#CUDA_VISIBLE_DEVICES=0 python main.py -c config/cfg_patch_stage_a_emb.py --datasets config/datasets_patch_stage_a_raw_local.json --output_dir outputs/stageA_emb --pretrain_model_path weights/groundingdino_swint_ogc.pth --num_workers 8 --amp# --resume outputs/stageA_emb/checkpoint.pth
# CUDA_VISIBLE_DEVICES=0 python main.py -c config/cfg_patch_stage_a.py --datasets config/datasets_patch_stage_a_coco2017_local.json --output_dir outputs/stageA_patch --pretrain_model_path weights/groundingdino_swint_ogc.pth --num_workers 8 --amp --resume outputs/stageA_patch/checkpoint.pth 
# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict, match_name_keywords
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    """
    PyTorch >= 2.6 defaults `torch.load(..., weights_only=True)`, which can fail on older
    training checkpoints that include non-tensor objects (e.g. argparse.Namespace).
    """
    import torch as _torch

    try:
        return _torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        # Allowlist argparse.Namespace for safe weights-only loading (our checkpoints store `args`).
        try:
            from torch import serialization as _serialization  # type: ignore

            _serialization.add_safe_globals([argparse.Namespace])  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return _torch.load(path, map_location=map_location)
        except Exception:
            # Fall back to full unpickling (unsafe for untrusted files).
            return _torch.load(path, map_location=map_location, weights_only=False)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument("--datasets", type=str, required=True, help='path to datasets json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def _trainable_param_summary(model: torch.nn.Module):
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    by_module = {}
    for name, count in trainable.items():
        root = name.split(".", 1)[0]
        by_module[root] = by_module.get(root, 0) + int(count)
    return trainable, by_module


def main(args):
    

    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    # Some flags exist both in argparse and config; allow config to override a small safe subset.
    allow_cfg_override = {"fix_size"}
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        elif k in allow_cfg_override:
            setattr(args, k, v)
            if args.rank == 0:
                print(f"[WARN] Config overrides argparse key: {k}={v}")
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")
    patch_only = bool(getattr(args, "patch_only", False))
    if patch_only and args.eval:
        raise ValueError("patch_only training does not support --eval (postprocessors/eval prompt are text-based).")


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params before freezing:\n"+json.dumps({n: p.numel() for n, p in model_without_ddp.named_parameters() if p.requires_grad}, indent=2))

    # freeze some layers BEFORE building optimizer param groups
    if args.freeze_keywords is not None:
        for name, parameter in model_without_ddp.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break

    # Optional: unfreeze last N decoder layers (useful for patch-only adaptation).
    unfreeze_n = int(getattr(args, "unfreeze_decoder_last_n_layers", 0) or 0)
    if unfreeze_n <= 0 and bool(getattr(args, "unfreeze_decoder_last_layer", False)):
        # Backward compatibility with older configs.
        unfreeze_n = 1
    if unfreeze_n > 0:
        try:
            decoder = model_without_ddp.transformer.decoder
            layers = list(getattr(decoder, "layers", []))
            if not layers:
                raise RuntimeError("transformer.decoder.layers is empty or missing.")
            n = min(int(unfreeze_n), len(layers))
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            logger.info(f"unfreeze_decoder_last_n_layers={unfreeze_n}: transformer.decoder.layers[-{n}:] are trainable.")
        except Exception as e:
            logger.warning(f"unfreeze_decoder_last_n_layers={unfreeze_n} but failed to unfreeze decoder layers: {e}")

    only_train_keywords = getattr(args, "only_train_keywords", None)
    if only_train_keywords:
        if isinstance(only_train_keywords, str):
            only_train_keywords = [only_train_keywords]
        for _, parameter in model_without_ddp.named_parameters():
            parameter.requires_grad_(False)
        for name, parameter in model_without_ddp.named_parameters():
            if match_name_keywords(name, only_train_keywords):
                parameter.requires_grad_(True)

        unexpected = [
            name
            for name, parameter in model_without_ddp.named_parameters()
            if parameter.requires_grad and not match_name_keywords(name, only_train_keywords)
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable parameters outside only_train_keywords: {unexpected[:20]}")

    trainable_params, trainable_modules = _trainable_param_summary(model_without_ddp)
    logger.info("params after freezing:\n" + json.dumps(trainable_params, indent=2))
    logger.info("trainable module summary:\n" + json.dumps(trainable_modules, indent=2))
    if only_train_keywords and (not trainable_params):
        raise RuntimeError("No trainable parameters remain after applying only_train_keywords.")

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    logger.debug("build dataset ... ...")
    dataset_train_list = None
    train_mix_weights = None
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
        else:
            from torch.utils.data import ConcatDataset
            dataset_train_list = []
            train_mix_weights = []
            for idx in range(len(dataset_meta["train"])):
                datasetinfo = dataset_meta["train"][idx]
                dataset_train_list.append(build_dataset(image_set='train', args=args, datasetinfo=datasetinfo))
                train_mix_weights.append(float(datasetinfo.get("mix_weight", 1.0)))
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(f'number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}')

    dataset_val = None
    if not patch_only:
        dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False) if dataset_val is not None else None
        if not args.eval:
            if train_mix_weights is not None and any(abs(w - 1.0) > 1e-12 for w in train_mix_weights):
                raise NotImplementedError("mix_weight sampling is currently only supported in non-distributed training.")
            sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        if not args.eval:
            if train_mix_weights is not None and any(abs(w - 1.0) > 1e-12 for w in train_mix_weights):
                sample_weights = []
                for ds, mix_weight in zip(dataset_train_list, train_mix_weights):
                    ds_len = max(1, len(ds))
                    sample_weights.extend([float(mix_weight) / float(ds_len)] * len(ds))
                sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
                sampler_train = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset_train),
                    replacement=True,
                )
                expected = []
                total_mix = sum(train_mix_weights)
                for idx, (ds, mix_weight) in enumerate(zip(dataset_train_list, train_mix_weights)):
                    expected.append(
                        {
                            "dataset_idx": idx,
                            "len": len(ds),
                            "mix_weight": float(mix_weight),
                            "expected_fraction": (float(mix_weight) / float(total_mix)) if total_mix > 0 else 0.0,
                        }
                    )
                logger.info("using mix_weight weighted sampling:\n" + json.dumps(expected, indent=2))
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        def _worker_init_fn(_worker_id: int):
            # Avoid CPU thread oversubscription when using many DataLoader workers.
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        dl_train_kwargs = dict(
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
        if str(args.device).startswith("cuda"):
            dl_train_kwargs["pin_memory"] = True
        if int(args.num_workers) > 0:
            dl_train_kwargs["persistent_workers"] = True
            dl_train_kwargs["prefetch_factor"] = 2
            dl_train_kwargs["worker_init_fn"] = _worker_init_fn
        data_loader_train = DataLoader(dataset_train, **dl_train_kwargs)

    data_loader_val = None
    if dataset_val is not None:
        dl_val_kwargs = dict(
            batch_size=4,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
        if str(args.device).startswith("cuda"):
            dl_val_kwargs["pin_memory"] = True
        if int(args.num_workers) > 0:
            dl_val_kwargs["persistent_workers"] = True
            dl_val_kwargs["prefetch_factor"] = 2
            dl_val_kwargs["worker_init_fn"] = _worker_init_fn
        data_loader_val = DataLoader(dataset_val, **dl_val_kwargs)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    base_ds = get_coco_api_from_dataset(dataset_val) if dataset_val is not None else None

    if args.frozen_weights is not None:
        checkpoint = _torch_load_compat(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = _torch_load_compat(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)


        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
            except Exception as e:
                logger.warning(
                    f"Failed to restore optimizer/scheduler state from resume checkpoint; "
                    f"continuing with fresh optimizer state. Error: {e}"
                )

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = _torch_load_compat(args.pretrain_model_path, map_location="cpu")["model"]
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

 
    
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return
    
 
    
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False) if not patch_only else None

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None))
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # Store as plain dict to stay compatible with `weights_only=True` safe loading.
                    'args': vars(args),
                }

                utils.save_on_master(weights, checkpoint_path)
                
        if not patch_only:
            # eval
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}}


        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if (not patch_only) and coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
