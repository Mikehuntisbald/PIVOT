#CUDA_VISIBLE_DEVICES=0 python main.py -c config/cfg_patch_stage_a_emb.py --datasets config/datasets_patch_stage_a_raw_local.json --output_dir outputs/stageA_emb --pretrain_model_path weights/groundingdino_swint_ogc.pth --num_workers 8 --amp# --resume outputs/stageA_emb/checkpoint.pth
# CUDA_VISIBLE_DEVICES=0 python main.py -c config/cfg_patch_stage_a.py --datasets config/datasets_patch_stage_a_coco2017_local.json --output_dir outputs/stageA_patch --pretrain_model_path weights/groundingdino_swint_ogc.pth --num_workers 8 --amp --resume outputs/stageA_patch/checkpoint.pth 
# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import signal
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
from engine import GracefulTrainingExit, evaluate, train_one_epoch

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


def _make_grad_scaler(enabled: bool):
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return amp_mod.GradScaler(device_type="cuda", enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


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
    parser.add_argument(
        '--prefetch_factor',
        default=1,
        type=int,
        help='DataLoader batches prefetched per worker when num_workers > 0; lower values use fewer shared-memory file descriptors',
    )
    parser.add_argument(
        '--pin_memory',
        dest='pin_memory',
        action='store_true',
        default=None,
        help='enable DataLoader pin_memory; default is enabled for CUDA devices',
    )
    parser.add_argument(
        '--no_pin_memory',
        '--no-pin-memory',
        dest='pin_memory',
        action='store_false',
        help='disable DataLoader pin_memory',
    )
    parser.add_argument(
        '--persistent_workers',
        dest='persistent_workers',
        action='store_true',
        default=None,
        help='keep DataLoader workers alive between epochs; default is enabled when num_workers > 0',
    )
    parser.add_argument(
        '--no_persistent_workers',
        '--no-persistent-workers',
        dest='persistent_workers',
        action='store_false',
        help='disable persistent DataLoader workers',
    )
    parser.add_argument(
        '--mp_sharing_strategy',
        default=os.environ.get("TORCH_MP_SHARING_STRATEGY", "file_system"),
        choices=("file_system", "file_descriptor", "none"),
        help='torch multiprocessing CPU tensor sharing strategy; file_system avoids one fd per shared storage',
    )
    parser.add_argument(
        '--min_nofile',
        default=_env_int("GDINO_MIN_NOFILE", 65536),
        type=int,
        help='try to raise the process open-file soft limit to at least this value; 0 disables',
    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument(
        '--iter_checkpoint_interval',
        default=0,
        type=int,
        help='save output_dir/checkpoint_iter.pth every N finished train iterations; 0 disables periodic saves',
    )

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


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _install_signal_checkpoint_handlers(args):
    def _handler(signum, _frame):
        if getattr(args, "_stop_requested", False):
            raise KeyboardInterrupt(f"Received signal {signum} twice.")
        args._stop_requested = True
        args._stop_signal = int(signum)
        print(
            f"Received signal {signum}; will save checkpoint_iter.pth after the current iteration.",
            flush=True,
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def _configure_torch_multiprocessing(args, logger):
    strategy = str(getattr(args, "mp_sharing_strategy", "file_system") or "none")
    if strategy == "none":
        return
    try:
        available = torch.multiprocessing.get_all_sharing_strategies()
        if strategy not in available:
            logger.warning(
                f"Requested mp_sharing_strategy={strategy!r}, but available strategies are {sorted(available)}."
            )
            return
        torch.multiprocessing.set_sharing_strategy(strategy)
        logger.info(f"torch multiprocessing sharing strategy: {torch.multiprocessing.get_sharing_strategy()}")
    except Exception as e:
        logger.warning(f"Failed to set torch multiprocessing sharing strategy to {strategy!r}: {e}")


def _get_nofile_limit():
    try:
        import resource

        return resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return None


def _raise_nofile_limit(args, logger):
    minimum = int(getattr(args, "min_nofile", 0) or 0)
    if minimum <= 0:
        return
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(int(soft), minimum), int(hard))
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info(f"Raised RLIMIT_NOFILE soft limit from {soft} to {target} (hard={hard}).")
        elif soft < minimum:
            logger.warning(
                f"RLIMIT_NOFILE soft/hard is {soft}/{hard}; cannot raise to requested minimum {minimum}."
            )
    except Exception as e:
        logger.warning(f"Failed to adjust RLIMIT_NOFILE: {e}")


def _resolve_dataloader_runtime(args):
    num_workers = int(args.num_workers)
    if num_workers < 0:
        raise ValueError(f"--num_workers must be >= 0, got {num_workers}.")
    prefetch_factor = int(getattr(args, "prefetch_factor", 1))
    if prefetch_factor < 1:
        raise ValueError(f"--prefetch_factor must be >= 1, got {prefetch_factor}.")
    pin_memory_arg = getattr(args, "pin_memory", None)
    pin_memory = str(args.device).startswith("cuda") if pin_memory_arg is None else bool(pin_memory_arg)
    persistent_arg = getattr(args, "persistent_workers", None)
    persistent_workers = num_workers > 0 if persistent_arg is None else bool(persistent_arg)
    return {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }


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
    skip_eval = bool(getattr(args, "skip_eval", False)) and (not args.eval)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    _raise_nofile_limit(args, logger)
    _configure_torch_multiprocessing(args, logger)
    nofile_limit = _get_nofile_limit()
    if nofile_limit is not None:
        logger.info(f"RLIMIT_NOFILE soft/hard: {nofile_limit[0]}/{nofile_limit[1]}")
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval and (args.eval or not skip_eval):
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
    if (not patch_only) and (args.eval or not skip_eval):
        dataset_val = build_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False) if dataset_val is not None else None
        if not args.eval:
            has_dataset_sample_weights = (
                dataset_train_list is not None
                and any(getattr(ds, "sample_weights", None) is not None for ds in dataset_train_list)
            )
            if train_mix_weights is not None and (
                any(abs(w - 1.0) > 1e-12 for w in train_mix_weights) or has_dataset_sample_weights
            ):
                raise NotImplementedError(
                    "mix_weight / dataset sample_weight sampling is currently only supported in non-distributed training."
                )
            sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
        if not args.eval:
            has_dataset_sample_weights = (
                dataset_train_list is not None
                and any(getattr(ds, "sample_weights", None) is not None for ds in dataset_train_list)
            )
            has_explicit_mix_weights = train_mix_weights is not None and any(
                abs(w - 1.0) > 1e-12 for w in train_mix_weights
            )
            if train_mix_weights is not None and (has_explicit_mix_weights or has_dataset_sample_weights):
                sample_weights = []
                for ds, mix_weight in zip(dataset_train_list, train_mix_weights):
                    ds_len = max(1, len(ds))
                    if has_explicit_mix_weights:
                        base_weight = float(mix_weight) / float(ds_len)
                    else:
                        # Preserve RandomSampler's length-proportional dataset mix when all mix weights are 1.
                        base_weight = 1.0
                    ds_sample_weights = getattr(ds, "sample_weights", None)
                    if ds_sample_weights is not None and len(ds_sample_weights) == len(ds):
                        sample_weights.extend([base_weight * float(w) for w in ds_sample_weights])
                    else:
                        sample_weights.extend([base_weight] * len(ds))
                sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
                sampler_train = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset_train),
                    replacement=True,
                )
                expected = []
                total_mix = sum(train_mix_weights) if has_explicit_mix_weights else sum(len(ds) for ds in dataset_train_list)
                for idx, (ds, mix_weight) in enumerate(zip(dataset_train_list, train_mix_weights)):
                    expected_fraction = (
                        (float(mix_weight) / float(total_mix))
                        if has_explicit_mix_weights and total_mix > 0
                        else (float(len(ds)) / float(total_mix) if total_mix > 0 else 0.0)
                    )
                    expected.append(
                        {
                            "dataset_idx": idx,
                            "len": len(ds),
                            "mix_weight": float(mix_weight),
                            "expected_fraction": expected_fraction,
                            "tn_balance_stats": getattr(ds, "tn_balance_stats", None),
                        }
                    )
                logger.info("using mix_weight weighted sampling:\n" + json.dumps(expected, indent=2))
            elif getattr(dataset_train, "sample_weights", None) is not None:
                sample_weights = torch.as_tensor(getattr(dataset_train, "sample_weights"), dtype=torch.double)
                sampler_train = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset_train),
                    replacement=True,
                )
                logger.info(
                    "using dataset-level weighted sampling:\n"
                    + json.dumps(getattr(dataset_train, "tn_balance_stats", {}), indent=2)
                )
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

        dataloader_runtime = _resolve_dataloader_runtime(args)
        logger.info("DataLoader runtime settings: " + json.dumps(dataloader_runtime, indent=2))
        dl_train_kwargs = dict(
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=dataloader_runtime["num_workers"],
        )
        dl_train_kwargs["pin_memory"] = dataloader_runtime["pin_memory"]
        if dataloader_runtime["num_workers"] > 0:
            dl_train_kwargs["persistent_workers"] = dataloader_runtime["persistent_workers"]
            dl_train_kwargs["prefetch_factor"] = dataloader_runtime["prefetch_factor"]
            dl_train_kwargs["worker_init_fn"] = _worker_init_fn
        data_loader_train = DataLoader(dataset_train, **dl_train_kwargs)

    data_loader_val = None
    if dataset_val is not None:
        dataloader_runtime = _resolve_dataloader_runtime(args)
        logger.info("Validation DataLoader runtime settings: " + json.dumps(dataloader_runtime, indent=2))
        dl_val_kwargs = dict(
            batch_size=4,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=dataloader_runtime["num_workers"],
        )
        dl_val_kwargs["pin_memory"] = dataloader_runtime["pin_memory"]
        if dataloader_runtime["num_workers"] > 0:
            dl_val_kwargs["persistent_workers"] = dataloader_runtime["persistent_workers"]
            dl_val_kwargs["prefetch_factor"] = dataloader_runtime["prefetch_factor"]
            dl_val_kwargs["worker_init_fn"] = _worker_init_fn
        data_loader_val = DataLoader(dataset_val, **dl_val_kwargs)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    scaler = _make_grad_scaler(enabled=args.amp)
    resume_iter = 0
    resume_epoch_rng_state = None
    resume_runtime_rng_state = None

    base_ds = get_coco_api_from_dataset(dataset_val) if dataset_val is not None else None

    if args.frozen_weights is not None:
        checkpoint = _torch_load_compat(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    output_dir = Path(args.output_dir)
    auto_resume_checkpoint = output_dir / 'checkpoint.pth'
    if (not args.resume) and auto_resume_checkpoint.exists():
        logger.info(
            f"Found existing checkpoint at {auto_resume_checkpoint}, but auto-resume is disabled. "
            "Pass --resume explicitly to restore model/optimizer/scheduler from it."
        )
    if args.resume:
        logger.info(f"Loading resume checkpoint from {args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = _torch_load_compat(args.resume, map_location="cpu")
        load_output = model_without_ddp.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)
        logger.info(f"Loaded resume model state: {load_output}")


        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                restored_scaler = False
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                    restored_scaler = True
                ckpt_epoch = int(checkpoint['epoch'])
                ckpt_iter = int(checkpoint.get('iteration', 0) or 0)
                epoch_finished = bool(checkpoint.get('epoch_finished', 'iteration' not in checkpoint))
                logger.info(
                    "Restored resume training state: "
                    f"epoch={ckpt_epoch}, iteration={ckpt_iter}, "
                    f"epoch_finished={epoch_finished}, scaler_restored={restored_scaler}"
                )
                if (not epoch_finished) and ckpt_iter > 0 and ckpt_iter < len(data_loader_train):
                    args.start_epoch = ckpt_epoch
                    resume_iter = ckpt_iter
                    resume_epoch_rng_state = checkpoint.get('epoch_rng_state', None)
                    resume_runtime_rng_state = checkpoint.get('rng_state', None)
                    logger.info(
                        f"Resuming mid-epoch from epoch={ckpt_epoch}, "
                        f"next_iter={resume_iter}/{len(data_loader_train)}"
                    )
                else:
                    args.start_epoch = ckpt_epoch + 1
                    logger.info(
                        f"Resuming from next epoch: checkpoint_epoch={ckpt_epoch}, "
                        f"start_epoch={args.start_epoch}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to restore optimizer/scheduler state from resume checkpoint; "
                    f"continuing with fresh optimizer state. Error: {e}"
                )
        elif not args.eval:
            logger.info(
                "Resume checkpoint did not include optimizer/lr_scheduler/epoch; "
                "loaded model weights only and will use fresh training state."
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
    best_map_holder = BestMetricHolder(use_ema=False) if (not patch_only and not skip_eval) else None
    _install_signal_checkpoint_handlers(args)

    current_epoch_rng_state = None
    coco_evaluator = None

    def _checkpoint_payload(epoch, *, iteration=0, epoch_finished=True, reason=None):
        payload = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': int(epoch),
            'iteration': int(iteration),
            'epoch_finished': bool(epoch_finished),
            'rng_state': _capture_rng_state(),
            'epoch_rng_state': current_epoch_rng_state,
            # Store as plain dict to stay compatible with `weights_only=True` safe loading.
            'args': vars(args),
        }
        if reason is not None:
            payload['checkpoint_reason'] = str(reason)
        return payload

    def _save_iter_checkpoint(*, epoch, iteration, scaler=None, epoch_finished=False, reason=None):
        if not args.output_dir:
            return
        checkpoint_path = output_dir / 'checkpoint_iter.pth'
        utils.save_on_master(
            _checkpoint_payload(epoch, iteration=iteration, epoch_finished=epoch_finished, reason=reason),
            checkpoint_path,
        )
        msg = f"Saved iteration checkpoint to {checkpoint_path} (epoch={epoch}, next_iter={iteration}, reason={reason})."
        logger.info(msg) if args.save_log else print(msg, flush=True)

    try:
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)

            this_start_iter = resume_iter if epoch == args.start_epoch else 0
            if this_start_iter > 0 and resume_epoch_rng_state is not None:
                current_epoch_rng_state = resume_epoch_rng_state
            else:
                current_epoch_rng_state = _capture_rng_state()

            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler,
                args=args, logger=(logger if args.save_log else None), scaler=scaler,
                start_iter=this_start_iter, epoch_rng_state=current_epoch_rng_state,
                runtime_rng_state=(resume_runtime_rng_state if this_start_iter > 0 else None),
                iter_checkpoint_fn=_save_iter_checkpoint)
            resume_iter = 0
            resume_epoch_rng_state = None
            resume_runtime_rng_state = None
            if getattr(args, "_stop_requested", False):
                _save_iter_checkpoint(
                    epoch=epoch,
                    iteration=len(data_loader_train),
                    scaler=scaler,
                    epoch_finished=True,
                    reason="signal_after_epoch",
                )
                return
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
                    weights = _checkpoint_payload(epoch, iteration=0, epoch_finished=True, reason="epoch")

                    utils.save_on_master(weights, checkpoint_path)
                
            if not patch_only and not skip_eval:
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
                        'scaler': scaler.state_dict(),
                        'epoch': epoch,
                        'iteration': 0,
                        'epoch_finished': True,
                        'args': vars(args),
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
    except GracefulTrainingExit as e:
        msg = str(e) or "Training stopped after writing iteration checkpoint."
        logger.info(msg) if args.save_log else print(msg)
        return
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
