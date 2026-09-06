#!/usr/bin/env python3
"""Independent MDETR Python layer over an immutable, explicitly bound torch env."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-python", type=Path, required=True)
    p.add_argument("--environment", type=Path, required=True)
    args = p.parse_args()
    base_site = subprocess.check_output([str(args.base_python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip()
    if not args.environment.exists():
        subprocess.run([str(args.base_python), "-m", "venv", str(args.environment)], check=True)
    py = args.environment / "bin/python"
    site = Path(subprocess.check_output([str(py), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip())
    pointer = site / "mdetr_read_only_base_dependencies.pth"
    text = base_site + "\n"
    if pointer.exists() and pointer.read_text() != text:
        raise RuntimeError("different base environment already bound")
    if not pointer.exists():
        with pointer.open("x") as f:
            f.write(text)
    system_paths = json.loads(subprocess.check_output([str(args.base_python), "-c", "import sys,json;print(json.dumps([p for p in sys.path if p.startswith('/usr/') and p.endswith(('site-packages','dist-packages'))]))"], text=True))
    system_pointer = site / "mdetr_read_only_system_dependencies.pth"
    system_text = "\n".join(system_paths) + "\n"
    if system_pointer.exists() and system_pointer.read_text() != system_text:
        raise RuntimeError("different system dependencies already bound")
    if not system_pointer.exists():
        with system_pointer.open("x") as f:
            f.write(system_text)
    # --ignore-installed prevents any uninstall/update in the read-only base.
    subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--ignore-installed", "transformers==4.40.2", "tokenizers==0.19.1", "timm==0.9.16"], check=True)
    code = "import sys,torch,torchvision,transformers,tokenizers,timm,json;print(json.dumps({'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'timm':timm.__version__,'torch_path':torch.__file__,'transformers_path':transformers.__file__}))"
    versions = json.loads(subprocess.check_output([str(py), "-c", code], text=True))
    receipt = {"schema": "arrow.confidence_readout.mdetr_environment/v1", "environment": str(args.environment.resolve()), "base_read_only_site_packages": base_site, "system_read_only_packages": system_paths, "versions": versions, "base_environment_mutated": False, "official_requirements_note": "upstream requests transformers 4.5.1; this Python 3.12 compatible adapter pins 4.40.2 eager Roberta, no upstream source patch"}
    out = args.environment / "runtime_receipt.json"
    if out.exists() and json.loads(out.read_text()) != receipt:
        raise RuntimeError("environment receipt changed")
    if not out.exists():
        with out.open("x") as f:
            json.dump(receipt, f, indent=2)
            f.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
