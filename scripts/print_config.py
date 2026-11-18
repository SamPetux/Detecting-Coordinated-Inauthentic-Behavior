"""Print the effective configuration and its hash, to be checked line by line against docs/06_protocol.md."""
from __future__ import annotations

import argparse
import hashlib

import yaml


def flatten(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        out += flatten(v, key + ".") if isinstance(v, dict) else [(key, v)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/confirm.yaml")
    args = ap.parse_args()
    raw = open(args.config, "rb").read()
    print(f"# {args.config} sha256={hashlib.sha256(raw).hexdigest()[:16]}")
    for k, v in flatten(yaml.safe_load(raw)):
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()
