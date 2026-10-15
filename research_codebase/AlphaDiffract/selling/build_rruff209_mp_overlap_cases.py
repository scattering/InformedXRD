#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Stage validated RRUFF-209 cases for MP overlap search.")
    p.add_argument("--reference-json", required=True)
    p.add_argument("--validation-root", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--stage-processed-dir", required=True)
    return p.parse_args()


def centering_from_symbol(symbol: str) -> str:
    for ch in symbol:
        if ch.isalpha():
            return ch.upper()
    raise ValueError(f"cannot infer centering from symbol: {symbol}")


def main():
    args = parse_args()
    ref = json.loads(Path(args.reference_json).read_text())
    rows = ref["rows"]
    validation_root = Path(args.validation_root)
    stage_dir = Path(args.stage_processed_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for row in rows:
        rid = row["rid"]
        lebail_json = validation_root / rid / f"{rid}_lebail.json"
        d = json.loads(lebail_json.read_text())
        src_xy = Path(d["processed_xy"])
        dst_xy = stage_dir / src_xy.name
        if not dst_xy.exists():
            shutil.copy2(src_xy, dst_xy)

        fitted = row["fitted_cell"]
        out_rows.append(
            {
                "rid": rid,
                "name": row["name"],
                "standard_sg_number": int(row["standard_sg_number"]),
                "standard_sg_symbol": row["standard_sg_symbol"],
                "centering": centering_from_symbol(row["standard_sg_symbol"]),
                "crystal_system": row["crystal_system"],
                "processed_xy": dst_xy.name,
                "cell": [
                    float(fitted["a"]),
                    float(fitted["b"]),
                    float(fitted["c"]),
                    float(fitted["alpha"]),
                    float(fitted["beta"]),
                    float(fitted["gamma"]),
                ],
                "winning_transform": row["winning_transform"],
                "best_icsd_id": row["best_icsd_id"],
                "best_icsd_name": row["best_icsd_name"],
            }
        )

    payload = {"n_rows": len(out_rows), "rows": out_rows}
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"n_rows": len(out_rows), "output_json": str(out), "stage_processed_dir": str(stage_dir)}, indent=2))


if __name__ == "__main__":
    main()
