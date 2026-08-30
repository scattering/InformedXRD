import argparse
import ast
import json
from pathlib import Path

import ase.db
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0, help="0 means full DB")
    return parser.parse_args()


def flush_shard(output_dir, shard_idx, intensities, spg, crysystem):
    shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
    payload = {
        "intensity": torch.stack(intensities, dim=0),
        "spg": torch.tensor(spg, dtype=torch.int64),
        "crysystem": torch.tensor(crysystem, dtype=torch.int64),
    }
    torch.save(payload, shard_path)
    return shard_path, payload["intensity"].shape[0]


def main():
    args = parse_args()
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = ase.db.connect(str(db_path))
    total_rows = len(db)
    target_rows = min(args.limit, total_rows) if args.limit > 0 else total_rows

    manifest = {
        "source_db": str(db_path),
        "total_rows_in_db": total_rows,
        "target_rows": target_rows,
        "shard_size": args.shard_size,
        "shards": [],
    }

    intensities = []
    spg = []
    crysystem = []
    shard_idx = 0
    rows_written = 0
    seq_len = None

    for row_idx, row in enumerate(db.select(limit=target_rows), start=1):
        intensity = torch.tensor(ast.literal_eval(getattr(row, "intensity")), dtype=torch.float32)
        tager = ast.literal_eval(getattr(row, "tager"))
        this_seq_len = int(intensity.shape[0])
        if seq_len is None:
            seq_len = this_seq_len
        elif this_seq_len != seq_len:
            raise ValueError(f"Inconsistent sequence length at row {row.id}: {this_seq_len} vs {seq_len}")

        intensities.append(intensity)
        spg.append(int(tager[0]))
        crysystem.append(int(tager[1]))

        if len(intensities) >= args.shard_size:
            shard_path, count = flush_shard(output_dir, shard_idx, intensities, spg, crysystem)
            manifest["shards"].append({"path": str(shard_path), "rows": count})
            rows_written += count
            print(f"[shard] {shard_path} rows={count} cumulative={rows_written}/{target_rows}", flush=True)
            shard_idx += 1
            intensities, spg, crysystem = [], [], []

    if intensities:
        shard_path, count = flush_shard(output_dir, shard_idx, intensities, spg, crysystem)
        manifest["shards"].append({"path": str(shard_path), "rows": count})
        rows_written += count
        print(f"[shard] {shard_path} rows={count} cumulative={rows_written}/{target_rows}", flush=True)

    manifest["rows_written"] = rows_written
    manifest["sequence_length"] = seq_len
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "manifest": str(manifest_path),
        "rows_written": rows_written,
        "sequence_length": seq_len,
        "num_shards": len(manifest["shards"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
