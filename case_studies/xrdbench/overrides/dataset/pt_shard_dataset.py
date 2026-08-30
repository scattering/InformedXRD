import json
import time

import torch
from torch.utils.data import Dataset


class PTShardDataset(Dataset):
    def __init__(self, manifest_path: str):
        t0 = time.time()
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        shards = manifest.get("shards", [])
        if not shards:
            raise ValueError(f"No shards listed in manifest: {manifest_path}")

        intensities = []
        spg = []
        crysystem = []
        for shard in shards:
            payload = torch.load(shard["path"], map_location="cpu")
            intensities.append(payload["intensity"])
            spg.append(payload["spg"])
            crysystem.append(payload["crysystem"])

        self.intensity = torch.cat(intensities, dim=0).contiguous()
        self.spg = torch.cat(spg, dim=0).contiguous()
        self.crysystem = torch.cat(crysystem, dim=0).contiguous()
        self.sequence_length = int(self.intensity.shape[1])
        self.load_seconds = time.time() - t0
        print(
            f"Loaded PTShardDataset from {manifest_path}: "
            f"rows={self.intensity.shape[0]} seq_len={self.sequence_length} "
            f"load_seconds={self.load_seconds:.2f}",
            flush=True,
        )

    def __len__(self):
        return int(self.intensity.shape[0])

    def __getitem__(self, idx):
        return {
            "intensity": self.intensity[idx],
            "spg": self.spg[idx],
            "crysystem": self.crysystem[idx],
            "element": torch.tensor([], dtype=torch.float32),
        }
