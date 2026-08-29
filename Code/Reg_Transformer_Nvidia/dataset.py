import torch
import h5py
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ====== Lazy-loading Dataset with chunked batch reads ======
class H5ChunkedDataset(Dataset):
    def __init__(self, h5_path, split):
        self.h5_path = h5_path
        self.split = split
        self.file = None  # will be opened once per worker
        with h5py.File(h5_path, "r") as f:
            self.len = f[f"X_{split}"].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        X_ds = self.file[f"X_{self.split}"]
        y_ds = self.file[f"y_{self.split}"]

        # If idx is a list/tensor (batch mode), fetch all at once
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, list):
            #X = X_ds[idx, :700]
            X = X_ds[idx, :]
            y = y_ds[idx] - 1
        else:  # single sample
            #X = X_ds[idx, :700]
            X = X_ds[idx]
            y = y_ds[idx] - 1

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ====== Worker init to keep file open per worker ======
def _init_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, H5ChunkedDataset):
        dataset.file = h5py.File(dataset.h5_path, "r")


# ====== Custom collate to handle streaming batches from HDF5 ======
def h5_collate(batch):
    # If the dataset returned a whole batch (chunk read), it’s already a tuple of tensors
    if isinstance(batch[0][0], torch.Tensor) and batch[0][0].ndim > 1:
        return batch[0]
    else:
        # Default PyTorch collate
        return torch.utils.data._utils.collate.default_collate(batch)


# ====== Loader function ======
def get_dataloaders(h5_file="xrd_data.h5", batch_size=32, world_size=1, rank=0,
                    num_workers=8, num_classes=230, prefetch_factor=4):

    train_dataset = H5ChunkedDataset(h5_file, "train")
    val_dataset = H5ChunkedDataset(h5_file, "val")
    test_dataset = H5ChunkedDataset(h5_file, "test")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=h5_collate, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=h5_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=h5_collate
    )

    with h5py.File(h5_file, "r") as f:
        intensity_points = f["X_train"].shape[1]

    return train_loader, val_loader, test_loader, num_classes, intensity_points

# ====== Loader function (test only) ======
def get_dataloaders_test(h5_file="xrd_data.h5", batch_size=32, world_size=1, rank=0,
                         num_workers=8, num_classes=230, prefetch_factor=4):

    test_dataset = H5ChunkedDataset(h5_file, "test")

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=h5_collate
    )

    with h5py.File(h5_file, "r") as f:
        intensity_points = f["X_test"].shape[1]

    return None, None, test_loader, num_classes, intensity_points
