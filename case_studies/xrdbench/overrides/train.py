# Modified from the training script of the public XRDBench benchmark suite
# (SimXRD; MIT License -- see UPSTREAM_LICENSE in this directory).
# Modifications for this work: PT-shard dataset loading, model-module
# registry, and evaluation-export hooks. Drop-in override for the upstream repo.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from importlib import import_module
from dataset.dataset import ASEDataset
from dataset.pt_shard_dataset import PTShardDataset
from tqdm import tqdm
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os
from utils.tools import EarlyStopping, adjust_learning_rate_withWarmup
import json
import wandb
import warnings

time_exp_dic = {"time": 0, "counter": 0}


MODEL_MODULES = {
    "cnn1": ("model.CNN1", "Model", None),
    "cnn2": ("model.CNN2", "Model", None),
    "cnn3": ("model.CNN3", "Model", None),
    "cnn4": ("model.CNN4", "Model", None),
    "cnn5": ("model.CNN5", "Model", None),
    "cnn6": ("model.CNN6", "Model", None),
    "cnn7": ("model.CNN7", "Model", (0.7,)),
    "cnn8": ("model.CNN8", "Model", None),
    "cnn9": ("model.CNN9", "Model", None),
    "cnn10": ("model.CNN10", "Model", None),
    "cnn11": ("model.CNN11", "Model", None),
    "mlp": ("model.MLP", "Model", None),
    "rnn": ("model.RNN", "Model", None),
    "lstm": ("model.LSTM", "Model", None),
    "gru": ("model.GRU", "Model", None),
    "birnn": ("model.BiRNN", "Model", None),
    "bilstm": ("model.BiLSTM", "Model", None),
    "bigru": ("model.BiGRU", "Model", None),
    "transformer": ("model.Transformer", "Model", None),
    "iTransformer": ("model.iTransformer", "Model", None),
    "PatchTST": ("model.PatchTST", "Model", None),
}


def build_model(args):
    if args.model not in MODEL_MODULES:
        raise ValueError(f"Unknown model: {args.model}")
    module_name, class_name, extra_args = MODEL_MODULES[args.model]
    module = import_module(module_name)
    model_cls = getattr(module, class_name)
    if extra_args is None:
        return model_cls(args)
    return model_cls(*extra_args, args)


def _atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, tmp_path)
    if not os.path.exists(tmp_path):
        raise FileNotFoundError(f"Temporary checkpoint was not created: {tmp_path}")
    os.replace(tmp_path, path)


def save_periodic_checkpoint(model, optimizer, epoch, batch_index, path):
    latest_model_path = os.path.join(path, "checkpoint_latest.pth")
    latest_resume_path = os.path.join(path, "checkpoint_latest_resume.pt")
    _atomic_torch_save(model.state_dict(), latest_model_path)
    _atomic_torch_save(
        {
            "epoch": epoch,
            "batch_index": batch_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        latest_resume_path,
    )
    print(f"Periodic checkpoint saved at epoch={epoch} batch={batch_index + 1}")


def _build_loader(dataset, batch_size, shuffle, num_workers, use_gpu, drop_last=False):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": bool(use_gpu),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def train(args, nowtime):
    warnings.filterwarnings("ignore")
    if args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    train_shard_manifest = os.environ.get("XRDBENCH_TRAIN_SHARD_MANIFEST", "").strip()
    val_shard_manifest = os.environ.get("XRDBENCH_VAL_SHARD_MANIFEST", "").strip()
    test_shard_manifest = os.environ.get("XRDBENCH_TEST_SHARD_MANIFEST", "").strip()
    if train_shard_manifest:
        print(f"Using shard-backed train dataset: {train_shard_manifest}", flush=True)
        train_dataset = PTShardDataset(train_shard_manifest)
    else:
        train_db_names = [
            name.strip()
            for name in os.environ.get("XRDBENCH_TRAIN_DB_NAMES", "train_1,train_2").split(",")
            if name.strip()
        ]
        train_db_paths = [os.path.join(args.datapath, name, "binxrd.db") for name in train_db_names]
        train_dataset = ASEDataset(train_db_paths, False)

    if val_shard_manifest:
        print(f"Using shard-backed val dataset: {val_shard_manifest}", flush=True)
        val_dataset = PTShardDataset(val_shard_manifest)
    else:
        val_dataset = ASEDataset([os.path.join(args.datapath, "val_db/test_binxrd.db")], False)
    if test_shard_manifest:
        print(f"Using shard-backed test dataset: {test_shard_manifest}", flush=True)
        test_dataset = PTShardDataset(test_shard_manifest)
    else:
        test_dataset = ASEDataset([os.path.join(args.datapath, "test_db/binxrd.db")], False)

    train_loader = _build_loader(train_dataset, args.batch_size, True, args.num_workers, args.use_gpu)
    val_loader = _build_loader(val_dataset, args.batch_size, True, args.num_workers, args.use_gpu, drop_last=False)
    test_loader = _build_loader(test_dataset, args.batch_size, False, args.num_workers, args.use_gpu, drop_last=False)

    model = build_model(args)

    save_path = f"./checkpoints/{args.task}-{args.model}_lr{args.lr}_bs{args.batch_size}_{nowtime}_{args.seed}"
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + "/args.json", "w") as f:
        json.dump(args.__dict__, f)
    model = model.to(device)
    checkpoint_every_steps = int(os.environ.get("XRDBENCH_CHECKPOINT_EVERY_STEPS", "0"))
    max_train_steps = int(os.environ.get("XRDBENCH_MAX_TRAIN_STEPS", "0"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(
            model,
            optimizer,
            criterion,
            epoch,
            train_loader,
            device,
            args,
            checkpoint_every_steps=checkpoint_every_steps,
            checkpoint_path=save_path,
            max_steps=max_train_steps,
        )
        val_loss, _ = run_epoch(model, optimizer, criterion, epoch, val_loader, device, args, backprop=False)
        test_loss, res = run_epoch(model, optimizer, criterion, epoch, test_loader, device, args, backprop=False)
        if checkpoint_every_steps > 0:
            save_periodic_checkpoint(model, optimizer, epoch, len(train_loader) - 1, save_path)

        wandb.log(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "test_f1": res["macro_f1"],
                "test_precision": res["macro_precision"],
                "test_recall": res["macro_recall"],
                "test_acc": res["accuracy"],
                "test_top5_acc": res["top5_accuracy"],
                "infer_time": time_exp_dic["time"] / max(time_exp_dic["counter"], 1),
            }
        )
        early_stopping(epoch, val_loss, test_loss, res, model, save_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    wandb.log(
        {
            "epoch": early_stopping.best_epoch,
            "val_loss": early_stopping.val_loss_min,
            "test_loss": early_stopping.best_test_loss,
            "test_f1": early_stopping.best_test_macrof1,
            "test_precision": early_stopping.best_test_precision,
            "test_recall": early_stopping.best_test_recall,
            "test_acc": early_stopping.best_test_accuracy,
            "test_top5_acc": getattr(early_stopping, "best_test_top5_accuracy", 0),
        }
    )

    print(
        "*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best Test Accuracy: %.5f \t Best Test Top-5: %.5f \t Best Macro-F1: %.5f\t Best epoch %d"
        % (
            early_stopping.val_loss_min,
            early_stopping.best_test_loss,
            early_stopping.best_test_accuracy,
            getattr(early_stopping, "best_test_top5_accuracy", 0),
            early_stopping.best_test_macrof1,
            early_stopping.best_epoch,
        )
    )

    return early_stopping


def run_epoch(
    model,
    optimizer,
    criterion,
    epoch,
    loader,
    device,
    args,
    backprop=True,
    checkpoint_every_steps=0,
    checkpoint_path=None,
    max_steps=0,
):
    if backprop:
        model.train()
    else:
        model.eval()
    res = {
        "epoch": epoch,
        "loss": 0,
        "accuracy": 0,
        "top5_accuracy": 0,
        "top5_correct": 0,
        "macro_f1": 0,
        "counter": 0,
        "macro_precision": 0,
        "macro_recall": 0,
    }
    all_labels = []
    all_predicted = []
    for batch_index, data in enumerate(tqdm(loader)):
        if backprop and max_steps > 0 and batch_index >= max_steps:
            print(f"Stopping train epoch early at batch {batch_index} due to XRDBENCH_MAX_TRAIN_STEPS={max_steps}")
            break
        intensity = data["intensity"].to(device)
        crysystem_labels = data["crysystem"].to(device)
        spg_labels = data["spg"].to(device)
        element = data["element"].to(device)
        intensity = intensity.unsqueeze(1)
        element = element.unsqueeze(1)
        if args.task in ("spg", "eg"):
            labels = spg_labels - 1
        elif args.task == "crysystem":
            labels = crysystem_labels - 1
        batch_size = intensity.shape[0]
        if backprop:
            optimizer.zero_grad()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            outputs = model(intensity)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic["time"] += t2 - t1
            time_exp_dic["counter"] += 1

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if checkpoint_every_steps > 0 and checkpoint_path and (batch_index + 1) % checkpoint_every_steps == 0:
                save_periodic_checkpoint(model, optimizer, epoch, batch_index, checkpoint_path)
        else:
            outputs = model(intensity)
            loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        k = min(5, outputs.shape[1])
        topk = torch.topk(outputs, k=k, dim=1).indices
        res["top5_correct"] += int((topk == labels.unsqueeze(1)).any(dim=1).sum().item())

        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size

        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())
        if args.model == "CPICANN":
            adjust_learning_rate_withWarmup(optimizer, epoch + batch_index / len(loader), args)

    accuracy = accuracy_score(all_labels, all_predicted)
    top5_accuracy = res.get("top5_correct", 0) / len(all_labels) if all_labels else 0
    macro_f1 = f1_score(all_labels, all_predicted, average="macro")
    macro_precision = precision_score(all_labels, all_predicted, average="macro")
    macro_recall = recall_score(all_labels, all_predicted, average="macro")

    res["accuracy"] = accuracy
    res["top5_accuracy"] = top5_accuracy
    res["macro_f1"] = macro_f1
    res["macro_precision"] = macro_precision
    res["macro_recall"] = macro_recall

    if not backprop:
        prefix = "==> "
    else:
        prefix = " "
    print("%s epoch %d avg loss: %.5f top1: %.5f top5: %.5f" % (prefix, epoch, res["loss"] / res["counter"], res["accuracy"], res["top5_accuracy"]))

    return res["loss"] / res["counter"], res
