import os
import json
import argparse
import torch
import wandb
from model import transformer_model
from dataset import get_dataloaders
import torch.optim as optim
import intel_extension_for_pytorch as ipex
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--sweep", action="store_true", help="Flag to indicate if this is a sweep run")
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel (DDP)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--sweep_id", type=str, required=True, help="W&B Sweep ID")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def compute_top_k_accuracy(output, target, k=1):
    """Compute top-k accuracy: fraction of samples where true label is in top-k"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct_k = correct.any(dim=1).float().sum().item()  # only 1 per sample
        return correct_k
    
def train(config, train_loader, val_loader, test_loader, device, rank, world_size):
    # Initialize wandb on all ranks
    wandb.init(project="reg-sweeps", config=config)
    config = wandb.config  # Get hyperparameters from sweep
    
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
    
    # Use actual input size
    actual_spec_length = inputs.shape[1]
    
    # Initialize model
    model = transformer_model(
        spec_length=actual_spec_length,
        num_output=config["num_classes"],
        embed_dim=config["embed_dim"],
        ff_dim=config["ff_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_units=config["mlp_units"],
        dropout=config["dropout"],
    ).to(device)

    # Wrap model with DDP if using distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[device])

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]) if config["optimizer"] == "Adam" else optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Optimize the model with IPEX for better performance
    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    torch.xpu.empty_cache()

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Set epoch for distributed sampler
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        # Use tqdm on all ranks (with rank prefix)
        iterator = tqdm(train_loader, desc=f"Rank {rank} - Epoch {epoch + 1}/{config['num_epochs']}")
        
        for batch in iterator:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Rank {rank} - Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Rank {rank} - Epoch {epoch + 1}/{config['num_epochs']}, Validation Loss: {avg_val_loss}")

        # Log metrics per epoch on all ranks
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })


    model_to_save = model.module if world_size > 1 else model
    model_path = f"{config['model_path']}xrd_model_{wandb.run.id}.pth"
    torch.save(model_to_save.state_dict(), model_path)
    wandb.save(model_path)

    # Evaluation (only on rank 0)
    model.eval()
    test_loss, correct, total = 0, 0, 0
    top1_correct, top3_correct, top5_correct = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            top1_correct += compute_top_k_accuracy(outputs, targets, k=1)
            top3_correct += compute_top_k_accuracy(outputs, targets, k=3)
            top5_correct += compute_top_k_accuracy(outputs, targets, k=5)

    # Accuracy (make sure to divide by total!)
    test_accuracy = 100 * correct / total
    top1_accuracy = 100 * top1_correct / total
    top3_accuracy = 100 * top3_correct / total
    top5_accuracy = 100 * top5_correct / total

    print(f"Total samples: {total}")

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    # Log final test metrics
    wandb.log({
        "total_samples": total,
        "test_loss": test_loss,
        "test_accuracy_top1": top1_accuracy,
        "test_accuracy_top3": top3_accuracy,
        "test_accuracy_top5": top5_accuracy,
    })

    wandb.finish()

def run_sweep(sweep_id, config, rank, world_size):
    print(f"Using Sweep ID: {sweep_id}")
    
    def train_wrapper():
        # Data loading with distributed parameters
        data_path = config["parameters"]["data_path"]["value"]
        batch_size = config["parameters"]["batch_size"]["value"]

        train_loader, val_loader, test_loader, num_classes, intensity_points = get_dataloaders(
            data_path, batch_size=batch_size, world_size=world_size, rank=rank,
            num_workers=config["parameters"]["num_workers"]["value"], 
            num_classes=config["parameters"]["num_classes"]["value"], 
            prefetch_factor=config["parameters"]["prefetch_factor"]["value"]
        )

        # Set device to current XPU device
        device = torch.device(f"xpu:{rank}" if torch.xpu.is_available() else "cpu")
        
        train(config, train_loader, val_loader, test_loader, device, rank, world_size)
    
    # Run wandb agent on all ranks
    wandb.agent(sweep_id, function=train_wrapper, count=5, project="sweep-runs")

def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup distributed training
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group(backend='ccl', rank=rank, world_size=world_size)
        torch.xpu.set_device(local_rank)

    # Run sweep
    run_sweep(args.sweep_id, config, rank, world_size)

if __name__ == "__main__":
    main()