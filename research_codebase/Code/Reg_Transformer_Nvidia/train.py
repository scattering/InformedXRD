import torch
import wandb
import json
import argparse
from model import transformer_model
from dataset import get_dataloaders
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ExponentialLR, CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--sweep", action="store_true", help="Flag to indicate if this is a sweep run")
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel (DDP)")
    parser.add_argument("--node_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--sweep_id", type=str, required=True, help="W&B Sweep ID")
    parser.add_argument("--num_sweeps", type=int, required=True, help="W&B Number of Sweeps")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def get_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    scheduler_type = config.get("scheduler_type", None)
    
    if scheduler_type == "multistep":
        milestones = config.get("lr_milestones", [3, 5, 7])  # epochs to reduce LR
        gamma = config.get("lr_gamma", 0.2)  # factor to reduce LR by
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == "step":
        step_size = config.get("lr_step_size", 10)
        gamma = config.get("lr_gamma", 0.2)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == "exponential":
        gamma = config.get("lr_gamma", 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == "cosine":
        T_max = config.get("T_max", config["num_epochs"])
        eta_min = config.get("eta_min", 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    else:
        return None

def compute_top_k_accuracy(output, target, k=1):
    """Compute top-k accuracy: fraction of samples where true label is in top-k"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct_k = correct.any(dim=1).float().sum().item()  # only 1 per sample
        return correct_k

def broadcast_config(config, rank):
    if rank == 0:
        obj_list = [config]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

# At module level
def set_global_context(train_loader, val_loader, test_loader, device, rank, local_rank, distributed):
    global _global_train_loader, _global_val_loader, _global_test_loader
    global _global_device, _global_rank, _global_local_rank, _global_distributed

    _global_train_loader = train_loader
    _global_val_loader = val_loader
    _global_test_loader = test_loader
    _global_device = device
    _global_rank = rank
    _global_local_rank = local_rank
    _global_distributed = distributed


def train(config, train_loader, val_loader, test_loader, device, rank, local_rank, distributed):
    if rank == 0:
        wandb.init(project="08-17", config=config)
        run = wandb.run
        config = wandb.config  # Get hyperparameters from sweep
    else:
        run = None

    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
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
        pos_encoding=config["pos_encoding"],
    ).to(device)

    model = model.to(device)
   
    # Only trace the model on rank 0 (avoid DDP duplicates)
    if rank == 0:
        wandb.watch(model, log="all", log_freq=100)  # Can change to "gradients" if overhead is high

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer setup
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # summary(model, input_size=(config["parameters"]["batch_size"]["value"], 1, config["spec_length"]))
    
    scheduler = get_scheduler(optimizer, config)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}")

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

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else config["learning_rate"]
        else:
            current_lr = config["learning_rate"]
            
        if rank == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}")
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Loss: {avg_val_loss}")
            print(f"End of epoch {epoch+1}, New LR = {current_lr}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            })

    if rank == 0:
        # Save model after training
        model_path = f"{config['model_path']}xrd_model_{wandb.run.id}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        # Evaluation
        model.eval()
        test_loss, correct, total = 0, 0, 0
        top1_correct, top3_correct, top5_correct = 0, 0, 0

        # Confusion Matrix
        all_preds = []
        all_targets = []
        all_top5_preds = []

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

                # Collect predictions and targets for confusion matrix
                all_preds.append(predicted.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                # For top-5 confusion matrix:
                top5_preds = torch.topk(outputs, k=5, dim=1).indices.cpu().numpy()
                targets_np = targets.cpu().numpy()
                top5_selected_preds = []
                for i in range(targets_np.shape[0]):
                    if targets_np[i] in top5_preds[i]:
                        # pick ground truth label as predicted (simulate "correct")
                        top5_selected_preds.append(targets_np[i])
                    else:
                        # pick top-1 prediction (simulate "incorrect")
                        top5_selected_preds.append(top5_preds[i][0])
                all_top5_preds.append(np.array(top5_selected_preds))

                


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

        # Concatenate all predictions and targets for confusion matrix
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_top5_preds = np.concatenate(all_top5_preds)

        # Log final test metrics
        wandb.log({
            "total_samples": total,
            "test_loss": test_loss,
            "test_accuracy_top1": top1_accuracy,
            "test_accuracy_top3": top3_accuracy,
            "test_accuracy_top5": top5_accuracy,
        })


    if rank == 0:
        wandb.finish()



def run_sweep(sweep_id, config, train_loader, val_loader, test_loader, device,
              num_sweeps, rank, local_rank, distributed):
    print(f"Using Sweep ID: {sweep_id}")

    if rank == 0:
        # This lambda keeps your train_entry_point exactly as-is, just partially applied
        wandb.agent(
            sweep_id,
            function=lambda: train_entry_point(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                rank=rank,
                local_rank=local_rank,
                distributed=distributed
            ),
            count=num_sweeps,
            project="08-17"
        )
    else:
        # All other ranks receive the config via broadcast and run training
        for _ in range(num_sweeps):
            broadcasted_config = broadcast_config(None, rank)
            train(
                broadcasted_config,
                train_loader,
                val_loader,
                test_loader,
                device,
                rank,
                local_rank,
                distributed
            )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def train_entry_point(config=None, train_loader=None, val_loader=None, test_loader=None,
                      device=None, rank=None, local_rank=None, distributed=None):
    run = wandb.init() 

    if config is None:
        config = dict(wandb.config)  # Only fetch from wandb if not passed explicitly

    # Broadcast to other ranks
    broadcasted_config = broadcast_config(dict(config), rank)


    # Run training with the provided config and data loaders
    train(
        config,
        train_loader,
        val_loader,
        test_loader,
        device,
        rank,
        local_rank,
        distributed
    )


def main():
    
    args = parse_args()
    config = load_config(args.config)

    # Global rank
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    print("local rank: " + str(local_rank))

    # Wait for master TCP port (if not rank 0)
    #wait_for_master()

    if args.distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0

    # sanity test of DDP connectivity
    #test_internode_connectivity()

    # Infer num_workers based on CPU cores per process
    total_cores = os.cpu_count() or 16  # fallback to 16 if None
    max_workers = 16
    num_workers = min(max_workers, max(1, total_cores // world_size))
    num_workers=8
    

    # Data loading
    data_path = config["parameters"]["data_path"]["value"]
    batch_size = config["parameters"]["batch_size"]["value"]

    train_loader, val_loader, test_loader, num_classes, intensity_points = get_dataloaders(
        data_path, 
        batch_size=batch_size, 
        world_size=world_size, 
        num_workers=num_workers, 
        num_classes=config["parameters"]["num_classes"]["value"], 
        prefetch_factor=config["parameters"]["prefetch_factor"]["value"]
    )


    run_sweep(args.sweep_id, config, train_loader, val_loader, test_loader, device, args.num_sweeps, rank, local_rank, args.distributed)


if __name__ == "__main__":
    main()
