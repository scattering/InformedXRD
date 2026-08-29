import torch
import wandb
import argparse
import json
import numpy as np
from collections import OrderedDict
import torch.nn as nn
from model import transformer_model
from dataset import get_dataloaders, get_dataloaders_test


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference")

    # File paths
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--project", required=True, help="Name of the W&B project")

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    config = config["parameters"]
    return config

def compute_top_k_accuracy(output, target, k=1):
    """Compute top-k accuracy: fraction of samples where true label is in top-k"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct_k = correct.any(dim=1).float().sum().item()  # only 1 per sample
        return correct_k

def infer(model, test_loader, device):
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
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

def main(args):
    # --- Init W&B run ---
    config = load_config(args.config)
    run = wandb.init(project=args.project, config=config)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    data_path = config["data_path"]["value"]
    batch_size = config["batch_size"]["value"]
    num_workers = config["num_workers"]["value"]
    world_size = 1

    _, _, test_loader, num_classes, intensity_points = get_dataloaders_test(
        data_path, 
        batch_size=batch_size, 
        world_size=world_size, 
        num_workers=num_workers, 
        num_classes=config["num_classes"]["value"], 
        prefetch_factor=config["prefetch_factor"]["value"]
    )

    sample_batch = next(iter(test_loader))
    inputs, targets = sample_batch
    actual_spec_length = inputs.shape[1]

    # --- Load model ---
    model = transformer_model(
        spec_length=actual_spec_length,
        num_output=config["num_classes"]["value"],
        embed_dim=config["embed_dim"]["values"][0],  # Note: values[0]!
        ff_dim=config["ff_dim"]["values"][0],
        depth=config["depth"]["values"][0],
        num_heads=config["num_heads"]["values"][0],
        mlp_units=config["mlp_units"]["values"][0],
        dropout=0,
        pos_encoding=config["pos_encoding"]["values"][0],
    )
    # Load checkpoint
    checkpoint = torch.load(config["checkpoint"]["value"], map_location=device)

    # Strip "module." prefix
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace("module.", "")] = v

    # Load weights
    model.load_state_dict(new_state_dict, strict=False)

    # Send to device
    model = model.to(device).eval()
    
    infer(model, test_loader, device)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
