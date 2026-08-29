import torch
import wandb
import json
import argparse
from resnet_model import ResnetClassifier
from dataset import get_dataloaders
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import math
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
    
def train(config, device):
    config = wandb.config

    with wandb.init(project="project-cnn", config=config) as run:
        config = wandb.config
        lr = config.learning_rate
        lr_str = f"{lr/10**int(math.log10(lr)):.0f}e{int(math.log10(lr))}"
        run_name = (
        f"BS-{config.batch_size}_"
        f"EPS-{config.num_epochs}_"
        f"LR-{lr_str}_"
        f"wd-{config.weight_decay}_"
        f"opt-{config.optimizer}_"
        f"dropout-{config.dropout}_"
        f"id-{run.id}")
        wandb.run.name = run_name

        data_path = config["data_path"]
        batch_size = config["batch_size"]

        train_loader, val_loader, test_loader, num_classes, intensity_points = get_dataloaders(
            data_path, batch_size=config['batch_size'], num_workers=config["num_workers"], num_classes=config["num_classes"], prefetch_factor=config["prefetch_factor"]
        )

        # Initialize model
        model = ResnetClassifier(
            input_dim = config['input_dim'],
            res_dims = config['res_dims'],
            res_kernel = config['res_kernel'],
            res_stride = config['res_stride'],
            num_blocks = config['num_blocks'],
            first_kernel_size = config['first_kernel_size'],
            first_stride = config['first_stride'],
            first_pool_kernel_size = config['first_pool_kernel_size'],
            first_pool_stride = config['first_pool_stride'],
            num_classes= config['num_classes'],
        ).to(device)
        
        weight_decay = float(config["weight_decay"])
        learning_rate = float(config["learning_rate"])

        if config["optimizer"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Optimize the model with IPEX for better performance
      #  model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(config["num_epochs"]):
            model.train()
            running_loss = 0.0
            train_correct, train_total = 0,0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy:.2f}%")

            # Validation loop
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Accuracy
                    _, predicted = torch.max(outputs, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.2f}%")


            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            })

        # Save model after training
        model_path = f"{config['model_path']}xrd_{config['model_type']}_{wandb.run.id}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        # Evaluation
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

def run_sweep(sweep_id, config, device):
    print(f"Using Sweep ID: {sweep_id}")
    #sweep_id = wandb.sweep(config, project="XRD_ViT_Code_sweep")
    #wandb.init(project="project", config=config)
    wandb.agent(sweep_id, function=lambda: train(config, device), count=2, project="project-cnn") #COUNT = Amount of Runs

def main():
    args = parse_args()
    config = load_config(args.config)

    # Data loading
    #data_path = config["parameters"]["data_path"]["value"]
    #batch_size = config["parameters"]["batch_size"]["value"]

    #train_loader, val_loader, test_loader, num_classes, intensity_points = get_dataloaders(
      #  data_path, batch_size=batch_size, num_workers=config["parameters"]["num_workers"]["value"], num_classes=config["parameters"]["num_classes"]["value"], prefetch_factor=config["parameters"]["prefetch_factor"]["value"]
   # )

    # Set device to XPU (Intel GPU) or CPU fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run sweep
    run_sweep(args.sweep_id, config, device)

if __name__ == "__main__":
    main()
