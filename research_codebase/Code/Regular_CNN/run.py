import os
from dotenv import load_dotenv
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import h5py
import sys

# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Model definition
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, ff_dim, num_layers, mlp_units, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Linear(1, embed_size)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_size, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, mlp_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, n_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.mlp(x)

def train_wandb(model, train_loader, val_loader, criterion, optimizer, epochs=4):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loss and accuracy
        model.eval()
        val_loss = 0
        total = 0
        correct_top1, correct_top3, correct_top5 = 0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted_top1 = torch.max(outputs, 1)
                _, predicted_top3 = outputs.topk(3, 1, True, True)
                _, predicted_top5 = outputs.topk(5, 1, True, True)

                total += y_batch.size(0)
                correct_top1 += (predicted_top1 == y_batch).sum().item()
                correct_top3 += predicted_top3.eq(y_batch.view(-1, 1).expand_as(predicted_top3)).sum().item()
                correct_top5 += predicted_top5.eq(y_batch.view(-1, 1).expand_as(predicted_top5)).sum().item()

        val_acc_top1 = 100 * correct_top1 / total
        val_acc_top3 = 100 * correct_top3 / total
        val_acc_top5 = 100 * correct_top5 / total
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss/len(train_loader),
            "val_loss": val_loss/len(val_loader),
            "val_accuracy_top1": val_acc_top1,
            "val_accuracy_top3": val_acc_top3,
            "val_accuracy_top5": val_acc_top5,
        },
        step=epoch+1)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Top-1 Val Acc: {val_acc_top1:.2f}%, Top-3 Val Acc: {val_acc_top3:.2f}%, Top-5 Val Acc: {val_acc_top5:.2f}%")

def get_top5(model, val_loader):
    
    model.eval()
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)

            _, predicted_top5 = outputs.topk(5, 1, True, True)
            total += y_batch.size(0)
            correct_top5 += predicted_top5.eq(y_batch.view(-1, 1).expand_as(predicted_top5)).sum().item()
    
    return 100 * correct_top5 / total

#@wandbc.track_in_wandb()
def objective(trial):

    wandb.init(project="03-26-run", name=f"trial-{trial.number}", reinit=True)
    
    # Suggest values of the hyperparameters using a trial object.
    embed_size = trial.suggest_categorical('embed_size', [64, 128, 256, 512])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    ff_dim = trial.suggest_categorical('ff_dim', [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 8)
    dropout = trial.suggest_float('dropout', 0.20, 0.40)
    mlp_units = trial.suggest_categorical('mlp_units', [32, 64, 128, 256, 512])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 3, 6)

    model = TimeSeriesTransformer(
        input_dim=X_train.shape[1],
        embed_size=embed_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        mlp_units=mlp_units,
        dropout=dropout,
        n_classes=num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_wandb(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)
    
    val_accuracy = get_top5(model, val_loader)
    
    wandb.finish()
    return val_accuracy

if __name__ == "__main__":
    
    args = sys.argv
    device_id = int(args[1])
    torch.xpu.set_device(device_id)
    print(f"Using XPU device: {torch.xpu.current_device()}")

    load_dotenv()  # This loads the variables from the .env file
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    num_classes = 230
    device="xpu"
    file_path = "PowCOD_small_500_2theta_4.0_44.0_0.2_0.2.h5"
    with h5py.File(file_path, "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        X_val = f["X_val"][:]
        y_val = f["y_val"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]

    # Reshape and convert to float32
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1)).astype(np.float32)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)).astype(np.float32)

    # Cross entropy loss requires 0 indices

    y_train = y_train - 1  # Convert range 1-230 → 0-229
    y_test = y_test - 1
    y_val = y_val - 1

    batch_size = 16
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.long)


    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = TensorDataset(X_train, y_train)
    val_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
   
    
    study = optuna.load_study(
        study_name="03-26-max-run",
        storage="mysql://root:@localhost:3307/example?unix_socket=/home1/ALLOC1/UserA/mysql/mysql.sock"
    )
    study.optimize(objective, n_trials=8)



