import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from model2 import FinalModel  

def load_data(file_paths):
    data_list = []
    label_list = []
    
    for file_path in file_paths:
        data = np.load(file_path)
        X = data['X']  
        y = data['y']  

        X = X[:, :147]  
        X = X.reshape(X.shape[0], 7, 7, 3) 

        data_list.append(X)
        label_list.append(y)
    
    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(label_list, axis=0)

    X = np.transpose(X, (0, 3, 1, 2))  
    
    return X, y

def preprocess_labels(y):
    label_mapping = {'male_masculine': 0, 'female_feminine': 1}
    return np.array([label_mapping[label] for label in y])

def compute_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(np.float32)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    return accuracy, precision, recall, f1

def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        return model, optimizer, epoch, best_val_loss
    return model, optimizer, 0, float('inf')

def train_model(model, train_loader, val_loader, num_epochs, device):
    best_val_loss = float('inf')
    current_checkpoint_path = 'model_2_checkpoints/current_model.pth'
    best_checkpoint_path = 'model_2_checkpoints/best_model.pth'

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, best_checkpoint_path)

    if os.path.exists(current_checkpoint_path):
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, current_checkpoint_path)

    criterion = nn.BCEWithLogitsLoss() 

    train_losses, val_losses = [], []
    train_metrics = []
    val_metrics = []

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()  # Ensure labels are float
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            y_train_pred = []
            y_train_true = []
            val_loss = 0.0

            for inputs, labels in tqdm(train_loader, desc="Train Evaluation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
                outputs = model(inputs)
                y_train_pred.append(outputs.cpu().numpy())
                y_train_true.append(labels.cpu().numpy())

            y_train_pred = np.concatenate(y_train_pred)
            y_train_true = np.concatenate(y_train_true)
            train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(y_train_true, y_train_pred)

            train_metrics.append((train_accuracy, train_precision, train_recall, train_f1))

            y_val_pred = []
            y_val_true = []

            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                y_val_pred.append(outputs.cpu().numpy())
                y_val_true.append(labels.cpu().numpy())

            val_losses.append(val_loss / len(val_loader))

            y_val_pred = np.concatenate(y_val_pred)
            y_val_true = np.concatenate(y_val_true)
            val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(y_val_true, y_val_pred)

            val_metrics.append((val_accuracy, val_precision, val_recall, val_f1))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, best_checkpoint_path)

            save_checkpoint(model, optimizer, epoch, val_loss, current_checkpoint_path)

            scheduler.step(val_loss)

        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")

    return train_losses, val_losses, train_metrics, val_metrics

def plot_metrics(train_losses, val_losses, train_metrics, val_metrics):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for i, label in enumerate(metrics_labels):
        plt.subplot(1, 2, 2)
        plt.plot([m[i] for m in train_metrics], label=f'Train {label}', linestyle='--')
        plt.plot([m[i] for m in val_metrics], label=f'Val {label}', linestyle='-')

    plt.title('Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    os.makedirs('model_2_checkpoints', exist_ok=True) 

    # file_paths = ['normalized_reg.npz', 'normalized_aug.npz'] 
    data_dir = r"C:\Users\vijay\DLProj\data"
    file_paths = [
        os.path.join(data_dir, 'normalized_reg.npz'),
        os.path.join(data_dir, 'normalized_aug.npz')
    ]
    X, y = load_data(file_paths)
    y = preprocess_labels(y)

    batch_size = 32
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FinalModel(num_blocks=3, num_classes=1, embed_size=32, heads=4, forward_expansion=4, dropout=0.6).to(device)

    num_epochs = 7  
    train_losses, val_losses, train_metrics, val_metrics = train_model(model, train_loader, val_loader, num_epochs, device)

    plot_metrics(train_losses, val_losses, train_metrics, val_metrics)
