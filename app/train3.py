import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from model3 import FinalModel3  # Import the FinalModel from model3.py

# Load normalized images from npz files
def load_data(file_paths):
    data_list = []
    label_list = []
    
    for file_path in file_paths:
        data = np.load(file_path)
        X = data['X']  
        y = data['y']  # Labels in string format

        X = X[:, :147]
        X = X.reshape(X.shape[0], 7, 7, 3)

        data_list.append(X)
        label_list.append(y)
    
    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(label_list, axis=0)

    X = np.transpose(X, (0, 3, 1, 2))  # Shape (N, 3, 7, 7)
    
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
    current_checkpoint_path = 'model_3_checkpoints/current_model.pth'
    best_checkpoint_path = 'model_3_checkpoints/best_model.pth'

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, best_checkpoint_path)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Adjust step_size and gamma

    criterion = nn.BCEWithLogitsLoss()
    
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
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

        # Print metrics
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}\n")

        save_checkpoint(model, optimizer, epoch + 1, val_loss / len(val_loader), current_checkpoint_path)

        if val_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_loss / len(val_loader)
            save_checkpoint(model, optimizer, epoch + 1, best_val_loss, best_checkpoint_path)

        # Step the scheduler
        scheduler.step()

    # Plot training and validation losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    os.makedirs('model_3_checkpoints', exist_ok=True)

    # file_paths = [
    #     'normalized_reg.npz',
    #     'normalized_aug.npz',
    # ]
    data_dir = r"C:\Users\vijay\DLProj\data"
    file_paths = [
        os.path.join(data_dir, 'normalized_reg.npz'),
        os.path.join(data_dir, 'normalized_aug.npz')
    ]
    X, y = load_data(file_paths)
    y = preprocess_labels(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FinalModel3(num_blocks=4, num_classes=1, dropout_rate=0.5).to(device)

    train_model(model, train_loader, val_loader, num_epochs=7, device=device)
