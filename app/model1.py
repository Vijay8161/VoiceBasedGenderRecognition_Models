import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(file_paths):
    all_X = []
    all_y = []
    for file_path in file_paths:
        data = np.load(file_path)
        all_X.append(data['X'])
        all_y.append(data['y'])
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(LabelEncoder().fit_transform(y), dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            noise = torch.randn_like(x) * 0.1  
            x += noise

        return x, y

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        scores = self.attention_weights(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)
        weighted_sum = torch.bmm(weights, x).squeeze(1)
        return weighted_sum, weights

class HybridModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, transformer_dim, dropout=0.5):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.attention = AttentionLayer(lstm_hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4, dropout=dropout, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_out, _ = self.attention(lstm_out)
        transformer_out = self.transformer(attention_out.unsqueeze(1)).squeeze(1)
        output = self.fc(transformer_out)
        return output

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with loss {loss}.")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")
    return model, optimizer, start_epoch

def compute_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(np.float32)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    return accuracy, precision, recall, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    best_val_loss = float('inf')
    current_checkpoint_path = os.path.join(save_path, 'current_model.pth')
    best_checkpoint_path = os.path.join(save_path, 'best_model.pth')

    model, optimizer, start_epoch = load_checkpoint(model, optimizer, current_checkpoint_path)
    
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training') as progress:
            for inputs, labels in progress:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(outputs.detach().cpu().numpy())
                progress.set_postfix(loss=train_loss / len(train_loader))

        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(
            np.array(all_train_labels), np.array(all_train_preds)
        )

        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation') as progress:
                for inputs, labels in progress:
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(outputs.cpu().numpy())
                    progress.set_postfix(val_loss=val_loss / len(val_loader))

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(
            np.array(all_val_labels), np.array(all_val_preds)
        )

        save_checkpoint(model, optimizer, epoch, train_loss, current_checkpoint_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, best_checkpoint_path)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Best Val Loss: {best_val_loss:.4f}\n")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.show()

def main():
    os.makedirs('model_checkpoints', exist_ok=True)

    # file_paths = ['normalized_reg.npz', 'normalized_aug.npz']
    data_dir = r"C:\Users\vijay\DLProj\data"
    file_paths = [
        os.path.join(data_dir, 'normalized_reg.npz'),
        os.path.join(data_dir, 'normalized_aug.npz')
    ]

    # file_paths = [
    #     os.path.join('..', 'data', 'normalized_reg.npz'),
    #     os.path.join('..', 'data', 'normalized_aug.npz')
    # ]

    X, y = load_data(file_paths)
    epsilon = 1e-8
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0)+epsilon)

    dataset = AudioDataset(X, y, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=75, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=75, shuffle=False)

    model = HybridModel(input_dim=X.shape[1], lstm_hidden_dim=64, transformer_dim=64, dropout=0.6).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=7, save_path='model_checkpoints')

if __name__ == '__main__':
    main()
