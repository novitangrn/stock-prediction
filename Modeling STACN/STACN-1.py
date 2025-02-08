import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# **Dataset STACN**
class STACN_Dataset(Dataset):
    def __init__(self, df, lookback=10, forecast=1, target_col="close", scaler=None, close_scaler=None):
        self.data = []
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.close_scaler = close_scaler if close_scaler is not None else StandardScaler()

        df = df.copy()
        df["return_close"] = df["close"].pct_change().fillna(0)

        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(df[["close", "open", "high", "low"]].values)
        if not hasattr(self.close_scaler, 'mean_'):
            self.close_scaler.fit(df[["close"]].values)

        df[["close", "open", "high", "low"]] = self.scaler.transform(df[["close", "open", "high", "low"]].values)

        for i in range(lookback, len(df) - forecast):
            thought_vector = np.array(df.iloc[i]["thought_vector"])
            historical_data = df.iloc[i-lookback:i][["close", "open", "high", "low"]].values
            target = df.iloc[i+forecast][target_col]

            if not (np.isnan(historical_data).any() or np.isnan(target)):
                self.data.append((thought_vector, historical_data, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        thought_vector, historical_data, target = self.data[idx]
        return (torch.tensor(thought_vector, dtype=torch.float32),
                torch.tensor(historical_data, dtype=torch.float32),
                torch.tensor([target], dtype=torch.float32))

# **Collate function untuk padding**
def collate_fn(batch):
    thought_vectors, historical_data, targets = zip(*batch)
    max_threshold = max(tv.shape[0] for tv in thought_vectors)
    padded_thought_vectors = []

    for tv in thought_vectors:
        pad_size = max_threshold - tv.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, tv.shape[1]), dtype=torch.float32)
            tv = torch.cat([tv, pad], dim=0)
        padded_thought_vectors.append(tv)

    return (torch.stack(padded_thought_vectors),
            torch.stack(historical_data),
            torch.stack(targets))

# **Model STACN**
class STACN(nn.Module):
    def __init__(self, input_size=4, hidden_size=512, num_layers=2, cnn_out_channels=128, dropout=0.01):
        super(STACN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv3d(4, cnn_out_channels, kernel_size=(3,5,5), stride=(1,1,2), padding=(1,2,2)),
            nn.BatchNorm3d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        )

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_channels + hidden_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, thought_vectors, stock_features):
        if len(thought_vectors.shape) == 4:
            thought_vectors = thought_vectors.unsqueeze(1)
        thought_vectors = thought_vectors.permute(0, 4, 1, 2, 3)

        x_news = self.cnn(thought_vectors)
        x_news = torch.mean(x_news, dim=[2,3,4])

        x_stock, _ = self.lstm(stock_features)
        x_stock = x_stock[:, -1, :]

        x_combined = torch.cat([x_news, x_stock], dim=1)
        output = self.fc(x_combined)
        return output

# **1️⃣ Persiapkan Device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **2️⃣ Normalisasi Data**
price_scaler = StandardScaler()
close_scaler = StandardScaler()

all_prices = np.concatenate([df[["close", "open", "high", "low"]].values for df in synced_data.values()])
all_close_prices = np.concatenate([df[["close"]].values for df in synced_data.values()]).reshape(-1, 1)  # Ensure 2D

price_scaler.fit(all_prices)
close_scaler.fit(all_close_prices)

# **3️⃣ Fungsi untuk Training Model**
def train_model(sector, train_loader, val_loader, close_scaler, device, num_epochs=100, patience=10):
    """Fungsi untuk melatih model per sektor"""
    model = STACN().to(device)
    trainer = ModelTrainer(model, device, close_scaler)

    best_val_loss = float('inf')
    best_model_state = None
    no_improve = 0  # Counter untuk Early Stopping

    for epoch in range(num_epochs):
        trainer.train(train_loader)
        mse, mae, predictions, actuals = trainer.validate(val_loader)
        val_loss = trainer.val_losses[-1]

        # **Cek apakah model membaik**
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_predictions = predictions
            best_actuals = actuals
            no_improve = 0  # Reset counter
        else:
            no_improve += 1

        print(f"Sektor {sector} - Epoch {epoch+1}: MSE: {mse:.6f}, MAE: {mae:.6f}, "
              f"Train Loss: {trainer.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")

        # **Cek Early Stopping**
        if no_improve >= patience:
            print(f"Early stopping for {sector} at epoch {epoch+1}")
            break

    # **Simpan Model Terbaik**
    if best_model_state:
        torch.save(best_model_state, f"best_model_{sector}.pth")
        print(f"Best model saved for {sector}")

    # **Visualisasi Training & Validation Loss**
    plt.figure(figsize=(12, 5))

    # **Plot Loss**
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Losses - {sector}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # **Plot Prediksi vs Aktual**
    plt.subplot(1, 2, 2)
    plt.plot(best_actuals, label='Actual Values', color='blue', alpha=0.7)
    plt.plot(best_predictions, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'Predicted vs Actual Values - {sector}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'visualization_{sector}.png')
    plt.show()

# **4️⃣ Buat DataLoader untuk Setiap Sektor**
data_loaders = {}
for sector, df in synced_data.items():
    dataset = STACN_Dataset(df=df, lookback=10, forecast=1, target_col='close', scaler=price_scaler, close_scaler=close_scaler)

    # **Pisahkan data 80% train, 20% validasi (tanpa shuffle)**
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    data_loaders[sector] = {'train': train_loader, 'val': val_loader}

# **5️⃣ Training Model untuk Semua Sektor**
for sector, loaders in data_loaders.items():
    print(f"\nTraining model for sector: {sector}")
    train_model(sector, loaders['train'], loaders['val'], close_scaler, device)

print("Training completed for all sectors!")
