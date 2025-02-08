import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# **Dataset STACN**
class STACN_Dataset(Dataset):
    def __init__(self, df, lookback=10, forecast=1, target_col="close", scaler=None, close_scaler=None):
        self.data = []
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.close_scaler = close_scaler if close_scaler is not None else StandardScaler()

        # Create a copy of the dataframe
        df = df.copy()

        if "return_close" not in df.columns:
            df["return_close"] = df["close"].pct_change().fillna(0)

        # Fit scalers if not already fit
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(df[["close", "open", "high", "low"]].values)

        if not hasattr(self.close_scaler, 'mean_'):
            self.close_scaler.fit(df[["close"]].values)

        # Transform the data
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, cnn_out_channels),
            nn.Tanh(),
            nn.Softmax(dim=1),
            nn.Dropout(dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_channels + hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
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

        attention_weights = self.attention(x_stock)
        x_news = x_news * attention_weights

        x_combined = torch.cat([x_news, x_stock], dim=1)
        output = self.fc(x_combined)

        return output


class ModelTrainer:
    def __init__(self, model, device, close_scaler, learning_rate=0.001, patience=3):
        self.model = model
        self.device = device
        self.close_scaler = close_scaler
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        self.patience = patience

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')  # Initialize best validation loss to infinity
        self.epochs_without_improvement = 0  # Counter for early stopping
        self.best_model_state = None  # Store best model state

    def train(self, train_loader):
        self.model.train()
        epoch_loss = 0
        for batch in train_loader:
            thought_vectors, stock_features, targets = [item.to(self.device) for item in batch]
            self.optimizer.zero_grad()
            outputs = self.model(thought_vectors, stock_features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        self.train_losses.append(epoch_loss / len(train_loader))

    def validate(self, val_loader):
        self.model.eval()
        epoch_loss = 0
        predictions, actuals = [], []

        with torch.no_grad():
            for batch in val_loader:
                thought_vectors, stock_features, targets = [item.to(self.device) for item in batch]
                outputs = self.model(thought_vectors, stock_features)
                loss = self.criterion(outputs, targets)
                epoch_loss += loss.item()

                # Properly reshape and transform back to original scale
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Reshape if needed
                if len(outputs_np.shape) == 1:
                    outputs_np = outputs_np.reshape(-1, 1)
                if len(targets_np.shape) == 1:
                    targets_np = targets_np.reshape(-1, 1)

                # Inverse transform
                outputs_np = self.close_scaler.inverse_transform(outputs_np)
                targets_np = self.close_scaler.inverse_transform(targets_np)

                predictions.extend(outputs_np.flatten())
                actuals.extend(targets_np.flatten())

        self.val_losses.append(epoch_loss / len(val_loader))

        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        return mse, mae, predictions, actuals


    def predict(self, val_loader):
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for batch in val_loader:
                thought_vectors, stock_features, targets = [item.to(self.device) for item in batch]
                outputs = self.model(thought_vectors, stock_features)

                outputs_np = outputs.cpu().numpy().reshape(-1, 1)
                targets_np = targets.cpu().numpy().reshape(-1, 1)

                outputs_np = self.close_scaler.inverse_transform(outputs_np)
                targets_np = self.close_scaler.inverse_transform(targets_np)

                predictions.extend(outputs_np.flatten())
                actuals.extend(targets_np.flatten())

        return predictions, actuals


# **Preprocessing dan Training Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loaders = {}

# Initialize scalers and fit them on all data
price_scaler = StandardScaler()
close_scaler = StandardScaler()

# Prepare all price data for fitting
all_prices = np.concatenate([df[["close", "open", "high", "low"]].values for df in synced_data.values()])
all_close_prices = np.concatenate([df[["close"]].values for df in synced_data.values()])

# Fit the scalers
price_scaler.fit(all_prices)
close_scaler.fit(all_close_prices)


for sector, df in synced_data.items():
    dataset = STACN_Dataset(df=df, lookback=10, forecast=1, target_col='close', scaler=price_scaler, close_scaler=close_scaler)
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=16, shuffle=False, collate_fn=collate_fn)

    data_loaders[sector] = {'train': train_loader, 'val': val_loader}

for sector, loaders in data_loaders.items():
    print(f"\nTraining model for sector: {sector}")
    model = STACN().to(device)
    trainer = ModelTrainer(model, device, close_scaler)

    best_val_loss = float('inf')
    best_model_state = None
    all_predictions = []
    all_actuals = []

    for epoch in range(100):  # Tetap berjalan hingga 100 epoch
        trainer.train(loaders['train'])
        mse, mae, predictions, actuals = trainer.validate(loaders['val'])
        val_loss = trainer.val_losses[-1]

        # Simpan model terbaik berdasarkan validasi loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            all_predictions = predictions
            all_actuals = actuals

        print(f"Saham {sector} - Epoch {epoch+1}: MSE: {mse:.6f}, MAE: {mae:.6f}, "
              f"Train Loss: {trainer.train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")

    # Simpan model terbaik untuk sektor ini
    if best_model_state is not None:
        torch.save(best_model_state, f"best_model_{sector}.pth")
        print(f"Best model saved for {sector}")

    # Plot training dan validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Losses - {sector}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot predicted vs actual values
    plt.subplot(1, 2, 2)
    plt.plot(all_actuals, label='Actual Values', color='blue', alpha=0.7)
    plt.plot(all_predictions, label='Predicted Values', color='red', alpha=0.7)
    plt.title(f'Predicted vs Actual Values - {sector}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Simpan visualisasi
    plt.savefig(f'visualization_{sector}.png')
    plt.close()

print("Training completed for all sectors!")
