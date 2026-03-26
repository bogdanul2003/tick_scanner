"""
LSTM-based Time Series Forecaster for MACD prediction.

This module provides a PyTorch LSTM model that can be trained on MACD data
and exported to Core ML for NPU inference on Apple Silicon.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import os


class LSTMForecaster(nn.Module):
    """
    LSTM model for time series forecasting (MACD values).
    
    Architecture:
    - Input: sequence of MACD values (batch, seq_len, 1)
    - LSTM layers with dropout
    - Fully connected output layer
    - Output: forecasted values (batch, forecast_horizon)
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5
    ):
        """
        Initialize the LSTM forecaster.
        
        Args:
            input_size: Number of input features (1 for univariate MACD)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
            forecast_horizon: Number of steps to forecast
        """
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Forecasted values of shape (batch, forecast_horizon)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last timestep's output
        last_output = lstm_out[:, -1, :]
        
        # Project to forecast horizon
        forecast = self.fc(last_output)
        
        return forecast


class MACDForecasterTrainer:
    """Trainer for the LSTM MACD forecaster."""
    
    def __init__(
        self,
        seq_length: int = 30,
        forecast_horizon: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            seq_length: Length of input sequences
            forecast_horizon: Number of days to forecast
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu', 'mps', or 'cuda')
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        # Auto-select device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = LSTMForecaster(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For normalization
        self.mean = 0.0
        self.std = 1.0
    
    def prepare_sequences(
        self, 
        data: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training sequences from time series data.
        
        Args:
            data: 1D numpy array of MACD values
            
        Returns:
            Tuple of (X, y) tensors for training
        """
        # Normalize
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-8
        normalized = (data - self.mean) / self.std
        
        X, y = [], []
        for i in range(len(normalized) - self.seq_length - self.forecast_horizon + 1):
            X.append(normalized[i:i + self.seq_length])
            y.append(normalized[i + self.seq_length:i + self.seq_length + self.forecast_horizon])
        
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
        y = torch.tensor(np.array(y), dtype=torch.float32)  # (N, forecast_horizon)
        
        return X, y
    
    def train(
        self,
        train_data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> dict:
        """
        Train the model on MACD data.
        
        Args:
            train_data: 1D numpy array of MACD values
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        X, y = self.prepare_sequences(train_data)
        
        # Split into train/val
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train))
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            train_loss = total_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = self.criterion(val_pred, y_val).item()
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return history
    
    def evaluate(
        self,
        test_data: np.ndarray,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate the model on held-out test data.
        
        Args:
            test_data: 1D numpy array of MACD values for testing
            verbose: Print evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        X_test, y_test = self.prepare_sequences(test_data)
        
        if len(X_test) == 0:
            return {"error": "Not enough test data"}
        
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)
        
        # Compute metrics
        mse = torch.mean((predictions - y_test) ** 2).item()
        mae = torch.mean(torch.abs(predictions - y_test)).item()
        rmse = np.sqrt(mse)
        
        # Denormalize for interpretable metrics
        pred_np = predictions.cpu().numpy() * self.std + self.mean
        actual_np = y_test.cpu().numpy() * self.std + self.mean
        
        mae_denorm = np.mean(np.abs(pred_np - actual_np))
        rmse_denorm = np.sqrt(np.mean((pred_np - actual_np) ** 2))
        
        # Directional accuracy: did we correctly predict if MACD will increase/decrease?
        # Compare last input value to first forecasted value
        X_test_np = X_test.cpu().numpy()
        last_input = X_test_np[:, -1, 0] * self.std + self.mean  # Last value of input sequence
        first_pred = pred_np[:, 0]  # First forecasted value
        first_actual = actual_np[:, 0]  # First actual value
        
        pred_direction = (first_pred > last_input).astype(int)
        actual_direction = (first_actual > last_input).astype(int)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        # Positive prediction accuracy: when MACD is negative, did we correctly predict positive?
        negative_mask = last_input < 0
        if np.sum(negative_mask) > 0:
            pred_positive = first_pred[negative_mask] > 0
            actual_positive = first_actual[negative_mask] > 0
            positive_accuracy = np.mean(pred_positive == actual_positive)
        else:
            positive_accuracy = None
        
        metrics = {
            "mse_normalized": mse,
            "mae_normalized": mae,
            "rmse_normalized": rmse,
            "mae": float(mae_denorm),
            "rmse": float(rmse_denorm),
            "directional_accuracy": float(directional_accuracy),
            "positive_prediction_accuracy": float(positive_accuracy) if positive_accuracy else None,
            "test_samples": len(X_test)
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print("Test Evaluation Results")
            print("=" * 50)
            print(f"Test samples: {metrics['test_samples']}")
            print(f"MAE (Mean Absolute Error): {metrics['mae']:.6f}")
            print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.6f}")
            print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
            if metrics['positive_prediction_accuracy'] is not None:
                print(f"Positive Prediction Accuracy: {metrics['positive_prediction_accuracy']:.2%}")
            print("=" * 50)
        
        return metrics
    
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Make a prediction given an input sequence.
        
        Args:
            sequence: 1D array of length seq_length
            
        Returns:
            Forecasted values (forecast_horizon,)
        """
        self.model.eval()
        
        # Normalize using training stats
        normalized = (sequence - self.mean) / self.std
        
        # Prepare input
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x)
        
        # Denormalize
        pred_np = pred.cpu().numpy()[0]
        forecast = pred_np * self.std + self.mean
        
        return forecast
    
    def save(self, path: str):
        """Save model and normalization parameters."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "mean": self.mean,
            "std": self.std,
            "seq_length": self.seq_length,
            "forecast_horizon": self.forecast_horizon,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and normalization parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.seq_length = checkpoint["seq_length"]
        self.forecast_horizon = checkpoint["forecast_horizon"]
        print(f"Model loaded from {path}")
    
    def export_to_coreml(self, output_path: str) -> str:
        """
        Export the model to Core ML format for NPU inference.
        
        Args:
            output_path: Path to save the .mlpackage
            
        Returns:
            Path to the saved Core ML model
        """
        import coremltools as ct
        
        self.model.eval()
        self.model.to("cpu")
        
        # Create example input
        example_input = torch.randn(1, self.seq_length, 1)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, example_input)
        
        # Convert to Core ML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, self.seq_length, 1), name="input_sequence")],
            outputs=[ct.TensorType(name="forecast")],
            compute_precision=ct.precision.FLOAT16,  # Use FP16 for NPU efficiency
            compute_units=ct.ComputeUnit.ALL  # Allow NPU, GPU, and CPU
        )
        
        # Add metadata
        mlmodel.author = "Tick Scanner"
        mlmodel.short_description = "LSTM-based MACD forecaster for time series prediction"
        mlmodel.version = "1.0"
        
        # Save with normalization parameters in metadata
        mlmodel.user_defined_metadata["mean"] = str(self.mean)
        mlmodel.user_defined_metadata["std"] = str(self.std)
        mlmodel.user_defined_metadata["seq_length"] = str(self.seq_length)
        mlmodel.user_defined_metadata["forecast_horizon"] = str(self.forecast_horizon)
        
        mlmodel.save(output_path)
        print(f"Core ML model saved to {output_path}")
        
        return output_path


def get_model_path(signal_type: str = "macd") -> str:
    """
    Get the path to the Core ML model.
    
    Args:
        signal_type: Type of signal model - "macd" or "signal_line"
        
    Returns:
        Path to the .mlpackage file
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "models", f"{signal_type}_forecaster.mlpackage")


def get_pytorch_model_path(signal_type: str = "macd") -> str:
    """
    Get the path to the PyTorch model checkpoint.
    
    Args:
        signal_type: Type of signal model - "macd" or "signal_line"
        
    Returns:
        Path to the .pt file
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "models", f"{signal_type}_forecaster.pt")
