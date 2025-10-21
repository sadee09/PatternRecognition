import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import sys
import os

# Simple path addition
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config.config import CNN_LEARNING_RATE, CNN_MAX_EPOCHS, CNN_BATCH_SIZE, CNN_DROPOUT_RATE
from src.utils.data_loader import load_mnist_for_cnn


class CNNClassifier(nn.Module):
    
    def __init__(self, num_classes=10, dropout_rate=CNN_DROPOUT_RATE):

        super(CNNClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Conv(1→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1→32 channels, 3×3 kernel
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for better training
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32→64 channels, 3×3 kernel
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size after convolutions and pooling
        # Input: 28x28 -> Conv1 -> 28x28 -> Pool -> 14x14
        # -> Conv2 -> 14x14 -> Pool -> 7x7
        # So final size is 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        """
        # First convolutional block: Conv(1→32, 3×3) → BatchNorm → ReLU → MaxPool
        x = self.conv1(x)  # (batch_size, 32, 28, 28)
        x = self.bn1(x)    # Batch normalization
        x = self.relu(x)
        x = self.pool(x)   # (batch_size, 32, 14, 14)
        
        # Second convolutional block: Conv(32→64, 3×3) → BatchNorm → ReLU → MaxPool
        x = self.conv2(x)  # (batch_size, 64, 14, 14)
        x = self.bn2(x)    # Batch normalization
        x = self.relu(x)
        x = self.pool(x)   # (batch_size, 64, 7, 7)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 64*7*7)
        
        # Fully connected layers
        x = self.fc1(x)   # (batch_size, 128)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)   # (batch_size, num_classes)
        
        return x


def train_cnn_classifier(X_train, y_train, learning_rate=CNN_LEARNING_RATE, 
                        max_epochs=CNN_MAX_EPOCHS, batch_size=CNN_BATCH_SIZE, 
                        dropout_rate=CNN_DROPOUT_RATE, verbose=True):
    """
    Train CNN classifier using cross-entropy loss.
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    num_classes = len(np.unique(y_train))
    model = CNNClassifier(num_classes, dropout_rate)
    
    # Loss function and optimizer - Assignment requirement: Cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {avg_train_loss:.4f}")
    
    return model, train_losses


def evaluate_cnn_performance(X_train, X_test, y_train, y_test):
    """
    Evaluate CNN performance.
    """
    print("Starting CNN evaluation...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train.shape[1:]}")
    
    # Train model
    start_time = time.time()
    model, train_losses = train_cnn_classifier(
        X_train, y_train, verbose=True
    )
    fit_time = time.time() - start_time
    
    # Make predictions on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        start_time = time.time()
        test_outputs = model(X_test_tensor)
        predictions = torch.argmax(test_outputs, dim=1).numpy()
        predict_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    
    # Get class-wise accuracy
    class_accuracies = {}
    for class_label in np.unique(y_test):
        class_mask = (y_test == class_label)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predictions[class_mask] == y_test[class_mask])
            class_accuracies[class_label] = class_acc
    
    results = {
        'accuracy': accuracy,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'train_losses': train_losses,
        'predictions': predictions,
        'class_accuracies': class_accuracies,
        'model': model
    }
    
    print(f"CNN - Accuracy: {accuracy:.4f}, "
          f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")
    
    print("Class-wise accuracies:")
    for class_label, acc in class_accuracies.items():
        print(f"  Class {class_label}: {acc:.4f}")
    
    return results


if __name__ == "__main__":
    # Load MNIST data for CNN (keeps 2D shape)
    print("Loading MNIST data for CNN...")
    X_train, X_test, y_train, y_test = load_mnist_for_cnn()
    
    # Evaluate CNN performance
    results = evaluate_cnn_performance(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("CNN RESULTS SUMMARY")
    print("="*60)
    
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Fit Time: {results['fit_time']:.2f}s")
    print(f"Predict Time: {results['predict_time']:.2f}s")