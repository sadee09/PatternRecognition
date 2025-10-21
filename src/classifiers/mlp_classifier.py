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
from config.config import MLP_HIDDEN_SIZES, MLP_LEARNING_RATE, MLP_MAX_EPOCHS, MLP_BATCH_SIZE, MLP_DROPOUT_RATE
from src.utils.data_loader import load_mnist_data


class MLPClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=MLP_HIDDEN_SIZES, num_classes=10, dropout_rate=MLP_DROPOUT_RATE):

        super(MLPClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build the network layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # ReLU nonlinearity
        layers.append(nn.Dropout(dropout_rate))  # Dropout for regularization
        
        # Additional hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # ReLU after each hidden layer
            layers.append(nn.Dropout(dropout_rate))  # Dropout after each hidden layer
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.network(x)


def train_mlp_classifier(X_train, y_train, hidden_sizes=MLP_HIDDEN_SIZES, 
                        learning_rate=MLP_LEARNING_RATE, max_epochs=MLP_MAX_EPOCHS, 
                        batch_size=MLP_BATCH_SIZE, dropout_rate=MLP_DROPOUT_RATE, 
                        X_val=None, y_val=None, verbose=True):
    """
    Train MLP classifier using SGD optimizer.
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = MLPClassifier(input_size, hidden_sizes, num_classes, dropout_rate)
    
    # Loss function and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
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


def evaluate_mlp_performance(X_train, X_test, y_train, y_test, hidden_sizes=MLP_HIDDEN_SIZES,
                           learning_rate=MLP_LEARNING_RATE, max_epochs=MLP_MAX_EPOCHS,
                           batch_size=MLP_BATCH_SIZE, dropout_rate=MLP_DROPOUT_RATE):
    """
    Evaluate MLP performance.
    """
    print("Starting MLP evaluation...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Architecture: {X_train.shape[1]} -> {' -> '.join(map(str, hidden_sizes))} -> {len(np.unique(y_train))}")
    
    # Train model
    start_time = time.time()
    model, train_losses = train_mlp_classifier(
        X_train, y_train, hidden_sizes=hidden_sizes, 
        learning_rate=learning_rate, max_epochs=max_epochs,
        batch_size=batch_size, dropout_rate=dropout_rate, verbose=True
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
    
    print(f"MLP - Accuracy: {accuracy:.4f}, "
          f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")
    
    print("Class-wise accuracies:")
    for class_label, acc in class_accuracies.items():
        print(f"  Class {class_label}: {acc:.4f}")
    
    return results


if __name__ == "__main__":
    # Load MNIST data
    print("Loading MNIST data for MLP...")
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Evaluate MLP performance with architecture
    # Example: 784 → 256 → 128 → 10
    hidden_sizes = [256, 128]  # At least one hidden layer with ReLU
    
    results = evaluate_mlp_performance(
        X_train, X_test, y_train, y_test, hidden_sizes=hidden_sizes
    )
    
    print("\n" + "="*60)
    print("MLP RESULTS SUMMARY")
    print("="*60)
    
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Fit Time: {results['fit_time']:.2f}s")
    print(f"Predict Time: {results['predict_time']:.2f}s")