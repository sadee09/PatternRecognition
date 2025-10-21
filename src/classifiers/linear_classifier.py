import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

# Simple path addition
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config.config import LINEAR_LEARNING_RATE, LINEAR_MAX_EPOCHS
from src.utils.data_loader import load_mnist_data


class LinearClassifierNumPy:
    """
    Linear classifier implemented using NumPy.
    """
    def __init__(self, n_features, n_classes, learning_rate=LINEAR_LEARNING_RATE, max_epochs=LINEAR_MAX_EPOCHS):

        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Initialize weights: y = Wx
        self.W = np.random.randn(n_classes, n_features) * 0.01
        
        self.is_fitted = False
    
    def _forward(self, X):
        """
        Forward pass: compute y = Wx.
        """
        return np.dot(X, self.W.T)
    
    def _l2_loss(self, y_pred, y_true):
        """
        Computing L2 loss (Mean Squared Error).
        """
        # Convert labels to one-hot encoding
        y_true_onehot = np.eye(self.n_classes)[y_true]
        
        # L2 loss: MSE between predictions and one-hot labels
        loss = np.mean((y_pred - y_true_onehot) ** 2)
        return loss
    
    def _backward(self, X, y_true, y_pred):
        """
        Backward pass: Computing gradients for L2 loss.
        """
        batch_size = X.shape[0]
        
        # Convert labels to one-hot encoding
        y_true_onehot = np.eye(self.n_classes)[y_true]
        
        # Gradient of L2 loss w.r.t. predictions
        dloss = 2 * (y_pred - y_true_onehot) / batch_size
        
        # Gradient w.r.t. weights: dW = dloss^T * X
        dW = np.dot(dloss.T, X)
        
        return dW
    
    def fit(self, X_train, y_train, verbose=True):
        """
        Train the linear classifier using gradient descent.
        """
        train_losses = []
        
        for epoch in range(self.max_epochs):
            # Forward pass: y = Wx
            y_pred = self._forward(X_train)
            
            # Compute L2 loss
            train_loss = self._l2_loss(y_pred, y_train)
            train_losses.append(train_loss)
            
            # Backward pass: compute gradients
            dW = self._backward(X_train, y_train, y_pred)
            
            # Update weights using gradient descent
            self.W -= self.learning_rate * dW
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {train_loss:.4f}")
        
        self.is_fitted = True
        return train_losses
    
    def predict(self, X_test):
        """
        Predict class labels for test samples.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        y_pred = self._forward(X_test)
        return np.argmax(y_pred, axis=1)
    
    def score(self, X_test, y_test):
        """
        Compute accuracy score on test data.
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)


class LinearClassifierPyTorch(nn.Module):
    """
    Linear classifier implemented using PyTorch.
    """
    
    def __init__(self, n_features, n_classes):

        super(LinearClassifierPyTorch, self).__init__()
        # Use nn.Linear but set bias=False to match y = Wx (no bias)
        self.linear = nn.Linear(n_features, n_classes, bias=False)
        
        # Initialize weights to match NumPy implementation
        with torch.no_grad():
            self.linear.weight.data = torch.randn(n_classes, n_features) * 0.01
    
    def forward(self, x):
        """
        Forward pass: y = Wx.
        """
        return self.linear(x)


def train_pytorch_linear_classifier(X_train, y_train, batch_size=64, learning_rate=LINEAR_LEARNING_RATE, 
                                  max_epochs=LINEAR_MAX_EPOCHS, verbose=True):
    """
    Train PyTorch linear classifier using L2 loss and gradient descent with batching.
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = LinearClassifierPyTorch(n_features, n_classes)
    
    # Loss function: L2 loss (MSE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop with batching
    train_losses = []
    
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass: y = Wx
            outputs = model(batch_X)
            
            # Convert labels to one-hot for L2 loss
            y_true_onehot = torch.zeros(len(batch_y), n_classes)
            y_true_onehot.scatter_(1, batch_y.unsqueeze(1), 1)
            
            # Compute L2 loss
            loss = criterion(outputs, y_true_onehot)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.4f}")
    
    return model, train_losses


def evaluate_linear_classifier_performance(X_train, X_test, y_train, y_test, implementation='both'):
    """
    Evaluate linear classifier performance using NumPy and/or PyTorch implementations.
    """
    print("Starting Linear Classifier evaluation...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    results = {}
    
    if implementation in ['numpy', 'both']:
        print("\nEvaluating NumPy Linear Classifier...")
        
        # Create and train NumPy classifier
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        numpy_classifier = LinearClassifierNumPy(n_features, n_classes)
        
        start_time = time.time()
        train_losses = numpy_classifier.fit(X_train, y_train, verbose=True)
        fit_time = time.time() - start_time
        
        start_time = time.time()
        predictions = numpy_classifier.predict(X_test)
        predict_time = time.time() - start_time
        
        accuracy = numpy_classifier.score(X_test, y_test)
        
        results['numpy'] = {
            'accuracy': accuracy,
            'fit_time': fit_time,
            'predict_time': predict_time,
            'train_losses': train_losses,
            'predictions': predictions,
            'weights': numpy_classifier.W
        }
        
        print(f"NumPy Linear - Accuracy: {accuracy:.4f}, "
              f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")
    
    if implementation in ['pytorch', 'both']:
        print("\nEvaluating PyTorch Linear Classifier...")
        
        start_time = time.time()
        pytorch_model, train_losses = train_pytorch_linear_classifier(
            X_train, y_train, verbose=True
        )
        fit_time = time.time() - start_time
        
        # Make predictions
        pytorch_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            start_time = time.time()
            outputs = pytorch_model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
            predict_time = time.time() - start_time
        
        accuracy = np.mean(predictions == y_test)
        
        results['pytorch'] = {
            'accuracy': accuracy,
            'fit_time': fit_time,
            'predict_time': predict_time,
            'train_losses': train_losses,
            'predictions': predictions,
            'model': pytorch_model,
            'weights': pytorch_model.linear.weight.data.numpy()
        }
        
        print(f"PyTorch Linear - Accuracy: {accuracy:.4f}, "
              f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")
    
    return results


if __name__ == "__main__":
    # Load MNIST data
    print("Loading MNIST data for Linear Classifier...")
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Evaluate both implementations
    results = evaluate_linear_classifier_performance(
        X_train, X_test, y_train, y_test, implementation='both'
    )
    
    print("\n" + "="*60)
    print("LINEAR CLASSIFIER RESULTS SUMMARY")
    print("="*60)
    
    print("Results:")
    for impl, result in results.items():
        print(f"{impl.capitalize()}: Accuracy={result['accuracy']:.4f}, "
              f"Fit={result['fit_time']:.2f}s, Predict={result['predict_time']:.2f}s")