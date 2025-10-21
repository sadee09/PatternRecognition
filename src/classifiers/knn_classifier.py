import numpy as np
from collections import Counter
import time
import sys
import os

# Simple path addition
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config.config import KNN_K_VALUES
from src.utils.data_loader import load_mnist_data


class KNNClassifier:
    """
    K-Nearest Neighbors classifier using Euclidean distance and majority vote.
    """
    
    def __init__(self, k=3):

        self.k = k
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Store training data
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.is_fitted = True
    
    def predict(self, X_test):
        """
        Predict labels for test samples using majority vote among k nearest neighbors.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        predictions = []
        
        # For each test sample, find k nearest neighbors
        for test_sample in X_test:
            # Compute Euclidean distances to all training samples
            distances = np.sqrt(np.sum((self.X_train - test_sample) ** 2, axis=1))
            
            # Find k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # Majority vote
            label_counts = Counter(k_nearest_labels)
            prediction = label_counts.most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """
        Compute accuracy score on test data.
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)


def evaluate_knn_performance(X_train, X_test, y_train, y_test, k_values=KNN_K_VALUES):
    """
    Evaluate KNN performance for different values of k.
    """
    print("Starting KNN evaluation...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"K values to test: {k_values}")
    
    results = {}
    
    for k in k_values:
        print(f"\nEvaluating KNN with k={k}...")
        
        # Create and fit classifier
        knn = KNNClassifier(k=k)
        start_time = time.time()
        knn.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        predictions = knn.predict(X_test)
        predict_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = knn.score(X_test, y_test)
        
        # Store results
        results[k] = {
            'accuracy': accuracy,
            'fit_time': fit_time,
            'predict_time': predict_time,
            'predictions': predictions
        }
        
        print(f"KNN (k={k}) - Accuracy: {accuracy:.4f}, "
              f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")
    
    return results


if __name__ == "__main__":
    # Load MNIST data - use smaller subset for faster computation
    X_train, X_test, y_train, y_test = load_mnist_data(test_size=0.01)  # Only 1% for testing
    
    # Test with assignment required k values: 1, 3, 5
    k_values = [1, 3, 5]
    
    results = evaluate_knn_performance(X_train, X_test, y_train, y_test, k_values)
    
    print("\n" + "="*60)
    print("KNN RESULTS SUMMARY")
    print("="*60)
    
    # Find best k value
    best_k = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_k]['accuracy']
    
    print(f"Best k value: {best_k}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("\nAll Results:")
    
    for k, result in results.items():
        print(f"k={k}: Accuracy={result['accuracy']:.4f}, "
              f"Fit={result['fit_time']:.2f}s, Predict={result['predict_time']:.2f}s")