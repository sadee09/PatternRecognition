import numpy as np
import time
import sys
import os

# Simple path addition
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config.config import TEST_SIZE, RANDOM_STATE, DATA_DIR
from src.utils.data_loader import load_mnist_data


class NaiveBayesClassifier:
    def __init__(self):

        self.class_priors = None
        self.pixel_probs = None  # P(pixel=1|class)
        self.classes = None
        self.n_features = None
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Fit the Naive Bayes classifier by estimating probabilities.
        """
        self.classes = np.unique(y_train)
        self.n_features = X_train.shape[1]
        
        # Estimate class priors P(class)
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        self.class_priors = class_counts / total_samples
        
        # Estimate conditional probabilities P(pixel=1|class) for each class
        self.pixel_probs = np.zeros((len(self.classes), self.n_features))
        
        for i, class_label in enumerate(self.classes):
            # Get samples belonging to this class
            class_mask = (y_train == class_label)
            class_samples = X_train[class_mask]
            
            # Count pixels that are "on" (value = 1) for this class
            pixel_counts = np.sum(class_samples, axis=0)
            class_size = len(class_samples)
            
            # Estimate P(pixel=1|class) with Laplace smoothing
            # P(pixel=1|class) = (count + 1) / (class_size + 2)
            self.pixel_probs[i] = (pixel_counts + 1) / (class_size + 2)
        
        self.is_fitted = True
    
    def predict(self, X_test):
        """
        Predict class labels for test samples using Bayes' rule.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        predictions = np.zeros(X_test.shape[0], dtype=int)
        
        for i, test_sample in enumerate(X_test):
            # Compute posterior probability for each class
            log_posteriors = np.zeros(len(self.classes))
            
            for j, class_label in enumerate(self.classes):
                # P(class|x) ∝ P(x|class) * P(class)
                # Taking log: log P(class|x) = log P(x|class) + log P(class)
                
                # Compute log likelihood: log P(x|class)
                log_likelihood = 0.0
                pixel_probs = self.pixel_probs[j]
                
                for k in range(len(test_sample)):
                    if test_sample[k] == 1:
                        # Pixel is "on"
                        log_likelihood += np.log(pixel_probs[k])
                    else:
                        # Pixel is "off"
                        log_likelihood += np.log(1 - pixel_probs[k])
                
                # Add log prior
                log_prior = np.log(self.class_priors[j])
                log_posteriors[j] = log_likelihood + log_prior
            
            # Find class with highest posterior probability
            predictions[i] = self.classes[np.argmax(log_posteriors)]
        
        return predictions
    
    def score(self, X_test, y_test):
        """
        Compute accuracy score on test data.
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)


def evaluate_naive_bayes_performance(X_train, X_test, y_train, y_test):
    """
    Evaluate Naive Bayes performance.
    """
    print("Starting Naive Bayes evaluation...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Binary data range: [{X_train.min()}, {X_train.max()}]")
    
    # Create and train classifier
    nb = NaiveBayesClassifier()
    start_time = time.time()
    nb.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    predictions = nb.predict(X_test)
    predict_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = nb.score(X_test, y_test)
    
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
        'predictions': predictions,
        'class_accuracies': class_accuracies,
        'probabilities': nb.pixel_probs
    }
    
    print(f"Naive Bayes - Accuracy: {accuracy:.4f}, "
          f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")
    
    print("Class-wise accuracies:")
    for class_label, acc in class_accuracies.items():
        print(f"  Class {class_label}: {acc:.4f}")
    
    return results


if __name__ == "__main__":
    # Load MNIST data with binary preprocessing
    print("Loading MNIST data for Naive Bayes...")
    X_train, X_test, y_train, y_test = load_mnist_data(binary=True)
    
    # Evaluate performance
    results = evaluate_naive_bayes_performance(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("NAIVE BAYES RESULTS SUMMARY")
    print("="*60)
    
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Fit Time: {results['fit_time']:.2f}s")
    print(f"Predict Time: {results['predict_time']:.2f}s")