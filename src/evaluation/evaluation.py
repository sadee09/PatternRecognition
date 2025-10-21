import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
import time
import sys
import os

# Simple path addition
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


class ModelEvaluator:
    """
    Comprehensive model evaluation class for comparing different classifiers.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.results = {}
        self.class_names = [str(i) for i in range(10)]  # MNIST digits 0-9
    
    def add_model_result(self, model_name, predictions, true_labels, 
                        fit_time=None, predict_time=None, additional_info=None,
                        train_losses=None, train_accuracies=None):
        """
        Add model results to the evaluator.
        """
        # Calculate basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Store results
        self.results[model_name] = {
            'predictions': predictions,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'fit_time': fit_time,
            'predict_time': predict_time,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'additional_info': additional_info or {}
        }
    
    
    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model based on a specific metric.
        """
        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1][metric]
    
    def get_results_for_visualization(self):
        """
        Get results in a format suitable for visualization.
        """
        return self.results


def run_comprehensive_evaluation():

    print("Running Comprehensive MNIST Classification Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load data for different classifiers
    from src.utils.data_loader import load_mnist_data, load_mnist_for_cnn
    
    print("\n1. Loading data...")
    X_train, X_test, y_train, y_test = load_mnist_data()
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = load_mnist_for_cnn()
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = load_mnist_data(binary=True)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # 1. K-Nearest Neighbor
    print("\n2. Evaluating K-Nearest Neighbor...")
    from src.classifiers.knn_classifier import evaluate_knn_performance
    knn_results = evaluate_knn_performance(X_train, X_test, y_train, y_test, k_values=[1, 3, 5])
    
    for k, result in knn_results.items():
        evaluator.add_model_result(
            f"KNN (k={k})", 
            result['predictions'], 
            y_test,
            fit_time=result['fit_time'],
            predict_time=result['predict_time'],
            train_losses=None  # KNN doesn't have training losses
        )
    
    # 2. Naive Bayes
    print("\n3. Evaluating Naive Bayes...")
    from src.classifiers.naive_bayes_classifier import evaluate_naive_bayes_performance
    nb_results = evaluate_naive_bayes_performance(X_train_bin, X_test_bin, y_train_bin, y_test_bin)
    
    evaluator.add_model_result(
        "Naive Bayes",
        nb_results['predictions'],
        y_test_bin,  # Use the same test labels that were used for training
        fit_time=nb_results['fit_time'],
        predict_time=nb_results['predict_time'],
        train_losses=None,  # Naive Bayes doesn't have training losses
        additional_info={'probabilities': nb_results.get('probabilities')}
    )
    
    # 3. Linear Classifier (Original)
    print("\n4. Evaluating Linear Classifier (Original)...")
    from src.classifiers.linear_classifier import evaluate_linear_classifier_performance as evaluate_linear_original
    linear_original_results = evaluate_linear_original(
        X_train, X_test, y_train, y_test, implementation='numpy'
    )
    
    for impl, result in linear_original_results.items():
        evaluator.add_model_result(
            f"Linear Original ({impl.capitalize()})",
            result['predictions'],
            y_test,
            fit_time=result['fit_time'],
            predict_time=result['predict_time'],
            train_losses=result['train_losses'],
            additional_info={'weights': result.get('weights')}
        )
    
    # 4. Linear Classifier (Improved)
    print("\n5. Evaluating Linear Classifier (Improved)...")
    from src.classifiers.linear_classifer_improved import evaluate_linear_classifier_performance as evaluate_linear_improved
    linear_improved_results = evaluate_linear_improved(
        X_train, X_test, y_train, y_test, implementation='numpy'
    )
    
    for impl, result in linear_improved_results.items():
        evaluator.add_model_result(
            f"Linear Improved ({impl.capitalize()})",
            result['predictions'],
            y_test,
            fit_time=result['fit_time'],
            predict_time=result['predict_time'],
            train_losses=result['train_losses'],
            additional_info={'weights': result.get('weights')}
        )
    
    # 5. MLP
    print("\n6. Evaluating MLP...")
    from src.classifiers.mlp_classifier import evaluate_mlp_performance
    mlp_results = evaluate_mlp_performance(X_train, X_test, y_train, y_test)
    
    evaluator.add_model_result(
        "MLP",
        mlp_results['predictions'],
        y_test,
        fit_time=mlp_results['fit_time'],
        predict_time=mlp_results['predict_time'],
        train_losses=mlp_results['train_losses']
    )
    
    # 6. CNN
    print("\n7. Evaluating CNN...")
    from src.classifiers.cnn_classifier import evaluate_cnn_performance
    cnn_results = evaluate_cnn_performance(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn)
    
    evaluator.add_model_result(
        "CNN",
        cnn_results['predictions'],
        y_test_cnn,
        fit_time=cnn_results['fit_time'],
        predict_time=cnn_results['predict_time'],
        train_losses=cnn_results['train_losses']
    )
    
    # Get best model
    best_model, best_accuracy = evaluator.get_best_model('accuracy')
    print(f"\nBest performing model: {best_model} with accuracy: {best_accuracy:.4f}")
    
    return evaluator


# if __name__ == "__main__":
#     # Run comprehensive evaluation
#     evaluator = run_comprehensive_evaluation()