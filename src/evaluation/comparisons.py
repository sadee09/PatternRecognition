import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.utils.data_loader import load_mnist_data, load_mnist_for_cnn


class ModelComparisons:
    """
    Comprehensive comparison analysis for different model configurations.
    """
    
    def __init__(self):
        """Initialize the comparison analyzer."""
        self.comparison_results = {}
        self.class_names = [str(i) for i in range(10)]
    
    def compare_loss_functions(self, evaluator_results, save_dir="results/comparisons"):
        """
        Compare L2 Loss vs Cross-Entropy Loss using evaluator results.
        """
        print("Comparing Loss Functions: L2 vs Cross-Entropy")
        print("-" * 50)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract only NumPy linear classifier results from evaluator
        linear_original_numpy = None
        linear_improved_numpy = None
        
        for model_name, result in evaluator_results.items():
            if "Linear Original (Numpy)" in model_name:
                linear_original_numpy = result
            elif "Linear Improved (Numpy)" in model_name:
                linear_improved_numpy = result
        
        # Compile comparison results (NumPy only)
        comparison_data = []
        
        # Compare NumPy implementations only
        if linear_original_numpy and linear_improved_numpy:
            comparison_data.append({
                'Implementation': 'NumPy',
                'Loss Function': 'L2 Loss',
                'Accuracy': linear_original_numpy['accuracy'],
                'Fit Time': linear_original_numpy['fit_time'],
                'Predict Time': linear_original_numpy['predict_time'],
                'Final Loss': linear_original_numpy['train_losses'][-1] if linear_original_numpy['train_losses'] else None,
                'Epochs': len(linear_original_numpy['train_losses']) if linear_original_numpy['train_losses'] else None
            })
            
            comparison_data.append({
                'Implementation': 'NumPy',
                'Loss Function': 'Cross-Entropy',
                'Accuracy': linear_improved_numpy['accuracy'],
                'Fit Time': linear_improved_numpy['fit_time'],
                'Predict Time': linear_improved_numpy['predict_time'],
                'Final Loss': linear_improved_numpy['train_losses'][-1] if linear_improved_numpy['train_losses'] else None,
                'Epochs': len(linear_improved_numpy['train_losses']) if linear_improved_numpy['train_losses'] else None
            })
        
        if not comparison_data:
            print("No linear classifier results found for comparison")
            return None
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Store results
        self.comparison_results['loss_functions'] = {
            'dataframe': df
        }
        
        # Create visualization
        self._plot_loss_function_comparison(df, save_dir)
        
        print("Loss function comparison completed")
        return df
    
    def compare_mlp_architectures(self, X_train, X_test, y_train, y_test, save_dir="results/comparisons"):
        """
        Compare different MLP architectures.
        """
        print("Comparing MLP Architectures")
        print("-" * 50)
        
        os.makedirs(save_dir, exist_ok=True)
        
        from src.classifiers.mlp_classifier import evaluate_mlp_performance
        
        # Define different architectures to compare
        architectures = [
            [128],           # Single hidden layer
            [256, 128],      # Two hidden layers
            [512, 256, 128], # Three hidden layers
            [256, 256, 128, 64]  # Four hidden layers
        ]
        
        print("Testing different MLP architectures...")
        results = {}
        
        for i, hidden_sizes in enumerate(architectures):
            print(f"\nEvaluating Architecture {i+1}: {hidden_sizes}")
            
            arch_results = evaluate_mlp_performance(
                X_train, X_test, y_train, y_test,
                hidden_sizes=hidden_sizes
            )
            
            results[f'Architecture_{i+1}'] = {
                'hidden_sizes': hidden_sizes,
                'accuracy': arch_results['accuracy'],
                'fit_time': arch_results['fit_time'],
                'predict_time': arch_results['predict_time']
            }
        
        # Compile comparison data
        comparison_data = []
        for arch_name, result in results.items():
            comparison_data.append({
                'Architecture': arch_name,
                'Hidden Layers': str(result['hidden_sizes']),
                'Accuracy': result['accuracy'],
                'Fit Time': result['fit_time'],
                'Predict Time': result['predict_time']
            })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Store results
        self.comparison_results['mlp_architectures'] = {
            'dataframe': df,
            'detailed_results': results
        }
        
        # Create visualization
        self._plot_architecture_comparison(df, save_dir)
        
        print("MLP architecture comparison completed")
        return df
    
    def compare_learning_rates(self, X_train, X_test, y_train, y_test, save_dir="results/comparisons"):
        """
        Compare different learning rates for Linear Classifiers.
        """
        print("Comparing Learning Rates")
        print("-" * 50)
        
        os.makedirs(save_dir, exist_ok=True)
        
        from src.classifiers.linear_classifer_improved import evaluate_linear_classifier_performance
        
        learning_rates = [0.001, 0.01, 0.1, 0.5]
        comparison_data = []
        
        for lr in learning_rates:
            print(f"Testing learning rate: {lr}")
            
            try:
                # Test only NumPy implementation
                results = evaluate_linear_classifier_performance(
                    X_train, X_test, y_train, y_test, 
                    implementation='numpy', learning_rate=lr
                )
                
                # Add NumPy results
                if 'numpy' in results:
                    numpy_result = results['numpy']
                    comparison_data.append({
                        'Implementation': 'NumPy',
                        'Learning Rate': lr,
                        'Accuracy': numpy_result['accuracy'],
                        'Fit Time': numpy_result['fit_time'],
                        'Final Loss': numpy_result['train_losses'][-1] if numpy_result['train_losses'] else None,
                        'Convergence': 'Yes' if len(numpy_result['train_losses']) > 10 and numpy_result['train_losses'][-1] < numpy_result['train_losses'][0] else 'No'
                    })
                    
            except Exception as e:
                print(f"  NumPy failed for LR {lr}: {e}")
                comparison_data.append({
                    'Implementation': 'NumPy',
                    'Learning Rate': lr,
                    'Accuracy': 0.0,
                    'Fit Time': 0.0,
                    'Final Loss': None,
                    'Convergence': 'Failed'
                })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Store results
        self.comparison_results['learning_rates'] = {
            'dataframe': df
        }
        
        # Create visualization
        self._plot_learning_rate_comparison(df, save_dir)
        
        print("Learning rate comparison completed")
        return df
    

    
    def _plot_loss_function_comparison(self, df, save_dir):
        """Plot loss function comparison."""
        numpy_data = df[df['Implementation'] == 'NumPy']
        
        l2_acc = numpy_data[numpy_data['Loss Function'] == 'L2 Loss']['Accuracy'].iloc[0]
        ce_acc = numpy_data[numpy_data['Loss Function'] == 'Cross-Entropy']['Accuracy'].iloc[0]
        
        plt.figure(figsize=(10, 6))
        plt.bar(['L2 Loss', 'Cross-Entropy'], [l2_acc, ce_acc])
        plt.title('Loss Function Comparison')
        plt.ylabel('Accuracy')
        plt.savefig(f"{save_dir}/loss_function_comparison.png")
        plt.close()
    
    def _plot_architecture_comparison(self, df, save_dir):
        """Plot MLP architecture comparison."""
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(df['Architecture'], df['Accuracy'], color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, df['Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('MLP Architecture Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Architecture', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{save_dir}/mlp_architecture_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_rate_comparison(self, df, save_dir):
        """Plot learning rate comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs Learning Rate (NumPy only)
        numpy_data = df[df['Implementation'] == 'NumPy']
        axes[0].plot(numpy_data['Learning Rate'], numpy_data['Accuracy'], 
                    marker='o', label='NumPy', linewidth=2, color='blue')
        
        axes[0].set_title('Accuracy vs Learning Rate (NumPy)', fontweight='bold')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training Time vs Learning Rate (NumPy only)
        axes[1].plot(numpy_data['Learning Rate'], numpy_data['Fit Time'], 
                    marker='s', label='NumPy', linewidth=2, color='green')
        
        axes[1].set_title('Training Time vs Learning Rate (NumPy)', fontweight='bold')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Training Time (seconds)')
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/learning_rate_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

def run_all_comparisons(evaluator_results=None):
    """
    Run all comparison analyses.
    """
    print("Running Comprehensive Comparison Analysis")
    print("=" * 60)
    
    
    print("Loading MNIST data...")
    X_train, X_test, y_train, y_test = load_mnist_data()
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = load_mnist_for_cnn()
    
    # Initialize comparison analyzer
    analyzer = ModelComparisons()
    
    # Run all comparisons
    print("\n" + "="*60)
    print("RUNNING COMPARISON ANALYSES")
    print("="*60)
    
    # 1. Loss Function Comparison (using evaluator results)
    if evaluator_results:
        analyzer.compare_loss_functions(evaluator_results)
    else:
        print("No evaluator results provided for loss function comparison")
    
    # 2. MLP Architecture Comparison
    analyzer.compare_mlp_architectures(X_train, X_test, y_train, y_test)
    
    # 3. Learning Rate Comparison
    analyzer.compare_learning_rates(X_train, X_test, y_train, y_test)
    
    return analyzer

# if __name__ == "__main__":
#     analyzer = run_all_comparisons()
