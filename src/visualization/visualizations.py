import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def create_training_curves(evaluator_results, save_dir="results/plots"):
    """
    Create training loss curves for models that have training data.
    """
    print("Creating training curves...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Find models with training losses
    models_with_losses = {name: result for name, result in evaluator_results.items() 
                         if result.get('train_losses') is not None}
    
    if not models_with_losses:
        print("  No models with training losses found")
        return
    
    plt.figure(figsize=(12, 6))
    
    for model_name, result in models_with_losses.items():
        train_losses = result['train_losses']
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f'{model_name}', linewidth=2, marker='o', markersize=4)
    
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Training curves saved")

def create_confusion_matrices(evaluator_results, save_dir="results/plots"):
    """
    Create confusion matrices for all models.
    """
    print("Creating confusion matrices...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = [str(i) for i in range(10)]
    n_models = len(evaluator_results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (model_name, result) in enumerate(evaluator_results.items()):
        cm = confusion_matrix(result['true_labels'], result['predictions'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[i])
        axes[i].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.4f}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    # Hide empty subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Confusion matrices saved")

def create_model_comparison(evaluator_results, save_dir="results/plots"):
    """
    Create model performance comparison chart.
    """
    print("Creating model comparison...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(evaluator_results.keys())
    accuracies = [evaluator_results[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Performance Comparison - Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Model comparison saved")

def create_failure_analysis(evaluator_results, save_dir="results/plots"):
    """
    Create failure mode analysis showing common misclassifications.
    """
    print("Creating failure analysis...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = [str(i) for i in range(10)]
    
    # Find the best performing model for detailed analysis
    best_model_name = max(evaluator_results.keys(), 
                          key=lambda x: evaluator_results[x]['accuracy'])
    best_result = evaluator_results[best_model_name]
    
    cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
    
    # Create failure analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               ax=axes[0, 0])
    axes[0, 0].set_title(f'{best_model_name} - Confusion Matrix', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # 2. Most Common Misclassifications
    misclassifications = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                misclassifications.append((i, j, cm[i, j]))
    
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    top_misclass = misclassifications[:10]
    
    if top_misclass:
        true_labels = [f"{x[0]}" for x in top_misclass]
        pred_labels = [f"{x[1]}" for x in top_misclass]
        counts = [x[2] for x in top_misclass]
        
        x_pos = range(len(top_misclass))
        axes[0, 1].bar(x_pos, counts, color='red', alpha=0.7)
        axes[0, 1].set_title('Most Common Misclassifications', fontweight='bold')
        axes[0, 1].set_xlabel('True → Predicted')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([f"{t}→{p}" for t, p in zip(true_labels, pred_labels)], 
                                  rotation=45, ha='right')
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            axes[0, 1].text(i, count + 1, str(count), ha='center', va='bottom')
    
    # 3. Per-Class Accuracy
    per_class_acc = np.diag(cm) / cm.sum(axis=1)
    bars = axes[1, 0].bar(class_names, per_class_acc, color='green', alpha=0.7)
    axes[1, 0].set_title('Per-Class Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Digit Class')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, per_class_acc):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Model Performance Summary
    models = list(evaluator_results.keys())
    accuracies = [evaluator_results[model]['accuracy'] for model in models]
    
    bars = axes[1, 1].bar(models, accuracies, color='skyblue', alpha=0.7)
    axes[1, 1].set_title('All Models Performance', fontweight='bold')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/failure_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Failure analysis saved")

def create_weight_visualization(evaluator_results, save_dir="results/plots"):
    """
    Visualize weight matrices of linear classifiers as digit-like images.
    """
    print("Creating weight visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Find linear classifier models
    linear_models = {name: result for name, result in evaluator_results.items() 
                    if 'Linear' in name and 'weights' in result.get('additional_info', {})}
    
    if not linear_models:
        print("  No linear classifier weights found")
        return
    
    for model_name, result in linear_models.items():
        weights = result['additional_info']['weights']
        
        # Create weight visualization
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(10):
            # Reshape weight vector to 28x28 image
            weight_image = weights[i].reshape(28, 28)
            
            # Normalize for better visualization
            weight_image = (weight_image - weight_image.min()) / (weight_image.max() - weight_image.min())
            
            im = axes[i].imshow(weight_image, cmap='RdBu_r', vmin=0, vmax=1)
            axes[i].set_title(f'Digit {i}', fontweight='bold')
            axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes, shrink=0.8, aspect=20)
        
        plt.suptitle(f'{model_name} - Weight Matrix Visualization', fontsize=16, fontweight='bold')
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        
        # Save with model name
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f"{save_dir}/weights_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Weight visualization saved for {model_name}")

def create_probability_maps(evaluator_results, save_dir="results/plots"):
    """
    Visualize Naive Bayes probability maps as digit-like images.
    """
    print("Creating probability maps...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Find Naive Bayes model
    nb_models = {name: result for name, result in evaluator_results.items() 
                if 'Naive' in name and 'probabilities' in result.get('additional_info', {})}
    
    if not nb_models:
        print("  No Naive Bayes probability maps found")
        return
    
    for model_name, result in nb_models.items():
        probabilities = result['additional_info']['probabilities']
        
        # Create probability map visualization
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(10):
            # Reshape probability vector to 28x28 image
            prob_image = probabilities[i].reshape(28, 28)
            
            im = axes[i].imshow(prob_image, cmap='hot', vmin=0, vmax=1)
            axes[i].set_title(f'Digit {i}', fontweight='bold')
            axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes, shrink=0.8, aspect=20)
        
        plt.suptitle(f'{model_name} - Pixel Probability Maps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with model name
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f"{save_dir}/probabilities_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Probability maps saved for {model_name}")

def create_all_visualizations(evaluator_results, save_dir="results/plots"):
    """
    Create all visualizations from evaluation results.
    """
    print("Creating All Visualizations")
    print("=" * 40)
    
    # Create all visualizations
    create_training_curves(evaluator_results, save_dir)
    create_confusion_matrices(evaluator_results, save_dir)
    create_model_comparison(evaluator_results, save_dir)
    create_failure_analysis(evaluator_results, save_dir)
    create_weight_visualization(evaluator_results, save_dir)
    create_probability_maps(evaluator_results, save_dir)