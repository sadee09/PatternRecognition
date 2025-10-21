import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path

# Simple path addition to use the config file
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config.config import TEST_SIZE, RANDOM_STATE, DATA_DIR


def load_mnist_data(data_dir=DATA_DIR, test_size=TEST_SIZE, random_state=RANDOM_STATE, binary=False):
    """
    Load MNIST data from image files, preprocess the images and create train/test splits.
    """
    images = []
    labels = []
    
    # Use pathlib for better path handling
    data_path = Path(data_dir)
    
    # Load all images and labels
    for digit in range(10):
        digit_path = data_path / str(digit)
        if not digit_path.exists():
            continue
            
        # Use glob for pattern matching
        for img_path in digit_path.glob('*.png'):
            # Load and preprocess image
            image = Image.open(img_path).convert('L')
            # Normalize pixel values to [0,1]
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            if binary:
                # Binarize for Naive Bayes
                image_array = (image_array > 0.5).astype(np.float32)
            
            #Flatten into vectors (784 features)
            images.append(image_array.flatten())
            labels.append(digit)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def load_mnist_for_cnn(data_dir=DATA_DIR, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Load MNIST data for CNN models, keeping 2D image shape and preprocess the images.
    """
    images = []
    labels = []
    
    data_path = Path(data_dir)

    for digit in range(10):
        digit_path = data_path / str(digit)
        if not digit_path.exists():
            continue
            
        for img_path in digit_path.glob('*.png'):
            image = Image.open(img_path).convert('L')
            image_array = np.array(image, dtype=np.float32) / 255.0
        
            #Keep 2D shape for CNNs
            images.append(image_array)
            labels.append(digit)
    
    X = np.array(images)
    y = np.array(labels)
    
    # Add channel dimension for CNN (batch_size, height, width)
    X = X.reshape(X.shape[0], 1, 28, 28) #[28x28 -> 784 features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_data_info(X_train, X_test, y_train, y_test):
    """
    Print information about the loaded dataset.
    """
    print("Dataset Information:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")

