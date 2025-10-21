# MNIST Classification Project Configuration
import os

# Data Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'MNIST')
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2   
RANDOM_STATE = 42

# Model Configuration
KNN_K_VALUES = [1, 3, 5]
LINEAR_LEARNING_RATE = 0.01
LINEAR_MAX_EPOCHS = 30
MLP_HIDDEN_SIZES = [256, 128]
MLP_LEARNING_RATE = 0.001
MLP_MAX_EPOCHS = 30
MLP_BATCH_SIZE = 32
MLP_DROPOUT_RATE = 0.2
CNN_LEARNING_RATE = 0.001
CNN_MAX_EPOCHS = 30
CNN_BATCH_SIZE = 32
CNN_DROPOUT_RATE = 0.5

# Evaluation Configuration
QUICK_MODE = False
SAVE_PLOTS = True
SAVE_REPORTS = True
RESULTS_DIR = "results"

# Visualization Configuration
FIGURE_SIZE = (12, 8)
DPI = 300
PLOT_STYLE = "seaborn-v0_8"
