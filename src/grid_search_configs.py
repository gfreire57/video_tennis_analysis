"""
Grid Search Configuration - Predefined Parameter Grids

This module defines different parameter grids for systematic hyperparameter tuning:
- SMALL_GRID: Quick exploration (~20-30 runs)
- MEDIUM_GRID: Balanced exploration (~80-120 runs)
- LARGE_GRID: Comprehensive exploration (~300+ runs)
- ARCHITECTURE_FOCUSED: Focus on model architecture
- HYPERPARAMETER_FOCUSED: Focus on training hyperparameters
"""

# ==============================================================================
# PREDEFINED GRID SEARCH CONFIGURATIONS
# ==============================================================================

"""Quick exploration grid - approximately 20-30 training runs

Good for:
- Initial exploration
- Quick validation of grid search setup
- Limited computational resources
"""
SMALL_GRID = {
    # Model Architecture
    'lstm_layers': [
        [64, 128, 64],   # 3-layer:
        [128, 64],       # 2-layer:
        [64, 32],        # 2-layer:
    ],
    'dense_units': [
        32 #, 64
        ],
    'dropout_rates': [
        [0.3, 0.3, 0.2],  
        [0.4, 0.4, 0.3],  
    ],
    'use_batch_norm': [False, True],
    'use_bidirectional': [True],

    # Training Hyperparameters
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64],

    # Data Configuration
    'window_size': [30], # 45
    'overlap': [20], # 15, 10
}

"""Balanced exploration grid - approximately 80-120 training runs

Good for:
- Systematic parameter exploration
- Finding optimal architecture
- Moderate computational resources
"""
MEDIUM_GRID = {
    # Model Architecture
    'lstm_layers': [
        [64, 32],         # 2-layer: small
        [128, 64],        # 2-layer: medium (current V2)
        [256, 128],       # 2-layer: large
        [128, 96, 64],    # 3-layer: similar to V1
    ],
    'dense_units': [32, 64, 128],
    'dropout_rates': [
        [0.2, 0.2, 0.1],  # Lower dropout
        [0.3, 0.3, 0.2],  # Current configuration
        [0.4, 0.4, 0.3],  # Higher dropout
    ],
    'use_batch_norm': [False, True],  # Test with and without

    # Training Hyperparameters
    'learning_rate': [0.0001, 0.0005, 0.001],
    'batch_size': [16, 32, 64],

    # Data Configuration
    'window_size': [30, 45, 60],
    'overlap': [10, 15, 20],
}

"""Comprehensive exploration grid - approximately 300+ training runs

Good for:
- Exhaustive parameter search
- Final optimization
- Abundant computational resources (GPU cluster)

WARNING: This will take significant time to complete!
"""
LARGE_GRID = {
    # Model Architecture
    'lstm_layers': [
        [64, 32],          # 2-layer: small
        [128, 64],         # 2-layer: medium
        [256, 128],        # 2-layer: large
        [128, 96, 64],     # 3-layer: V1-like
        [256, 128, 64],    # 3-layer: large
        [512, 256, 128],   # 3-layer: very large
    ],
    'dense_units': [32, 64, 128, 256],
    'dropout_rates': [
        [0.1, 0.1, 0.05],  # Very low dropout
        [0.2, 0.2, 0.1],   # Low dropout
        [0.3, 0.3, 0.2],   # Current configuration
        [0.4, 0.4, 0.3],   # High dropout
        [0.5, 0.5, 0.4],   # Very high dropout
    ],
    'use_batch_norm': [False, True],

    # Training Hyperparameters
    'learning_rate': [0.00005, 0.0001, 0.0005, 0.001, 0.005],
    'batch_size': [16, 32, 64, 128],

    # Data Configuration
    'window_size': [30, 45, 60, 75],
    'overlap': [10, 15, 20, 25],
}

"""Focus on architecture exploration with fixed training hyperparameters

Good for:
- Finding optimal model architecture
- Comparing different layer configurations
- When you've already tuned hyperparameters
"""
ARCHITECTURE_FOCUSED = {
    # Model Architecture - VARIED
    'lstm_layers': [
        # 2-layer variations
        [64, 32],
        [128, 64],
        [256, 128],
        [512, 256],
        # 3-layer variations
        [128, 96, 64],
        [256, 128, 64],
        [128, 64, 32],
        # 4-layer variation (experimental)
        [256, 128, 64, 32],
    ],
    'dense_units': [32, 64, 128],
    'dropout_rates': [
        [0.3, 0.3, 0.2],  # Keep fixed at current
    ],
    'use_batch_norm': [False, True],

    # Training Hyperparameters - FIXED
    'learning_rate': [0.001],  # Keep fixed
    'batch_size': [32],        # Keep fixed

    # Data Configuration - FIXED
    'window_size': [45],
    'overlap': [15],
}

"""Focus on training hyperparameters with fixed architecture

Good for:
- Fine-tuning training parameters
- Optimizing learning rate and batch size
- When you've already found good architecture
"""
HYPERPARAMETER_FOCUSED = {
    # Model Architecture - FIXED
    'lstm_layers': [
        [128, 64],  # Keep fixed at current V2
    ],
    'dense_units': [64],  # Keep fixed
    'dropout_rates': [
        [0.3, 0.3, 0.2],  # Keep fixed
    ],
    'use_batch_norm': [False],

    # Training Hyperparameters - VARIED
    'learning_rate': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
    'batch_size': [8, 16, 32, 64, 128],

    # Data Configuration - VARIED
    'window_size': [30, 37, 45, 52, 60],
    'overlap': [10, 12, 15, 18, 20],
}

"""Focus on data parameters (window size, overlap) with fixed model

Good for:
- Finding optimal sequence length
- Optimizing temporal window configuration
- Testing FPS scaling impact
"""
DATA_FOCUSED = {
    # Model Architecture - FIXED
    'lstm_layers': [[128, 64]],
    'dense_units': [64],
    'dropout_rates': [[0.3, 0.3, 0.2]],
    'use_batch_norm': [False],

    # Training Hyperparameters - FIXED
    'learning_rate': [0.001],
    'batch_size': [32],

    # Data Configuration - VARIED
    'window_size': [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80],
    'overlap': [5, 10, 15, 20, 25, 30],
    'enable_fps_scaling': [False, True],  # Test both modes
}

"""Minimal grid for testing the grid search script itself

Good for:
- Verifying grid search implementation
- Testing MLflow integration
- Quick smoke test

Only 4 combinations total!
"""
MINIMAL_TEST = {
    'lstm_layers': [
        [128, 64],
        [64, 32],
    ],
    'dense_units': [64],
    'dropout_rates': [[0.3, 0.3, 0.2]],
    'use_batch_norm': [False],
    'use_bidirectional': [False],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32],
    'window_size': [45],
    'overlap': [15],
}

# ==============================================================================
# RANDOM SEARCH PARAMETER RANGES
# ==============================================================================

"""Parameter ranges for random search

Each parameter can be:
- List of discrete values: randomly sampled
- Tuple (min, max, 'int'/'float'/'log'): randomly sampled from range
"""
RANDOM_SEARCH_RANGES = {
    # Model Architecture
    'lstm_layers': [
        [64, 32],
        [128, 64],
        [256, 128],
        [128, 96, 64],
        [256, 128, 64],
    ],
    'dense_units': (32, 256, 'int'),  # Random int between 32-256
    'dropout_rates': [
        [0.1, 0.1, 0.05],
        [0.2, 0.2, 0.1],
        [0.3, 0.3, 0.2],
        [0.4, 0.4, 0.3],
        [0.5, 0.5, 0.4],
    ],
    'use_batch_norm': [False, True],
    'use_bidirectional': [False],

    # Training Hyperparameters
    'learning_rate': (0.00001, 0.01, 'log'),  # Log-uniform sampling
    'batch_size': [16, 32, 64, 128],

    # Data Configuration
    'window_size': (20, 80, 'int'),
    'overlap': (5, 30, 'int'),
    'enable_fps_scaling': [False, True],
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_grid(grid_name):
    """Get a predefined grid by name

    Args:
        grid_name: One of 'small', 'medium', 'large', 'architecture',
                   'hyperparameter', 'data', 'minimal'

    Returns:
        Dictionary with parameter grid
    """
    grids = {
        'small': SMALL_GRID,
        'medium': MEDIUM_GRID,
        'large': LARGE_GRID,
        'architecture': ARCHITECTURE_FOCUSED,
        'hyperparameter': HYPERPARAMETER_FOCUSED,
        'data': DATA_FOCUSED,
        'minimal': MINIMAL_TEST,
        'random': RANDOM_SEARCH_RANGES,
    }

    if grid_name.lower() not in grids:
        available = ', '.join(grids.keys())
        raise ValueError(f"Unknown grid '{grid_name}'. Available: {available}")

    return grids[grid_name.lower()]

def count_combinations(param_grid):
    """Count total number of combinations in a grid

    Args:
        param_grid: Dictionary with parameter lists

    Returns:
        Total number of combinations
    """
    import itertools

    total = 1
    for param_name, param_values in param_grid.items():
        total *= len(param_values)

    return total

def print_grid_summary(grid_name):
    """Print summary of a grid configuration

    Args:
        grid_name: Name of the grid
    """
    grid = get_grid(grid_name)
    num_combinations = count_combinations(grid)

    print(f"\n{'='*70}")
    print(f"GRID CONFIGURATION: {grid_name.upper()}")
    print(f"{'='*70}")
    print(f"Total combinations: {num_combinations}")
    print(f"\nParameters:")

    for param_name, param_values in grid.items():
        print(f"  {param_name}: {len(param_values)} values")
        if len(param_values) <= 5:
            print(f"    → {param_values}")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Print summaries of all grids
    import sys

    if len(sys.argv) > 1:
        grid_name = sys.argv[1]
        print_grid_summary(grid_name)
    else:
        print("Available grid configurations:\n")
        for grid_name in ['minimal', 'small', 'medium', 'large', 'architecture', 'hyperparameter', 'data']:
            grid = get_grid(grid_name)
            num_comb = count_combinations(grid)
            print(f"  {grid_name:15s} - {num_comb:4d} combinations")

        print("\nUsage: python grid_search_configs.py <grid_name>")
        print("Example: python grid_search_configs.py small")
