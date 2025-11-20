"""
Grid Search for Tennis Stroke Recognition Model

Systematic exploration of model architectures and hyperparameters using MLflow tracking.
Supports full grid search, random search, and predefined configurations.

Usage:
    poetry run python src/grid_search.py --grid small
    poetry run python src/grid_search.py --grid medium --max-runs 50
    poetry run python src/grid_search.py --random --n-samples 30
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import argparse
import itertools
import random
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import training functions from train_model
# We'll import the module and use its functions
import train_model
from grid_search_configs import get_grid, count_combinations, print_grid_summary

import mlflow
import mlflow.tensorflow
from tensorflow import keras


# ==============================================================================
# GRID GENERATION
# ==============================================================================

def generate_full_grid(param_grid):
    """Generate all combinations from parameter grid

    Args:
        param_grid: Dictionary with parameter names and value lists

    Returns:
        List of dictionaries, each representing one combination
    """
    # Get all parameter names and their values
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    # Generate all combinations
    combinations = list(itertools.product(*param_values))

    # Convert to list of dictionaries
    param_combinations = []
    for combo in combinations:
        param_dict = {name: value for name, value in zip(param_names, combo)}
        param_combinations.append(param_dict)

    return param_combinations


def generate_random_samples(param_ranges, n_samples, seed=42):
    """Generate random parameter combinations

    Args:
        param_ranges: Dictionary with parameter ranges
        n_samples: Number of random samples to generate
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries, each representing one random combination
    """
    random.seed(seed)
    np.random.seed(seed)

    param_combinations = []

    for _ in range(n_samples):
        param_dict = {}

        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list):
                # Discrete values: random choice
                param_dict[param_name] = random.choice(param_range)

            elif isinstance(param_range, tuple) and len(param_range) == 3:
                # Continuous range: (min, max, type)
                min_val, max_val, range_type = param_range

                if range_type == 'int':
                    param_dict[param_name] = random.randint(min_val, max_val)
                elif range_type == 'float':
                    param_dict[param_name] = random.uniform(min_val, max_val)
                elif range_type == 'log':
                    # Log-uniform sampling (good for learning rates)
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    param_dict[param_name] = 10 ** random.uniform(log_min, log_max)

        param_combinations.append(param_dict)

    return param_combinations


# ==============================================================================
# MODEL BUILDING WITH PARAMETERS
# ==============================================================================

def build_model_from_params(params, input_shape, num_classes):
    """Build LSTM model from parameter dictionary

    Args:
        params: Dictionary with model parameters
        input_shape: Tuple (window_size, num_features)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    lstm_layers = params['lstm_layers']
    dense_units = params['dense_units']
    dropout_rates = params['dropout_rates']
    use_batch_norm = params['use_batch_norm']
    use_bidirectional = params.get('use_bidirectional', False)
    
    learning_rate = params['learning_rate']

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    # Add LSTM layers
    num_lstm_layers = len(lstm_layers)

    for i, units in enumerate(lstm_layers):
        is_last_lstm = (i == num_lstm_layers - 1)
        return_sequences = not is_last_lstm

        if use_bidirectional and i == 0:  # Only first layer bidirectional
            model.add(keras.layers.Bidirectional(
                keras.layers.LSTM(units, return_sequences=return_sequences)
            ))
        else:
            model.add(keras.layers.LSTM(units, return_sequences=return_sequences))

        if use_batch_norm:
            model.add(keras.layers.BatchNormalization())

        if i < len(dropout_rates):
            model.add(keras.layers.Dropout(dropout_rates[i]))

    # Add Dense layer(s)
    model.add(keras.layers.Dense(dense_units, activation='relu'))

    if use_batch_norm:
        model.add(keras.layers.BatchNormalization())

    # Add dropout after dense layer
    dense_dropout_idx = num_lstm_layers
    if dense_dropout_idx < len(dropout_rates):
        model.add(keras.layers.Dropout(dropout_rates[dense_dropout_idx]))

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================================================================
# TRAINING WITH PARAMETERS
# ==============================================================================

def train_with_params(params, base_config, run_number, total_runs):
    """Train a single model with given parameters

    Args:
        params: Dictionary with parameters for this run
        base_config: Base CONFIG from train_model.py
        run_number: Current run number (for progress tracking)
        total_runs: Total number of runs

    Returns:
        Dictionary with results (metrics, run_id, etc.)
    """
    print("\n" + "="*70)
    print(f"GRID SEARCH RUN {run_number}/{total_runs}")
    print("="*70)
    print(f"Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")

    # Update CONFIG with parameters from grid
    config = base_config.copy()

    # Update data parameters if present
    if 'window_size' in params:
        config['window_size'] = params['window_size']
    if 'overlap' in params:
        config['overlap'] = params['overlap']
    if 'batch_size' in params:
        config['batch_size'] = params['batch_size']
    if 'learning_rate' in params:
        config['learning_rate'] = params['learning_rate']
    if 'enable_fps_scaling' in params:
        config['enable_fps_scaling'] = params['enable_fps_scaling']

    # Temporarily override train_model.CONFIG
    original_config = train_model.CONFIG.copy()
    train_model.CONFIG.update(config)

    try:
        # Load and process data (reuse functions from train_model)
        print("Loading data...")
        X_all, y_all, label_encoder = load_all_data(config)

        if len(X_all) == 0:
            print("❌ No sequences created. Skipping this combination.")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        # Calculate input shape
        num_features = len(train_model.SELECTED_LANDMARKS) * 4
        input_shape = (config['window_size'], num_features)
        num_classes = len(label_encoder.classes_)

        # Build model with grid parameters
        print("\nBuilding model...")
        model = build_model_from_params(params, input_shape, num_classes)
        model.summary()

        # Start MLflow run
        run_name = f"grid_search_{run_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Log all parameters (both from grid and from config)
            mlflow.log_param("grid_search_run", run_number)
            mlflow.log_param("total_grid_runs", total_runs)

            # Log grid parameters
            for key, value in params.items():
                if key == 'lstm_layers':
                    mlflow.log_param("lstm_layers", str(value))
                elif key == 'dropout_rates':
                    mlflow.log_param("dropout_rates", str(value))
                else:
                    mlflow.log_param(key, value)

            # Log data parameters
            mlflow.log_param("window_size", config['window_size'])
            mlflow.log_param("overlap", config['overlap'])
            mlflow.log_param("batch_size", config['batch_size'])
            mlflow.log_param("learning_rate", config['learning_rate'])
            mlflow.log_param("enable_fps_scaling", config['enable_fps_scaling'])
            mlflow.log_param("reference_fps", config['reference_fps'])
            mlflow.log_param("min_annotation_length", config['MIN_ANNOTATION_LENGTH'])
            mlflow.log_param("epochs", config['epochs'])
            mlflow.log_param("group_classes", config['group_classes'])

            # Log dataset statistics
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("num_features", num_features)
            mlflow.log_param("num_landmarks", len(train_model.SELECTED_LANDMARKS))
            mlflow.log_param("total_sequences", len(X_train) + len(X_test))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("input_shape", str(input_shape))

            # Log architecture details (extracted from model)
            lstm_layer_count = len(params['lstm_layers'])
            mlflow.log_param("num_lstm_layers", lstm_layer_count)
            mlflow.log_param("num_dense_layers", 1)  # Currently always 1 dense layer
            for i, units in enumerate(params['lstm_layers'], 1):
                mlflow.log_param(f"lstm_layer_{i}_units", units)
            mlflow.log_param("dense_layer_1_units", params['dense_units'])
            mlflow.log_param("uses_batch_normalization", params['use_batch_norm'])
            mlflow.log_param("uses_bidirectional", params.get('use_bidirectional', False))

            # Log architecture summary as a single string for easy viewing
            arch_summary = f"LSTM{params['lstm_layers']} → Dense[{params['dense_units']}]"
            if params['use_batch_norm']:
                arch_summary += " + BatchNorm"
            mlflow.log_param("architecture_summary", arch_summary)

            # Calculate class weights
            from sklearn.utils.class_weight import compute_class_weight
            class_weights_array = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = dict(enumerate(class_weights_array))

            # Log class weights
            for idx, weight in class_weights.items():
                class_name = label_encoder.classes_[idx]
                mlflow.log_param(f"class_weight_{class_name}", f"{weight:.3f}")

            # Log class distribution (as metrics, not params)
            y_all_encoded = np.concatenate([y_train, y_test])
            for idx, label in enumerate(label_encoder.classes_):
                count = np.sum(y_all_encoded == idx)
                mlflow.log_metric(f"class_{label}_count", count)

            # Setup callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.00005,
                    verbose=1
                )
            ]

            # Train model
            print("\nTraining...")
            training_start = time.time()

            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )

            training_duration = time.time() - training_start

            # Evaluate
            print("\nEvaluating...")
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            # Predictions
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)

            # Calculate metrics
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test,
                y_pred_classes,
                labels=range(num_classes),
                zero_division=0
            )

            # Log training time metrics
            mlflow.log_metric("training_time_seconds", training_duration)
            mlflow.log_metric("training_time_minutes", training_duration / 60)

            # Log performance metrics
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
            mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
            mlflow.log_metric("final_train_loss", history.history['loss'][-1])
            mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
            mlflow.log_metric("epochs_trained", len(history.history['loss']))

            # Per-class metrics
            for idx, class_name in enumerate(label_encoder.classes_):
                mlflow.log_metric(f"{class_name}_precision", float(precision[idx]))
                mlflow.log_metric(f"{class_name}_recall", float(recall[idx]))
                mlflow.log_metric(f"{class_name}_f1_score", float(f1[idx]))
                mlflow.log_metric(f"{class_name}_support", int(support[idx]))

            # Macro average metrics
            macro_f1 = float(f1.mean())
            mlflow.log_metric("macro_avg_f1_score", macro_f1)
            mlflow.log_metric("macro_avg_precision", float(precision.mean()))
            mlflow.log_metric("macro_avg_recall", float(recall.mean()))

            # Weighted average metrics
            total_support = support.sum()
            weighted_precision = (precision * support).sum() / total_support
            weighted_recall = (recall * support).sum() / total_support
            weighted_f1 = (f1 * support).sum() / total_support

            mlflow.log_metric("weighted_avg_precision", float(weighted_precision))
            mlflow.log_metric("weighted_avg_recall", float(weighted_recall))
            mlflow.log_metric("weighted_avg_f1_score", float(weighted_f1))

            # Log the trained model
            mlflow.keras.log_model(model, "model")


            print(f"Model logged in run ")

            # Get run ID
            run_id = mlflow.active_run().info.run_id

            print(f"\n✅ Run {run_number}/{total_runs} complete:")
            print(f"   Test Accuracy: {test_acc:.4f}")
            print(f"   Macro F1-Score: {macro_f1:.4f}")
            print(f"   Training Time: {training_duration/60:.2f} min")
            print(f"   MLflow Run ID: {run_id}")

            # Return results
            results = {
                'run_number': run_number,
                'run_id': run_id,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'macro_f1_score': macro_f1,
                'training_time_minutes': training_duration / 60,
                'epochs_trained': len(history.history['loss']),
                **params  # Include all parameters
            }

            return results

    except Exception as e:
        print(f"\n❌ Error in run {run_number}: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Restore original config
        train_model.CONFIG = original_config


def load_all_data(config):
    """Load and preprocess all data

    Reuses functions from train_model.py
    Uses saved poses if config['use_saved_poses'] is True

    Returns:
        X_all: All sequences
        y_all: All labels (encoded)
        label_encoder: Fitted label encoder
    """
    from sklearn.preprocessing import LabelEncoder

    all_X, all_y = [], []

    if config.get('use_saved_poses', False):
        # ========== FAST PATH: Load from saved poses ==========
        print("\n🚀 FAST MODE: Loading from saved pose data (Grid Search)")

        all_pose_data = train_model.load_saved_poses(config['pose_data_dir'])

        for pose_data in all_pose_data:
            video_filename = pose_data['video_filename']
            landmarks = pose_data['landmarks']
            fps = pose_data['fps']
            annotations = pose_data['annotations']

            print(f"\n  Processing: {video_filename}")
            print(f"    Frames: {len(landmarks)}, FPS: {fps}")

            # Create training sequences
            X, y = train_model.create_sequences_from_frames(
                landmarks,
                annotations,
                fps,
                window_size=config['window_size'],
                overlap=config['overlap'],
                group_classes=config['group_classes'],
                min_annotation_length=config['MIN_ANNOTATION_LENGTH']
            )

            if len(X) == 0:
                print("    ⚠️  No sequences created")
                continue

            # Print label distribution
            unique, counts = np.unique(y, return_counts=True)
            print("    Label distribution:")
            for label, count in zip(unique, counts):
                print(f"      {label}: {count}")

            all_X.append(X)
            all_y.append(y)

    else:
        # ========== SLOW PATH: Extract poses from videos ==========
        print("\n🐢 SLOW MODE: Extracting poses from videos (Grid Search)")
        print("   (Consider running: poetry run python src/extract_poses.py)")

        # Initialize pose extractor
        pose_extractor = train_model.PoseExtractor()

        # Process all Label Studio JSON files
        json_files = list(Path(config['label_studio_exports']).glob('*.json'))
        print(f"\nFound {len(json_files)} annotation files")

        for json_file in json_files:
            print(f"\nProcessing: {json_file.name}")

            # Parse annotations
            video_filename, annotations = train_model.parse_label_studio_json_frames(json_file)
            video_path = str(Path(config['video_base_path']) / video_filename)

            print(f"  Video: {video_filename}")
            print(f"  Annotations: {len(annotations)}")

            # Check if video exists
            if not Path(video_path).exists():
                print(f"  WARNING: Video not found at {video_path}, skipping...")
                continue

            # Extract poses
            landmarks, fps = pose_extractor.extract_from_video(video_path)

            if landmarks is None:
                print(f"  Skipping {video_filename} - could not process video")
                continue

            print(f"  Extracted {len(landmarks)} pose sequences")

            # Create training sequences
            X, y = train_model.create_sequences_from_frames(
                landmarks,
                annotations,
                fps,
                window_size=config['window_size'],
                overlap=config['overlap'],
                group_classes=config['group_classes'],
                min_annotation_length=config['MIN_ANNOTATION_LENGTH']
            )

            # Skip videos with no sequences
            if len(X) == 0:
                print("  ⚠️  No sequences created for this video")
                continue

            # Print label distribution
            unique, counts = np.unique(y, return_counts=True)
            print("  Label distribution:")
            for label, count in zip(unique, counts):
                print(f"    {label}: {count}")

            all_X.append(X)
            all_y.append(y)

    # Combine all data
    if len(all_X) == 0:
        print("\nERROR: No data was processed. Check video paths and annotations.")
        return np.array([]), np.array([]), LabelEncoder()

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    print(f"\nTotal sequences: {len(X_all)}")
    print(f"Sequence shape: {X_all.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)

    print(f"\nLabel mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == idx)
        print(f"  {label} -> {idx} ({count} samples)")

    return X_all, y_encoded, label_encoder


# ==============================================================================
# MAIN GRID SEARCH LOGIC
# ==============================================================================

def run_grid_search(grid_name='small', max_runs=None, random_search=False, n_samples=30):
    """Run grid search with specified configuration

    Args:
        grid_name: Name of predefined grid or 'random'
        max_runs: Maximum number of runs (None = all combinations)
        random_search: If True, use random search instead of full grid
        n_samples: Number of samples for random search
    """
    print("\n" + "="*70)
    print("GRID SEARCH FOR TENNIS STROKE RECOGNITION")
    print("="*70)

    # Get parameter grid
    if random_search:
        print(f"\nMode: Random Search ({n_samples} samples)")
        param_ranges = get_grid('random')
        param_combinations = generate_random_samples(param_ranges, n_samples)
    else:
        print(f"\nMode: Full Grid Search")
        print(f"Grid: {grid_name}")
        param_grid = get_grid(grid_name)
        param_combinations = generate_full_grid(param_grid)

    total_combinations = len(param_combinations)

    # Limit runs if max_runs specified
    if max_runs is not None and max_runs < total_combinations:
        print(f"\nLimiting to {max_runs} of {total_combinations} combinations")
        param_combinations = param_combinations[:max_runs]
        total_combinations = max_runs

    print(f"\nTotal runs to execute: {total_combinations}")
    print("="*70)

    print("\nParameter combinations to be tested:")
    for x in param_combinations:
        print(x)

    # ------------------------
    # ------------------------
    # ------------------------
    
    # Ask user to confirm before running this configuration (useful to avoid long unwanted runs)
    # auto_confirm = os.environ.get('GRID_SEARCH_AUTO_CONFIRM', '0').lower() in ('1', 'true', 'yes')

    # if not auto_confirm:
    print("\nCONFIRMATION REQUIRED:")
    print("  You are about to start a training run with the parameters shown above.")
    print("  Options: [y]es to proceed, [n]o to skip this combination, [a]bort to stop the entire grid search.")
    try:
        choice = input("Proceed? (y/n/a): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNon-interactive or interrupted input detected. Set GRID_SEARCH_AUTO_CONFIRM=1 to auto-approve. Skipping this run.")
        return None

    if choice in ('y', 'yes'):
        pass  # continue with the run
    elif choice in ('n', 'no'):
        print("Skipped this combination by user choice.")
        return None
    elif choice in ('a', 'abort'):
        print("Aborting entire grid search by user request.")
        raise KeyboardInterrupt("User aborted grid search")
    else:
        print("Unrecognized choice. Skipping this run.")
        return None
    # ------------------------
    # ------------------------
    # ------------------------


    # Setup MLflow
    mlflow.set_experiment(train_model.CONFIG['mlflow_experiment_name'])

    # Run all combinations
    all_results = []
    start_time = time.time()

    for i, params in enumerate(param_combinations, 1):
        results = train_with_params(
            params,
            train_model.CONFIG,
            run_number=i,
            total_runs=total_combinations
        )

        if results is not None:
            all_results.append(results)

        # Estimate remaining time
        elapsed = time.time() - start_time
        avg_time_per_run = elapsed / i
        remaining_runs = total_combinations - i
        estimated_remaining = avg_time_per_run * remaining_runs

        print(f"\n📊 Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        print(f"⏱️  Elapsed: {elapsed/60:.1f} min | Estimated remaining: {estimated_remaining/60:.1f} min")

    # Generate summary
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE!")
    print("="*70)

    if all_results:
        # Save results to CSV
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = Path(train_model.CONFIG['output_dir']) / f"grid_search_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")

        # Find best model
        best_idx = results_df['macro_f1_score'].idxmax()
        best_result = results_df.loc[best_idx]

        print(f"\n🏆 BEST MODEL:")
        print(f"   Run: {best_result['run_number']}")
        print(f"   Macro F1-Score: {best_result['macro_f1_score']:.4f}")
        print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"   MLflow Run ID: {best_result['run_id']}")
        print(f"\n   Parameters:")
        for key in ['lstm_layers', 'dense_units', 'dropout_rates', 'learning_rate', 'batch_size']:
            if key in best_result:
                print(f"     {key}: {best_result[key]}")

        # Print top 5 models
        print(f"\n📈 TOP 5 MODELS (by Macro F1-Score):")
        top_5 = results_df.nlargest(5, 'macro_f1_score')
        for idx, row in top_5.iterrows():
            print(f"   {row['run_number']:3d}. F1={row['macro_f1_score']:.4f}, Acc={row['test_accuracy']:.4f}, Time={row['training_time_minutes']:.1f}min")

    total_time = time.time() - start_time
    print(f"\n⏱️  Total grid search time: {total_time/3600:.2f} hours")
    print(f"✅ View all results in MLflow: mlflow ui")
    print("="*70 + "\n")


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Grid search for tennis stroke recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--grid', type=str, default='small',
                       choices=['minimal', 'small', 'medium', 'large', 'architecture', 'hyperparameter', 'data'],
                       help='Predefined grid configuration to use')
    parser.add_argument('--random', action='store_true',
                       help='Use random search instead of full grid')
    parser.add_argument('--n-samples', type=int, default=30,
                       help='Number of samples for random search')
    parser.add_argument('--max-runs', type=int, default=None,
                       help='Maximum number of runs to execute (limits grid search)')
    parser.add_argument('--show-grid', action='store_true',
                       help='Show grid configuration and exit (no training)')

    args = parser.parse_args()

    if args.show_grid:
        print_grid_summary(args.grid)
        sys.exit(0)

    # Run grid search
    run_grid_search(
        grid_name=args.grid,
        max_runs=args.max_runs,
        random_search=args.random,
        n_samples=args.n_samples
    )
