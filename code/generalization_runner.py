import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import Namespace
from scipy import stats
import gc
import torch

# Import train.py functions
from train import run_experiment

# Define the orthographies to test (as described in the paper)
ORTHOGRAPHIES = [
    '10ebased',   # "3 10e1 2 10e0"
    '10based',    # "3 10 2"
    'words',      # "thirty-two"
    'underscore', # "3_2"
    'character_fixed', # "0 0 3 2"
    'character',  # "3 2"
    'decimal',    # "32"
]

# Define the digit lengths to test
DIGIT_LENGTHS = [2, 5, 10, 15, 20, 25, 30]

# Default is a single seed for faster experimentation
SEEDS = [42]
# For multiple seeds, uncomment the following line
# SEEDS = [42, 123, 456, 789, 101]

# Default parameters for experiments
DEFAULT_PARAMS = {
    'operation': 'addition',
    'model_name_or_path': 't5-base',  # Closest to T5-220M in the paper
    'train_size': 10000,              # 10,000 examples for training
    'val_size': 1000,                 # 1,000 examples for checkpoint selection
    'test_size': 1000,                # Test set size
    'min_digits_train': 2,
    'min_digits_test': 30,            # Will be overridden
    'base_number': 10,
    'train_batch_size': 128,           # Reduced from 128 to save memory
    'val_batch_size': 256,             # Reduced from 256 to save memory
    'max_seq_length': 512,
    'num_workers': 4,
    'max_epochs': 25,
    'check_val_every_n_epoch': 10,
    'precision': 32,
    'gradient_clip_val': 1.0,
    'accumulate_grad_batches': 4,
    'optimizer': 'AdamW',
    'lr': 4e-4,
    'weight_decay': 5e-5,
    'scheduler': 'StepLR',
    'gamma': 1.0,
    'step_size': 1000,
    'balance_train': True,            # Balanced sampling for training
    'balance_val': True,              # Balanced sampling for validation
    'balance_test': True,             # Will be configurable via command line
    'invert_question': False,
    'invert_answer': False,
}

# Orthography-specific parameter overrides to handle memory issues
ORTHOGRAPHY_PARAMS = {
    '10ebased': {
        'train_batch_size': 32,  # Reduced batch size
        'val_batch_size': 64,    # Reduced batch size
        'precision': 16,         # Use 16-bit precision
        'accumulate_grad_batches': 2,  # Reduced accumulation
        'max_epochs': 15,        # Fewer epochs
    },
    'words': {
        'train_batch_size': 48,  # Slightly reduced batch size
        'precision': 16,         # Use 16-bit precision
    }
}

def load_results(results_dir):
    """Load existing results from the results directory."""
    results_file = os.path.join(results_dir, 'generalization_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}

def save_results(results, results_dir):
    """Save results to a JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'generalization_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

def calculate_statistics(accuracies):
    """Calculate mean and 95% confidence interval."""
    mean = np.mean(accuracies)
    
    # Calculate 95% confidence interval
    if len(accuracies) > 1:
        ci = stats.t.ppf(0.975, len(accuracies)-1) * stats.sem(accuracies)
        ci_lower = mean - ci
        ci_upper = mean + ci
    else:
        # No confidence interval with just one sample
        ci_lower = mean
        ci_upper = mean
    
    return mean, ci_lower, ci_upper

def train_model(orthography, max_digits, seed, output_dir, params=None, delete_checkpoints=False):
    """Train a model on the maximum digit length."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Update parameters for this experiment
    experiment_params = params.copy()
    experiment_params['orthography'] = orthography
    experiment_params['max_digits_train'] = max_digits
    experiment_params['min_digits_train'] = 2  # Always start from 2 digits
    experiment_params['max_digits_test'] = max_digits  # Initially test on same length
    experiment_params['min_digits_test'] = max_digits  # Initially test on same length
    experiment_params['seed'] = seed
    experiment_params['output_dir'] = os.path.join(output_dir, f"{orthography}_trained_on_{max_digits}_digits_seed{seed}")
    
    # Apply orthography-specific parameter overrides
    orthography_params = ORTHOGRAPHY_PARAMS.get(orthography, {})
    experiment_params.update(orthography_params)
    
    # Check if this experiment has already been run
    results_file = os.path.join(experiment_params['output_dir'], 'results.json')
    if os.path.exists(results_file):
        print(f"Experiment already completed for {orthography}, {max_digits} digits, seed {seed}. Skipping training.")
        try:
            with open(results_file, 'r') as f:
                result = json.load(f)
            test_accuracy = result.get('test_exact_match', 0.0)
            return experiment_params['output_dir'], test_accuracy
        except Exception as e:
            print(f"Error loading existing results: {e}. Will retrain.")
    
    # Convert to Namespace object for compatibility with train.py
    args = Namespace(**experiment_params)
    
    # Run the experiment directly by importing train.py
    print(f"\n{'='*80}\nTraining model: {orthography}, {max_digits} digits, seed {seed}\n{'='*80}\n")
    print(f"Configuration: {args}")
    
    try:
        # Call run_experiment from train.py
        result = run_experiment(args)
        test_accuracy = result.get('test_exact_match', 0.0)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return experiment_params['output_dir'], test_accuracy
    except Exception as e:
        print(f"Error training model: {e}")
        # Clean up GPU memory even if there's an error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None, 0.0

def test_model_on_digit_length(model_dir, orthography, test_digits, seed, balance_test, params=None):
    """Test a trained model on a specific digit length."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Create a temporary directory for test results
    test_output_dir = os.path.join(model_dir, f"test_on_{test_digits}_digits")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Check if this test has already been run
    results_file = os.path.join(test_output_dir, 'results.json')
    if os.path.exists(results_file):
        print(f"Test already completed for {orthography} on {test_digits} digits. Skipping test.")
        try:
            with open(results_file, 'r') as f:
                result = json.load(f)
            test_accuracy = result.get('test_exact_match', 0.0)
            return test_accuracy
        except Exception as e:
            print(f"Error loading existing test results: {e}. Will retest.")
    
    # Update parameters for testing
    test_params = params.copy()
    test_params['orthography'] = orthography
    test_params['max_digits_test'] = test_digits
    test_params['min_digits_test'] = test_digits
    test_params['max_digits_train'] = test_digits  # Needed for dataset creation
    test_params['min_digits_train'] = test_digits  # Needed for dataset creation
    test_params['seed'] = seed
    test_params['output_dir'] = test_output_dir
    test_params['balance_test'] = balance_test
    test_params['train_size'] = 10  # Small train set since we're not training
    test_params['val_size'] = 10    # Small val set since we're not validating
    test_params['test_size'] = 1000 # Standard test size
    
    # Apply orthography-specific parameter overrides
    orthography_params = ORTHOGRAPHY_PARAMS.get(orthography, {})
    test_params.update(orthography_params)
    
    # Find the best checkpoint in the model directory
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        return 0.0
    
    # Sort by validation score (assuming format: epoch=X-val_exact_match=Y.ckpt)
    checkpoints.sort(key=lambda x: float(x.split('val_exact_match=')[1].split('.ckpt')[0]) if 'val_exact_match=' in x else 0, reverse=True)
    best_checkpoint = os.path.join(model_dir, checkpoints[0])
    
    # Set the checkpoint path for loading
    test_params['checkpoint_path'] = best_checkpoint
    
    # Convert to Namespace object
    args = Namespace(**test_params)
    
    print(f"Testing model on {test_digits} digits (balanced={balance_test})")
    print(f"Using checkpoint: {best_checkpoint}")
    
    try:
        # Import and use evaluate function
        from evaluate import evaluate_model
        result = evaluate_model(args)
        test_accuracy = result.get('test_exact_match', 0.0)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return test_accuracy
    except Exception as e:
        print(f"Error testing model: {e}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        # Clean up GPU memory even if there's an error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return 0.0

def run_generalization_experiment(output_dir, orthographies=None, digit_lengths=None, 
                                 train_digits=30, params=None, seeds=None, 
                                 balance_test=True, delete_checkpoints=False):
    """Run generalization experiment: train on max_digits, test on all digit lengths."""
    # Set default values
    if orthographies is None:
        orthographies = ORTHOGRAPHIES
    if digit_lengths is None:
        digit_lengths = DIGIT_LENGTHS
    if seeds is None:
        seeds = SEEDS
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing results if available
    results = load_results(output_dir)
    
    # Initialize results structure if needed
    for orthography in orthographies:
        if orthography not in results:
            results[orthography] = {}
        
        for seed in seeds:
            seed_key = str(seed)
            if seed_key not in results[orthography]:
                results[orthography][seed_key] = {
                    'train_digits': train_digits,
                    'test_results': {}
                }
    
    # Run experiments for each orthography
    for orthography in orthographies:
        print(f"\n{'='*80}\nRunning generalization experiment for orthography: {orthography}\n{'='*80}\n")
        
        for seed in seeds:
            seed_key = str(seed)
            
            # Train model on the maximum digit length
            try:
                model_dir, train_accuracy = train_model(
                    orthography=orthography,
                    max_digits=train_digits,
                    seed=seed,
                    output_dir=output_dir,
                    params=params,
                    delete_checkpoints=delete_checkpoints
                )
                
                if model_dir is None:
                    print(f"Failed to train model for {orthography} with seed {seed}")
                    continue
                
                # Update training results
                results[orthography][seed_key]['train_accuracy'] = train_accuracy
                
                # Save intermediate results
                save_results(results, output_dir)
                
                # Test on all digit lengths
                for test_digits in digit_lengths:
                    # Skip if already tested
                    if str(test_digits) in results[orthography][seed_key]['test_results']:
                        print(f"Already tested {orthography} on {test_digits} digits with seed {seed}")
                        continue
                    
                    try:
                        test_accuracy = test_model_on_digit_length(
                            model_dir=model_dir,
                            orthography=orthography,
                            test_digits=test_digits,
                            seed=seed,
                            balance_test=balance_test,
                            params=params
                        )
                        
                        # Update test results
                        results[orthography][seed_key]['test_results'][str(test_digits)] = test_accuracy
                        
                        # Save after each test to enable resuming
                        save_results(results, output_dir)
                        
                    except Exception as e:
                        print(f"Error testing model on {test_digits} digits: {e}")
                        import traceback
                        traceback.print_exc()
                        # Clean up GPU memory even if there's an error
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Try to continue with next digit length
                        continue
            
            except Exception as e:
                print(f"Error in experiment for {orthography} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                # Clean up GPU memory even if there's an error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Try to continue with next seed
                continue
    
    # Calculate statistics and format final results
    final_results = {}
    for orthography in results:
        final_results[orthography] = {}
        
        # Get all seeds for this orthography
        seed_results = results[orthography]
        
        # For each digit length, calculate statistics across seeds
        for digit_length in digit_lengths:
            digit_key = str(digit_length)
            
            # Collect accuracies across seeds
            accuracies = []
            for seed_key in seed_results:
                if 'test_results' in seed_results[seed_key] and digit_key in seed_results[seed_key]['test_results']:
                    accuracies.append(seed_results[seed_key]['test_results'][digit_key])
            
            if accuracies:
                # Calculate statistics
                mean, ci_lower, ci_upper = calculate_statistics(accuracies)
                
                # Store in final results
                final_results[orthography][digit_key] = {
                    'mean': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n_samples': len(accuracies)
                }
    
    # Save the final processed results
    final_results_file = os.path.join(output_dir, 'generalization_final_results.json')
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Plot the results
    plot_generalization_results(final_results, output_dir, train_digits)
    
    return final_results

def plot_generalization_results(results, output_dir, train_digits):
    """Plot generalization results similar to Figure 1 in the paper."""
    plt.figure(figsize=(10, 6))
    
    # Define colors for different orthographies
    colors = {
        '10ebased': 'blue',
        '10based': 'green',
        'words': 'red',
        'underscore': 'purple',
        'character_fixed': 'orange',
        'character': 'brown',
        'decimal': 'black'
    }
    
    # Sort orthographies by their mean performance on the training digit length
    sorted_orthographies = sorted(
        results.keys(),
        key=lambda orth: results[orth].get(str(train_digits), {}).get('mean', 0),
        reverse=True
    )
    
    # Plot each orthography
    for orthography in sorted_orthographies:
        if orthography not in results:
            continue
        
        # Extract digit lengths and accuracies
        digit_lengths = []
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for digit_key in sorted(results[orthography].keys(), key=lambda x: int(x)):
            digit_lengths.append(int(digit_key))
            means.append(results[orthography][digit_key]['mean'])
            ci_lowers.append(results[orthography][digit_key]['ci_lower'])
            ci_uppers.append(results[orthography][digit_key]['ci_upper'])
        
        # Plot mean with confidence interval
        color = colors.get(orthography, 'gray')
        plt.plot(digit_lengths, means, marker='o', label=orthography, color=color)
        plt.fill_between(digit_lengths, ci_lowers, ci_uppers, alpha=0.2, color=color)
    
    # Add vertical line at training digit length
    plt.axvline(x=train_digits, color='gray', linestyle='--', 
                label=f'Training digit length ({train_digits})')
    
    # Set plot labels and title
    plt.xlabel('Number of Digits')
    plt.ylabel('Accuracy')
    plt.title(f'Generalization Performance: Models Trained on {train_digits} Digits')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.ylim(0, 1.05)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'generalization_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    # Show the plot
    plt.show()

def main():
    """Parse arguments and run the generalization experiment."""
    parser = argparse.ArgumentParser(description='Run generalization experiments for arithmetic transformer')
    parser.add_argument('--orthographies', nargs='+', default=ORTHOGRAPHIES,
                        help='Orthographies to test (e.g., 10ebased, 10based, words)')
    parser.add_argument('--digit_lengths', nargs='+', type=int, default=DIGIT_LENGTHS,
                        help='Digit lengths to test on')
    parser.add_argument('--train_digits', type=int, default=30,
                        help='Maximum number of digits to train on')
    parser.add_argument('--output_dir', type=str, default='generalization_results',
                        help='Directory to save results')
    parser.add_argument('--seeds', nargs='*', type=int, default=SEEDS,
                        help='Random seeds to use. If none provided, uses default seed 42.')
    parser.add_argument('--balance_test', action='store_true', 
                        help='Use balanced sampling for test sets (default: False = random sampling)')
    parser.add_argument('--delete_checkpoints', action='store_true',
                        help='Delete model checkpoints after experiment evaluation to save disk space')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only plot existing results without running experiments')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides default in DEFAULT_PARAMS)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size (overrides default in DEFAULT_PARAMS)')
    parser.add_argument('--precision', type=int, choices=[16, 32], default=None,
                        help='Training precision (16 or 32, overrides default in DEFAULT_PARAMS)')
    parser.add_argument('--gradient_accumulation', type=int, default=None,
                        help='Gradient accumulation steps (overrides default in DEFAULT_PARAMS)')
    parser.add_argument('--dataset_size', type=int, default=None,
                        help='Training dataset size (overrides default in DEFAULT_PARAMS)')
    
    args = parser.parse_args()
    
    # If no seeds are provided, use the default
    seeds = args.seeds if args.seeds else SEEDS
    
    # Update parameters if needed
    params = DEFAULT_PARAMS.copy()
    params['balance_test'] = args.balance_test
    if args.epochs is not None:
        params['max_epochs'] = args.epochs
    if args.batch_size is not None:
        params['train_batch_size'] = args.batch_size
    if args.precision is not None:
        params['precision'] = args.precision
    if args.gradient_accumulation is not None:
        params['accumulate_grad_batches'] = args.gradient_accumulation
    if args.dataset_size is not None:
        params['train_size'] = args.dataset_size
    
    if args.plot_only:
        # Only plot existing results
        results = load_results(args.output_dir)
        
        # Process results into the format expected by plot_generalization_results
        final_results = {}
        for orthography in results:
            final_results[orthography] = {}
            for seed_key in results[orthography]:
                seed_data = results[orthography][seed_key]
                for digit_key, accuracy in seed_data['test_results'].items():
                    if digit_key not in final_results[orthography]:
                        final_results[orthography][digit_key] = {
                            'accuracies': [],
                        }
                    final_results[orthography][digit_key]['accuracies'].append(accuracy)
        
        # Calculate statistics
        for orthography in final_results:
            for digit_key in final_results[orthography]:
                accuracies = final_results[orthography][digit_key]['accuracies']
                mean, ci_lower, ci_upper = calculate_statistics(accuracies)
                final_results[orthography][digit_key]['mean'] = mean
                final_results[orthography][digit_key]['ci_lower'] = ci_lower
                final_results[orthography][digit_key]['ci_upper'] = ci_upper
                final_results[orthography][digit_key]['n_samples'] = len(accuracies)
        
        plot_generalization_results(final_results, args.output_dir, args.train_digits)
    else:
        # Run the generalization experiment
        run_generalization_experiment(
            output_dir=args.output_dir,
            orthographies=args.orthographies,
            digit_lengths=args.digit_lengths,
            train_digits=args.train_digits,
            params=params,
            seeds=seeds,
            balance_test=args.balance_test,
            delete_checkpoints=args.delete_checkpoints
        )

if __name__ == '__main__':
    main()
