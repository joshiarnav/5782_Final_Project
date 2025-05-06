#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for testing trained models on different digit lengths.

This script allows you to evaluate models trained with generalization_runner.py
across different digit lengths without retraining them.
"""

import os
import json
import argparse
import glob
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from scipy import stats

# Import from other modules
from evaluate import evaluate_model

# Define the digit lengths to test
DIGIT_LENGTHS = [2, 5, 10, 15, 20, 25, 30]

# Default parameters for testing
DEFAULT_PARAMS = {
    'operation': 'addition',
    'model_name_or_path': 't5-base',
    'train_size': 10,                 # Small train set since we're not training
    'val_size': 10,                   # Small val set since we're not validating
    'test_size': 1000,                # Test set size
    'base_number': 10,
    'train_batch_size': 64,           # Required for dataloader creation
    'val_batch_size': 64,             # Batch size for testing
    'max_seq_length': 512,
    'num_workers': 4,
    'max_epochs': 25,                 # Required for compatibility
    'check_val_every_n_epoch': 10,    # Required for compatibility
    'precision': 32,
    'gradient_clip_val': 1.0,         # Required for compatibility
    'accumulate_grad_batches': 4,     # Required for compatibility
    'optimizer': 'AdamW',             # Required for compatibility
    'lr': 0.0004,                     # Required for compatibility
    'weight_decay': 5e-05,            # Required for compatibility
    'scheduler': 'StepLR',            # Required for compatibility
    'gamma': 1.0,                     # Required for compatibility
    'step_size': 1000,                # Required for compatibility
    'balance_test': True,             # Will be configurable via command line
    'balance_train': True,            # Required for dataset creation
    'balance_val': True,              # Required for dataset creation
    'invert_question': False,
    'invert_answer': False,
}

def find_trained_models(base_dir):
    """Find all trained models in the base directory.
    
    Returns a dictionary mapping orthography to model directory.
    """
    models = {}
    
    # Look for directories that match the pattern {orthography}_trained_on_{digits}_digits_seed{seed}
    model_dirs = glob.glob(os.path.join(base_dir, "*_trained_on_*_digits_seed*"))
    
    for model_dir in model_dirs:
        dir_name = os.path.basename(model_dir)
        parts = dir_name.split('_trained_on_')
        
        if len(parts) != 2:
            continue
            
        orthography = parts[0]
        rest = parts[1]
        
        # Extract digits and seed
        try:
            digits_part = rest.split('_digits_seed')[0]
            seed_part = rest.split('_digits_seed')[1]
            train_digits = int(digits_part)
            seed = int(seed_part)
            
            if orthography not in models:
                models[orthography] = []
                
            # Check if the directory has checkpoint files
            checkpoints = glob.glob(os.path.join(model_dir, "*.ckpt"))
            if checkpoints:
                models[orthography].append({
                    'dir': model_dir,
                    'train_digits': train_digits,
                    'seed': seed,
                    'checkpoint': sorted(checkpoints, key=os.path.getmtime)[-1]  # Get the latest checkpoint
                })
        except:
            # Skip directories that don't match the expected format
            continue
    
    return models

def test_model_on_digit_length(model_info, test_digits, balance_test, params=None):
    """Test a trained model on a specific digit length."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    model_dir = model_info['dir']
    orthography = model_info['orthography']
    seed = model_info['seed']
    checkpoint_path = model_info['checkpoint']
    
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
    test_params['max_digits_train'] = test_digits  # Required by create_datasets
    test_params['min_digits_train'] = test_digits  # Match min and max for consistency
    test_params['seed'] = seed
    test_params['output_dir'] = test_output_dir
    test_params['balance_test'] = balance_test
    test_params['checkpoint_path'] = checkpoint_path
    
    # Convert to Namespace object
    args = Namespace(**test_params)
    
    print(f"Testing model on {test_digits} digits (balanced={balance_test})")
    print(f"Using checkpoint: {checkpoint_path}")
    
    try:
        # Use evaluate_model from evaluate.py
        result = evaluate_model(args)
        test_accuracy = result.get('test_exact_match', 0.0)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return test_accuracy
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        # Clean up GPU memory even if there's an error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return 0.0

def save_results(results, output_dir):
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'generalization_results.json')
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

def evaluate_generalization(base_dir, digit_lengths=None, balance_test=True, orthographies=None):
    """Evaluate trained models on different digit lengths."""
    if digit_lengths is None:
        digit_lengths = DIGIT_LENGTHS
    
    # Find trained models
    all_models = find_trained_models(base_dir)
    
    if not all_models:
        print(f"No trained models found in {base_dir}")
        return
    
    print(f"Found models for orthographies: {list(all_models.keys())}")
    
    # Filter orthographies if specified
    if orthographies:
        all_models = {k: v for k, v in all_models.items() if k in orthographies}
    
    # Initialize results structure
    results = {}
    for orthography, models in all_models.items():
        results[orthography] = {}
        
        # Group models by train_digits
        models_by_digits = {}
        for model in models:
            train_digits = model['train_digits']
            if train_digits not in models_by_digits:
                models_by_digits[train_digits] = []
            models_by_digits[train_digits].append(model)
        
        # For each train_digits, test on all digit_lengths
        for train_digits, train_models in models_by_digits.items():
            # Add orthography to each model info
            for model in train_models:
                model['orthography'] = orthography
            
            # Test each model on all digit lengths
            for test_digits in digit_lengths:
                digit_key = str(test_digits)
                
                # Collect accuracies for all models with the same train_digits
                accuracies = []
                for model in train_models:
                    print(f"\nTesting {orthography} model trained on {train_digits} digits (seed {model['seed']}) on {test_digits} digits")
                    accuracy = test_model_on_digit_length(
                        model_info=model,
                        test_digits=test_digits,
                        balance_test=balance_test,
                        params=DEFAULT_PARAMS
                    )
                    accuracies.append(accuracy)
                
                # Calculate statistics
                if accuracies:
                    mean, ci_lower, ci_upper = calculate_statistics(accuracies)
                    
                    # Store in results
                    if digit_key not in results[orthography]:
                        results[orthography][digit_key] = {}
                    
                    results[orthography][digit_key] = {
                        'mean': mean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n_samples': len(accuracies),
                        'train_digits': train_digits
                    }
                    
                    # Save results after each test to avoid losing progress
                    save_results(results, base_dir)
    
    # Plot results for each train_digits
    for train_digits in set(model['train_digits'] for models in all_models.values() for model in models):
        # Filter results for this train_digits
        filtered_results = {}
        for orthography, digit_results in results.items():
            filtered_results[orthography] = {}
            for digit_key, result in digit_results.items():
                if result.get('train_digits') == train_digits:
                    filtered_results[orthography][digit_key] = result
        
        # Plot results
        plot_generalization_results(filtered_results, base_dir, train_digits)
    
    return results

def main():
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained models on different digit lengths')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing trained models')
    parser.add_argument('--digit_lengths', nargs='+', type=int, default=DIGIT_LENGTHS,
                        help='Digit lengths to test on')
    parser.add_argument('--balance_test', action='store_true',
                        help='Use balanced sampling for test sets (default: False = random sampling)')
    parser.add_argument('--orthographies', nargs='+', default=None,
                        help='Orthographies to evaluate (if not specified, all found models will be evaluated)')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_generalization(
        base_dir=args.base_dir,
        digit_lengths=args.digit_lengths,
        balance_test=args.balance_test,
        orthographies=args.orthographies
    )

if __name__ == '__main__':
    main()
