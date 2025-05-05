"""Experiment runner for arithmetic transformer models.

This module implements the experiment setup from the paper, with the option to run
with a single seed by default for faster experimentation.
"""

import os
import json
import glob
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from scipy import stats
from argparse import Namespace

# Import the train module
import sys
sys.path.append(os.getcwd())
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

# # Number of runs per configuration (for statistical significance)
# NUM_RUNS = 5

# Random seeds for each run
# Default is a single seed for faster experimentation
SEEDS = [42]
# For multiple seeds, uncomment the following line
# SEEDS = [42, 123, 456, 789, 101]

# Default parameters matching the paper
DEFAULT_PARAMS = {
    'operation': 'addition',
    'model_name_or_path': 't5-base',  # Closest to T5-220M in the paper
    'train_size': 1000,               # 1,000 examples as in the paper
    'val_size': 1000,                 # 1,000 examples for checkpoint selection
    'test_size': 1000,                # Test set size
    'min_digits_train': 2,
    'min_digits_test': 2,
    'base_number': 10,
    'train_batch_size': 128,
    'val_batch_size': 512,
    'max_seq_length': 512,
    'num_workers': 4,
    'max_epochs': 100,                # 100 epochs as in the paper
    'check_val_every_n_epoch': 5,
    'precision': 32,
    'gradient_clip_val': 1.0,
    'accumulate_grad_batches': 4,
    'optimizer': 'AdamW',
    'lr': 3e-4,
    'weight_decay': 5e-5,
    'scheduler': 'StepLR',
    'gamma': 1.0,
    'step_size': 1000,
    'balance_train': True,            # Balanced sampling as in the paper
    'balance_val': True,
    'balance_test': False,
    'invert_question': False,
    'invert_answer': False,
}

def load_results(results_dir):
    """Load existing results from the results directory."""
    results_file = os.path.join(results_dir, 'experiment_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}

def save_results(results, results_dir):
    """Save results to the results directory."""
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

def run_single_experiment(orthography, max_digits, seed, output_dir, params=None):
    """Run a single experiment with the given parameters."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Update parameters for this experiment
    experiment_params = params.copy()
    experiment_params['orthography'] = orthography
    experiment_params['max_digits_train'] = max_digits
    experiment_params['max_digits_test'] = max_digits
    experiment_params['seed'] = seed
    experiment_params['output_dir'] = os.path.join(output_dir, f"{orthography}_{max_digits}_digits_seed{seed}")
    
    # Convert to Namespace object for compatibility with train.py
    args = Namespace(**experiment_params)
    
    # Run the experiment directly by importing train.py
    print(f"\n{'='*80}\nRunning experiment: {orthography}, {max_digits} digits, seed {seed}\n{'='*80}\n")
    print(f"Configuration: {args}")
    
    try:
        # Call run_experiment from train.py
        result = run_experiment(args)
        return result.get('test_exact_match', 0.0)
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 0.0

def calculate_statistics(accuracies):
    """Calculate mean and 95% confidence interval."""
    mean = np.mean(accuracies)
    
    # Calculate 95% confidence interval
    if len(accuracies) > 1:
        ci = stats.t.interval(0.95, len(accuracies)-1, loc=mean, scale=stats.sem(accuracies))
        ci_lower = ci[0]
        ci_upper = ci[1]
    else:
        ci_lower = mean
        ci_upper = mean
    
    return mean, ci_lower, ci_upper

def plot_results(results, output_dir):
    """Plot the results similar to Figure 1 in the paper."""
    plt.figure(figsize=(10, 6))
    
    # Define markers and colors
    markers = ['o-', 's-', '^-', 'D-', 'x-', '*-', 'p-']
    colors = ['red', 'black', 'blue', 'gray', 'green', 'orange', 'purple']
    
    # Plot each orthography
    for i, orthography in enumerate(ORTHOGRAPHIES):
        if orthography in results:
            # Get x and y values
            x_values = sorted([int(digit) for digit in results[orthography].keys()])
            y_values = [results[orthography][str(x)]['mean'] for x in x_values]
            y_errors = [(results[orthography][str(x)]['mean'] - results[orthography][str(x)]['ci_lower'],
                        results[orthography][str(x)]['ci_upper'] - results[orthography][str(x)]['mean']) 
                       for x in x_values]
            
            # Transpose the error values for plt.errorbar
            y_errors = np.array(y_errors).T
            
            # Plot with label and error bars
            label = orthography
            if orthography == '10ebased':
                label = '10E-BASED "3 10e1 2 10e0"'
            elif orthography == '10based':
                label = '10-BASED "3 10 2"'
            elif orthography == 'words':
                label = 'WORDS "thirty-two"'
            elif orthography == 'underscore':
                label = 'UNDERSCORE "3_2"'
            elif orthography == 'character_fixed':
                label = 'FIXED-CHARACTER "0 0 3 2"'
            elif orthography == 'character':
                label = 'CHARACTER "3 2"'
            elif orthography == 'decimal':
                label = 'DECIMAL "32"'
                
            plt.errorbar(x_values, y_values, yerr=y_errors, fmt=markers[i % len(markers)], 
                        color=colors[i % len(colors)], label=label, capsize=4)
    
    # Set labels and title
    plt.xlabel('# of digits')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy of different number representations on the addition task')
    plt.grid(True)
    plt.legend(loc='best')
    plt.ylim(0, 1.05)  # Set y-axis from 0 to 1.05 for better visualization
    
    # Save the plot
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'figure1_reproduction.png')
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    
    # Also save as PDF for publication-quality figure
    pdf_file = os.path.join(output_dir, 'figure1_reproduction.pdf')
    plt.savefig(pdf_file)
    print(f"PDF plot saved to {pdf_file}")
    
    # Show the plot
    plt.show()

def run_experiments(output_dir, orthographies=None, digit_lengths=None, params=None, resume=True, seeds=None):
    """Run all experiments and generate plots."""
    # Set default values
    if orthographies is None:
        orthographies = ORTHOGRAPHIES
    if digit_lengths is None:
        digit_lengths = DIGIT_LENGTHS
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if seeds is None:
        seeds = SEEDS
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing results if resuming
    results = {}
    if resume:
        results = load_results(output_dir)
    
    # Run experiments for each orthography and digit length
    for orthography in orthographies:
        if orthography not in results:
            results[orthography] = {}
        
        for max_digits in digit_lengths:
            digit_key = str(max_digits)
            if digit_key not in results[orthography]:
                results[orthography][digit_key] = {'runs': {}}
            
            # Run with all specified seeds
            accuracies = []
            for seed in seeds:
                seed_key = str(seed)
                
                # Skip if already done
                if resume and seed_key in results[orthography][digit_key]['runs']:
                    print(f"Skipping experiment for {orthography}, {max_digits} digits, seed {seed} (already done)")
                    accuracies.append(results[orthography][digit_key]['runs'][seed_key])
                else:
                    # Run the experiment
                    print(f"\n{'='*80}\nRunning experiment: {orthography}, {max_digits} digits, seed {seed}\n{'='*80}\n")
                    accuracy = run_single_experiment(orthography, max_digits, seed, output_dir, params)
                    
                    # Save the result
                    results[orthography][digit_key]['runs'][seed_key] = accuracy
                    accuracies.append(accuracy)
                    
                    # Save after each experiment
                    save_results(results, output_dir)
            
            # Calculate statistics
            mean, ci_lower, ci_upper = calculate_statistics(accuracies)
            results[orthography][digit_key]['mean'] = mean
            results[orthography][digit_key]['ci_lower'] = ci_lower
            results[orthography][digit_key]['ci_upper'] = ci_upper
            
            # Save after calculating statistics
            save_results(results, output_dir)
    
    # Plot the final results
    plot_results(results, output_dir)
    
    return results

def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Run arithmetic transformer experiments')
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                        help='Directory to save results')
    parser.add_argument('--orthographies', type=str, nargs='+', choices=ORTHOGRAPHIES,
                        help='Specific orthographies to test')
    parser.add_argument('--digit_lengths', type=int, nargs='+',
                        help='Specific digit lengths to test')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not resume from previous experiments')
    parser.add_argument('--train_size', type=int, default=DEFAULT_PARAMS['train_size'],
                        help='Number of training examples')
    parser.add_argument('--max_epochs', type=int, default=DEFAULT_PARAMS['max_epochs'],
                        help='Maximum number of training epochs')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=DEFAULT_PARAMS['check_val_every_n_epoch'],
                        help='Run validation every N epochs')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only plot existing results without running experiments')
    parser.add_argument('--seed', type=int, default=SEEDS[0], nargs='*',
                        help='Random seed(s) for the experiment. Specify multiple seeds for statistical significance.')
    
    args = parser.parse_args()
    
    # Update parameters
    params = DEFAULT_PARAMS.copy()
    params['train_size'] = args.train_size
    params['max_epochs'] = args.max_epochs
    params['check_val_every_n_epoch'] = args.check_val_every_n_epoch
    
    # Plot only if requested
    if args.plot_only:
        results = load_results(args.output_dir)
        if results:
            plot_results(results, args.output_dir)
        else:
            print("No results found to plot.")
        return
    
    # Ensure seeds is a list
    seeds = args.seed
    if not seeds:  # If empty list, use default SEEDS
        seeds = SEEDS
    elif not isinstance(seeds, list):  # If single value, convert to list
        seeds = [seeds]
    
    # Run experiments
    run_experiments(
        output_dir=args.output_dir,
        orthographies=args.orthographies,
        digit_lengths=args.digit_lengths,
        params=params,
        resume=not args.no_resume,
        seeds=seeds
    )

if __name__ == '__main__':
    main()