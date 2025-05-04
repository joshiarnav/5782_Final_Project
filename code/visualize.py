"""Visualization utilities for arithmetic model results.

This script provides functions to visualize the performance of arithmetic models
across different number representations and digit lengths.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_dir):
    """Load results from JSON files in the specified directory.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        List of result dictionaries
    """
    results = []
    for results_file in glob.glob(os.path.join(results_dir, '**/results.json'), recursive=True):
        try:
            with open(results_file, 'r') as f:
                result = json.load(f)
                # Add the directory name as experiment ID
                result['experiment_id'] = os.path.basename(os.path.dirname(results_file))
                results.append(result)
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
    
    return results


def plot_accuracy_by_digits(results, output_file=None):
    """Plot accuracy by maximum number of digits for different orthographies.
    
    Args:
        results: List of result dictionaries
        output_file: Optional path to save the plot
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Group by orthography and max_digits_test
    grouped = df.groupby(['orthography', 'max_digits_test'])['test_exact_match'].mean().reset_index()
    
    # Get unique orthographies
    orthographies = sorted(grouped['orthography'].unique())
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Color map for different orthographies
    colors = plt.cm.tab10(np.linspace(0, 1, len(orthographies)))
    
    for i, orth in enumerate(orthographies):
        data = grouped[grouped['orthography'] == orth]
        plt.plot(data['max_digits_test'], data['test_exact_match'], 
                 marker='o', linestyle='-', linewidth=2, label=orth, color=colors[i])
    
    plt.xlabel('Maximum Number of Digits', fontsize=14)
    plt.ylabel('Test Accuracy (Exact Match)', fontsize=14)
    plt.title('Model Accuracy by Number of Digits and Representation', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Number Representation', fontsize=12)
    plt.ylim(0, 1.05)  # Accuracy is between 0 and 1
    
    # Add horizontal lines for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_accuracy_by_orthography(results, output_file=None):
    """Plot accuracy by orthography for different operations.
    
    Args:
        results: List of result dictionaries
        output_file: Optional path to save the plot
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Group by operation and orthography
    grouped = df.groupby(['operation', 'orthography'])['test_exact_match'].mean().reset_index()
    
    # Get unique operations
    operations = sorted(grouped['operation'].unique())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Width of a bar 
    width = 0.35
    
    # Get unique orthographies and sort them
    orthographies = sorted(grouped['orthography'].unique())
    x = np.arange(len(orthographies))
    
    # Plot bars for each operation
    for i, operation in enumerate(operations):
        data = grouped[grouped['operation'] == operation]
        # Align the bars
        offset = width * (i - len(operations)/2 + 0.5)
        ax.bar(x + offset, data['test_exact_match'], width, label=operation)
    
    # Add labels and title
    ax.set_xlabel('Number Representation', fontsize=14)
    ax.set_ylabel('Test Accuracy (Exact Match)', fontsize=14)
    ax.set_title('Model Accuracy by Number Representation and Operation', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(orthographies, rotation=45, ha='right')
    ax.legend(title='Operation')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add horizontal lines for reference
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function to visualize results."""
    parser = argparse.ArgumentParser(description='Visualize arithmetic model results.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--plot_type', type=str, choices=['digits', 'orthography', 'all'], default='all',
                        help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found. Make sure the directory contains valid result files.")
        return
    
    print(f"Loaded {len(results)} result files.")
    
    # Generate plots
    if args.plot_type in ['digits', 'all']:
        output_file = os.path.join(args.output_dir, 'accuracy_by_digits.png') if args.output_dir else None
        plot_accuracy_by_digits(results, output_file)
    
    if args.plot_type in ['orthography', 'all']:
        output_file = os.path.join(args.output_dir, 'accuracy_by_orthography.png') if args.output_dir else None
        plot_accuracy_by_orthography(results, output_file)


if __name__ == '__main__':
    main()
