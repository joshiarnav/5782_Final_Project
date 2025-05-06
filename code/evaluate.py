"""Evaluation script for arithmetic models.

This script allows evaluating a trained model on custom arithmetic examples.
"""

import argparse
import glob
import os

import torch
from transformers import AutoTokenizer

from model import ArithmeticTransformer
from dataset import ArithmeticDataset
from number_utils import (
    convert_to_base,
    convert_to_character,
    convert_to_10based,
    convert_to_10ebased
)
from num2words import num2words


def format_number(number, orthography, max_digits, base_number, invert_number):
    """Format a number according to the specified orthography."""
    # Convert to string representation
    number_str = str(number)
    
    # Convert to specified base if not decimal
    if base_number != 10:
        number_str = convert_to_base(num=number, base=base_number)

    # Apply the appropriate formatting based on orthography
    if orthography == 'decimal':
        return convert_to_character(
            number=number_str, separator='', invert_number=invert_number,
            max_digits=-1)
    elif orthography == 'character':
        return convert_to_character(
            number=number_str, separator=' ', invert_number=invert_number,
            max_digits=-1)
    elif orthography == 'character_fixed':
        return convert_to_character(
            number=number_str, separator=' ', invert_number=invert_number,
            max_digits=max_digits)
    elif orthography == 'underscore':
        return convert_to_character(
            number=number_str, separator='_', invert_number=invert_number,
            max_digits=-1)
    elif orthography == 'words':
        return num2words(int(number_str))
    elif orthography == '10based':
        return convert_to_10based(number_str, invert_number=invert_number)
    elif orthography == '10ebased':
        return convert_to_10ebased(
            number_str, split_type=None, invert_number=invert_number)
    else:
        raise ValueError(f'Unsupported orthography: {orthography}')


def evaluate_model(args):
    """Evaluate a trained model on a test set.
    
    This function is used by the generalization_runner to evaluate
    a model trained on one digit length across different test digit lengths.
    """
    from train import create_datasets, create_dataloaders
    import pytorch_lightning as pl
    import gc
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        pl.seed_everything(args.seed)
    
    # Create test dataset with the specified digit length
    print(f"Creating test dataset with digits {args.min_digits_test} to {args.max_digits_test}")
    _, _, dataset_test = create_datasets(args)
    
    if dataset_test is None or len(dataset_test) == 0:
        print(f"Error: Failed to create test dataset. Check digit length parameters.")
        return {'test_exact_match': 0.0}
    
    # Create a dummy train and val dataset with minimal examples to avoid None errors
    dummy_dataset = ArithmeticDataset(
        n_examples=1,
        min_digits=args.min_digits_test,
        max_digits=args.max_digits_test,
        operation=args.operation,
        orthography=args.orthography,
        base_number=args.base_number,
        invert_question=args.invert_question,
        invert_answer=args.invert_answer,
        balanced=args.balance_test
    )
    
    # Create dataloaders with dummy datasets for train and val
    _, _, test_dataloader = create_dataloaders(args, dummy_dataset, dummy_dataset, dataset_test)
    
    # Load the model from checkpoint
    model = ArithmeticTransformer.load_from_checkpoint(
        args.checkpoint_path,
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()
    
    # Set up a simple trainer for evaluation only
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        logger=False,
        enable_progress_bar=True,
    )
    
    # Test the model
    results = trainer.test(model, test_dataloader, verbose=True)
    
    # Clean up to prevent memory leaks
    del model
    del trainer
    del test_dataloader
    del dataset_test
    if 'dummy_dataset' in locals():
        del dummy_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Return the test results
    if results and len(results) > 0:
        return {'test_exact_match': results[0].get('test_exact_match', 0.0)}
    return {'test_exact_match': 0.0}


def main():
    """Main function to evaluate the model on custom examples."""
    parser = argparse.ArgumentParser(description='Evaluate a trained arithmetic model on custom examples.')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--operation', type=str, required=True, choices=['addition', 'subtraction'], 
                        help='Arithmetic operation')
    parser.add_argument('--orthography', type=str, required=True, 
                        choices=['decimal', 'character', 'character_fixed', 'underscore', 'words', '10based', '10ebased'],
                        help='Number representation format')
    parser.add_argument('--max_digits', type=int, default=15, help='Maximum number of digits')
    parser.add_argument('--base_number', type=int, default=10, help='Base of the number system')
    parser.add_argument('--invert_question', action='store_true', help='Invert digits in questions')
    parser.add_argument('--invert_answer', action='store_true', help='Invert digits in answers')
    parser.add_argument('--examples', type=str, nargs='+', required=True, 
                        help='Examples to evaluate in format "num1,num2"')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Find the latest checkpoint
    checkpoint_paths = glob.glob(os.path.join(args.checkpoint_dir, '*.ckpt'))
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")
    
    best_checkpoint_path = checkpoint_paths[0]
    for path in checkpoint_paths[1:]:
        if os.path.getmtime(path) > os.path.getmtime(best_checkpoint_path):
            best_checkpoint_path = path
    
    print(f"Loading model from {best_checkpoint_path}")
    
    # Load model
    model = ArithmeticTransformer.load_from_checkpoint(best_checkpoint_path)
    model = model.to(args.device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model.config.model_name_or_path)
    if args.orthography.endswith('_fixed'):
        tokenizer.add_special_tokens({'additional_special_tokens': ['0']})
    
    # Process examples
    for example in args.examples:
        try:
            num1, num2 = map(int, example.split(','))
            
            # Format numbers
            num1_str = format_number(
                num1, args.orthography, args.max_digits, args.base_number, args.invert_question)
            num2_str = format_number(
                num2, args.orthography, args.max_digits, args.base_number, args.invert_question)
            
            # Compute expected result
            if args.operation == 'addition':
                operation_term = 'plus'
                result = num1 + num2
            else:  # subtraction
                operation_term = 'minus'
                result = num1 - num2
                
            expected_result_str = format_number(
                result, args.orthography, args.max_digits, args.base_number, args.invert_answer)
            
            # Format question
            question = f"What is {num1_str} {operation_term} {num2_str}?"
            print(f"\nQuestion: {question}")
            print(f"Expected: {expected_result_str}")
            
            # Tokenize input
            inputs = tokenizer(question, return_tensors="pt").to(args.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = model.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=512,
                    do_sample=False
                )
                
            # Decode prediction
            predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Predicted: {predicted}")
            
            # Check if correct
            is_correct = predicted.strip().lower() == expected_result_str.strip().lower()
            print(f"Correct: {is_correct}")
            
        except Exception as e:
            print(f"Error processing example {example}: {e}")


if __name__ == '__main__':
    main()
