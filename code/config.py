import argparse

def get_argument_parser():
    """
    Create and return the argument parser for arithmetic tasks.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Train and evaluate transformer models on arithmetic problems.')
    
    # Data configuration
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save checkpoint and results.')
    parser.add_argument('--operation', type=str, required=True, help='Either "addition" or "subtraction".')
    parser.add_argument(
        '--orthography', type=str, required=True,
        help='Number representation: "decimal", "character", "character_fixed", "underscore", ' +
             '"words", "10based", or "10ebased"')
    parser.add_argument(
        '--invert_question', action='store_true',
        help="If passed, numbers in the question will be in the inverted order.")
    parser.add_argument(
        '--invert_answer', action='store_true',
        help="If passed, numbers in the answer will be in the inverted order.")
    parser.add_argument(
        '--balance_train', action='store_true',
        help='If passed, numbers in the training set will be sampled using the "balanced" method.')
    parser.add_argument(
        '--balance_val', action='store_true',
        help='If passed, numbers in the validation set will be sampled using the "balanced" method.')
    parser.add_argument(
        '--balance_test', action='store_true',
        help='If passed, numbers in the test set will be sampled using the "balanced" method.')
    
    # Number configuration
    parser.add_argument(
        '--min_digits_train', type=int, default=2,
        help='Minimum number of digits sampled for training and validation examples.')
    parser.add_argument(
        '--min_digits_test', type=int, default=2,
        help='Minimum number of digits sampled for test examples.')
    parser.add_argument(
        '--max_digits_train', type=int, required=True,
        help='Maximum number of digits sampled for training and validation examples.')
    parser.add_argument(
        '--max_digits_test', type=int, required=True,
        help='Maximum number of digits sampled for test examples.')
    parser.add_argument(
        '--base_number', type=int, default=10,
        help="Base of the number (e.g.; 2 -> binary, 10 -> decimal).")
    
    # Dataset configuration
    parser.add_argument("--seed", default=123, type=int, help="Random seed.")
    parser.add_argument("--train_size", default=1000, type=int, help="Number of examples for training.")
    parser.add_argument("--val_size", default=1000, type=int, help="Number of examples for validation.")
    parser.add_argument("--test_size", default=2000, type=int, help="Number of examples for testing.")
    
    # Model configuration
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length (in tokens).')
    
    # Training configuration
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use')
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--scheduler', type=str, default='StepLR',
                        help='learning rate scheduler. Currently, only StepLR is supported.')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma factor for ExponentialLR or StepLR')
    parser.add_argument('--step_size', type=int, default=2, help='period of learning rate decay (StepLR)')
    parser.add_argument('--t_0', type=int, default=2,
                        help='number of iterations for the first restart (CosineAnnealingWarmRestarts)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='a factor increases t_i after a restart (CosineAnnealingWarmRestarts)')
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPU workers for loading data.")
    
    return parser
