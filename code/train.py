"""Main training script for arithmetic tasks.

This script handles the training and evaluation of transformer models on arithmetic tasks.
"""

import os
import json
import glob
import random

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from config import get_argument_parser
from dataset import ArithmeticDataset
from model import ArithmeticTransformer

# Import GPU optimizations
try:
    from gpu_optimizations import apply_optimizations
except ImportError:
    # Define a dummy function if the module is not available
    def apply_optimizations():
        pass

def main():
    """Main function to train and evaluate the model."""
    # Parse arguments
    parser = get_argument_parser()
    # PyTorch Lightning 2.0+ no longer has add_argparse_args method
    # Instead, we'll add the trainer arguments manually
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32, help='Precision for training')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Validation check frequency')
    parser.add_argument('--amp_level', type=str, default='O1', help='Automatic mixed precision level')
    
    args = parser.parse_args()
    
    print(f"Configuration: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        pl.seed_everything(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seed set to {args.seed}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Apply GPU optimizations early
    apply_optimizations()
    
    # Create datasets
    print("Creating datasets...")
    dataset_train = ArithmeticDataset(
        n_examples=args.train_size, 
        min_digits=args.min_digits_train,
        max_digits=args.max_digits_train,
        operation=args.operation, 
        orthography=args.orthography,
        base_number=args.base_number, 
        invert_question=args.invert_question,
        invert_answer=args.invert_answer, 
        balanced=args.balance_train
    )
    
    dataset_val = ArithmeticDataset(
        n_examples=args.val_size, 
        min_digits=args.min_digits_train,
        max_digits=args.max_digits_train,
        operation=args.operation, 
        orthography=args.orthography,
        base_number=args.base_number, 
        invert_question=args.invert_question,
        invert_answer=args.invert_answer, 
        balanced=args.balance_val
    )
    
    dataset_test = ArithmeticDataset(
        n_examples=args.test_size,
        min_digits=args.min_digits_test,
        max_digits=args.max_digits_test,
        operation=args.operation, 
        orthography=args.orthography,
        base_number=args.base_number, 
        invert_question=args.invert_question,
        invert_answer=args.invert_answer, 
        balanced=args.balance_test
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    # For TPU compatibility, set num_workers=0 and disable multiprocessing
    is_tpu = 'TPU_NAME' in os.environ
    num_workers = 0 if is_tpu else args.num_workers
    persistent_workers = False if is_tpu else (num_workers > 0)
    
    train_dataloader = DataLoader(
        dataset_train, 
        batch_size=args.train_batch_size,
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=not is_tpu  # Pin memory only for GPU, not for TPU
    )
    
    val_dataloader = DataLoader(
        dataset_val, 
        batch_size=args.val_batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=not is_tpu  # Pin memory only for GPU, not for TPU
    )
    
    test_dataloader = DataLoader(
        dataset_test, 
        batch_size=args.val_batch_size,  # Use val_batch_size since test_batch_size doesn't exist
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=not is_tpu  # Pin memory only for GPU, not for TPU
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch}-{val_exact_match:.4f}',
        verbose=True, 
        save_last=False, 
        save_top_k=1, 
        mode='max', 
        monitor='val_exact_match',
        save_weights_only=False, 
        every_n_epochs=args.check_val_every_n_epoch
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'gradient_clip_val': args.gradient_clip_val,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'accelerator': 'auto',
        'devices': 'auto',
        'precision': args.precision,  # Use original precision setting (32)
        'logger': True,
        'enable_progress_bar': True,
        'callbacks': [checkpoint_callback],
        'default_root_dir': args.output_dir,
        'benchmark': True,  # Enable cuDNN benchmarking for better performance
    }
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Create model
    print("Creating model...")
    model = ArithmeticTransformer(
        config=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model)
    
    # Load best checkpoint
    print("Loading best checkpoint for testing...")
    checkpoint_paths = glob.glob(os.path.join(args.output_dir, '*.ckpt'))
    if checkpoint_paths:
        best_checkpoint_path = checkpoint_paths[0]
        for path in checkpoint_paths[1:]:
            if os.path.getmtime(path) > os.path.getmtime(best_checkpoint_path):
                best_checkpoint_path = path
                
        model = ArithmeticTransformer.load_from_checkpoint(
            best_checkpoint_path,
            config=args,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader
        )
    else:
        print("Warning: No checkpoint found. Using the last model state for testing.")
    
    # Test model
    print("Starting testing...")
    results = trainer.test(model)
    
    # Save results
    output = {
        'seed': args.seed,
        'max_digits_train': args.max_digits_train,
        'max_digits_test': args.max_digits_test,
        'operation': args.operation,
        'orthography': args.orthography,
        'base_number': args.base_number,
        'test_exact_match': results[0]['test_exact_match'] if results else None
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print("Done!")


if __name__ == '__main__':
    main()
