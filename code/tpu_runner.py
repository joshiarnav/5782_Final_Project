"""TPU-specific runner for arithmetic transformer model.

This script is specifically designed to run on Google Colab TPUs.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from dataset import ArithmeticDataset
from model import ArithmeticTransformer
from config import get_argument_parser


def train_on_tpu(args=None):
    """Run training on TPU with optimized settings."""
    # Parse args if not provided
    if args is None:
        parser = get_argument_parser()
        # Add trainer args
        parser.add_argument('--max_epochs', type=int, default=20)
        parser.add_argument('--accumulate_grad_batches', type=int, default=4)
        parser.add_argument('--gradient_clip_val', type=float, default=1.0)
        parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
        args = parser.parse_args()
    
    print(f"Configuration: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    pl.seed_everything(args.seed)
    
    print("Creating datasets...")
    # Create datasets with smaller sizes for TPU testing
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
    
    print("Creating dataloaders...")
    # Critical TPU settings: num_workers=0, persistent_workers=False
    train_dataloader = DataLoader(
        dataset_train, 
        batch_size=args.train_batch_size,
        shuffle=True, 
        num_workers=0,  # Must be 0 for TPU
        persistent_workers=False,
        pin_memory=False  # TPUs don't benefit from pinned memory
    )
    
    val_dataloader = DataLoader(
        dataset_val, 
        batch_size=args.val_batch_size, 
        shuffle=False,
        num_workers=0,  # Must be 0 for TPU
        persistent_workers=False,
        pin_memory=False  # TPUs don't benefit from pinned memory
    )
    
    test_dataloader = DataLoader(
        dataset_test, 
        batch_size=args.val_batch_size,
        shuffle=False, 
        num_workers=0,  # Must be 0 for TPU
        persistent_workers=False,
        pin_memory=False  # TPUs don't benefit from pinned memory
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
        save_weights_only=True,  # Save only weights for TPU compatibility
        every_n_epochs=args.check_val_every_n_epoch
    )
    
    print("Creating trainer...")
    # TPU-specific trainer settings
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'callbacks': [checkpoint_callback],
        'gradient_clip_val': args.gradient_clip_val,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'deterministic': False,  # TPUs work better with non-deterministic operations
        'accelerator': 'tpu',
        'devices': 'auto',
        'precision': 'bf16-mixed',  # Use bfloat16 for TPU
        'logger': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'num_sanity_val_steps': 0,  # Skip sanity check to avoid hanging
    }
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    print("Creating model...")
    # Create model with TPU-specific settings
    model = ArithmeticTransformer(
        config=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    
    # Explicitly set model to train mode
    model.train()
    
    print("Starting training...")
    trainer.fit(model)
    
    print("Training complete. Running evaluation...")
    results = trainer.test(model)
    
    print(f"Test results: {results}")
    return results


def run_from_colab(train_size=1000, val_size=200, test_size=200, 
                  max_digits_train=5, max_digits_test=5,
                  batch_size=32, max_epochs=5):
    """Helper function to run from a Colab notebook with custom parameters."""
    # Create a simple namespace object to hold args
    class Args:
        pass
    
    args = Args()
    
    # Set all required parameters
    args.output_dir = './output'
    args.model_name_or_path = 't5-base'
    args.operation = 'addition'
    args.orthography = '10ebased'
    args.balance_train = True
    args.balance_val = True
    args.balance_test = False
    args.min_digits_train = 2
    args.min_digits_test = 2
    args.max_digits_train = max_digits_train
    args.max_digits_test = max_digits_test
    args.base_number = 10
    args.seed = 1
    args.train_size = train_size
    args.val_size = val_size
    args.test_size = test_size
    args.train_batch_size = batch_size
    args.val_batch_size = batch_size * 2  # Smaller ratio for TPU
    args.optimizer = 'AdamW'
    args.lr = 3e-4
    args.weight_decay = 5e-5
    args.scheduler = 'StepLR'
    args.gamma = 1.0
    args.step_size = 1000
    args.t_0 = 2
    args.t_mult = 2
    args.max_seq_length = 512
    args.max_epochs = max_epochs
    args.accumulate_grad_batches = 4
    args.gradient_clip_val = 1.0
    args.check_val_every_n_epoch = 1
    
    return train_on_tpu(args)


if __name__ == '__main__':
    train_on_tpu()
