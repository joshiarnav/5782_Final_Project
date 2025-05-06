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


def create_datasets(args):
    """Create training, validation, and test datasets."""
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
    
    return dataset_train, dataset_val, dataset_test


def create_dataloaders(args, dataset_train, dataset_val, dataset_test):
    """Create DataLoaders for training, validation, and testing."""
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
        pin_memory=not is_tpu
    )
    
    test_dataloader = DataLoader(
        dataset_test, 
        batch_size=args.val_batch_size,
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=not is_tpu
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def create_trainer(args):
    """Create a PyTorch Lightning trainer."""
    print("Creating trainer...")
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='model-{epoch:02d}-{val_exact_match:.4f}',
        monitor='val_exact_match',
        mode='max',
        save_top_k=1,
        verbose=True,
    )
    
    # Create trainer
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'gradient_clip_val': args.gradient_clip_val,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'accelerator': 'auto',
        'devices': 'auto',
        'precision': args.precision,
        'logger': True,
        'enable_progress_bar': True,
        'callbacks': [checkpoint_callback],
        'default_root_dir': args.output_dir,
        'benchmark': True,  # Enable cuDNN benchmarking for better performance
    }
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer, checkpoint_callback


def create_model(args, train_dataloader, val_dataloader, test_dataloader):
    """Create and initialize the model."""
    print("Creating model...")
    model = ArithmeticTransformer(
        config=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    
    return model


def train_and_evaluate(args, trainer, model):
    """Train the model and evaluate it."""
    # Train model
    print("Starting training...")
    trainer.fit(model)
    
    # Test model
    print("\nEvaluating model...")
    results = trainer.test(ckpt_path='best')
    
    return results


def save_results(args, results):
    """Save experiment results to a JSON file."""
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
    
    return output


def run_experiment(args=None):
    """Run a complete experiment with the given arguments."""
    # Parse arguments if not provided
    if args is None:
        parser = get_argument_parser()
        args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print(f"Configuration: {args}")
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        pl.seed_everything(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Apply GPU optimizations
    apply_optimizations()
    
    # Create datasets and dataloaders
    dataset_train, dataset_val, dataset_test = create_datasets(args)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        args, dataset_train, dataset_val, dataset_test
    )
    
    # Create trainer and model
    trainer, checkpoint_callback = create_trainer(args)
    model = create_model(args, train_dataloader, val_dataloader, test_dataloader)
    
    # Train and evaluate
    results = train_and_evaluate(args, trainer, model)
    
    # Save results
    output = save_results(args, results)
    
    # Clean up memory
    del trainer
    del model
    del train_dataloader
    del val_dataloader
    del test_dataloader
    del dataset_train
    del dataset_val
    del dataset_test
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Done!")
    return output


def main():
    """Main function to run from command line."""
    run_experiment()


if __name__ == '__main__':
    main()
