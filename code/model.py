"""Transformer model for arithmetic tasks.

This module provides a PyTorch Lightning implementation of a transformer-based model
for solving arithmetic problems.
"""

import os
import json
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime

import torch
import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def compute_exact_match(predicted_answer: str, correct_answer: str) -> bool:
    """
    Compute exact match between predicted and correct answers.
    
    Args:
        predicted_answer: The model's prediction
        correct_answer: The ground truth answer
        
    Returns:
        True if the answers match (case-insensitive), False otherwise
    """
    predicted_answer = predicted_answer.strip().lower()
    correct_answer = correct_answer.strip().lower()
    return predicted_answer == correct_answer


class ArithmeticTransformer(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning transformer models on arithmetic tasks.
    """
    
    def __init__(self, 
                 config,
                 train_dataloader=None, 
                 val_dataloader=None, 
                 test_dataloader=None):
        """
        Initialize the model with the given configuration.
        
        Args:
            config: Configuration object with hyperparameters
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: DataLoader for test data
        """
        super().__init__()
        self.save_hyperparameters(config)
        
        # Convert config to Namespace if it's a dictionary
        from argparse import Namespace
        if isinstance(config, dict):
            self.config = Namespace(**config)
        else:
            self.config = config
        
        # Ensure model_name_or_path is set
        if not hasattr(self.config, 'model_name_or_path'):
            self.config.model_name_or_path = 't5-base'  # Default value
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name_or_path)
        
        # Add special tokens if needed for fixed character representation
        if getattr(self.config, 'orthography', '').endswith('_fixed'):
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['0']})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Store dataloaders
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        
        # For storing validation and test outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Setup logging
        self.log_file = self._setup_logging()
        
        # Set model to training mode
        self.model.train()
    
    def _setup_logging(self):
        """
        Set up a log file for storing string-based logs.
        
        Returns:
            Path to the log file
        """
        # Create logs directory in the output directory if it doesn't exist
        output_dir = getattr(self.config, 'output_dir', './output')
        log_dir = os.path.join(output_dir, 'logs')
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create log directory: {e}")
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'model_log_{timestamp}.txt')
        
        # Write header to log file
        try:
            with open(log_file, 'w') as f:
                f.write(f"=== Arithmetic Transformer Log - {timestamp} ===\n\n")
                f.write(f"Model: {getattr(self.config, 'model_name_or_path', 't5-base')}\n")
                f.write(f"Operation: {getattr(self.config, 'operation', 'addition')}\n")
                f.write(f"Orthography: {getattr(self.config, 'orthography', 'decimal')}\n\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
        
        return log_file
    
    def log_to_file(self, message):
        """
        Log a message to the log file.
        
        Args:
            message: Message to log
        """
        # Print to console
        print(message)
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def prepare_batch(self, questions: List[str], answers: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of data for the model.
        
        Args:
            questions: List of question strings
            answers: List of answer strings
            
        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        # Tokenize inputs
        input_dict = self.tokenizer.batch_encode_plus(
            questions, padding=True, truncation=True, max_length=self.config.max_seq_length, return_tensors='pt')
        
        # Tokenize outputs
        labels = self.tokenizer.batch_encode_plus(
            answers, padding=True, truncation=True, max_length=self.config.max_seq_length, return_tensors='pt')['input_ids']
        
        # Move tensors to the model's device
        input_ids = input_dict['input_ids'].to(self.device)
        attention_mask = input_dict['attention_mask'].to(self.device)
        labels = labels.to(self.device)
        
        return input_ids, attention_mask, labels
    
    def forward(self, **kwargs):
        """
        Forward pass through the model.
        
        Args:
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model outputs
        """
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and logs
        """
        questions, correct_answers = batch
        
        # Log sample data occasionally (on powers of 2)
        if batch_idx & (batch_idx - 1) == 0:
            # Log to file and print to console
            try:
                self.log_to_file(f"\n=== Training Batch {batch_idx} ===\n")
                self.log_to_file(f"Sample question: {questions[0]}")
                self.log_to_file(f"Sample answer: {correct_answers[0]}")
            except Exception as e:
                print(f"Warning: Error logging training sample: {e}")
        
        # Prepare batch and compute loss
        input_ids, attention_mask, labels = self.prepare_batch(
            questions=questions, answers=correct_answers)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return {'loss': loss}
    
    def inference_step(self, batch, batch_idx: int):
        """
        Common step for validation and testing.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        questions, correct_answers = batch
        
        # Prepare batch
        input_ids, attention_mask, _ = self.prepare_batch(
            questions=questions, answers=correct_answers)
        
        # Calculate appropriate max_length based on the longest correct answer
        # This ensures we generate enough tokens for the full answer
        max_answer_length = max([len(self.tokenizer.encode(answer)) for answer in correct_answers])
        # Add a buffer to ensure we have enough space
        max_length = max(max_answer_length + 10, 128)
        
        # Generate predictions
        batch_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=max_length,  # Dynamic max_length based on answer length
            num_beams=1,                # Use greedy decoding for validation
            early_stopping=True,        # Stop when a complete sequence is generated
            use_cache=True,             # Use KV cache for faster generation
            # return_dict_in_generate=False,  # Return tensor for efficiency
            # output_scores=False         # Don't compute scores for validation
        )
        
        # Decode predictions
        predicted_answers = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs
        ]
        
        # Compute exact matches
        exact_matches = [
            compute_exact_match(predicted_answer=predicted_answer, correct_answer=correct_answer)
            for predicted_answer, correct_answer in zip(predicted_answers, correct_answers)
        ]
        
        # Log sample data occasionally (on powers of 2)
        if batch_idx & (batch_idx - 1) == 0:
            # Log to file and print to console
            try:
                self.log_to_file(f"\n=== Validation Batch {batch_idx} ===\n")
                self.log_to_file(f"Sample question: {questions[0]}")
                self.log_to_file(f"Sample correct answer: {correct_answers[0]}")
                self.log_to_file(f"Sample predicted answer: {predicted_answers[0]}")
                self.log_to_file(f"Exact match: {exact_matches[0]}")
                self.log_to_file(f"Max answer length: {max_answer_length}, Max generation length: {max_length}")
            except Exception as e:
                print(f"Warning: Error logging validation sample: {e}")
        
        return {'exact_matches': exact_matches}
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        outputs = self.inference_step(batch, batch_idx)
        self.validation_step_outputs.append(outputs)
        return outputs
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with metrics
        """
        outputs = self.inference_step(batch, batch_idx)
        self.test_step_outputs.append(outputs)
        return outputs
    
    def on_validation_epoch_end(self):
        """
        Compute and log validation metrics at the end of the epoch.
        """
        exact_matches = []
        for x in self.validation_step_outputs:
            exact_matches.extend(x['exact_matches'])
        
        exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        
        # Log metrics
        metrics = {'val_exact_match': exact_match}
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        
        # Log to file
        try:
            self.log_to_file(f"\n=== Validation Epoch End ===\n")
            self.log_to_file(f"Validation Exact Match: {exact_match:.4f}")
        except Exception as e:
            print(f"Warning: Error logging validation epoch end: {e}")
        
        # Clear the outputs list
        self.validation_step_outputs = []
    
    def on_test_epoch_end(self):
        """
        Compute and log test metrics at the end of the epoch.
        """
        exact_matches = []
        for x in self.test_step_outputs:
            exact_matches.extend(x['exact_matches'])
        
        exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        
        # Log metrics
        metrics = {'test_exact_match': exact_match}
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        
        # Log to file
        try:
            self.log_to_file(f"\n=== Test Epoch End ===\n")
            self.log_to_file(f"Test Exact Match: {exact_match:.4f}")
        except Exception as e:
            print(f"Warning: Error logging test epoch end: {e}")
        
        # Clear the outputs list
        self.test_step_outputs = []
    
    def train_dataloader(self):
        """
        Return the training dataloader.
        
        Returns:
            Training dataloader
        """
        return self._train_dataloader
    
    def val_dataloader(self):
        """
        Return the validation dataloader.
        
        Returns:
            Validation dataloader
        """
        return self._val_dataloader
    
    def test_dataloader(self):
        """
        Return the test dataloader.
        
        Returns:
            Test dataloader
        """
        return self._test_dataloader
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Tuple of (optimizers, schedulers)
        """
        optimizer_name = getattr(self.config, 'optimizer', 'AdamW')
        scheduler_name = getattr(self.config, 'scheduler', 'StepLR')
        lr = getattr(self.config, 'lr', 0.0004)
        weight_decay = getattr(self.config, 'weight_decay', 5e-5)
        
        # Prepare optimizer parameters with weight decay differentiation
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        
        # Create optimizer
        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        
        # Create scheduler
        if scheduler_name == 'StepLR':
            step_size = getattr(self.config, 'step_size', 1000)
            gamma = getattr(self.config, 'gamma', 1.0)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f'Unsupported scheduler: {scheduler_name}')
        
        return [optimizer], [scheduler]
