# Arithmetic Transformer

This repository contains a modular implementation of the experiments from the paper:

[Nogueira, Jiang, Lin "Investigating the Limitations of Transformers with Simple Arithmetic Tasks", 2021](https://arxiv.org/abs/2102.13019)

## Installation

First, install the required packages:
```
pip install -r requirements.txt
```

## Project Structure

The codebase is organized into the following modules:

- `train.py` - Main script for training and evaluating models
- `evaluate.py` - Script for evaluating trained models on custom examples
- `visualize.py` - Script for visualizing model performance across different configurations
- `model.py` - Implementation of the transformer-based arithmetic model
- `dataset.py` - Dataset implementation for generating arithmetic problems
- `number_utils.py` - Utilities for number representation and conversion
- `config.py` - Configuration and argument parsing

## Training a Model

The command below trains and evaluates a T5-base model on the task of adding up to 15-digits:

```
python train.py 
    --output_dir=./output 
    --model_name_or_path=t5-base 
    --operation=addition 
    --orthography=10ebased 
    --balance_train 
    --balance_val 
    --train_size=100000 
    --val_size=10000 
    --test_size=10000 
    --min_digits_train=2 
    --max_digits_train=15 
    --min_digits_test=2 
    --max_digits_test=15 
    --base_number=10 
    --seed=1 
    --train_batch_size=4 
    --accumulate_grad_batches=32 
    --val_batch_size=32 
    --max_seq_length=512 
    --num_workers=4 
    --gpus=1 
    --optimizer=AdamW 
    --lr=3e-4 
    --weight_decay=5e-5 
    --scheduler=StepLR 
    --t_0=2 
    --t_mult=2 
    --gamma=1.0 
    --step_size=1000 
    --max_epochs=20 
    --check_val_every_n_epoch=2 
    --amp_level=O0 
    --precision=32 
    --gradient_clip_val=1.0
```

This training should take approximately 10 hours on a V100 GPU.

## Evaluating a Model

To evaluate a trained model on custom examples:

```
python evaluate.py 
    --checkpoint_dir=./output 
    --operation=addition 
    --orthography=10ebased 
    --max_digits=15 
    --examples "123,456" "7890,1234" "9999,9999"
```

## Visualizing Results

To visualize the performance of models across different configurations:

```
python visualize.py 
    --results_dir=./experiments 
    --output_dir=./plots 
    --plot_type=all
```

## Number Representations

The paper investigates how different number representations affect model performance:

- `decimal`: Standard decimal representation (e.g., "832")
- `character`: Space-separated digits (e.g., "8 3 2")
- `character_fixed`: Fixed-length space-separated digits (e.g., "0 8 3 2")
- `underscore`: Underscore-separated digits (e.g., "8_3_2")
- `words`: Word representation (e.g., "eight hundred thirty-two")
- `10based`: Digits with explicit place values (e.g., "8 100 3 10 2")
- `10ebased`: Digits with scientific notation place values (e.g., "8 10e2 3 10e1 2 10e0")

The results show that explicit position tokens (as in `10ebased`) enable the model to learn addition of numbers up to 60 digits with high accuracy.