# Introduction
In this project, we reproduce the paper Investigating the Limitations of Transformers with Simple Arithmetic Tasks by Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. The work addresses the inability of standard pretrained transformer models to perform basic arithmetic such as addition. Motivated by the human capacity for abstraction and seeking to see if existing transformer models could attain the same ability, the authors of the original paper inject explicit digit‐position representations into the model’s input. The authors fine-tune T5 models of various sizes on generated addition and subtraction datasets using six distinct input representations and show that positional markers enable near-perfect accuracy on numbers up to 60 digits. 


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

## Running Experiments

To reproduce the experiments from the paper, use the `experiment_runner.py` script which tests different number representations across various digit lengths:

```
python experiment_runner.py 
    --output_dir=./experiment_results 
    --orthographies 10ebased 10based words underscore character_fixed character decimal
    --digit_lengths 2 5 10 15 20 25 30
    --train_size=10000
    --max_epochs=25
    --seed=42
```

For faster experimentation, you can run a subset of orthographies and digit lengths:

```
python experiment_runner.py 
    --output_dir=./experiment_results_test 
    --orthographies 10ebased decimal 
    --digit_lengths 2 5 
    --train_size=1000 
    --max_epochs=2
```

## Testing Generalization

To test how well models trained on one digit length generalize to other digit lengths, use the `generalization_runner.py` script:

```
python generalization_runner.py 
    --output_dir=./generalized_results 
    --train_digits=30 
    --orthographies 10ebased 10based
    --digit_lengths 2 5 10 15 20 25 30
    --balance_test
```

This trains models on the specified digit length (default: 30) and tests them on all digit lengths in the list.

## Evaluating Trained Models

To evaluate a trained model on custom examples:

```
python evaluate.py 
    --checkpoint_dir=./output 
    --operation=addition 
    --orthography=10ebased 
    --max_digits=15 
    --examples "123,456" "7890,1234" "9999,9999"
```

To evaluate generalization capabilities of previously trained models:

```
python evaluate_generalization.py 
    --base_dir=./generalized_results 
    --digit_lengths 2 5 10 15 20 25 30
    --balance_test
    --orthographies 10ebased 10based
```

## Visualizing Results

To visualize the performance of models across different configurations:

```
python visualize.py 
    --results_dir=./experiments 
    --output_dir=./plots 
    --plot_type=all
```

## Experiment Implementation

The file `experiment_implementation.ipynb` is a Jupyter notebook that was used to run all the experiments and generate the results presented in our project. It contains the complete workflow including:

1. Setting up the environment
2. Running the experiments with `experiment_runner.py`
3. Testing generalization with `generalization_runner.py`
4. Visualizing and analyzing the results

This notebook can be run in Google Colab with GPU acceleration for reproducing our experimental results.

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
