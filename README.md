# Introduction
In this project, we reproduce the paper Investigating the Limitations of Transformers with Simple Arithmetic Tasks by Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. The work addresses the inability of standard pretrained transformer models to perform basic arithmetic such as addition. Motivated by the human capacity for abstraction and seeking to see if existing transformer models could attain the same ability, the authors of the original paper inject explicit digit‐position representations into the model’s input. The authors fine-tune T5 models of various sizes on generated addition and subtraction datasets using six distinct input representations and show that positional markers enable near-perfect accuracy on numbers up to 60 digits. 

# Chosen Result
Results
We aimed to reproduce Figure 1 from the original paper, which presents the test accuracies of various number format types on adding two numbers with between 2 to 30 digits. This result is significant because it highlights how the input representations affect the performance of transformers. 
<img width="347" alt="Screenshot 2025-05-12 at 10 07 41 PM" src="https://github.com/user-attachments/assets/22b8d340-0376-4cca-a900-4f72cf2baa6b" />

As shown above, the input representation of the arithmetic can greatly boost the performance of the transformer, achieving near perfect accuracies in some scenarios. We chose to reproduce this due to its central role in demonstrating the paper’s key claim: that transformers, when fine-tuned, can learn arithmetic operations to varying degrees depending on the input format. 

# GitHub Contents
In the code folder, the codebase is organized into the following modules:

- `train.py` - Main script for training and evaluating models
- `evaluate.py` - Script for evaluating trained models on custom examples
- `visualize.py` - Script for visualizing model performance across different configurations
- `model.py` - Implementation of the transformer-based arithmetic model
- `dataset.py` - Dataset implementation for generating arithmetic problems
- `number_utils.py` - Utilities for number representation and conversion
- `config.py` - Configuration and argument parsing


# Re-implementation Details
We generated our dataset through a process called balanced sampling. Given a maximum number of digits D (D = 30 in our implementation), sample a number d from [2, D] and further sample a number between [10d-1, 10d-1 - 1]. Repeat this twice to get two numbers to add, and substitute them into the format “What is [number1] plus [number2]?”. Using this sampling method and question format, a dataset of 10000 was created along with a validation set of 1000. Each dataset was then formatted to one of the 7 different input format types
<img width="347" alt="Screenshot 2025-05-12 at 10 20 24 PM" src="https://github.com/user-attachments/assets/52adf68c-712e-4ae3-bcca-367578154ac7" />
The 7 input formats are shown above in Table 1 (taken from the paper). They vary from no change to the number (decimal) to more tokenized versions and those with digit positions. 

Each formatted dataset was used to finetune, validate, and test a T5-base (220M parameter) transformer model. Finetuning was done with an AdamW optimizer with a learning rate of 3e-4 to 4e-4, weight decay of 5e-5, and gradient clipping at 1.0. Batch size was 64 over 25 epochs. The model was validated every two epochs and the best accuracy checkpoint was used. 

Compared to the original paper, we made some minor changes to speed training and reduce memory usage. Batch size was reduced from 128, but gradient accumulation steps was set to 4 to simulate larger batches. Additionally, more training data was used at the expense of less epochs, but the paper mentioned that they performed similar experiments and found that this configuration didn’t affect overall performance much.  

# Reproduction Steps

## Installation

First, install the required packages:
```
pip install -r requirements.txt
```

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


# Results/Insights
<img width="291" alt="Screenshot 2025-05-12 at 10 24 09 PM" src="https://github.com/user-attachments/assets/8c70ff99-9599-4ccd-9e21-277bcf80a35c" />
Our results (shown above) align with the paper’s: that 10e-based and 10-based formatting are the most effective with basic arithmetic tasks in transformers. Other formats did well with 2 digit numbers, but their performance plummeted with larger numbers, just like those in the paper. 

While the reproduced accuracies are slightly lower, they still achieved decent accuracies around 80% and reflected the advantages of each format. We believe that most of the discrepancies come from using smaller batch sizes (due to memory constraints) and minor changes to the AdamW. Regardless, our results reflect the paper and show that input formatting can be a key element of using transformers on arithmetic, as they can drastically change performance. 

# Conclusion
Transformers learn most accurately when introducing position tokens (e.g. “3 10e1 2”) with 10e based words having the highest test accuracy. This result is exhibited even with varying amounts of data, epochs, and batch sizes, highlighting the robustness of this technique. 

# References
Nogueira, R., Jiang, Z., & Lin, J. (2021). Investigating the limitations of transformers with simple arithmetic tasks. arXiv preprint arXiv:2102.13019.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140), 1-67.

