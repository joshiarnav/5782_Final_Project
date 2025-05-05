"""Dataset module for arithmetic tasks.

This module provides dataset classes for generating and processing arithmetic problems.
"""

import random
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
from num2words import num2words

from number_utils import (
    convert_to_base,
    convert_to_character,
    convert_to_10based,
    convert_to_10ebased
)


class ArithmeticDataset(Dataset):
    """Dataset for generating arithmetic problems with different number representations."""
    
    def __init__(self, 
                 n_examples: int, 
                 min_digits: int, 
                 max_digits: int, 
                 operation: str,
                 orthography: str, 
                 base_number: int = 10, 
                 invert_question: bool = False, 
                 invert_answer: bool = False,
                 balanced: bool = False):
        """
        Initialize the dataset with the specified parameters.
        
        Args:
            n_examples: Number of examples to generate
            min_digits: Minimum number of digits in the generated numbers
            max_digits: Maximum number of digits in the generated numbers
            operation: Type of arithmetic operation ('addition' or 'subtraction')
            orthography: Number representation format
            base_number: Base of the number system (default: 10 for decimal)
            invert_question: Whether to invert digits in questions
            invert_answer: Whether to invert digits in answers
            balanced: Whether to use balanced sampling
        """
        self.operation = operation
        self.orthography = orthography
        self.invert_answer = invert_answer
        self.invert_question = invert_question
        self.base_number = base_number
        self.max_digits = max_digits
        
        # Validate configuration
        if self.base_number != 10:
            assert self.orthography != 'words', 'Cannot convert to words when base is different than 10.'
            assert self.operation == 'addition', f'Cannot perform {self.operation} when base is different than 10.'

        if self.invert_question or self.invert_answer:
            assert self.orthography != 'words', 'Cannot invert number when orthography = "words".'

        # Generate examples
        self.examples = self._generate_examples(n_examples, min_digits, max_digits, balanced)

    def _generate_examples(self, n_examples: int, min_digits: int, max_digits: int, balanced: bool) -> List[Tuple[int, int]]:
        """Generate number pairs for arithmetic operations.
        
        This implementation matches the original paper's code in test.py.
        """
        if balanced:
            # Balanced sampling as described in the paper
            examples = []
            for _ in range(n_examples):
                # Randomly select number of digits between min and max
                digits = random.randint(min_digits, max_digits)
                # Generate number with exactly 'digits' digits
                min_number = 10**(digits-1) if digits > 1 else 0
                max_number = (10**digits) - 1
                examples.append((random.randint(min_number, max_number), 
                               random.randint(min_number, max_number)))
        else:
            # Random sampling as described in the paper
            # This generates numbers with potentially any number of digits up to max_digits
            max_number = (10**max_digits) - 1
            examples = [
                (random.randint(0, max_number), random.randint(0, max_number))
                for _ in range(n_examples)
            ]
        
        return examples

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get an example at the specified index."""
        first_term, second_term = self.examples[idx]

        # Compute result based on operation
        if self.operation == 'addition':
            operation_term = 'plus'
            result = first_term + second_term
        elif self.operation == 'subtraction':
            operation_term = 'minus'
            result = first_term - second_term
        else:
            raise ValueError(f'Invalid operation: {self.operation}')

        # Convert numbers to the specified representation
        first_term_str = self._format_number(first_term, self.invert_question)
        second_term_str = self._format_number(second_term, self.invert_question)
        answer_str = self._format_number(result, self.invert_answer)

        # Format the question
        question = f'What is {first_term_str} {operation_term} {second_term_str}?'
        
        return question, answer_str

    def _format_number(self, number: int, invert_number: bool) -> str:
        """Format a number according to the specified orthography."""
        # Convert to string representation
        number_str = str(number)
        
        # Convert to specified base if not decimal
        if self.base_number != 10:
            number_str = convert_to_base(num=number, base=self.base_number)

        # Apply the appropriate formatting based on orthography
        if self.orthography == 'decimal':
            return convert_to_character(
                number=number_str, separator='', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'character':
            return convert_to_character(
                number=number_str, separator=' ', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'character_fixed':
            return convert_to_character(
                number=number_str, separator=' ', invert_number=invert_number,
                max_digits=self.max_digits)
        elif self.orthography == 'underscore':
            return convert_to_character(
                number=number_str, separator='_', invert_number=invert_number,
                max_digits=-1)
        elif self.orthography == 'words':
            return num2words(int(number_str))
        elif self.orthography == '10based':
            return convert_to_10based(number_str, invert_number=invert_number)
        elif self.orthography == '10ebased':
            return convert_to_10ebased(
                number_str, split_type=None, invert_number=invert_number)
        else:
            raise ValueError(f'Unsupported orthography: {self.orthography}')
