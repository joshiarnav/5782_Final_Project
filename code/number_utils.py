"""Utilities for number representation and conversion.

This module provides functions to convert numbers between different representations,
as described in the paper 'Investigating the Limitations of Transformers with Simple Arithmetic Tasks'.
"""

def convert_to_base(num: int, base: int, numerals="0123456789abcdefghijklmnopqrstuvwxyz") -> str:
    """
    Convert a decimal number to another base representation.
    
    Args:
        num: The decimal number to convert
        base: The target base (2-36)
        numerals: The characters to use for each digit
        
    Returns:
        String representation of the number in the target base
    """
    if num == 0:
        return numerals[0]
    
    result = ""
    while num > 0:
        result = numerals[num % base] + result
        num //= base
        
    return result


def convert_to_character(number: str, separator: str, invert_number: bool, max_digits: int) -> str:
    """
    Convert a number to a character-based representation with optional padding and inversion.
    
    Args:
        number: String representation of the number
        separator: Character to place between digits
        invert_number: Whether to invert the order of digits
        max_digits: Maximum number of digits (for padding). -1 for no padding.
        
    Returns:
        The formatted number string
    """
    # Handle negative numbers
    prefix = ""
    if number.startswith('-'):
        prefix = '-'
        number = number[1:]
    
    # Pad with leading zeros if needed
    if max_digits > 0:
        number = number.zfill(max_digits)
    
    # Apply inversion if requested
    if invert_number:
        number = number[::-1]
    
    # Add the prefix back if it was present
    result = prefix + separator.join(number)
    
    return result


def convert_to_10based(number: str, invert_number: bool) -> str:
    """
    Convert a number to a representation where each digit is followed by its place value.
    
    Args:
        number: String representation of the number
        invert_number: Whether to invert the order of digits
        
    Returns:
        The number in 10-based representation
    """
    # Handle negative numbers
    prefix = ""
    if number.startswith('-'):
        prefix = '-'
        number = number[1:]
    
    # Process each digit with its place value
    output = []
    for i, digit in enumerate(number[::-1]):
        if i > 0:
            output.append('1' + i * '0')
        output.append(digit)
    
    # Add the negative sign if needed
    if prefix:
        output.append(prefix)
    
    # The output is already inverted. If we want it to not be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]
    
    return ' '.join(output)


def convert_to_10ebased(number: str, split_type: str, invert_number: bool) -> str:
    """
    Convert a number to a representation where each digit is followed by its place value in scientific notation.
    
    Args:
        number: String representation of the number
        split_type: How to format the exponent (None, 'underscore', or 'character')
        invert_number: Whether to invert the order of digits
        
    Returns:
        The number in 10e-based representation
    """
    # Handle negative numbers
    prefix = ""
    if number.startswith('-'):
        prefix = '-'
        number = number[1:]
    
    # Process each digit with its place value in scientific notation
    output = []
    for i, digit in enumerate(number[::-1]):
        if split_type is None:
            output.append(f'10e{i}')
        elif split_type == 'underscore':
            output.append(f'10e{"_".join(str(i))}')
        elif split_type == 'character':
            output.append(' '.join(f'D{i}E'))
        else:
            raise ValueError(f'Unsupported split_type: {split_type}')
        
        output.append(digit)
    
    # Add the negative sign if needed
    if prefix:
        output.append(prefix)
    
    # The output is already inverted. If we want it to not be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]
    
    return ' '.join(output)
