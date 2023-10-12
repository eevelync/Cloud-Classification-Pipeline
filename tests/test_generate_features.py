import sys
import pandas.testing as pdt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pandas as pd
import numpy as np
import pytest
from generate_features import log_transform, multiply_columns, calculate_norm_range, calculate_range, generate_features



def test_log_transform_happy():
    data = pd.DataFrame({'a': [1, 2, 3]})
    result = log_transform(data, 'a', 'log_a')
    expected = pd.DataFrame({'a': [1, 2, 3], 'log_a': [0.0, 0.6931471805599453, 1.0986122886681098]})
    assert result.equals(expected)

def test_log_transform_unhappy():
    data = pd.DataFrame({'a': [-1, 0, 1]})
    with pytest.raises(ValueError):
        _ = log_transform(data, 'a', 'log_a')

def test_multiply_columns_happy():
    data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = multiply_columns(data, 'a', 'b', 'a_x_b')
    expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'a_x_b': [4, 10, 18]})
    assert result.equals(expected)

def test_multiply_columns_unhappy():
    data = pd.DataFrame({'a': [1, 2, 3]})
    with pytest.raises(KeyError):
        _ = multiply_columns(data, 'a', 'b', 'a_x_b')

def test_calculate_norm_range_happy():
    data = pd.DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6], 'mean': [2.5, 3.5, 4.5]})
    result = calculate_norm_range(data, 'min', 'max', 'mean', 'norm_range')
    expected = pd.DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6], 'mean': [2.5, 3.5, 4.5], 'norm_range': [1.2, 0.8571428571428571, 0.6666666666666666]})
    assert result.equals(expected)

def test_calculate_norm_range_unhappy():
    data = pd.DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6]})
    with pytest.raises(KeyError):
        _ = calculate_norm_range(data, 'min', 'max', 'mean', 'norm_range')

def test_calculate_range_happy():
    data = pd.DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6]})
    result = calculate_range(data, 'min', 'max', 'range')
    expected = pd.DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6], 'range': [3, 3, 3]})
    assert result.equals(expected)

def test_calculate_range_unhappy():
    data = pd.DataFrame({'min': [1, 2, 3]})
    with pytest.raises(KeyError):
        _ = calculate_range(data, 'min', 'max', 'range')

import pandas as pd
import numpy as np
import pytest
from generate_features import log_transform, multiply_columns, calculate_norm_range, calculate_range, generate_features

# ... previous tests ...

def test_generate_features_happy():
    data = pd.DataFrame({'visible_entropy': [1, 2, 3],
                         'visible_contrast': [4, 5, 6],
                         'IR_min': [1, 2, 3],
                         'IR_max': [4, 5, 6],
                         'IR_mean': [2.5, 3.5, 4.5]})

    config = {
        'calculate_norm_range': {
            'IR_norm_range': {
                'min_col': 'IR_min',
                'max_col': 'IR_max',
                'mean_col': 'IR_mean'
            }
        },
        'log_transform': {
            'log_entropy': 'visible_entropy'
        },
        'multiply': {
            'entropy_x_contrast': {
                'col_a': 'visible_contrast',
                'col_b': 'visible_entropy'
            }
        },
        'calculate_range': {
            'IR_range': {
                'min_col': 'IR_min',
                'max_col': 'IR_max'
            }
        }
    }

    result = generate_features(data, config)
    expected = pd.DataFrame({'visible_entropy': [1, 2, 3],
                         'visible_contrast': [4, 5, 6],
                         'IR_min': [1, 2, 3],
                         'IR_max': [4, 5, 6],
                         'IR_mean': [2.5, 3.5, 4.5],
                         'IR_norm_range': [1.2, 0.8571428571428571, 0.6666666666666666],
                         'log_entropy': [0.0, 0.6931471805599453, 1.0986122886681098],  
                         'entropy_x_contrast': [4, 10, 18],
                         'IR_range': [3, 3, 3]})

    
    if not result.equals(expected):
        print("Result DataFrame:")
        print(result)
        print("\nExpected DataFrame:")
        print(expected)
    
    # Sort the columns in both DataFrames
    result = result.sort_index(axis=1)
    expected = expected.sort_index(axis=1)
    
    pdt.assert_frame_equal(result, expected, check_exact=False, atol=1e-6)

def test_generate_features_unhappy():
    data = pd.DataFrame({'visible_entropy': [1, 2, 3],
                         'visible_contrast': [4, 5, 6],
                         'IR_min': [1, 2, 3],
                         'IR_max': [4, 5, 6],
                         'IR_mean': [2.5, 3.5, 4.5]})

    config = {
        'calculate_norm_range': {
            'IR_norm_range': {
                'min_col': 'wrong_col',
                'max_col': 'IR_max',
                'mean_col': 'IR_mean'
            }
        },
        'log_transform': {
            'log_entropy': 'visible_entropy'
        },
        'multiply': {
            'entropy_x_contrast': {
                'col_a': 'visible_contrast',
                'col_b': 'visible_entropy'
            }
        },
        'calculate_range': {
            'IR_range': {
                'min_col': 'IR_min',
                'max_col': 'IR_max'
            }
        }
    }

    with pytest.raises(KeyError):
        _ = generate_features(data, config)
