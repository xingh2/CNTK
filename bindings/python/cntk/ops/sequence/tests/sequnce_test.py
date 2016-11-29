# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for sequence operations, each operation is tested for
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import _test_unary_op,_test_binary_op, AA, precision, PRECISION_TO_TYPE

def test_op_is_first(seq):
	expected_forward = np.eye(1, len(seq), dtype=seq.dtype)
	
	from .. import is_first
	_test_unary_op()


def test_op_is_last(seq):
	expected_forward = np.eye(1, len(seq), dtype=seq.dtype)[::1]

	from .. import is_last

def test_op_slice(seq,begin_index, end_index):
	expected_forward = seq[begin_index:end_index]

def test_op_first(seq):
	expected_forward = seq[0]

def test_op_last(seq):
	expected_forward = seq[-1]

def test_op_where(condition):
	expected_forward = np.where(seq)

def test_op_gather(seq,condition):
	expected_forward = seq[np.nonzero(condition)]

def test_op_scatter(seq,condition):
	a = np.zeros(shape=(len(condition[0]),) + seq.shape[1:])
	a[np.nonzero(condition[0])] = seq
	expected_forward = a

def test_op_broadcast_as(operand, broadcast_as_operand):
	a = np.zeros(shape=(len(broadcast_as_operand[0]),) + operand.shape[1:])
	for i in range(len(a)):
    	a[i]=operand
    expected_forward = a

def test_op_reduce_sum(seq):
	expected_forward = np.sum(seq, axis=0)