#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:30:49 2022

@author: ochwada
"""

# libraries

import numpy as np
import scipy
from scipy import sparse

# ------------ CREATING VECTORS ---------->
#as a row
vector_row = np.array([1,2,3,5])

#as a column
vector_col = np.array([ [1], [2], [3] ])

# ------------  ---------->
# ------------ CREATING MATRIX ---------->
#2D array

matrix = np.array([[1,2],
                   [3,4],
                   [5,6]
                   ])

# NOT recommended because - arrays are the de facto std data structure, and np operations return np arrays
matrix_object = np.mat([ [1,2],
                   [3,4],
                   [5,6] ]) #

# ------------  ---------->
# ------------ CREATING SPARSE MATRIX ---------->
# for data with very few NON-zero values

matrix2 = np.array([[0, 0],
                    [0, 1],
                    [3, 0]])

#create a Compressed sparse row (CSR)

matrix_sparse = sparse.csr_matrix(matrix2)

# ------------  ---------->
# ------------ SELECTING ELEMENTS---------->
vector= np.array([1,2,3,4,5,6])

matrix3 = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]
                   ])

#selecting all elements of a vector
allElements = vector[:]

# Select everything up to  and including the third element
upTo3rdElement = vector[:3]


# Select everything after the 3rd Element
from3rdElement =  vector[3:]

#Select the last element
lastElement = vector[-1]


# Select the 1st two rows and all columns of a matrix
matrix_1sts2rowAllCln = matrix3[:2, :]


















