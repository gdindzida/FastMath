# General Matrix Multiplication Experiment

[Go back](../../../README.md#experiments)

## Problem setup

We have two matrices $A$ and $B$ which are $n \times n$ matrices. Then we perform matrix multiplication to get matrix $C$ and each element of matrix $C$ we calculate in following way:

$$
C_{ij} = \sum_{k=0}^{n}A_{ik}B_{kj} 
$$

Computational bandwith = $O(n^3)$

Memory bandwith = $O(n^2)$

Theoretically given the large $n$ this algorithm will become compute bounded.

## Naive GEMM

