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

Naive Gemm implementation loops through rows and columns and calculates given sum. Implementation can be found [here](../../algorithms/gemm.cpp#19). Problem with this implementation is that retrieving values from cache is not optimized. For matrix $a$ cache value retrieval is fine since inner matrix memory is organized in per row basis. But for matrix $b$ values are fetched column by column so CPU needs to jump between memory adresses that are not close to each other. By transposing second matrix we get faster fatching and faster computation even though we have more computation to do (since we first need to transpose the matrix). Implementation can be found [here](../../algorithms/gemm.cpp#34). Static memory is used in order to avoid memory initialization every time. 

Without OpenMP:

![alt text](image.png)

With OpenMP:

![alt text](image-1.png)

