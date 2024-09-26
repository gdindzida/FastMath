# FastMath
Experimenting with optimization

## Theoretical performance of my CPU

My CPU info:
| CurrentClockSpeed | MaxClockSpeed | Name | NumberOfCores |
| ----------------- | ------------- | ---- | ------------- |
| 2208              |          2208 | Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz |  6 |

Detailed info can be found [here](https://ark.intel.com/content/www/us/en/ark/products/134906/intel-core-i7-8750h-processor-9m-cache-up-to-4-10-ghz.html).

6 cores x 2.2GHz = 13.2GHz =  $13.2 \times 10^{9}$ cycles per second.

If each cycle we can do 8 operations (as per extended operation set) then we have 105.6 GFLOP/s.

Max memory bandwith = 41.8 GB/s ~ 10.45 GFLOP/s (float takes 4 bytes).

CPU speed / Max memory bandwith = 105.6 / 10.45 ~ 10. 

This means memory fetching is 10 times slower.

If task requires less than 10 cycles per memory access then it is MEMORY BOUNDED. Otherwise it is COMPUTE BOUNDED.

## Memory bounded theoretical example

If problem is memory bounded caching won't help. If we take a look at an example where we have two vectors $x$ and $y$ of size $n$. Lets say we want to calculate following expression: $y = ax + y$ where $a$ is some scalar. In order to do calculate this expression we need to do $2n$ floating point operations ($n$ multiplications and $n$ additions). But we also need to fetch $2n$ floating point numbers from memory into cache. Which shows us that this algorithm is memory bounded and we lose optimal CPU speed.

So if we want for this algorithm to perform faster we need to increase memory bandwith. If we transfer this task to GPU we will get only performance increase of memory bandwith which is usually bigger. But more on that later.

## Experiments

- [General Matrix-Matrix Multiplication](src/experiments/experiment_1/README.md)
