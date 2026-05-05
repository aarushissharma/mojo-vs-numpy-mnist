# Mojo vs NumPy: MNIST Neural Networl Benchmark

A from-scratch nural network trained on MNSIT, implemented in both Python and Mojo, with a benchmarked comparison of forward pass performance.

## motivation/backrgound
Created as a 'deconstruction' experiment. 

After learning about Modular's work, I wanted to explore Mojo's performance in ML workloads, especially with its lower-level hardware emphasis.

I rebuilt forward pass in Mojo to see what was actaully happening behind-the-scenes inside NumPy's black box. Using SIMD vectorization and parallelization, I tried to understand where the performance comes from.

## project structure

## network architecture
- input layer: 784 neurons (28x28 flattened MNIST images)
- hidden layer: 128 neurons (ReLU activation)
- output layer: 10 neurons (Softmax activation)
- trained from scratch using only NumPy — no ML frameworks
- test accuracy: ~97%

## results

| Implementation              | Avg Forward Pass (10k samples, 100 runs) | Speedup vs naive |
|-----------------------------|------------------------------------------|------------------|
| Mojo naive                  | 803 ms                                   | baseline         |
| Mojo SIMD                   | 138 ms                                   | 5.8x             |
| Mojo SIMD + Parallel        | 30  ms                                   | 26x              |
| Mojo SIMD + Parallel Float32| 13 ms                                    | 62x              |
| NumPy (Python)              | 6.34 ms                                  | reference        |

**analysis:** each optimization step produced a meaningful & explainable speedup
- SIMD gave 5.8x, parallelization added another 4.6x, and switching from Float64 to Float32 doubled it
- remaining gap with numpy is likely due to BLAS (basic linear algebra subprograms) optimization rather than language limitation

## what i learned


## tech stack
- python3/numpy
- mojo 1.0.0b2
- mnist dataset via tensorflow
- benchmarked on apple macbook pro (apple silicon)


## Read More
*medium post coming soon*