from std.time import perf_counter_ns
from std.random import random_float64
from std.sys import simd_width_of
from std.algorithm import parallelize

def relu_val(x: Float32) -> Float32:
    if x > 0:
        return x
    return 0.0

def matmul_parallel(A: List[Float32], B: List[Float32],
                    m: Int, n: Int, k: Int) -> List[Float32]:
    var C = List[Float32](capacity=m*k)
    for _ in range(m*k):
        C.append(0.0)
    comptime simd_width = simd_width_of[DType.float32]()
    var A_ptr = A.unsafe_ptr()
    var B_ptr = B.unsafe_ptr()
    var C_ptr = C.unsafe_ptr()
    @parameter
    def compute_row(i: Int):
        for l in range(n):
            var a_val = A_ptr[i*n + l]
            var j = 0
            while j + simd_width <= k:
                var b_vec = (B_ptr + l*k + j).load[width=simd_width]()
                var c_vec = (C_ptr + i*k + j).load[width=simd_width]()
                c_vec = c_vec + a_val * b_vec
                (C_ptr + i*k + j).store(c_vec)
                j += simd_width
            while j < k:
                C_ptr[i*k + j] += a_val * B_ptr[l*k + j]
                j += 1
    parallelize[compute_row](m)
    return C^

def main():
    print("Mojo SIMD + Parallel (Float32) neural network forward pass benchmark")
    var batch_size = 10000
    var input_size = 784
    var hidden_size = 128
    var output_size = 10
    var X  = List[Float32](capacity=batch_size*input_size)
    var W1 = List[Float32](capacity=input_size*hidden_size)
    var W2 = List[Float32](capacity=hidden_size*output_size)
    for _ in range(batch_size*input_size):
        X.append(random_float64(0.0, 1.0).cast[DType.float32]())
    for _ in range(input_size*hidden_size):
        W1.append(random_float64(-0.01, 0.01).cast[DType.float32]())
    for _ in range(hidden_size*output_size):
        W2.append(random_float64(-0.01, 0.01).cast[DType.float32]())
    print("Starting benchmark...")
    var runs = 100
    var start = perf_counter_ns()
    for _ in range(runs):
        var z1 = matmul_parallel(X, W1, batch_size, input_size, hidden_size)
        for i in range(len(z1)):
            z1[i] = relu_val(z1[i])
        var _ = matmul_parallel(z1, W2, batch_size, hidden_size, output_size)
    var end = perf_counter_ns()
    var avg_ms = (end - start) / UInt(runs) / UInt(1_000_000)
    print("Avg forward pass time over", runs, "runs:", avg_ms, "ms")