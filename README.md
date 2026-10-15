# Lab 9: Introduction to CUDA (Google Colab Edition)
## Name: Heshanth T
## Student ID: IT24101633
## Module: SE3082 – Parallel Computing

---

### Overview & Objectives
This repository contains the complete implementation, Google Colaboratory notebook, verification benchmarks, and laboratory report for **Lab 09: Introduction to CUDA C++**.

The practical explores general-purpose GPU computing (GPGPU) on an **NVIDIA Tesla T4 GPU**, covering:
1. GPU runtime verification via `nvidia-smi` and CUDA compiler `nvcc`
2. Host/Device memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
3. 1D thread/block execution configurations (`blockIdx.x`, `threadIdx.x`)
4. Hardware block limits (1024 maximum threads per block)
5. Large-scale vector multiplication (10,000,000 elements) with boundary guards
6. 2D grid matrix element-wise multiplication (100,000,000 elements)

---

### Directory Structure
```
SE3082-lab-09-2026/
├── SE3082_Lab09_Report_IT24101633.pdf  # Final Lab Report (PDF with SLIIT Cover Page)
├── SE3082_Lab09_IT24101633.ipynb      # Google Colab Jupyter Notebook
├── README.md                          # Repository Documentation
├── src/                               # CUDA C++ Source Code
│   ├── hello.cu                       # Exercise 1: Multi-threaded Smoke Test
│   ├── ex2_add.cu                     # Exercise 2: Scalar Addition on GPU
│   ├── ex3_blocks.cu                  # Exercise 3: 512 Blocks, 1 Thread each
│   ├── ex4_threads.cu                 # Exercise 4: 1 Block, 512 Threads
│   ├── ex5_vecmul.cu                  # Exercise 5: 10M Vector Multiplication
│   └── ex6_matrix.cu                  # Exercise 6: 100M Matrix Element Product
└── screenshots/                       # High-Resolution Execution Captures
    ├── ex1_nvidia_smi.png
    ├── ex1_compiler_arch.png
    ├── ex1_hello_smoke.png
    ├── ex2_add.png
    ├── ex3_blocks.png
    ├── ex4_threads.png
    ├── ex4_threads_limit_err.png
    ├── ex5_vecmul.png
    └── ex6_matrix.png
```

---

### Exercise Summary & Verification Results

| Exercise | Launch Configuration | Threads Launched | Elements Processed | Warp Occupancy | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ex 1: Smoke Test** | `<<<1, 4>>>` | 4 threads | N/A | 12.5% (4/32) | Threads 0–3 printed |
| **Ex 2: Scalar Add** | `<<<1, 1>>>` | 1 thread | 1 scalar ($2+7$) | 3.125% (1/32) | Result is 9 |
| **Ex 3: 512 Blocks** | `<<<512, 1>>>` | 512 threads | 512 integers | 3.125% (1/32) | **0 mismatches / 512** |
| **Ex 4: 512 Threads**| `<<<1, 512>>>` | 512 threads | 512 integers | 100.0% (32/32) | **0 mismatches / 512** |
| **Ex 4: Limit Test** | `<<<1, 2048>>>` | Failed Launch | 2048 integers | 0.0% | Hardware limit hit (>1024) |
| **Ex 5: Vector 10M** | `<<<19532, 512>>>`| 10,000,384 threads | 10,000,000 integers | 100.0% (32/32) | **0 mismatches / 10000000** |
| **Ex 6: Matrix 100M**| `<<<dim3(313,625), dim3(32,16)>>>` | 100,160,000 threads | 100,000,000 integers | 100.0% (32/32) | **0 mismatches / 100000000** |

---

### Compilation & Execution Commands
For execution in Google Colab (or any system equipped with NVIDIA CUDA Compiler):

```bash
# Exercise 1: Smoke Test
nvcc -arch=sm_75 hello.cu -o hello && ./hello

# Exercise 2: Scalar Addition
nvcc -arch=sm_75 ex2_add.cu -o ex2_add && ./ex2_add

# Exercise 3: Vector Addition (512 Blocks)
nvcc -arch=sm_75 ex3_blocks.cu -o ex3_blocks && ./ex3_blocks

# Exercise 4: Vector Addition (512 Threads)
nvcc -arch=sm_75 ex4_threads.cu -o ex4_threads && ./ex4_threads

# Exercise 5: Vector Multiplication (10M elements)
nvcc -arch=sm_75 ex5_vecmul.cu -o ex5_vecmul && ./ex5_vecmul

# Exercise 6: 2D Matrix Element Multiplication (10,000 x 10,000)
nvcc -arch=sm_75 ex6_matrix.cu -o ex6_matrix && ./ex6_matrix
```
