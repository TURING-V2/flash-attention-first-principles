# Flash Attention v1: CUDA Implementation from First Principles

A from-scratch implementation of Flash Attention v1 algorithm in CUDA, following the [original paper](https://arxiv.org/pdf/2205.14135) by Dao et al. This project demonstrates the core algorithmic innovations behind Flash Attention through direct CUDA kernel implementation.

## Overview

Flash Attention is a groundbreaking algorithm that makes attention computation faster and more memory-efficient by being **IO-aware**. Instead of materializing the full attention matrix in GPU memory (which grows quadratically with sequence length), it uses clever tiling and online softmax computation to achieve **O(N) memory complexity** instead of **O(N^2)**.

### Key Results from Testing

**Tested on:**
- NVIDIA Tesla T4 (16GB VRAM) - Google Colab
- NVIDIA GTX 1650 Ti (4GB VRAM) - Local machine

**Memory Efficiency:** Spectacular Success
- At seq_len=8192: **52x less memory** than standard attention (2190MB → 42MB)
- Memory scaling follows theoretical O(N) vs O(N^2) prediction
- Enables processing sequences that would OOM with standard attention

**Compute Performance:** Learning Experience
- Custom implementation is ~6x slower than PyTorch's optimized attention
- This highlights the engineering complexity of production implementations
- Memory efficiency ≠ compute efficiency without extensive optimization

## Deep Dive: The Algorithm

### The Core Problem

Standard attention computation:
```
S = Q @ K^T                    # [batch, heads, seq_len, seq_len] - O(N^2) memory!
P = softmax(S)                 # Requires materializing full attention matrix
O = P @ V                      # Final output
```

For a sequence of length 4096 with 8 heads:
- Attention matrix: **536 MB per sample**
- With batch size 32: **17 GB just for attention scores**

### Flash Attention's Solution: Three Key Ideas

#### 1. **Tiling Strategy** - Keep Data in Fast Memory

Instead of computing the full attention matrix, we:
- Split Q into blocks of size `Br` (e.g., 16 rows)
- Split K, V into blocks of size `Bc` (e.g., 16 rows)  
- Process one Q block at a time, streaming through all K,V blocks
- Keep all intermediate results in SRAM (fast memory), not HBM (slow memory)

```cuda
//each thread block processes Br queries
for q_block in Q_blocks:
    load Q_block into SRAM  //small 16×64 elements
    
    for kv_block in KV_blocks:
        load K_block, V_block into SRAM
        compute attention for this Q-KV block pair
        update running statistics
```

**Why this matters:**
- SRAM access: ~1 cycle latency
- HBM access: ~hundreds of cycles latency
- Keeping data in SRAM = massive speedup potential

#### 2. **Online Softmax** - One-Pass Computation

The clever trick: You don't need to see all attention scores before computing softmax, refer below youtube explanation of matrix chain multiplication. 

Instead of:
```python
# standard two-pass
scores = Q @ K.T           # pass 1: compute all scores
probs = softmax(scores)    # pass 2: normalize
output = probs @ V         # pass 3: compute output
```

Flash Attention does:
```python
# online single-pass
m = -∞                     # running max
l = 0                      # running sum of exponentials
O = 0                      # running output

for each KV block:
    scores = Q_block @ K_block.T
    m_new = max(m, max(scores))
    
    # rescale previous results
    scale = exp(m - m_new)
    l = l * scale
    O = O * scale
    
    # add new contribution
    exp_scores = exp(scores - m_new)
    l = l + sum(exp_scores)
    O = O + exp_scores @ V_block
    
    m = m_new

O = O / l  # final normalization
```

**The math:** When we encounter a new max value, we rescale everything:
```
O_new = O_old × exp(m_old - m_new) + P_tilde × V_new
```

This ensures correctness while only making one pass through the data.

#### 3. **IO Complexity Analysis**

Let N = sequence length, d = head dimension, M = SRAM size

**Standard Attention:**
- Memory reads/writes: O(Nd + N^2)
- The N^2 term dominates for long sequences
- Must read/write full attention matrix to HBM

**Flash Attention:**
- Memory reads/writes: O(N^2d^2/M)
- With typical values (d=64, M=100KB): **25-40x fewer HBM accesses**
- Only reads/writes final output to HBM

### Implementation Details

My Implementation provides two kernel variants:

#### Simple Kernel (`flash_attention_v1_kernel`)
- One thread per query row
- Straightforward implementation following algorithm directly
- Good for understanding the core algorithm
- Block sizes: 16×16 for Q and KV

#### Optimized Kernel (`flash_attention_v1_vectorized`)  
- Warp-level parallelism for better GPU utilization
- Vectorized memory access patterns
- Uses `__shfl_xor_sync` for efficient reductions
- Processes 4 elements per thread with explicit unrolling

**Note:** The warp-level optimizations (warp reductions, vectorized access) are not specified in the v1 paper they are additional GPU-specific optimizations attempted beyond the algorithm description. The v1 paper focuses on the tiling strategy and online softmax algorithm at a higher level of abstraction, leaving specific CUDA implementation details to the implementer.

Key CUDA optimizations attempted:
- Shared memory for Q, K, V blocks
- Warp-level primitives for reductions (general CUDA best practices)
- Coalesced memory access patterns
- Float4 vectorization where possible

## Running the Code

### Requirements
```bash
pip install torch ninja matplotlib numpy
```

**GPU Required:** CUDA-capable NVIDIA GPU (tested on Turing arch)

### Quick Start

```python
from flash_attention import load_kernel, FlashAttention

# compile and load CUDA kernels (takes ~30-60 seconds)
cuda_module = load_kernel()

# create Flash Attention module
flash_attn = FlashAttention(cuda_module, use_optimized=True)

# use it like standard attention
Q = torch.randn(1, 8, 512, 64, device='cuda')  # [batch, heads, seq, dim]
K = torch.randn(1, 8, 512, 64, device='cuda')
V = torch.randn(1, 8, 512, 64, device='cuda')

output = flash_attn(Q, K, V)  # same shape as Q
```

### Run Full Benchmark Suite

```bash
# in notebook or Python script
python flash_attention_experiments.py
```

This will:
- Test correctness against PyTorch's standard attention
- Benchmark across 12 different configurations (128 to 8192 sequence length)
- Generate comprehensive performance plots
- Show memory scaling analysis

### Example Benchmark Output

```
Configuration    Seq Len    Standard      Flash Opt     Memory Saved
─────────────────────────────────────────────────────────────────────
large            1024       1.10ms        7.66ms        74.3%
xlarge           2048       9.35ms        57.84ms       88.9%  
xxlarge          3072       22.73ms       129.91ms      92.6%
xxxlarge         4096       29.82ms       175.47ms      94.3%
massive          6144       46.15ms       261.10ms      96.2%
```

## Results Analysis

### What Works Exceptionally Well

1. **Memory Efficiency**
   - Theoretical O(N) scaling confirmed experimentally
   - At seq=8192: 2190MB → 42MB (52x reduction)
   - Enables sequences that would otherwise OOM

2. **Correctness**
   - Max difference from PyTorch: < 1e-6
   - Mean difference: < 5e-8  
   - Numerically stable across all tested configurations

3. **Scaling Behavior**
   - Memory grows linearly with sequence length (as predicted)
   - Standard attention grows quadratically (as expected)

### What Reveals Optimization Complexity

1. **Raw Compute Speed**
   - My implementation: 6-8x slower than PyTorch's standard attention
   - PyTorch SDPA: 100-200x faster than my implementation
   - This gap shows the difference between "algorithm correct" and "production optimized"

2. **Why the Speed Gap?**
   - PyTorch uses highly optimized BLAS kernels (cuBLAS)
   - Missing optimizations: tensor cores, better occupancy, memory coalescing
   - Softmax and reduction operations not fully optimized
   - Loop overhead and suboptimal block sizes

3. **Key Takeaway**
   - This implementation is **educational**, not production-ready
   - Official `flash-attention` library uses Triton + extensive CUDA optimization
   - Shows why systems engineering matters as much as algorithms

## Key Learnings

### From Implementation Perspective

1. **Online Softmax is Tricky**
   - Numerical stability requires careful max tracking
   - Rescaling logic must be exact or accuracy degrades
   - The `exp(m_old - m_new)` term is the key to correctness

2. **Memory Hierarchy Matters**
   - SRAM vs HBM access patterns dominate performance
   - Block size choices impact everything
   - Shared memory size is a hard constraint (48-96KB typically)

3. **CUDA Optimization is Deep**
   - Warp-level primitives help but aren't magic
   - Memory coalescing requires careful data layout
   - Occupancy vs shared memory usage trade-offs are subtle

### From Algorithm Perspective

1. **IO-Aware Algorithms are Different**
   - Flash Attention isn't about reducing FLOPs (it does the same math)
   - It's about reducing expensive memory movements
   - This mindset applies to many other GPU algorithms

2. **Tiling + Online Computation is Powerful**
   - Pattern applies beyond attention (e.g., online softmax, layer norm)
   - Enables "impossible" computations by rethinking memory access

3. **Theoretical Analysis Matches Reality**
   - O(N) vs O(N^2) memory prediction holds in practice
   - IO complexity analysis (O(N^2d^2/M)) explains the speedup potential

## Code Structure

```
flash_attention_v1.ipynb
├── Cell 1: Environment setup (CUDA, PyTorch)
├── Cell 2: CUDA kernels (flash_attention_kernel.cu)
│   ├── flash_attention_v1_kernel: Simple implementation
│   └── flash_attention_v1_vectorized: Optimized version
├── Cell 3: C++ bindings (flash_attention_kernel.cpp)
├── Cell 4: Python wrapper (flash_attention.py)
│   ├── FlashAttention class
│   └── benchmark() function
├── Cell 5-7: Run benchmarks and tests
└── Cell 8: Comprehensive experiments (flash_attention_experiments.py)
    ├── Multi-scale benchmarking
    └── Visualization generation
```

## Understanding the Math

### Attention Mechanism Recap

Given queries Q, keys K, values V (all of shape [batch, heads, seq_len, head_dim]):

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

Where:
- Q K^T produces attention scores [batch, heads, seq_len, seq_len]
- softmax normalizes scores per query
- Matrix multiply with V produces output [batch, heads, seq_len, head_dim]

### Online Softmax Derivation

The key insight enabling single-pass computation:

Given a sequence of elements x_1, x_2, ..., x_n, we want:
```
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

**Traditional:** Requires two passes (find max, then normalize)

**Online:** Maintain running statistics:
- m: current maximum
- l: sum of exponentials

When we see a new block with maximum m_new:
1. Update maximum: `m ← max(m, m_new)`  
2. Rescale previous sum: `l ← l × exp(m_old - m_new)`
3. Add new contributions: `l ← l + Σ exp(x_i - m_new)`

This ensures the final result is identical to the two-pass version, but we only need one pass through the data!

### Memory Complexity Proof Sketch

**Standard Attention:**
- Must store S = QK^T which is [N × N]
- Memory: O(N^2)

**Flash Attention:**
- Stores only:
  - Q blocks in SRAM: O(Br × d)
  - KV blocks in SRAM: O(Bc × d)
  - Final output O: O(N × d)
- Total HBM memory: O(N × d) = O(N)

Where Br, Bc, d are constants, so only O scaling matters.

## Educational Value
This implementation is ideal for:
1. Studying Flash Attention with annotated, from‑scratch CUDA code.
2. Practicing CUDA patterns: shared memory, warp primitives, coalescing, and profiling.
3. Observing production trade‑offs: correctness vs. heavy optimization and why specialized libraries exist.

## References & Resources

### Original Paper
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) - Dao et al., 2022

### Official Implementation  
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - Production-ready Triton/CUDA implementation

### Educational Resources
- [Aleksa Gordić - Flash Attention Explained](https://youtu.be/gBMO1JZav44) - Excellent video walkthrough
- [Running CUDA on Google Colab](https://medium.com/@zubair09/running-cuda-on-google-colab-d8992b12f767)
- [CUDA Programming Tutorials](https://www.youtube.com/watch?v=LKwyHWYEIMQ)
- [GPU Memory Hierarchy](https://www.youtube.com/watch?v=N1EZpa7lZc8)
- [Optimizing CUDA Kernels](https://www.youtube.com/watch?v=eMlx5fFNoYc)

## Limitations & Future Work

### Current Limitations
- Forward pass only (no backward/gradient computation)
- Float32 only (no FP16/BF16 support)
- Max head dimension: 128
- No causal masking support
- Performance gap vs production implementations

### Potential Improvements
- Implement backward pass for training
- Add FP16/BF16 kernels using CUDA cores or Tensor Cores
- Support causal attention masks
- Better block size tuning
- More aggressive loop unrolling and optimization
- Profile-guided optimization using Nsight Compute

### Production Considerations
For real-world usage, please use:
- [Official flash-attention](https://github.com/Dao-AILab/flash-attention) - Highly optimized
- PyTorch's `scaled_dot_product_attention` - Built-in and fast
- This implementation is for **learning purposes only**
