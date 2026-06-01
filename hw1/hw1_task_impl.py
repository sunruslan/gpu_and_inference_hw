import torch
import numpy as np

# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # TODO (1 line): implement a lowest-AI op
    return torch.clone(x)


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros_like(x)
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    # TODO (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # TODO: time `rep` runs using CUDA events and return median latency (ms)
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times)


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # Each `acc = acc * x + x` iteration: one multiply + one add = 2 FLOPs/element.
    flops_per_element = 2 * num_ops
    total_flops = num_elements * flops_per_element

    if variant == "compiled":
        # Fused kernel: one read of x and one write of the result at the boundary.
        total_bytes = num_elements * 2 * bytes_per_element
    elif variant == "eager":
        # Separate mul/add kernels per iteration; each touches 3 tensors (2 reads + 1 write).
        bytes_per_iter = 6 * num_elements * bytes_per_element
        init_write = num_elements * bytes_per_element  # acc = zeros_like(x)
        total_bytes = init_write + num_ops * bytes_per_iter
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1. On our L40S run, compiled latency stayed ~0.82–0.83 ms from 1 through 64 ops
# while achieved throughput rose from ~0.16 to ~10.4 TFLOP/s. torch.compile fuses the
# loop so traffic at the kernel boundary is still one read and one write of the 64M
# float32 vector (~256 MB) regardless of num_ops; only the FLOP count grows (2K per
# element). Arithmetic intensity therefore rises with K, but bytes moved do not. We
# are still on the memory-bandwidth (left) side of the roofline (ridge ≈106 FLOP/Byte;
# AI at 64 ops is only 16), so time is set mainly by bytes/BW while reported FLOP/s
# increases as AI increases (achievable FLOP/s ≈ BW × AI on that slope).
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2. Our run was close (matmul 1024 ≈22.6 TFLOP/s vs 128-op compiled ≈20.9 TFLOP/s),
# but a small GEMM can still underperform a large fused elementwise kernel because:
# (1) a 1024³ problem is too small to fill all SMs/warps on a big GPU, so compute
# units sit idle; (2) cuBLAS/launch and workspace overhead matter more relative to
# useful work, whereas the compiled elementwise kernel sustains a long, regular
# memory-bound sweep over 256 MB. A huge H100 is built for large tiles; tiny matmuls
# do not always win on FLOP/s even when their nominal AI is higher.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3. For compiled ops on our L40S trace, ms was still ~flat from 64 to 128 ops
# (~0.823 vs ~0.822 ms) while TFLOP/s doubled again (~10.4 → ~20.9), so we were
# still memory-bound (AI 32 ≪ ridge 106). In general, when extra ops finally make
# latency climb after staying flat at low K, the kernel is moving toward the compute
# ceiling: FLOPs are no longer “free” on top of fixed HBM traffic and FP32 throughput
# starts to limit step time. (The eager series shows the opposite pattern—ms grows
# roughly linearly with K because each step re-reads/writes the growing sequence.)
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4. Eager PyTorch runs separate mul/add kernels each loop iteration, materializes
# intermediates, and (in our model) keeps ~0.08 FLOP/Byte and ~0.05 TFLOP/s while
# latency grows from ~2.9 ms (1 op) to ~164 ms (64 ops). Compiled code fuses the loop
# into one kernel with one read/write per element, so AI scales with K (0.25–32 in
# our run), latency stays ~0.82 ms, and points move right/up on the roofline. Same math,
# different bytes and launch pattern.
