import torch

# Enable TF32 matmul on Ampere+ GPUs (silences transformers warning for fp32 paths).
torch.set_float32_matmul_precision("high")

from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Generation loops (progressive optimizations)
# ---------------------------------------------------------------------------


def v0_loop(model, input_ids, n_steps):
    """Same as the original unoptimized loop (full re-forward, .item(), cat)."""
    generated_ids = input_ids.clone()
    generated_tokens = []
    for _ in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_value = next_token_id.item()
        generated_tokens.append(token_value)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens


def v1_loop(model, input_ids, n_steps):
    """V1: KV cache — prefill once, then decode one token per step."""
    generated_ids = input_ids.clone()
    generated_tokens = []
    past_key_values = None
    for _ in range(n_steps):
        if past_key_values is None:
            model_input = generated_ids
        else:
            model_input = generated_ids[:, -1:]
        outputs = model(
            input_ids=model_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_value = next_token_id.item()
        generated_tokens.append(token_value)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens


@torch.inference_mode()
def v2_loop(model, input_ids, n_steps):
    """V2: V1 + no per-step .item() / cat (GPU token buffer)."""
    past_key_values = None
    cur = input_ids
    out = torch.empty(n_steps, device=input_ids.device, dtype=torch.long)
    for i in range(n_steps):
        outputs = model(
            input_ids=cur,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out[i] = next_token.view(-1)
        cur = next_token
    return out.cpu().tolist()


# V3 uses the same loop as V2; torch.compile is applied in generate_v3.
v3_loop = v2_loop


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to {trace_path}")


# ---------------------------------------------------------------------------
# Model setup + per-version runners
# ---------------------------------------------------------------------------


def _compile_for_generation(model):
    """torch.compile for decode loops with KV cache.

    Avoid CUDA graph capture: HuggingFace past_key_values are updated in-place
    each step, which conflicts with reduce-overhead / cudagraph_trees replay.
    """
    import torch._inductor.config as inductor_config

    inductor_config.triton.cudagraph_trees = False
    # PyTorch 2.9+: cannot pass mode and options together; use a no-cudagraphs mode.
    for mode in ("max-autotune-no-cudagraphs", "default"):
        try:
            return torch.compile(model, mode=mode, fullgraph=False)
        except RuntimeError:
            continue
    return torch.compile(model, fullgraph=False)


def _warmup_loop(loop_fn, model, input_ids, n_steps=2):
    loop_fn(model, input_ids, n_steps)
    torch.cuda.synchronize()


def _run_version(
    label: str,
    loop_fn,
    trace_name: str,
    dtype=torch.float32,
    compile_model: bool = False,
) -> float:
    """Profile (Chrome trace) then time a version for grading."""
    model = build_model(dtype)
    input_ids = get_input_ids()
    if compile_model:
        model = _compile_for_generation(model)
        _warmup_loop(loop_fn, model, input_ids)
    profile(loop_fn, model, input_ids, trace_name)
    elapsed = time_generation(loop_fn, model, input_ids, label)
    del model
    torch.cuda.empty_cache()
    return elapsed


def generate_v0() -> float:
    return _run_version("V0", v0_loop, "v0_trace.json")


def generate_v1() -> float:
    return _run_version("V1 (+KV cache)", v1_loop, "v1_trace.json")


def generate_v2() -> float:
    return _run_version("V2 (+no .item())", v2_loop, "v2_trace.json")


def generate_v3() -> float:
    return _run_version(
        "V3 (+compile)",
        v3_loop,
        "v3_trace.json",
        dtype=torch.bfloat16,
        compile_model=True,
    )


def generate_optimized() -> float:
    """Final optimized path (V3) for grading speedup vs slow baseline."""
    return generate_v3()


# Aliases for homework / imports
optimized_loop = v3_loop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_progressive_summary(slow_elapsed: float, timings: dict[str, float]):
    print("\n" + "=" * 60)
    print("PROGRESSIVE OPTIMIZATIONS")
    print("=" * 60)
    print(f"{'Version':<28} {'Time (s)':>10} {'vs Slow':>10} {'vs Prev':>10}")
    print("-" * 60)

    order = [
        ("Slow (utils baseline)", slow_elapsed),
        ("V0 (same loop as slow)", timings["V0"]),
        ("V1 (+KV cache)", timings["V1"]),
        ("V2 (+no .item())", timings["V2"]),
        ("V3 (+torch.compile)", timings["V3"]),
    ]

    prev_elapsed = None
    for label, elapsed in order:
        vs_slow = slow_elapsed / elapsed if elapsed > 0 else float("inf")
        if prev_elapsed is None:
            vs_prev = "-"
        else:
            vs_prev = f"{prev_elapsed / elapsed:.2f}x" if elapsed > 0 else "inf"
        print(f"{label:<28} {elapsed:10.2f} {vs_slow:9.2f}x {vs_prev:>10}")
        prev_elapsed = elapsed

    v3 = timings["V3"]
    print("-" * 60)
    if v3 > 0:
        print(f"  Total speedup (Slow → V3): {slow_elapsed / v3:.2f}x")


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Progressive versions (profile + timing) ---")
    timings = {
        "V0": generate_v0(),
        "V1": generate_v1(),
        "V2": generate_v2(),
        "V3": generate_v3(),
    }
    optimized_elapsed = timings["V3"]

    _print_progressive_summary(slow_elapsed, timings)

    print("\nChrome traces saved to results/ (open at https://ui.perfetto.dev):")
    for trace in (
        "slow_trace.json",
        "v0_trace.json",
        "v1_trace.json",
        "v2_trace.json",
        "v3_trace.json",
    ):
        print(f"  {RESULTS_DIR / trace}")

    print("\n" + "=" * 60)
    print("SUMMARY (grading)")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print(
            "generate_optimized() did not return a positive elapsed time; "
            "cannot compute speedup."
        )
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#   V1 (+KV cache): ...
#   V2 (+no .item()): ...
#   V3 (+torch.compile, bf16): ...
#
# Biggest impact and why:
#
