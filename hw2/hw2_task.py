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
        # Required when model is torch.compile'd with CUDA graphs + KV cache:
        # each decode step reuses past_key_values without overwriting the graph.
        torch.compiler.cudagraph_mark_step_begin()
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


# V3/V4 use the same loop; model dtype / compile differ in generate_*.
v3_loop = v2_loop
v4_loop = v2_loop


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


def _warmup_loop(loop_fn, model, input_ids, n_steps=2):
    loop_fn(model, input_ids, n_steps)
    torch.cuda.synchronize()


def _time_version(label, loop_fn, dtype, compile_model=False):
    model = build_model(dtype)
    input_ids = get_input_ids()
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
        _warmup_loop(loop_fn, model, input_ids)
    elapsed = time_generation(loop_fn, model, input_ids, label)
    del model
    torch.cuda.empty_cache()
    return elapsed


def generate_v0(trace_name: str | None = "v0_trace.json") -> float:
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    if trace_name:
        profile(v0_loop, model, input_ids, trace_name)
    elapsed = time_generation(v0_loop, model, input_ids, "V0")
    del model
    torch.cuda.empty_cache()
    return elapsed


def generate_v1(trace_name: str | None = None) -> float:
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    if trace_name:
        profile(v1_loop, model, input_ids, trace_name)
    elapsed = time_generation(v1_loop, model, input_ids, "V1 (+KV cache)")
    del model
    torch.cuda.empty_cache()
    return elapsed


def generate_v2(trace_name: str | None = None) -> float:
    return _time_version("V2 (+no .item())", v2_loop, torch.float32)


def generate_v3(trace_name: str | None = None) -> float:
    return _time_version("V3 (+bf16)", v3_loop, torch.bfloat16)


def generate_v4(trace_name: str | None = "v4_trace.json") -> float:
    model = build_model(torch.bfloat16)
    model = torch.compile(model, mode="reduce-overhead")
    input_ids = get_input_ids()
    _warmup_loop(v4_loop, model, input_ids)
    if trace_name:
        profile(v4_loop, model, input_ids, trace_name)
    elapsed = time_generation(v4_loop, model, input_ids, "V4 (+compile)")
    del model
    torch.cuda.empty_cache()
    return elapsed


def generate_optimized(optimized_trace_name: str = "v4_trace.json") -> float:
    """Final optimized path (V4) for grading speedup vs slow baseline."""
    return generate_v4(optimized_trace_name)


# Aliases for homework / imports
optimized_loop = v4_loop


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
        ("V3 (+bf16)", timings["V3"]),
        ("V4 (+torch.compile)", timings["V4"]),
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

    v4 = timings["V4"]
    print("-" * 60)
    if v4 > 0:
        print(f"  Total speedup (Slow → V4): {slow_elapsed / v4:.2f}x")


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Progressive versions (timing only) ---")
    timings = {
        "V0": generate_v0(trace_name=None),
        "V1": generate_v1(),
        "V2": generate_v2(),
        "V3": generate_v3(),
    }

    print("\n--- Part 3: V4 (+compile) timing and Chrome trace ---")
    optimized_elapsed = generate_v4(trace_name="v4_trace.json")
    timings["V4"] = optimized_elapsed

    _print_progressive_summary(slow_elapsed, timings)

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
#   V3 (+bf16): ...
#   V4 (+torch.compile): ...
#
# Biggest impact and why:
#
