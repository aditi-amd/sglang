# MiniMax-M3 optimization campaign: planning doc

Branch `M3-opt-0929` (from `M3-perf-rebase` `276976ba11`). Box: 8x MI350X VF (see `/scratch/m3/ENV_HANDOFF.md` for the
proot environment and the hostcall workarounds every server here needs). This file is the plan of record; the status tracker at the
bottom is updated as work lands.

---

## Part A: the request (verbatim)

# MiniMax-M3: SGLang Optimization Handoff

Hi team — could you review and integrate the two MiniMax-M3 optimization PRs below, and take ownership of developing the two additional serving optimizations described here?

## 1. Review and integrate the existing PRs

| PR | Optimization | Review focus |
|---|---|---|
| [#41488 — Indexer-only decode CP](https://github.com/sgl-project/sglang/pull/41488) | Partitions indexer context-block reads across TP ranks and exchanges compact top-k candidates. | Correctness, communication overhead, and the batch/context crossover where CP becomes beneficial. |
| [#41397 — FlyDSL paged attention + GPU work planner](https://github.com/sgl-project/sglang/pull/41397) | Integrates AITER FlyDSL paged attention and GPU planning for dense, uneven-context graph decode. Sparse calls retain static partitioning. | Numerical qualification, graph integration, and end-to-end serving performance. |

Confirm each PR's current EAGLE3 support and implement any missing compatibility before qualifying the combined **TP4 + EAGLE3** serving configuration.

Below are the two optimizatison we suggest:

## 2. Develop graph capture tuning for actual execution batches

Profile the running **decode and EAGLE3 verification batches**, then tune the capture-size list around frequently observed shapes.

- Record live request count, verification tokens per request, selected graph size, and replay frequency.
- Add capture sizes where frequent padding causes measurable overhead.
- Measure padding using the same row definition for live execution and graph capture:

  ```text
  padding = (captured rows - live rows) / captured rows
  ```

- Track capture time, graph memory usage, and remaining GPU KV-cache capacity. Additional graphs must justify any reduction in cache capacity.
- Preserve the existing fallback for unsupported shapes.

**Goal:** reduce decode and verification latency without reducing cache capacity enough to offset the gain.

## 3. Develop CPU-backed prefix caching for M3

Evaluate and extend **HiCache or an EAGLE3-compatible LMCache integration** when the reusable prefix working set exceeds GPU capacity.

- **Complete state:** offload and restore both **main K/V and index-K**, preserving consistent token/page mappings.
- **Cross-rank consistency:** reuse only the **contiguous prefix available across every required TP rank and cache component**.
- **Retention:** compare **LRU and SLRU** under the same CPU-memory budget, explicitly identifying which policy governs each tier.
- **Buffer lifetime:** protect source and destination buffers during transfers. Release references after completion and handle misses, cancellation, and errors correctly.
- **Transfer efficiency:** measure CPU-to-GPU restore time and NUMA placement against the prefill computation avoided.

**Goal:** reduce prefix recomputation and sustain higher concurrency while maintaining interactivity.

## 4. Validate each change independently, then combine

Run matched A/B tests at concurrency **24, 32, 40, and 48**, using identical agentic traces, hardware, model settings, and **real EAGLE3 acceptance**.

| Experiment | Primary measurements |
|---|---|
| Indexer CP off vs on | Indexer-chain latency including communication, selection correctness, batch/context crossover, serving throughput and interactivity |
| Existing attention vs FlyDSL + planner | Attention latency, numerical correctness, graph-replay behavior, serving throughput and interactivity |
| Default vs tuned graph sizes | Padding, decode/verify latency, capture time, graph memory, available GPU KV capacity |
| CPU tier off vs on | Recomputed prompt tokens, restored tokens, transfer time, cache hit rates |
| LRU vs SLRU | Prefix retention, recomputation, eviction behavior at equal CPU-memory budgets |
| Combined configuration | Output throughput, p90 interactivity, TTFT, correctness |

Keep concurrency fixed when measuring the benefit of CPU offload. Increasing concurrency while enabling offload is a separate operating-point comparison and does not isolate the offload benefit.

**Requested outcome:** integration of the two existing PRs, development of the two additional optimizations by the RadixArk / SGLang team, and a validated serving configuration with reproducible benchmark results.

---

## Part B: execution plan

### Fixed setup (all experiments)

- Model `amd/MiniMax-M3-MXFP4` + EAGLE3 `Inferact/MiniMax-M3-EAGLE3-GQA` (3 steps / 4 draft tokens, **real acceptance**), TP4, the
  `reproduce.sh real` server config (`rebase_0923/serve_m3.sh`) plus the node's hostcall knobs.
- Workload: the AgentX agentic traces (`inferencex-agentx-mvp`, SemiAnalysis AIPerf fork at InferenceX's pinned commit, seed 42), as in
  `OPTIMIZATIONS.md`. Concurrency 24 / 32 / 40 / 48. Same traces, same seed, same GPUs class for every A/B.
- Matched A/B: the box has 8 GPUs, so A and B run as two TP4 servers side by side (GPUs 0-3 and 4-7), then swap sides for a second
  pass to cancel any per-GPU difference. Where a change needs the whole box, run A then B on the same GPUs.
- `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ`: report every number with the value used. The AgentX configs use 4 (corrupts long-context tool
  calls, `endpoint/ENDPOINT.md`); the production endpoint uses 1. Primary A/Bs at 1; note if an optimization interacts with 4.
- Quality gate for any kept config: GSM8K-500 5-shot in 0.85-0.89, plus the long-context tool-call check if attention/indexer changes.

### Work items, in order

1. **Benchmark harness.** Set up the AIPerf AgentX client (from `reproduce.sh bench`) under `/scratch/m3`, confirm a short c=24 run
   reproduces the order of magnitude in `OPTIMIZATIONS.md`, and add a metrics sampler for per-step shapes where needed.
2. **PR #41397 (FlyDSL paged attention + planner).** Fetch, check its base against this branch, merge, and check EAGLE3: does the
   verify/draft-extend path (multi-token per request) go through the new dense decode kernel, or fall back? Implement what is missing.
   Qualify numerics (kernel-level vs the existing Triton path, then GSM8K) and graph replay, then A/B at c=24-48.
3. **PR #41488 (indexer-only decode CP).** Same flow. Measure the indexer chain with communication and the batch/context crossover
   (context length x batch sweep at TP4), then serving A/B.
4. **Graph capture tuning.** Instrument the decode/verify graph runner to log live requests, tokens per request, selected graph size and
   replay count; compute padding as (captured rows - live rows) / captured rows. Derive a capture list from the observed histogram,
   measure capture time, graph memory, KV capacity (`max_total_num_tokens`) and decode/verify latency, A/B against the default list.
5. **CPU-backed prefix cache.** Survey HiCache support for M3 (main K/V + index-K, fp8 pools, TP consistency) and LMCache/EAGLE3.
   Extend HiCache where state is missing, measure restore time and NUMA placement, then CPU tier off vs on at fixed concurrency and
   LRU vs SLRU at equal CPU budget.
6. **Combine** the kept changes, run the full table at c=24-48, quality gate, and record the final config and commands.

### Measurement definitions

- Throughput: AIPerf total token throughput per GPU (prompt incl. cache hits + completion) and output tok/s per GPU; interactivity: p90
  per-user output tok/s; TTFT p50/p90.
- Padding: (captured rows - live rows) / captured rows, rows = tokens entering the graph (decode: requests; verify: requests x draft tokens).
- KV capacity: `max_total_num_tokens` from the server log.

---

## Status tracker

| Item | Status | Result / notes |
|---|---|---|
| Planning doc | done | this file |
| Benchmark harness | done | AIPerf SA fork 754356e9 at `/scratch/m3/aiperf-sa-venv`; `/scratch/m3/bin/agentx.sh PORT CONC 900 OUT` (the scenario enforces >= 900 s). A/A at c=24, identical servers on GPUs 0-3 vs 4-7: 26,108 vs 26,284 total tok/s/GPU (0.7%), ITL p90 20.7 / 20.7 ms, TTFT p50 833 / 733 ms. Side-by-side A/B resolves ~1-2% throughput; TTFT medians need repeats. Baseline c=24 (TOPK_FREQ=1): 25.4-26.3K total, 208-213 out tok/s/GPU. |
| #41397 FlyDSL PA + planner | measured, integration deferred | Branch `opt/flydsl` (PR diff applies cleanly). Blockers for our config: `--attention-backend aiter`, page size 16/64/128 (ours 1), no speculation, SHUFFLE-5D main KV for all layers (breaks this branch's NHD readers and HiCache's per-token host copies). EAGLE3 is reachable: AITER `pa_decode` supports `query_length` > 1 with dense causal masking (MTP). Microbench on M3 dense-layer verify shapes (16 q heads, 1 KV head, fp8, 4 draft tokens, uneven ~170K contexts; `test/manual/minimax_m3/flydsl/bench_dense_verify.py`): FlyDSL **planned** 98 / 243 / 324 / 378 / 571 us at 8/16/24/32/48 requests vs our `_verify_mla_prefix_stage1` 149 / 416 / 531 / 676 / 882 us (**1.5-1.8x**), max diff vs fp32 ~2e-4; FlyDSL static partitions are 2x slower than ours, so the planner is the win. Dense verify is 6.5-7.6% of the steady c=32 step, so ~3% of step time. Tried the planner idea inside our Triton kernel (length-proportional splits): <= 10% on the same shapes, reverted. Next step if pursued: SHUFFLE layout for the 3 dense layers only, FlyDSL for their decode/verify. |
| #41488 indexer CP | **kept** | `opt/indexer-cp`. Fixes on top of the PR: (1) gate rejected all speculation, now chain EAGLE verify; (2) read KV heads from `ModelConfig` (VL config); (3) **packed CP scorer** for verify rows (4 draft rows x 4 heads per 16-row tile, one K read per request): 0.31-0.40x native packed time at >=24 req x >=128K, exact selected-ID parity on 30 shapes; (4) the PR only passed `indexer_cp` from ordinary decode, so **verify never used CP** until wired (serving profiles confirm `_score_shard_packed` replaces `_decode_score_kernel`). Steady c=32 baseline profile: indexer 30.8% of GPU time (`_decode_score_kernel` 26%). Serving A/B vs baseline (same box halves, TOPK_FREQ=1): c=24 25,441 -> **28,182** (+10.8%), ITL p50 13.1 -> 11.5 ms, interactivity p90 46 -> 56; c=32 32,393 -> **33,952** (+4.8%), ITL p50 17.7 -> 12.8 ms; c=40 31,571 -> **33,229** (+5.3%), ITL p50 21.1 -> 15.2 ms. TTFT p50 rises (980 -> 1,769 ms at c=32). GSM8K-500 0.862-0.872. |
| Graph capture tuning | measured | `SGLANG_GRAPH_SHAPE_STATS`. Padding over c=24-48: verify 3.8%, draft 3.8% (2.0 padded rows per verify replay); live batches mostly 3-18 requests. Greedy additions 9/11/13/15/17/33/35/37 -> 0.9%. Verify graphs cost ~1.7 s capture and ~95 MB each from the post-KV reserve (27 GB free), so no KV capacity loss. Expected gain < 1% (memory-bound step); tested inside the combined config. |
| Serving fixes (`opt/kv-indices-parallel`) | kept | (1) `create_flashinfer_kv_indices_triton` ran one program per request over ~170K tokens every step (639 us x2, 3.1% of GPU time): token-block-parallel launch. (2) **c=48 crash**: `PrefillBudget.available_chunk_tokens` returned the whole chunk when decode headroom exceeded the pool; the allocator had 104 tokens for a 4096-token continuation and the scheduler died (baseline and CP both crashed at c=48). Continuation now claims only existing tokens or waits; regression test in `test_prefill_memory_budget.py`. (3) HiCache index-K host pool lacked `can_use_write_back_jit` (M3 could not start with HiCache). |
| CPU prefix cache | running | HiCache M3 stack covers main KV + index-K + EAGLE3 draft KV host pools; reuse clamped to the shortest prefix across components. Restore verified (59,999/60,001 tokens from host after a 9.3M-token flood). Host budget: sglang counts charged page cache as used; dropping our own checkpoint pages (posix_fadvise) allows 200 GB main + 95 GB index per rank = 13.0M host tokens (1.58x the 8.25M GPU pool). Baseline recompute: 5.5% (c=24), 8.8% (c=32), 10.9% (c=40), 22% (c=48, 36.7M tokens in 920 s, more than the prefill capacity: c=48 thrashes at 13K tok/s/GPU, ITL p90 200 ms). HiCache-LRU sweep running. |
| Combined config | running | `opt/combined` = indexer CP + serving fixes + shape stats. GSM8K-500 0.860. Sweep running. |
