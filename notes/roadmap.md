# vLLM / LMCache Study Roadmap

Last updated: 2026-06-21

## Phase 0 — Environment & Foundation (mostly done)
- WSL2 + VS Code Remote setup
- `uv` virtual environment; vLLM and LMCache both installed via `-e .` (editable mode)
- CLAUDE.md established (trace-first behavior, English-primary with Chinese
  bridge sentences on request, RFC-vs-inferred distinction, note-writing conventions)
- First skill created: `trace-call-path`
- `notes/` folder set up for accumulating trace notes

Status: Done. Skill tested and working.

---

## Phase 1 — Single-GPU Inference Lifecycle (in progress)
Goal: Understand the full path of a request from entry to result, without
yet touching multi-process/IPC complexity.

Scope:
- `LLM.generate()` → request normalization → `add_request` (traced)
- Scheduler core logic: how continuous batching decides what runs each step
  (prefill vs decode mixing)
- KV cache block allocation (PagedAttention)
- Model forward pass
- Detokenization, result return

Progress so far:
- Traced: `LLM.generate()` → `_run_completion` → `_add_completion_requests`
  → `_render_and_add_requests` → `_add_request` → `LLMEngine.add_request`
  → `self.engine_core.add_request(request)`
- Found: by default, `engine_core` is a `SyncMPClient` (multiprocessing mode
  is on by default via `VLLM_ENABLE_V1_MULTIPROCESSING=1`). This means the
  scheduler does NOT run in-process — it lives in a separate background
  EngineCore process, reached over ZMQ. [inferred from code, not yet
  confirmed against an RFC]

Open question to self-check before moving on:
> Why does vLLM separate "adding requests" from "running the engine loop"
> into two phases? How does this relate to continuous batching?

> **What I understood you were asking:** Why is enqueuing a request
> (`add_request`) a separate operation from advancing the engine (`step`),
> and why does that separation matter for continuous batching?
>
> **Answer:** The split *is* the mechanism that enables continuous
> (iteration-level) batching. Three points, all `[inferred from code]`:
> 1. **Different rates / different processes.** `add_request` is called
>    once per request from the LLM process (`llm_engine.py:276`); `step()`
>    runs in a tight loop in the background EngineCore (driver at
>    `core.py:1302`, `_process_engine_step`). Enqueue ≠ run, so request B
>    can be added while request A is mid-generation.
> 2. **Decoupled mutable queue.** `add_request` only enqueues into the
>    scheduler's waiting set; it does not run the model. So a new request
>    can land between any two `step()` calls, and the next `schedule()`
>    (`core.py:490`) is free to fold it into the running batch — mixing a
>    fresh prefill with in-flight decodes. That mid-flight injection is
>    exactly what "continuous" means. If enqueue and run were fused, the
>    batch would freeze once started → static batching.
> 3. **Single decision-maker.** N producers may enqueue concurrently, but
>    exactly one loop calls `schedule()`. Admission stays cheap; the policy
>    (what runs, how the GPU is filled) is centralized — which is why the
>    multiprocess-vs-inproc transport never changes the decision logic.
>
> Caveat: term "continuous batching" / iteration-level scheduling is the
> established vLLM (Orca-style) design, but not yet confirmed here against
> a specific GitHub RFC. `[inferred from code]`

Status: At Phase 1 part 2 (scheduler core logic). Entry plumbing (part 1) done;
next is tracing `Scheduler.schedule()` in-process.

Next concrete step: trace from `LLMEngine.add_request()` (llm_engine.py:276,
already traced) forward into `EngineCore.step()` (core.py:490), which calls
`Scheduler.schedule()`. Goal: understand what schedule() actually decides
on each call — specifically the continuous batching logic for mixing
prefill and decode requests in one batch.

Anatomy of `EngineCore.step()` (core.py:478–508), one "tick" of continuous
batching, in execution order: `[inferred from code]`
- `core.py:488` — `has_requests()` early-out: nothing waiting/running → return.
- `core.py:490` — `schedule(self._should_throttle_prefills())`: PLAN the step
  (which requests, token counts, prefill/decode mix). Pure bookkeeping, no GPU.
  Note the arg `_should_throttle_prefills()` — a prefill-throttle signal feeding
  the mixing decision; not yet traced.
- `core.py:491` — `execute_model(scheduler_output, non_block=True)`: dispatch
  the forward pass, returns a Future (doesn't block).
- `core.py:492` — `get_grammar_bitmask(...)`: build guided-decoding logit mask,
  overlapped with the model running.
- `core.py:497` — `future.result()`: block for the forward pass.
- `core.py:498–499` — `sample_tokens(grammar_output)` if not already sampled.
- `core.py:503` — `_process_aborts_queue()`: handle aborts that arrived during
  execution.
- `core.py:504` — `update_from_output(scheduler_output, model_output)`: CLOSE the
  step — append new tokens, free/alloc KV blocks, detect finished requests,
  build `EngineCoreOutputs`.
- `core.py:508` — return `(outputs, total_num_scheduled_tokens > 0)`.
Symmetry to remember: `schedule()` (490) opens the step, `update_from_output()`
(504) closes it — both are Scheduler methods; `step()` is the sandwich between.
(Pipelined variant `step_with_batch_queue()` at core.py:519, default path uses
the above.)

Note: To study scheduler logic in isolation, temporarily forced
VLLM_ENABLE_V1_MULTIPROCESSING=0 (InprocClient path) to bypass ZMQ.
Verified that this exercises the same Scheduler class/logic as the
default SyncMPClient path — only the transport layer differs.
[ZMQ itself deferred to Phase 2 as originally planned.]
  - Why safe: both `InprocClient` (core_client.py:287) and `EngineCoreProc`
    (subclass, core.py:894) wrap the same `EngineCore`. Scheduler built once in
    `EngineCore.__init__` (core.py:150); `schedule()` called from inherited
    `EngineCore.step()` (core.py:490). Multiprocessing changes only transport +
    who drives `step()`, not the decision logic. Full evidence:
    `phase1-scheduler.md` → "Finding — does multiprocessing change the scheduler".
    [inferred from code]

---

## Phase 2 — Engine Internal Communication (ZMQ) (next)
Goal: Understand how the API/LLM process and the background EngineCore
process communicate.

Known entry point:
- `self.engine_core.add_request(request)` at `llm_engine.py:276` is where
  the call crosses from the main process into the EngineCore process.

To investigate:
- How the ZMQ socket is established (PUB/SUB? REQ/REP? push/pull?)
- Message serialization format (msgpack/pickle/etc.)
- Why vLLM chose a separate-process design — look for a GitHub RFC/discussion
  to confirm or refute the working theory (theory: avoid HTTP serving from
  blocking GPU scheduling)

Status: Not started.

---

## Phase 3 — Connector Abstraction Layer
Goal: Before looking at any concrete implementation (LMCache), understand
the KV connector interface contract vLLM itself defines.

Scope:
- The abstract base class and its hooks (get/put/lookup-style methods)
- What the scheduler side must implement vs. what the worker side must implement
- Why this abstraction exists: letting KV cache be taken over by external
  systems (offload, sharing, persistent cache, etc.)

Status: Not started.

---

## Phase 4 — LMCache as a Connector Implementation
Goal: With Phase 3's interface knowledge in hand, understand how LMCache
answers each part of that interface.

Key questions (original motivating interest):
- How does it handle non-contiguous chunks (mapping to PagedAttention's
  non-contiguous block layout)?
- What bottleneck does the custom CUDA kernel solve? (working theory:
  gather/scatter inefficiency — needs verification)
- How does it coordinate cache hit/miss timing with the scheduler
  (prefix caching logic)?

Status: Not started.

---

## Phase 5 — Self-Synthesis
Goal: Reverse the roles — write my own architecture summary/diagrams first;
Claude only reviews and flags gaps or errors, doesn't author from scratch.

Status: Not started.