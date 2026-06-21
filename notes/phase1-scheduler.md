# Phase 1 — Tracing LLM.generate() → scheduler's first decision

## Trace A: generate() → the ZMQ boundary (where the scheduler lives)

Goal: follow a single offline `LLM.generate()` call from the entrypoint down to
the point where the request is handed to the EngineCore that owns the scheduler.

### Step 1 — `LLM.generate(...)`
File: `vllm/vllm/entrypoints/llm.py:422`
Validates `runner_type == "generate"` (466), fills default sampling params (473), then delegates.
→ calls `self._run_completion(...)` at llm.py:476

### Step 2 — `_run_completion(...)`
File: `vllm/vllm/entrypoints/offline_utils.py:326`
Two-phase: first *adds* all requests, then *drives* the engine loop.
Key shape — adding and running are separate.
→ calls `self._add_completion_requests(...)` at offline_utils.py:340
→ then calls `self._run_engine(...)` at offline_utils.py:349

### Step 3 — `_add_completion_requests(...)`
File: `vllm/vllm/entrypoints/offline_utils.py:290`
Normalizes prompts/params/lora/priority into parallel sequences (303–306), then renders each prompt lazily.
→ calls `self._render_and_add_requests(...)` at offline_utils.py:308

### Step 4 — `_render_and_add_requests(...)`
File: `vllm/vllm/entrypoints/offline_utils.py:523`
Loops over prompts (534), adds one at a time, collects request IDs.
On failure aborts everything already added (547).
→ calls `self._add_request(...)` per prompt at offline_utils.py:535

### Step 5 — `_add_request(...)`
File: `vllm/vllm/entrypoints/offline_utils.py:552`
Forces `output_kind = FINAL_ONLY` (561) — offline mode only wants the final result, not per-token streaming.
Generates a request ID from a counter (563).
→ calls `self.llm_engine.add_request(...)` at offline_utils.py:565

#### Q&A — can `FINAL_ONLY` be a streaming mode instead?

> **What I understood you were asking:** Since Step 5 hard-codes `FINAL_ONLY`,
> does that mean the request can never stream? Is `FINAL_ONLY` simply incapable
> of streaming output?
>
> **Answer:** The `output_kind` field itself fully supports streaming — it has
> three modes (`vllm/vllm/sampling_params.py:182–188`): `CUMULATIVE = 0` (every
> `RequestOutput` carries the whole output so far; the default at
> sampling_params.py:301), `DELTA = 1` (each carries only the new tokens), and
> `FINAL_ONLY = 2` (no intermediate outputs at all). `DELTA`/`CUMULATIVE` are
> exactly how the online server streams tokens.
>
> The constraint is not in the enum — it's that the offline `LLM.generate()`
> path *forces* `FINAL_ONLY` (`offline_utils.py:561`), overriding whatever the
> caller set. Reason: the driver loop `_run_engine` only keeps an output when
> `output.finished` is true (`offline_utils.py:598`) and returns
> `list[RequestOutput]` all at once. Intermediate snapshots would be discarded
> anyway, so forcing `FINAL_ONLY` tells the engine not to waste work building
> them. Streaming (`DELTA`/`CUMULATIVE`) lives on the async/online `AsyncLLM`
> path, not `LLM.generate()`. `[inferred from code]`

### Step 6 — `LLMEngine.add_request(...)`
File: `vllm/vllm/v1/engine/llm_engine.py:218`
Runs `input_processor.process_inputs(...)` (250) → raw prompt becomes an `EngineCoreRequest`
(tokenization, mm processing). Then branches on `n` (number of sampling sequences):
- **Branch taken: `n == 1`** (llm_engine.py:272) — normal single-sample request.
  Registers output bookkeeping via `output_processor.add_request` (274), then pushes to engine core (276).
- (Not taken: `n > 1` at 279 fans out child requests.)
→ calls `self.engine_core.add_request(request)` at llm_engine.py:276

#### Q&A — what is "mm processing", and what does "branches on `n`" mean?

> **What I understood you were asking:** Two terms in Step 6 are unexplained —
> (a) what does "mm processing" inside `process_inputs` actually refer to, and
> (b) what is `n`, and what does it mean for the code to "branch on `n`"?
>
> **Answer (a) — mm = multimodal:** `process_inputs` (`llm_engine.py:250`) may
> run the model's HuggingFace multimodal processor to convert raw media (images
> / audio / video) into tensors/feature specs before the request enters the
> engine. Evidence in `vllm/vllm/v1/engine/input_processor.py`: imports
> `MULTIMODAL_REGISTRY` / `MultiModalFeatureSpec` (lines 19–21);
> `self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)`
> (line 58); the mm-budget setup only runs `if self.supports_mm_inputs:`
> (line 61). For a text-only model (our scenario) `supports_mm_inputs` is
> `False`, so this is skipped and `process_inputs` only tokenizes/renders.
> `[inferred from code]`
>
> **Answer (b) — `n` = number of completions per prompt:** `n` is
> `SamplingParams.n` (`vllm/vllm/sampling_params.py:213`, "Number of outputs to
> return for the given prompt request"; default `1`). It's the OpenAI `n`
> parameter. The branch at `llm_engine.py:272`: when `n == 1` (our scenario) one
> prompt → one sequence, add a single request; when `n > 1` (line 279) it builds
> a `ParentRequest` (280) and fans out `n` child requests (281–292), each its own
> `EngineCoreRequest` with a distinct `request_id` and child sampling params,
> sharing the same prompt. The scheduler then treats them as `n` independent
> sequences (they can share the prompt's KV-cache blocks via prefix caching).
> `[inferred from code]`

### ── Decisive branch: where does the scheduler run? ──
`self.engine_core` was bound at construction (`llm_engine.py:105`, `EngineCoreClient.make_client`).

Dispatch in `make_client` — `vllm/vllm/v1/engine/core_client.py:83`:
- `multiprocess_mode and not asyncio_mode` → returns **`SyncMPClient`** (core_client.py:103)
- else → `InprocClient` (core_client.py:105)

`multiprocess_mode` traces back to a **default-on** flag:
- `LLM` builds via `LLMEngine.from_engine_args` (llm.py:349)
- sets `enable_multiprocessing=True` when `envs.VLLM_ENABLE_V1_MULTIPROCESSING` (llm_engine.py:174–176)
- which defaults to `True` (`vllm/vllm/envs.py:139`; env var defaults to `"1"` at envs.py:1269–1270)

**Conclusion:** by default `engine_core` is a `SyncMPClient`, so the scheduler does NOT run in the
Python process. It runs in a separate background `EngineCore` process, reached over ZMQ.
Docstring confirms intent: "SyncMPClient: ZMQ + background proc EngineCore (for LLM)" (core_client.py:78).
Source: `[inferred from code]` — read from dispatch logic + env default, not an RFC.

```
── PROCESS BOUNDARY (ZMQ, about to cross) ──
   LLM process  →  background EngineCore process (where Scheduler lives)
```

The `engine_core.add_request(request)` call at llm_engine.py:276 is about to cross this boundary.

#### Finding — does multiprocessing change the scheduler itself, or only how requests reach it?

Question: if I force `VLLM_ENABLE_V1_MULTIPROCESSING=0` (InprocClient path), do I get the
*same* `Scheduler` class and the *same* decision logic that runs inside `EngineCoreProc`?

**Answer: same `Scheduler`, same decision logic.** Multiprocessing only changes the transport
(how requests arrive) and the driver (who calls `step()`), NOT how `schedule()` decides.

Both client paths converge on the same `EngineCore`:
- `InprocClient.__init__` wraps a plain `EngineCore`: `self.engine_core = EngineCore(...)` (core_client.py:287).
- The multiprocess path's `EngineCoreProc` is a **subclass**: `class EngineCoreProc(EngineCore)` (core.py:894).

The scheduler is built once, in `EngineCore.__init__`, independent of multiprocessing:
- `Scheduler = vllm_config.scheduler_config.get_scheduler_cls()` (core.py:137)
- `self.scheduler = Scheduler(...)` (core.py:150) — the only `self.scheduler =` in the file;
  `EngineCoreProc` inherits `__init__` and does not rebuild it.

The decision logic lives in `EngineCore.step()`, inherited unchanged (no `def step`/`def schedule`
after line 894):
- `scheduler_output = self.scheduler.schedule(...)` (core.py:490, also 547)

What multiprocessing actually changes (transport + driver only):
| Concern | In-process (`InprocClient`) | Multiprocess (`EngineCoreProc`) |
|---|---|---|
| Who calls `step()` | your thread via `get_output` → `step_fn()` (core_client.py:290) | background busy loop → `_process_engine_step` → `step_fn()` (core.py:1302) |
| How requests arrive | direct method call (core_client.py:297–299) | ZMQ → input queue |
| `schedule()` decision logic | **same** `EngineCore.step` / `Scheduler` | **same** `EngineCore.step` / `Scheduler` |

Both drivers call the same `self.step_fn()` (core_client.py:290 and core.py:1302).

**Implication:** studying scheduling with `VLLM_ENABLE_V1_MULTIPROCESSING=0` is faithful — it's the
same `Scheduler.schedule()` production runs. Caveat `[inferred from code]`: multiprocess mode adds
machinery *around* scheduling (busy loop's `time.sleep(0.001)` GIL-yield at core.py:1312–1313; DP
coordination in `DPEngineCoreProc` at core.py:1743) — these affect timing/concurrency, not what
`schedule()` picks. Source: `[inferred from code]`.

## Trace B: from Step 6 to `Scheduler.schedule()` — the connective tissue

Key finding: `add_request` and `schedule()` are **NOT in one call chain**. They are
two separate chains joined only by a shared queue, `self.scheduler.waiting`
(`scheduler.py:180`, built once via `create_request_queue(self.policy)`).
One chain deposits, the other drains. This is the concrete proof of the
"two phases" answer in roadmap.md.  `[inferred from code]`

(In-process / `InprocClient` path — same `EngineCore`/`Scheduler` as multiproc, see Trace A.)

### Producer chain (deposit a Request into `self.waiting`)
```
llm_engine.py:276   self.engine_core.add_request(request)
 └─core_client.py:297  InprocClient.add_request
     ├─:298  EngineCore.preprocess_add_request(request)   → (req, request_wave)
     └─:299  EngineCore.add_request(req, request_wave)
         └─core.py:403   self.scheduler.add_request(request)
             └─scheduler.py:1976  Scheduler.add_request → _enqueue_waiting_request
                 └─scheduler.py:1816  self.waiting.add_request(request)
```

### Consumer chain (drain `self.waiting` via the engine loop)
```
offline_utils.py:594  while self.llm_engine.has_unfinished_requests():
offline_utils.py:595    self.llm_engine.step()
 └─llm_engine.py:304   outputs = self.engine_core.get_output()
     └─core_client.py:290  InprocClient.get_output → self.engine_core.step_fn()
         # step_fn bound at core.py:221-223 → EngineCore.step (batch_queue is None)
         └─core.py:490   self.scheduler.schedule(...)
             └─scheduler.py:387   Scheduler.schedule → reads self.waiting at :628
```

Bridge = `self.waiting` (+ `self.running`). To verify yourself: breakpoint/print
at `scheduler.py:1816` (deposit) and `scheduler.py:628` (drain) — they fire on
different stack traces.

中：`add_request` 和 `schedule()` 之間沒有直接呼叫，它們靠 `self.waiting` 佇列接起來——一邊存、一邊取。

## Trace C: inside `Scheduler.schedule()` (`scheduler.py:387`)

Design note at `scheduler.py:389–398`: there is **no "prefill phase" vs "decode
phase."** Every request just has `num_computed_tokens` trying to catch up to
`num_tokens_with_spec`. Prefill and decode are the *same* operation with
different token counts — the heart of continuous batching.

Shared resource: `token_budget = self.max_num_scheduled_tokens` (`scheduler.py:407`).
Two passes spend from this one budget, both writing the same `num_scheduled_tokens`
dict → one `SchedulerOutput` → one model forward.

### Pass 1 — RUNNING first (`scheduler.py:429–612`)
- `while req_index < len(self.running) and token_budget > 0:` (`:431`)
- catch-up token count: `num_new_tokens = num_tokens_with_spec +
  num_output_placeholders − num_computed_tokens` (`:462–466`), capped by
  `token_budget` (`:469`).  Decode ≈ 1 token; in-progress chunked-prefill = remaining chunk.
- allocate KV: `kv_cache_manager.allocate_slots(...)` (`:524`). If `None` (no free
  blocks) → **preempt** lowest-priority / last running req (`:536–567`) and retry.
- on success: `num_scheduled_tokens[id] = num_new_tokens`; `token_budget -= num_new_tokens` (`:577–578`).

### Pass 2 — WAITING (new prefills) (`scheduler.py:624–...`)
- `while (self.waiting or self.skipped_waiting) and token_budget > 0:` (`:628`)
- stop if `len(self.running) == self.max_num_running_reqs` (`:629–630`)
- pull next waiting request (`:632–635`) and admit it (allocate + schedule prompt tokens).

**Where prefill+decode mix:** running decodes get FIRST claim on `token_budget`;
leftover budget admits new prefills from `waiting`. A newly-deposited request gets
folded in here on the next `step()`. `[inferred from code]`

中：跑中的 decode 先吃 token 預算，剩下的才拿去做新的 prefill——同一個 step 混進同一批。

### Next / open thread
- Open checkpoint (to answer before tracing on): why does the RUNNING pass run
  *before* the WAITING pass? What breaks if new prefills are admitted first?
  (hint: preemption at `:561`, and a half-generated request losing its turn.)
- Next to trace: `kv_cache_manager.allocate_slots` (`scheduler.py:524`) and the
  preemption path (`:536–567`).
