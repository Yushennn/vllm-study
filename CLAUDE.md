# Learning Context: vLLM & LMCache Deep Dive

## My goal
I am NOT building a product. I am reverse-engineering vLLM and LMCache
to deeply understand: continuous batching, the scheduler, the KV-cache
connector abstraction, ZMQ-based IPC between API server and EngineCore,
and LMCache's non-contiguous chunk offloading + custom CUDA kernels.

## How to work with me
- Default to TRACING, not SUMMARIZING. When I ask about a mechanism,
  walk me through the actual call path (file:line, function names, in
  execution order) rather than giving me a high-level paragraph first.
- Always answer in English. Only when a concept is genuinely hard to
  bridge, add ONE short Chinese sentence (not a paragraph) prefixed
  with "中:". Do not translate routine explanations. And please ALWAYS
  USE TRADITIONAL CHINESE.
- Prefer asking me "what do you think happens here?" before revealing
  the answer, when we're tracing a new code path together — but only
  when I haven't already shown understanding. Don't do this for
  trivial lookups.
- When citing code, always give exact file paths and line numbers so
  I can open them myself.
- Do not generate full architecture diagrams or summary docs unless I
  explicitly ask — I want to build understanding incrementally, not
  receive finished artifacts.
- Flag explicitly when something is a vLLM design decision documented
  in a GitHub RFC/discussion vs. something you're inferring from code.

## Repo locations

### vLLM (`./vllm/`)
Python package root: `./vllm/vllm/`

Key top-level files:
- `sequence.py` — core data structures (IntermediateTensors, pipeline hidden states)
- `sampling_params.py` — SamplingParams dataclass
- `envs.py` / `env_override.py` — runtime env-var config (separate from config/ dataclasses)
- `forward_context.py` — BatchDescriptor passed through the forward pass (used by CUDA graphs)

Key subsystem directories:
| Path | What lives there |
|---|---|
| `vllm/v1/` | Current-generation engine (everything below is inside here) |
| `vllm/v1/engine/` | AsyncLLMEngine, core.py (EngineCore), coordinator, detokenizer, input/output processors, tensor_ipc |
| `vllm/v1/core/` | KV cache manager, block pool, encoder cache manager, kv_cache_utils; scheduler lives in `core/sched/` (scheduler.py, async_scheduler.py, request_queue.py) |
| `vllm/v1/worker/` | gpu_worker.py, gpu_model_runner.py, cpu_worker.py, xpu_worker.py, worker_base.py, block_table.py, gpu_input_batch.py, ubatching.py, mamba_utils.py, dp_utils.py, ec/kv/lora connector mixins |
| `vllm/v1/attention/` | Attention abstraction; backends in `attention/backends/`, ops in `attention/ops/` |
| `vllm/v1/kv_offload/` | Tiered KV cache offloading to CPU/disk; `cpu/`, `tiering/`, `worker/` sub-dirs |
| `vllm/v1/simple_kv_offload/` | Simpler single-tier KV offload variant |
| `vllm/v1/kv_cache_interface.py` | KV cache spec types, quantization modes, per-backend shape contracts |
| `vllm/v1/kv_cache_spec_registry.py` | Maps attention backends → KV cache specs |
| `vllm/v1/cudagraph_dispatcher.py` | Runtime CUDA graph dispatch (FULL/PIECEWISE/NONE) |
| `vllm/v1/request.py` | v1 Request class (sampling params, multimodal, structured output) |
| `vllm/v1/spec_decode/` | Speculative decoding (`dynamic/` sub-dir) |
| `vllm/v1/sample/` | Sampling logic and logits processors |
| `vllm/v1/structured_output/` | Guided/constrained decoding |
| `vllm/v1/metrics/` | v1 metrics |
| `vllm/v1/pool/` | Memory pool |
| `vllm/engine/` | Legacy (pre-v1) engine |
| `vllm/entrypoints/` | Server entrypoints: `openai/` (OpenAI-compat), `anthropic/`, `mcp/`, `cli/`, `generate/`, `pooling/`, `serve/`, `speech_to_text/` |
| `vllm/model_executor/` | Model loading (`model_loader/`), custom layers (`layers/`), GPU kernels (`kernels/`), LoRA, offloader, warmup |
| `vllm/model_executor/models/` | Per-architecture model implementations |
| `vllm/distributed/` | TP/PP/DP parallel state (`parallel_state.py`), device communicators, KV transfer for disaggregated prefill (`kv_transfer/`), encoder cache transfer (`ec_transfer/`), elastic expert parallelism (`elastic_ep/`), EPLB (`eplb/`), weight transfer (`weight_transfer/`) |
| `vllm/config/` | All config dataclasses (VllmConfig, etc.) |
| `vllm/compilation/` | torch.compile integration and passes |
| `vllm/multimodal/` | Multimodal input processing (`media/`, `processing/`) |
| `vllm/lora/` | LoRA adapter support (`layers/`, `ops/`, `punica_wrapper/`) |
| `vllm/reasoning/` | Per-model reasoning parsers (~25 models: DeepSeek R1/V3, Qwen3, Gemma4, Granite, Mistral, etc.) |
| `vllm/tokenizers/` | Tokenizer utilities |
| `vllm/tool_parsers/` | Tool-call output parsers |
| `vllm/transformers_utils/` | HuggingFace transformers utilities, chat templates, processor configs |
| `vllm/ir/` | Intermediate representation and ops (used by compilation passes) |
| `vllm/kernels/` | Triton and Helion kernels |
| `vllm/platforms/` | Platform abstraction (CUDA, ROCm, CPU, XPU, etc.) |
| `vllm/plugins/` | Plugin system (io_processors, lora_resolvers) |
| `vllm/ray/` | Ray integration |
| `vllm/tracing/` | OpenTelemetry tracing |
| `vllm/profiler/` | Profiling utilities |
| `vllm/utils/` | General utilities |
| `vllm/vllm_flash_attn/` | FlashAttention integration shim |
| `csrc/` | C++/CUDA kernels: `attention/`, `moe/`, `quantization/`, `cpu/`, `rocm/`, `cutlass_extensions/` |

### LMCache (`./LMCache/`)
Python package root: `./LMCache/lmcache/`

Compiled native extensions (built C++ backends, at package root):
- `lmcache_fs.so` — filesystem storage backend
- `lmcache_redis.so` — Redis storage backend
- `native_storage_ops.so` — native storage ops (fallback in `python_ops_fallback.py`)

Key top-level files:
- `observability.py` — top-level observability hook
- `usage_context.py` — tracks serving mode/usage context
- `python_ops_fallback.py` — pure-Python fallback when native extensions absent

Key subsystem directories:
| Path | What lives there |
|---|---|
| `lmcache/v1/` | Core v1 engine (everything below is inside here) |
| `lmcache/v1/cache_engine.py` | **Main KV cache engine** — stores/retrieves KV caches across tiers |
| `lmcache/v1/ec_engine.py` | Encoder Cache engine — per-hash encoder tensors for multimodal |
| `lmcache/v1/manager.py` | LMCacheManager — unified lifecycle manager, decouples vLLM adapter from internals |
| `lmcache/v1/config.py` / `config_base.py` | Config system: YAML, env-vars, CLI overrides |
| `lmcache/v1/cache_interface.py` | Public cache interface contract |
| `lmcache/v1/protocol.py` | Protocol/message definitions |
| `lmcache/v1/gpu_connector/` | Pulls KV tensors off GPU: `gpu_connectors.py`, `gpu_ops.py`, `kv_format/`, platform variants (musa, xpu, hpu) |
| `lmcache/v1/kv_codec/` | KV cache encoding/decoding |
| `lmcache/v1/lookup_client/` | Cache lookup client; `record_strategies/` for eviction/admission policies |
| `lmcache/v1/transfer_channel/` | **Data transfer layer**: `nixl_channel.py`, `py_socket_channel.py`, `zmq_transport.py`, `abstract.py` |
| `lmcache/v1/rpc/` | RPC transport: `transport.py`, `zmq_transport.py` |
| `lmcache/v1/multiprocess/` | Multi-process pipeline: `modules/`, `protocols/`, `transfer_context/`, `http_apis/` |
| `lmcache/v1/mp_coordinator/` | Multi-process coordinator + HTTP API (`http_apis/`, `l2/`) |
| `lmcache/v1/distributed/` | Distributed KV store: `l2_adapters/`, `transfer_channel/`, `memory_manager/`, `eviction_policy/`, `serde/`, `storage_controllers/` |
| `lmcache/v1/mp_observability/` | Metrics and tracing for the MP pipeline (`subscribers/`, `trace/`) |
| `lmcache/v1/cache_controller/` | Cache controller logic: `commands/`, `controllers/`, `frontend/` |
| `lmcache/v1/api_server/` | External-facing API server |
| `lmcache/v1/internal_api_server/` | Internal API for vLLM/controller communication: `common/`, `controller/`, `vllm/` |
| `lmcache/v1/compute/` | Attention compute, blend, model helpers (`attention/`, `blend/`, `models/`) |
| `lmcache/v1/platform/` | Platform-specific code (`cuda/`, `cpu/`, `musa/`) |
| `lmcache/v1/offload_server/` | Server-side offload handling |
| `lmcache/v1/standalone/` | Standalone (non-vLLM) deployment: `manager.py`, `standalone_service_factory.py` |
| `lmcache/v1/storage_backend/` | v1 storage backend (separate from legacy top-level one) |
| `lmcache/v1/health_monitor/` | Health monitoring with checks (`checks/`) |
| `lmcache/v1/event_manager.py` | Event pub/sub for internal components |
| `lmcache/v1/memory_management.py` / `lazy_memory_allocator.py` | Memory management + lazy allocation |
| `lmcache/v1/token_database.py` | Token → hash database for cache lookup |
| `lmcache/v1/kv_layer_groups.py` | KV layer group definitions |
| `lmcache/v1/pin_monitor.py` | Pinned CPU memory monitor |
| `lmcache/v1/system_detection.py` | Hardware/platform detection at startup |
| `lmcache/v1/plugin/` | Plugin system |
| `lmcache/v1/server/` | Server implementation |
| `lmcache/integration/` | Adapters for inference engines |
| `lmcache/integration/vllm/` | vLLM connectors: `lmcache_connector_v1.py`, `lmcache_mp_connector.py` (+ versioned variants `_085`, `_0180`, `_0201`), `vllm_v1_adapter.py`, `vllm_ec_adapter.py`, `vllm_multi_process_adapter.py`, `vllm_service_factory.py` |
| `lmcache/integration/sglang/` | SGLang adapter |
| `lmcache/integration/tensorrt_llm/` | TensorRT-LLM adapter |
| `lmcache/cli/` | CLI commands: `bench/`, `query/`, `quota/`, `trace/`, `tool/` |
| `lmcache/storage_backend/` | Legacy storage backend + serialization (`serde/`) |
| `lmcache/tools/` | Utilities: `cache_simulator/`, `controller_benchmark/`, `mp_status_viewer/`, `transfer_channel_benchmark/` |
| `lmcache/lmcache_frontend/` | Web frontend and MP plugin |
| `csrc/` | C++ storage backends: `fs/`, `redis/`, `aerospike/`, `mooncake/`; storage manager |

## Verification habit
When asked to summarize codebase structure, always verify by actually listing directories and opening at least one representative file per claimed subsystem — never summarize purely from folder names.

## Note-writing conventions
When saving Q&A into trace notes, never copy my literal question wording.
Instead write your own interpretation of what you understood I was asking,
labeled "What I understood you were asking" — this is a deliberate check
so I can catch misunderstandings between what I meant and what you parsed. Format:

> **What I understood you were asking:** ...
>
> **Answer:** ...

## Current phase
Phase 1: tracing LLM.generate() -> scheduler -> continuous batching loop


## My running notes
Located in `./notes/` — read before re-explaining something I've already covered.
