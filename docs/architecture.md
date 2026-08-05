# MLX Serving Gateway — Architecture

## Overview

MLXServingGateway is a Swift-native inference server that exposes an OpenAI-compatible HTTP API over locally-loaded MLX models on Apple Silicon. The gateway's core responsibility is to turn a stream of independent HTTP requests into efficient batched MLX inference calls, then stream results back to each caller.

The technology stack is fixed: Swift 6 + Hummingbird 2 for the server layer, the MLX Swift framework for on-device inference, and Swift's structured concurrency (`async`/`await`, actors) throughout.

---

## Request Lifecycle

```
Client
  │
  ▼
[Hummingbird HTTP] ─── POST /v1/chat/completions ───────────────────┐
  │  deserialize JSON → ChatCompletionRequest                        │
  │  validate (model name, token budget, stop sequences)             │
  │                                                                   │
  ▼                                                                   │
[Admission Gate]                                                      │
  │  check concurrent-request cap                                     │
  │  reject with 503 if at limit                                      │
  │                                                                   │
  ▼                                                                   │
[Request Queue]  ← AsyncStream<PendingRequest>                        │
  │  enqueue with deadline = now + max_wait_ms                        │
  │                                                                   │
  ▼                                                                   │
[Batch Assembler]  (one per loaded model)                             │
  │  wait until batch is full OR deadline fires                       │
  │  group by (model, sequence-length bin)                            │
  │  emit InferenceBatch                                              │
  │                                                                   │
  ▼                                                                   │
[Model Pool]                                                          │
  │  acquire model from LRU pool (load if absent, evict LRU if full) │
  │                                                                   │
  ▼                                                                   │
[KV Cache Layer]                                                      │
  │  walk prefix trie, reuse cached KV blocks for shared prefixes     │
  │  allocate new blocks for unique suffixes                          │
  │                                                                   │
  ▼                                                                   │
[MLX Inference]                                                       │
  │  forward pass in batch; generate tokens via sampling              │
  │  emit token stream (AsyncStream<TokenEvent>)                      │
  │                                                                   │
  ▼                                                                   │
[Response Splitter]                                                   │
  │  demux batch output → per-request token streams                   │
  │                                                                   │
  ▼                                                                   │
[Stream Writer]                                                       │
  │  stream=true  → buffered JSON (SSE not yet wired)                  │
  │  stream=false → buffer all tokens, emit ChatCompletionResponse    │
  │                                                                   └──▶ Client
```

Each stage is an actor or structured-concurrency task; backpressure propagates naturally because the `AsyncStream` buffer is bounded.

---

## Batching Strategy

### Goal

Maximise GPU utilisation by grouping requests that can share a single matrix multiplication kernel, while bounding latency with a hard deadline.

### Bin Assignment

When a request is enqueued it is assigned to a **(model, bin)** pair:

```
bin = ceil(prompt_token_count / BIN_SIZE) * BIN_SIZE
```

`BIN_SIZE` defaults to 64. Requests in the same bin have padded sequence lengths that differ by at most `BIN_SIZE − 1` tokens, which keeps padding waste under one bin width and avoids jagged tensor shapes that inhibit kernel fusion.

### Batch Assembly Loop

```
repeat every tick (or when a new request arrives):
    for each (model, bin) with queued requests:
        candidates = requests where deadline not yet passed
        if |candidates| >= MIN_BATCH_SIZE
            OR oldest_candidate.deadline <= now:
                emit batch(candidates[0 ..< MAX_BATCH_SIZE])
```

| Parameter        | Default | Notes                                         |
|-----------------|---------|-----------------------------------------------|
| `BIN_SIZE`       | 64      | Token-count granularity for sequence grouping |
| `MAX_BATCH_SIZE` | 8       | Upper bound; constrained by VRAM              |
| `MIN_BATCH_SIZE` | 1       | Emit immediately when deadline fires          |
| `max_wait_ms`    | 20 ms   | Per-request deadline from enqueue time        |

Setting `max_wait_ms = 0` disables batching (useful for single-request debugging). Setting it higher increases throughput at the cost of tail latency.

### Padding & Masking

The batch assembler pads all sequences to the bin length with the model's `pad_token_id`, then constructs an attention mask so padding positions are excluded from softmax.

---

## KV Cache Design

### Motivation

Many concurrent requests share a common prefix — a system prompt, few-shot examples, or a RAG context. Re-computing the key-value tensors for that prefix on every request wastes compute proportional to the prefix length.

### Prefix Trie

The KV cache is indexed by a *prefix trie* keyed on token-ID sequences:

```
TrieNode {
    children:    [TokenID: TrieNode]
    kv_block:    KVBlock?          // nil for non-terminal nodes
    ref_count:   Int               // live requests using this block
    last_used:   ContinuousClock.Instant
}

KVBlock {
    keys:    MLXArray  // shape [layers, heads, seq_len, head_dim]
    values:  MLXArray
    pinned:  Bool      // true while a request holds a reference
}
```

**Cache lookup (at inference time):**

1. Tokenise the full prompt.
2. Walk the trie token-by-token; record the deepest node with a `kv_block`.
3. The matching prefix up to that node is the *cache hit region* — its KV tensors are reused without recomputation.
4. The remaining suffix is the *cache miss region* — computed during the forward pass and inserted into the trie as new nodes.

**Eviction:**

Blocks are evicted in LRU order when total KV memory exceeds a configurable `kv_cache_max_bytes` limit, subject to `ref_count == 0` (pinned blocks are never evicted).

### Block Granularity

Blocks are stored at the full prefix length of the matching trie node, not in fixed-size pages. This simplifies the implementation at the cost of internal fragmentation when prefixes grow incrementally. A page-quantised variant (e.g. blocks of 16 or 32 tokens) can be introduced later if fragmentation becomes measurable.

---

## Multi-Model LRU Pool

### Purpose

Apple Silicon has unified memory, so loading a model means allocating a large contiguous MLX buffer. The pool manages a bounded set of simultaneously resident models, evicting the least-recently-used one when a new model must be loaded.

### Structure

```
actor ModelPool {
    maxResidentModels: Int          // e.g. 3; configurable at startup
    loaded: OrderedDictionary<ModelID, LoadedModel>
    loading: [ModelID: Task<LoadedModel, Error>]
}

struct LoadedModel {
    model:      MLXModel
    tokenizer:  Tokenizer
    config:     ModelConfig
    lastUsed:   ContinuousClock.Instant
    batcher:    BatchAssembler     // one per resident model
}
```

**Acquire flow:**

```
func acquire(modelID: ModelID) async throws -> LoadedModel {
    if let m = loaded[modelID] {
        m.lastUsed = .now
        return m
    }
    if let t = loading[modelID] {
        return try await t.value   // coalesce concurrent load requests
    }
    if loaded.count >= maxResidentModels {
        evict(lru: loaded.min(by: \.lastUsed))
    }
    let task = Task { try await loadFromDisk(modelID) }
    loading[modelID] = task
    let m = try await task.value
    loaded[modelID] = m
    loading.removeValue(forKey: modelID)
    return m
}
```

**Eviction** calls `model.unload()` (releases MLX buffers) and shuts down the associated `BatchAssembler`. Any requests queued to that batcher receive a 503 and must be retried; the client-visible error message indicates the model is being swapped.

### Model Discovery

At startup, the server scans a configurable `models_dir` (default: `~/Models`) for directories containing `config.json` + `*.safetensors`. Each discovered directory registers as an available `ModelID` equal to its directory name (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`). Models are loaded lazily on first request.

---

## API Surface

The gateway exposes an OpenAI-compatible REST API. All endpoints accept and return `application/json` unless noted.

### POST /v1/chat/completions

Creates a chat completion.

**Request** — `ChatCompletionRequest`

| Field               | Type                   | Required | Notes                                   |
|--------------------|------------------------|----------|-----------------------------------------|
| `model`             | string                 | yes      | Must match a discovered model directory |
| `messages`          | array of ChatMessage   | yes      | `role` ∈ {system, user, assistant}      |
| `temperature`       | float [0, 2]           | no       | Default 1.0                             |
| `top_p`             | float (0, 1]           | no       | Default 1.0                             |
| `max_tokens`        | int                    | no       | Hard cap on generated tokens            |
| `stream`            | bool                   | no       | Default false                           |
| `stop`              | string or string[]     | no       | Stop sequences                          |
| `presence_penalty`  | float [-2, 2]          | no       |                                         |
| `frequency_penalty` | float [-2, 2]          | no       |                                         |
| `seed`              | int                    | no       | For reproducibility                     |

**Non-streaming response** — `ChatCompletionResponse`

```jsonc
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1722124800,
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "..." },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

**Streaming response** — not yet implemented

Requests with `"stream": true` currently receive the same buffered `ChatCompletionResponse` as non-streaming requests. The `ChatRouter` SSE code path (`text/event-stream`, `ChatCompletionChunk`) exists but is not wired to the inference engine.

### GET /v1/models

Lists all discovered models.

```jsonc
{
  "object": "list",
  "data": [
    { "id": "mlx-community/Qwen2.5-7B-Instruct-4bit", "object": "model", "owned_by": "local" }
  ]
}
```

### GET /health

Returns `200 OK` with `{"status":"ok"}` when the server is ready to accept requests. Returns `503` during startup or when all model slots are occupied by loading tasks.

### Error Shape

All errors follow the OpenAI error envelope:

```jsonc
{
  "error": {
    "message": "Model 'unknown-model' is not available.",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

| HTTP status | `type`                  | When                                   |
|-------------|-------------------------|----------------------------------------|
| 400         | `invalid_request_error` | Malformed JSON, missing required field |
| 404         | `invalid_request_error` | Model not found                        |
| 429         | `rate_limit_error`      | Admission gate full                    |
| 503         | `server_error`          | Model evicted mid-request, load failed |

---

## Component Ownership Map

```
GatewayServer (executable)
└── Application (Hummingbird)
    └── ChatRouter              ← HTTP edge; decodes/encodes; validates
        └── InferenceGateway    ← admission gate; enqueue
            └── ModelPool       ← LRU resident-model manager
                └── LoadedModel
                    ├── BatchAssembler  ← (model, bin) grouping; max_wait_ms
                    ├── KVCacheManager  ← prefix trie; block eviction
                    └── MLXInferenceEngine  ← forward pass; token sampling
                        └── ResponseSplitter ← demux → per-request streams
```

---

## Key Design Decisions

**Why bin-based batching instead of continuous batching?**
Continuous batching (PagedAttention style) maximises GPU utilisation but requires custom CUDA kernels. MLX's JIT traces static shapes; bin-based grouping keeps tensor shapes predictable across a batch without requiring hand-written Metal kernels for every model architecture.

**Why a prefix trie instead of a flat LRU cache?**
A flat cache keyed on the full prompt string cannot share a system-prompt block between two requests that have different user turns. The trie lets any number of requests reuse the same KV block as long as their prompts share a common prefix, which is the overwhelmingly common case with a fixed system prompt.

**Why actors for the pool and batcher?**
Swift 6 strict concurrency means any shared mutable state must be protected. Actors provide this guarantee with zero boilerplate and compose cleanly with `async`/`await`-based Hummingbird handlers. The alternative (locking with `NSLock` or `OSAllocatedUnfairLock`) would require manual careful reasoning about every access point.

**Why lazy model loading?**
Loading a 7B 4-bit model takes 3–8 seconds and consumes 4–5 GB of unified memory. Loading every discovered model at startup would block the health endpoint and exhaust memory for any realistic `models_dir`. Lazy loading with coalesced concurrent-load tasks (the `loading` dictionary in `ModelPool`) gives sub-second first-token latency for resident models and correct serialised behaviour for cold starts.
