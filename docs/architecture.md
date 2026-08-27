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
  │  stream=true  → per-token SSE chunks over text/event-stream        │
  │  stream=false → buffer all tokens, emit ChatCompletionResponse    │
  │                                                                   └──▶ Client
```

> **Design document vs. implementation.** This file describes the intended architecture. Where the shipped code diverges, the divergence is called out inline. The two largest today: `MLXInferenceEngine` is an actor that runs each assembled group's requests **sequentially** rather than as a batched forward pass, and the KV prefix cache's restored state has not yet been shown to reduce prefill wall-clock time (last measured: 0.1%, see `benchmarks/kv-cache-hits.json`).

Each stage is an actor or structured-concurrency task; backpressure propagates naturally because the `AsyncStream` buffer is bounded.

---

## Batching Strategy

### Goal

Maximise GPU utilisation by grouping requests that can share a single matrix multiplication kernel, while bounding latency with a hard deadline.

### Bin Assignment

When a request is enqueued it is assigned to a **(model, bin)** pair:

```
bin = smallest b in [128, 256, 512, 1024, 2048, 4096] with b >= prompt_token_count
      (or 8192 if none fits)
```

Requests in the same bin have sequence lengths within one bin width of each other, which keeps padding waste bounded and avoids jagged tensor shapes that inhibit kernel fusion. `prompt_token_count` is currently *estimated* as `characters / 4` rather than tokenised, since binning happens before the model container is acquired.

### Batch Assembly Loop

```
repeat every tick (or when a new request arrives):
    for each (model, bin) with queued requests:
        candidates = requests where deadline not yet passed
        if |candidates| >= MIN_BATCH_SIZE
            OR oldest_candidate.deadline <= now:
                emit batch(candidates[0 ..< MAX_BATCH_SIZE])
```

| Parameter       | Default | Notes                                                        |
|-----------------|---------|--------------------------------------------------------------|
| bin ladder      | 128…4096 | Geometric; overflow bin at 8192                             |
| `maxBatchSize`  | 8       | Upper bound on group size                                     |
| `maxWaitMs`     | 100 ms  | Per-group deadline from first enqueue (`GatewayServer.swift`) |

`BatchAssemblerConfig`'s own default for `maxWaitMs` is 50 ms; `GatewayServer` constructs it with 100 ms. Setting `maxWaitMs = 0` dispatches essentially immediately (useful for single-request debugging). Setting it higher increases the chance of a full group at the cost of tail latency.

### Padding & Masking — not implemented

The design calls for padding all sequences in a group to the bin length with the model's `pad_token_id` and masking the padding positions out of softmax. **The shipped `batchedInfer` does not do this**, because it does not build a batched tensor at all: it acquires the model container once and iterates the group's requests one at a time, each with its own prompt length. Padding and masking become necessary only once a real batched forward pass lands.

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

**Measured effect:** the lookup, restore, and store path is implemented and hits on 79% of the benchmark workload, but the last end-to-end run found a **0.1%** P50 prefill latency reduction versus no cache — effectively nothing (`benchmarks/kv-cache-hits.json`). That measurement predates the store-path correctness fix in `ad92b78` and has not been repeated. Treat the design below as intent and the 0.1% as the current evidence.

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

There is no startup directory scan. A `ModelID` is a Hugging Face Hub repo ID (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`) passed in the request's `model` field; the engine downloads and caches it via the HuggingFace Swift SDK on first use, then keeps it resident in the pool. Any Hub repo in MLX format is therefore usable without pre-registration.

---

## API Surface

The gateway exposes an OpenAI-compatible REST API. All endpoints accept and return `application/json` unless noted.

### POST /v1/chat/completions

Creates a chat completion.

**Request** — `ChatCompletionRequest`

| Field               | Type                   | Required | Notes                                   |
|--------------------|------------------------|----------|-----------------------------------------|
| `model`             | string                 | yes      | Hugging Face Hub repo ID in MLX format  |
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

**Streaming response** — `text/event-stream`

Requests with `"stream": true` receive a real per-token SSE stream. `MLXInferenceEngine.makeStreamHandler()` drives `TokenIterator` one token at a time and yields each decoded fragment as a `StreamChunk.token` through an `AsyncThrowingStream`, followed by a single `StreamChunk.done(FinishReason)`. `ChatRouter.sseResponse()` emits these as `ChatCompletionChunk` events with `Content-Type: text/event-stream` and `Cache-Control: no-cache`, opening with a role-establishing chunk and closing with `data: [DONE]`.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Paris"},"index":0}],...}

data: [DONE]
```

### GET /v1/models

Lists the models **currently resident in the pool** — those that have served at least one request since startup. Returns an empty list before any request is processed; it is not a catalogue of what the Hub offers.

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

| HTTP status | `type`                  | When                                   | Status |
|-------------|-------------------------|----------------------------------------|--------|
| 400         | `invalid_request_error` | Malformed JSON, missing required field | shipped |
| 401         | `invalid_request_error` | Missing or unrecognised bearer token (only when `API_KEY_FILE` is set) | shipped |
| 500         | `server_error`          | Model load failure, inference error    | shipped |
| 504         | `server_error`          | Exceeded `REQUEST_TIMEOUT_SECONDS`     | shipped |
| 404         | `invalid_request_error` | Model not found                        | planned — currently surfaces as 500 |
| 429         | `rate_limit_error`      | Admission gate full                    | planned — no admission gate is wired yet |
| 503         | `server_error`          | Model evicted mid-request              | planned |

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
Loading a 7B 4-bit model takes 3–8 seconds and consumes 4–5 GB of unified memory. Since models are addressed by Hugging Face Hub repo ID, there is no bounded set to preload — and eagerly loading anything would block the health endpoint and exhaust memory. Lazy loading with coalesced concurrent-load tasks (the `loading` dictionary in `ModelPool`) gives fast first-token latency for resident models and correct serialised behaviour for cold starts.
