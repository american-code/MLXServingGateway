# MLX Serving Gateway

A Swift-native inference server that exposes an **OpenAI-compatible HTTP API** over locally-loaded MLX models on Apple Silicon. It groups concurrent requests, shares KV cache prefixes across requests with common system prompts, and manages a multi-model LRU pool — all without leaving unified memory.

> **Current performance status.** `MLXInferenceEngine` is an actor and processes each request group sequentially — there is no batched forward pass yet, so aggregate throughput is flat across concurrency (36.5 → 39.5 tok/s from C=1 to C=16) and latency grows with queue depth. KV prefix sharing hits 79% of requests but the last measured wall-clock prefill saving was ~0.1%. See [Benchmark Results](#benchmark-results) and [`posts/mlx-serving-gateway-benchmark.md`](posts/mlx-serving-gateway-benchmark.md) for the full data and the roadmap out of it.

**Stack:** Swift 6 · Hummingbird 2 · MLX Swift · structured concurrency

---

## Requirements

| Requirement | Minimum |
|-------------|---------|
| macOS | 14 Sonoma |
| Xcode / Swift toolchain | Swift 6.0 |
| Apple Silicon | M1 or later (required for MLX) |

---

## Installation

```bash
git clone https://github.com/american-code/MLXServingGateway
cd MLXServingGateway
swift build -c release
```

The release binary lands at `.build/release/GatewayServer`.

---

## Model Loading

Models are fetched from the **Hugging Face Hub** on demand using their Hub repository IDs — no local directory scanning or pre-registration is required. Pass the Hub ID in the `"model"` field of your request:

```json
"model": "mlx-community/Qwen2.5-7B-Instruct-4bit"
```

On the first request for a model the server downloads and caches it via the HuggingFace Swift SDK. A cold-start for a 7B 4-bit model takes 3–8 seconds; subsequent requests are served from the resident pool (up to `MAX_MODELS` models at once). The least-recently-used model is evicted when the limit is reached.

The `GET /v1/models` endpoint lists models **currently resident in the pool** — not all models available on the Hub. A model only appears there after at least one successful request has loaded it.

---

## Starting the Server

```bash
.build/release/GatewayServer
```

The server binds to `127.0.0.1:8080` by default. Verify it is ready:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

---

## API Key Setup

Authentication is opt-in. When `API_KEY_FILE` is set, every request must carry a valid `Authorization: Bearer <token>` header. When the variable is unset, the server accepts all requests.

**Create a key file:**

```
# ~/.config/mlx-gateway/keys.txt
# Lines starting with # are comments and are ignored.
# One key per line. Any non-empty string works — these are your own keys,
# not provider-issued tokens.
gw-local-key-1
gw-local-key-2
```

**Start the server with auth enabled:**

```bash
API_KEY_FILE=~/.config/mlx-gateway/keys.txt .build/release/GatewayServer
```

A missing or unrecognised token returns `401 Unauthorized` with a JSON error body.

---

## Example curl Commands

All examples assume the server is running at `localhost:8080`. If auth is enabled, add `-H "Authorization: Bearer gw-local-key-1"` to each command.

### Non-streaming completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is the capital of France?"}
    ],
    "max_tokens": 128
  }'
```

### Streaming completion

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is the capital of France?"}
    ],
    "max_tokens": 128,
    "stream": true
  }'
```

The response is `text/event-stream` (chunked transfer). Each token is delivered as a Server-Sent Event:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Paris"},"index":0}],...}

data: [DONE]
```

### List resident models

Returns models currently loaded in the pool (those that have received at least one request since the server started). Returns an empty list if no requests have been processed yet.

```bash
curl http://localhost:8080/v1/models
```

### Health check

```bash
curl http://localhost:8080/health
```

---

## Configuration Reference

All configuration is via environment variables. No config file is required.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_MODELS` | `3` | Maximum number of models resident in memory simultaneously. Least-recently-used model is evicted when the limit is reached |
| `API_KEY_FILE` | *(unset — auth disabled)* | Path to a text file of valid API keys, one per line; `#`-prefixed lines are comments |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Per-request deadline in seconds. Requests that exceed this return `504 Gateway Timeout` |

### Batching parameters

Batching is configured at compile time in `GatewayServer.swift`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxBatchSize` | `8` | Maximum requests released to the engine as one group |
| `maxWaitMs` | `100` | Maximum time (ms) to wait for a full group before dispatching. Set to `0` to dispatch immediately |

Requests are keyed by `(model, sequence-length bin)`, with bins at 128 / 256 / 512 / 1024 / 2048 / 4096 tokens (anything larger falls into a single 8192 bin). Token counts are estimated from character length, not tokenised, at this stage.

**A group is not a tensor batch.** `MLXInferenceEngine.batchedInfer` acquires the model container once for the whole group and then runs each request through the token iterator **sequentially**. Grouping amortises model acquisition and scheduling; it does not currently produce a batched forward pass, which is why throughput does not rise with concurrency.

---

## Architecture Overview

```
POST /v1/chat/completions
       │
  [Auth Middleware]   ← optional Bearer token check
       │
  [ChatRouter]        ← JSON decode, validate
       │
  [BatchAssembler]    ← group by (model, sequence-length bin); wait ≤ maxWaitMs
       │
  [ModelPool]         ← acquire model; LRU evict if at capacity
       │
  [KVPrefixCache]     ← walk prefix trie; restore cached KV state for shared prefixes
       │
  [MLXInferenceEngine]← autoregressive decoding, one request at a time (actor-serialized)
       │
  [ChatRouter]        ← buffered JSON or SSE streaming response
       │
     Client
```

See [`docs/architecture.md`](docs/architecture.md) for the full design, including the prefix trie eviction policy, bin-based batching rationale, and multi-model LRU pool internals.

---

## Benchmark Results

All numbers below were measured on a MacBookPro18,3 (Apple M1 Pro, 16 GB unified memory) on 2026-07-30. Raw data lives in `benchmarks/`. Full write-up: [`posts/mlx-serving-gateway-benchmark.md`](posts/mlx-serving-gateway-benchmark.md).

### Throughput and Latency

**Model:** `mlx-community/Llama-3.2-3B-bf16` (unquantized). 130-token prompt, 64 output tokens, KV prefix cache forced to miss. Source: `benchmarks/load-test-results.json`.

| Concurrency | Throughput | P50 | P95 | Mean |
|---|---|---|---|---|
| 1  | 36.5 tok/s | 1,754 ms | 1,763 ms | 1,751 ms |
| 4  | 38.8 tok/s | 6,600 ms | 6,612 ms | 6,599 ms |
| 16 | 39.5 tok/s | 25,894 ms | 25,912 ms | 21,591 ms |
| 32 | 39.4 tok/s | 32,455 ms | 51,927 ms | 32,458 ms |

Throughput moves 8.2% across a 32× concurrency range — that is flat. Because inference is actor-serialized, concurrency buys latency, not throughput.

Single-request throughput on 4-bit weights (`mlx-community/Llama-3.2-3B-Instruct-4bit`, 8 runs, 128 output tokens, `benchmarks/gateway-vs-ollama.json`): **116.74 tok/s** mean, P50 latency 1,095 ms, 1,933 MB resident.

> **No Ollama comparison exists.** `benchmarks/gateway-vs-ollama.json` has a real gateway measurement and an `HTTP 500` where the Ollama result would be. The run was never repeated. Nothing in this repo supports a faster-than-Ollama claim.

### KV Cache Prefix Sharing

**Workload:** 100 requests; 80% share a 512-token system-prompt prefix; unique suffix up to 64 tokens. Source: `benchmarks/kv-cache-hits.json`.

| Metric | Result |
|--------|--------|
| Cache hit rate | **79%** |
| Average tokens saved (trie accounting) | **70.1%** |
| P50 prefill latency reduction — **measured** | **0.1%** |
| P50 prefill latency reduction — *modeled from token savings* | *87.2%* |
| Trie lookup P50 | 53.3 µs |
| Trie lookup P95 | 82.7 µs |
| Trie store P50 | 231.7 µs |

The end-to-end measurement (100 requests per phase, `max_tokens=1`) found 585.9 ms P50 with unique prompts versus 585.3 ms P50 with a shared prefix. The trie finds the overlap, but on that run the restored KV state did not displace GPU prefill work. Trie-lookup overhead (P50 53 µs) is negligible either way.

That measurement predates commit `ad92b78`, which fixed a prefix-cache store bug (a stray `trim(1)` left the trie key and the stored KV state off by one token). The KV figures have **not** been re-measured since that fix.

```bash
swift run KVCacheBenchmark
```

Note this runs the Foundation-only trie simulation and **overwrites** `benchmarks/kv-cache-hits.json` with trie-only metrics, including the *modeled* latency reduction. The measured wall-clock block in the committed file comes from a separate end-to-end HTTP run.

---

## Running Tests

```bash
swift test
```

---

## License

MIT — see [`LICENSE`](LICENSE).
