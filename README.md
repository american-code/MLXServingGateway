# MLX Serving Gateway

A Swift-native inference server that exposes an **OpenAI-compatible HTTP API** over locally-loaded MLX models on Apple Silicon. It batches concurrent requests, shares KV cache prefixes across requests with common system prompts, and manages a multi-model LRU pool — all without leaving unified memory.

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
git clone https://github.com/your-org/MLXServingGateway
cd MLXServingGateway
swift build -c release
```

The release binary lands at `.build/release/GatewayServer`.

---

## Model Placement

The server scans `~/Models` (configurable via `MODELS_DIR`) for directories that contain both a `config.json` and at least one `*.safetensors` file. Each matching directory is registered as an available model whose ID equals its directory name.

**Recommended layout:**

```
~/Models/
├── mlx-community/
│   ├── Qwen2.5-7B-Instruct-4bit/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer.json
│   └── Llama-3.2-3B-Instruct-4bit/
│       ├── config.json
│       └── ...
```

Models are loaded lazily on the first request that names them. A cold-start for a 7B 4-bit model takes 3–8 seconds; subsequent requests are served from the resident pool.

**Download a model with mlx-lm:**

```bash
pip install mlx-lm
python -m mlx_lm.convert \
  --hf-path mlx-community/Qwen2.5-7B-Instruct-4bit \
  --mlx-path ~/Models/mlx-community/Qwen2.5-7B-Instruct-4bit
```

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
# One key per line.
sk-local-abc123
sk-local-xyz789
```

**Start the server with auth enabled:**

```bash
API_KEY_FILE=~/.config/mlx-gateway/keys.txt .build/release/GatewayServer
```

A missing or unrecognised token returns `401 Unauthorized` with a JSON error body.

---

## Example curl Commands

All examples assume the server is running at `localhost:8080`. If auth is enabled, add `-H "Authorization: Bearer sk-local-abc123"` to each command.

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

### Streaming completion (SSE)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5 slowly."}
    ],
    "stream": true,
    "max_tokens": 64
  }'
```

Each SSE event is a `ChatCompletionChunk`. The stream closes with `data: [DONE]`.

### List available models

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
| `MODELS_DIR` | `~/Models` | Directory scanned for MLX model subdirectories |
| `MAX_MODELS` | `3` | Maximum number of models resident in memory simultaneously. Least-recently-used model is evicted when the limit is reached |
| `API_KEY_FILE` | *(unset — auth disabled)* | Path to a text file of valid API keys, one per line; `#`-prefixed lines are comments |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Per-request deadline in seconds. Requests that exceed this return `504 Gateway Timeout` |

### Batching parameters

Batching is configured at compile time in `GatewayServer.swift`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxBatchSize` | `8` | Maximum requests per inference batch |
| `maxWaitMs` | `100` | Maximum time (ms) to wait for a full batch before dispatching. Set to `0` to disable batching |
| `BIN_SIZE` | `64` | Token-count granularity for sequence-length binning |

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
  [KVPrefixCache]     ← walk prefix trie; reuse cached KV blocks for shared prefixes
       │
  [MLXInferenceEngine]← forward pass; autoregressive decoding; SSE token stream
       │
  [ChatRouter]        ← stream=true → SSE  |  stream=false → buffered JSON
       │
     Client
```

See [`docs/architecture.md`](docs/architecture.md) for the full design, including the prefix trie eviction policy, bin-based batching rationale, and multi-model LRU pool internals.

---

## Benchmark Results

### KV Cache Prefix Sharing

**Workload:** 100 simulated requests; 80% share a 512-token system-prompt prefix; unique suffix up to 64 tokens.

| Metric | Result |
|--------|--------|
| Cache hit rate | **79%** |
| P50 prefill latency reduction | **87.2%** |
| Average tokens saved | **70.1%** |
| Trie lookup P50 | 53.3 µs |
| Trie lookup P95 | 82.7 µs |
| Trie store P50 | 231.7 µs |

The latency reduction figure uses a token-savings model: prefill cost scales linearly with sequence length, so `saved_prefix_tokens / total_prompt_tokens` is a conservative lower bound on wall-clock prefill savings.

Run the benchmark yourself:

```bash
swift run KVCacheBenchmark
# Results saved to benchmarks/kv-cache-hits.json
```

---

## Running Tests

```bash
swift test
```

---

## License

MIT
