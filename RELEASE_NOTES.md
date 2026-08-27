# MLX Serving Gateway v0.1.0

**Tagged 2026-07-29 · notes revised 2026-08-27**

First tagged release. This is a working inference server that runs MLX models on Apple Silicon and exposes an OpenAI-compatible REST API.

> These notes describe `master` at its current head, which is ahead of the `v0.1.0` tag. Since the tag, SSE streaming was wired to the inference engine (`d70cd61`), stop sequences and `finish_reason: length` were implemented, and a KV prefix-cache correctness bug was fixed (`ad92b78`). The "What's Included" and "Known Limitations" sections below reflect the current head, not the tagged commit.

---

## What's Included

### Request Grouping
`BatchAssembler` collects concurrent requests into groups keyed by `(model, sequence-length bin)` and releases a group when it reaches `maxBatchSize` (8) or `maxWaitMs` (100 ms) elapses. This amortises model acquisition and scheduling across the group. It is **not yet a tensor-level batch**: `MLXInferenceEngine.batchedInfer` acquires the model container once and then runs the group's requests sequentially. Measured throughput is consequently flat across concurrency — 36.5 tok/s at C=1 versus 39.5 tok/s at C=16 (Llama-3.2-3B-bf16, Apple M1 Pro; see `benchmarks/load-test-results.json`).

### KV Cache Prefix Sharing
`KVPrefixCache` indexes prompt token prefixes in a trie and restores cached KV state for identical prefixes (common with shared system prompts), feeding only the suffix through the token iterator. Hit rate on the benchmark workload is 79% with 70.1% of prompt tokens covered — but the last end-to-end measurement found a **0.1%** wall-clock prefill reduction, not the ~87% the token-savings model predicts. The plumbing is in place; the payoff is not yet realised. See `benchmarks/kv-cache-hits.json`.

### Multi-Model LRU Pool
`ModelPool` maintains a resident set of loaded models bounded by a configurable capacity (`MAX_MODELS`, default 3). The least-recently-used model is evicted when the pool is full. Models are fetched from the Hugging Face Hub by repo ID and load lazily on first request; a 7B 4-bit model cold-starts in 3–8 seconds on M-series hardware.

### SSE Streaming
Requests with `"stream": true` receive a real per-token Server-Sent Event stream. `MLXInferenceEngine.makeStreamHandler()` drives `TokenIterator` a token at a time and yields each decoded fragment through an `AsyncThrowingStream`, which `ChatRouter.sseResponse()` emits as `ChatCompletionChunk` events over `text/event-stream` (chunked transfer, `Cache-Control: no-cache`), terminated by `data: [DONE]`. The stream path uses the same KV-prefix-cache lookup as the buffered path, and carries the correct `finish_reason` (`stop` or `length`) in the final chunk.

### API Key Authentication
`AuthMiddleware` optionally enforces bearer-token authentication. Set `API_KEY_FILE` to a path containing one token per line; unset it to allow unauthenticated access. All other middleware runs regardless.

### OpenAI-Compatible HTTP API
`ChatRouter` exposes `/v1/chat/completions`, plus `/v1/models` and `/health` on the server. Request and response shapes follow the OpenAI schema so existing clients work without modification. `/v1/models` lists the models **currently resident in the pool**, not everything available on the Hub.

---

## Minimum Hardware

| Requirement | Minimum |
|---|---|
| Apple Silicon | **M1 or later** (MLX requires Apple Silicon; Intel Macs are not supported) |
| macOS | 14.0 Sonoma |
| RAM | 8 GB unified memory (16 GB recommended for 7B models) |
| Swift toolchain | Swift 6.0 |

---

## Known Limitations

- **Inference is serialized.** `MLXInferenceEngine` is an actor and processes one request at a time, so aggregate throughput does not increase with concurrency; additional load turns into queue-wait latency. Real batched inference and continuous batching are the top roadmap items.
- **KV prefix sharing does not yet reduce wall-clock prefill.** The trie hits, the token accounting shows savings, but the last end-to-end measurement showed ~0.1% latency reduction. Not re-measured since the prefix-cache correctness fix in this release.
- **No Ollama (or other server) comparison.** `benchmarks/gateway-vs-ollama.json` records an `HTTP 500` from the Ollama side rather than a number. No cross-server performance claim is supported by this repo.
- **No quantization in-process.** Models must already be in MLX format (`.safetensors` + `config.json`) on the Hugging Face Hub. Use `mlx_lm.convert` or an existing mlx-community repo.
- **Single-node only.** No distributed inference; all layers must fit in one machine's unified memory.
- **No LoRA / adapter support.** Base model weights only in this release.
- **Prefix cache is in-process and ephemeral.** Cache is lost on server restart; no disk persistence.
- **Sequence-length binning is coarse.** Bins are 128 / 256 / 512 / 1024 / 2048 / 4096 tokens and prompt length is estimated from character count, not tokenised.
- **Authentication is all-or-nothing.** No per-model or per-route scoping of API keys.
- **Benchmarks cover one machine and one model family.** Everything in `benchmarks/` was measured on a single Apple M1 Pro against Llama-3.2-3B; real-world numbers will vary by model, quantization level, and concurrent request count.

---

## Installation

```bash
git clone https://github.com/american-code/MLXServingGateway
cd MLXServingGateway
swift build -c release
.build/release/GatewayServer
```

See `README.md` for full setup instructions, model loading behaviour, and the API reference.
