# MLX Serving Gateway v0.1.0

**2026-07-29**

First tagged release. This is a working inference server that runs quantized MLX models on Apple Silicon and exposes an OpenAI-compatible REST API. All core subsystems are present and integrated.

---

## What's Included

### Request Batching
`BatchAssembler` collects concurrent requests into micro-batches before dispatch. Configurable `batchSize` and `waitDeadline` let you trade latency against throughput. On a loaded M2 Ultra, batching 4–8 requests roughly halves per-token latency versus serial execution.

### KV Cache Prefix Sharing
`KVPrefixCache` hashes the token prefix of each prompt and reuses any cached KV state for identical prefixes (common with shared system prompts). Reduces redundant attention computation across requests that share a long preamble.

### Multi-Model LRU Pool
`ModelPool` maintains a resident set of loaded models bounded by a configurable capacity. The least-recently-used model is evicted when the pool is full. Models load lazily on first request; a 7B 4-bit model cold-starts in 3–8 seconds on M-series hardware.

### SSE Streaming
`MLXInferenceEngine` runs autoregressive decoding and streams tokens back via Server-Sent Events as they are generated. Clients receive `data: {"choices": [...]}` lines in real time, compatible with the OpenAI streaming protocol.

### API Key Authentication
`AuthMiddleware` optionally enforces bearer-token authentication. Set `API_KEY_FILE` to a path containing one token per line; unset it to allow unauthenticated access. All other middleware runs regardless.

### OpenAI-Compatible HTTP API
`ChatRouter` exposes `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, and `/health`. Request and response shapes follow the OpenAI schema so existing clients work without modification.

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

- **No quantization in-process.** Models must be pre-converted to MLX format (`.safetensors` + `config.json`). Use `mlx_lm.convert` or download from mlx-community on Hugging Face.
- **Single-node only.** No distributed inference; all layers must fit in one machine's unified memory.
- **No LoRA / adapter support.** Base model weights only in this release.
- **Prefix cache is in-process and ephemeral.** Cache is lost on server restart; no disk persistence.
- **Batch assembler does not handle variable sequence lengths optimally.** Long prompts mixed with short ones in the same batch may waste compute on padding.
- **Authentication is all-or-nothing.** No per-model or per-route scoping of API keys.
- **No automatic model discovery refresh.** `~/Models` is scanned once at startup; models added afterward require a server restart.
- **Benchmarks are synthetic.** The included `posts/mlx-serving-gateway-benchmark.md` reports timings under controlled load; real-world numbers will vary by model, quantization level, and concurrent request count.

---

## Installation

```bash
git clone <repo-url>
cd MLXServingGateway
swift build -c release
.build/release/GatewayServer
```

See `README.md` for full setup instructions, model placement conventions, and API reference.
