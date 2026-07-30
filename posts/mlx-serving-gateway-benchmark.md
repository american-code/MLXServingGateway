# MLX Serving Gateway: Throughput, Latency, and KV Cache Benchmarks on Apple Silicon

*July 29, 2026 · Jacob Melton*

---

Apple Silicon's unified memory architecture has always promised that local LLM inference could be both fast and practical. The problem has been the software stack: most inference servers were designed for NVIDIA GPUs and ported to Metal as an afterthought, leaving significant performance on the table. [MLX Serving Gateway](https://github.com/your-org/MLXServingGateway) is a Swift-native server built from the start around MLX and Apple's accelerator stack. This post covers the motivation, how it was benchmarked, and what we found.

---

## Why Another Local Inference Server?

Ollama is the de facto standard for running models locally on a Mac. It works, it's polished, and the Docker-inspired model management is genuinely nice. But it has two architectural properties that limit throughput under concurrent load:

1. **No request batching.** Concurrent requests queue up and are dispatched serially. Tensor cores sit idle while one request waits for the previous one to finish.
2. **No KV prefix sharing.** Every request pays the full prefill cost from token zero, even when 512 of those tokens are an identical system prompt shared with every other request in flight.

MLX Serving Gateway addresses both. A `BatchAssembler` groups concurrent requests by model and sequence-length bin, dispatching them as a single batched forward pass up to `maxBatchSize` (default 8). A prefix trie tracks KV blocks for common prompt prefixes, so the second request with the same system prompt skips the prefill work for those tokens entirely.

The rest of the stack — written in Swift 6 with structured concurrency, served through Hummingbird 2, running on MLX Swift — stays entirely inside unified memory. No copies between CPU and GPU address spaces; no JNI boundary.

---

## Methodology

All benchmarks were run on a **MacBookPro18,3 (Apple M1 Pro, 16 GB unified memory)** under macOS 15.4. Model used:

| Model | Format | Disk size |
|-------|--------|-----------|
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 4-bit quantized safetensors | 1.8 GB |

**Throughput/latency test:** Concurrency levels of 1, 4, 16, and 32 clients, each sending a 128-token prompt requesting 64 tokens of completion. Requests per run: 20 at C=1 and C=4, 48 at C=16, 96 at C=32. Throughput is aggregate generated tokens divided by wall-clock time. Latency (P50/P95) is end-to-end from request dispatch to final token received.

**KV cache test:** 100 requests with an 80% hit-rate workload — 80 requests share a 512-token system prompt prefix, 20 requests use unique prompts. Unique per-request suffixes are up to 64 tokens. This reflects a realistic RAG or customer-support scenario where every user shares the same large system context.

> All numbers are reproducible: run the included load-test script for throughput/latency and `swift run KVCacheBenchmark` for the cache analysis.

---

## Throughput Results

Throughput scales with concurrency until the `BatchAssembler` fills its maximum batch size, then plateaus.

<svg viewBox="0 0 640 310" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, 'Cascadia Code', monospace" font-size="13">
  <!-- Background -->
  <rect width="640" height="310" fill="#0f1117" rx="8"/>

  <!-- Title -->
  <text x="320" y="28" fill="#e2e8f0" font-size="15" font-weight="bold" text-anchor="middle">Aggregate Throughput by Concurrency — Llama-3.2-3B-4bit (tokens/sec)</text>

  <!-- Y-axis gridlines: chart area y=50 (150 tok/s) to y=250 (0 tok/s) -->
  <line x1="110" y1="50"  x2="580" y2="50"  stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="54"  fill="#718096" text-anchor="end">150</text>
  <line x1="110" y1="117" x2="580" y2="117" stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="121" fill="#718096" text-anchor="end">100</text>
  <line x1="110" y1="183" x2="580" y2="183" stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="187" fill="#718096" text-anchor="end">50</text>
  <line x1="110" y1="250" x2="580" y2="250" stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="254" fill="#718096" text-anchor="end">0</text>

  <!-- Y axis -->
  <line x1="110" y1="50" x2="110" y2="250" stroke="#4a5568" stroke-width="1.5"/>

  <!-- C=1: 100.0 tok/s → h=133, top=117 -->
  <rect x="134" y="117" width="70" height="133" fill="#4299e1" rx="3"/>
  <text x="169" y="111" fill="#4299e1" font-size="12" text-anchor="middle" font-weight="bold">100.0</text>
  <text x="169" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=1</text>

  <!-- C=4: 122.3 tok/s → h=163, top=87 -->
  <rect x="251" y="87" width="70" height="163" fill="#4299e1" rx="3"/>
  <text x="286" y="81" fill="#4299e1" font-size="12" text-anchor="middle" font-weight="bold">122.3</text>
  <text x="286" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=4</text>

  <!-- C=16: 130.5 tok/s → h=174, top=76 -->
  <rect x="369" y="76" width="70" height="174" fill="#48bb78" rx="3"/>
  <text x="404" y="70" fill="#48bb78" font-size="12" text-anchor="middle" font-weight="bold">130.5</text>
  <text x="404" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=16</text>

  <!-- C=32: 130.1 tok/s → h=173, top=77 -->
  <rect x="486" y="77" width="70" height="173" fill="#48bb78" rx="3"/>
  <text x="521" y="71" fill="#48bb78" font-size="12" text-anchor="middle" font-weight="bold">130.1</text>
  <text x="521" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=32</text>

  <!-- Legend -->
  <rect x="170" y="287" width="12" height="12" fill="#4299e1" rx="2"/>
  <text x="187" y="298" fill="#a0aec0" font-size="11">Scaling</text>
  <rect x="265" y="287" width="12" height="12" fill="#48bb78" rx="2"/>
  <text x="282" y="298" fill="#a0aec0" font-size="11">Saturated (batch full)</text>
</svg>

At C=1 the gateway runs single-request inference: **100.0 tok/s**. Adding concurrency lets the `BatchAssembler` group requests into batches of up to 8, pushing throughput to **130.5 tok/s** at C=16 — a ~30% lift. Throughput is flat from C=16 to C=32, confirming the batch assembler is fully saturated: additional requests queue behind the current batch rather than expanding it.

---

## Request Latency

Static batching trades latency for throughput. Each request waits at the batch boundary until `maxBatchSize` (8) is reached or the deadline fires, then the entire batch is dispatched together.

| Concurrency | P50 | P95 | Mean |
|-------------|-----|-----|------|
| 1  | 622 ms | 839 ms | 640 ms |
| 4  | 2,088 ms | 2,119 ms | 2,093 ms |
| 16 | 7,848 ms | 7,855 ms | 7,194 ms |
| 32 | 15,732 ms | 15,777 ms | 13,778 ms |

Two patterns stand out. First, the P50–P95 spread is very tight (≤220 ms at C=1, ≤9 ms at C=32): batched requests exit in lock-step, producing highly predictable latency within each concurrency tier. Second, absolute latency grows roughly linearly: at C=32 each request waits behind approximately four full batches of 8, yielding about 16× the C=1 latency.

This is the expected trade-off for static batching. **Continuous batching** — interleaving new requests into a running decode loop rather than waiting for a batch boundary — would substantially reduce the queue-wait and is the primary item on the roadmap.

---

## KV Cache Prefix Sharing Analysis

The KV cache numbers came from running `swift run KVCacheBenchmark` against the workload described above. Results:

| Metric | Result |
|--------|--------|
| Cache hit rate | **79%** |
| P50 prefill latency reduction | **87.2%** |
| Average tokens saved per request | **70.1%** |
| Trie lookup P50 | 53.3 µs |
| Trie lookup P95 | 82.7 µs |
| Trie store P50 | 231.7 µs |

A 79% hit rate in a workload where 80% of requests share a prefix is the expected ceiling; the 1% gap is cold-start misses on the first appearance of the shared prefix. The **87.2% prefill latency reduction** on hits is close to what the token math predicts: `512 prefix tokens / (512 + 32 mean suffix tokens) ≈ 94%` of the total prompt is skipped, and prefill cost scales roughly linearly with token count.

The trie itself is cheap. A lookup that resolves in 53 µs at P50 adds essentially nothing to the response path — well under the ~5 ms of network overhead a local HTTP call carries. Even a trie store at 231.7 µs is negligible relative to a multi-hundred-millisecond prefill.

**When does prefix sharing matter most?**

- **System-prompt-heavy workloads**: Customer support bots, RAG pipelines, and tool-use agents often carry 200–2 000 token system contexts that are identical across every call in a session.
- **Multi-turn conversations**: The assistant context grows with each turn. Without prefix sharing, each follow-up message re-processes the entire history. With it, only the new user message is prefilled.
- **Single-host, high-RPS serving**: The benefit compounds with concurrency. At 1 RPS the savings are modest; at 16+ RPS, 79% of prefill work vanishes.

---

## Getting Started

```bash
# 1. Clone and build
git clone https://github.com/your-org/MLXServingGateway
cd MLXServingGateway
swift build -c release

# 2. Download a model
pip install mlx-lm
python -m mlx_lm.convert \
  --hf-path mlx-community/Llama-3.2-3B-Instruct-4bit \
  --mlx-path ~/Models/mlx-community/Llama-3.2-3B-Instruct-4bit

# 3. Start the server
.build/release/GatewayServer

# 4. Send a request (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

The server is OpenAI API-compatible, so any client that targets `api.openai.com` can be pointed at `localhost:8080` instead — including the Python and Node SDKs, LangChain, LlamaIndex, and Open WebUI.

For auth-gated deployments (shared development machines, team homelab servers):

```bash
echo "sk-local-abc123" > ~/.config/mlx-gateway/keys.txt
API_KEY_FILE=~/.config/mlx-gateway/keys.txt .build/release/GatewayServer
```

To reproduce the KV cache benchmark:

```bash
swift run KVCacheBenchmark
# Results saved to benchmarks/kv-cache-hits.json
```

---

## What's Next

The current architecture processes one batch at a time per model. The obvious next steps are **continuous batching** (interleaving new requests into a running decode loop) and **speculative decoding** (using a small draft model to propose tokens that the main model verifies in bulk). Both would shrink the queue-wait latency and improve throughput at low concurrency where the batch assembler's wait window adds unnecessary delay.

On the hardware side, M3 Pro/Max and M4 Pro/Max offer higher memory bandwidth that MLX can exploit directly — the gateway is purely software-bounded and will scale with the chip.

The benchmark harness and all source code are MIT-licensed. Issues and pull requests welcome.

---

*Hardware: MacBookPro18,3, Apple M1 Pro, 16 GB unified memory, macOS 15.4. MLX Swift 0.21. All benchmarks run with the machine otherwise idle.*
