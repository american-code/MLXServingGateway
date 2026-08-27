# MLX Serving Gateway: Throughput, Latency, and KV Cache Benchmarks on Apple Silicon

*July 29, 2026 · revised August 27, 2026 · Jacob Melton*

---

Apple Silicon's unified memory architecture has always promised that local LLM inference could be both fast and practical. The problem has been the software stack: most inference servers were designed for NVIDIA GPUs and ported to Metal as an afterthought. [MLX Serving Gateway](https://github.com/american-code/MLXServingGateway) is a Swift-native server built from the start around MLX and Apple's accelerator stack. This post covers the motivation, how it was benchmarked, and what we found — including the parts that did not work yet.


---

## Why Another Local Inference Server?

The design bet is that a serving layer written natively against MLX — rather than ported onto it — can exploit two things that matter under concurrent load:

1. **Request grouping.** Concurrent requests arriving within a short window can be collected and handed to the engine together, amortising model acquisition and scheduling overhead instead of paying it per request.
2. **KV prefix sharing.** When 512 tokens of a prompt are an identical system preamble shared with every other in-flight request, recomputing those key/value tensors per request is waste.

MLX Serving Gateway implements the plumbing for both. A `BatchAssembler` actor groups concurrent requests by `(model, sequence-length bin)` and releases a group when it reaches `maxBatchSize` (8) or `maxWaitMs` (100 ms) elapses. A prefix trie tracks KV state for common prompt prefixes, and the inference path restores that state and feeds only the suffix through the token iterator.

**What the plumbing does not yet do is convert into wall-clock wins**, and the measurements below are mostly about that gap. Two structural reasons, both visible in the source:

- `MLXInferenceEngine` is an `actor`, and `batchedInfer` acquires the model container once and then iterates the group **sequentially**. There is no batched forward pass — no single matmul spanning eight requests. A "batch" today is a scheduling group, not a tensor-level batch.
- The prefix trie's savings are real in token accounting but, at the time of the KV measurement run, did not show up as reduced prefill wall-clock.

The rest of the stack — Swift 6 with structured concurrency, served through Hummingbird 2, running on MLX Swift — stays entirely inside unified memory. No copies between CPU and GPU address spaces; no JNI boundary. That part works as intended.

### On the Ollama comparison

Ollama is the obvious baseline, and a comparison harness exists (`benchmarks/gateway-vs-ollama.json`). **It has no Ollama data in it.** Every attempt against the local Ollama endpoint returned `HTTP 500`, the run was never repeated, and the file records the error rather than a substitute number. The gateway side of that file is real — 116.74 tok/s mean over 8 runs, detailed below — but there is nothing to compare it against.

So: **no claim is made here about MLX Serving Gateway being faster or slower than Ollama.** That comparison is not yet measured.

---

## Methodology

All benchmarks ran on a **MacBookPro18,3 (Apple M1 Pro, 16 GB unified memory)**, machine otherwise idle, against MLX Swift 0.31.4 (via `mlx-swift-lm` 3.31.4).

Three separate runs are reported, and they do **not** all use the same model. This matters — the throughput numbers are not comparable across sections:

| Run | Model | Weights | Date |
|-----|-------|---------|------|
| Concurrency / latency sweep | `mlx-community/Llama-3.2-3B-bf16` | bf16, unquantized | 2026-07-30 |
| Single-request throughput | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 4-bit quantized | 2026-07-30 |
| KV prefix-sharing | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 4-bit quantized | 2026-07-30 |

**Concurrency/latency sweep** (`benchmarks/load-test-results.json`): concurrency levels 1, 4, 16, and 32, each request carrying a 130-token prompt and requesting 64 output tokens. Request counts were 10 at C=1, 16 at C=4, 24 at C=16, and 32 at C=32. Throughput is aggregate generated tokens over wall-clock time; latency is end-to-end from dispatch to final token. The bf16 base model has no chat template, so the Llama-3.2-Instruct template was injected for prompt formatting. Each request carries a unique `[#NNNN]` prefix, which forces the KV prefix cache to miss — this sweep measures the cold path deliberately.

**Single-request throughput** (`benchmarks/gateway-vs-ollama.json`): 8 timed runs after 2 warmups, 128 output tokens, 4-bit quantized weights.

**KV prefix-sharing** (`benchmarks/kv-cache-hits.json`): two parts. (a) A Foundation-only trie simulation — `swift run KVCacheBenchmark` — over 100 requests where 80 share a 512-token prefix and 20 are unique, measuring hit rate and trie lookup/store cost. (b) A separate end-to-end HTTP run, 100 requests per phase, `max_tokens=1` to isolate prefill: phase 1 with unique system prompts, phase 2 with one shared 512-token system prompt.

> Note that `swift run KVCacheBenchmark` measures the trie in isolation and **overwrites** `benchmarks/kv-cache-hits.json` with trie-only metrics — including a *modeled* latency reduction, not a measured one. The measured wall-clock block in the committed file came from the separate end-to-end run described in (b). Don't confuse the two; the difference between them is the most interesting result in this post.

---

## Throughput Results

Throughput is essentially flat across a 32× range of concurrency.

<svg viewBox="0 0 640 310" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, 'Cascadia Code', monospace" font-size="13">
  <!-- Background -->
  <rect width="640" height="310" fill="#0f1117" rx="8"/>

  <!-- Title -->
  <text x="320" y="28" fill="#e2e8f0" font-size="15" font-weight="bold" text-anchor="middle">Aggregate Throughput by Concurrency — Llama-3.2-3B-bf16 (tokens/sec)</text>

  <!-- Y-axis gridlines: chart area y=50 (50 tok/s) to y=250 (0 tok/s), 4 px per tok/s -->
  <line x1="110" y1="50"  x2="580" y2="50"  stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="54"  fill="#718096" text-anchor="end">50</text>
  <line x1="110" y1="90"  x2="580" y2="90"  stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="94"  fill="#718096" text-anchor="end">40</text>
  <line x1="110" y1="130" x2="580" y2="130" stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="134" fill="#718096" text-anchor="end">30</text>
  <line x1="110" y1="170" x2="580" y2="170" stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="174" fill="#718096" text-anchor="end">20</text>
  <line x1="110" y1="210" x2="580" y2="210" stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="214" fill="#718096" text-anchor="end">10</text>
  <line x1="110" y1="250" x2="580" y2="250" stroke="#2d3748" stroke-width="1"/>
  <text x="100" y="254" fill="#718096" text-anchor="end">0</text>

  <!-- Y axis -->
  <line x1="110" y1="50" x2="110" y2="250" stroke="#4a5568" stroke-width="1.5"/>

  <!-- C=1: 36.5 tok/s → h=146, top=104 -->
  <rect x="134" y="104" width="70" height="146" fill="#4299e1" rx="3"/>
  <text x="169" y="98" fill="#4299e1" font-size="12" text-anchor="middle" font-weight="bold">36.5</text>
  <text x="169" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=1</text>

  <!-- C=4: 38.8 tok/s → h=155, top=95 -->
  <rect x="251" y="95" width="70" height="155" fill="#4299e1" rx="3"/>
  <text x="286" y="89" fill="#4299e1" font-size="12" text-anchor="middle" font-weight="bold">38.8</text>
  <text x="286" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=4</text>

  <!-- C=16: 39.5 tok/s → h=158, top=92 -->
  <rect x="369" y="92" width="70" height="158" fill="#4299e1" rx="3"/>
  <text x="404" y="86" fill="#4299e1" font-size="12" text-anchor="middle" font-weight="bold">39.5</text>
  <text x="404" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=16</text>

  <!-- C=32: 39.4 tok/s → h=158, top=92 -->
  <rect x="486" y="92" width="70" height="158" fill="#4299e1" rx="3"/>
  <text x="521" y="86" fill="#4299e1" font-size="12" text-anchor="middle" font-weight="bold">39.4</text>
  <text x="521" y="270" fill="#a0aec0" font-size="11" text-anchor="middle">C=32</text>

  <!-- Footnote -->
  <text x="320" y="296" fill="#718096" font-size="11" text-anchor="middle">Flat within 8.2% across 32x concurrency — inference is serialized, so grouping does not add throughput</text>
</svg>

At C=1 the gateway sustains **36.5 tok/s** on bf16 weights. Raising concurrency to 16 moves that to **39.5 tok/s** — an **8.2% change**, and C=32 comes back down fractionally to 39.4 tok/s.

That is not batching working. That is batching not being there yet. `MLXInferenceEngine` is an actor and `batchedInfer` walks its group sequentially inside a single `container.perform`, so the GPU sees one request's forward pass at a time no matter how many clients are connected. The small gain from C=1 to C=16 is amortised per-request overhead — model acquisition, tokenisation scheduling, actor hops — not parallel compute. The `benchmarks/load-test-results.json` notes say this plainly: *"MLXInferenceEngine actor serializes all inference, so throughput (tok/s) is stable across concurrency levels while latency scales with batch depth."*

For a sense of what the same hardware does with quantized weights: the single-request 4-bit run reaches **116.74 tok/s mean** (P50 116.84, P50 latency 1,095 ms, 1,933 MB resident). Roughly 3.2× the bf16 rate, which is what you would expect from 4-bit weights on a bandwidth-bound decode loop. It is a different model configuration and does not belong on the same axis as the sweep above.

---

## Request Latency

With inference serialized, latency is the metric that actually moves.

| Concurrency | P50 | P95 | Mean | Requests |
|-------------|-----|-----|------|----------|
| 1  | 1,754 ms | 1,763 ms | 1,751 ms | 10 |
| 4  | 6,600 ms | 6,612 ms | 6,599 ms | 16 |
| 16 | 25,894 ms | 25,912 ms | 21,591 ms | 24 |
| 32 | 32,455 ms | 51,927 ms | 32,458 ms | 32 |

Two things stand out.

**Latency scales with queue depth.** P50 goes 3.8× from C=1 to C=4 and 14.8× from C=1 to C=16 — close to the 4× and 16× that pure serialization predicts. Each request waits for every request ahead of it to finish generating all 64 of its tokens, because nothing overlaps. This is the direct latency cost of the serialization described above; it is not the classic static-batching trade (accepting queue-wait to gain throughput), because there is no throughput gain being bought. At C=32 the P50 lands at 32,455 ms rather than the ~56,000 ms a strict 32× would imply, but the P95 of 51,927 ms sits right about where that projection does — the run has 32 requests draining in order, so the distribution is a ramp and which percentile you read matters more than at lower concurrency.

**The tail opens up at C=32.** Through C=16 the P50–P95 spread is under 20 ms: the upper half of requests exits in a tight cluster, because they are drained in order at a near-constant rate. (At C=16 the mean, 21,591 ms, sits 4.3 s *below* the P50 — the handful of early-served requests finish much sooner and pull the average down, while the rest bunch at the back.) At C=32 the spread jumps to 19.5 seconds (32,455 ms P50 vs 51,927 ms P95) and even the tight upper cluster disappears: with a 32-deep queue the last-served request waits behind all 31 others, so the ramp spans the full run.

The fix for both is the same and is not a tuning knob: the engine needs a genuinely batched forward pass, or continuous batching that interleaves new requests into a running decode loop. Until then, the gateway is a correct and convenient OpenAI-compatible front end for a **single-stream** MLX backend, and should be sized accordingly.

---

## KV Cache Prefix Sharing Analysis

This is where prediction and measurement disagree most sharply, so both are reported.

### Trie behaviour (simulation)

| Metric | Result |
|--------|--------|
| Cache hit rate | **79%** |
| Average tokens saved per request | **70.1%** |
| Trie lookup P50 | 53.3 µs |
| Trie lookup P95 | 82.7 µs |
| Trie store P50 | 231.7 µs |

A 79% hit rate on a workload where 80% of requests share a prefix is the expected ceiling; the 1% gap is the cold-start miss on the shared prefix's first appearance. The trie itself is cheap: a 53 µs P50 lookup is nothing next to a multi-hundred-millisecond prefill, and even a 232 µs store is negligible.

From those token savings, the harness *models* a P50 prefill latency reduction of **87.2%**, on the reasoning that prefill cost scales linearly with sequence length. That model is stated in the JSON as a model (`latency_model.p50_reduction_pct_modeled`); it is not a measurement.

### End-to-end wall clock (measurement)

| Phase (100 requests each, `max_tokens=1`) | P50 | Mean | P95 |
|---|---|---|---|
| Unique system prompts (no cache benefit) | 585.9 ms | 584.9 ms | 594.6 ms |
| Identical 512-token system prompt (cache warm) | 585.3 ms | 583.1 ms | 597.3 ms |
| **Measured reduction** | **0.1%** | — | **−0.45%** |

**Measured P50 prefill reduction: 0.1%. Predicted: 87.2%.**

The 512 shared tokens are found by the trie, and the token accounting says 70.1% of prompt tokens were avoidable — but the wall clock did not move. As the benchmark's own note records, the prefix trie tracked the token overlap without the saved KV state actually displacing GPU prefill work on that run.

One caveat worth stating rather than burying: this measurement predates commit `ad92b78`, which fixed a real correctness bug in the prefix-cache store path (an erroneous `trim(1)` left the trie key covering N tokens while the stored KV state covered N−1). That fix changes the cache's behaviour. **The KV numbers above have not been re-measured since**, so treat 0.1% as the last honest measurement and not as a settled verdict on the current code. Re-running this benchmark is the top item on the list below.

### Where prefix sharing would pay off

The workload shapes that motivate the feature are unchanged even though the current payoff is zero:

- **System-prompt-heavy workloads**: customer-support bots, RAG pipelines, and tool-use agents routinely carry 200–2,000 token system contexts identical across every call.
- **Multi-turn conversations**: without prefix sharing, each follow-up re-processes the entire history; with it, only the new turn needs prefilling.
- **Single-host serving**: the savings scale with how much of the prompt is shared, which in these workloads is most of it.

All of that is the argument for finishing the feature, not a description of what it currently delivers.

---

## Getting Started

```bash
# 1. Clone and build
git clone https://github.com/american-code/MLXServingGateway
cd MLXServingGateway
swift build -c release

# 2. Start the server
.build/release/GatewayServer

# 3. Send a request (OpenAI-compatible).
#    Models are fetched from the Hugging Face Hub by repo ID on first use —
#    no pre-conversion or local model directory required.
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

The server is OpenAI API-compatible, so any client that targets `api.openai.com` can be pointed at `localhost:8080` instead — including the Python and Node SDKs, LangChain, LlamaIndex, and Open WebUI. Streaming works: `"stream": true` returns `text/event-stream` with a real per-token SSE chunk sequence terminated by `data: [DONE]`.

For auth-gated deployments (shared development machines, team homelab servers):

```bash
echo "gw-local-key-1" > ~/.config/mlx-gateway/keys.txt
API_KEY_FILE=~/.config/mlx-gateway/keys.txt .build/release/GatewayServer
```

To re-run the trie microbenchmark:

```bash
swift run KVCacheBenchmark
# Overwrites benchmarks/kv-cache-hits.json with trie-only metrics,
# including a MODELED latency reduction. Back the file up first if you
# want to keep the committed end-to-end measurement block.
```

---

## What's Next

In priority order, based on what the numbers above actually show:

1. **Make the KV prefix cache pay off.** Re-measure end-to-end prefill savings on top of the `ad92b78` correctness fix, and if the wall clock still does not move, find where the restored KV state stops displacing GPU work.
2. **Real batched inference.** `batchedInfer` currently iterates its group sequentially. A genuine batched forward pass is the prerequisite for concurrency doing anything for throughput.
3. **Continuous batching.** Interleaving new requests into a running decode loop, rather than draining a queue in order, is what fixes the C=32 latency ramp.
4. **Speculative decoding.** A small draft model proposing tokens for bulk verification would help the single-stream case that the gateway is currently good at.

On the hardware side, M3 Pro/Max and M4 Pro/Max offer higher memory bandwidth that MLX can exploit directly — but on this evidence the gateway is bounded by its own scheduling, not by the chip, and that has to be fixed first.

The benchmark harness and all source code are MIT-licensed. Issues and pull requests welcome.

---

*Hardware: MacBookPro18,3, Apple M1 Pro, 16 GB unified memory. MLX Swift 0.31.4 via mlx-swift-lm 3.31.4, Swift 6.0, Hummingbird 2. All benchmarks run with the machine otherwise idle. Raw data: `benchmarks/load-test-results.json`, `benchmarks/kv-cache-hits.json`, `benchmarks/gateway-vs-ollama.json`.*
