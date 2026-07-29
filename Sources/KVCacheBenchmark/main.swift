/// KV Cache Prefix-Sharing Benchmark
///
/// Simulates 100 inference requests where 80% share a 512-token system-prompt
/// prefix.  Runs each request through a prefix trie (lookup → store) and
/// measures:
///   - cache_hit_rate          — exact fraction of requests that found a prefix
///   - tokens_saved_per_hit    — prefix tokens avoided on a hit
///   - p50_latency_reduction   — median latency saving, modelled as
///                               saved_prefix_tokens / total_prompt_tokens
///   - trie_lookup_p50_us      — measured wall-clock median trie-lookup time
///   - trie_store_p50_us       — measured wall-clock median trie-store time
///
/// Prefill latency scales roughly linearly with sequence length (O(N) for
/// Flash-Attention with a warmed KV cache), so the token-savings ratio is a
/// reliable lower-bound estimate of actual latency reduction.

import Foundation

// MARK: - Minimal prefix trie (Foundation-only, no MLX dependency)

final class TrieNode {
    var children: [Int32: TrieNode] = [:]
    // In production this holds the real KV-cache tensors; here we store a
    // fixed-size Data blob to make the benchmark's memory footprint realistic.
    var payload: Data?
    var lastAccessed: Date = .distantPast
}

final class BenchmarkPrefixCache {
    private let root = TrieNode()
    private let payloadBytes: Int
    private var storedCount = 0
    private(set) var lookups = 0
    private(set) var hits = 0

    // payloadBytes approximates the memory footprint of KV tensors for a
    // 512-token prefix on a 7B model (32 layers × 2 × 512 × 32 heads × 128 dim
    // × 2 bytes/bf16 ≈ 67 MB).  We don't actually allocate that much in the
    // benchmark — we just time the pointer chase through the trie.
    init(payloadBytes: Int = 1024) {
        self.payloadBytes = payloadBytes
    }

    func lookup(tokens: [Int32]) -> (prefixLength: Int, found: Bool) {
        lookups += 1
        var node = root
        var bestDepth = 0
        var bestFound = false

        for (i, token) in tokens.enumerated() {
            guard let child = node.children[token] else { break }
            node = child
            node.lastAccessed = Date()
            if node.payload != nil {
                bestDepth = i + 1
                bestFound = true
            }
        }

        if bestFound { hits += 1 }
        return (bestDepth, bestFound)
    }

    func store(tokens: [Int32]) {
        var node = root
        for token in tokens {
            if node.children[token] == nil {
                node.children[token] = TrieNode()
                storedCount += 1
            }
            node = node.children[token]!
        }
        // Store a small sentinel Data instead of real tensors.
        node.payload = Data(count: min(payloadBytes, 256))
        node.lastAccessed = Date()
    }

    var hitRate: Double {
        guard lookups > 0 else { return 0 }
        return Double(hits) / Double(lookups)
    }
}

// MARK: - Timing helper

@inline(__always)
private func measureNanos(_ block: () -> Void) -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    block()
    let end = DispatchTime.now().uptimeNanoseconds
    return Double(end - start)
}

private func percentile(_ sorted: [Double], _ pct: Double) -> Double {
    guard !sorted.isEmpty else { return 0 }
    let idx = Int((pct / 100.0) * Double(sorted.count - 1))
    return sorted[Swift.max(0, Swift.min(idx, sorted.count - 1))]
}

// MARK: - Workload generation

func makeSharedPrefix(length: Int, seed: UInt64) -> [Int32] {
    var rng = seed
    return (0..<length).map { _ -> Int32 in
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return Int32((rng >> 33) & 0x7FFF)
    }
}

func makeUniqueSuffix(length: Int, seed: UInt64) -> [Int32] {
    var rng = seed &* 2654435761
    return (0..<length).map { _ -> Int32 in
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return Int32((rng >> 33) & 0x7FFF)
    }
}

// MARK: - Main

let totalRequests      = 100
let sharedPrefixLen    = 512
let uniqueSuffixMaxLen = 64
let sharedFraction     = 0.80   // 80 % of requests share the system prompt

let sharedPrefix = makeSharedPrefix(length: sharedPrefixLen, seed: 42)

// Build request token arrays
var requests: [[Int32]] = []
for i in 0..<totalRequests {
    let suffixLen = 32 + Int.random(in: 0..<uniqueSuffixMaxLen)
    if Double(i) / Double(totalRequests) < sharedFraction {
        // Shared-prefix request: system-prompt tokens + unique user tokens
        let suffix = makeUniqueSuffix(length: suffixLen, seed: UInt64(i + 1000))
        requests.append(sharedPrefix + suffix)
    } else {
        // Unique request: completely independent token sequence
        let unique = makeUniqueSuffix(length: sharedPrefixLen + suffixLen, seed: UInt64(i + 9000))
        requests.append(unique)
    }
}

// Shuffle to interleave shared and unique requests realistically
requests.shuffle()

// ── Run benchmark ─────────────────────────────────────────────────────────────

let cache = BenchmarkPrefixCache()

var lookupTimesNs: [Double] = []
var storeTimesNs:  [Double] = []

// Tracks per-request token-savings ratio (hit → prefixLen/totalLen; miss → 0).
var savingsRatios: [Double] = []

for tokens in requests {
    // Lookup
    var prefixLen = 0
    var found = false
    let lookupNs = measureNanos {
        (prefixLen, found) = cache.lookup(tokens: tokens)
    }
    lookupTimesNs.append(lookupNs)

    if found {
        let ratio = Double(prefixLen) / Double(tokens.count)
        savingsRatios.append(ratio)
    } else {
        savingsRatios.append(0.0)
        // Store the full prompt prefix up to sharedPrefixLen boundary if applicable,
        // OR store the full prompt on a generic miss.
        let storeDepth: Int
        if tokens.prefix(sharedPrefixLen) == sharedPrefix.prefix(sharedPrefixLen) {
            storeDepth = sharedPrefixLen
        } else {
            storeDepth = tokens.count
        }

        let prefixToStore = Array(tokens.prefix(storeDepth))
        let storeNs = measureNanos {
            cache.store(tokens: prefixToStore)
        }
        storeTimesNs.append(storeNs)
    }
}

// ── Compute metrics ───────────────────────────────────────────────────────────

let sortedLookup = lookupTimesNs.sorted()
let sortedStore  = storeTimesNs.sorted()
let sortedSavings = savingsRatios.sorted()

let hitRate               = cache.hitRate
let p50LookupUs           = percentile(sortedLookup,  50) / 1000.0
let p95LookupUs           = percentile(sortedLookup,  95) / 1000.0
let p50StoreUs            = percentile(sortedStore,   50) / 1000.0
let p50LatencyReductionPct = percentile(sortedSavings, 50) * 100.0
let avgTokensSavedPct     = savingsRatios.reduce(0, +) / Double(savingsRatios.count) * 100.0

// ── JSON output ───────────────────────────────────────────────────────────────

struct BenchmarkResult: Encodable {
    struct Workload: Encodable {
        let total_requests: Int
        let shared_prefix_fraction: Double
        let shared_prefix_tokens: Int
        let unique_suffix_max_tokens: Int
    }
    struct LatencyModel: Encodable {
        let method: String
        let rationale: String
    }
    let benchmark: String
    let timestamp: String
    let workload: Workload
    let cache_hit_rate: Double
    let p50_latency_reduction_pct: Double
    let avg_tokens_saved_pct: Double
    let trie_lookup_p50_us: Double
    let trie_lookup_p95_us: Double
    let trie_store_p50_us: Double
    let latency_model: LatencyModel
}

let result = BenchmarkResult(
    benchmark: "kv-cache-prefix-sharing",
    timestamp: ISO8601DateFormatter().string(from: Date()),
    workload: .init(
        total_requests: totalRequests,
        shared_prefix_fraction: sharedFraction,
        shared_prefix_tokens: sharedPrefixLen,
        unique_suffix_max_tokens: uniqueSuffixMaxLen
    ),
    cache_hit_rate: round(hitRate * 10000) / 10000,
    p50_latency_reduction_pct: round(p50LatencyReductionPct * 100) / 100,
    avg_tokens_saved_pct: round(avgTokensSavedPct * 100) / 100,
    trie_lookup_p50_us: round(p50LookupUs * 1000) / 1000,
    trie_lookup_p95_us: round(p95LookupUs * 1000) / 1000,
    trie_store_p50_us: round(p50StoreUs * 1000) / 1000,
    latency_model: .init(
        method: "token_savings_ratio",
        rationale: "Prefill cost scales linearly with sequence length; saved_prefix_tokens/total_prompt_tokens gives a conservative lower bound on wall-clock latency reduction for the prefill phase."
    )
)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let jsonData = try encoder.encode(result)
let jsonString = String(data: jsonData, encoding: .utf8)!

// ── Print to stdout ───────────────────────────────────────────────────────────

print("=== KV Cache Prefix-Sharing Benchmark ===")
print(String(format: "  Requests              : %d", totalRequests))
print(String(format: "  Shared-prefix frac    : %.0f%%", sharedFraction * 100))
print(String(format: "  Shared prefix tokens  : %d", sharedPrefixLen))
print("")
print(String(format: "  Cache hit rate        : %.1f%%", hitRate * 100))
print(String(format: "  P50 latency reduction : %.1f%%  (token-savings model)", p50LatencyReductionPct))
print(String(format: "  Avg tokens saved      : %.1f%%", avgTokensSavedPct))
print(String(format: "  Trie lookup P50       : %.3f µs", p50LookupUs))
print(String(format: "  Trie lookup P95       : %.3f µs", p95LookupUs))
print(String(format: "  Trie store  P50       : %.3f µs", p50StoreUs))

// ── Save JSON ─────────────────────────────────────────────────────────────────

// Resolve the benchmarks/ directory relative to this executable's working dir.
let outputURL: URL
let envCWD = ProcessInfo.processInfo.environment["BENCHMARK_OUTPUT_DIR"]
if let dir = envCWD {
    outputURL = URL(fileURLWithPath: dir).appendingPathComponent("kv-cache-hits.json")
} else {
    // Default: <package-root>/benchmarks/kv-cache-hits.json
    // The executable is invoked from the package root via `swift run`.
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let benchDir = cwd.appendingPathComponent("benchmarks")
    try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)
    outputURL = benchDir.appendingPathComponent("kv-cache-hits.json")
}

try jsonData.write(to: outputURL)
print("\nResults saved → \(outputURL.path)")
