import Foundation
import MLX
import MLXLMCommon

// Snapshot of KV state for one attention layer: [keys, values].
// Marked @unchecked Sendable because MLXArray is backed by GPU memory that is
// safe to share across isolation boundaries once eval'd.
public struct KVCacheEntry: @unchecked Sendable {
    /// Per-layer KV state: `layerStates[i] == [keys, values]` for layer i.
    public let layerStates: [[MLXArray]]
    /// Number of tokens whose KV state is captured here.
    public let tokenCount: Int
}

// MARK: - Trie

private final class TrieNode {
    var children: [Int32: TrieNode] = [:]
    var entry: KVCacheEntry?
    var lastAccessed: Date = .distantPast
}

// MARK: - KVPrefixCache

/// Actor that maintains a prefix trie mapping token sequences to KV cache snapshots.
///
/// On a lookup hit the caller receives a `KVCacheEntry` covering the longest
/// matching prefix. The caller should restore KV caches from that entry and
/// feed only the remaining suffix tokens to the model — skipping prefill for
/// the cached portion entirely.
public actor KVPrefixCache {
    private let root = TrieNode()
    private let maxEntries: Int
    private var storedCount = 0

    private var totalLookups = 0
    private var totalHits = 0

    public init(maxEntries: Int = 500) {
        self.maxEntries = maxEntries
    }

    // MARK: Lookup

    /// Returns the longest prefix of `tokens` whose KV state is cached.
    ///
    /// - Returns: `(prefixLength, entry)` where `prefixLength == 0` on miss.
    public func lookup(tokens: [Int32]) -> (prefixLength: Int, entry: KVCacheEntry?) {
        totalLookups += 1
        var node = root
        var bestDepth = 0
        var bestEntry: KVCacheEntry?

        for (i, token) in tokens.enumerated() {
            guard let child = node.children[token] else { break }
            node = child
            node.lastAccessed = Date()
            if let e = node.entry {
                bestDepth = i + 1
                bestEntry = e
            }
        }

        if bestEntry != nil { totalHits += 1 }
        return (bestDepth, bestEntry)
    }

    // MARK: Store

    /// Cache the KV state for the given token prefix.
    ///
    /// `layerStates[i]` must be `[keys, values]` for layer i, already eval'd on
    /// the GPU so the computation graph is fully materialised before storage.
    public func store(tokens: [Int32], layerStates: [[MLXArray]]) {
        guard !tokens.isEmpty else { return }

        var node = root
        for token in tokens {
            if node.children[token] == nil {
                node.children[token] = TrieNode()
                storedCount += 1
            }
            node = node.children[token]!
        }

        if node.entry == nil { /* only count genuinely new leaf entries */ }
        node.entry = KVCacheEntry(layerStates: layerStates, tokenCount: tokens.count)
        node.lastAccessed = Date()

        if storedCount > maxEntries { evictOldestLeaf(root) }
    }

    // MARK: Stats

    public var hitRate: Double {
        guard totalLookups > 0 else { return 0 }
        return Double(totalHits) / Double(totalLookups)
    }

    public var stats: (lookups: Int, hits: Int, stored: Int) {
        (totalLookups, totalHits, storedCount)
    }

    // MARK: Eviction (simple LRU-like: evict oldest leaf)

    @discardableResult
    private func evictOldestLeaf(_ node: TrieNode) -> Bool {
        var oldestDate: Date = .distantFuture
        var oldestKey: Int32?

        for (key, child) in node.children {
            if child.entry != nil && child.lastAccessed < oldestDate {
                oldestDate = child.lastAccessed
                oldestKey = key
            }
        }
        if let key = oldestKey {
            let child = node.children[key]!
            child.entry = nil
            storedCount -= 1
            if child.children.isEmpty { node.children.removeValue(forKey: key) }
            return true
        }
        for (_, child) in node.children {
            if evictOldestLeaf(child) { return true }
        }
        return false
    }
}
