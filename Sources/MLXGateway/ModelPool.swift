import Foundation

/// LRU pool of concurrently-loaded models, bounded by `maxModels`.
///
/// When adding a model would exceed capacity, the least-recently-used entry is
/// evicted first. Concurrent requests for the same model ID are coalesced: only
/// one load task runs; all waiters share its result.
public actor ModelPool<Container: Sendable> {
    public typealias Loader = @Sendable (String) async throws -> Container

    private struct Entry {
        let container: Container
        var lastUsed: Date
    }

    public let maxModels: Int
    private let loader: Loader
    private var cache: [String: Entry] = [:]
    private var loadTasks: [String: Task<Container, Error>] = [:]

    public init(maxModels: Int = 3, loader: @escaping Loader) {
        self.maxModels = maxModels
        self.loader = loader
    }

    /// Returns the container for `modelID`, loading it if necessary and updating
    /// the LRU timestamp on every successful access.
    public func acquire(modelID: String) async throws -> Container {
        // Fast path: already in cache
        if let entry = cache[modelID] {
            cache[modelID]!.lastUsed = Date()
            return entry.container
        }

        // Coalesce concurrent requests for the same model
        if let task = loadTasks[modelID] {
            let container = try await task.value
            cache[modelID]?.lastUsed = Date()
            return container
        }

        // Evict LRU entries until there is room
        while cache.count >= maxModels {
            evictLRU()
        }

        // Kick off the load and register it so concurrent callers can coalesce
        let id = modelID
        let load = self.loader
        let task = Task { try await load(id) }
        loadTasks[id] = task

        do {
            let container = try await task.value
            loadTasks.removeValue(forKey: id)
            cache[id] = Entry(container: container, lastUsed: Date())
            return container
        } catch {
            loadTasks.removeValue(forKey: id)
            throw error
        }
    }

    /// IDs of models currently held in the cache (unordered).
    public var loadedModelIDs: [String] {
        Array(cache.keys)
    }

    private func evictLRU() {
        guard let key = cache.min(by: { $0.value.lastUsed < $1.value.lastUsed })?.key else { return }
        cache.removeValue(forKey: key)
    }
}
