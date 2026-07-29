import XCTest
@testable import MLXGateway

// MARK: - Helpers

private actor LoadCounter {
    var callCount = 0
    func increment() { callCount += 1 }
}

// Fake container — plain Int so we need no MLX dependency in these tests
private typealias FakePool = ModelPool<Int>

private func makePool(maxModels: Int, counter: LoadCounter? = nil) -> FakePool {
    ModelPool(maxModels: maxModels) { id in
        await counter?.increment()
        return id.count          // deterministic, unique-ish value per id
    }
}

// MARK: - Tests

final class ModelPoolTests: XCTestCase {

    // MARK: Cache hit — loader is not called again

    func testCacheHit_doesNotReload() async throws {
        let counter = LoadCounter()
        let pool = makePool(maxModels: 3, counter: counter)

        _ = try await pool.acquire(modelID: "a")
        _ = try await pool.acquire(modelID: "a")
        _ = try await pool.acquire(modelID: "a")

        let calls = await counter.callCount
        XCTAssertEqual(calls, 1, "Loader must be called exactly once on repeated cache hits")
    }

    // MARK: LRU eviction — least recently used model is dropped

    func testLRUEviction_evictsLeastRecentlyUsed() async throws {
        let pool = makePool(maxModels: 2)

        _ = try await pool.acquire(modelID: "first")
        _ = try await pool.acquire(modelID: "second")

        // Re-access "first" to make "second" the LRU
        _ = try await pool.acquire(modelID: "first")

        // Loading "third" must evict "second" (LRU), not "first"
        _ = try await pool.acquire(modelID: "third")

        let loaded = await pool.loadedModelIDs
        XCTAssertTrue(loaded.contains("first"), "\"first\" was recently used — must stay in pool")
        XCTAssertFalse(loaded.contains("second"), "\"second\" was LRU — must have been evicted")
        XCTAssertTrue(loaded.contains("third"), "\"third\" was just loaded — must be in pool")
        XCTAssertEqual(loaded.count, 2, "Pool must not exceed maxModels=2 after eviction")
    }

    // MARK: Pool never exceeds maxModels

    func testPoolSize_neverExceedsMax() async throws {
        let max = 3
        let pool = makePool(maxModels: max)

        for i in 0..<10 {
            _ = try await pool.acquire(modelID: "model-\(i)")
            let count = await pool.loadedModelIDs.count
            XCTAssertLessThanOrEqual(count, max,
                "Pool size must never exceed maxModels=\(max); got \(count) after loading model-\(i)")
        }
    }

    // MARK: Concurrent requests for the same model coalesce into one load

    func testConcurrentSameModel_loaderCalledOnce() async throws {
        let counter = LoadCounter()
        // Slow loader to ensure overlap
        let pool: ModelPool<Int> = ModelPool(maxModels: 4) { id in
            await counter.increment()
            try await Task.sleep(nanoseconds: 20_000_000)   // 20 ms
            return id.count
        }

        let concurrency = 8
        _ = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<concurrency {
                group.addTask { try await pool.acquire(modelID: "shared-model") }
            }
            var results: [Int] = []
            for try await v in group { results.append(v) }
            return results
        }

        let calls = await counter.callCount
        XCTAssertEqual(calls, 1,
            "Concurrent requests for the same model must coalesce into one loader invocation; got \(calls)")
    }

    // MARK: Failed load does not pollute the cache

    func testFailedLoad_doesNotCacheEntry() async throws {
        struct LoadError: Error {}
        let pool: ModelPool<Int> = ModelPool(maxModels: 2) { _ in throw LoadError() }

        do {
            _ = try await pool.acquire(modelID: "bad-model")
            XCTFail("Expected loader error to propagate")
        } catch {}

        let loaded = await pool.loadedModelIDs
        XCTAssertFalse(loaded.contains("bad-model"),
            "A failed load must not leave a stale entry in the cache")
    }

    // MARK: Access order updates LRU correctly across multiple models

    func testAccessOrder_updatesLRU() async throws {
        let pool = makePool(maxModels: 2)

        _ = try await pool.acquire(modelID: "x")
        _ = try await pool.acquire(modelID: "y")

        // Re-access "x" — now "y" is LRU
        _ = try await pool.acquire(modelID: "x")

        _ = try await pool.acquire(modelID: "z")   // should evict "y"

        let loaded = await pool.loadedModelIDs
        XCTAssertFalse(loaded.contains("y"), "\"y\" should have been evicted as LRU")
        XCTAssertTrue(loaded.contains("x"), "\"x\" was re-accessed and must survive")
        XCTAssertTrue(loaded.contains("z"), "\"z\" was just loaded and must be present")
    }
}
