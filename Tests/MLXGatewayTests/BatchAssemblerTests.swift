import XCTest
@testable import MLXGateway

// MARK: - Helpers

/// Thread-safe log for recording batches dispatched during tests.
private actor BatchLog {
    var batches: [[ChatCompletionRequest]] = []

    func record(_ batch: [ChatCompletionRequest]) {
        batches.append(batch)
    }
}

private func makeResponse(for request: ChatCompletionRequest) -> ChatCompletionResponse {
    ChatCompletionResponse(
        id: "chatcmpl-\(UUID().uuidString)",
        created: Int(Date().timeIntervalSince1970),
        model: request.model,
        choices: [
            Choice(
                index: 0,
                message: ChatMessage(role: .assistant, content: "ok"),
                finishReason: .stop
            )
        ],
        usage: Usage(promptTokens: 5, completionTokens: 3)
    )
}

private func echoHandler(_ requests: [ChatCompletionRequest]) async throws -> [ChatCompletionResponse] {
    requests.map { makeResponse(for: $0) }
}

// MARK: - Tests

final class BatchAssemblerTests: XCTestCase {

    // MARK: Stress test — 100 concurrent requests, none dropped

    func testHundredConcurrentRequests_allComplete() async throws {
        let requestCount = 100
        let assembler = BatchAssembler(
            configuration: .init(maxBatchSize: 8, maxWaitMs: 30),
            handler: echoHandler
        )

        let responses = try await withThrowingTaskGroup(of: ChatCompletionResponse.self) { group in
            for i in 0..<requestCount {
                group.addTask {
                    let req = ChatCompletionRequest(
                        model: "test-model",
                        messages: [ChatMessage(role: .user, content: "Hello \(i)")]
                    )
                    return try await assembler.submit(req)
                }
            }
            var collected: [ChatCompletionResponse] = []
            collected.reserveCapacity(requestCount)
            for try await resp in group {
                collected.append(resp)
            }
            return collected
        }

        XCTAssertEqual(responses.count, requestCount,
                       "All \(requestCount) concurrent requests must complete — none dropped")
    }

    // MARK: Stress test — mixed models, 100 concurrent requests

    func testHundredConcurrentRequests_mixedModels_allComplete() async throws {
        let models = ["llama-3b", "mistral-7b", "phi-2"]
        let requestCount = 100
        let assembler = BatchAssembler(
            configuration: .init(maxBatchSize: 12, maxWaitMs: 40),
            handler: echoHandler
        )

        let responses = try await withThrowingTaskGroup(of: ChatCompletionResponse.self) { group in
            for i in 0..<requestCount {
                let model = models[i % models.count]
                group.addTask {
                    let req = ChatCompletionRequest(
                        model: model,
                        messages: [ChatMessage(role: .user, content: "Query \(i)")]
                    )
                    return try await assembler.submit(req)
                }
            }
            var collected: [ChatCompletionResponse] = []
            collected.reserveCapacity(requestCount)
            for try await resp in group {
                collected.append(resp)
            }
            return collected
        }

        XCTAssertEqual(responses.count, requestCount,
                       "All \(requestCount) mixed-model requests must complete — none dropped")
    }

    // MARK: Batches are grouped by model

    func testBatches_groupedByModel() async throws {
        let log = BatchLog()

        let assembler = BatchAssembler(
            configuration: .init(maxBatchSize: 20, maxWaitMs: 60)
        ) { requests in
            await log.record(requests)
            return requests.map { makeResponse(for: $0) }
        }

        try await withThrowingTaskGroup(of: ChatCompletionResponse.self) { group in
            for _ in 0..<6 {
                group.addTask {
                    try await assembler.submit(ChatCompletionRequest(
                        model: "model-a",
                        messages: [ChatMessage(role: .user, content: "hi")]
                    ))
                }
                group.addTask {
                    try await assembler.submit(ChatCompletionRequest(
                        model: "model-b",
                        messages: [ChatMessage(role: .user, content: "hi")]
                    ))
                }
            }
            for try await _ in group {}
        }

        let batches = await log.batches
        for batch in batches {
            let distinctModels = Set(batch.map(\.model))
            XCTAssertEqual(distinctModels.count, 1,
                           "Every batch must contain requests for exactly one model; got \(distinctModels)")
        }
    }

    // MARK: max_batch_size triggers immediate flush

    func testMaxBatchSize_triggersImmediateFlush() async throws {
        let log = BatchLog()
        let maxBatchSize = 5

        let assembler = BatchAssembler(
            // Long timer — flush must be triggered by size, not timeout
            configuration: .init(maxBatchSize: maxBatchSize, maxWaitMs: 5_000)
        ) { requests in
            await log.record(requests)
            return requests.map { makeResponse(for: $0) }
        }

        // Send exactly maxBatchSize requests so the size threshold is hit
        try await withThrowingTaskGroup(of: ChatCompletionResponse.self) { group in
            for i in 0..<maxBatchSize {
                group.addTask {
                    try await assembler.submit(ChatCompletionRequest(
                        model: "test-model",
                        messages: [ChatMessage(role: .user, content: "msg \(i)")]
                    ))
                }
            }
            for try await _ in group {}
        }

        let batches = await log.batches
        XCTAssertEqual(batches.count, 1)
        XCTAssertEqual(batches[0].count, maxBatchSize,
                       "Batch of exactly max_batch_size should be flushed as a single full batch")
    }

    // MARK: max_wait_ms triggers flush for partial batches

    func testMaxWaitMs_flushesPartialBatch() async throws {
        let maxWaitMs = 60

        let assembler = BatchAssembler(
            // Large batch size — timer must trigger, not size
            configuration: .init(maxBatchSize: 200, maxWaitMs: maxWaitMs),
            handler: echoHandler
        )

        let start = ContinuousClock.now
        // Submit fewer requests than maxBatchSize
        try await withThrowingTaskGroup(of: ChatCompletionResponse.self) { group in
            for i in 0..<3 {
                group.addTask {
                    try await assembler.submit(ChatCompletionRequest(
                        model: "test-model",
                        messages: [ChatMessage(role: .user, content: "msg \(i)")]
                    ))
                }
            }
            for try await _ in group {}
        }
        let elapsed = ContinuousClock.now - start
        let elapsedMs = Double(elapsed.components.seconds) * 1000
                      + Double(elapsed.components.attoseconds) / 1e15

        XCTAssertGreaterThanOrEqual(elapsedMs, Double(maxWaitMs) * 0.5,
                                    "Partial batch should wait for the timer before flushing")
        XCTAssertLessThan(elapsedMs, Double(maxWaitMs) * 8,
                          "Partial batch should not wait unreasonably long beyond the timer")
    }

    // MARK: Sequence-length binning separates short and long prompts

    func testSequenceLengthBinning_separatesShortAndLongPrompts() async throws {
        let log = BatchLog()

        let assembler = BatchAssembler(
            configuration: .init(maxBatchSize: 20, maxWaitMs: 80)
        ) { requests in
            await log.record(requests)
            return requests.map { makeResponse(for: $0) }
        }

        // Short prompt: ~5 tokens → bin 128
        let shortContent = "Hi"
        // Long prompt: ~600 tokens → bin 2048  (≈2400 chars / 4)
        let longContent = String(repeating: "word ", count: 480)

        try await withThrowingTaskGroup(of: ChatCompletionResponse.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    try await assembler.submit(ChatCompletionRequest(
                        model: "test-model",
                        messages: [ChatMessage(role: .user, content: shortContent)]
                    ))
                }
                group.addTask {
                    try await assembler.submit(ChatCompletionRequest(
                        model: "test-model",
                        messages: [ChatMessage(role: .user, content: longContent)]
                    ))
                }
            }
            for try await _ in group {}
        }

        let batches = await log.batches
        // Every batch must be internally homogeneous (all short or all long)
        for batch in batches {
            let contentLengths = Set(batch.map { req -> String in
                if case .text(let s) = req.messages.first?.content { return s }
                return ""
            })
            XCTAssertEqual(contentLengths.count, 1,
                           "Requests with different sequence lengths should land in separate bins/batches")
        }
    }

    // MARK: Response count mismatch surfaced as error

    func testHandlerResponseMismatch_failsAffectedRequests() async throws {
        let assembler = BatchAssembler(
            configuration: .init(maxBatchSize: 3, maxWaitMs: 50)
        ) { requests in
            // Return one fewer response than requested — BatchAssembler must handle this
            Array(requests.map { makeResponse(for: $0) }.dropLast())
        }

        // Return true = succeeded, false = threw; collect sequentially via `for await`
        var successCount = 0
        await withTaskGroup(of: Bool.self) { group in
            for i in 0..<3 {
                group.addTask {
                    do {
                        _ = try await assembler.submit(ChatCompletionRequest(
                            model: "test-model",
                            messages: [ChatMessage(role: .user, content: "msg \(i)")]
                        ))
                        return true
                    } catch {
                        return false
                    }
                }
            }
            for await succeeded in group {
                if succeeded { successCount += 1 }
            }
        }

        XCTAssertEqual(successCount, 2, "Two requests should succeed")
        XCTAssertEqual(3 - successCount, 1, "One request should fail with a mismatch error")
    }
}
