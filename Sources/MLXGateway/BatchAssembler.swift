import Foundation

// MARK: - Supporting Types

public struct BatchKey: Hashable, Sendable {
    public let modelID: String
    public let sequenceLengthBin: Int
}

public struct BatchAssemblerConfig: Sendable {
    public let maxBatchSize: Int
    public let maxWaitMs: Int

    public init(maxBatchSize: Int = 8, maxWaitMs: Int = 50) {
        self.maxBatchSize = maxBatchSize
        self.maxWaitMs = maxWaitMs
    }
}

public enum BatchAssemblerError: Error, Sendable {
    case responseMismatch(expected: Int, got: Int)
}

// MARK: - BatchAssembler

/// Accepts inference requests, groups them by (model_id, sequence_length_bin),
/// and emits a batch when max_batch_size is reached or max_wait_ms elapses.
public actor BatchAssembler {
    /// Receives a batch of requests; must return exactly one response per request, in order.
    public typealias BatchHandler = @Sendable ([ChatCompletionRequest]) async throws -> [ChatCompletionResponse]

    private struct PendingRequest: Sendable {
        let request: ChatCompletionRequest
        let continuation: CheckedContinuation<ChatCompletionResponse, Error>
    }

    private let config: BatchAssemblerConfig
    private let handler: BatchHandler
    private var queues: [BatchKey: [PendingRequest]] = [:]
    private var timers: [BatchKey: Task<Void, Never>] = [:]

    public init(configuration: BatchAssemblerConfig = .init(), handler: @escaping BatchHandler) {
        self.config = configuration
        self.handler = handler
    }

    /// Enqueues a request and suspends until its batch is dispatched and a response is returned.
    public nonisolated func submit(_ request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        try await withCheckedThrowingContinuation { continuation in
            Task {
                await self.enqueue(request, continuation: continuation)
            }
        }
    }

    // MARK: - Private

    private func enqueue(
        _ request: ChatCompletionRequest,
        continuation: CheckedContinuation<ChatCompletionResponse, Error>
    ) {
        let key = batchKey(for: request)
        let item = PendingRequest(request: request, continuation: continuation)
        queues[key, default: []].append(item)

        if queues[key]!.count >= config.maxBatchSize {
            flush(key: key)
        } else if timers[key] == nil {
            scheduleTimer(for: key)
        }
    }

    private func scheduleTimer(for key: BatchKey) {
        let delay = UInt64(config.maxWaitMs) * 1_000_000
        timers[key] = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: delay)
            } catch {
                return
            }
            guard let self else { return }
            await self.timerFired(key: key)
        }
    }

    private func timerFired(key: BatchKey) {
        timers[key] = nil
        flush(key: key)
    }

    private func flush(key: BatchKey) {
        timers[key]?.cancel()
        timers[key] = nil
        guard let pending = queues.removeValue(forKey: key), !pending.isEmpty else { return }

        let requests = pending.map(\.request)
        let continuations = pending.map(\.continuation)
        let handler = self.handler

        Task.detached {
            do {
                let responses = try await handler(requests)
                for (cont, resp) in zip(continuations, responses) {
                    cont.resume(returning: resp)
                }
                if responses.count < continuations.count {
                    let err = BatchAssemblerError.responseMismatch(
                        expected: continuations.count,
                        got: responses.count
                    )
                    for cont in continuations.dropFirst(responses.count) {
                        cont.resume(throwing: err)
                    }
                }
            } catch {
                for cont in continuations {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Binning

    private func batchKey(for request: ChatCompletionRequest) -> BatchKey {
        let tokens = estimateTokenCount(request)
        return BatchKey(modelID: request.model, sequenceLengthBin: binForTokenCount(tokens))
    }

    private func estimateTokenCount(_ request: ChatCompletionRequest) -> Int {
        let chars = request.messages.reduce(0) { count, msg in
            switch msg.content {
            case .text(let s): return count + s.count
            case .parts(let parts): return count + parts.compactMap(\.text).reduce(0) { $0 + $1.count }
            }
        }
        return max(1, chars / 4)
    }

    private func binForTokenCount(_ tokens: Int) -> Int {
        let bins = [128, 256, 512, 1024, 2048, 4096]
        return bins.first { $0 >= tokens } ?? 8192
    }
}
