import Foundation
import HuggingFace
@preconcurrency import Tokenizers
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon

/// Actor that runs batched MLX inference with KV prefix caching.
///
/// Model loading is delegated to a `ModelPool<ModelContainer>` that enforces an
/// LRU eviction policy when more models are requested than `maxModels` allows.
/// Requests in a batch share a single model-lock acquisition: all prompts are
/// tokenised, then each sequence is decoded sequentially.
///
/// For each request, the engine:
///  1. Looks up the prompt token sequence in a per-model prefix trie.
///  2. On hit: restores cached KV state and feeds only the suffix tokens to the
///     model, skipping the prefill for the shared prefix.
///  3. On miss: runs the full prefill, snapshots the KV state at the prompt
///     boundary, and stores it.
public actor MLXInferenceEngine {
    private let pool: ModelPool<ModelContainer>
    // One prefix cache per model ID, isolated within this actor.
    private var prefixCaches: [String: KVPrefixCache] = [:]

    public init(maxModels: Int = 3) {
        pool = ModelPool(maxModels: maxModels) { modelID in
            try await #huggingFaceLoadModelContainer(
                configuration: ModelConfiguration(id: modelID)
            )
        }
    }

    /// Returns the IDs of models currently resident in the pool.
    public func loadedModelIDs() async -> [String] {
        await pool.loadedModelIDs
    }

    /// Returns a `BatchHandler` closure suitable for `BatchAssembler`.
    public nonisolated func makeBatchHandler() -> BatchAssembler.BatchHandler {
        { [self] requests in
            try await self.inferBatch(requests)
        }
    }

    /// Returns a `StreamHandler` closure for per-token SSE streaming.
    /// The returned stream yields `StreamChunk.token` for each decoded fragment,
    /// followed by a single `StreamChunk.done` carrying the finish reason.
    public nonisolated func makeStreamHandler() -> StreamHandler {
        { [self] request in
            AsyncThrowingStream { continuation in
                Task {
                    do {
                        try await self.inferStream(request: request, continuation: continuation)
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    // MARK: - Private

    private func inferStream(
        request: ChatCompletionRequest,
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws {
        let container = try await pool.acquire(modelID: request.model)
        let prefixCache = getOrCreatePrefixCache(modelID: request.model)

        try await container.perform { context in
            let messages: [[String: any Sendable]] = request.messages.map { msg in
                ["role": msg.role.rawValue as any Sendable,
                 "content": msg.content.textContent as any Sendable]
            }
            let promptTokenIds = try context.tokenizer.applyChatTemplate(messages: messages)
            let promptInt32 = promptTokenIds.map(Int32.init)

            let params = GenerateParameters(
                maxTokens: request.maxTokens ?? 512,
                temperature: Float(request.temperature ?? 0.7),
                topP: Float(request.topP ?? 1.0),
                seed: request.seed.map(UInt64.init)
            )

            let (prefixLen, cachedEntry) = await prefixCache.lookup(tokens: promptInt32)

            let kvCaches: [any KVCache]
            let inputTokenIds: [Int]

            if let entry = cachedEntry, prefixLen > 0 {
                let restored = entry.layerStates.map { layerState in
                    let c = KVCacheSimple()
                    c.state = layerState
                    return c as any KVCache
                }
                let suffix = Array(promptTokenIds[prefixLen...])
                if suffix.isEmpty {
                    // Full prefix hit: trim one so TokenIterator can re-run the last
                    // prompt token through the model and sample the first output token.
                    restored.forEach { $0.trim(1) }
                    inputTokenIds = [promptTokenIds.last!]
                } else {
                    inputTokenIds = suffix
                }
                kvCaches = restored
            } else {
                kvCaches = context.model.newCache(parameters: params)
                inputTokenIds = promptTokenIds
            }

            let lmInput = LMInput(
                tokens: MLXArray(inputTokenIds.map(Int32.init), [inputTokenIds.count])
            )

            let iterator = try TokenIterator(
                input: lmInput,
                model: context.model,
                cache: kvCaches,
                parameters: params
            )

            // Snapshot KV state at the prompt boundary on a cold miss.
            // After init, the cache holds exactly all prompt tokens — no trim needed.
            if cachedEntry == nil || prefixLen == 0 {
                nonisolated(unsafe) let layerStates: [[MLXArray]] = kvCaches.map { c in
                    let snap = c.copy()
                    let st = snap.state
                    eval(st)
                    return st
                }
                await prefixCache.store(tokens: promptInt32, layerStates: layerStates)
            }

            let eosId = context.tokenizer.eosTokenId
            let stopStrings = request.stop?.strings ?? []
            let maxStopLen = stopStrings.map(\.count).max() ?? 0
            var pendingText = ""
            var finishReason: FinishReason = .length

            for token in iterator {
                if let eosId, token == eosId {
                    finishReason = .stop
                    break
                }
                let piece = context.tokenizer.decode(tokenIds: [token])

                if stopStrings.isEmpty {
                    continuation.yield(.token(piece))
                } else {
                    pendingText += piece

                    // Check for any stop sequence in accumulated text.
                    var hitStop = false
                    for stop in stopStrings {
                        if let range = pendingText.range(of: stop) {
                            let safe = String(pendingText[..<range.lowerBound])
                            if !safe.isEmpty { continuation.yield(.token(safe)) }
                            pendingText = ""
                            finishReason = .stop
                            hitStop = true
                            break
                        }
                    }
                    if hitStop { break }

                    // Yield the safe prefix — everything more than maxStopLen chars back
                    // — so a stop sequence that spans the boundary is still detectable.
                    if pendingText.count > maxStopLen {
                        let splitOffset = pendingText.count - maxStopLen
                        let splitIdx = pendingText.index(pendingText.startIndex, offsetBy: splitOffset)
                        let safe = String(pendingText[..<splitIdx])
                        pendingText = String(pendingText[splitIdx...])
                        continuation.yield(.token(safe))
                    }
                }
            }

            // Flush any remaining buffered text.
            if !pendingText.isEmpty {
                var flushed = false
                for stop in stopStrings {
                    if pendingText.hasSuffix(stop) {
                        let trimmed = String(pendingText.dropLast(stop.count))
                        if !trimmed.isEmpty { continuation.yield(.token(trimmed)) }
                        finishReason = .stop
                        flushed = true
                        break
                    }
                }
                if !flushed { continuation.yield(.token(pendingText)) }
            }

            continuation.yield(.done(finishReason))
        }
    }

    private func inferBatch(
        _ requests: [ChatCompletionRequest]
    ) async throws -> [ChatCompletionResponse] {
        guard !requests.isEmpty else { return [] }
        let container = try await pool.acquire(modelID: requests[0].model)
        return try await batchedInfer(container: container, requests: requests)
    }

    private func getOrCreatePrefixCache(modelID: String) -> KVPrefixCache {
        if let existing = prefixCaches[modelID] { return existing }
        let cache = KVPrefixCache()
        prefixCaches[modelID] = cache
        return cache
    }

    /// Acquires the model lock once, then processes every request sequentially.
    ///
    /// For each request:
    ///  - Lookup: find the longest cached prefix in the prefix trie.
    ///  - Hit: restore KV state; feed only suffix tokens to `TokenIterator`.
    ///  - Miss: full prefill via `TokenIterator`; snapshot and store KV state.
    private func batchedInfer(
        container: ModelContainer,
        requests: [ChatCompletionRequest]
    ) async throws -> [ChatCompletionResponse] {
        let modelID = requests[0].model
        let prefixCache = getOrCreatePrefixCache(modelID: modelID)

        return try await container.perform { context in
            var responses: [ChatCompletionResponse] = []

            for request in requests {
                let messages: [[String: any Sendable]] = request.messages.map { msg in
                    let role: any Sendable = msg.role.rawValue
                    let content: any Sendable = msg.content.textContent
                    return ["role": role, "content": content]
                }

                let promptTokenIds = try context.tokenizer.applyChatTemplate(
                    messages: messages
                )
                let promptInt32 = promptTokenIds.map(Int32.init)

                let params = GenerateParameters(
                    maxTokens: request.maxTokens ?? 512,
                    temperature: Float(request.temperature ?? 0.7),
                    topP: Float(request.topP ?? 1.0),
                    seed: request.seed.map(UInt64.init)
                )

                // --- KV prefix cache lookup ---
                let (prefixLen, cachedEntry) = await prefixCache.lookup(tokens: promptInt32)

                let kvCaches: [any KVCache]
                let inputTokenIds: [Int]

                if let entry = cachedEntry, prefixLen > 0 {
                    // Hit: restore cached KV state, process only the suffix.
                    let restored = entry.layerStates.map { layerState in
                        let c = KVCacheSimple()
                        c.state = layerState
                        return c as any KVCache
                    }
                    let suffix = Array(promptTokenIds[prefixLen...])
                    if suffix.isEmpty {
                        // Full prefix hit: trim one so TokenIterator can re-run the last
                        // prompt token and sample the first output token correctly.
                        restored.forEach { $0.trim(1) }
                        inputTokenIds = [promptTokenIds.last!]
                    } else {
                        inputTokenIds = suffix
                    }
                    kvCaches = restored
                } else {
                    // Miss: fresh cache, process the whole prompt.
                    kvCaches = context.model.newCache(parameters: params)
                    inputTokenIds = promptTokenIds
                }

                let lmInput = LMInput(
                    tokens: MLXArray(inputTokenIds.map(Int32.init), [inputTokenIds.count])
                )

                // Creates the iterator; runs prefill + primes the first token in-place
                // on `kvCaches` (the cache class instances are mutated).
                let iterator = try TokenIterator(
                    input: lmInput,
                    model: context.model,
                    cache: kvCaches,
                    parameters: params
                )

                // Snapshot KV state at the prompt boundary on a cold miss.
                // After init, the cache holds exactly all prompt tokens — no trim needed.
                if cachedEntry == nil || prefixLen == 0 {
                    // eval() materialises the graph before the actor send; nonisolated(unsafe)
                    // suppresses region-isolation false positives on @unchecked Sendable MLXArrays.
                    nonisolated(unsafe) let layerStates: [[MLXArray]] = kvCaches.map { c in
                        let snap = c.copy()
                        let st = snap.state
                        eval(st)
                        return st
                    }
                    await prefixCache.store(tokens: promptInt32, layerStates: layerStates)
                }

                // Decode until EOS, stop sequence, or maxTokens.
                let eosId = context.tokenizer.eosTokenId
                let stopStrings = request.stop?.strings ?? []
                var generatedText = ""
                var completionTokens = 0
                var finishReason: FinishReason = .length

                for token in iterator {
                    if let eosId, token == eosId {
                        finishReason = .stop
                        break
                    }
                    completionTokens += 1
                    generatedText += context.tokenizer.decode(tokenIds: [token])

                    var hitStop = false
                    for stop in stopStrings {
                        if generatedText.hasSuffix(stop) {
                            generatedText = String(generatedText.dropLast(stop.count))
                            finishReason = .stop
                            hitStop = true
                            break
                        }
                    }
                    if hitStop { break }
                }

                responses.append(
                    ChatCompletionResponse(
                        id: "chatcmpl-\(UUID().uuidString)",
                        created: Int(Date().timeIntervalSince1970),
                        model: request.model,
                        choices: [
                            Choice(
                                index: 0,
                                message: ChatMessage(role: .assistant, content: generatedText),
                                finishReason: finishReason
                            )
                        ],
                        usage: Usage(
                            promptTokens: promptTokenIds.count,
                            completionTokens: completionTokens
                        )
                    )
                )
            }

            return responses
        }
    }
}
