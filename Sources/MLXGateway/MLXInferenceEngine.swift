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
///     boundary (trimming off the first generated token), and stores it.
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

    /// Returns a `BatchHandler` closure suitable for `BatchAssembler`.
    public nonisolated func makeBatchHandler() -> BatchAssembler.BatchHandler {
        { [self] requests in
            try await self.inferBatch(requests)
        }
    }

    // MARK: - Private

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
                    kvCaches = entry.layerStates.map { layerState in
                        let c = KVCacheSimple()
                        c.state = layerState
                        return c as any KVCache
                    }
                    inputTokenIds = Array(promptTokenIds[prefixLen...])
                } else {
                    // Miss: fresh cache, process the whole prompt.
                    kvCaches = context.model.newCache(parameters: params)
                    inputTokenIds = promptTokenIds
                }

                let lmInput = LMInput(
                    tokens: MLXArray(inputTokenIds.map(Int32.init), [inputTokenIds.count])
                )

                // Creates the iterator; runs prefill + generates the first token
                // in-place on `kvCaches` (the cache class instances are mutated).
                var iterator = try TokenIterator(
                    input: lmInput,
                    model: context.model,
                    cache: kvCaches,
                    parameters: params
                )

                // On a cold miss, snapshot KV state at the prompt boundary.
                // The iterator has already generated one token, so trim 1 off.
                if cachedEntry == nil || prefixLen == 0 {
                    let layerStates: [[MLXArray]] = kvCaches.map { c in
                        let snap = c.copy()
                        snap.trim(1)
                        let st = snap.state
                        eval(st)    // materialise graph before crossing actor boundary
                        return st
                    }
                    await prefixCache.store(tokens: promptInt32, layerStates: layerStates)
                }

                // Decode until EOS or maxTokens.
                let eosId = context.tokenizer.eosTokenId
                var generatedIds: [Int] = []
                for token in iterator {
                    if let eosId, token == eosId { break }
                    generatedIds.append(token)
                }

                let outputText = context.tokenizer.decode(tokenIds: generatedIds)

                responses.append(
                    ChatCompletionResponse(
                        id: "chatcmpl-\(UUID().uuidString)",
                        created: Int(Date().timeIntervalSince1970),
                        model: request.model,
                        choices: [
                            Choice(
                                index: 0,
                                message: ChatMessage(role: .assistant, content: outputText),
                                finishReason: .stop
                            )
                        ],
                        usage: Usage(
                            promptTokens: promptTokenIds.count,
                            completionTokens: generatedIds.count
                        )
                    )
                )
            }

            return responses
        }
    }
}
