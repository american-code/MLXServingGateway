import Foundation
import HuggingFace
@preconcurrency import Tokenizers
import MLX
import MLXLLM
import MLXHuggingFace
import MLXLMCommon

/// Actor that lazily loads MLX models and runs batched inference.
///
/// Requests in a batch share a single model-lock acquisition: all prompts are
/// tokenized, then each sequence is decoded sequentially using its own
/// `GenerateParameters` (temperature / top-p / seed). This amortises the
/// per-batch overhead of acquiring the `SerialAccessContainer` lock and keeps
/// the model hot in the Metal cache between requests.
public actor MLXInferenceEngine {
    private var modelCache: [String: ModelContainer] = [:]

    public init() {}

    /// Returns a `BatchHandler` closure suitable for `BatchAssembler`.
    ///
    /// The closure captures `self` (the actor) so every batch funnels through
    /// the actor's serial executor before touching the model cache.
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
        let container = try await ensureLoaded(modelID: requests[0].model)
        return try await batchedInfer(container: container, requests: requests)
    }

    private func ensureLoaded(modelID: String) async throws -> ModelContainer {
        if let cached = modelCache[modelID] { return cached }
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(id: modelID)
        )
        modelCache[modelID] = container
        return container
    }

    /// Acquires the model lock once, then processes every request sequentially.
    ///
    /// Sequential-within-lock means:
    /// - Prefill for request N+1 starts only after decode for request N finishes.
    /// - The model weights stay hot between requests in the same batch.
    /// - At most one concurrent Metal command stream is active at any time.
    private func batchedInfer(
        container: ModelContainer,
        requests: [ChatCompletionRequest]
    ) async throws -> [ChatCompletionResponse] {
        try await container.perform { context in
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

                let params = GenerateParameters(
                    maxTokens: request.maxTokens ?? 512,
                    temperature: Float(request.temperature ?? 0.7),
                    topP: Float(request.topP ?? 1.0),
                    seed: request.seed.map(UInt64.init)
                )

                let lmInput = LMInput(
                    tokens: MLXArray(promptTokenIds.map(Int32.init), [promptTokenIds.count])
                )

                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: params, context: context
                )

                var outputText = ""
                var completionTokenCount = 0
                for await generation in stream {
                    switch generation {
                    case .chunk(let text):
                        outputText += text
                    case .info(let info):
                        completionTokenCount = info.generationTokenCount
                    case .toolCall:
                        break
                    }
                }

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
                            completionTokens: completionTokenCount
                        )
                    )
                )
            }

            return responses
        }
    }
}
