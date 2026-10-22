import Foundation
import CoreFoundation
#if canImport(OSLog)
import OSLog
#endif
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

enum HostedAskPLONNError: Error, Equatable {
    case invalidEndpoint, authenticationRequired, capacity, rateLimited, dailyAllowance, unavailable
    case oversizedPayload, invalidResponse, unverifiedModel

    var userFacingMessage: String {
        switch self {
        case .invalidEndpoint: return "AI chat is unavailable in this build. Recorded research is still available."
        case .authenticationRequired: return "Sign in again to use Gemma 4 31B. Recorded research is still available."
        case .capacity: return "AI is handling another request. Try again shortly. Recorded research is still available."
        case .rateLimited: return "AI is receiving too many requests. Try again shortly. Recorded research is still available."
        case .dailyAllowance: return "AI’s daily request allowance has been reached. It resets at midnight UTC. Recorded research is still available."
        case .unavailable: return "AI couldn’t complete this request. Please try again. Recorded research is still available."
        case .oversizedPayload: return "This request or response exceeded AI chat limits. Recorded research is still available."
        case .invalidResponse, .unverifiedModel: return "The AI response could not be verified. Recorded research is still available."
        }
    }
}

/// A fixed vocabulary keeps failure diagnostics free of request/response data.
enum AskPLONNFailureDiagnostic {
    static func category(for error: Error) -> String {
        if error is CancellationError { return "cancelled" }
        if error is URLError { return "network" }
        if let hosted = error as? HostedAskPLONNError {
            switch hosted {
            case .invalidEndpoint: return "configuration"
            case .authenticationRequired: return "authentication"
            case .capacity: return "capacity"
            case .rateLimited: return "rate_limit"
            case .dailyAllowance: return "daily_allowance"
            case .unavailable: return "service_unavailable"
            case .oversizedPayload: return "payload_limit"
            case .invalidResponse: return "response_schema"
            case .unverifiedModel: return "model_identity"
            }
        }
        if let core = error as? AskPLONNError {
            switch core {
            case .unsupportedAnswer: return "answer_contract"
            case .unsupportedQuestion: return "unsupported_question"
            default: return "request_context"
            }
        }
        return "other"
    }

    /// Safe to show or copy: this vocabulary never includes payload/error details.
    static func matchupMessage(for error: Error) -> String {
        if let hosted = error as? HostedAskPLONNError {
            return hosted.userFacingMessage + " Tap Generate to retry when ready."
        }
        switch error {
        case AskPLONNError.unsupportedAnswer:
            return "The AI explanation did not match the recorded evidence. Observed data is available below."
        case AskPLONNError.unsupportedQuestion:
            return "The AI could not support an explanation from this evidence. Observed data is available below."
        default:
            return "The AI explanation could not be completed. Observed data is available below."
        }
    }

    static func recordMatchupFailure(_ error: Error) {
        #if canImport(OSLog)
        let category = category(for: error)
        Logger(subsystem: "ai.oakas.plonn", category: "MatchupInsights")
            .notice("Matchup explanation failed: \(category, privacy: .public)")
        #endif
    }
}

/// Local account eligibility; service availability is confirmed when a request completes.
enum HostedAskPLONNConnectionState: Equatable {
    case unavailable, signInRequired, consentRequired, ready

    static func resolve(serviceAvailable: Bool, signedIn: Bool, consented: Bool) -> Self {
        guard serviceAvailable else { return .unavailable }
        guard signedIn else { return .signInRequired }
        return consented ? .ready : .consentRequired
    }
    var statusText: String {
        switch self {
        case .unavailable: return "AI chat unavailable in this build · Research summaries available"
        case .signInRequired: return "Sign in to use Gemma 4 31B · Research summaries available"
        case .consentRequired: return "Enable Gemma 4 31B to chat · Research summaries available"
        case .ready: return "Gemma 4 31B · Beta · Hosted AI"
        }
    }
}

/// The account-bound caller supplies a fresh token for each request. Neither
/// account identifiers nor runtime configuration are part of the wire format.
struct HostedAskPLONNProvider: AskPLONNProvider {
    typealias TokenProvider = @Sendable () async throws -> String
    typealias Transport = @Sendable (URLRequest) async throws -> (Data, HTTPURLResponse)
    static let endpointURL = URL(string: "https://api.oakas.ai/plonn/v1/chat/cloud")!
    static let modelName = "gemma4:31b"
    static let pinnedDigest = "08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68"
    static let maximumRequestBytes = 16 * 1024
    static let maximumResponseBytes = 32 * 1024
    static let timeout: TimeInterval = 45
    var displayName: String { "Gemma 4 31B" }
    private let endpoint: URL
    private let tokenProvider: TokenProvider
    private let transport: Transport

    init(endpoint: URL = Self.endpointURL, tokenProvider: @escaping TokenProvider, transport: Transport? = nil) {
        self.endpoint = endpoint
        self.tokenProvider = tokenProvider
        self.transport = transport ?? { try await Self.hostedRead($0) }
    }

    static func allowedEndpoint(_ url: URL) -> Bool {
        url.absoluteString == endpointURL.absoluteString
    }

    private struct WireEvidence: Encodable {
        let id: String
        let kind: AskPLONNIntent
        let title: String
        let text: String
        let source: String
        let source_urls: [String]
        init(_ evidence: AskPLONNEvidence) {
            id = evidence.id; kind = evidence.kind; title = evidence.title
            text = evidence.text; source = evidence.source; source_urls = evidence.sourceURLs
        }
    }
    private struct WireTurn: Encodable {
        let question: String
        let intent: AskPLONNIntent
        let general_response: String?
        init(_ turn: AskPLONNQuestionContext) {
            question = turn.question; intent = turn.intent; general_response = turn.generalResponse
        }
    }
    private struct WireRequest: Encodable {
        let schema_version = 1
        let consent_version = AccountAIConsentPolicy.version
        let request_id = UUID().uuidString.lowercased()
        let question: String
        let intent: AskPLONNIntent
        let evidence: [WireEvidence]
        let prior_turns: [WireTurn]
        let purpose: AskPLONNResearchPurpose?
        let matchup_teams: [String]?
        let model_draft: AskPLONNModelDraft?
    }

    private static func wireText(_ value: String, maximum: Int) -> Bool {
        !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && value.unicodeScalars.count <= maximum &&
            value.unicodeScalars.allSatisfy { $0.value >= 32 || $0 == "\n" || $0 == "\t" }
    }

    func answer(_ request: AskPLONNRequest) async throws -> AskPLONNAnswer {
        try Task.checkCancellation()
        // Validate the destination before calling any credential closure.
        guard Self.allowedEndpoint(endpoint) else { throw HostedAskPLONNError.invalidEndpoint }
        _ = try AskPLONNCore.boundedRequest(request)
        guard Self.wireText(request.question, maximum: 1000) else { throw AskPLONNError.questionTooLong }
        let facts = request.evidence.filter {
            $0.kind == request.intent && $0.availability == .available &&
                !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
                !$0.source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        guard request.intent == .generalChat || (!facts.isEmpty && facts.count <= 4) else { throw AskPLONNError.unsupportedQuestion }
        guard facts.allSatisfy({
            Self.wireText($0.id, maximum: 200) && Self.wireText($0.title, maximum: 200) &&
                Self.wireText($0.text, maximum: 2000) && Self.wireText($0.source, maximum: 300) &&
                $0.sourceURLs.count <= 8 && $0.sourceURLs.allSatisfy {
                    Self.wireText($0, maximum: 1000) && !$0.unicodeScalars.contains(where: CharacterSet.whitespacesAndNewlines.contains)
                }
        }) else { throw AskPLONNError.invalidEvidence }
        let prior = request.priorTurns ?? []
        guard prior.allSatisfy({ Self.wireText($0.question, maximum: 1000) &&
            ($0.generalResponse == nil || Self.wireText($0.generalResponse!, maximum: 2000)) }),
            prior.reduce(0, { $0 + $1.question.unicodeScalars.count + ($1.generalResponse?.unicodeScalars.count ?? 0) }) <= 3000
        else { throw AskPLONNError.invalidHistory }
        let body = try JSONEncoder().encode(WireRequest(question: request.question, intent: request.intent,
                                                       evidence: facts.map(WireEvidence.init), prior_turns: prior.map(WireTurn.init),
                                                       purpose: request.researchPurpose, matchup_teams: request.matchupTeams, model_draft: request.modelDraft))
        guard body.count <= Self.maximumRequestBytes else { throw HostedAskPLONNError.oversizedPayload }
        try Task.checkCancellation()
        let token: String
        do { token = try await tokenProvider() }
        catch let error as URLError where error.code == .cancelled { throw CancellationError() }
        try Task.checkCancellation()
        guard !token.isEmpty, token.utf8.count <= 8192,
              token.utf8.allSatisfy({ (33...126).contains($0) }) else { throw HostedAskPLONNError.authenticationRequired }
        var http = URLRequest(url: endpoint, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: Self.timeout)
        http.httpMethod = "POST"
        http.httpBody = body
        http.httpShouldHandleCookies = false
        http.setValue("application/json", forHTTPHeaderField: "Content-Type")
        http.setValue("application/json", forHTTPHeaderField: "Accept")
        http.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        let data: Data
        do {
            let (received, response) = try await transport(http)
            try Task.checkCancellation()
            guard response.url?.absoluteString == endpoint.absoluteString else { throw HostedAskPLONNError.invalidResponse }
            guard received.count <= Self.maximumResponseBytes else { throw HostedAskPLONNError.oversizedPayload }
            try Self.validateStatus(response.statusCode, body: received)
            data = received
        } catch let error as URLError where error.code == .cancelled { throw CancellationError() }
        catch let error as URLError where error.code == .timedOut { throw HostedAskPLONNError.unavailable }
        let payload: [String: Any]
        do {
            var shape = HostedChatJSONShape(data: data)
            try shape.validate()
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw HostedAskPLONNError.invalidResponse }
            payload = object
        } catch { throw HostedAskPLONNError.invalidResponse }
        guard Set(payload.keys) == Set(["schema_version", "model", "provider", "response"]),
              let version = payload["schema_version"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(),
              !["f", "d"].contains(String(cString: version.objCType)), version.intValue == 2,
              let response = payload["response"] as? [String: Any] else { throw HostedAskPLONNError.invalidResponse }
        guard payload["model"] as? String == Self.modelName, payload["provider"] as? String == "ollama_cloud"
        else { throw HostedAskPLONNError.unverifiedModel }
        let answer: AskPLONNAnswer
        if request.researchPurpose == .modelCreation {
            guard Set(response.keys) == ["draft"], let value = response["draft"] as? [String: Any],
                  Set(value.keys) == AskPLONNModelDraft.wireKeys else { throw HostedAskPLONNError.invalidResponse }
            if let line = value["line"] as? NSNumber, CFGetTypeID(line) == CFBooleanGetTypeID() { throw HostedAskPLONNError.invalidResponse }
            let draft: AskPLONNModelDraft
            do { draft = try JSONDecoder().decode(AskPLONNModelDraft.self, from: JSONSerialization.data(withJSONObject: value)) }
            catch { throw HostedAskPLONNError.invalidResponse }
            guard draft.isValid else { throw HostedAskPLONNError.invalidResponse }
            answer = AskPLONNAnswer(text: draft.guidance, evidence: [], mode: .generalExplanation,
                                   providerLabel: displayName, modelDraft: draft)
        } else if request.intent == .generalChat {
            guard Set(response.keys) == Set(["kind", "message"]),
                  let rawKind = response["kind"] as? String, let kind = AskPLONNGeneralResponseKind(rawValue: rawKind),
                  let message = response["message"] as? String,
                  Self.wireText(message, maximum: 2000), AskPLONNCore.validGeneralText(message)
            else { throw HostedAskPLONNError.invalidResponse }
            let safeMessage: String
            switch kind {
            case .explanation: safeMessage = message
            case .needs_research: safeMessage = AskPLONNCore.generalResearchGuidance().text
            case .unsupported: safeMessage = AskPLONNCore.generalUnsupportedGuidance().text
            }
            answer = AskPLONNAnswer(text: safeMessage, evidence: [], mode: .generalExplanation,
                                   providerLabel: displayName, requiresResearchContext: kind == .needs_research)
        } else {
            let isMatchup = request.researchPurpose == .matchupInsights
            let expectedKeys = isMatchup ? ["supported", "evidence_ids", "summary"] : ["supported", "evidence_ids"]
            guard Set(response.keys) == Set(expectedKeys),
                  let supported = Self.boolean(response["supported"]), let ids = response["evidence_ids"] as? [String]
            else { throw HostedAskPLONNError.invalidResponse }
            var summary: String?
            if isMatchup {
                guard let text = response["summary"] as? String else { throw HostedAskPLONNError.invalidResponse }
                guard supported || text.isEmpty else { throw AskPLONNError.unsupportedAnswer }
                summary = text
            }
            guard ids.count <= (isMatchup ? 3 : 4), Set(ids).count == ids.count,
                  ids.allSatisfy({ id in facts.contains { $0.id == id } }), supported == !ids.isEmpty
            else { throw AskPLONNError.unsupportedAnswer }
            guard supported else { throw AskPLONNError.unsupportedQuestion }
            let selected = ids.compactMap { id in facts.first { $0.id == id } }
            answer = AskPLONNAnswer(text: selected.map(\.text).joined(separator: "\n\n"), evidence: selected,
                                   mode: .provider, providerLabel: displayName, matchupSummary: summary)
        }
        try Task.checkCancellation()
        return try AskPLONNCore.validateProviderAnswer(answer, for: request)
    }

    private static func boolean(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }

    static func validateStatus(_ status: Int, body: Data? = nil) throws {
        // Only a known gateway reason changes presentation. Proxy/provider 429s
        // are temporary rate limits, not proof that a daily allowance was used.
        // Never surface server text; malformed/duplicate JSON uses the fallback.
        if (status == 502 || status == 429), let body, body.count <= maximumResponseBytes {
            var shape = HostedChatJSONShape(data: body)
            if (try? shape.validate()) != nil,
               let payload = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
               payload["status"] as? String == "unavailable" {
                if status == 502, payload["reason"] as? String == "chat_response_invalid" {
                    throw HostedAskPLONNError.invalidResponse
                }
                if status == 429, payload["reason"] as? String == "chat_quota_exceeded" {
                    throw HostedAskPLONNError.dailyAllowance
                }
            }
        }
        switch status {
        case 200: return
        case 401, 403: throw HostedAskPLONNError.authenticationRequired
        case 409: throw HostedAskPLONNError.capacity
        case 429: throw HostedAskPLONNError.rateLimited
        case 502, 503, 504: throw HostedAskPLONNError.unavailable
        case 413: throw HostedAskPLONNError.oversizedPayload
        default: throw HostedAskPLONNError.invalidResponse
        }
    }

    static func sessionConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.httpShouldSetCookies = false
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.connectionProxyDictionary = [:]
        return configuration
    }

    private static func hostedRead(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let delegate = BoundedHostedChatResponse(expectedURL: endpointURL)
        let session = URLSession(configuration: sessionConfiguration(), delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        return try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                delegate.start(session: session, request: request, continuation: continuation)
            }
        }, onCancel: { delegate.cancel() })
    }
}

/// Foundation otherwise silently overwrites duplicate JSON keys. Bound nesting
/// and reject duplicate keys before decoding the gateway's small fixed schema.
private struct HostedChatJSONShape {
    private let bytes: [UInt8]
    private var cursor = 0
    init(data: Data) { bytes = Array(data) }
    mutating func validate() throws {
        try value(depth: 0)
        whitespace()
        guard cursor == bytes.count else { throw HostedAskPLONNError.invalidResponse }
    }
    private mutating func whitespace() {
        while cursor < bytes.count && [9, 10, 13, 32].contains(bytes[cursor]) { cursor += 1 }
    }
    private mutating func take(_ byte: UInt8) -> Bool {
        whitespace()
        guard cursor < bytes.count, bytes[cursor] == byte else { return false }
        cursor += 1
        return true
    }
    private mutating func string() throws -> String {
        whitespace()
        let start = cursor
        guard take(34) else { throw HostedAskPLONNError.invalidResponse }
        while cursor < bytes.count {
            let byte = bytes[cursor]
            cursor += 1
            if byte == 92 {
                guard cursor < bytes.count else { throw HostedAskPLONNError.invalidResponse }
                cursor += 1
            } else if byte == 34 {
                let token = Data(bytes[start..<cursor])
                guard String(data: token, encoding: .utf8) != nil,
                      let text = try JSONSerialization.jsonObject(with: token, options: [.fragmentsAllowed]) as? String
                else { throw HostedAskPLONNError.invalidResponse }
                return text
            }
        }
        throw HostedAskPLONNError.invalidResponse
    }
    private mutating func value(depth: Int) throws {
        whitespace()
        guard depth <= 8, cursor < bytes.count else { throw HostedAskPLONNError.invalidResponse }
        switch bytes[cursor] {
        case 123:
            cursor += 1
            if take(125) { return }
            var keys = Set<String>()
            while true {
                let key = try string()
                guard keys.insert(key).inserted, take(58) else { throw HostedAskPLONNError.invalidResponse }
                try value(depth: depth + 1)
                if take(125) { return }
                guard take(44) else { throw HostedAskPLONNError.invalidResponse }
            }
        case 91:
            cursor += 1
            if take(93) { return }
            while true {
                try value(depth: depth + 1)
                if take(93) { return }
                guard take(44) else { throw HostedAskPLONNError.invalidResponse }
            }
        case 34: _ = try string()
        default:
            let start = cursor
            while cursor < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[cursor]) { cursor += 1 }
            guard cursor > start else { throw HostedAskPLONNError.invalidResponse }
            // JSONSerialization subsequently validates literal/number syntax.
        }
    }
}

/// Enforces the body limit while receiving chunks, before unbounded allocation.
final class BoundedHostedChatResponse: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let expectedURL: URL
    private let lock = NSLock()
    private var task: URLSessionDataTask?
    private var continuation: CheckedContinuation<(Data, HTTPURLResponse), Error>?
    private var terminal: Result<(Data, HTTPURLResponse), Error>?
    private var data = Data()
    private var response: HTTPURLResponse?

    init(expectedURL: URL) { self.expectedURL = expectedURL }
    func start(session: URLSession, request: URLRequest, continuation: CheckedContinuation<(Data, HTTPURLResponse), Error>) {
        lock.lock()
        if let terminal { lock.unlock(); continuation.resume(with: terminal); return }
        self.continuation = continuation
        let task = session.dataTask(with: request); self.task = task
        lock.unlock(); task.resume()
    }
    func cancel() { finish(.failure(CancellationError())) }
    private func finish(_ result: Result<(Data, HTTPURLResponse), Error>) {
        lock.lock()
        guard terminal == nil else { lock.unlock(); return }
        terminal = result
        let continuation = self.continuation; self.continuation = nil
        let task = self.task; self.task = nil
        lock.unlock()
        task?.cancel(); continuation?.resume(with: result)
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        do {
            guard let http = response as? HTTPURLResponse, http.url?.absoluteString == expectedURL.absoluteString
            else { throw HostedAskPLONNError.invalidResponse }
            // These statuses have known structured reasons. Read their bounded
            // bodies before classification; other failures still fail early.
            if http.statusCode != 502 && http.statusCode != 429 {
                try HostedAskPLONNProvider.validateStatus(http.statusCode)
            }
            guard response.expectedContentLength <= HostedAskPLONNProvider.maximumResponseBytes
            else { throw HostedAskPLONNError.oversizedPayload }
            lock.lock()
            guard terminal == nil else { lock.unlock(); completionHandler(.cancel); return }
            self.response = http
            lock.unlock(); completionHandler(.allow)
        } catch { completionHandler(.cancel); finish(.failure(error)) }
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive chunk: Data) {
        lock.lock()
        guard terminal == nil else { lock.unlock(); return }
        guard chunk.count <= HostedAskPLONNProvider.maximumResponseBytes - data.count
        else { lock.unlock(); finish(.failure(HostedAskPLONNError.oversizedPayload)); return }
        data.append(chunk); lock.unlock()
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            if (error as? URLError)?.code == .cancelled { finish(.failure(CancellationError())) }
            else { finish(.failure(error)) }
            return
        }
        lock.lock(); let response = self.response; let data = self.data; lock.unlock()
        guard let response else { finish(.failure(HostedAskPLONNError.invalidResponse)); return }
        finish(.success((data, response)))
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
        finish(.failure(HostedAskPLONNError.invalidResponse))
    }
}
