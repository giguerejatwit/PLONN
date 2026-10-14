import Foundation

/// A bounded set of observations from MatchupInsightEngine for one account and matchup.
/// Original observations, sources and player-prop links always stay intact.
struct GemmaMatchupInsightSnapshot: Sendable, Equatable {
    let contextID: String
    let homeTeam: String
    let awayTeam: String
    let ownerID: String?
    let insights: [MatchupInsight]
}

struct GemmaMatchupInsightResult: Sendable, Equatable {
    let insights: [MatchupInsight]
    let summary: String?
}

actor GemmaMatchupInsightSession {
    static let maximumObservationCount = 4
    static let maximumSelectionCount = 3

    private final class WaiterCancellation: @unchecked Sendable {
        private let lock = NSLock()
        private var cancelled = false
        func cancel() { lock.lock(); cancelled = true; lock.unlock() }
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    }

    private struct Waiter {
        let continuation: CheckedContinuation<GemmaMatchupInsightResult, Error>
        let cancellation: WaiterCancellation
    }

    private struct Pending {
        let token: UUID
        let task: Task<Void, Never>
        var waiters: [UUID: Waiter]
    }

    private var snapshot: GemmaMatchupInsightSnapshot?
    private var purpose: AskPLONNResearchPurpose?
    private var cached: GemmaMatchupInsightResult?
    private var pending: Pending?

    /// Drops all account and matchup evidence, and makes any late completion inert.
    func invalidate() {
        snapshot = nil
        purpose = nil
        cached = nil
        supersedePending()
    }

    func generate(_ snapshot: GemmaMatchupInsightSnapshot, provider: any AskPLONNProvider) async throws -> [MatchupInsight] {
        try await resolve(snapshot, purpose: nil, provider: provider).insights
    }

    /// Requests qualitative model prose separately from the exact observations.
    func explain(_ snapshot: GemmaMatchupInsightSnapshot, provider: any AskPLONNProvider) async throws -> GemmaMatchupInsightResult {
        try await resolve(snapshot, purpose: .matchupInsights, provider: provider)
    }

    private func resolve(_ snapshot: GemmaMatchupInsightSnapshot, purpose: AskPLONNResearchPurpose?,
                         provider: any AskPLONNProvider) async throws -> GemmaMatchupInsightResult {
        try Task.checkCancellation()
        if self.snapshot != snapshot || self.purpose != purpose {
            invalidate()
            self.snapshot = snapshot
            self.purpose = purpose
        }
        let request = try Self.request(for: snapshot, purpose: purpose)
        if let cached {
            try Task.checkCancellation()
            return cached
        }

        let waiterID = UUID()
        let cancellation = WaiterCancellation()
        let selected = try await withTaskCancellationHandler(operation: {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<GemmaMatchupInsightResult, Error>) in
                guard !cancellation.isCancelled else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                let waiter = Waiter(continuation: continuation, cancellation: cancellation)
                if pending != nil {
                    pending?.waiters[waiterID] = waiter
                    return
                }
                let token = UUID()
                let task = Task {
                    let result: Result<GemmaMatchupInsightResult, Error>
                    do {
                        try Task.checkCancellation()
                        let answer = try await provider.answer(request)
                        try Task.checkCancellation()
                        result = .success(try Self.selectedInsights(answer, request: request, snapshot: snapshot))
                    } catch let error as URLError where error.code == .cancelled {
                        result = .failure(CancellationError())
                    } catch {
                        result = .failure(error)
                    }
                    finish(token: token, result: result)
                }
                pending = Pending(token: token, task: task, waiters: [waiterID: waiter])
            }
        }, onCancel: {
            cancellation.cancel()
            Task { await self.cancelWaiter(waiterID) }
        })
        try Task.checkCancellation()
        guard self.snapshot == snapshot, self.purpose == purpose else { throw AskPLONNError.superseded }
        return selected
    }

    private func finish(token: UUID, result: Result<GemmaMatchupInsightResult, Error>) {
        guard let pending, pending.token == token else { return }
        self.pending = nil
        if case .success(let selected) = result, pending.waiters.values.contains(where: { !$0.cancellation.isCancelled }) {
            cached = selected
        }
        // Failures are never cached or retried automatically.
        for waiter in pending.waiters.values {
            if waiter.cancellation.isCancelled { waiter.continuation.resume(throwing: CancellationError()) }
            else { waiter.continuation.resume(with: result) }
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let waiter = pending?.waiters.removeValue(forKey: id) else { return }
        waiter.continuation.resume(throwing: CancellationError())
        if pending?.waiters.isEmpty == true {
            pending?.task.cancel()
            pending = nil
        }
    }

    private func supersedePending() {
        guard let pending else { return }
        self.pending = nil
        pending.task.cancel()
        for waiter in pending.waiters.values {
            waiter.continuation.resume(throwing: waiter.cancellation.isCancelled ? CancellationError() : AskPLONNError.superseded)
        }
    }

    private static let question = """
    Select up to three descriptive observations from the supplied matchup evidence. Use only supplied evidence IDs and retain their limitations. Select observations from both teams when relevant. Do not recommend a bet, predict a player result, or treat a team average as a player forecast.
    """
    private static let explanationQuestion = """
    Describe the supplied observed team rates neutrally, without advantage claims or predictions. Use full supplied team names. Numeric observations may only be copied as complete unchanged supplied evidence sentences: preserve every semicolon, team, metric, unit and sample. Do not rewrite numbers or combine sentence fragments. Describe sample limitations without numeric paraphrases. Prioritize up to three observations with evidence_ids; all supplied observations will appear below the explanation. Include no prices, forecasts or individual player claims. Follow the exact evidence sentence contract; otherwise use qualitative text without numbers. Digits already inside an exact full supplied team name are allowed.
    """

    private static func nonblank(_ value: String) -> Bool {
        !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func wireText(_ value: String, maximum: Int) -> Bool {
        nonblank(value) && value.unicodeScalars.count <= maximum &&
            value.unicodeScalars.allSatisfy { $0.value >= 32 || $0 == "\n" || $0 == "\t" }
    }

    private static func request(for snapshot: GemmaMatchupInsightSnapshot, purpose: AskPLONNResearchPurpose?) throws -> AskPLONNRequest {
        guard nonblank(snapshot.contextID), nonblank(snapshot.homeTeam), nonblank(snapshot.awayTeam),
              snapshot.homeTeam != snapshot.awayTeam,
              snapshot.ownerID.map(nonblank) ?? true,
              !snapshot.insights.isEmpty, snapshot.insights.count <= maximumObservationCount,
              Set(snapshot.insights.map(\.id)).count == snapshot.insights.count,
              snapshot.insights.allSatisfy({
                  [$0.team == snapshot.homeTeam, $0.team == snapshot.awayTeam].contains(true) &&
                      ($0.marketID.map(nonblank) ?? true)
              }) else { throw AskPLONNError.invalidEvidence }
        let facts = snapshot.insights.map {
            AskPLONNEvidence(id: $0.id, kind: .recentForm, title: $0.topic,
                             text: $0.text + "\n" + $0.evidence, source: $0.evidence)
        }
        guard facts.allSatisfy({ wireText($0.id, maximum: 200) && wireText($0.title, maximum: 200) &&
            wireText($0.text, maximum: 2000) && wireText($0.source, maximum: 300) }),
            snapshot.insights.allSatisfy({ nonblank($0.text) }) else { throw AskPLONNError.invalidEvidence }
        return try AskPLONNCore.boundedRequest(AskPLONNRequest(question: purpose == nil ? question : explanationQuestion,
            intent: .recentForm, evidence: facts, ownerID: snapshot.ownerID, researchPurpose: purpose,
            matchupTeams: purpose == nil ? nil : [snapshot.homeTeam, snapshot.awayTeam]))
    }

    private static func selectedInsights(_ answer: AskPLONNAnswer, request: AskPLONNRequest,
                                         snapshot: GemmaMatchupInsightSnapshot) throws -> GemmaMatchupInsightResult {
        guard answer.mode == .provider, answer.evidence.count <= maximumSelectionCount else {
            throw AskPLONNError.unsupportedAnswer
        }
        let verified = try AskPLONNCore.validateProviderAnswer(answer, for: request)
        // Core validation checks every evidence field, including URLs and source metadata.
        // Returning original values also prevents a model from changing a player-prop link.
        let originals = try verified.evidence.map { fact in
            guard let original = snapshot.insights.first(where: { $0.id == fact.id }) else {
                throw AskPLONNError.unsupportedAnswer
            }
            return original
        }
        // Commentary uses the complete bounded input. Keep every observation
        // visible beneath it, in provider priority order followed by the rest.
        let visible = request.researchPurpose == .matchupInsights
            ? originals + snapshot.insights.filter { original in !originals.contains(where: { $0.id == original.id }) }
            : originals
        return GemmaMatchupInsightResult(insights: visible, summary: verified.matchupSummary)
    }
}

#if DEBUG && targetEnvironment(simulator)
/// Launch-only synthetic evidence and HTTP response. Never reads credentials or uses a network transport.
enum MatchupOfflineFixture {
    static var mode: String { ProcessInfo.processInfo.environment["PLONN_MATCHUP_INSIGHTS_FIXTURE"] ?? "" }
    static var active: Bool { ["success", "invalid"].contains(mode) }
    static let home = "Seattle Seahawks"
    static let away = "Denver Broncos"
    static let summary = "Seattle Seahawks passing production contrasts with Denver Broncos allowance. These observations describe a limited sample."
    static var insights: [MatchupInsight] {
        [MatchupInsight(id: "offline-passing", text: "Seattle Seahawks average 240 passing yards per game; Denver Broncos allow 210 passing yards per game.",
            evidence: "Offline fixture · Synthetic season rates · 5 completed games per team · Not actual team statistics.", marketID: nil, topic: "Passing", team: home, hasPairedEvidence: true)]
    }
    static var provider: HostedAskPLONNProvider {
        let invalid = mode == "invalid"
        return HostedAskPLONNProvider(tokenProvider: { "offline-fixture-not-a-credential" }, transport: { request in
            // Inject the response below the hosted wire parser; no URLSession or upstream service is called.
            let response: [String: Any] = ["schema_version": 1, "model": HostedAskPLONNProvider.modelName,
                "digest": HostedAskPLONNProvider.pinnedDigest,
                "response": ["supported": true, "evidence_ids": ["offline-passing"],
                             "summary": invalid ? "Seattle Seahawks will score 99 points." : summary]]
            return (try JSONSerialization.data(withJSONObject: response),
                    HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!)
        })
    }
}
#endif
