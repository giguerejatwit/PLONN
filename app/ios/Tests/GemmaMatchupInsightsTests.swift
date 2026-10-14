import XCTest
@testable import PLONNCore

final class GemmaMatchupInsightsTests: XCTestCase {
    private func insight(_ id: String = "fixture-1", team: String = "Fixture Home", text: String = "Fixture Home averages 12.3 points per game.",
                         evidence: String = "Fixture Home: fixture season rates · 12 games. Team history, not a player forecast.",
                         marketID: String? = "fixture-prop", topic: String = "Scoring") -> MatchupInsight {
        MatchupInsight(id: id, text: text, evidence: evidence, marketID: marketID, topic: topic, team: team, hasPairedEvidence: true)
    }

    private func snapshot(_ insights: [MatchupInsight]? = nil, contextID: String = "fixture-matchup", ownerID: String? = "fixture-owner",
                          home: String = "Fixture Home", away: String = "Fixture Away") -> GemmaMatchupInsightSnapshot {
        GemmaMatchupInsightSnapshot(contextID: contextID, homeTeam: home, awayTeam: away, ownerID: ownerID,
                                   insights: insights ?? [insight()])
    }

    private func assertFailure(_ expected: AskPLONNError, action: () async throws -> [MatchupInsight],
                               file: StaticString = #filePath, line: UInt = #line) async {
        do {
            _ = try await action()
            XCTFail("Expected \(expected)", file: file, line: line)
        } catch { XCTAssertEqual(error as? AskPLONNError, expected, file: file, line: line) }
    }

    private func assertCancelled(_ task: Task<[MatchupInsight], Error>, file: StaticString = #filePath, line: UInt = #line) async {
        do {
            _ = try await task.value
            XCTFail("Expected cancellation", file: file, line: line)
        } catch { XCTAssertTrue(error is CancellationError, file: file, line: line) }
    }

    func testRequestIsBoundedStaticDescriptiveResearchAndSelectionPreservesOriginals() async throws {
        let originals = [insight(), insight("fixture-2", team: "Fixture Away", marketID: nil),
                         insight("fixture-3", topic: "Assists"), insight("fixture-4", team: "Fixture Away", topic: "Rebounds")]
        let provider = InsightRecordingProvider { request in
            InsightRecordingProvider.selection(request, indices: [2, 1, 0])
        }
        let result = try await GemmaMatchupInsightSession().generate(snapshot(originals), provider: provider)
        XCTAssertEqual(result, [originals[2], originals[1], originals[0]])
        let requests = await provider.requests
        let request = try XCTUnwrap(requests.first)
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(request.intent, .recentForm)
        XCTAssertEqual(request.ownerID, "fixture-owner")
        XCTAssertNil(request.priorTurns)
        XCTAssertTrue(request.question.contains("three descriptive observations"))
        XCTAssertTrue(request.question.contains("Do not recommend a bet"))
        XCTAssertTrue(request.question.contains("team average as a player forecast"))
        XCTAssertEqual(request.evidence.count, 4)
        for (fact, original) in zip(request.evidence, originals) {
            XCTAssertEqual(fact.id, original.id)
            XCTAssertEqual(fact.title, original.topic)
            XCTAssertEqual(fact.text, original.text + "\n" + original.evidence)
            XCTAssertEqual(fact.source, original.evidence)
            XCTAssertEqual(fact.kind, .recentForm)
            XCTAssertEqual(fact.availability, .available)
            XCTAssertNil(fact.capturedAt)
            XCTAssertNil(fact.sampleSize)
            XCTAssertTrue(fact.sourceURLs.isEmpty)
        }
    }

    func testInvalidInputNeverCallsProviderIncludingConflictingDuplicateIDs() async {
        let valid = insight()
        let invalid: [GemmaMatchupInsightSnapshot] = [
            snapshot([]), snapshot((0..<5).map { insight("fixture-\($0)") }),
            snapshot([valid, valid]), snapshot([valid, insight(text: "Conflicting 999 points.")]),
            snapshot([insight(" \n")]), snapshot([insight(text: " \n")]), snapshot([insight(evidence: " \n")]),
            snapshot([insight(topic: " \n")]), snapshot([insight(team: "Unrelated Team")]),
            snapshot([insight(marketID: " \n")]), snapshot(contextID: " \n"), snapshot(ownerID: " \n"),
            snapshot(home: ""), snapshot(away: "Fixture Home"),
            snapshot([insight(String(repeating: "i", count: 201))]),
            snapshot([insight(topic: String(repeating: "t", count: 201))]),
            snapshot([insight(text: String(repeating: "t", count: 2001))]),
            snapshot([insight(evidence: String(repeating: "s", count: 301))]),
            snapshot([insight(text: "Observed \u{0} control character")])
        ]
        let provider = InsightRecordingProvider()
        let session = GemmaMatchupInsightSession()
        for candidate in invalid {
            await assertFailure(.invalidEvidence) { try await session.generate(candidate, provider: provider) }
        }
        let requests = await provider.requests
        XCTAssertTrue(requests.isEmpty)
    }

    func testUnsafeStatisticsCannotReachModelThroughObservationEngine() async {
        let matchup = InsightMatchup(sport: "NBA", eventID: "fixture", homeTeam: "Fixture Home", awayTeam: "Fixture Away", commence: nil)
        let provider = InsightRecordingProvider()
        for invalidRate in [Double.nan, .infinity, -.infinity, -1] {
            let profile = InsightTeamProfile(team: "Fixture Home", sample: 12, offense: ["avgPoints": invalidRate], allowed: [:], source: "fixture")
            let observed = MatchupInsightEngine.generate(matchup: matchup, profiles: [profile], props: [])
            XCTAssertTrue(observed.isEmpty)
            await assertFailure(.invalidEvidence) {
                try await GemmaMatchupInsightSession().generate(self.snapshot(observed), provider: provider)
            }
        }
        let requests = await provider.requests
        XCTAssertTrue(requests.isEmpty)
    }

    func testInventedNumericProseCitationsAndPropLinksAreRejected() async {
        let additions = [" Fixture Home will score 999 points.", " [1]", " https://example.com/fake-source", " Related prop: Forged Athlete, over 99 points.", " Recommended bet: over."]
        for addition in additions {
            let provider = InsightRecordingProvider { request in
                let selected = InsightRecordingProvider.selection(request)
                return AskPLONNAnswer(text: selected.text + addition, evidence: selected.evidence, mode: .provider)
            }
            await assertFailure(.unsupportedAnswer) {
                try await GemmaMatchupInsightSession().generate(self.snapshot(), provider: provider)
            }
        }
    }

    func testEveryChangedEvidenceFieldAndForeignIdentityIsRejected() async {
        let mutations: [@Sendable (inout AskPLONNEvidence) -> Void] = [
            { $0 = AskPLONNEvidence(id: "other-matchup", kind: $0.kind, title: $0.title, text: $0.text, source: $0.source) },
            { $0 = AskPLONNEvidence(id: $0.id, kind: .comparePrices, title: $0.title, text: $0.text, source: $0.source) },
            { $0 = AskPLONNEvidence(id: $0.id, kind: $0.kind, title: "Invented topic", text: $0.text, source: $0.source) },
            { $0 = AskPLONNEvidence(id: $0.id, kind: $0.kind, title: $0.title, text: "Fixture Home averages 999 points.", source: $0.source) },
            { $0 = AskPLONNEvidence(id: $0.id, kind: $0.kind, title: $0.title, text: $0.text, source: "forged source") },
            { $0.capturedAt = Date(timeIntervalSince1970: 1_000) }, { $0.sampleSize = 99 },
            { $0.availability = .stale }, { $0.availability = .unavailable },
            { $0.sourceURLs = ["https://site.api.espn.com/forged-citation"] }
        ]
        for mutation in mutations {
            let provider = InsightRecordingProvider { request in
                var fact = request.evidence[0]
                mutation(&fact)
                return AskPLONNAnswer(text: fact.text, evidence: [fact], mode: .provider)
            }
            await assertFailure(.unsupportedAnswer) {
                try await GemmaMatchupInsightSession().generate(self.snapshot(), provider: provider)
            }
        }
    }

    func testEmptyDuplicateExcessAndNonProviderSelectionsAreRejected() async {
        let originals = (0..<4).map { insight("fixture-\($0)") }
        let transforms: [@Sendable (AskPLONNRequest) -> AskPLONNAnswer] = [
            { InsightRecordingProvider.selection($0, indices: []) },
            { InsightRecordingProvider.selection($0, indices: [0, 0]) },
            { InsightRecordingProvider.selection($0, indices: [0, 1, 2, 3]) },
            { request in
                var answer = InsightRecordingProvider.selection(request)
                answer.mode = .localResearch
                return answer
            },
            { request in
                var answer = InsightRecordingProvider.selection(request)
                answer.mode = .generalExplanation
                return answer
            },
            { request in
                var answer = InsightRecordingProvider.selection(request)
                answer.requiresResearchContext = true
                return answer
            }
        ]
        for transform in transforms {
            await assertFailure(.unsupportedAnswer) {
                try await GemmaMatchupInsightSession().generate(self.snapshot(originals), provider: InsightRecordingProvider(transform: transform))
            }
        }
    }

    func testExactSnapshotIsCachedButEveryOriginalFieldChangeRegenerates() async throws {
        let session = GemmaMatchupInsightSession(), provider = InsightRecordingProvider()
        let first = try await session.generate(snapshot(), provider: provider)
        let cached = try await session.generate(snapshot(), provider: provider)
        XCTAssertEqual(first, cached)
        var requests = await provider.requests
        XCTAssertEqual(requests.count, 1)
        var changedPairing = insight()
        changedPairing.hasPairedEvidence = false
        let changes = [insight(text: "Fixture Home averages 13.3 points per game."),
                       insight(evidence: "Fixture official source · 13 games. Team history, not a player forecast."),
                       insight(marketID: "changed-prop"), insight(topic: "Changed topic"), insight(team: "Fixture Away"),
                       insight("changed-id"), changedPairing]
        for change in changes {
            let result = try await session.generate(snapshot([change]), provider: provider)
            XCTAssertEqual(result, [change])
        }
        requests = await provider.requests
        XCTAssertEqual(requests.count, 8)
        let pair = [insight(), insight("fixture-2", team: "Fixture Away")]
        _ = try await session.generate(snapshot(pair), provider: provider)
        let reordered = try await session.generate(snapshot(Array(pair.reversed())), provider: provider)
        XCTAssertEqual(reordered, Array(pair.reversed()))
        requests = await provider.requests
        XCTAssertEqual(requests.count, 10)
    }

    func testContextReopeningOwnerChangesAndExplicitInvalidationDropCache() async throws {
        let session = GemmaMatchupInsightSession(), provider = InsightRecordingProvider()
        for candidate in [snapshot(), snapshot(contextID: "other"), snapshot(), snapshot(ownerID: "other-owner"), snapshot(ownerID: nil)] {
            _ = try await session.generate(candidate, provider: provider)
        }
        await session.invalidate()
        _ = try await session.generate(snapshot(ownerID: nil), provider: provider)
        let requests = await provider.requests
        XCTAssertEqual(requests.count, 6)
        XCTAssertEqual(requests.map(\.ownerID), ["fixture-owner", "fixture-owner", "fixture-owner", "other-owner", nil, nil])
    }

    func testFailureIsNotCachedOrAutomaticallyRetried() async throws {
        let provider = InsightRecordingProvider { _ in throw AskPLONNError.providerUnavailable }
        let session = GemmaMatchupInsightSession()
        await assertFailure(.providerUnavailable) { try await session.generate(self.snapshot(), provider: provider) }
        var requests = await provider.requests
        XCTAssertEqual(requests.count, 1)
        await assertFailure(.providerUnavailable) { try await session.generate(self.snapshot(), provider: provider) }
        requests = await provider.requests
        XCTAssertEqual(requests.count, 2)
        let working = InsightRecordingProvider()
        let result = try await session.generate(snapshot(), provider: working)
        XCTAssertEqual(result, [insight()])
    }

    func testConcurrentIdenticalRequestsShareOneProviderCall() async throws {
        let session = GemmaMatchupInsightSession(), provider = InsightSuspendedProvider(), candidate = snapshot()
        let first = Task { try await session.generate(candidate, provider: provider) }
        let second = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(1)
        await provider.complete(0)
        let firstResult = try await first.value, secondResult = try await second.value
        XCTAssertEqual(firstResult, candidate.insights)
        XCTAssertEqual(secondResult, candidate.insights)
        let count = await provider.callCount
        XCTAssertEqual(count, 1)
    }

    func testCancelledCallerReturnsCancellationAndLateSuccessDoesNotPopulateCache() async throws {
        let session = GemmaMatchupInsightSession(), provider = InsightSuspendedProvider(), candidate = snapshot()
        let first = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(1)
        first.cancel()
        await assertCancelled(first)
        await provider.complete(0)
        let reopened = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(2)
        await provider.complete(1)
        let result = try await reopened.value
        XCTAssertEqual(result, candidate.insights)
    }

    func testCancellationRacingProviderCompletionCannotCacheCancelledSuccess() async throws {
        let session = GemmaMatchupInsightSession(), provider = InsightSuspendedProvider(), candidate = snapshot()
        let first = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(1)
        first.cancel()
        // Complete without waiting for the actor's cancellation callback to run.
        await provider.complete(0)
        await assertCancelled(first)
        let reopened = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(2)
        await provider.complete(1)
        let result = try await reopened.value
        XCTAssertEqual(result, candidate.insights)
    }

    func testCancellingOneWaiterDoesNotCancelOtherIdenticalRequest() async throws {
        let session = GemmaMatchupInsightSession(), provider = InsightSuspendedProvider(), candidate = snapshot()
        let first = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(1)
        let second = Task { try await session.generate(candidate, provider: provider) }
        // Let the second caller enter the actor before cancelling the first waiter.
        for _ in 0..<10 { await Task.yield() }
        first.cancel()
        await assertCancelled(first)
        await provider.complete(0)
        let result = try await second.value
        XCTAssertEqual(result, candidate.insights)
        let count = await provider.callCount
        XCTAssertEqual(count, 1)
    }

    func testChangedContextOwnerAndFactsSupersedePendingResults() async throws {
        for changed in [snapshot(contextID: "different-matchup"), snapshot(ownerID: "different-owner"),
                        snapshot([insight(evidence: "Changed official source · 10 games.")])] {
            let session = GemmaMatchupInsightSession(), provider = InsightSuspendedProvider(), original = snapshot()
            let first = Task { try await session.generate(original, provider: provider) }
            await provider.waitForCalls(1)
            let second = Task { try await session.generate(changed, provider: provider) }
            await provider.waitForCalls(2)
            await assertFailure(.superseded) { try await first.value }
            // A provider that ignores cancellation must still be unable to publish old evidence.
            await provider.complete(0)
            await provider.complete(1)
            let result = try await second.value
            XCTAssertEqual(result, changed.insights)
            let cached = try await session.generate(changed, provider: provider)
            XCTAssertEqual(cached, changed.insights)
            let count = await provider.callCount
            XCTAssertEqual(count, 2)
        }
    }

    func testInvalidationSupersedesPendingAndRejectsLateCompletion() async throws {
        let session = GemmaMatchupInsightSession(), provider = InsightSuspendedProvider(), candidate = snapshot()
        let task = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(1)
        await session.invalidate()
        await assertFailure(.superseded) { try await task.value }
        await provider.complete(0)
        let next = Task { try await session.generate(candidate, provider: provider) }
        await provider.waitForCalls(2)
        await provider.complete(1)
        let result = try await next.value
        XCTAssertEqual(result, candidate.insights)
    }

    func testProviderCancellationIsPropagatedAndNotCached() async {
        for provider in [InsightRecordingProvider { _ in throw CancellationError() },
                         InsightRecordingProvider { _ in throw URLError(.cancelled) }] {
            let session = GemmaMatchupInsightSession(), candidate = snapshot()
            let task = Task { try await session.generate(candidate, provider: provider) }
            await assertCancelled(task)
            let requests = await provider.requests
            XCTAssertEqual(requests.count, 1)
        }
    }

    func testExplanationHasItsOwnCacheAndCarriesModelProseWithOriginalLinks() async throws {
        let candidate = snapshot()
        let summary = "Fixture Home's recorded scoring production meets Fixture Away's observed allowance. The historical sample limits the comparison."
        let provider = InsightRecordingProvider { request in
            var answer = InsightRecordingProvider.selection(request)
            if request.researchPurpose == .matchupInsights { answer.matchupSummary = summary }
            return answer
        }
        let session = GemmaMatchupInsightSession()
        let originalSelection = try await session.generate(candidate, provider: provider)
        let result = try await session.explain(candidate, provider: provider)
        XCTAssertEqual(result.insights, originalSelection)
        XCTAssertEqual(result.summary, summary)
        XCTAssertNotEqual(result.summary, result.insights.first?.text)
        let reopened = try await session.explain(candidate, provider: provider)
        XCTAssertEqual(reopened, result)
        var requests = await provider.requests
        XCTAssertEqual(requests.count, 2)
        XCTAssertNil(requests[0].researchPurpose)
        XCTAssertEqual(requests[1].researchPurpose, .matchupInsights)
        XCTAssertEqual(requests[1].matchupTeams, [candidate.homeTeam, candidate.awayTeam])
        await session.invalidate()
        let refreshed = try await session.explain(candidate, provider: provider)
        XCTAssertEqual(refreshed, result)
        requests = await provider.requests
        XCTAssertEqual(requests.count, 3)
    }

    func testSelectionOnlyProviderCannotPretendItGeneratedAnExplanation() async {
        let provider = InsightRecordingProvider(), session = GemmaMatchupInsightSession()
        for _ in 0..<2 {
            do {
                _ = try await session.explain(snapshot(), provider: provider)
                XCTFail("A selected original sentence is not generated commentary.")
            } catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        }
        let requests = await provider.requests
        XCTAssertEqual(requests.count, 2, "An invalid summary must not populate the successful cache.")
    }

    func testNHLExplanationRetainsFullBasisAfterGemmaPrioritizesThreeOfFourRows() async throws {
        let original = [
            insight("calgary-goals", team: "Calgary Flames", text: "Calgary Flames averages 1.5 goals per game; Anaheim Ducks allows 2.8 goals per game.", marketID: nil, topic: "Goals"),
            insight("anaheim-goals", team: "Anaheim Ducks", text: "Anaheim Ducks averages 3.5 goals per game; Calgary Flames allows 5.8 goals per game.", marketID: nil, topic: "Goals"),
            insight("calgary-shots", team: "Calgary Flames", text: "Calgary Flames averages 27.8 shots on goal per game; Anaheim Ducks allows 24.0 shots on goal per game.", marketID: nil, topic: "Shots"),
            insight("anaheim-shots", team: "Anaheim Ducks", text: "Anaheim Ducks averages 20.0 shots on goal per game; Calgary Flames allows 30.0 shots on goal per game.", marketID: nil, topic: "Shots")]
        let generated = "The Calgary Flames average fewer goals per game than the Anaheim Ducks, while the Calgary Flames allow more goals per game than the Anaheim Ducks allow. Regarding shot patterns, the Calgary Flames average more shots on goal per game than the Anaheim Ducks allow, whereas the Anaheim Ducks average fewer shots on goal per game than the Calgary Flames allow. These observations are based on a limited sample of a few games."
        let candidate = snapshot(original, home: "Calgary Flames", away: "Anaheim Ducks")
        let provider = InsightRecordingProvider { request in
            var response = InsightRecordingProvider.selection(request, indices: [2, 0, 1])
            if request.researchPurpose == .matchupInsights { response.matchupSummary = generated }
            return response
        }
        let session = GemmaMatchupInsightSession()
        let selection = try await session.generate(candidate, provider: provider)
        XCTAssertEqual(selection, [original[2], original[0], original[1]])
        let result = try await session.explain(candidate, provider: provider)
        XCTAssertEqual(result.summary, generated)
        XCTAssertEqual(result.insights, [original[2], original[0], original[1], original[3]])
        XCTAssertTrue(result.insights.allSatisfy { $0.marketID == nil })
        let reopened = try await session.explain(candidate, provider: provider)
        XCTAssertEqual(reopened, result)
        let requests = await provider.requests
        XCTAssertEqual(requests.count, 2)
    }

    func testLiveNHLGoalOnlyFallbackGetsSeparateGeneratedExplanationWithoutInventingShots() async throws {
        let start = Date(timeIntervalSince1970: 1_800_000_000)
        let matchup = InsightMatchup(sport: "NHL", eventID: "fixture-live-nhl", homeTeam: "Fixture Home", awayTeam: "Fixture Away",
                                    commence: ISO8601DateFormatter().string(from: start))
        let season = ["Fixture Home", "Fixture Away"].map {
            InsightTeamProfile(team: $0, sample: 4, offense: ["avgGoals": 3, "avgShots": 25],
                               allowed: ["avgGoals": 4, "avgShots": 30], source: "Fixture season", periodKey: "fixture-season",
                               through: start.addingTimeInterval(60))
        }
        let recent = ["Fixture Home", "Fixture Away"].compactMap { team in
            MatchupInsightSources.recent(team: team, sport: "NHL", games: [
                InsightGameObservation(id: team + "-earlier-a", date: start.addingTimeInterval(-172800), scored: 3, allowed: 2),
                InsightGameObservation(id: team + "-earlier-b", date: start.addingTimeInterval(-86400), scored: 4, allowed: 3)
            ], cutoff: start)
        }
        let original = MatchupInsightEngine.generate(matchup: matchup, profiles: season, props: [], recentProfiles: recent)
        XCTAssertEqual(original.count, 2)
        XCTAssertTrue(original.allSatisfy { $0.topic == "Goals" && $0.hasPairedEvidence && $0.marketID == nil })
        XCTAssertFalse(original.contains { $0.text.lowercased().contains("shots") })
        let candidate = snapshot(original, contextID: "fixture-live-nhl")
        let summary = "Fixture Home and Fixture Away have similar recorded goal production and opposing goal allowance. These short historical samples limit the comparison, not a prediction."
        let provider = InsightRecordingProvider { request in
            var answer = InsightRecordingProvider.selection(request, indices: [1, 0])
            if request.researchPurpose == .matchupInsights { answer.matchupSummary = summary }
            return answer
        }
        let session = GemmaMatchupInsightSession()
        let selected = try await session.generate(candidate, provider: provider)
        await session.invalidate()
        let reselected = try await session.generate(candidate, provider: provider)
        XCTAssertEqual(selected, reselected, "Refreshing the old selector legitimately returns the same NHL sentences.")
        let explained = try await session.explain(candidate, provider: provider)
        XCTAssertEqual(explained.insights, selected)
        XCTAssertEqual(explained.summary, summary)
        XCTAssertFalse(explained.summary!.contains("shots"))
        XCTAssertNotEqual(explained.summary, explained.insights.map(\.text).joined(separator: "\n\n"))
        let requests = await provider.requests
        XCTAssertEqual(requests.count, 3)
    }
}

private actor InsightRecordingProvider: AskPLONNProvider {
    private(set) var requests: [AskPLONNRequest] = []
    let transform: @Sendable (AskPLONNRequest) throws -> AskPLONNAnswer
    init(transform: @escaping @Sendable (AskPLONNRequest) throws -> AskPLONNAnswer = { InsightRecordingProvider.selection($0) }) {
        self.transform = transform
    }
    func answer(_ request: AskPLONNRequest) async throws -> AskPLONNAnswer {
        requests.append(request)
        return try transform(request)
    }
    nonisolated static func selection(_ request: AskPLONNRequest, indices: [Int]? = nil) -> AskPLONNAnswer {
        let selected = (indices ?? Array(request.evidence.indices.prefix(3))).map { request.evidence[$0] }
        return AskPLONNAnswer(text: selected.map(\.text).joined(separator: "\n\n"), evidence: selected,
                             mode: .provider, providerLabel: "Fixture Gemma")
    }
}

/// Deliberately ignores task cancellation so tests can deliver a late model result.
private actor InsightSuspendedProvider: AskPLONNProvider {
    private var requests: [AskPLONNRequest] = []
    private var completions: [Int: CheckedContinuation<AskPLONNAnswer, Error>] = [:]
    private var observers: [(Int, CheckedContinuation<Void, Never>)] = []
    var callCount: Int { requests.count }
    func answer(_ request: AskPLONNRequest) async throws -> AskPLONNAnswer {
        let index = requests.count
        requests.append(request)
        return try await withCheckedThrowingContinuation { continuation in
            completions[index] = continuation
            let ready = observers.filter { $0.0 <= requests.count }
            observers.removeAll { $0.0 <= requests.count }
            for observer in ready { observer.1.resume() }
        }
    }
    func waitForCalls(_ count: Int) async {
        if requests.count >= count { return }
        await withCheckedContinuation { observers.append((count, $0)) }
    }
    func complete(_ index: Int) {
        completions.removeValue(forKey: index)?.resume(returning: InsightRecordingProvider.selection(requests[index]))
    }
}
