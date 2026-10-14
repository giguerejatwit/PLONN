import XCTest
@testable import PLONNCore

final class AnalyticsFeedCoreTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-10-13T12:00:00Z")!
    private func stamp(_ date: Date) -> String { ISO8601DateFormatter().string(from: date) }
    private func event() -> AnalyticsEvent { AnalyticsEvent(sport: "NBA", providerID: "espn-1", home: "Fixture Home", away: "Fixture Away", start: now.addingTimeInterval(3600), homeID: "1", awayID: "2") }
    private func quote(_ id: String = "quote-1", player: Bool = false, key: String? = nil) -> Market {
        Market(id: id, event: "odds-1", sport: "NBA", kind: player ? "prop" : "team", name: player ? "Fixture Player" : "Fixture Home", opponent: "Fixture Away", label: "Full game", aliases: "", p: 0.99, price: -110, market: key ?? (player ? "player_points" : "spreads"), book: "Fixture Book", quotes: [Quote(book: "Fixture Book", price: -110, updated: stamp(now.addingTimeInterval(-30)))], updated: stamp(now.addingTimeInterval(-30)), mode: "live", line: player ? 10.5 : -2.5, eventLabel: "Fixture Away at Fixture Home", commence: stamp(event().start), side: player ? "over" : "home")
    }
    private func proof(player: Bool = false, keys: [String]? = nil) -> AnalyticsFeedVerifiedSubject {
        AnalyticsFeedVerifiedSubject(sport: "NBA", subject: player ? .player : .team, name: player ? "Fixture Player" : "Fixture Home", providerID: player ? "77" : "1", teamName: player ? "Fixture Home" : nil, teamID: player ? "1" : nil, marketKeys: keys ?? (player ? ["player_points"] : ["team_totals"]), event: event(), sourceURL: player ? "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/1/roster" : "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20261013", verifiedAt: now)
    }
    private func bettingCandidate(_ quotes: [Market], player: Bool = false) throws -> AnalyticsFeedCandidate {
        try XCTUnwrap(AnalyticsFeedCore.build(snapshots: quotes, verifiedSubjects: [proof(player: player, keys: [player ? "player_points" : "spreads"])], now: now).candidates.first)
    }
    private func bettingHistory(_ candidate: AnalyticsFeedCandidate) -> AnalyticsHistoryResult {
        let observation = AnalyticsObservation(id: "past", date: now.addingTimeInterval(-86400), opponent: "Fixture Away", value: 20, scored: 100, allowed: 95)
        var history = AnalyticsHistoryEngine.statistics(market: candidate.market, observations: [observation], now: now)
        history.verifiedPlayerID = candidate.subject == .player ? "77" : nil
        return history
    }
    func testBettingFeedRequiresHistoryAndRecentExactRecordedQuote() throws {
        let raw = quote()
        let candidate = try bettingCandidate([raw])
        let history = bettingHistory(candidate)
        XCTAssertEqual(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: history, now: now), [raw])
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: nil, now: now).isEmpty)
        let noStats = AnalyticsHistoryEngine.statistics(market: candidate.market, observations: [], now: now)
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: noStats, now: now).isEmpty)
        let noQuotes = try bettingCandidate([])
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: noQuotes, history: bettingHistory(noQuotes), now: now).isEmpty)
        XCTAssertEqual(AnalyticsFeedCore.search(candidates: [noQuotes], query: "Fixture Home").count, 1)
    }
    func testRecordedQuoteWindowFiltersIndividuallyWithoutMakingCacheLive() throws {
        var boundary = quote("boundary"); boundary.updated = stamp(now.addingTimeInterval(-900))
        var old = quote("old"); old.updated = stamp(now.addingTimeInterval(-901))
        var future = quote("future"); future.updated = stamp(now.addingTimeInterval(1))
        var missing = quote("missing"); missing.updated = nil
        let candidate = try bettingCandidate([boundary, old, future, missing])
        let history = bettingHistory(candidate)
        XCTAssertEqual(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: history, now: now).map(\.id), ["boundary"])
        XCTAssertEqual(AnalyticsFeedCore.eligibleComparisons(candidate: candidate, history: history, now: now).count, 2)
        let expired = try bettingCandidate([old])
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: expired, history: bettingHistory(expired), now: now).isEmpty)
    }
    func testBettingFeedRejectsWrongPlayerIdentityAndInvalidLines() throws {
        let candidate = try bettingCandidate([quote(player: true)], player: true)
        var history = bettingHistory(candidate)
        XCTAssertEqual(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: history, now: now).count, 1)
        history.verifiedPlayerID = "different"
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: history, now: now).isEmpty)
        var invalid = quote(player: true); invalid.line = nil
        let noLine = try bettingCandidate([invalid], player: true)
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: noLine, history: bettingHistory(noLine), now: now).isEmpty)
        var proof = proof(); proof.teamID = "incorrect"
        let wrongID = AnalyticsFeedVerifiedSubject(sport: "NBA", subject: .team, name: "Fixture Home", providerID: "999", marketKeys: ["spreads"], event: event(), sourceURL: proof.sourceURL, verifiedAt: now)
        let wrong = try XCTUnwrap(AnalyticsFeedCore.build(snapshots: [quote()], verifiedSubjects: [wrongID], now: now).candidates.first)
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: wrong, history: bettingHistory(wrong), now: now).isEmpty)
    }
    func testPublicSourcesPopulateMixedStatisticsFeedWithoutBookQuotes() {
        let result = AnalyticsFeedCore.build(snapshots: [], verifiedSubjects: [proof(player: true), proof()], now: now)
        XCTAssertEqual(result.candidates.map(\.subject), [.team, .player])
        XCTAssertTrue(result.candidates.allSatisfy { $0.quotes.isEmpty && $0.market.line == nil && $0.market.side == nil && $0.market.p == nil && $0.market.price == 0 && $0.market.event.isEmpty })
        XCTAssertEqual(result.candidates.first?.market.market, "team_totals")
        XCTAssertEqual(result.candidates.last?.market.market, "player_points")
    }
    func testBookSideAndLineCopiesBecomeOneStableCardWithExactQuotes() throws {
        let first = quote()
        var second = quote("quote-2"); second.book = "Other Book"; second.line = -3.5; second.price = 105; second.side = "away"
        let original = [first, second]
        let result = AnalyticsFeedCore.build(snapshots: original + [first], verifiedSubjects: [proof()], now: now)
        let candidate = try XCTUnwrap(result.candidates.first { $0.market.market == "spreads" })
        XCTAssertEqual(candidate.quotes, original)
        XCTAssertNil(candidate.market.line); XCTAssertNil(candidate.market.side)
        XCTAssertEqual(candidate.verifiedSubject?.providerID, "1")
        let reversed = AnalyticsFeedCore.build(snapshots: Array(original.reversed()), verifiedSubjects: [proof()], now: now)
        XCTAssertEqual(result.candidates.map(\.id), reversed.candidates.map(\.id))
        XCTAssertEqual(result.candidates.map { $0.quotes.map(\.id) }, reversed.candidates.map { $0.quotes.map(\.id) })
    }
    func testSearchAcrossNamesSportStatAndTeamNeedsNoSelector() {
        let candidates = AnalyticsFeedCore.build(snapshots: [], verifiedSubjects: [proof(), proof(player: true)], now: now).candidates
        XCTAssertEqual(AnalyticsFeedCore.search(candidates: candidates, query: "  nba PLAYER points ").map(\.subject), [.player])
        XCTAssertEqual(AnalyticsFeedCore.search(candidates: candidates, query: "TEAM total").map(\.subject), [.team])
        XCTAssertEqual(AnalyticsFeedCore.search(candidates: candidates, query: "Fixture Home").count, 2)
        XCTAssertEqual(AnalyticsFeedCore.search(candidates: candidates, query: "   ").map(\.id), candidates.map(\.id))
        XCTAssertTrue(AnalyticsFeedCore.search(candidates: candidates, query: "unrecorded name").isEmpty)
    }
    func testNoWinningRatePriceOrForecastPreferenceChangesIdentityOrOrder() {
        let originals = [quote(player: true), quote("team-quote")]
        let changed = originals.map { original -> Market in var copy = original; copy.p = 0.01; copy.price = 10000; copy.line = 999; return copy }
        let low = AnalyticsFeedCore.build(snapshots: originals, now: now)
        let high = AnalyticsFeedCore.build(snapshots: changed, now: now)
        XCTAssertEqual(low.candidates.map(\.id), high.candidates.map(\.id))
        XCTAssertEqual(low.candidates.map(\.subject), [.team, .player])
        XCTAssertTrue(low.candidates.allSatisfy { $0.market.p == nil && $0.market.logs == nil && $0.market.modelValidated == nil })
        var missingOdds = quote(); missingOdds.price = 0; missingOdds.line = nil; missingOdds.book = ""
        let seed = AnalyticsFeedCore.build(snapshots: [missingOdds], now: now)
        XCTAssertEqual(seed.candidates.count, 1); XCTAssertTrue(seed.candidates[0].quotes.isEmpty)
    }
    func testConflictingSnapshotsAndPublicIdentitiesAreQuarantined() {
        let first = quote()
        var conflict = first; conflict.name = "Wrong Team"
        XCTAssertTrue(AnalyticsFeedCore.build(snapshots: [first, conflict], now: now).candidates.isEmpty)
        let other = AnalyticsFeedVerifiedSubject(sport: "NBA", subject: .team, name: "Fixture Home", providerID: "999", marketKeys: ["h2h"], event: event(), sourceURL: proof().sourceURL, verifiedAt: now)
        let result = AnalyticsFeedCore.build(snapshots: [first], verifiedSubjects: [proof(), other], now: now)
        XCTAssertTrue(result.candidates.isEmpty)
        XCTAssertFalse(result.warnings.isEmpty)
        var eventConflict = quote("second"); eventConflict.eventLabel = "Wrong Away at Wrong Home"
        let scopes = AnalyticsFeedCore.build(snapshots: [first, eventConflict], now: now)
        XCTAssertEqual(scopes.candidates.count, 1); XCTAssertTrue(scopes.candidates[0].quotes.isEmpty)
    }
    func testPlayerComparisonRequiresCurrentRosterProofAndExactScope() throws {
        let raw = quote(player: true)
        let noProof = try XCTUnwrap(AnalyticsFeedCore.build(snapshots: [raw], now: now).candidates.first)
        XCTAssertEqual(noProof.quotes, [raw])
        XCTAssertTrue(AnalyticsFeedCore.eligibleComparisons(candidate: noProof, now: now).isEmpty)
        let verified = try XCTUnwrap(AnalyticsFeedCore.build(snapshots: [raw], verifiedSubjects: [proof(player: true)], now: now).candidates.first)
        let observation = AnalyticsObservation(id: "history", date: now.addingTimeInterval(-86400), opponent: "Fixture Away", value: 20)
        var history = AnalyticsHistoryEngine.statistics(market: verified.market, observations: [observation], now: now)
        history.verifiedPlayerID = "77"
        XCTAssertTrue(AnalyticsFeedCore.eligibleComparisons(candidate: verified, now: now).isEmpty)
        XCTAssertEqual(AnalyticsFeedCore.eligibleComparisons(candidate: verified, history: history, now: now), [raw])
        history.verifiedPlayerID = "wrong-athlete"
        XCTAssertTrue(AnalyticsFeedCore.eligibleComparisons(candidate: verified, history: history, now: now).isEmpty)
        XCTAssertTrue(AnalyticsFeedCore.eligibleComparisons(candidate: verified, now: now.addingTimeInterval(901)).isEmpty)
    }
    func testGameTotalsAreNotTeamPropsAndLegacyMislabeledTeamTotalIsExcluded() throws {
        var gameTotal = quote(key: "totals"); gameTotal.kind = "game"; gameTotal.name = "Over"; gameTotal.opponent = "Fixture Home"; gameTotal.side = "over"; gameTotal.line = 205.5
        let total = try XCTUnwrap(AnalyticsFeedCore.build(snapshots: [gameTotal], now: now).candidates.first)
        XCTAssertEqual(total.subject, .team); XCTAssertEqual(total.market.market, "totals"); XCTAssertEqual(total.market.name, "Fixture Home")
        var mislabeled = quote(key: "team_totals"); mislabeled.kind = "prop"; mislabeled.line = 100.5
        XCTAssertTrue(AnalyticsFeedCore.build(snapshots: [mislabeled], now: now).candidates.isEmpty)
        var genuine = quote(key: "team_totals"); genuine.line = 100.5; genuine.side = "over"
        XCTAssertEqual(AnalyticsFeedCore.build(snapshots: [genuine], now: now).candidates.first?.market.market, "team_totals")
        genuine.name = "Unknown Team"
        XCTAssertTrue(AnalyticsFeedCore.build(snapshots: [genuine], now: now).candidates.isEmpty)
    }
    func testExpiredOrUntrustedDirectoryCannotPopulateAndCapacityIsExplicit() {
        let bad = AnalyticsFeedVerifiedSubject(sport: "NBA", subject: .player, name: "Fixture Player", providerID: "77", teamName: "Fixture Home", teamID: "1", marketKeys: ["player_points"], sourceURL: "https://example.com/roster", verifiedAt: now)
        let expired = AnalyticsFeedVerifiedSubject(sport: "NBA", subject: .team, name: "Fixture Home", providerID: "1", marketKeys: ["h2h"], sourceURL: proof().sourceURL, verifiedAt: now.addingTimeInterval(-901))
        XCTAssertTrue(AnalyticsFeedCore.build(snapshots: [], verifiedSubjects: [bad, expired], now: now).candidates.isEmpty)
        let overload = AnalyticsFeedCore.build(snapshots: Array(repeating: quote(), count: 10_001), now: now)
        XCTAssertTrue(overload.candidates.isEmpty); XCTAssertFalse(overload.warnings.isEmpty)
    }
}
