import XCTest
@testable import PLONNCore

final class AnalyticsTrendInsightsTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-10-13T12:00:00Z")!
    private func candidate(player: Bool = false, proof: Bool = false) -> AnalyticsFeedCandidate {
        let event = AnalyticsEvent(sport: "NBA", providerID: "event-1", home: "Fixture Home", away: "Fixture Away", start: now.addingTimeInterval(3600), homeID: "1", awayID: "2")
        let verified = proof ? AnalyticsFeedVerifiedSubject(sport: "NBA", subject: player ? .player : .team, name: player ? "Fixture Player" : "Fixture Home", providerID: player ? "77" : "1", teamName: player ? "Fixture Home" : nil, teamID: player ? "1" : nil, marketKeys: [player ? "player_points" : "team_totals"], event: event, sourceURL: player ? "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/1/roster" : "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", verifiedAt: now) : nil
        let market = Market(id: "candidate", event: "selected-event", sport: "NBA", kind: player ? "prop" : "team", name: player ? "Fixture Player" : "Fixture Home", opponent: "", label: "", aliases: "", p: nil, price: 0, market: player ? "player_points" : "team_totals", book: "", quotes: [], updated: nil, mode: "analytics")
        return AnalyticsFeedCandidate(id: "candidate", subject: player ? .player : .team, market: market, quotes: [], verifiedSubject: verified)
    }
    private func rows(_ count: Int, metadata: Bool = false) -> [AnalyticsObservation] {
        return (0..<count).map { index -> AnalyticsObservation in
            let value = Double(index + 1)
            let date = now.addingTimeInterval(-Double(index + 1) * 86400)
            let venue: String? = metadata ? (index.isMultiple(of: 2) ? "home" : "away") : nil
            let season: Int? = metadata ? 2026 : nil
            return AnalyticsObservation(id: String(index), date: date, opponent: "Fixture Away", value: value, scored: value, allowed: 100, homeAway: venue, season: season)
        }
    }
    private func history(_ observations: [AnalyticsObservation], playerID: String? = nil) -> AnalyticsHistoryResult {
        AnalyticsHistoryResult(summary: nil, rows: [], observations: observations, source: "ESPN", warnings: [], availability: "Fixture verified observations", verifiedPlayerID: playerID)
    }
    func testRecentFiveAndEarlierBaselineDoNotOverlapAndHaveOwnDates() throws {
        let insights = AnalyticsTrendInsights.make(candidate: candidate(), history: history(rows(20)), now: now)
        let recent = try XCTUnwrap(insights.first { $0.kind == .recentBaseline })
        XCTAssertEqual(recent.headline, "3.0 vs 13.0 team score")
        XCTAssertEqual(recent.sample, 20)
        XCTAssertEqual(recent.startDate, now.addingTimeInterval(-20 * 86400))
        XCTAssertEqual(recent.endDate, now.addingTimeInterval(-86400))
        XCTAssertTrue(recent.detail.contains("15 games immediately before them"))
        XCTAssertEqual(recent.source, "ESPN")
        XCTAssertNil(recent.availability)
    }
    func testDistributionExplainsMiddleValueAndRangeWithoutStatisticalJargon() throws {
        let card = try XCTUnwrap(AnalyticsTrendInsights.make(candidate: candidate(), history: history(rows(20)), now: now).first { $0.kind == .consistency })
        XCTAssertEqual(card.headline, "Middle result 10.5")
        XCTAssertTrue(card.detail.contains("5.8 to 15.2"))
        XCTAssertTrue(card.detail.contains("Last 20 games with usable team score statistics"))
        XCTAssertEqual(card.sample, 20)
        XCTAssertFalse(card.detail.contains("forecast"))
        XCTAssertFalse(card.detail.contains("IQR"))
        XCTAssertFalse(card.detail.contains("UTC"))
    }
    func testVenueAndSeasonAreNeverInferredFromDatesOrMatchupNames() {
        let insights = AnalyticsTrendInsights.make(candidate: candidate(), history: history(rows(20)), now: now)
        XCTAssertEqual(insights.map(\.kind), [.recentBaseline, .consistency])
        let explicit = AnalyticsTrendInsights.make(candidate: candidate(), history: history(rows(15, metadata: true)), now: now)
        XCTAssertNotNil(explicit.first { $0.kind == .homeAway })
        let season = explicit.first { $0.kind == .seasonBaseline }
        XCTAssertEqual(season?.sample, 15)
        XCTAssertTrue(season?.detail.contains("10 earlier recorded games from the same 2026 season") == true)
    }
    func testVenueMinimumAndUnknownNeutralExclusions() throws {
        var observations = rows(8)
        for index in 0..<3 { observations[index].homeAway = "home" }
        for index in 3..<6 { observations[index].homeAway = "away" }
        observations[6].homeAway = "neutral"
        let card = try XCTUnwrap(AnalyticsTrendInsights.make(candidate: candidate(), history: history(observations), now: now).first { $0.kind == .homeAway })
        XCTAssertEqual(card.sample, 6)
        XCTAssertEqual(card.headline, "Home 2.0 · Away 5.0")
        observations[5].homeAway = nil
        XCTAssertNil(AnalyticsTrendInsights.make(candidate: candidate(), history: history(observations), now: now).first { $0.kind == .homeAway })
    }
    func testOpponentSubsetRequiresExactSourceOpponentAndVerifiedContext() throws {
        var observations = rows(10)
        for index in 3..<7 { observations[index] = AnalyticsObservation(id: observations[index].id, date: observations[index].date, opponent: "Other Opponent", scored: Double(index + 1), allowed: 100) }
        for index in 7..<10 { observations[index] = AnalyticsObservation(id: observations[index].id, date: observations[index].date, opponent: "Fixture", scored: Double(index + 1), allowed: 100) }
        XCTAssertNil(AnalyticsTrendInsights.make(candidate: candidate(), history: history(observations), now: now).first { $0.kind == .opponent })
        let context = try XCTUnwrap(AnalyticsTrendInsights.make(candidate: candidate(proof: true), history: history(observations), now: now).first { $0.kind == .opponent })
        XCTAssertEqual(context.sample, 3)
        XCTAssertEqual(context.headline, "2.0 team score vs Fixture Away")
    }
    func testPlayerOpponentNeverUsesNameOnlyOrWrongHistoricalAthlete() {
        let subject = candidate(player: true, proof: true)
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: subject, history: history(rows(10)), now: now).isEmpty)
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: subject, history: history(rows(10), playerID: "wrong"), now: now).isEmpty)
        XCTAssertNotNil(AnalyticsTrendInsights.make(candidate: subject, history: history(rows(10), playerID: "77"), now: now).first { $0.kind == .opponent })
    }
    func testDuplicatesFutureMissingVoidAndNonfiniteCannotInflateSamples() throws {
        let originals = rows(5)
        let excluded = [AnalyticsObservation(id: "future", date: now, opponent: "Fixture Away", scored: 100), AnalyticsObservation(id: "void", date: now.addingTimeInterval(-100), opponent: "Fixture Away", scored: 100, status: .void), AnalyticsObservation(id: "missing", date: now.addingTimeInterval(-100), opponent: "Fixture Away", scored: 100, status: .missing), AnalyticsObservation(id: "nonfinite", date: now.addingTimeInterval(-100), opponent: "Fixture Away", scored: .infinity)]
        let clean = AnalyticsTrendInsights.make(candidate: candidate(), history: history(originals + [originals[0]] + excluded), now: now)
        XCTAssertEqual(clean.count, 1); XCTAssertEqual(clean.first?.sample, 5)
        var conflict = originals[0]; conflict.scored = 200
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: candidate(), history: history(originals + [conflict] + excluded), now: now).isEmpty)
    }
    func testInsufficientSamplesAndMixedSeasonDoNotProduceClaims() {
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: candidate(), history: history(rows(4)), now: now).isEmpty)
        let ten = AnalyticsTrendInsights.make(candidate: candidate(), history: history(rows(10)), now: now)
        XCTAssertEqual(ten.map(\.kind), [.recentBaseline, .consistency])
        var mixed = rows(20, metadata: true); mixed[0].season = 2025
        XCTAssertNil(AnalyticsTrendInsights.make(candidate: candidate(), history: history(mixed), now: now).first { $0.kind == .seasonBaseline })
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: candidate(), history: history(Array(repeating: rows(1)[0], count: 2001)), now: now).isEmpty)
    }
    func testEveryPlayerInsightRequiresVerifiedIdentityAndCompatibleNamespace() {
        let unscoped = candidate(player: true)
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: unscoped, history: history(rows(20)), now: now).isEmpty)
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: unscoped, history: history(rows(20), playerID: "  "), now: now).isEmpty)
        XCTAssertEqual(AnalyticsTrendInsights.make(candidate: unscoped, history: history(rows(20), playerID: "77"), now: now).map(\.kind), [.recentBaseline, .consistency])
        let scoped = candidate(player: true, proof: true)
        let incompatible = AnalyticsHistoryResult(summary: nil, rows: [], observations: rows(20), source: "MLB Stats API", warnings: [], availability: "Fixture incompatible provider ID", verifiedPlayerID: "77")
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: scoped, history: incompatible, now: now).isEmpty)
        // Without current-roster context, verified MLB history supports MLB descriptive cards.
        var mlbMarket = unscoped.market; mlbMarket.sport = "MLB"; mlbMarket.market = "batter_hits"
        let mlbUnscoped = AnalyticsFeedCandidate(id: "mlb-candidate", subject: .player, market: mlbMarket, quotes: [], verifiedSubject: nil)
        XCTAssertEqual(AnalyticsTrendInsights.make(candidate: mlbUnscoped, history: incompatible, now: now).map(\.kind), [.recentBaseline, .consistency])
        let espnRoster = AnalyticsFeedVerifiedSubject(sport: "MLB", subject: .player, name: mlbMarket.name, providerID: "77", teamName: "Fixture Home", teamID: "1", marketKeys: ["batter_hits"], sourceURL: "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/1/roster", verifiedAt: now)
        let mlbScoped = AnalyticsFeedCandidate(id: "mlb-candidate", subject: .player, market: mlbMarket, quotes: [], verifiedSubject: espnRoster)
        XCTAssertTrue(AnalyticsTrendInsights.make(candidate: mlbScoped, history: incompatible, now: now).isEmpty)
    }

    func testDisplayCopyKeepsExactCountsAndDatesWithoutDateRangesOrBoilerplate() throws {
        var subject = candidate(proof: true)
        var market = subject.market; market.market = "h2h"
        subject = AnalyticsFeedCandidate(id: subject.id, subject: .team, market: market, quotes: [], verifiedSubject: subject.verifiedSubject)
        let cards = AnalyticsTrendInsights.make(candidate: subject, history: history(rows(20, metadata: true)), now: now)
        XCTAssertEqual(Set(cards.map(\.kind)), Set(AnalyticsTrendInsightKind.allCases))
        for card in cards {
            XCTAssertGreaterThan(card.sample, 0)
            XCTAssertNotNil(card.startDate); XCTAssertNotNil(card.endDate)
            let copy = [card.title, card.headline, card.detail].joined(separator: " ").lowercased()
            for removed in ["score margin", "iqr", "median", "forecast", "predict", "utc", "oct "] { XCTAssertFalse(copy.contains(removed), copy) }
            XCTAssertTrue(card.detail.contains("Score difference is this team's score minus its opponent's"))
        }
    }

}
