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
        XCTAssertEqual(recent.headline, "3.0 PTS/GM")
        XCTAssertTrue(recent.detail.contains("13.0 PTS/GM"))
        XCTAssertEqual(recent.sample, 20)
        XCTAssertEqual(recent.startDate, now.addingTimeInterval(-20 * 86400))
        XCTAssertEqual(recent.endDate, now.addingTimeInterval(-86400))
        XCTAssertTrue(recent.detail.contains("15 games immediately before them"))
        XCTAssertEqual(recent.source, "ESPN")
        XCTAssertNil(recent.availability)
    }
    func testRangeUsesObservedExtremesWithoutStatisticalJargon() throws {
        let card = try XCTUnwrap(AnalyticsTrendInsights.make(candidate: candidate(), history: history(rows(20)), now: now).first { $0.kind == .consistency })
        XCTAssertEqual(card.headline, "1–20 points")
        XCTAssertTrue(card.detail.contains("Lowest and highest results"))
        XCTAssertTrue(card.detail.contains("20 games with usable statistics"))
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
        XCTAssertEqual(card.headline, "2.0 PTS/GM")
        XCTAssertTrue(card.detail.contains("Away: 5.0 PTS/GM"))
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
        XCTAssertEqual(context.headline, "2.0 PTS/GM")
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
            for removed in ["score difference", "middle result", "decisions", "score margin", "iqr", "median", "forecast", "predict", "utc", "oct "] { XCTAssertFalse(copy.contains(removed), copy) }
            XCTAssertTrue((card.headline.contains("points") || card.headline.contains("PTS/GM")))
            XCTAssertTrue(card.headline.contains("Outscored by") || card.headline.contains("Lost by"))
        }
    }

    func testScoringResultsUseSportUnitsAndDoNotCallAnAverageMarginAWinsAverage() {
        for (sport, unit, abbreviation) in [("NFL", "points", "PTS/GM"), ("NBA", "points", "PTS/GM"), ("MLB", "runs", "R/GM"), ("NHL", "goals", "G/GM")] {
            var market = candidate().market; market.sport = sport; market.market = "spreads"
            // A positive average can include losses; do not say every game was won.
            XCTAssertEqual(AnalyticsCardCopy.average(2, market: market), "Outscored opponents by 2.0 \(abbreviation)")
            XCTAssertEqual(AnalyticsCardCopy.average(-2, market: market), "Outscored by 2.0 \(abbreviation)")
            XCTAssertEqual(AnalyticsCardCopy.average(0, market: market), "Even with opponents on average")
            XCTAssertEqual(AnalyticsCardCopy.result(-1, market: market), "Lost by 1 \(unit.dropLast())")
            XCTAssertEqual(AnalyticsCardCopy.result(0, market: market), "Tied")
            XCTAssertEqual(AnalyticsCardCopy.range(low: -3, high: 4, market: market), "Lost by 3 \(unit) · Won by 4 \(unit)")
            market.market = "totals"
            XCTAssertEqual(AnalyticsCardCopy.average(6, market: market), "6.0 \(abbreviation) (both teams)")
        }
        var hitter = candidate(player: true).market; hitter.sport = "MLB"; hitter.market = "batter_hits"
        XCTAssertEqual(AnalyticsCardCopy.average(3, market: hitter), "3.0 H/GM", "H is the baseball hits statistic")
    }
    func testPerGameAbbreviationsKeepSportAndStatMeaning() {
        for (sport, stat, label) in [("NBA", "player_points", "PTS/GM"), ("NBA", "player_rebounds", "REB/GM"),
                                    ("NBA", "player_points_rebounds_assists", "PTS+REB+AST/GM"),
                                    ("NFL", "player_reception_yds", "REC YDS/GM"), ("NFL", "player_receptions", "REC/GM"),
                                    ("MLB", "pitcher_strikeouts", "K/GM"), ("MLB", "batter_rbis", "RBI/GM"),
                                    ("NHL", "player_shots_on_goal", "SOG/GM"), ("NHL", "player_total_saves", "SV/GM")] {
            var market = candidate(player: true).market; market.sport = sport; market.market = stat
            XCTAssertEqual(AnalyticsCardCopy.perGameUnit(market), label)
            XCTAssertEqual(AnalyticsCardCopy.average(5, market: market), "5.0 " + label)
            XCTAssertTrue(AnalyticsCardCopy.average(5, market: market, abbreviated: false).contains("per game"))
        }
    }
    func testLineComparisonCountsExcludeExactLineAndUnavailableResults() throws {
        var market = candidate(player: true).market
        market.price = -110; market.line = 20; market.side = "Over"
        let observations = [21.0, 19, 20, 25, 24, 10].enumerated().map { index, value in
            AnalyticsObservation(id: "line-\(index)", date: now.addingTimeInterval(-Double(index + 1) * 86400), opponent: "Other", value: value)
        } + [
            AnalyticsObservation(id: "void", date: now.addingTimeInterval(-7 * 86400), opponent: "Other", value: 50, status: .void),
            AnalyticsObservation(id: "missing", date: now.addingTimeInterval(-8 * 86400), opponent: "Other", status: .missing)
        ]
        let summary = try XCTUnwrap(AnalyticsHistoryEngine.evaluate(market: market, observations: observations, now: now).summary)
        XCTAssertEqual(AnalyticsCardCopy.comparison(summary, market: market), "Met this line in 3 of 5 games")
        XCTAssertEqual(AnalyticsCardCopy.exclusions(summary, market: market), "Not counted: 1 game landed exactly on the line; 2 games had no usable result.")
        let onlyEqual = try XCTUnwrap(AnalyticsHistoryEngine.evaluate(market: market, observations: [observations[2]], now: now).summary)
        XCTAssertEqual(AnalyticsCardCopy.comparison(onlyEqual, market: market), "No games to compare")
        market.market = "h2h"; market.kind = "team"; market.side = market.name
        let tied = AnalyticsObservation(id: "tie", date: now.addingTimeInterval(-86400), opponent: "Other", scored: 10, allowed: 10)
        let tieSummary = try XCTUnwrap(AnalyticsHistoryEngine.evaluate(market: market, observations: [tied], now: now).summary)
        XCTAssertEqual(AnalyticsCardCopy.exclusions(tieSummary, market: market), "Not counted: 1 game was tied or had no usable result.")
    }

}
