import XCTest
@testable import PLONNCore

final class AnalyticsHistoryTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)
    private func market(_ key: String = "player_points", side: String = "over", line: Double? = 10) -> Market {
        Market(id: "manual", event: "selected", sport: "NBA", kind: key.hasPrefix("player_") ? "prop" : "team", name: "Fixture Team", opponent: "Opponent", label: "", aliases: "", p: nil, price: 0, market: key, book: "", quotes: [], updated: nil, mode: "analytics", line: line, side: side)
    }
    private func observation(_ id: String, value: Double? = nil, scored: Double? = nil, allowed: Double? = nil, status: AnalyticsObservationStatus = .completed) -> AnalyticsObservation {
        AnalyticsObservation(id: id, date: now.addingTimeInterval(-100), opponent: "Opponent", value: value, scored: scored, allowed: allowed, status: status)
    }
    func testIntegerPushAndExcludedObservedRows() {
        let result = AnalyticsHistoryEngine.evaluate(market: market(), observations: [observation("hit", value: 11), observation("miss", value: 9), observation("push", value: 10), observation("void", status: .void), observation("missing")], now: now)
        XCTAssertEqual(result.summary?.n, 5)
        XCTAssertEqual(result.summary?.hits, 1); XCTAssertEqual(result.summary?.misses, 1)
        XCTAssertEqual(result.summary?.pushes, 1); XCTAssertEqual(result.summary?.voids, 1); XCTAssertEqual(result.summary?.missing, 1)
        XCTAssertEqual(result.summary?.hitFrequency, 0.5)
        XCTAssertEqual(result.summary?.smallSample, true)
    }
    func testZeroDenominatorAndLocalLineOverride() {
        let observations = [observation("equal", value: 10)]
        XCTAssertNil(AnalyticsHistoryEngine.evaluate(market: market(), observations: observations, now: now).summary?.hitFrequency)
        let changed = AnalyticsHistoryEngine.evaluate(market: market(), observations: observations, line: 9.5, now: now)
        XCTAssertEqual(changed.summary?.hits, 1); XCTAssertEqual(changed.observations, observations)
    }
    func testDirectionForSpreadTotalsAndUnderProps() {
        let game = observation("one", scored: 105, allowed: 100)
        XCTAssertEqual(AnalyticsHistoryEngine.evaluate(market: market("spreads", line: -5), observations: [game], now: now).summary?.pushes, 1)
        XCTAssertEqual(AnalyticsHistoryEngine.evaluate(market: market("spreads", line: -6), observations: [game], now: now).summary?.misses, 1)
        XCTAssertEqual(AnalyticsHistoryEngine.evaluate(market: market("totals", side: "under", line: 206), observations: [game], now: now).summary?.hits, 1)
        XCTAssertEqual(AnalyticsHistoryEngine.evaluate(market: market(side: "under"), observations: [observation("one", value: 9)], now: now).summary?.hits, 1)
    }
    func testDrawIsUndeterminedAndUnsupportedContractFailsClosed() {
        let draw = observation("draw", scored: 100, allowed: 100)
        let result = AnalyticsHistoryEngine.evaluate(market: market("h2h", line: nil), observations: [draw], now: now)
        XCTAssertEqual(result.summary?.missing, 1); XCTAssertEqual(result.summary?.pushes, 0); XCTAssertNil(result.summary?.hitFrequency)
        XCTAssertNil(AnalyticsHistoryEngine.evaluate(market: market("h2h_3_way"), observations: [draw], now: now).summary)
    }
    func testConflictingDuplicatesCurrentGameAndFutureAreExcluded() {
        let observations = [observation("conflict", value: 11), observation("conflict", value: 12), observation("selected", value: 11), AnalyticsObservation(id: "future", date: now, opponent: "Opponent", value: 11), observation("valid", value: 11), observation("valid", value: 11)]
        let result = AnalyticsHistoryEngine.evaluate(market: market(), observations: observations, now: now)
        XCTAssertEqual(result.summary?.n, 1); XCTAssertEqual(result.warnings.count, 1)
        var selected = market(); selected.commence = ISO8601DateFormatter().string(from: now.addingTimeInterval(-200))
        XCTAssertNil(AnalyticsHistoryEngine.evaluate(market: selected, observations: observations, now: now).summary)
    }
    func testLastTenObservedIncludesMissingAndNonfiniteCannotHit() {
        let observations = (0..<12).map { AnalyticsObservation(id: String($0), date: now.addingTimeInterval(Double(-$0 - 1)), opponent: "Opponent", value: $0 == 0 ? Double.infinity : 11) }
        let result = AnalyticsHistoryEngine.evaluate(market: market(), observations: observations, limit: 1000, now: now)
        XCTAssertEqual(result.summary?.n, 10); XCTAssertEqual(result.summary?.missing, 1); XCTAssertEqual(result.summary?.hits, 9)
        XCTAssertNil(AnalyticsHistoryEngine.evaluate(market: market(line: .nan), observations: observations, now: now).summary)
    }
    func testDirectoryIdentityAndMalformedSchedule() throws {
        let team: [String: Any] = ["team": ["id": "1", "displayName": "Fixture Team"]]
        let directory: [String: Any] = ["sports": [["leagues": [["teams": [team]]]]]]
        XCTAssertEqual(AnalyticsHistoryService.teamID(directory: directory, name: "Fixture Team"), "1")
        XCTAssertNil(AnalyticsHistoryService.teamID(directory: directory, name: "Fixture"))
        XCTAssertThrowsError(try AnalyticsHistoryService.schedule(payload: ["events": "invalid"], teamID: "1"))
    }
    func testScheduleCountsMissingFinalScoreAndSkipsUnfinished() throws {
        func event(_ id: String, completed: Bool, score: Any) -> [String: Any] {
            ["id": id, "date": "2026-01-01T00:00:00Z", "competitions": [["status": ["type": ["completed": completed, "state": completed ? "post" : "in"]], "competitors": [["team": ["id": "1", "displayName": "Fixture Team"], "score": score], ["team": ["id": "2", "displayName": "Opponent"], "score": ["value": 90]]]]]]
        }
        let rows = try AnalyticsHistoryService.schedule(payload: ["events": [event("final", completed: true, score: ["value": 100]), event("missing", completed: true, score: true), event("live", completed: false, score: "100")]], teamID: "1")
        XCTAssertEqual(rows.count, 2); XCTAssertEqual(rows.first?.scored, 100); XCTAssertEqual(rows.last?.status, .missing)
        let conflicting = try AnalyticsHistoryService.schedule(payload: ["events": [event("duplicate", completed: true, score: "100"), event("duplicate", completed: true, score: "101")]], teamID: "1")
        XCTAssertTrue(conflicting.isEmpty)
    }
    func testServiceBoundedReloadAndCancellationWithFixtures() async {
        actor Calls {
            var urls: [URL] = []
            func record(_ url: URL) { urls.append(url) }
            func count() -> Int { urls.count }
        }
        let calls = Calls()
        let cached = PlayerHistoryService(transport: { url in
            await calls.record(url)
            if url.path.hasSuffix("/teams") { return Data(#"{"sports":[{"leagues":[{"teams":[{"team":{"id":"1","displayName":"Fixture Team"}}]}]}]}"#.utf8) }
            return Data(#"{"events":[]}"#.utf8)
        })
        let service = AnalyticsHistoryService(reader: { try await cached.read($0) })
        let first = await service.load(market: market("h2h", line: nil), now: now)
        XCTAssertEqual(first.sourceURLs.count, 5)
        _ = await service.load(market: market("h2h", line: nil), now: now)
        let count = await calls.count(); XCTAssertEqual(count, 5)
        let cancelled = Task { await service.load(market: market("h2h", line: nil), now: now) }; cancelled.cancel()
        let result = await cancelled.value
        XCTAssertTrue(result.availability.contains("cancelled"))
    }
    func testIneligibleSelectionsPerformNoProviderReads() async {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
            func total() -> Int { count }
        }
        let counter = Counter()
        let service = AnalyticsHistoryService(reader: { _ in
            await counter.increment()
            throw PlayerHistoryError.invalidPayload
        })
        var badDate = market("h2h"); badDate.commence = "invalid"
        var unsupportedSport = market("h2h"); unsupportedSport.sport = "SOCCER"
        var unsupportedProp = market(); unsupportedProp.market = "player_anytime_td"
        var unknownTotal = market("totals"); unknownTotal.name = "Over"; unknownTotal.opponent = ""
        let selections = [market(line: .nan), market(line: nil), market(line: -1), market(side: "yes"), market("totals", line: -1), market("h2h_3_way"), badDate, unsupportedSport, unsupportedProp, unknownTotal]
        for selection in selections {
            XCTAssertNotNil(AnalyticsHistoryEngine.unavailableReason(market: selection, now: now))
            let result = await service.load(market: selection, now: now)
            XCTAssertNil(result.summary)
            XCTAssertTrue(result.sourceURLs.isEmpty)
        }
        let total = await counter.total()
        XCTAssertEqual(total, 0)
    }

    func testStatisticsWithoutLineIncludesActualDistributionAndTeamTies() {
        var selection = market(line: nil); selection.side = nil
        let observations = [observation("a", value: 9), observation("b", value: 15), observation("c", status: .void), observation("d")]
        let stats = AnalyticsHistoryEngine.statistics(market: selection, observations: observations, now: now)
        XCTAssertNil(stats.summary)
        XCTAssertEqual(stats.statistics?.n, 4); XCTAssertEqual(stats.statistics?.valid, 2)
        XCTAssertEqual(stats.statistics?.mean, 12); XCTAssertEqual(stats.statistics?.minimum, 9); XCTAssertEqual(stats.statistics?.maximum, 15)
        XCTAssertEqual(stats.statistics?.voids, 1); XCTAssertEqual(stats.statistics?.missing, 1)
        XCTAssertNil(stats.statistics?.teamWins)
        XCTAssertNil(AnalyticsHistoryEngine.statsUnavailableReason(market: selection, now: now))
        let team = AnalyticsHistoryEngine.statistics(market: market("totals", line: nil), observations: [observation("w", scored: 101, allowed: 100), observation("l", scored: 90, allowed: 100), observation("t", scored: 100, allowed: 100)], now: now)
        XCTAssertEqual(team.statistics?.teamWins, 1); XCTAssertEqual(team.statistics?.teamLosses, 1); XCTAssertEqual(team.statistics?.teamTies, 1)
        XCTAssertEqual(team.statistics?.valid, 3)
    }
    func testStatsLoadDoesNotRequireLineOrSideWithFixtureReader() async {
        let service = AnalyticsHistoryService(reader: { url in
            if url.path.hasSuffix("/teams") { return Data(#"{"sports":[{"leagues":[{"teams":[{"team":{"id":"1","displayName":"Fixture Team"}}]}]}]}"#.utf8) }
            return Data(#"{"events":[]}"#.utf8)
        })
        for key in ["spreads", "team_totals"] {
            var selection = market(key, line: nil); selection.side = nil
            let result = await service.loadStats(market: selection, now: now)
            XCTAssertEqual(result.sourceURLs.count, 5)
            XCTAssertNil(result.summary)
            XCTAssertTrue(result.warnings.contains { $0.contains("coverage") })
        }
    }

}

extension AnalyticsHistoryTests {
    func testTeamTotalsUseOwnScoreAndKeepGameTotalsAndMarginsDistinct() {
        let game = AnalyticsObservation(id: "team-total", date: now.addingTimeInterval(-3600), opponent: "Other", scored: 105, allowed: 100)
        for (key, expected) in [("team_totals", 105.0), ("totals", 205.0), ("spreads", 5.0), ("h2h", 5.0)] {
            XCTAssertEqual(AnalyticsHistoryEngine.statistics(market: market(key), observations: [game], now: now).statistics?.mean, expected)
        }
        XCTAssertEqual(AnalyticsHistoryEngine.evaluate(market: market("team_totals", side: "under", line: 106), observations: [game], now: now).summary?.hits, 1)
        XCTAssertEqual(AnalyticsHistoryEngine.evaluate(market: market("team_totals", side: "over", line: 104), observations: [game], now: now).summary?.hits, 1)
        XCTAssertEqual(AnalyticsHistoryEngine.evaluate(market: market("team_totals", side: "over", line: 105), observations: [game], now: now).summary?.pushes, 1)
        let ownOnly = AnalyticsObservation(id: "own-only", date: game.date, opponent: "Other", scored: 105)
        XCTAssertEqual(AnalyticsHistoryEngine.statistics(market: market("team_totals"), observations: [ownOnly], now: now).statistics?.mean, 105)
        for selection in [market("team_totals", side: "home"), market("team_totals", line: -1), market("team_totals", line: .infinity)] {
            XCTAssertNil(AnalyticsHistoryEngine.evaluate(market: selection, observations: [game], now: now).summary)
        }
        var legacy = market("team_totals"); legacy.kind = "prop"
        XCTAssertNotNil(AnalyticsHistoryEngine.statsUnavailableReason(market: legacy, now: now))
    }
}

extension AnalyticsHistoryTests {
    func testValidObservationsExposeBroaderCleanedWindowAndMetadataConflicts() {
        let selection = market("team_totals", line: nil)
        var rows: [AnalyticsObservation] = []
        for index in 0..<16 {
            let day = now.addingTimeInterval(-Double(index + 1) * 86400)
            let venue = index % 2 == 0 ? "home" : "away"
            rows.append(AnalyticsObservation(id: "game-\(index)", date: day, opponent: "Other", scored: Double(100 + index), allowed: 90, homeAway: venue, season: 2026))
        }
        let cleaned = AnalyticsHistoryEngine.validObservations(market: selection, observations: rows, now: now)
        XCTAssertEqual(cleaned.count, 16)
        XCTAssertEqual(cleaned.first?.value, 100)
        XCTAssertEqual(cleaned.first?.homeAway, "home")
        XCTAssertEqual(cleaned.first?.season, 2026)
        XCTAssertEqual(AnalyticsHistoryEngine.statistics(market: selection, observations: rows, now: now).statsRows.count, 10)
        var conflict = rows[0]; conflict.homeAway = "away"
        XCTAssertEqual(AnalyticsHistoryEngine.validObservations(market: selection, observations: rows + [conflict], now: now).count, 15)
        conflict = rows[0]; conflict.season = 2025
        XCTAssertEqual(AnalyticsHistoryEngine.validObservations(market: selection, observations: rows + [conflict], now: now).count, 15)
        var missing = rows[0]; missing.status = .missing
        let future = AnalyticsObservation(id: "future", date: now.addingTimeInterval(60), opponent: "Other", scored: 101, allowed: 90)
        XCTAssertTrue(AnalyticsHistoryEngine.validObservations(market: selection, observations: [missing, future], now: now).isEmpty)
        var unknown = rows[0]; unknown.homeAway = "neutral"; unknown.season = 0
        let unknownCleaned = AnalyticsHistoryEngine.validObservations(market: selection, observations: [unknown], now: now)
        XCTAssertNil(unknownCleaned.first?.homeAway); XCTAssertNil(unknownCleaned.first?.season)
    }
    func testScheduleVenueAndSeasonRequireExplicitProof() throws {
        func event(venue: Any? = nil, season: Any? = nil) -> [String: Any] {
            var own: [String: Any] = ["team": ["id": "1", "displayName": "Fixture Team"], "score": 100]
            own["homeAway"] = venue
            var row: [String: Any] = ["id": "final", "date": "2026-01-01T00:00:00Z", "competitions": [["status": ["type": ["completed": true, "state": "post"]], "competitors": [own, ["team": ["id": "2", "displayName": "Opponent"], "score": 90]]]]]
            row["season"] = season
            return row
        }
        func parse(_ row: [String: Any], season: Int? = nil) throws -> AnalyticsObservation? {
            try AnalyticsHistoryService.schedule(payload: ["events": [row]], teamID: "1", season: season).first
        }
        let explicit = try parse(event(venue: "away", season: ["year": 2026]))
        XCTAssertEqual(explicit?.homeAway, "away"); XCTAssertEqual(explicit?.season, 2026)
        let absent = try parse(event())
        XCTAssertNil(absent?.homeAway); XCTAssertNil(absent?.season)
        XCTAssertEqual(try parse(event(), season: 2026)?.season, 2026)
        for invalid: Any in [true, "2026", 2026.5, 0, 1e200] {
            XCTAssertNil(try parse(event(venue: "neutral", season: ["year": invalid]), season: 2026)?.season)
        }
        XCTAssertNil(try parse(event(season: ["year": 2025]), season: 2026)?.season)
        XCTAssertNil(try parse(event(venue: "HOME"))?.homeAway)
        XCTAssertTrue(try AnalyticsHistoryService.schedule(payload: ["events": [event(venue: "home"), event(venue: "away")]], teamID: "1").isEmpty)
    }
}

extension AnalyticsHistoryTests {
    func testNeutralAndMalformedNeutralSiteSuppressTeamVenueOnly() throws {
        for competitionFlag in [false, true] {
            for (flag, expected): (Any, String?) in [(true, nil), (false, "home"), (1, nil), (0, nil), ("false", nil), (NSNull(), nil)] {
                var competition: [String: Any] = ["status": ["type": ["completed": true, "state": "post"]], "competitors": [["team": ["id": "1"], "score": 100, "homeAway": "home"], ["team": ["id": "2", "displayName": "Other"], "score": 90, "homeAway": "away"]]]
                if competitionFlag { competition["neutralSite"] = flag }
                var event: [String: Any] = ["id": "final", "date": "2026-01-01T00:00:00Z", "competitions": [competition], "season": ["year": 2026]]
                if !competitionFlag { event["neutralSite"] = flag }
                let row = try XCTUnwrap(AnalyticsHistoryService.schedule(payload: ["events": [event]], teamID: "1").first)
                XCTAssertEqual(row.homeAway, expected)
                XCTAssertEqual(row.scored, 100); XCTAssertEqual(row.season, 2026)
            }
        }
    }
}
