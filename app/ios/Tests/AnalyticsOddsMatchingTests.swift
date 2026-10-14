import XCTest
@testable import PLONNCore

final class AnalyticsOddsMatchingTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-10-13T12:00:00Z")!
    private func stamp(_ date: Date) -> String { ISO8601DateFormatter().string(from: date) }
    private func event() -> AnalyticsEvent { AnalyticsEvent(sport: "NBA", providerID: "espn-1", home: "Fixture Home", away: "Fixture Away", start: now.addingTimeInterval(3600), homeID: "1", awayID: "2") }
    private func target(prop: Bool = false) -> AnalyticsStatsTarget {
        AnalyticsStatsTarget(sport: "NBA", kind: prop ? "prop" : "team", name: prop ? "Fixture Player" : "Fixture Home", marketKey: prop ? "player_points" : "spreads", teamName: "Fixture Home", teamID: "1", playerID: prop ? "athlete-1" : nil, rosterVerifiedAt: prop ? now : nil)
    }
    private func quote(prop: Bool = false) -> Market {
        Market(id: "quote-1", event: "odds-1", sport: "NBA", kind: prop ? "prop" : "team", name: prop ? "Fixture Player" : "Fixture Home", opponent: "Fixture Away", label: "Full game", aliases: "", p: nil, price: -110, market: prop ? "player_points" : "spreads", book: "Fixture Book", quotes: [], updated: stamp(now.addingTimeInterval(-60)), mode: "live", line: prop ? 10.5 : -2.5, eventLabel: "Fixture Away at Fixture Home", eventState: "pre", commence: stamp(event().start), side: prop ? "over" : "home")
    }
    private func response(_ quotes: [Market], stale: Bool = false, fetched: Date? = nil, coverageEvent: String = "odds-1", start: Date? = nil, keys: [String]? = nil) -> AnalyticsScopedOddsResponse {
        AnalyticsScopedOddsResponse(schemaVersion: 1, coverage: AnalyticsOddsCoverage(sport: "NBA", eventID: coverageEvent, home: "Fixture Home", away: "Fixture Away", commence: stamp(start ?? event().start), markets: keys ?? [quotes.first?.market ?? "spreads"]), markets: quotes, fetchedAt: stamp(fetched ?? now), expiresAt: stamp(now.addingTimeInterval(300)), stale: stale)
    }
    func testExactScopedEventAndStatAndCodableContract() throws {
        let request = try XCTUnwrap(AnalyticsOddsMatcher.request(target: target(), event: event(), now: now))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any])
        XCTAssertEqual(json["schema_version"] as? Int, 1); XCTAssertEqual(json["markets"] as? [String], ["spreads"])
        XCTAssertNil(json["owner"])
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response(Array(repeating: quote(), count: 501)), now: now).markets.isEmpty)
        let result = AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([quote()]), now: now)
        XCTAssertEqual(result.markets.count, 1); XCTAssertEqual(result.freshness, .fresh); XCTAssertEqual(result.fetchedAt, now)
    }
    func testWrongEventSportCategoryAndAmbiguousRowsFailClosed() {
        var wrong = quote(); wrong.eventLabel = "Unrelated Away at Unrelated Home"
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([wrong]), now: now).markets.isEmpty)
        wrong = quote(); wrong.sport = "NFL"
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([wrong]), now: now).markets.isEmpty)
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([quote()], keys: ["spreads", "totals"]), now: now).markets.isEmpty)
        var conflict = quote(); conflict.line = -3.5
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([quote(), conflict]), now: now).markets.isEmpty)
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([quote()], start: event().start.addingTimeInterval(61)), now: now).markets.isEmpty)
    }
    func testPlayerNameAloneCannotMatchAndVerifiedRosterCan() {
        var unverified = target(prop: true); unverified = AnalyticsStatsTarget(sport: unverified.sport, kind: unverified.kind, name: unverified.name, marketKey: unverified.marketKey, teamName: unverified.teamName)
        XCTAssertNil(AnalyticsOddsMatcher.request(target: unverified, event: event(), now: now))
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: unverified, event: event(), response: response([quote(prop: true)]), now: now).markets.isEmpty)
        XCTAssertEqual(AnalyticsOddsMatcher.match(target: target(prop: true), event: event(), response: response([quote(prop: true)]), now: now).markets.count, 1)
        let staleRoster = AnalyticsStatsTarget(sport: "NBA", kind: "prop", name: "Fixture Player", marketKey: "player_points", teamName: "Fixture Home", teamID: "1", playerID: "athlete-1", rosterVerifiedAt: now.addingTimeInterval(-901))
        XCTAssertNil(AnalyticsOddsMatcher.request(target: staleRoster, event: event(), now: now))
    }
    func testFreshnessIndependentOfMatchingAndCachedUnknownIsStale() {
        let stale = AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([quote()], stale: true), now: now)
        XCTAssertEqual(stale.markets.count, 1); XCTAssertEqual(stale.freshness, .stale)
        var oldQuote = quote(); oldQuote.updated = stamp(now.addingTimeInterval(-901))
        XCTAssertEqual(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([oldQuote]), now: now).freshness, .stale)
        let cached = AnalyticsOddsMatcher.matchCached(target: target(), event: event(), markets: [quote()], now: now)
        XCTAssertEqual(cached.markets.count, 1); XCTAssertEqual(cached.freshness, .stale); XCTAssertNil(cached.fetchedAt)
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([quote()], fetched: now.addingTimeInterval(60)), now: now).markets.isEmpty)
    }
    func testInvalidBookLineSidePeriodAndUpdateAreRejected() {
        for update in ["", "malformed"] {
            var invalid = quote(); invalid.updated = update
            XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([invalid]), now: now).markets.isEmpty)
        }
        var invalid = quote(); invalid.line = .nan
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([invalid]), now: now).markets.isEmpty)
        invalid = quote(); invalid.mode = "demo"
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([invalid]), now: now).markets.isEmpty)
        invalid = quote(); invalid.book = ""
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([invalid]), now: now).markets.isEmpty)
        invalid = quote(prop: true); invalid.side = "yes"
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(prop: true), event: event(), response: response([invalid]), now: now).markets.isEmpty)
        invalid = quote(); invalid.label = "First half spread"
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([invalid]), now: now).markets.isEmpty)
        invalid = quote(); invalid.market = "spreads_h1"
        XCTAssertTrue(AnalyticsOddsMatcher.match(target: target(), event: event(), response: response([invalid]), now: now).markets.isEmpty)
    }
    func testOnlySameDayUpcomingEventsAndDuplicateIDsAreUnambiguous() throws {
        func fixture(_ id: String, start: Date, home: String = "Fixture Home") -> [String: Any] {
            ["id": id, "date": stamp(start), "competitions": [["status": ["type": ["state": "pre", "completed": false, "name": "STATUS_SCHEDULED"]], "competitors": [["homeAway": "home", "team": ["id": "1", "displayName": home]], ["homeAway": "away", "team": ["id": "2", "displayName": "Fixture Away"]]]]]]
        }
        let first = fixture("a", start: event().start)
        let parsed = try AnalyticsHistoryService.upcomingEvents(payload: ["events": [first, first, fixture("tomorrow", start: now.addingTimeInterval(86400)), fixture("past", start: now.addingTimeInterval(-60))]], sport: "NBA", teamID: "1", now: now)
        XCTAssertEqual(parsed.count, 1)
        let conflicts = try AnalyticsHistoryService.upcomingEvents(payload: ["events": [first, fixture("a", start: event().start, home: "Wrong Team")]], sport: "NBA", teamID: "1", now: now)
        XCTAssertTrue(conflicts.isEmpty)
        let doubleheader = try AnalyticsHistoryService.upcomingEvents(payload: ["events": [first, fixture("b", start: event().start.addingTimeInterval(3600))]], sport: "NBA", teamID: "1", now: now)
        XCTAssertEqual(doubleheader.count, 2)
    }
    func testRosterRequiresExactIdentityAndTeam() {
        let roster: [String: Any] = ["team": ["id": "1"], "athletes": [["id": "athlete-1", "fullName": "Fixture Player"]]]
        XCTAssertEqual(AnalyticsHistoryService.rosterPlayerID(payload: roster, teamID: "1", name: "Fixture Player"), "athlete-1")
        XCTAssertNil(AnalyticsHistoryService.rosterPlayerID(payload: roster, teamID: "2", name: "Fixture Player"))
        XCTAssertNil(AnalyticsHistoryService.rosterPlayerID(payload: roster, teamID: "1", name: "Fixture"))
    }
    func testContextResolverIsBoundedAndDoesNotFanOutAcrossTeams() async throws {
        actor Reads {
            var paths: [String] = []
            func add(_ path: String) { paths.append(path) }
            func count() -> Int { paths.count }
        }
        let reads = Reads()
        let start = stamp(event().start)
        let scoreboard: [String: Any] = ["events": [["id": "espn-1", "date": start, "competitions": [["status": ["type": ["state": "pre", "completed": false, "name": "STATUS_SCHEDULED"]], "competitors": [["homeAway": "home", "team": ["id": "1", "displayName": "Fixture Home"]], ["homeAway": "away", "team": ["id": "2", "displayName": "Fixture Away"]]]]]]]]
        let scoreboardData = try JSONSerialization.data(withJSONObject: scoreboard)
        let service = AnalyticsHistoryService(reader: { url in
            await reads.add(url.path)
            if url.path.hasSuffix("/teams") { return Data(#"{"sports":[{"leagues":[{"abbreviation":"NBA","teams":[{"team":{"id":"1","displayName":"Fixture Home"}}]}]}]}"#.utf8) }
            if url.path.hasSuffix("/scoreboard") { return scoreboardData }
            return Data(#"{"team":{"id":"1"},"athletes":[{"id":"athlete-1","fullName":"Fixture Player"}]}"#.utf8)
        })
        let teamContext = await service.resolveContext(market: quote(), now: now)
        XCTAssertEqual(teamContext.event?.providerID, "espn-1")
        let teamCount = await reads.count(); XCTAssertEqual(teamCount, 2)
        let unknownPlayer = await service.resolveContext(market: quote(prop: true), now: now)
        XCTAssertNil(unknownPlayer.target)
        let unknownCount = await reads.count(); XCTAssertEqual(unknownCount, 2)
        let history = AnalyticsHistoryResult(summary: nil, rows: [], observations: [], source: "ESPN", warnings: [], availability: "Fixture", verifiedPlayerID: "athlete-1", candidateTeamID: "1")
        let playerContext = await service.resolveContext(market: quote(prop: true), history: history, now: now)
        XCTAssertEqual(playerContext.target?.playerID, "athlete-1")
        XCTAssertEqual(playerContext.target?.rosterVerifiedAt, now)
        let finalCount = await reads.count(); XCTAssertEqual(finalCount, 4)
    }

}

extension AnalyticsOddsMatchingTests {
    func testTeamTotalMatchingRequiresExactTeamAndExplicitFullGameContract() throws {
        let selected = AnalyticsStatsTarget(sport: "NBA", kind: "team", name: "Fixture Home", marketKey: "team_totals", teamName: "Fixture Home", teamID: "1")
        XCTAssertEqual(try XCTUnwrap(AnalyticsOddsMatcher.request(target: selected, event: event(), now: now)).markets, ["team_totals"])
        var valid = quote(); valid.market = "team_totals"; valid.side = "over"; valid.line = 105.5
        XCTAssertEqual(AnalyticsOddsMatcher.match(target: selected, event: event(), response: response([valid]), now: now).markets.count, 1)
        for variant in 0..<7 {
            var invalid = valid
            switch variant {
            case 0: invalid.kind = "prop"
            case 1: invalid.name = "Fixture Away"
            case 2: invalid.name = "Fixture"
            case 3: invalid.side = "home"
            case 4: invalid.line = -1
            case 5: invalid.label = "First half team total"
            default: invalid.market = "totals"
            }
            XCTAssertTrue(AnalyticsOddsMatcher.match(target: selected, event: event(), response: response([invalid]), now: now).markets.isEmpty)
        }
        let unrelated = AnalyticsStatsTarget(sport: "NBA", kind: "team", name: "Other Team", marketKey: "team_totals", teamName: "Other Team")
        XCTAssertFalse(AnalyticsOddsMatcher.eligible(target: unrelated, event: event(), now: now))
    }
}
