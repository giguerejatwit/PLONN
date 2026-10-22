import XCTest
@testable import PLONNCore

final class MatchupOutlookSelectionTests: XCTestCase {
    typealias Engine = MatchupOutlookSelection
    let now = sportsDate("2026-10-15T16:00:00Z")!
    let stamp = "2026-10-15T15:59:30Z"
    let start = "2026-10-16T00:15:00Z"
    func event(_ sport: String = "NFL") -> SharedMatchupEvent {
        .init(sport: sport, eventID: "espn:1", homeTeam: "Home", awayTeam: "Away", startISO: start)
    }
    func market(_ sport: String = "NFL", key: String = "spreads", name: String = "Home", side: String = "home", line: Double? = -2.5, price: Double = -110, id: String = "quote") -> Market {
        Market(id: id, event: "provider:1", sport: sport, kind: key.hasPrefix("player_") || key.hasPrefix("batter_") ? "prop" : "team", name: name, opponent: "Away", label: key, aliases: "", p: nil, price: price, market: key, book: "Book A", quotes: [], updated: stamp, mode: "live", logs: nil, line: line, eventLabel: "Away at Home", modelValidated: nil, modelUpdated: nil, modelVersion: nil, eventState: "pre", commence: start, side: side)
    }
    func fact(_ team: String, _ role: String, _ metric: String, _ value: Double, sample: Int = 20, player: String? = nil, period: String = "2026 regular season", unit: String = "units per game") -> Engine.Fact {
        .init(id: [team, role, metric, player ?? ""].joined(separator: ":"), team: team, role: role, metric: metric, unit: unit, period: period, start: "2026-09-01T00:00:00Z", end: "2026-10-14T12:00:00Z", value: value, sample: sample, player: player)
    }
    func scoring(_ sport: String, home: Double = 30, away: Double = 15, homeAllowed: Double = 15, awayAllowed: Double = 30, sample: Int = 20) -> [Engine.Fact] {
        let metric = ["NFL": "pointsPerGame", "NBA": "pointsPerGame", "MLB": "runsPerGame", "NHL": "goalsPerGame"][sport]!
        return [fact("Home", "offense", metric, home, sample: sample), fact("Away", "offense", metric, away, sample: sample), fact("Home", "defense", metric, homeAllowed, sample: sample), fact("Away", "defense", metric, awayAllowed, sample: sample)]
    }
    func model(_ market: Market, p: Double) -> Market {
        var m = market; m.p = p; m.modelValidated = true; m.modelUpdated = stamp; m.modelVersion = "verified-v1"; return m
    }

    func testLargePropSlateDoesNotChangeTeamDirection() {
        let base=market()
        let expected=Engine.directions(event:event(),facts:scoring("NFL"),markets:[base],now:now)
        let props=(0..<1000).map {market(key:"player_rush_yds",name:"Player \($0)",side:"over",line:45.5,id:"prop-\($0)")}
        let began=Date()
        XCTAssertEqual(Engine.directions(event:event(),facts:scoring("NFL"),markets:[base]+props,now:now),expected)
        print("1000-prop outlook check: \(Date().timeIntervalSince(began)) seconds")
    }

    func testBestPositiveModelEdgeWinsAcrossAllSportsAndMarketKinds() throws {
        for (sport, prop) in ["NFL": "player_rush_yds", "NBA": "player_rebounds", "MLB": "batter_hits", "NHL": "player_shots_on_goal"] {
            let total = model(market(sport, key: "totals", name: "Over", side: "over", line: 5.5, id: "total"), p: 0.55)
            let strong = model(market(sport, key: prop, name: "Player", side: "under", line: 2.5, price: 125, id: "prop"), p: 0.65)
            let spread = model(market(sport, id: "spread"), p: 0.57)
            let result = try XCTUnwrap(Engine.select(event: event(sport), facts: scoring(sport), markets: [total, spread, strong], now: now))
            XCTAssertEqual(result.marketID, "prop", sport)
            XCTAssertEqual(try XCTUnwrap(result.edge), 100 * (0.65 - 1 / 2.25), accuracy: 0.0001)
            XCTAssertTrue(result.selection.contains("Under 2.5")); XCTAssertTrue(result.selection.contains("+125"))
            XCTAssertTrue(result.selection.contains("Book A")); XCTAssertEqual(result.quoteDate, sportsDate(stamp))
        }
    }

    func testMoneylineLabelIncludesPriceAndNoInventedPointLine() throws {
        let m = model(market("MLB", key: "h2h", side: "home", line: nil, price: 128), p: 0.6)
        let result = try XCTUnwrap(Engine.select(event: event("MLB"), facts: [], markets: [m], now: now))
        XCTAssertEqual(result.selection, "Home moneyline · +128 · Book A")
    }

    func testHistoricalTeamDirectionCanLeadOverMixedScoringAcrossSports() throws {
        for sport in ["NFL", "NBA", "MLB", "NHL"] {
            let m = market(sport)
            let result = try XCTUnwrap(Engine.select(event: event(sport), facts: scoring(sport), markets: [m], now: now))
            XCTAssertEqual(result.marketID, m.id); XCTAssertNil(result.edge)
            XCTAssertTrue(result.selection.contains("-2.5")); XCTAssertTrue(result.selection.contains("-110"))
            XCTAssertEqual(result.qualification, Engine.trendQualification)
        }
    }

    func testTotalRequiresOffenseAndDefenseToAgreeWithTheActualLine() throws {
        for sport in ["NFL", "NBA", "MLB", "NHL"] {
            let m = market(sport, key: "totals", name: "Under", side: "under", line: 60.5)
            let result = try XCTUnwrap(Engine.select(event: event(sport), facts: scoring(sport), markets: [m], now: now))
            XCTAssertTrue(result.selection.contains("Under 60.5"))
            XCTAssertNil(Engine.select(event: event(sport), facts: scoring(sport, homeAllowed: 40, awayAllowed: 40), markets: [m], now: now))
        }
    }

    func testNonScoringPlayerDirectionRequiresPlayerAndOpponentEvidenceAcrossSports() throws {
        for (sport, pair) in ["NFL": ("player_rush_yds", "rushingYardsPerGame"), "NBA": ("player_rebounds", "reboundsPerGame"), "MLB": ("batter_hits", "hitsPerGame"), "NHL": ("player_shots_on_goal", "shotsPerGame")] {
            let m = market(sport, key: pair.0, name: "Player", side: "over", line: 2.5)
            let team = [fact("Home", "offense", pair.1, 12), fact("Away", "defense", pair.1, 13), fact("league", "reference", pair.1, 10)]
            let player = fact("Home", "player", pair.1, 3.5, player: "Player")
            let result = try XCTUnwrap(Engine.select(event: event(sport), facts: team + [player], markets: [m], now: now))
            XCTAssertEqual(result.marketID, m.id, sport); XCTAssertTrue(result.factIDs.contains(player.id))
            XCTAssertTrue(result.selection.contains("Over 2.5")); XCTAssertNil(result.edge)
            let withoutPlayer = try XCTUnwrap(Engine.select(event: event(sport), facts: team, markets: [m], now: now))
            XCTAssertNil(withoutPlayer.marketID, "Team history must not create a player pick")
            XCTAssertTrue(withoutPlayer.explanation.contains("No matching supported line and odds"))
        }
    }

    func testMixedScoringFallsBackToStrongestNonScoringMetricWithNoInventedOdds() throws {
        for (sport, metric) in ["NFL": "rushingYardsPerGame", "NBA": "reboundsPerGame", "MLB": "hitsPerGame", "NHL": "shotsPerGame"] {
            let fs = scoring(sport) + [fact("Home", "offense", metric, 15), fact("Away", "defense", metric, 13), fact("league", "reference", metric, 10)]
            let result = try XCTUnwrap(Engine.select(event: event(sport), facts: fs, markets: [], now: now))
            XCTAssertTrue(result.factIDs.allSatisfy { $0.contains(metric) })
            XCTAssertNil(result.edge); XCTAssertNil(result.marketID)
            XCTAssertFalse(result.selection.contains("Scoring rates point"))
        }
    }

    func testMixedAndSmallOrMismatchedSamplesNeverForceADirection() {
        let m = market()
        XCTAssertNil(Engine.select(event: event(), facts: scoring("NFL", home: 30, away: 15, homeAllowed: 30, awayAllowed: 15), markets: [m], now: now))
        XCTAssertNil(Engine.select(event: event(), facts: scoring("NFL", sample: 2), markets: [m], now: now))
        var facts = scoring("NFL")
        facts[0] = fact("Home", "offense", "pointsPerGame", 30, period: "2025 regular season")
        XCTAssertNil(Engine.select(event: event(), facts: facts, markets: [m], now: now))
        facts = scoring("NFL") + [scoring("NFL")[0]]
        XCTAssertNil(Engine.select(event: event(), facts: facts, markets: [m], now: now))
    }

    func testUnrelatedConflictingFutureAndClosedMarketsCannotLead() {
        let original = model(market(), p: 0.8)
        var bad = original; bad.eventLabel = "Other at Home"
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [bad], now: now))
        bad = original; bad.commence = "2026-10-17T00:15:00Z"
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [bad], now: now))
        bad = original; bad.updated = "2026-10-15T16:01:00Z"
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [bad], now: now))
        bad = original; bad.price = 150
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [original, bad], now: now))
        bad = original; bad.id = "closed"; bad.eventState = "final"
        XCTAssertNil(Engine.select(event: event(), facts: scoring("NFL"), markets: [original, bad], now: now))
        XCTAssertNil(Engine.select(event: event(), facts: scoring("NFL"), markets: [original], now: sportsDate(start)!))
    }

    func testStaleOddsAreSnapshotContextOnlyAndExpireAfterADay() throws {
        var m = model(market(), p: 0.8); m.updated = "2026-10-15T10:00:00Z"
        let lead = try XCTUnwrap(Engine.select(event: event(), facts: scoring("NFL"), markets: [m], now: now))
        XCTAssertNil(lead.edge); XCTAssertEqual(lead.quoteDate, sportsDate(m.updated!))
        m.updated = "2026-10-14T10:00:00Z"
        XCTAssertNil(Engine.select(event: event(), facts: scoring("NFL"), markets: [m], now: now))
    }

    func testModelMustBeFreshValidatedPositiveAndHaveSupportedSettlement() {
        var m = model(market(), p: 0.8)
        m.modelValidated = false
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [m], now: now))
        m.modelValidated = true; m.modelUpdated = "2026-10-15T15:00:00Z"
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [m], now: now))
        m.modelUpdated = stamp; m.p = 0.3
        XCTAssertNil(Engine.select(event: event(), facts: scoring("NFL"), markets: [m], now: now))
        m.p = 0.8; m.line = -2
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [m], now: now))
        m = model(market(key: "h2h", line: nil), p: 0.8)
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [m], now: now), "NFL tie settlement remains unsupported")
    }

    func testBestBookUsesExactLineAndOwnTimestamp() throws {
        var m = model(market(), p: 0.7)
        m.quotes = [Quote(book: "Book B", price: 125, updated: stamp), Quote(book: "Untimestamped", price: 300)]
        let result = try XCTUnwrap(Engine.select(event: event(), facts: [], markets: [m], now: now))
        XCTAssertTrue(result.selection.contains("+125 · Book B"))
        XCTAssertEqual(result.edge!, 100 * (0.7 - 1 / 2.25), accuracy: 0.0001)
    }

    func testRankingIsIndependentOfInputOrder() throws {
        let a = model(market(id: "a"), p: 0.7), b = model(market(id: "b"), p: 0.7)
        XCTAssertEqual(Engine.select(event: event(), facts: [], markets: [a,b], now: now), Engine.select(event: event(), facts: [], markets: [b,a], now: now))
    }

    func testNHLPuckDropToleranceRequiresUniqueMatchingEvent() throws {
        var m = model(market("NHL", key: "player_shots_on_goal", name: "Player", side: "over", line: 2.5), p: 0.7)
        m.commence = "2026-10-16T00:20:00Z"
        XCTAssertNotNil(Engine.select(event: event("NHL"), facts: [], markets: [m], now: now))
        var second = m; second.id = "other"; second.event = "different-provider-event"
        XCTAssertNil(Engine.select(event: event("NHL"), facts: [], markets: [m, second], now: now))
    }

    func testConflictingProbabilitiesFromSameModelSnapshotCannotLead() {
        let m = model(market(), p: 0.7)
        var other = m; other.id = "different-row"; other.p = 0.8
        XCTAssertNil(Engine.select(event: event(), facts: [], markets: [m, other], now: now))
    }

    func testRecordedSeattleDenverSnapshotCanChooseSideInsteadOfMixedScoringHeadline() throws {
        let ios = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let raw = try JSONDecoder().decode([String].self, from: Data(contentsOf: ios.appendingPathComponent("PLONN/FrozenMatchupReports-20261013.json")))
        let responses = try raw.map { try JSONDecoder().decode(SharedMatchupReportResponse.self, from: XCTUnwrap(Data(base64Encoded: $0))) }
        let response = try XCTUnwrap(responses.first { $0.event.homeTeam == "Denver Broncos" && $0.event.awayTeam == "Seattle Seahawks" })
        let checked = try SharedMatchupReportValidator.date(XCTUnwrap(response.generatedAt))
        let fs = Engine.facts(response, now: checked)
        var m = market(key: "h2h", name: "Seattle Seahawks", side: "Seattle Seahawks", line: nil, price: -140)
        m.eventLabel = "Seattle Seahawks at Denver Broncos"; m.commence = response.event.startISO
        m.updated = response.sourceCheckedAt
        let lead = try XCTUnwrap(Engine.select(event: response.event, facts: fs, markets: [m], now: checked))
        XCTAssertEqual(lead.selection, "Seattle Seahawks moneyline · -140 · Book A")
        XCTAssertNil(lead.edge)
        XCTAssertTrue(lead.explanation.contains("26.8")); XCTAssertTrue(lead.explanation.contains("19.4"))
        XCTAssertEqual(response.report?.outlook.first?.id, "game:mixed-scoring", "Original recorded report stays unchanged")
    }

    func testThreeDistinctSupportedDirectionsAreRankedAcrossAllSports() throws {
        for (sport, prop) in ["NFL": "player_rush_yds", "NBA": "player_rebounds", "MLB": "batter_hits", "NHL": "player_shots_on_goal"] {
            let player = model(market(sport, key: prop, name: "Player", side: "over", line: 2.5, price: 125, id: "player"), p: 0.7)
            let total = model(market(sport, key: "totals", name: "Over", side: "over", line: 40.5, id: "total"), p: 0.62)
            let spread = model(market(sport, id: "spread"), p: 0.6)
            let fourth = model(market(sport, key: prop, name: "Another Player", side: "over", line: 3.5, id: "fourth"), p: 0.55)
            let rows = [spread, fourth, total, player]
            let result = Engine.directions(event: event(sport), facts: [], markets: rows, now: now)
            XCTAssertEqual(result.map(\.lead.marketID), ["player", "total", "spread"], sport)
            XCTAssertEqual(result, Engine.directions(event: event(sport), facts: [], markets: rows.reversed(), now: now))
            XCTAssertTrue(result.allSatisfy { !$0.lead.shortReason.isEmpty && $0.lead.quoteDate != nil })
            XCTAssertTrue(result[2].correlationNote?.contains("direction 2") == true)
        }
    }

    func testSameBetAtDifferentBooksAndAlternateLinesConsumesOneSlot() {
        let best = model(market("NBA", key: "totals", name: "Over", side: "over", line: 210.5, price: 125, id: "a"), p: 0.7)
        var book = best; book.id = "another-book"; book.book = "Book B"; book.price = -110
        var alternate = best; alternate.id = "alternate"; alternate.market = "alternate_totals"; alternate.line = 215.5; alternate.p = 0.55
        var opposite = best; opposite.id = "under"; opposite.name = "Under"; opposite.side = "under"; opposite.p = 0.53; opposite.price = -110
        let prop = model(market("NBA", key: "player_assists", name: "Player", side: "over", line: 6.5, id: "prop"), p: 0.6)
        let result = Engine.directions(event: event("NBA"), facts: [], markets: [alternate, opposite, book, prop, best], now: now)
        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].lead.marketID, "a")
        XCTAssertTrue(result[0].lead.selection.contains("Over 210.5"))
        XCTAssertTrue(result[0].lead.selection.contains("+125 · Book A"))
        XCTAssertEqual(result[1].lead.marketID, "prop")
    }

    func testTeamSideAliasesAndOppositeSidesAreOneDirection() {
        let home = model(market("NBA", key: "spreads", side: "home", price: 125, id: "home"), p: 0.7)
        var alias = home; alias.id = "alias"; alias.market = "spread"; alias.side = "Home"
        let away = model(market("NBA", key: "spreads", name: "Away", side: "away", line: 2.5, id: "away"), p: 0.6)
        XCTAssertEqual(Engine.directions(event: event("NBA"), facts: [], markets: [home, alias, away], now: now).count, 1)
    }

    func testSamePlayerDifferentStatsStayDistinctButAreMarkedRelated() {
        let points = model(market("NBA", key: "player_points", name: "Player", side: "over", line: 20.5, id: "points"), p: 0.7)
        let rebounds = model(market("NBA", key: "player_rebounds", name: "Player", side: "over", line: 8.5, id: "rebounds"), p: 0.6)
        let result = Engine.directions(event: event("NBA"), facts: [], markets: [points, rebounds], now: now)
        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[1].correlationNote, "Potentially correlated with top direction · same player.")
    }

    func testPlayerScoringAndTeamTotalAreMarkedRelatedAcrossSports() {
        for (sport, prop) in ["NFL": "player_pass_tds", "NBA": "player_points", "MLB": "batter_home_runs", "NHL": "player_goals"] {
            let player = model(market(sport, key: prop, name: "Player", side: "over", line: 1.5, id: "player"), p: 0.7)
            let team = model(market(sport, key: "team_totals", name: "Home", side: "over", line: 20.5, id: "team"), p: 0.6)
            let result = Engine.directions(event: event(sport), facts: [], markets: [team, player], now: now)
            XCTAssertEqual(result.count, 2, sport)
            XCTAssertTrue(result.last?.correlationNote?.contains("shared scoring exposure") == true, sport)
        }
    }

    func testHistoricalAlternativesKeepTheirOwnEvidenceAndQuotes() {
        let spread = market("NBA", id: "spread")
        let total = market("NBA", key: "totals", name: "Under", side: "under", line: 60.5, price: 125, id: "total")
        let result = Engine.directions(event: event("NBA"), facts: scoring("NBA"), markets: [total, spread], now: now)
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result.allSatisfy { $0.lead.edge == nil && !$0.lead.factIDs.isEmpty && !$0.lead.shortReason.isEmpty })
        XCTAssertTrue(result.last?.lead.selection.contains("Under 60.5") == true)
        XCTAssertTrue(result.last?.lead.selection.contains("+125") == true)
        XCTAssertNotNil(result.last?.correlationNote)
    }

    func testOnlyQualifiedDirectionsAreShownWithoutFillingSlots() {
        let good = model(market("NBA", id: "good"), p: 0.7)
        let weak = model(market("NBA", key: "player_rebounds", name: "Player", side: "over", line: 8.5, id: "weak"), p: 0.2)
        var old = model(market("NBA", key: "totals", name: "Over", side: "over", line: 210.5, id: "old"), p: 0.8)
        old.updated = "2026-10-14T10:00:00Z"
        let result = Engine.directions(event: event("NBA"), facts: [], markets: [weak, old, good], now: now)
        XCTAssertEqual(result.map(\.lead.marketID), ["good"])
        XCTAssertNil(result.first?.correlationNote)
        XCTAssertTrue(Engine.directions(event: event("NBA"), facts: [], markets: [weak, old], now: now).isEmpty)
    }

    func testUnpricedDirectionsNeverInventOddsOrMeasuredEdges() {
        let metrics = ["rushingYardsPerGame", "passingYardsPerGame", "passingAttemptsPerGame", "rushingAttemptsPerGame"]
        let facts = metrics.flatMap { metric in
            [fact("Home", "offense", metric, 15), fact("Away", "defense", metric, 13), fact("league", "reference", metric, 10)]
        }
        let result = Engine.directions(event: event(), facts: facts, markets: [], now: now)
        XCTAssertEqual(result.count, 3)
        XCTAssertTrue(result.allSatisfy { $0.lead.marketID == nil && $0.lead.edge == nil && $0.lead.quoteDate == nil && $0.lead.shortReason.contains("odds unavailable") })
        XCTAssertTrue(result.dropFirst().allSatisfy { $0.correlationNote?.contains("same team context") == true })
    }

    func testRecordedStatisticsHaveTwoSupportedDirectionsWithFixtureQuotes() throws {
        let ios = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let bundle = try JSONDecoder().decode([String].self, from: Data(contentsOf: ios.appendingPathComponent("PLONN/FrozenMatchupReports-20261013.json")))
        let responses = try bundle.map { try JSONDecoder().decode(SharedMatchupReportResponse.self, from: XCTUnwrap(Data(base64Encoded: $0))) }
        let response = try XCTUnwrap(responses.first { $0.event.homeTeam == "Denver Broncos" && $0.event.awayTeam == "Seattle Seahawks" })
        let clock = try SharedMatchupReportValidator.date(XCTUnwrap(response.generatedAt))
        let rows = [market(key: "h2h", name: "Seattle Seahawks", side: "Seattle Seahawks", line: nil, price: -140, id: "side"),
                    market(key: "totals", name: "Over", side: "over", line: 40.5, price: -107, id: "total")].map { row -> Market in
            var m = row
            m.eventLabel = "Seattle Seahawks at Denver Broncos"; m.commence = response.event.startISO; m.updated = response.sourceCheckedAt
            return m
        }
        let result = Engine.directions(event: response.event, facts: Engine.facts(response, now: clock), markets: rows, now: clock)
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result[0].lead.selection.contains("Over 40.5"))
        XCTAssertTrue(result[1].lead.selection.contains("Seattle Seahawks moneyline"))
        XCTAssertNotNil(result[1].correlationNote)
    }
}
