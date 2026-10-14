import XCTest
@testable import PLONNCore

/// Entirely synthetic fixtures. Counts compare recorded statistics with today's line.
final class OpponentPropHistoryTests: XCTestCase {
    private func date(_ value: String) -> Date { ISO8601DateFormatter().date(from: value)! }
    private func context(sport: String = "NBA", key: String = "player_points", playerID: String = "athlete-7", opponentID: String = "opponent-9", playerName: String = "Fixture Athlete", opponentName: String = "Fixture Opponent", seasons: [Int] = [2024, 2025, 2026]) -> OpponentHistoryContext {
        OpponentHistoryContext(sport: sport, playerID: playerID, playerName: playerName, marketKey: key, opponentID: opponentID, opponentName: opponentName, targetStart: date("2026-10-07T00:00:00Z"), cutoff: date("2026-10-06T12:00:00Z"), targetEventID: "target-event", source: "fixture official logs", requestedSeasons: seasons, loadedSeasons: seasons)
    }
    private func game(_ id: String = "history-1", value: Double = 10, sport: String = "NBA", key: String = "player_points", timestamp: String = "2026-10-05T23:00:00Z", opponent: String = "Fixture Opponent", opponentID: String? = "opponent-9", playerID: String? = "athlete-7", teamID: String? = "team-old", season: Int? = 2026, workload: String = "32 minutes") -> PlayerGame {
        PlayerGame(id: id, date: date(timestamp), opponent: opponent, value: value, workload: workload, sport: sport, playerID: playerID, playerTeamID: teamID, opponentID: opponentID, marketKey: key, season: season)
    }
    private func summarize(_ games: [PlayerGame], context c: OpponentHistoryContext? = nil, line: Double? = 10, count: Int = 10) -> OpponentHistorySummary {
        OpponentPropHistoryEngine.summarize(games: games, context: c ?? context(), line: line, displayCount: count)
    }
    func testEverySportUsesActualResultsAgainstCurrentLine() {
        for (sport, key) in [("NBA", "player_points"), ("MLB", "batter_hits"), ("NFL", "player_rush_yds"), ("NHL", "player_shots_on_goal")] {
            let games = [game("a", value: 0, sport: sport, key: key), game("b", value: 1, sport: sport, key: key), game("c", value: 2, sport: sport, key: key)]
            let result = summarize(games, context: context(sport: sport, key: key), line: 1)
            XCTAssertEqual(result.n, 3, sport)
            XCTAssertEqual(result.average, 1)
            XCTAssertEqual(result.overs, 1)
            XCTAssertEqual(result.unders, 1)
            XCTAssertEqual(result.pushes, 1)
            for phrase in ["guaranteed", "will hit", "bet wins", "won their bets"] { XCTAssertFalse(result.text.lowercased().contains(phrase)) }
        }
    }
    func testOpponentIdentityOverridesDisplayNamesAndPlayerTeamChanges() {
        let renamed = game("a", opponent: "Historical Opponent Name", teamID: "former-team")
        let traded = game("b", teamID: "current-team")
        let impostor = game("c", opponent: "Fixture Opponent", opponentID: "different-team")
        let result = summarize([renamed, traded, impostor])
        XCTAssertEqual(Set(result.games.map(\.id)), ["a", "b"])
        XCTAssertEqual(result.n, 2)
    }
    func testMissingOrWrongIdentitiesCannotInventTrend() {
        let invalid = [game("a", opponentID: nil), game("b", playerID: nil), game("c", playerID: "another-athlete"), game("d", opponentID: "another-opponent")]
        let result = summarize(invalid)
        XCTAssertEqual(result.n, 0)
        XCTAssertNil(result.average)
        XCTAssertNil(result.overs)
        XCTAssertFalse(result.text.isEmpty)
    }
    func testSportAndMarketMustMatchEvenWhenPlayerPointsKeyCollides() {
        XCTAssertEqual(summarize([game(sport: "NHL")]).n, 0)
        XCTAssertEqual(summarize([game(key: "player_assists")]).n, 0)
        let hockey = summarize([game(sport: "NHL")], context: context(sport: "NHL"))
        XCTAssertEqual(hockey.n, 1)
    }
    func testEasternPriorCalendarDayExcludesSameDayFutureAndTarget() {
        let rows = [game("prior-day", timestamp: "2026-10-06T03:59:59Z"), game("same-day", timestamp: "2026-10-06T04:00:00Z"), game("future", timestamp: "2026-10-08T12:00:00Z"), game("target-event", timestamp: "2026-10-05T12:00:00Z")]
        XCTAssertEqual(summarize(rows).games.map(\.id), ["prior-day"])
    }
    func testEarlierTargetDayRestrictsLaterCutoff() {
        let c = OpponentHistoryContext(sport: "NBA", playerID: "athlete-7", playerName: "Fixture Athlete", marketKey: "player_points", opponentID: "opponent-9", opponentName: "Fixture Opponent", targetStart: date("2026-10-05T23:00:00Z"), cutoff: date("2026-10-06T12:00:00Z"), targetEventID: "target-event", source: "fixture official logs", requestedSeasons: [2026], loadedSeasons: [2026])
        XCTAssertEqual(summarize([game("before", timestamp: "2026-10-05T03:59:59Z"), game("same-day", timestamp: "2026-10-05T04:00:00Z")], context: c).games.map(\.id), ["before"])
    }
    func testMLBDoubleheadersRemainDistinctByEventID() {
        let result = summarize([game("doubleheader-1", value: 1, sport: "MLB", key: "batter_hits", timestamp: "2026-10-04T17:00:00Z"), game("doubleheader-2", value: 2, sport: "MLB", key: "batter_hits", timestamp: "2026-10-04T23:00:00Z")], context: context(sport: "MLB", key: "batter_hits"), line: 1.5)
        XCTAssertEqual(result.n, 2)
        XCTAssertEqual(result.overs, 1)
        XCTAssertEqual(result.unders, 1)
    }
    func testIdenticalDuplicatesDeduplicateRegardlessOfWorkloadPresentation() {
        let original = game()
        let anotherLabel = game(workload: "Different presentation")
        XCTAssertEqual(summarize([original, original, anotherLabel]).n, 1)
    }
    func testConflictingDuplicatesQuarantineEveryCopy() {
        let original = game()
        let variants = [game(value: 11), game(timestamp: "2026-10-04T23:00:00Z"), game(opponentID: "different-team"), game(playerID: "different-player"), game(season: 2025), game(sport: "NHL"), game(key: "player_assists")]
        for conflict in variants { XCTAssertEqual(summarize([original, conflict, original]).n, 0) }
    }
    func testInvalidCountOutcomesAreExcludedButActualZeroIsKept() {
        for (sport, key) in [("NBA", "player_points"), ("MLB", "batter_hits"), ("NFL", "player_pass_tds"), ("NHL", "player_goals")] {
            let rows = [game("zero", value: 0, sport: sport, key: key), game("negative", value: -1, sport: sport, key: key), game("fraction", value: 1.5, sport: sport, key: key), game("nan", value: .nan, sport: sport, key: key), game("infinity", value: .infinity, sport: sport, key: key)]
            let result = summarize(rows, context: context(sport: sport, key: key))
            XCTAssertEqual(result.games.map(\.id), ["zero"], sport)
        }
    }
    func testSignedNFLYardOutcomeIsRecordedWithoutClamping() {
        let result = summarize([game(value: -3, sport: "NFL", key: "player_rush_yds")], context: context(sport: "NFL", key: "player_rush_yds"), line: 0)
        XCTAssertEqual(result.n, 1)
        XCTAssertEqual(result.average, -3)
        XCTAssertEqual(result.unders, 1)
    }
    func testUnavailableLineDoesNotInventOverUnderCounts() {
        for line in [nil, Double.nan, Double.infinity] as [Double?] {
            let result = summarize([game()], line: line)
            XCTAssertEqual(result.n, 1)
            XCTAssertEqual(result.average, 10)
            XCTAssertNil(result.overs)
            XCTAssertNil(result.unders)
            XCTAssertNil(result.pushes)
        }
    }
    func testRequestedAndLoadedSeasonsRestrictObservedRows() {
        XCTAssertEqual(summarize([game("old", season: 2023), game("unknown", season: nil), game("current", season: 2026)]).games.map(\.id), ["current"])
    }
    func testWindowsSelectMostRecentDistinctGamesAndKeepHistoricalMeaning() {
        let rows = (1...12).map { game("game-\($0)", value: Double($0), timestamp: String(format: "2026-09-%02dT12:00:00Z", $0)) }
        let ten = summarize(rows, line: 6.5)
        let five = summarize(rows.reversed(), line: 6.5, count: 5)
        XCTAssertEqual(ten.n, 10)
        XCTAssertEqual(five.n, 5)
        XCTAssertEqual(five.games.map(\.id), ["game-12", "game-11", "game-10", "game-9", "game-8"])
        XCTAssertEqual(five.average, 10)
        XCTAssertEqual(five.overs, 5)
        XCTAssertFalse(five.text.lowercased().contains("forecast"))
    }
    func testSupportedMarketFamiliesHaveExactSportSpecificStatistics() {
        let families = [
            "NBA": ["player_points", "player_rebounds", "player_assists", "player_threes", "player_blocks", "player_steals", "player_points_rebounds_assists", "player_points_rebounds", "player_points_assists", "player_rebounds_assists"],
            "MLB": ["batter_hits", "batter_rbis", "batter_stolen_bases", "batter_home_runs", "batter_total_bases", "batter_runs_scored", "pitcher_strikeouts"],
            "NFL": ["player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_pass_completions", "player_pass_interceptions", "player_rush_yds", "player_rush_attempts", "player_reception_yds", "player_receptions", "player_rush_reception_yds"],
            "NHL": ["player_goals", "player_assists", "player_points", "player_shots_on_goal", "player_blocked_shots", "player_total_saves", "player_power_play_points"]
        ]
        for (sport, keys) in families {
            for key in keys {
                let result = summarize([game(sport: sport, key: key)], context: context(sport: sport, key: key))
                XCTAssertEqual(result.n, 1, sport + "/" + key)
                XCTAssertFalse(result.text.contains(key), "Raw provider key must not be shown")
            }
        }
        for (sport, key) in [("NBA", "batter_hits"), ("MLB", "player_points"), ("NFL", "player_points"), ("NHL", "player_rebounds")] {
            XCTAssertEqual(summarize([game(sport: sport, key: key)], context: context(sport: sport, key: key)).n, 0)
        }
    }
    func testActualZeroDoesNotRequireMinutesButLegacyDNPPlaceholderIsNotVerified() {
        let actual = game("played-zero", value: 0, workload: "")
        let placeholder = PlayerGame(id: "legacy-dnp", date: date("2026-10-05T23:00:00Z"), opponent: "Fixture Opponent", value: 0, workload: "DNP")
        XCTAssertEqual(summarize([actual, placeholder]).games.map(\.id), ["played-zero"])
    }
    func testSmallSampleAndEmptyHistoryDiscloseLimitsWithoutClaimingForecast() {
        let small = summarize([game()])
        XCTAssertTrue(small.evidence.lowercased().contains("small sample"))
        XCTAssertTrue(small.evidence.contains("today’s line"))
        XCTAssertTrue(small.evidence.contains("not historical bet results or a forecast"))
        XCTAssertNil(small.unavailableReason)
        let empty = summarize([])
        XCTAssertEqual(empty.n, 0)
        XCTAssertNotNil(empty.unavailableReason)
        XCTAssertTrue(empty.text.lowercased().contains("unavailable"))
        XCTAssertFalse(empty.text.contains("averaged 0"))
    }
    func testUnloadedSeasonRowsExcludedAndCoverageDisclosed() {
        let c = OpponentHistoryContext(sport: "NBA", playerID: "athlete-7", playerName: "Fixture Athlete", marketKey: "player_points", opponentID: "opponent-9", opponentName: "Fixture Opponent", targetStart: date("2026-10-07T00:00:00Z"), cutoff: date("2026-10-06T12:00:00Z"), targetEventID: "target-event", source: "fixture official logs", requestedSeasons: [2024, 2025, 2026], loadedSeasons: [2026])
        let result = summarize([game("unloaded", season: 2025), game("loaded", season: 2026)], context: c)
        XCTAssertEqual(result.games.map(\.id), ["loaded"])
        XCTAssertTrue(result.evidence.contains("unavailable seasons: 2024, 2025"))
    }
    func testNegativeLineOnlyAllowedForSignedNFLYards() {
        let yards = summarize([game(value: -3, sport: "NFL", key: "player_rush_yds")], context: context(sport: "NFL", key: "player_rush_yds"), line: -4)
        XCTAssertEqual(yards.overs, 1)
        XCTAssertNil(summarize([game()], line: -1).overs)
    }
    func testBlankEventIDAndMissingVerificationAreExcluded() {
        var missingSport = game("sport-missing"); missingSport.sport = nil
        var missingMarket = game("market-missing"); missingMarket.marketKey = nil
        let result = summarize([game(""), missingSport, missingMarket])
        XCTAssertEqual(result.n, 0)
    }

    func testInvalidOwnTeamIdentityCannotVerifyOpponentMatchup() {
        for own in [nil, "", " \n ", "opponent-9"] as [String?] {
            XCTAssertEqual(summarize([game(teamID: own)]).n, 0)
        }
        XCTAssertEqual(summarize([game(teamID: "valid-former-team")]).n, 1)
    }
    func testWhitespaceContextAndImplausibleSeasonsAreUnavailable() {
        let bad = [context(playerID: " \n"), context(opponentID: " "), context(playerName: " "), context(opponentName: " "), context(seasons: []), context(seasons: [0]), context(seasons: [1999]), context(seasons: [9999])]
        for c in bad {
            let result = summarize([game()], context: c)
            XCTAssertEqual(result.n, 0)
            XCTAssertNotNil(result.unavailableReason)
        }
    }
    func testMalformedGameDateAndFractionalYardStatAreExcluded() {
        var brokenDate = game(); brokenDate = PlayerGame(id: brokenDate.id, date: Date(timeIntervalSince1970: .nan), opponent: brokenDate.opponent, value: brokenDate.value, workload: brokenDate.workload, sport: brokenDate.sport, playerID: brokenDate.playerID, playerTeamID: brokenDate.playerTeamID, opponentID: brokenDate.opponentID, marketKey: brokenDate.marketKey, season: brokenDate.season)
        XCTAssertEqual(summarize([brokenDate]).n, 0)
        XCTAssertEqual(summarize([game(value: -1.5, sport: "NFL", key: "player_rush_yds")], context: context(sport: "NFL", key: "player_rush_yds")).n, 0)
    }

    private func selectorMatchup(_ sport: String = "NBA") -> InsightMatchup {
        InsightMatchup(sport: sport, eventID: "scoreboard-event", homeTeam: "Fixture Home", awayTeam: "Fixture Away", commence: "2026-10-07T00:00:00Z")
    }
    private func selectorQuote(_ id: String = "quote-1", sport: String = "NBA", key: String = "player_points", name: String = "Home Athlete") -> Market {
        Market(id: id, event: "odds-provider-event", sport: sport, kind: "prop", name: name, opponent: "Fixture Away", label: "Over 10.5", aliases: "", p: nil, price: -110, market: key, book: "Fixture Book", quotes: [], updated: "2026-10-06T12:00:00Z", mode: "live", line: 10.5, eventLabel: "Fixture Away at Fixture Home", eventState: "pre", commence: "2026-10-07T00:00:00Z", side: "Over")
    }
    private func select(_ quotes: [Market], sport: String = "NBA", teams: [String: String]? = nil, limit: Int = 3) -> [Market] {
        OpponentPropHistoryEngine.insightMarkets(markets: quotes, matchup: selectorMatchup(sport), verifiedPlayerTeams: teams ?? Dictionary(quotes.compactMap { q in q.name == "Home Athlete" ? (q.id, "Fixture Home") : (q.name == "Away Athlete" ? (q.id, "Fixture Away") : nil) }, uniquingKeysWith: { a, _ in a }), limit: limit)
    }
    func testSelectorAcceptsCanonicalCrossProviderMatchupAcrossEverySport() {
        for (sport, key) in [("NBA", "player_points"), ("MLB", "batter_hits"), ("NFL", "player_rush_yds"), ("NHL", "player_points")] {
            for label in ["Fixture Away at Fixture Home", "Fixture Home @ Fixture Away", "  FIXTURE  AWAY  @  Fixture Home "] {
                var quote = selectorQuote(sport: sport, key: key); quote.eventLabel = label
                XCTAssertEqual(select([quote], sport: sport).map(\.id), [quote.id], sport)
            }
        }
    }
    func testSelectorRequiresExactTeamsAndKickoffEvenWithSameEventID() {
        let original = selectorQuote()
        var variants: [Market] = []
        var wrong = original; wrong.event = "scoreboard-event"; wrong.eventLabel = nil; variants.append(wrong)
        wrong = original; wrong.eventLabel = "Fixture Away at Fixture Home Extra"; variants.append(wrong)
        wrong = original; wrong.eventLabel = "Other Away at Fixture Home"; variants.append(wrong)
        wrong = original; wrong.commence = nil; variants.append(wrong)
        wrong = original; wrong.commence = "bad date"; variants.append(wrong)
        wrong = original; wrong.commence = "2026-10-08T00:00:00Z"; variants.append(wrong)
        wrong = original; wrong.commence = "2026-10-07T00:01:01Z"; variants.append(wrong)
        for quote in variants { XCTAssertTrue(select([quote]).isEmpty) }
        var boundary = original; boundary.commence = "2026-10-07T00:01:00Z"
        XCTAssertEqual(select([boundary]).count, 1)
    }
    func testSelectorRejectsUnverifiedPlayersOtherSportsAndInvalidQuoteFields() {
        let original = selectorQuote()
        var variants: [Market] = []
        var wrong = original; wrong.sport = "NHL"; variants.append(wrong)
        wrong = original; wrong.kind = "game"; variants.append(wrong)
        wrong = original; wrong.mode = "demo"; variants.append(wrong)
        wrong = original; wrong.market = "batter_hits"; variants.append(wrong)
        wrong = original; wrong.name = "Unknown Athlete"; variants.append(wrong)
        wrong = original; wrong.name = " "; variants.append(wrong)
        wrong = original; wrong.book = " \n"; variants.append(wrong)
        wrong = original; wrong.id = " "; variants.append(wrong)
        wrong = original; wrong.price = .nan; variants.append(wrong)
        wrong = original; wrong.price = 99; variants.append(wrong)
        wrong = original; wrong.line = .infinity; variants.append(wrong)
        wrong = original; wrong.line = nil; variants.append(wrong)
        wrong = original; wrong.line = -1; variants.append(wrong)
        wrong = original; wrong.side = nil; variants.append(wrong)
        wrong = original; wrong.side = "Both"; variants.append(wrong)
        for quote in variants { XCTAssertTrue(select([quote]).isEmpty) }
        XCTAssertTrue(select([original], teams: [:]).isEmpty)
        XCTAssertTrue(select([original], teams: [original.id: "Unrelated Team"]).isEmpty)
        XCTAssertTrue(select([original], teams: [original.name: "Fixture Home"]).isEmpty)
    }
    func testSelectorDoesNotOfferCompletedOrUnavailableGameProps() {
        for state in ["post", "FINAL", "completed", "canceled", "cancelled", "postponed", "suspended"] {
            var quote = selectorQuote(); quote.eventState = state
            XCTAssertTrue(select([quote]).isEmpty, state)
        }
    }
    func testSelectorQuarantinesDuplicateIDsAndDeduplicatesBooksAndSides() {
        let original = selectorQuote("a")
        var anotherBook = selectorQuote("b"); anotherBook.book = "Other Book"
        var anotherSide = selectorQuote("c"); anotherSide.side = "Under"
        XCTAssertEqual(select([anotherSide, anotherBook, original]).map(\.id), ["a"])
        XCTAssertTrue(select([original, original]).isEmpty)
        var conflict = original; conflict.line = 99
        XCTAssertTrue(select([original, conflict]).isEmpty)
    }
    func testSelectorBalancesTeamsCapsWorkAndIsStableUnderOrdering() {
        let home1 = selectorQuote("a")
        let home2 = selectorQuote("b", key: "player_assists")
        let home3 = selectorQuote("c", key: "player_rebounds")
        let away = selectorQuote("z", name: "Away Athlete")
        let quotes = [home1, home2, home3, away]
        let result = select(quotes)
        XCTAssertEqual(result.map(\.id), ["a", "z", "b"])
        XCTAssertEqual(select(quotes.reversed()).map(\.id), result.map(\.id))
        XCTAssertEqual(select(quotes, limit: 999).count, 3)
        XCTAssertEqual(select(quotes, limit: 2).map(\.id), ["a", "z"])
        XCTAssertEqual(select(quotes, limit: 1).count, 1)
    }
    func testSelectorAcceptsNegativeLineOnlyForSignedNFLYards() {
        var yards = selectorQuote(sport: "NFL", key: "player_rush_yds"); yards.line = -2
        XCTAssertEqual(select([yards], sport: "NFL").count, 1)
        yards.market = "player_pass_tds"
        XCTAssertTrue(select([yards], sport: "NFL").isEmpty)
    }

}

extension OpponentPropHistoryTests {
    func testVenueConflictQuarantinesOtherwiseIdenticalOpponentGame() {
        var home = game(); home.homeAway = "home"
        var away = game(); away.homeAway = "away"
        XCTAssertEqual(summarize([home, away]).n, 0)
        XCTAssertEqual(summarize([home, home]).n, 1)
    }
}
