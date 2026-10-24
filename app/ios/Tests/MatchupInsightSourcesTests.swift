import XCTest
@testable import PLONNCore

final class MatchupInsightSourcesTests: XCTestCase {
    private func scheduleEvent(id: String, date: String = "2026-01-01T00:00:00Z", ownID: String = "A", allowed: Double = 100) -> [String: Any] {
        ["id": id, "date": date, "competitions": [["status": ["type": ["completed": true]],
            "competitors": [["id": ownID, "score": ["value": 110.0]], ["id": "B", "score": ["value": allowed]]]]]]
    }
    func testNBAScheduleAllowanceRequiresCompleteMatchingSample() {
        let cutoff = Date(timeIntervalSince1970: 1780000000)
        let first = scheduleEvent(id: "1", allowed: 100)
        let second = scheduleEvent(id: "2", allowed: 120)
        XCTAssertEqual(MatchupInsightSources.nbaScheduleAllowance(teamID: "A", payload: ["events": [first, first, second]], sample: 2, cutoff: cutoff, excludingEventID: "target"), 110)
        XCTAssertNil(MatchupInsightSources.nbaScheduleAllowance(teamID: "A", payload: ["events": [first]], sample: 2, cutoff: cutoff, excludingEventID: "target"))
        XCTAssertNil(MatchupInsightSources.nbaScheduleAllowance(teamID: "A", payload: ["events": [first, scheduleEvent(id: "1", allowed: 101)]], sample: 1, cutoff: cutoff, excludingEventID: "target"))
    }
    func testNBAScheduleAllowanceExcludesTargetAndFutureAndRejectsAmbiguity() {
        let cutoff = Date(timeIntervalSince1970: 1780000000)
        let first = scheduleEvent(id: "1")
        let target = scheduleEvent(id: "target")
        let future = scheduleEvent(id: "future", date: "2027-01-01T00:00:00Z")
        XCTAssertEqual(MatchupInsightSources.nbaScheduleAllowance(teamID: "A", payload: ["events": [first, target, future]], sample: 1, cutoff: cutoff, excludingEventID: "target"), 100)
        for invalid in [scheduleEvent(id: "wrong", ownID: "C"), scheduleEvent(id: "bad", allowed: .nan), scheduleEvent(id: "badDate", date: "invalid")] {
            XCTAssertNil(MatchupInsightSources.nbaScheduleAllowance(teamID: "A", payload: ["events": [invalid]], sample: 1, cutoff: cutoff, excludingEventID: "target"))
        }
    }

    func testComparisonGroupsAreSpecificToEachSport() {
        XCTAssertEqual(MatchupInsightSources.comparisonGroups(sport: "NFL"), ["Scoring", "Passing", "Rushing", "Disruption"])
        XCTAssertEqual(MatchupInsightSources.comparisonGroups(sport: "NBA"), ["Efficiency", "Shooting", "Possession", "Defense"])
        XCTAssertEqual(MatchupInsightSources.comparisonGroups(sport: "NHL"), ["Goals", "Shots", "Goal prevention"])
        XCTAssertEqual(MatchupInsightSources.comparisonGroups(sport: "MLB"), ["Batting", "Power", "Discipline", "Pitching"])
        XCTAssertTrue(MatchupInsightSources.comparisonGroups(sport: "unknown").isEmpty)
        let hockey = Set(MatchupInsightSources.comparisonGroups(sport: "NHL"))
        for foreignSport in ["MLB", "NBA"] {
            XCTAssertTrue(hockey.isDisjoint(with: MatchupInsightSources.comparisonGroups(sport: foreignSport)))
        }
    }

    private func payload(_ pairs: [(String, Double)], opponent: [(String, Double)] = []) -> [String: Any] {
        func categories(_ values: [(String, Double)]) -> [[String: Any]] {
            [["name": "general", "stats": values.map { ["name": $0.0, "value": $0.1] as [String: Any] }]]
        }
        return ["season": ["year": 2027, "type": 1],
                "requestedSeason": ["year": 2026, "type": 2, "displayName": "2025-26"],
                "results": ["stats": ["categories": categories(pairs)], "opponent": categories(opponent)]]
    }
    // Captured public ESPN response shapes; unrelated metrics trimmed.
    func testCapturedProviderShapes() throws {
        do {
            let data = #"{"status":"success","results":{"stats":{"categories":[{"name":"general","stats":[{"name":"avgRebounds","value":46.280487},{"name":"gamesPlayed","value":82.0}]},{"name":"offensive","stats":[{"name":"avgThreePointFieldGoalsMade","value":13.695122},{"name":"avgPoints","value":120.86585},{"name":"avgAssists","value":28.95122}]},{"name":"defensive","stats":[]}]}},"season":{"year":2027,"type":1,"name":"Preseason","displayName":"2026-27"},"requestedSeason":{"year":2026,"type":2,"name":"Regular Season","displayName":"2025-26","qualifiedPostSeason":false}}"#.data(using: .utf8)!
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let profile = try XCTUnwrap(MatchupInsightSources.espn(team: "Fixture", sport: "NBA", requestedYear: 2026, payload: payload))
            XCTAssertEqual(profile.sample, 82)
            XCTAssertFalse(profile.offense.isEmpty)
        }
        do {
            let data = #"{"status":"success","results":{"stats":{"categories":[{"name":"passing","stats":[{"name":"passingAttempts","value":98.0},{"name":"teamGamesPlayed","value":3.0},{"name":"passingYardsPerGame","value":236.0}]},{"name":"rushing","stats":[{"name":"rushingAttempts","value":105.0},{"name":"rushingYardsPerGame","value":184.3},{"name":"teamGamesPlayed","value":3.0}]},{"name":"receiving","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"miscellaneous","stats":[]},{"name":"defensive","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"defensiveInterceptions","stats":[]},{"name":"general","stats":[{"name":"gamesPlayed","value":3.0}]},{"name":"returning","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"kicking","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"punting","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"scoring","stats":[]}]},"opponent":[{"name":"passing","stats":[{"name":"passingAttempts","value":84.0},{"name":"teamGamesPlayed","value":3.0},{"name":"passingYardsPerGame","value":219.0}]},{"name":"rushing","stats":[{"name":"rushingAttempts","value":69.0},{"name":"rushingYardsPerGame","value":117.0},{"name":"teamGamesPlayed","value":3.0}]},{"name":"receiving","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"miscellaneous","stats":[]},{"name":"defensive","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"defensiveInterceptions","stats":[]},{"name":"general","stats":[{"name":"gamesPlayed","value":3.0}]},{"name":"returning","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"kicking","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"punting","stats":[{"name":"teamGamesPlayed","value":3.0}]},{"name":"scoring","stats":[]}]},"season":{"year":2026,"type":2,"name":"Regular Season","displayName":"2026"},"requestedSeason":{"year":2026,"type":2,"name":"Regular Season","displayName":"2026","qualifiedPostSeason":false}}"#.data(using: .utf8)!
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let profile = try XCTUnwrap(MatchupInsightSources.espn(team: "Fixture", sport: "NFL", requestedYear: 2026, payload: payload))
            XCTAssertEqual(profile.sample, 3)
            XCTAssertFalse(profile.offense.isEmpty)
        }
        do {
            let data = #"{"status":"success","results":{"stats":{"categories":[{"name":"general","stats":[{"name":"games","value":82.0}]},{"name":"offensive","stats":[{"name":"goals","value":282.0},{"name":"shotsTotal","value":2439.0}]},{"name":"defensive","stats":[{"name":"goalsAgainst","value":265.0},{"name":"shotsAgainst","value":2190.0}]},{"name":"penalties","stats":[]}]}},"season":{"year":2027,"type":2,"name":"Regular Season","displayName":"2026-27"},"requestedSeason":{"year":2026,"type":2,"name":"Regular Season","displayName":"2025-26","qualifiedPostSeason":true}}"#.data(using: .utf8)!
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let profile = try XCTUnwrap(MatchupInsightSources.espn(team: "Fixture", sport: "NHL", requestedYear: 2026, payload: payload))
            XCTAssertEqual(profile.sample, 82)
            XCTAssertFalse(profile.offense.isEmpty)
        }
    }

    func testNBAUsesRequestedPeriodAndNeverInventsDefense() {
        let profile = MatchupInsightSources.espn(team: "Celtics", sport: "NBA", requestedYear: 2026,
            payload: payload([("gamesPlayed", 82), ("avgPoints", 120.86585), ("avgRebounds", 46.280487)]))
        XCTAssertEqual(profile?.sample, 82)
        XCTAssertEqual(profile?.offense["avgPoints"], 120.86585)
        XCTAssertTrue(profile?.allowed.isEmpty == true)
        XCTAssertEqual(profile?.source, "2025-26 ESPN team statistics")
        XCTAssertNil(MatchupInsightSources.espn(team: "Celtics", sport: "NBA", requestedYear: 2025,
            payload: payload([("gamesPlayed", 82), ("avgPoints", 120)])))
    }
    func testNFLRepeatingGamesAndAttemptsAreTotals() {
        let profile = MatchupInsightSources.espn(team: "Bills", sport: "NFL", requestedYear: 2026,
            payload: payload([("teamGamesPlayed", 3), ("teamGamesPlayed", 3), ("rushingAttempts", 105),
                ("passingAttempts", 98), ("passingYardsPerGame", 236), ("rushingYardsPerGame", 184.3)],
                opponent: [("teamGamesPlayed", 3), ("passingYardsPerGame", 200)]))
        XCTAssertEqual(profile?.offense["rushingAttemptsPerGame"], 35)
        XCTAssertEqual(profile?.offense["passingAttemptsPerGame"], 98.0 / 3)
        XCTAssertEqual(profile?.allowed["passingYardsPerGame"], 200)
        XCTAssertNil(profile?.allowed["rushingYardsPerGame"])
    }
    func testNFLDefensiveSampleMustMatchOwnSeasonSample() {
        for counts in [[("teamGamesPlayed", 2.0)], [("teamGamesPlayed", 3.0), ("teamGamesPlayed", 2.0)], [("teamGamesPlayed", Double.nan)], []] {
            let profile = MatchupInsightSources.espn(team: "A", sport: "NFL", requestedYear: 2026,
                payload: payload([("teamGamesPlayed", 3), ("passingYardsPerGame", 236)],
                    opponent: counts + [("passingYardsPerGame", 200)]))
            XCTAssertEqual(profile?.offense["passingYardsPerGame"], 236)
            XCTAssertTrue(profile?.allowed.isEmpty == true)
        }
    }

    func testNHLUsesOwnDefensiveTotals() {
        let profile = MatchupInsightSources.espn(team: "Panthers", sport: "NHL", requestedYear: 2026,
            payload: payload([("games", 82), ("goals", 282), ("shotsTotal", 2439),
                ("goalsAgainst", 265), ("shotsAgainst", 2190), ("avgGoalsAgainst", 3.232)]))
        XCTAssertEqual(profile?.offense["avgGoals"], 282.0 / 82)
        XCTAssertEqual(profile?.offense["avgShots"], 2439.0 / 82)
        XCTAssertEqual(profile?.allowed["avgGoals"], 265.0 / 82)
        XCTAssertEqual(profile?.allowed["avgShots"], 2190.0 / 82)
    }
    func testNHLNeverConsumesBaseballOrBasketballMetrics() {
        let foreign: [(String, Double)] = [("games", 82), ("avg", 0.25), ("slg", 0.45), ("ops", 0.8),
            ("avgPoints", 110), ("fieldGoalsMade", 3000), ("avgRebounds", 40)]
        XCTAssertNil(MatchupInsightSources.espn(team: "A", sport: "NHL", requestedYear: 2026,
            payload: payload(foreign)))
        let partial = MatchupInsightSources.espn(team: "A", sport: "NHL", requestedYear: 2026,
            payload: payload(foreign + [("shotsTotal", 2460)]))
        XCTAssertEqual(partial?.offense, ["avgShots": 30])
        XCTAssertTrue(partial?.allowed.isEmpty == true)
        XCTAssertNil(MatchupInsightSources.espn(team: "A", sport: "NHL", requestedYear: 2026,
            payload: payload([("games", 0), ("goals", 0), ("shotsTotal", 0)])))
    }

    func testConflictsAndInvalidValuesAreUnavailable() {
        let profile = MatchupInsightSources.espn(team: "A", sport: "NBA", requestedYear: 2026,
            payload: payload([("gamesPlayed", 82), ("avgPoints", 100), ("avgPoints", 101),
                ("avgAssists", -.infinity), ("avgRebounds", 40)]))
        XCTAssertNil(profile?.offense["avgPoints"])
        XCTAssertNil(profile?.offense["avgAssists"])
        XCTAssertEqual(profile?.offense["avgRebounds"], 40)
        for games in [0.0, -1, 1.5, .infinity] {
            XCTAssertNil(MatchupInsightSources.espn(team: "A", sport: "NBA", requestedYear: 2026,
                payload: payload([("gamesPlayed", games), ("avgPoints", 100)])))
        }
        XCTAssertNil(MatchupInsightSources.espn(team: "A", sport: "NBA", requestedYear: 2026,
            payload: payload([("gamesPlayed", 82), ("gamesPlayed", 81), ("avgPoints", 100)])))
    }
    func testNFLScoringReconcilesRoundedRatesAgainstVerifiedTotals() throws {
        let rates:[(String,Double)] = [("teamGamesPlayed",6),("totalPoints",136),
            ("totalPointsPerGame",22.67),("totalPointsPerGame",22.667)]
        for ordering in [rates,Array(rates.reversed())] {
            let profile=try XCTUnwrap(MatchupInsightSources.espn(team:"Patriots",sport:"NFL",requestedYear:2026,
                payload:payload(ordering,opponent:[("teamGamesPlayed",6),("totalPoints",113),("totalPointsPerGame",18.833)])))
            XCTAssertEqual(try XCTUnwrap(profile.offense["totalPointsPerGame"]),136.0/6,accuracy:1e-9)
            XCTAssertEqual(try XCTUnwrap(profile.allowed["totalPointsPerGame"]),113.0/6,accuracy:1e-9)
        }
        for conflict in [("totalPointsPerGame",22.68),("totalPointsPerGame",22.669),("totalPointsPerGame",Double.nan),
                         ("totalPoints",137),("totalPoints",136.5)] {
            let profile=MatchupInsightSources.espn(team:"Patriots",sport:"NFL",requestedYear:2026,
                payload:payload(rates+[conflict,("rushingYardsPerGame",130.7)]))
            XCTAssertNil(profile?.offense["totalPointsPerGame"],"A real conflict must still be rejected")
            XCTAssertEqual(profile?.offense["rushingYardsPerGame"],130.7)
        }
        let noCount=MatchupInsightSources.espn(team:"Patriots",sport:"NFL",requestedYear:2026,
            payload:payload(rates.filter{$0.0 != "totalPoints"}+[("rushingYardsPerGame",130.7)]))
        XCTAssertNil(noCount?.offense["totalPointsPerGame"],"Rounding reconciliation requires a verified count")
    }
    func testMLBPartialSidesAndSeasonValidation() {
        let batting: [String: Any] = ["season": "2026", "gamesPlayed": 162, "avg": ".250", "slg": ".410", "homeRuns": 200]
        let profile = MatchupInsightSources.mlb(team: "Mets", requestedYear: 2026, batting: batting, pitching: nil)
        XCTAssertEqual(profile?.offense["avg"], 0.25)
        XCTAssertEqual(profile?.offense["homeRunsPerGame"], 200.0 / 162)
        XCTAssertTrue(profile?.allowed.isEmpty == true)
        XCTAssertNil(MatchupInsightSources.mlb(team: "Mets", requestedYear: 2025, batting: batting, pitching: nil))
        let pitching: [String: Any] = ["season": "2026", "gamesPlayed": 162, "avg": ".240", "slg": ".390", "homeRuns": 170]
        XCTAssertEqual(MatchupInsightSources.mlb(team: "Mets", requestedYear: 2026, batting: nil, pitching: pitching)?.allowed["slg"], 0.39)
    }
    func testRecentGamesExcludeCutoffAndGuardConflictingEvents() {
        let cutoff = Date(timeIntervalSince1970: 1000)
        let observed = InsightGameObservation(id: "1", date: Date(timeIntervalSince1970: 900), scored: 110, allowed: 100)
        let future = InsightGameObservation(id: "2", date: cutoff, scored: 200, allowed: 200)
        let profile = MatchupInsightSources.recent(team: "A", sport: "NBA", games: [observed, observed, future], cutoff: cutoff)
        XCTAssertEqual(profile?.sample, 1)
        XCTAssertEqual(profile?.offense["avgPoints"], 110)
        XCTAssertEqual(profile?.through, observed.date)
        let conflict = InsightGameObservation(id: "1", date: observed.date, scored: 111, allowed: 100)
        XCTAssertNil(MatchupInsightSources.recent(team: "A", sport: "NBA", games: [observed, conflict], cutoff: cutoff))
        XCTAssertNil(MatchupInsightSources.recent(team: "A", sport: "NFL", games: [observed], cutoff: cutoff))
        XCTAssertEqual(MatchupInsightSources.recent(team: "A", sport: "NHL", games: [observed], cutoff: cutoff)?.offense["avgGoals"], 110)
    }

    func testRosterRequiresUniquePlayerAcrossBothVerifiedTeams() {
        XCTAssertEqual(MatchupInsightSources.uniqueRosterTeam(player: " José Altuve ", rosters: ["A": ["Jose Altuve"], "B": ["Other Player"]], expectedTeams: ["A", "B"]), "A")
        for rosters in [["A": ["Jose Altuve"]], ["A": ["Jose Altuve"], "B": ["Jose Altuve"]], ["A": ["Jose Altuve", "Jose Altuve"], "B": []], ["A": ["Jose Altuve"], "B": [], "C": []]] {
            XCTAssertNil(MatchupInsightSources.uniqueRosterTeam(player: "Jose Altuve", rosters: rosters, expectedTeams: ["A", "B"]))
        }
    }
}
