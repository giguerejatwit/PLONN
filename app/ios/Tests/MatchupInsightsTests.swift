import XCTest
@testable import PLONNCore

/// Synthetic observations and quotes: none of these are claims about real teams.
final class MatchupInsightsTests: XCTestCase {
    private let start = "2026-10-03T18:00:00Z"
    private let cases: [(String, String, String, String)] = [
        ("NFL", "rushingYardsPerGame", "player_rush_yds", "rushing yards"),
        ("NFL", "passingYardsPerGame", "player_pass_yds", "passing yards"),
        ("NFL", "passingYardsPerGame", "player_reception_yds", "receiving yards"),
        ("NBA", "avgPoints", "player_points", "points"),
        ("NBA", "avgAssists", "player_assists", "assists"),
        ("NBA", "avgRebounds", "player_rebounds", "rebounds"),
        ("NBA", "avgThreePointFieldGoalsMade", "player_threes", "made threes"),
        ("MLB", "avg", "batter_hits", "hits"),
        ("MLB", "slg", "batter_total_bases", "total bases"),
        ("MLB", "homeRunsPerGame", "batter_home_runs", "home runs"),
        ("NHL", "avgGoals", "player_goals", "goals"),
        ("NHL", "avgShots", "player_shots_on_goal", "shots on goal")
    ]
    private func matchup(_ sport: String = "NFL") -> InsightMatchup {
        InsightMatchup(sport: sport, eventID: "fixture-event", homeTeam: "Fixture Home", awayTeam: "Fixture Away", commence: start)
    }
    private func profile(_ team: String, offense: [String: Double] = [:], allowed: [String: Double] = [:], sample: Int = 12, source: String = "fixture season rates") -> InsightTeamProfile {
        InsightTeamProfile(team: team, sample: sample, offense: offense, allowed: allowed, source: source)
    }
    private func profiles(_ key: String, offense: Double = 12.34, allowed: Double = 15.67) -> [InsightTeamProfile] {
        [profile("Fixture Home", offense: [key: offense]), profile("Fixture Away", allowed: [key: allowed])]
    }
    private func market(_ sport: String = "NFL", family: String = "player_rush_yds", id: String = "prop-fixture") -> Market {
        Market(id: id, event: "fixture-event", sport: sport, kind: "prop", name: "Fixture Athlete", opponent: "Fixture Away", label: "Over 6.5", aliases: "", p: nil, price: -110, market: family, book: "Fixture Book", quotes: [], updated: "2026-10-02T18:00:00Z", mode: "live", line: 6.5, eventState: "pre", commence: start, side: "Over")
    }
    private func generate(_ sport: String = "NFL", key: String = "rushingYardsPerGame", profiles supplied: [InsightTeamProfile]? = nil, markets: [Market]? = nil, team: String = "Fixture Home") -> [MatchupInsight] {
        MatchupInsightEngine.generate(matchup: matchup(sport), profiles: supplied ?? profiles(key), props: (markets ?? [market(sport)]).map { InsightPlayerProp(market: $0, verifiedPlayerTeam: team) })
    }

    func testEverySupportedSportAndMarketUsesActualEvidenceAndExactProp() throws {
        for (sport, key, family, label) in cases {
            let results = generate(sport, key: key, markets: [market(sport, family: family)])
            let insight = try XCTUnwrap(results.first, "\(sport)/\(family)")
            XCTAssertEqual(results.count, 1)
            XCTAssertEqual(insight.marketID, "prop-fixture")
            XCTAssertTrue(insight.text.contains("Fixture Home"))
            XCTAssertTrue(insight.text.contains("Fixture Away"))
            XCTAssertTrue(insight.text.contains("Fixture Athlete, over"))
            XCTAssertTrue(insight.text.contains(label + " (Fixture Book)"))
            XCTAssertTrue(insight.evidence.contains("fixture season rates · 12 games"))
            for phrase in ["lead back", "guaranteed", "will hit", "favors", "high", "runs often"] {
                XCTAssertFalse(insight.text.lowercased().contains(phrase), phrase)
            }
            // Decimal points do not count as sentence endings.
            XCTAssertLessThanOrEqual(insight.text.components(separatedBy: ". ").count, 2)
            XCTAssertLessThan(insight.text.count, 320)
        }
    }
    func testEmptyAndUnsupportedInputsProduceNoInventedInsight() {
        XCTAssertTrue(MatchupInsightEngine.generate(matchup: matchup(), profiles: [], props: []).isEmpty)
        XCTAssertTrue(generate("SOCCER").isEmpty)
        XCTAssertTrue(generate(profiles: [profile("Fixture Home"), profile("Fixture Away")]).isEmpty)
        XCTAssertTrue(generate(profiles: [profile("Unrelated", offense: ["rushingYardsPerGame": 99])]).isEmpty)
    }
    func testOneSidedEvidenceNeverLinksPropOrInventsMissingNumber() throws {
        let own = try XCTUnwrap(generate(profiles: [profile("Fixture Home", offense: ["rushingYardsPerGame": 12.34])]).first)
        XCTAssertNil(own.marketID)
        XCTAssertTrue(own.text.contains("12.3"))
        XCTAssertTrue(own.text.contains("defensive rate is unavailable"))
        XCTAssertFalse(own.text.contains("allows"))
        XCTAssertFalse(own.evidence.contains("Fixture Away:"))
        let defense = try XCTUnwrap(generate(profiles: [profile("Fixture Away", allowed: ["rushingYardsPerGame": 15.67])]).first)
        XCTAssertNil(defense.marketID)
        XCTAssertTrue(defense.text.contains("allows 15.7"))
        XCTAssertTrue(defense.text.contains("offensive rate is unavailable"))
        XCTAssertFalse(defense.text.contains("Fixture Home averages"))
        XCTAssertFalse(defense.evidence.contains("Fixture Home:"))
    }
    func testInvalidNumbersAreUnavailableAcrossAllSports() {
        for (sport, key, _, _) in cases {
            for value in [Double.nan, Double.infinity, -Double.infinity, -1] {
                XCTAssertTrue(generate(sport, key: key, profiles: profiles(key, offense: value, allowed: value)).isEmpty)
            }
            XCTAssertFalse(generate(sport, key: key, profiles: profiles(key, offense: 0, allowed: 0)).isEmpty)
        }
    }
    func testMissingSampleAndSourceInvalidateEvidence() {
        for bad in [profile("Fixture Home", offense: ["rushingYardsPerGame": 12], sample: 0), profile("Fixture Home", offense: ["rushingYardsPerGame": 12], sample: -1), profile("Fixture Home", offense: ["rushingYardsPerGame": 12], source: " \n")] {
            XCTAssertTrue(generate(profiles: [bad]).isEmpty)
        }
    }
    func testDuplicateConflictingProfilesAreNotAveragedOrArbitrarilySelected() throws {
        let base = profiles("rushingYardsPerGame")
        for extra in [base[0], profile("Fixture Home", offense: ["rushingYardsPerGame": 999])] {
            let insight = try XCTUnwrap(generate(profiles: base + [extra]).first)
            XCTAssertNil(insight.marketID)
            XCTAssertTrue(insight.text.contains("offensive rate is unavailable"))
            XCTAssertFalse(insight.text.contains("999"))
            XCTAssertFalse(insight.text.contains("12.3"))
        }
    }
    func testWrongMarketFamilyCannotConnectPropForAnySport() throws {
        for (sport, key, _, _) in cases {
            let insight = try XCTUnwrap(generate(sport, key: key, markets: [market(sport, family: "player_unrelated")]).first)
            XCTAssertNil(insight.marketID)
            XCTAssertFalse(insight.text.contains("Related prop"))
        }
    }
    func testRushAndPassCannotCrossLink() throws {
        XCTAssertNil(try XCTUnwrap(generate(markets: [market(family: "player_pass_yds")]).first).marketID)
        XCTAssertNil(try XCTUnwrap(generate(key: "passingYardsPerGame", markets: [market(family: "player_rush_yds")]).first).marketID)
    }
    func testWrongEventLeaguePlayerTeamAndStartCannotLink() throws {
        let original = market()
        var variants: [Market] = []
        var altered = original; altered.event = "another-event"; variants.append(altered)
        altered = original; altered.sport = "NBA"; variants.append(altered)
        altered = original; altered.commence = "2026-10-04T18:00:00Z"; variants.append(altered)
        altered = original; altered.commence = nil; variants.append(altered)
        altered = original; altered.commence = "invalid"; variants.append(altered)
        for invalid in variants { XCTAssertNil(try XCTUnwrap(generate(markets: [invalid]).first).marketID) }
        for team in ["Fixture Away", "", "Unrelated"] { XCTAssertNil(try XCTUnwrap(generate(team: team).first).marketID) }
    }
    func testInvalidAndSamplePropsDoNotLink() throws {
        let original = market()
        var variants: [Market] = []
        var altered = original; altered.line = .nan; variants.append(altered)
        altered = original; altered.line = -1; variants.append(altered)
        altered = original; altered.line = nil; variants.append(altered)
        altered = original; altered.price = .infinity; variants.append(altered)
        altered = original; altered.mode = "demo"; variants.append(altered)
        altered = original; altered.kind = "game"; variants.append(altered)
        altered = original; altered.side = "Both"; variants.append(altered)
        altered = original; altered.side = nil; variants.append(altered)
        altered = original; altered.name = " \n"; variants.append(altered)
        altered = original; altered.book = ""; variants.append(altered)
        altered = original; altered.id = ""; variants.append(altered)
        for invalid in variants { XCTAssertNil(try XCTUnwrap(generate(markets: [invalid]).first).marketID) }
    }
    func testAmbiguousDuplicatePropIDCannotLink() throws {
        let original = market()
        var conflict = original; conflict.line = 99
        for duplicate in [original, conflict] {
            XCTAssertNil(try XCTUnwrap(generate(markets: [original, duplicate]).first).marketID)
        }
    }
    func testStableSelectionIdentityAndNumbersDespiteInputOrdering() throws {
        let a = market(id: "a"), z = market(id: "z")
        let first = try XCTUnwrap(generate(markets: [z, a]).first)
        let second = try XCTUnwrap(generate(profiles: profiles("rushingYardsPerGame").reversed(), markets: [a, z]).first)
        XCTAssertEqual(first.id, "NFL|fixture-event|Fixture Home|rushingYardsPerGame")
        XCTAssertEqual(first.id, second.id)
        XCTAssertEqual(first.text, second.text)
        XCTAssertEqual(first.evidence, second.evidence)
        XCTAssertEqual(first.marketID, "a")
        XCTAssertTrue(first.text.contains("12.3"))
        XCTAssertTrue(first.text.contains("15.7"))
    }
    func testBaseballPreservesThreeDigitRatePrecision() throws {
        let insight = try XCTUnwrap(generate("MLB", key: "avg", profiles: profiles("avg", offense: 0.2714, allowed: 0.2536), markets: [market("MLB", family: "batter_hits")]).first)
        XCTAssertTrue(insight.text.contains("0.271 batting average"))
        XCTAssertTrue(insight.text.contains("0.254 opponent batting average"))
        XCTAssertFalse(insight.text.contains("27.1"))
    }
    func testOpponentPropDoesNotBorrowHomeEvidence() throws {
        let supplied = [profile("Fixture Home", offense: ["rushingYardsPerGame": 20], allowed: ["rushingYardsPerGame": 10]), profile("Fixture Away", offense: ["rushingYardsPerGame": 30], allowed: ["rushingYardsPerGame": 40])]
        let results = generate(profiles: supplied, team: "Fixture Away")
        let home = try XCTUnwrap(results.first(where: { $0.team == "Fixture Home" }))
        let away = try XCTUnwrap(results.first(where: { $0.team == "Fixture Away" }))
        XCTAssertNil(home.marketID)
        XCTAssertEqual(away.marketID, "prop-fixture")
        XCTAssertTrue(away.text.contains("Fixture Away averages 30.0"))
        XCTAssertTrue(away.text.contains("Fixture Home allows 10.0"))
    }
    func testCrossProviderEventIDsRequireExactCanonicalMatchupAndKickoff() throws {
        for label in ["Fixture Away at Fixture Home", "Fixture Home @ Fixture Away", "  FIXTURE   AWAY  @  fixture home  "] {
            var prop = market(); prop.event = "odds-provider-event"; prop.eventLabel = label
            XCTAssertEqual(try XCTUnwrap(generate(markets: [prop]).first).marketID, prop.id)
            for wrong in ["Fixture Away at Fixture Home Extra", "Other Away at Fixture Home", "Fixture Home vs Fixture Away"] {
                prop.eventLabel = wrong
                XCTAssertNil(try XCTUnwrap(generate(markets: [prop]).first).marketID)
            }
        }
        var prop = market(); prop.event = "odds-provider-event"; prop.eventLabel = "Fixture Away at Fixture Home"
        prop.commence = "2026-10-03T18:01:00Z"
        XCTAssertEqual(try XCTUnwrap(generate(markets: [prop]).first).marketID, prop.id)
        prop.commence = "2026-10-03T18:01:01Z"
        XCTAssertNil(try XCTUnwrap(generate(markets: [prop]).first).marketID)
        prop.commence = "2026-10-04T18:00:00Z"
        XCTAssertNil(try XCTUnwrap(generate(markets: [prop]).first).marketID)
    }
    func testUnknownMatchupKickoffNeverLinksEvenMatchingEventID() throws {
        let unknown = InsightMatchup(sport: "NFL", eventID: "fixture-event", homeTeam: "Fixture Home", awayTeam: "Fixture Away", commence: nil)
        let result = MatchupInsightEngine.generate(matchup: unknown, profiles: profiles("rushingYardsPerGame"), props: [InsightPlayerProp(market: market(), verifiedPlayerTeam: "Fixture Home")])
        XCTAssertNil(try XCTUnwrap(result.first).marketID)
    }
    func testCompletedUnavailableAndInvalidOddsPropsCannotLink() throws {
        for state in ["post", "FINAL", "completed", "canceled", "cancelled", "postponed", "suspended"] {
            var prop = market(); prop.eventState = state
            XCTAssertNil(try XCTUnwrap(generate(markets: [prop]).first).marketID, state)
        }
        for price in [0.0, 99, -99] {
            var prop = market(); prop.price = price
            XCTAssertNil(try XCTUnwrap(generate(markets: [prop]).first).marketID)
        }
        var prop = market(); prop.book = " \n "
        XCTAssertNil(try XCTUnwrap(generate(markets: [prop]).first).marketID)
    }
    func testKnownIncompatiblePeriodsCannotProduceComparison() {
        var supplied = profiles("rushingYardsPerGame")
        supplied[0].periodKey = "2026 regular season"
        supplied[1].periodKey = "2025 regular season"
        XCTAssertTrue(generate(profiles: supplied).isEmpty)
        supplied[1].periodKey = supplied[0].periodKey
        XCTAssertFalse(generate(profiles: supplied).isEmpty)
    }
    func testObservationsAtOrAfterKickoffAreNotUsedAsPregameEvidence() throws {
        for offset in [0.0, 1, 86400] {
            var supplied = profiles("rushingYardsPerGame")
            supplied[0].through = ISO8601DateFormatter().date(from: start)!.addingTimeInterval(offset)
            let result = try XCTUnwrap(generate(profiles: supplied).first)
            XCTAssertNil(result.marketID)
            XCTAssertTrue(result.text.contains("offensive rate is unavailable"))
            XCTAssertFalse(result.text.contains("12.3"))
        }
    }
    func testNFLRunTendencyOnlyAppearsWhenAttemptsAreObserved() throws {
        var supplied = profiles("rushingYardsPerGame")
        supplied[0] = profile("Fixture Home", offense: ["rushingYardsPerGame": 12.34, "rushingAttemptsPerGame": 28.14])
        XCTAssertTrue(try XCTUnwrap(generate(profiles: supplied).first).text.contains("runs 28.1 times per game"))
        XCTAssertFalse(try XCTUnwrap(generate().first).text.contains("times per game"))
    }

    func testRecentPairedPointsReplacePartialSeasonWithoutLosingPairedAssists() throws {
        let season = [profile("Fixture Home", offense: ["avgPoints": 110, "avgAssists": 25]), profile("Fixture Away", allowed: ["avgAssists": 22])]
        let recent = [profile("Fixture Home", offense: ["avgPoints": 115], sample: 5, source: "recent five games"), profile("Fixture Away", allowed: ["avgPoints": 108], sample: 5, source: "recent five games")]
        let results = MatchupInsightEngine.generate(matchup: matchup("NBA"), profiles: season, props: [], recentProfiles: recent)
        let points = try XCTUnwrap(results.first(where: { $0.team == "Fixture Home" && $0.topic == "Scoring" }))
        let assists = try XCTUnwrap(results.first(where: { $0.team == "Fixture Home" && $0.topic == "Assists" }))
        XCTAssertTrue(points.hasPairedEvidence)
        XCTAssertTrue(points.text.contains("115.0"))
        XCTAssertTrue(points.evidence.contains("recent five games"))
        XCTAssertTrue(assists.hasPairedEvidence)
        XCTAssertTrue(assists.text.contains("25.0"))
        XCTAssertTrue(assists.evidence.contains("fixture season rates"))
        XCTAssertEqual(Set(results.map(\.id)).count, results.count)
    }
    func testPartialRecentProfileCannotEraseOrMixPrimaryOneSidedEvidence() throws {
        var primary = [profile("Fixture Home", offense: ["avgPoints": 110])]
        primary[0].periodKey = "season2026"
        var recent = [profile("Fixture Away", allowed: ["avgPoints": 108], sample: 5, source: "recent five games")]
        recent[0].periodKey = "recent2026"
        let results = MatchupInsightEngine.generate(matchup: matchup("NBA"), profiles: primary, props: [], recentProfiles: recent)
        let points = try XCTUnwrap(results.first(where: { $0.team == "Fixture Home" && $0.topic == "Scoring" }))
        XCTAssertFalse(points.hasPairedEvidence)
        XCTAssertTrue(points.text.contains("110.0"))
        XCTAssertFalse(points.text.contains("108.0"))
        XCTAssertTrue(points.text.contains("defensive rate is unavailable"))
        XCTAssertFalse(points.evidence.contains("recent five games"))
    }
    func testCompleteSeasonPairWinsOverDifferentRecentNumbers() throws {
        let primary = [profile("Fixture Home", offense: ["avgPoints": 110]), profile("Fixture Away", allowed: ["avgPoints": 108])]
        let recent = [profile("Fixture Home", offense: ["avgPoints": 95], sample: 5, source: "recent five games"), profile("Fixture Away", allowed: ["avgPoints": 98], sample: 5, source: "recent five games")]
        let results = MatchupInsightEngine.generate(matchup: matchup("NBA"), profiles: primary, props: [], recentProfiles: recent)
        let points = try XCTUnwrap(results.first(where: { $0.team == "Fixture Home" && $0.topic == "Scoring" }))
        XCTAssertTrue(points.hasPairedEvidence)
        XCTAssertTrue(points.text.contains("110.0"))
        XCTAssertTrue(points.text.contains("108.0"))
        XCTAssertFalse(points.text.contains("95.0"))
        XCTAssertFalse(points.evidence.contains("recent five games"))
    }

}
