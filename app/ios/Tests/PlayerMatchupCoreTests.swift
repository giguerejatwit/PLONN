import XCTest
@testable import PLONNCore

final class PlayerMatchupCoreTests: XCTestCase {
    func testHistorySchemaFailureExplainsTheProblemWithoutBlamingTheSelection() {
        for sport in ["nfl", "nba", "nhl", "mlb"] {
            XCTAssertEqual(PlayerMatchupSetup.failure(["reason": sport + "_source_schema_mismatch"]),
                           "We couldn’t read this player’s game history. Try again shortly.")
        }
    }

    let now = sportsDate("2026-10-19T12:00:00Z")!
    func setup(_ league: String = "NFL", stat: String = "Rush Yds") -> PlayerMatchupSetup {
        .init(league: league, eventID: "123", home: "Home Team", away: "Away Team", startsAt: "2026-10-20T19:30:00Z", player: "Test Player", playerID: "77", teamID: "1", stat: stat, line: 59.5, side: "Over", book: "User-entered line", propOptions: [:])
    }
    func result(_ setup: PlayerMatchupSetup, available: Bool = true) throws -> [String: Any] {
        let projection: [String: Any] = available
            ? ["available":true,"distribution":"empirical","mean":60.0,"lower":40.0,"upper":80.0,"interval_level":0.8,"standard_deviation":18.0,"probabilities":["over":0.55,"under":0.45,"push":0.0],"selected_probability":0.55,"opponent_adjustment":2.0]
            : ["available":false,"distribution":"empirical","interval_level":0.8]
        return ["status":"ok","evaluation_mode":PlayerMatchupSetup.mode,"upcoming_prediction":true,"setup":try setup.payload(),"generated_at":"2026-10-19T11:00:00Z",
                "inference":["target":["event_id":setup.eventID,"league":setup.league,"date":"2026-10-20","starts_at":setup.startsAt,"player_id":setup.playerID,"player_name":setup.player,"team_id":setup.teamID,"team":setup.home,"opponent_id":"2","opponent":setup.away,"is_home":true],"projection":projection,"factors":[["kind":"workload","title":"Recent carries","detail":"Recorded workload is available.","direction":"neutral","sample_size":10]],"missing_inputs":["Historical lineup snapshots"]],
                "validation":["status":available ? "passed":"limited","training_count":30,"test_count":12,"brier":0.2,"baseline_brier":0.24,"calibration_error":0.05,"interval_coverage":0.8,"gates":["sample":available],"reasons":available ? []:["Insufficient historical evidence"]],
                "coverage":["usable_rows":42,"excluded_rows":1,"opponent_games":12,"workload_games":40],
                "availability":["version":1,"event_id":setup.eventID,"state":"limited","probability_adjusted":false,"historical_training_available":false,"checked_at":"2026-10-19T11:00:00Z","expires_at":"2026-10-19T11:30:00Z","injuries":[setup.home,setup.away].map{["team":$0,"state":"missing","players":[],"excluded":0] as [String:Any]},"lineups":[setup.home,setup.away].map{["team":$0,"state":"unknown","players":[],"detail":"Not confirmed"] as [String:Any]}]]
    }
    func change(_ body: [String: Any], projection changes: [String: Any]) -> [String: Any] {
        var body = body, inference = body["inference"] as! [String:Any], projection = (body["inference"] as! [String:Any])["projection"] as! [String:Any]
        for (key,value) in changes { projection[key] = value }
        inference["projection"] = projection; body["inference"] = inference; return body
    }
    func testEveryCatalogPropHasStrictFutureSetupAndNoTrainingGameChoice() throws {
        var count = 0
        for league in LabPropCatalog.leagues {
            for stat in LabPropCatalog.options(league:league,target:"Player") where stat != "Custom stat" {
                var request = setup(league,stat:stat)
                if LabPropCatalog.isYesNo(stat) { request.line = 0.5 }
                if LabPropCatalog.isFantasy(stat) { request.propOptions = ["scoring_preset":"draftkings_classic"] }
                if stat == "Method of first basket" { request.propOptions = ["first_basket_method":"two_pointer"] }
                let payload = try request.payload()
                XCTAssertNil(payload["historical_event_id"]); XCTAssertNil(payload["american_odds"])
                XCTAssertEqual(Set(payload.keys).count,14)
                XCTAssertEqual(try JSONDecoder().decode(PlayerMatchupSetup.self,from:JSONEncoder().encode(request)),request)
                count += 1
            }
        }
        XCTAssertGreaterThan(count,80)
    }
    func testRejectsWrongSportInvalidIDsUnknownPropsAndScoringRules() {
        var request = setup(); request.playerID = "mlb:123"; XCTAssertThrowsError(try request.payload())
        request = setup(); request.playerID = "0"; XCTAssertThrowsError(try request.payload())
        request = setup(); request.league = "Soccer"; XCTAssertThrowsError(try request.payload())
        request = setup(); request.stat = "Custom stat"; XCTAssertThrowsError(try request.payload())
        request = setup(); request.line = .infinity; XCTAssertThrowsError(try request.payload())
        request = setup(); request.line = 1.234; XCTAssertThrowsError(try request.payload())
        request = setup("NBA",stat:"Fantasy points"); XCTAssertThrowsError(try request.payload())
        request.propOptions = ["scoring_preset":"made_up"]; XCTAssertThrowsError(try request.payload())
        request = setup("NFL",stat:"Anytime touchdown scorer"); XCTAssertThrowsError(try request.payload())
    }
    func testValidProjectionAndLimitedEvidenceAreDistinct() throws {
        let request = setup(), forecast = try request.forecast(result(request),now:now)
        XCTAssertEqual(forecast.inference.projection.standardDeviation,18)
        XCTAssertEqual(forecast.inference.projection.probabilities?.over,0.55)
        let limited = try request.forecast(result(request,available:false),now:now)
        XCTAssertFalse(limited.inference.projection.available)
        XCTAssertNil(limited.inference.projection.mean)
        XCTAssertThrowsError(try request.forecast(change(result(request,available:false),projection:["mean":60]),now:now))
    }
    func testMismatchedSubjectLineOptionAndFutureGenerationNeverDisplay() throws {
        let request = setup(), good = try result(request)
        for field in ["player_id","team_id","event_id","stat","line","side","prop_options"] {
            var bad = good, payload = good["setup"] as! [String:Any]
            payload[field] = field == "line" ? 20.5 : field == "prop_options" ? ["unknown":"value"] : "wrong"
            bad["setup"] = payload; XCTAssertThrowsError(try request.forecast(bad,now:now),field)
        }
        var bad = good; bad["generated_at"] = "2026-10-20T20:00:00Z"
        XCTAssertThrowsError(try request.forecast(bad,now:now))
        bad = good; var inference = bad["inference"] as! [String:Any], target = inference["target"] as! [String:Any]
        target["opponent_id"] = request.teamID; inference["target"] = target; bad["inference"] = inference
        XCTAssertThrowsError(try request.forecast(bad,now:now))
    }
    func testRejectsInvalidProbabilitiesRangesGatesAndInsufficientValidation() throws {
        let request = setup(), good = try result(request)
        for changed: [String:Any] in [["probabilities":["over":0.6,"under":0.6,"push":0.0]], ["selected_probability":0.9], ["lower":100,"upper":20], ["standard_deviation":-2], ["interval_level":0.95], ["mean":true]] {
            XCTAssertThrowsError(try request.forecast(change(good,projection:changed),now:now))
        }
        for changed: [String:Any] in [["status":"limited"],["gates":["sample":false]],["test_count":9],["training_count":19]] {
            var bad = good, validation = good["validation"] as! [String:Any]
            for (key,value) in changed { validation[key] = value }; bad["validation"] = validation
            XCTAssertThrowsError(try request.forecast(bad,now:now))
        }
    }
    func testSavedSnapshotKeepsOriginalDateAndDoesNotBecomeLiveForecast() throws {
        let request = setup(), snapshot = try request.forecast(result(request),now:now.addingTimeInterval(7*86400))
        XCTAssertEqual(snapshot.generatedAt,sportsDate("2026-10-19T11:00:00Z"))
        XCTAssertLessThan(request.start!,now.addingTimeInterval(7*86400))
    }
    func testExistingAnalyticsMarketsMapToCanonicalProps() {
        let examples = ["NFL":["player_rush_yds":"Rush Yds","player_pass_yds_alternate":"Pass Yds"],"NBA":["player_points":"Points","player_points_rebounds_assists":"Pts + Reb + Ast"],"MLB":["pitcher_strikeouts":"Pitcher strikeouts","batter_total_bases":"Total bases"],"NHL":["player_total_saves":"Saves","player_shots_on_goal":"Shots on goal"]]
        for (league,markets) in examples { for (key,value) in markets { XCTAssertEqual(PlayerMatchupSetup.stat(sport:league,marketKey:key),value) } }
        XCTAssertNil(PlayerMatchupSetup.stat(sport:"NFL",marketKey:"unrecognized"))
    }
    func testCapturedBackendResponseUsesTheSameContract() throws {
        guard let paths = ProcessInfo.processInfo.environment["PLONN_MATCHUP_RESPONSE_FIXTURES"] else {
            throw XCTSkip("Set PLONN_MATCHUP_RESPONSE_FIXTURES to captured backend JSON paths for integration verification.")
        }
        for path in paths.split(separator: ":") {
            let data = try Data(contentsOf: URL(fileURLWithPath: String(path)))
            let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String:Any])
            let setupData = try JSONSerialization.data(withJSONObject: XCTUnwrap(body["setup"]))
            let request = try JSONDecoder().decode(PlayerMatchupSetup.self, from: setupData)
            let date = try XCTUnwrap((body["generated_at"] as? String).flatMap(sportsDate))
            let forecast = try request.forecast(body, now:date.addingTimeInterval(5))
            XCTAssertEqual(forecast.setup,request)
        }
    }
}
