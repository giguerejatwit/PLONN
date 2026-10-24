import XCTest
@testable import PLONNCore

final class SimulationMarketDiagnosticTests: XCTestCase {
    func testCapturedCurrentMatchups() async throws {
        guard let path = ProcessInfo.processInfo.environment["PLONN_MARKET_DIAGNOSTIC"] else { throw XCTSkip("Opt-in current matchup diagnostic") }
        let root = URL(fileURLWithPath: path)
        let index = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: root.appendingPathComponent("response-index.json")))
        let rows = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("games.json"))) as! [[String: Any]]
        let reader: MatchupSimulationSource.Reader = { url in
            guard let file = index[url.absoluteString] else { throw MatchupSimulation.Failure.source }
            return try Data(contentsOf: root.appendingPathComponent("responses/" + file))
        }
        var report: [[String: Any]] = []
        for row in rows {
            var histories: [PlayerSimulation.History] = [], availability: [SimulationAvailability] = []
            var record: [String: Any] = ["event": row["id"]!, "start": row["start"]!]
            for side in ["away", "home"] {
                let raw = row[side] as! [String: Any]
                let team = InterestTeam(sport: .nba, providerID: raw["id"] as! String, name: raw["displayName"] as! String, abbreviation: raw["abbreviation"] as! String)
                let base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
                let roster = try await reader(URL(string: base + "teams/\(team.providerID)/roster")!)
                let body = try JSONSerialization.jsonObject(with: roster) as! [String: Any]
                let now = try XCTUnwrap((body["timestamp"] as? String).flatMap(sportsDate))
                let injuries = try await reader(URL(string: base + "injuries")!)
                let snapshot = try SimulationAvailabilitySource.parse(roster, injuries: injuries, team: team, now: now)
                let games = try await MatchupSimulationSource(read: reader).history(team, cutoff: now)
                let history = try await PlayerSimulationSource(read: reader).history(games, cutoff: now)
                let profile = try SportPlaySimulation.team(history, availability: snapshot)
                histories.append(history); availability.append(snapshot)
                record[side] = team.abbreviation
                record[side + "_excluded"] = snapshot.observations.filter(\.excluded).map { ["id": $0.id, "name": $0.name, "status": $0.status] }
                record[side + "_missing_history"] = profile.missingHistory.map(\.name)
                record[side + "_history"] = history.boxes.map { ["event": $0.id, "date": ISO8601DateFormatter().string(from: $0.game.date), "scored": $0.game.scored, "allowed": $0.game.allowed] as [String: Any] }
                record[side + "_profiles"] = profile.profiles.map { ["id": $0.id, "name": $0.player.name, "means": $0.means, "samples": $0.samples] as [String: Any] }
                record[side + "_possessions"] = profile.possessions
                if team.providerID == "6" {
                    let flagg = try XCTUnwrap(snapshot.roster.first { $0.name == "Cooper Flagg" })
                    XCTAssertTrue(snapshot.excludedIDs.contains(flagg.id))
                    XCTAssertFalse(profile.profiles.contains { $0.id == flagg.id })
                }
            }
            let seed = UInt64(row["id"] as! String)!
            let quoteURL = root.appendingPathComponent("quotes.json")
            let quotes = FileManager.default.fileExists(atPath: quoteURL.path) ? try JSONDecoder().decode([String:[Market]].self, from: Data(contentsOf: quoteURL))[row["id"] as! String] ?? [] : []
            let now = max(availability.map(\.checked).max()!, quotes.compactMap { $0.updated.flatMap(sportsDate) }.max() ?? .distantPast)
            let event = PlayerMatchupEvent(game: .init(league: "NBA", eventID: row["id"] as! String, home: histories[1].team.name, away: histories[0].team.name, startsAt: row["start"] as! String), homeID: histories[1].team.providerID, awayID: histories[0].team.providerID, seasonType: 2)
            let market = SimulationForecast.market(event: event, markets: quotes, now: now)
            if !quotes.isEmpty { XCTAssertNotNil(market) }
            let result = try PlayerSimulation.currentRun(away: histories[0], home: histories[1], availability: availability, now: now, seed: seed, market: market)
            if let f = result.current?.forecast {
                record["targets"] = f.targets; record["effective_paths"] = f.effectivePaths; record["support_limited"] = f.limited
                record["model_version"] = f.version; record["market_used"] = f.market != nil
                record["market_home_probability"] = f.market?.homeProbability
                record["market_quote_ids"] = f.market?.quoteIDs
            }
            for i in [0,2499,4999] {
                let replay = try result.replay(i)
                XCTAssertEqual(replay.homeScore, result.scores.scores[i].home)
                XCTAssertEqual(replay.awayScore, result.scores.scores[i].away)
            }
            record["mean"] = [result.scores.awaySummary.average, result.scores.homeSummary.average]
            record["ahead_away_home_level"] = [result.scores.awayAhead, result.scores.homeAhead, result.scores.level]
            record["seed"] = String(seed)
            record["runs"] = 5000
            report.append(record)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("diagnostic.json"), options: .atomic)
            print("DIAGNOSTIC \(record["away"]!)–\(record["home"]!) \(record["mean"]!) \(record["ahead_away_home_level"]!)")
        }
    }
}
