import XCTest
@testable import PLONNCore

/// Opt-in, cached-only chronological reconstruction. Target scores, current
/// rosters, and current injuries are never read by this prediction stage.
final class SimulationHistoricalBacktestTests: XCTestCase {
    func testPastOnlyHistoricalReconstruction() async throws {
        guard let path = ProcessInfo.processInfo.environment["PLONN_HISTORICAL_BACKTEST"] else {
            throw XCTSkip("Requires a captured historical backtest dataset")
        }
        let root = URL(fileURLWithPath: path)
        let index = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: root.appendingPathComponent("response-index.json")))
        let rows = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("targets.json"))) as! [[String: Any]]
        let output = root.appendingPathComponent("predictions")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var report: [[String: Any]] = []
        for row in rows {
            let id = row["id"] as! String, sport = InterestSport(rawValue: row["sport"] as! String)!
            let cutoff = try XCTUnwrap(sportsDate(row["start"] as! String))
            let seed = UInt64(id)! &* 0x517cc1b727220a95 ^ 20_261_024
            var record: [String: Any] = ["event": id, "sport": sport.rawValue, "cutoff": row["start"]!, "seed": String(seed), "run_type": "past_only_roster_proxy", "injuries_verified": false]
            let reader: MatchupSimulationSource.Reader = { url in
                // Enforce the isolation even if a future loader implementation changes.
                guard !url.path.hasSuffix("/roster"), !url.path.hasSuffix("/injuries"),
                      URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "event" })?.value != id,
                      let name = index[url.absoluteString] else { throw MatchupSimulation.Failure.source }
                return try Data(contentsOf: root.appendingPathComponent("responses/" + name))
            }
            do {
                var histories: [PlayerSimulation.History] = [], availability: [SimulationAvailability] = []
                for side in ["away", "home"] {
                    let raw = row[side] as! [String: Any]
                    let team = InterestTeam(sport: sport, providerID: raw["id"] as! String, name: raw["displayName"] as! String, abbreviation: raw["abbreviation"] as! String)
                    record[side] = team.abbreviation
                    let games = try await MatchupSimulationSource(read: reader).history(team, cutoff: cutoff)
                    var boxes: [PlayerSimulation.Box] = [], failed: [[String: String]] = []
                    for game in games.games.prefix(10) {
                        do {
                            let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(sport.leaguePath)/summary?event=\(game.id)")!
                            boxes.append(try PlayerSimulationSource.parse(await reader(url), team: team, game: game, cutoff: cutoff))
                        } catch { failed.append(["event": game.id, "error": String(describing: error)]) }
                    }
                    record[side + "_rejected_boxes"] = failed
                    record[side + "_history"] = boxes.map { ["id": $0.id, "start": ISO8601DateFormatter().string(from: $0.game.date)] }
                    guard boxes.count >= 5 else { throw MatchupSimulation.Failure.history }
                    XCTAssertTrue(boxes.allSatisfy { $0.game.date < cutoff && $0.id != id })
                    // A documented pregame-only participation proxy, not a claimed
                    // historical active roster. Never consult the target box score.
                    var players: [String: PlayerMatchupPlayer] = [:]
                    for box in boxes.prefix(3) {
                        for line in box.lines where players[line.playerID] == nil {
                            players[line.playerID] = .init(id: line.playerID, name: line.name, teamID: team.providerID, team: team.name, position: "")
                        }
                    }
                    let roster = players.values.sorted { $0.id < $1.id }
                    record[side + "_proxy_roster"] = roster.map { ["id": $0.id, "name": $0.name] }
                    record[side + "_proxy_roster_games"] = boxes.prefix(3).map(\.id)
                    let history = PlayerSimulation.History(team: team, roster: roster, boxes: boxes, partial: games.partial || !failed.isEmpty)
                    histories.append(history)
                    availability.append(.init(roster: roster, observations: [], checked: cutoff, injuryFeedUpdated: nil, injuriesVerified: false))
                }
                let result = try PlayerSimulation.currentRun(away: histories[0], home: histories[1], availability: availability, now: cutoff, seed: seed, count: 5000)
                let base = histories.map { MatchupSimulation.History(team: $0.team, games: $0.boxes.map(\.game), partial: $0.partial) }
                let baseline = try MatchupSimulation.run(away: base[0], home: base[1], now: cutoff, seed: seed, count: 5000)
                XCTAssertEqual(result.scores.scores.count, 5000)
                XCTAssertTrue(result.scores.scores.allSatisfy { $0.away >= 0 && $0.home >= 0 })
                record["runs"] = result.scores.scores.count
                if let forecast = result.current?.forecast {
                    record["model_version"] = forecast.version
                    record["score_targets"] = forecast.targets
                    record["effective_games"] = forecast.effectiveGames
                    record["effective_paths"] = forecast.effectivePaths
                    record["support_limited"] = forecast.limited
                    record["market_used"] = forecast.market != nil
                }
                record["mean"] = [result.scores.awaySummary.average, result.scores.homeSummary.average]
                let saved: [String: Any] = ["record": record, "scores": result.scores.scores.map { [$0.away, $0.home] }, "baseline_scores": baseline.scores.map { [$0.away, $0.home] }]
                try JSONSerialization.data(withJSONObject: saved, options: [.sortedKeys]).write(to: output.appendingPathComponent("\(sport.rawValue)-\(id).json"), options: .atomic)
                print("BACKTEST \(sport.rawValue) \(record["away"]!)–\(record["home"]!) \(row["start"]!) complete \(report.count + 1)/\(rows.count)")
            } catch {
                record["error"] = String(describing: error)
                print("BACKTEST UNAVAILABLE \(sport.rawValue) \(id): \(error)")
            }
            report.append(record)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("prediction-index.json"), options: .atomic)
        }
        XCTAssertEqual(report.count, rows.count)
        for sport in InterestSport.allCases where rows.contains(where: { $0["sport"] as? String == sport.rawValue }) {
            XCTAssertTrue(report.contains { $0["sport"] as? String == sport.rawValue && $0["error"] == nil }, "No successful simulations for \(sport.rawValue)")
        }
    }
}
