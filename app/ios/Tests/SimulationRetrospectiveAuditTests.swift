import XCTest
@testable import PLONNCore

/// Opt-in offline audit against captured responses. Never performs network reads.
final class SimulationRetrospectiveAuditTests: XCTestCase {
    func testCapturedRetrospective() async throws {
        guard let path = ProcessInfo.processInfo.environment["PLONN_RETROSPECTIVE_DIRECTORY"] else {
            throw XCTSkip("Opt-in retrospective audit requires captured inputs")
        }
        let root = URL(fileURLWithPath: path)
        let output = URL(fileURLWithPath: ProcessInfo.processInfo.environment["PLONN_RETROSPECTIVE_OUTPUT"] ?? path)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let index = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: root.appendingPathComponent("response-index.json")))
        let rows = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("audit-games.json"))) as! [[String: Any]]
        let reader: MatchupSimulationSource.Reader = { url in
            guard let name = index[url.absoluteString] else { throw MatchupSimulation.Failure.source }
            return try Data(contentsOf: root.appendingPathComponent("responses/" + name))
        }
        let started = Date()
        var report: [[String: Any]] = []
        for row in rows {
            let id = row["id"] as! String, sport = InterestSport(rawValue: row["sport"] as! String)!
            let cutoff = try XCTUnwrap(sportsDate(row["start"] as! String))
            let seed = UInt64(id)! &* 0x517cc1b727220a95 ^ 20_261_024
            var record: [String: Any] = ["event": id, "sport": sport.rawValue, "cutoff": row["start"]!, "seed": String(seed), "run_type": "retrospective_current_availability", "generated_at": ISO8601DateFormatter().string(from: started)]
            do {
                var histories: [PlayerSimulation.History] = [], availability: [SimulationAvailability] = []
                for side in ["away", "home"] {
                    let raw = row[side] as! [String: Any]
                    let team = InterestTeam(sport: sport, providerID: raw["id"] as! String, name: raw["displayName"] as! String, abbreviation: raw["abbreviation"] as! String)
                    record[side] = team.abbreviation
                    let base = "https://site.api.espn.com/apis/site/v2/sports/" + sport.leaguePath
                    let roster = try await reader(URL(string: base + "/teams/\(team.providerID)/roster")!)
                    let body = try JSONSerialization.jsonObject(with: roster) as! [String: Any]
                    let checked = try XCTUnwrap((body["timestamp"] as? String).flatMap(sportsDate))
                    let injuries = try? await reader(URL(string: base + "/injuries")!)
                    let snapshot = try SimulationAvailabilitySource.parse(roster, injuries: injuries, team: team, now: checked)
                    availability.append(snapshot)
                    let games = try await MatchupSimulationSource(read: reader).history(team, cutoff: cutoff)
                    let history = try await PlayerSimulationSource(read: reader).history(games, cutoff: cutoff)
                    XCTAssertFalse(history.boxes.contains { $0.id == id })
                    XCTAssertTrue(history.boxes.allSatisfy { $0.game.date < cutoff })
                    histories.append(history)
                    record[side + "_history"] = history.boxes.map { $0.id }
                    record[side + "_availability_as_of"] = ISO8601DateFormatter().string(from: checked)
                    record[side + "_injuries_verified"] = snapshot.injuriesVerified
                    record[side + "_excluded"] = snapshot.observations.filter(\.excluded).map(\.name)
                }
                let simulationNow = availability.map(\.checked).max()!
                let result = try PlayerSimulation.currentRun(away: histories[0], home: histories[1], availability: availability, now: simulationNow, seed: seed)
                let scores = result.scores.scores
                // The target result is only consulted after the outcomes are generated.
                let actual = row["actual"] as! [Int]
                let exact = scores.filter { $0.away == actual[0] && $0.home == actual[1] }.count
                record["actual"] = actual
                record["runs"] = scores.count
                record["exact_matches"] = exact
                record["exact_frequency"] = Double(exact) / Double(scores.count)
                record["mean"] = [result.scores.awaySummary.average, result.scores.homeSummary.average]
                record["range80_away"] = [result.scores.awaySummary.low, result.scores.awaySummary.high]
                record["range80_home"] = [result.scores.homeSummary.low, result.scores.homeSummary.high]
                record["away_ahead"] = result.scores.awayAhead
                record["home_ahead"] = result.scores.homeAhead
                record["level"] = result.scores.level
                record["unique_scores"] = Set(scores.map { "\($0.away):\($0.home)" }).count
                record["regulation_only"] = true
                record["observed_periods"] = row["period"]
                let saved: [String: Any] = ["record": record, "scores": scores.map { [$0.away, $0.home] }]
                try JSONSerialization.data(withJSONObject: saved, options: [.sortedKeys]).write(to: output.appendingPathComponent("\(sport.rawValue)-\(id)-outcomes.json"), options: .atomic)
                print("AUDIT \(sport.rawValue) \(record["away"]!) \(record["home"]!) actual=\(actual) exact=\(exact)/\(scores.count)")
            } catch {
                record["error"] = String(describing: error)
                print("AUDIT FAILED \(sport.rawValue) \(id): \(error)")
            }
            report.append(record)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("results.json"), options: .atomic)
        }
        XCTAssertEqual(report.count, rows.count)
    }
}
