import XCTest
@testable import PLONNCore

final class CurrentRosterSimulationTests: XCTestCase {
    func fixtures() throws -> [(PlayerSimulation.History, SimulationAvailability)] {
        let helper = PlayerSimulationTests(), directory = try helper.fixtureDirectory()
        let current = directory.appendingPathComponent("availability-current")
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as! [[String: Any]]
        return try manifest.map { item in
            let sport = InterestSport(rawValue: item["sport"] as! String)!, raw = item["team"] as! [String: Any], id = raw["id"] as! String
            let team = InterestTeam(sport: sport, providerID: id, name: raw["displayName"] as! String, abbreviation: raw["abbreviation"] as! String)
            let data = try Data(contentsOf: current.appendingPathComponent("\(sport.rawValue)-\(id)-roster.json"))
            let root = try JSONSerialization.jsonObject(with: data) as! [String: Any], now = sportsDate(root["timestamp"] as! String)!.addingTimeInterval(5)
            let availability = try SimulationAvailabilitySource.parse(data, injuries: Data(contentsOf: current.appendingPathComponent("\(sport.rawValue)-injuries.json")), team: team, now: now)
            let boxes = try (item["events"] as! [[String: Any]]).map { row in
                try helper.parseFixture(Data(contentsOf: directory.appendingPathComponent("\(sport.rawValue)-\(row["id"]!)-box.json")), sport: sport, teamID: id).0
            }
            return (.init(team: team, roster: availability.roster, boxes: boxes, partial: false), availability)
        }
    }
    func testFourSportsUseOnlyCurrentEligibleActorsAndReplayMatchesEveryStat() throws {
        let rows = try fixtures()
        for sport in InterestSport.allCases {
            let selected = rows.filter { $0.0.team.sport == sport }
            let teams = try selected.map { try SportPlaySimulation.team($0.0, availability: $0.1) }
            XCTAssertEqual(teams.count, 2)
            for seed in 1...15 {
                let game = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: UInt64(seed))
                let replay = game.through(game.plays.count)
                XCTAssertFalse(game.plays.isEmpty)
                XCTAssertEqual(replay.awayScore, game.awayScore); XCTAssertEqual(replay.homeScore, game.homeScore)
                XCTAssertEqual(replay.away, game.away); XCTAssertEqual(replay.home, game.home)
                let finalOnly = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: UInt64(seed), recordPlays: false)
                XCTAssertEqual(finalOnly.away, game.away); XCTAssertEqual(finalOnly.home, game.home)
                XCTAssertEqual(finalOnly.awayScore, game.awayScore); XCTAssertEqual(finalOnly.homeScore, game.homeScore)
                for play in game.plays {
                    let own = teams[play.home ? 1 : 0]
                    XCTAssertTrue(own.availability.eligible.contains { $0.id == play.actorID }, sport.rawValue + ": " + play.title)
                    for delta in play.deltas { XCTAssertTrue(teams[delta.home ? 1 : 0].availability.eligible.contains { $0.id == delta.playerID }) }
                }
                for (side, lines) in [game.away, game.home].enumerated() {
                    let score = side == 0 ? game.awayScore : game.homeScore, other = side == 0 ? game.homeScore : game.awayScore
                    let stat = sport == .nba ? "points" : sport == .nhl ? "goals" : "runs"
                    if sport != .nfl { XCTAssertEqual(lines.reduce(0) { $0 + ($1.values[stat] ?? 0) }, Double(score)) }
                    if sport == .nba {
                        XCTAssertEqual(lines.reduce(0) { $0 + ($1.values["minutes"] ?? 0) }, 240, accuracy: 0.000001)
                        for line in lines { XCTAssertEqual(line.values["points"], 2 * line.values["fieldGoalsMade"]! + line.values["threePointFieldGoalsMade"]! + line.values["freeThrowsMade"]!) }
                    }
                    if sport == .nhl {
                        let goalie = try XCTUnwrap(lines.first { $0.group == "goalies" })
                        XCTAssertEqual(goalie.values["goalsAgainst"], Double(other))
                        XCTAssertEqual(goalie.values["shotsAgainst"], goalie.values["saves"]! + goalie.values["goalsAgainst"]!)
                        XCTAssertTrue(lines.filter { $0.group == "skaters" }.allSatisfy { $0.values["penaltyMinutes"] == nil })
                    }
                    if sport == .mlb { XCTAssertEqual(lines.filter { $0.group == "pitching" }.reduce(0) { $0 + ($1.values["earnedRuns"] ?? 0) }, Double(other)) }
                    if sport == .nfl {
                        let passing = lines.filter { $0.group == "passing" }.reduce(0) { $0 + ($1.values["passingYards"] ?? 0) }
                        let receiving = lines.filter { $0.group == "receiving" }.reduce(0) { $0 + ($1.values["receivingYards"] ?? 0) }
                        XCTAssertEqual(passing, receiving)
                        XCTAssertEqual(lines.filter { $0.group == "passing" }.reduce(0) { $0 + ($1.values["completions"] ?? 0) }, lines.filter { $0.group == "receiving" }.reduce(0) { $0 + ($1.values["receptions"] ?? 0) })
                    }
                    XCTAssertTrue(lines.allSatisfy { line in teams[side].availability.eligible.contains { $0.id == line.playerID } })
                }
            }
        }
    }
    func testGeneratedColumnsMatchReplayAndAvailabilityIsRequired() throws {
        let rows = try fixtures()
        for sport in InterestSport.allCases {
            let selected = rows.filter { $0.0.team.sport == sport }, checked = selected.map { $0.1.checked }.max()!
            let result = try PlayerSimulation.currentRun(away: selected[0].0, home: selected[1].0, availability: selected.map { $0.1 }, now: checked, seed: 22, count: 5_000)
            for index in [0, 24, 4999] {
                let replay = try result.replay(index)
                XCTAssertEqual(result.scores.scores[index].away, replay.awayScore)
                XCTAssertEqual(result.scores.scores[index].home, replay.homeScore)
                for side in [false, true] {
                    let lines = side ? replay.home : replay.away, stored = result.box(homeSide: side, trial: index)
                    XCTAssertEqual(lines.map(\.id), stored.lines.map(\.id))
                    for line in lines { for (key, value) in line.values {
                        XCTAssertEqual(try XCTUnwrap(stored.lines.first { $0.id == line.id }?.values[key]), value, accuracy: 0.001)
                    } }
                }
            }
            let date = ISO8601DateFormatter(), start = checked.addingTimeInterval(86400), a = selected[0].0.team, h = selected[1].0.team
            let event = PlayerMatchupEvent(game: .init(league: sport.rawValue, eventID: "401999999", home: h.name, away: a.name, startsAt: date.string(from: start)), homeID: h.providerID, awayID: a.providerID, seasonType: 2)
            let market = sport == .nba ? "player_points" : sport == .nfl ? "player_pass_yds" : sport == .mlb ? "batter_hits" : "player_total_saves"
            var supportedComparisons = 0
            for homeSide in [false, true] {
                let side = homeSide ? 1 : 0
                let box = result.box(homeSide: homeSide, trial: 0)
                guard let player = result.current?.teams[side].profiles.first(where: { profile in
                    SimulationOdds.value(box: box, playerID: profile.id, market: market, sport: sport.rawValue) != nil
                }), let baseline = SimulationOdds.value(box: box, playerID: player.id, market: market, sport: sport.rawValue) else { continue }
                let appearances = selected[side].0.boxes.filter {
                    SimulationOdds.value(box: $0, playerID: player.id, market: market, sport: sport.rawValue) != nil
                }.count
                let line = baseline + 0.5
                let quote = Market(id: "current-prop", event: "odds-current", sport: sport.rawValue, kind: "prop", name: player.player.name, opponent: homeSide ? a.name : h.name, label: "Over", aliases: "", p: nil, price: -110, market: market, book: "Example book", quotes: [], updated: date.string(from: checked), mode: "live", line: line, eventLabel: a.name + " at " + h.name, eventState: "pre", commence: date.string(from: start), side: "Over")
                let comparisons = SimulationOdds.compare(result, event: event, markets: [quote], now: checked)
                let comparison = try XCTUnwrap(comparisons.first, sport.rawValue)
                XCTAssertEqual(comparison.playerAppearances, appearances)
                // Current starters can have fewer than five verified appearances.
                // Preserve that coverage limit instead of requiring a departed or
                // excluded player to satisfy the positive comparison fixture.
                if appearances < 5 {
                    XCTAssertEqual(comparison.unavailable, result.trials.count)
                    XCTAssertEqual(comparison.counted, 0)
                    XCTAssertNil(comparison.hitRate)
                    continue
                }
                supportedComparisons += 1
                let expected = result.trials.indices.reduce(0) { count, index in
                    count + ((SimulationOdds.value(box: result.box(homeSide: homeSide, trial: index), playerID: player.id, market: market, sport: sport.rawValue) ?? -.infinity) > line ? 1 : 0)
                }
                XCTAssertEqual(comparison.hit, expected); XCTAssertEqual(comparison.counted, result.trials.count)
            }
            XCTAssertGreaterThan(supportedComparisons, 0, "No eligible modeled prop on either team for " + sport.rawValue)
            XCTAssertThrowsError(try PlayerSimulation.currentRun(away: selected[0].0, home: selected[1].0, availability: selected.map { $0.1 }, now: checked.addingTimeInterval(121), seed: 1, count: 100))
        }
    }
    func testAvailabilityChangeFailedRefreshAndExpiredChecksBlockReplay() throws {
        let item = try fixtures()[0].1
        var watch = SimulationAvailabilityWatch(original: [item, item], latest: [item, item])
        XCTAssertTrue(watch.usable(at: item.checked))
        XCTAssertFalse(watch.usable(at: item.checked.addingTimeInterval(121)))
        watch.failed = true; XCTAssertFalse(watch.usable(at: item.checked))
        let changed = SimulationAvailability(roster: Array(item.roster.dropLast()), observations: item.observations, checked: item.checked, injuryFeedUpdated: item.injuryFeedUpdated, injuriesVerified: item.injuriesVerified)
        watch.update([changed, item]); XCTAssertTrue(watch.changed); XCTAssertFalse(watch.usable(at: item.checked))
        let injury = SimulationAvailability.Observation(playerID: item.roster[0].id, name: item.roster[0].name, status: "Out", updated: item.checked, excluded: true)
        let out = SimulationAvailability(roster: item.roster, observations: [injury], checked: item.checked, injuryFeedUpdated: item.checked, injuriesVerified: true)
        XCTAssertFalse(out.eligible.contains { $0.id == injury.playerID })
        watch.update([out, item]); XCTAssertTrue(watch.changed)
        XCTAssertTrue(SimulationAvailability.excluded("Practice Squad")); XCTAssertTrue(SimulationAvailability.excluded("Injured/60")); XCTAssertTrue(SimulationAvailability.excluded("Out"))
        XCTAssertFalse(SimulationAvailability.excluded("Questionable")); XCTAssertFalse(SimulationAvailability.excluded("Day-To-Day"))
    }
    func testRosterIdentityAndFreshnessCannotBeFakedByFetchTime() throws {
        let helper = PlayerSimulationTests(), directory = try helper.fixtureDirectory().appendingPathComponent("availability-current")
        let data = try Data(contentsOf: directory.appendingPathComponent("NBA-2-roster.json"))
        let root = try JSONSerialization.jsonObject(with: data) as! [String: Any], date = sportsDate(root["timestamp"] as! String)!
        let team = InterestTeam(sport: .nba, providerID: "2", name: "Boston Celtics", abbreviation: "BOS")
        let unknown = try SimulationAvailabilitySource.parse(data, injuries: nil, team: team, now: date)
        XCTAssertFalse(unknown.injuriesVerified)
        XCTAssertThrowsError(try SimulationAvailabilitySource.parse(data, injuries: nil, team: team, now: date.addingTimeInterval(901)))
        XCTAssertThrowsError(try SimulationAvailabilitySource.parse(data, injuries: nil, team: .init(sport: .nba, providerID: "1", name: "Atlanta Hawks", abbreviation: "ATL"), now: date))
    }
}
