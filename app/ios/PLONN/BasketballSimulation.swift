import Foundation

/// Bundled, retrospective simulation data. Observed results never enter replay state.
struct BasketballSimulation: Decodable {
    /// Keeps source extensions, including nullable nested audit metadata, intact.
    enum JSONValue: Decodable, Equatable {
        case null, bool(Bool), number(Double), string(String), array([JSONValue]), object([String: JSONValue])
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() { self = .null }
            else if let v = try? c.decode(Bool.self) { self = .bool(v) }
            else if let v = try? c.decode(Double.self) { self = .number(v) }
            else if let v = try? c.decode(String.self) { self = .string(v) }
            else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
            else { self = .object(try c.decode([String: JSONValue].self)) }
        }
    }
    struct Metadata: Decodable {
        struct Training: Decodable { let season: String; let games: Int }
        struct OpeningNight: Decodable {
            let targetDateET, currentSeason, lastPlayerObservationET, residualCalibrationStatus: String
            let currentSeasonGamesUsed, residualCalibrationGames: Int
            let retrospectiveKnownAvailability: Bool
        }
        struct Backtest: Decodable { let status, label: String }
        let gameId, date, sourceUrl, modelVersion, rosterPolicy: String
        let actualScore: [Int]
        let zoneOrder: [String]
        let training: Training
        let limitations: [String]
        let openingNight: OpeningNight
        let headToHead: HeadToHead?
        let backtest: Backtest
        let provenance: [String: JSONValue]
        private enum CodingKeys: String, CodingKey {
            case gameId, date, sourceUrl, modelVersion, rosterPolicy, actualScore, zoneOrder, training, limitations, openingNight, headToHead, backtest
        }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            gameId = try c.decode(String.self, forKey: .gameId)
            date = try c.decode(String.self, forKey: .date)
            sourceUrl = try c.decode(String.self, forKey: .sourceUrl)
            modelVersion = try c.decode(String.self, forKey: .modelVersion)
            rosterPolicy = try c.decode(String.self, forKey: .rosterPolicy)
            actualScore = try c.decode([Int].self, forKey: .actualScore)
            zoneOrder = try c.decode([String].self, forKey: .zoneOrder)
            training = try c.decode(Training.self, forKey: .training)
            limitations = try c.decode([String].self, forKey: .limitations)
            openingNight = try c.decode(OpeningNight.self, forKey: .openingNight)
            headToHead = try c.decodeIfPresent(HeadToHead.self, forKey: .headToHead)
            backtest = try c.decode(Backtest.self, forKey: .backtest)
            provenance = try [String: JSONValue](from: decoder)
        }
    }
    struct HeadToHead: Decodable {
        struct Meeting: Decodable, Identifiable {
            let game_id, date_et, away_id, home_id: String
            let actual_away, actual_home: Int
            let usable: Bool
            var id: String { game_id }
        }
        let source_meeting_count, usable_meeting_count: Int
        let effective_meeting_count, half_life_days, shrinkage_games, maximum_absolute_points_per_team: Double
        let first_meeting_date_et, last_meeting_date_et: String
        let adjustment_by_team: [String: Double]
        let meetings: [Meeting]
    }
    struct Player: Decodable, Identifiable {
        let id, name, jersey, position: String
        let starter: Bool
        let historicalGames: Int
        let historicalMinutes, historicalFGA, projectedMinutes: Double
        let zones, accuracy: [Double]
        let adaptiveHistory: [String: JSONValue]
        var shortName: String {
            let parts = name.split(separator: " ")
            return parts.count > 1 ? String(parts[0].prefix(1)) + ". " + parts.dropFirst().joined(separator: " ") : name
        }
    }
    struct Team: Decodable, Identifiable {
        struct Parameters: Decodable { let defenseFactors: [Double] }
        let id, name, abbrev, color: String
        let players: [Player]
        let parameters: Parameters
    }
    struct Rosters: Decodable { let teams: [Team] }
    struct Ensemble: Decodable {
        let runs, seed: Int
        let mean_score: [Double]
        let win_probability: Double
        let total_p10, total_p90: Double
        let score_pairs: [[Int]]
        let structural_mean_score: [Double]
        let total_mean_mcse: Double
    }
    struct Event: Decodable {
        let period, clock, duration, team, points: Int
        let player, kind: String
        let zone: String?
        let made: Bool
        let assist, rebound: String?
        let offensive_rebound: Bool
        let score_before, score: [Int]
        let lineups: [[String]]
        let free_throws: Int?
        var hasIllustratedShotLocation: Bool { kind == "shot" && zone != "unlocated_two" }
        var periodLabel: String { period <= 4 ? "Q\(period)" : "OT\(period - 4)" }
    }
    struct FinalBox: Decodable {
        let name: String
        let pts, fga, fgm, tpa, tpm, fta, ftm, reb, ast, tov: Int
        let minutes: Double
        var counters: BoxLine {
            BoxLine(points: pts, rebounds: reb, assists: ast, turnovers: tov, attempts: fga, made: fgm,
                    threesAttempted: tpa, threesMade: tpm, freeThrowsAttempted: fta, freeThrowsMade: ftm)
        }
    }
    struct Replay: Decodable {
        let label: String
        let seed: Int
        let score: [Int]
        let events: [Event]
        let box: [String: FinalBox]

        func frame(at position: Double) -> Frame {
            let bounded = position.isFinite ? min(Double(events.count), max(0, position)) : 0
            let completed = Int(bounded)
            let index = min(completed, events.count - 1)
            let event = events[index]
            let finished = completed == events.count
            let phase = finished ? 1 : bounded - Double(index)
            return Frame(index: index, phase: phase, completed: completed,
                score: completed > 0 ? events[completed - 1].score : [0, 0],
                seconds: finished ? 0 : Double(event.clock) + Double(event.duration) * (1 - phase),
                period: event.period, finished: finished)
        }

        /// Score, minutes, counters and feed all use this same completed-event boundary.
        func counts(completed: Int) -> [String: BoxLine] {
            var result: [String: BoxLine] = [:]
            for event in events.prefix(max(0, min(completed, events.count))) {
                for id in event.lineups.flatMap({ $0 }) { result[id, default: BoxLine()].seconds += event.duration }
                var line = result[event.player, default: BoxLine()]
                line.points += event.points
                if event.kind == "shot" {
                    line.attempts += 1; line.made += event.made ? 1 : 0
                    if event.zone == "three" { line.threesAttempted += 1; line.threesMade += event.made ? 1 : 0 }
                } else if event.kind == "free throws" {
                    line.freeThrowsAttempted += 2; line.freeThrowsMade += event.points
                } else if event.kind == "turnover" { line.turnovers += 1 }
                result[event.player] = line
                if let id = event.assist { result[id, default: BoxLine()].assists += 1 }
                if let id = event.rebound { result[id, default: BoxLine()].rebounds += 1 }
            }
            return result
        }
    }
    struct Frame {
        let index: Int
        let phase: Double
        let completed: Int
        let score: [Int]
        let seconds: Double
        let period: Int
        let finished: Bool
        var clock: String {
            if finished { return "Final" }
            let value = Int(ceil(seconds))
            return String(format: "%d:%02d", value / 60, value % 60)
        }
        var periodLabel: String { period <= 4 ? "Q\(period)" : "OT\(period - 4)" }
    }
    struct BoxLine: Equatable {
        var points = 0, rebounds = 0, assists = 0, turnovers = 0
        var attempts = 0, made = 0, threesAttempted = 0, threesMade = 0, freeThrowsAttempted = 0, freeThrowsMade = 0
        var seconds = 0
        var minutes: Double { Double(seconds) / 60 }
    }
    let meta: Metadata
    let data: Rosters
    let result: Ensemble
    let replays: [Replay]
    var players: [String: Player] { Dictionary(uniqueKeysWithValues: data.teams.flatMap(\.players).map { ($0.id, $0) }) }
    var representativeScenario: MatchupSimulation.RepresentativeScenario? {
        MatchupSimulation.representative(in: result.score_pairs.map { .init(away: $0[0], home: $0[1]) },
            candidates: replays.map { .init(away: $0.score[0], home: $0.score[1]) })
    }
    static let zoneOrder = ["rim", "mid", "three", "unlocated_two"]
    static let zoneLabels = ["Rim", "Mid-range", "Three", "Unlocated 2"]

    enum LoadError: Error { case invalidReplay }
    /// Local adapter only. A future remote loader must satisfy the same validation.
    static func bundled() throws -> Self {
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        #else
        let bundle = Bundle.main
        #endif
        guard let url = bundle.url(forResource: "CelticsPistonsSimulation", withExtension: "json") else { throw LoadError.invalidReplay }
        return try decode(Data(contentsOf: url))
    }
    static func decode(_ bytes: Data) throws -> Self {
        guard bytes.count <= 5_000_000 else { throw LoadError.invalidReplay }
        let value = try JSONDecoder().decode(Self.self, from: bytes)
        guard value.data.teams.count == 2, Set(value.data.teams.map(\.id)).count == 2,
              !value.replays.isEmpty, value.replays.count <= 100,
              value.result.runs > 0, value.result.score_pairs.count == value.result.runs,
              value.result.mean_score.count == 2, value.result.structural_mean_score.count == 2,
              value.result.mean_score.allSatisfy({ $0.isFinite && $0 >= 0 }),
              value.result.total_p10 >= 0, value.result.total_p90 >= value.result.total_p10,
              value.meta.actualScore.count == 2, value.meta.actualScore.allSatisfy({ $0 >= 0 }),
              value.meta.zoneOrder == zoneOrder, !value.meta.modelVersion.isEmpty,
              (0...1).contains(value.result.win_probability),
              value.result.score_pairs.allSatisfy({ $0.count == 2 && $0.allSatisfy { $0 >= 0 } }) else { throw LoadError.invalidReplay }
        let rosterIDs = value.data.teams.map { Set($0.players.map(\.id)) }
        let allPlayers = value.data.teams.flatMap(\.players)
        guard Set(allPlayers.map(\.id)).count == allPlayers.count,
              value.data.teams.allSatisfy({ $0.players.count >= 5 && $0.parameters.defenseFactors.count == 4 }),
              allPlayers.allSatisfy({ player in
                  player.zones.count == 4 && player.accuracy.count == 4 && player.projectedMinutes >= 0 &&
                  player.zones.allSatisfy { (0...1).contains($0) } &&
                  player.accuracy.allSatisfy { (0...1).contains($0) } && abs(player.zones.reduce(0, +) - 1) < 0.0001
              }) else { throw LoadError.invalidReplay }
        for replay in value.replays {
            guard !replay.events.isEmpty, replay.events.count <= 1000, replay.score.count == 2,
                  Set(replay.box.keys) == Set(allPlayers.map(\.id)) else { throw LoadError.invalidReplay }
            var score = [0, 0], previousPeriod = 1, previousClock = 720
            for event in replay.events {
                let newPeriod = event.period > previousPeriod
                let startClock = newPeriod ? (event.period <= 4 ? 720 : 300) : previousClock
                guard (0...1).contains(event.team), (1...20).contains(event.period),
                      event.period == previousPeriod || (event.period == previousPeriod + 1 && previousClock == 0),
                      event.clock >= 0, event.duration > 0, event.duration <= 30,
                      event.clock + event.duration == startClock, event.score_before == score,
                      event.lineups.count == 2, event.score.count == 2 else { throw LoadError.invalidReplay }
                for team in 0...1 {
                    guard event.lineups[team].count == 5, Set(event.lineups[team]).count == 5,
                          Set(event.lineups[team]).isSubset(of: rosterIDs[team]) else { throw LoadError.invalidReplay }
                }
                guard event.lineups[event.team].contains(event.player),
                      event.assist.map({ event.lineups[event.team].contains($0) && $0 != event.player }) ?? true,
                      event.rebound.map({ event.lineups[event.offensive_rebound ? event.team : 1 - event.team].contains($0) }) ?? true,
                      !event.offensive_rebound || event.rebound != nil else { throw LoadError.invalidReplay }
                switch event.kind {
                case "shot":
                    guard let zone = event.zone, zoneOrder.contains(zone),
                          event.points == (event.made ? (zone == "three" ? 3 : 2) : 0),
                          event.assist == nil || event.made, event.rebound == nil || !event.made else { throw LoadError.invalidReplay }
                case "free throws":
                    guard event.zone == nil, (event.free_throws ?? 2) == 2, (0...2).contains(event.points),
                          event.made == (event.points > 0), event.assist == nil else { throw LoadError.invalidReplay }
                case "turnover":
                    guard event.zone == nil, event.points == 0, !event.made, event.assist == nil, event.rebound == nil else { throw LoadError.invalidReplay }
                default: throw LoadError.invalidReplay
                }
                score[event.team] += event.points
                guard score == event.score else { throw LoadError.invalidReplay }
                previousPeriod = event.period; previousClock = event.clock
            }
            guard score == replay.score, previousPeriod >= 4, previousClock == 0 else { throw LoadError.invalidReplay }
            let counts = replay.counts(completed: replay.events.count)
            for player in allPlayers {
                guard let exported = replay.box[player.id], exported.name == player.name else { throw LoadError.invalidReplay }
                var line = counts[player.id, default: .init()]
                guard abs(line.minutes - exported.minutes) <= 0.0051 else { throw LoadError.invalidReplay }
                line.seconds = 0
                guard line == exported.counters else { throw LoadError.invalidReplay }
            }
        }
        return value
    }

    func description(_ event: Event, completed: Bool = true) -> String {
        let name = players[event.player]?.shortName ?? "Player"
        if !completed {
            if event.zone == "unlocated_two" { return name + " · Unlocated 2" }
            return name + (event.kind == "free throws" ? " · Two-shot trip" : " with the ball")
        }
        switch event.kind {
        case "turnover": return name + " turns it over"
        case "free throws": return name + " · \(event.points)/2 free throws"
        default:
            let shot = event.zone == "three" ? "a three" : event.zone == "rim" ? "at the rim" : event.zone == "unlocated_two" ? "a two · location unknown" : "a mid-range shot"
            return name + (event.made ? " makes " : " misses ") + shot
        }
    }
}
