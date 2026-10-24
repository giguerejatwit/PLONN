import Foundation

/// Recent verified box scores supply player rates. Current-roster simulations
/// use sport-specific event ledgers; the legacy sampler remains for comparison tests.
struct PlayerSimulation: Sendable {
    struct Stat: Hashable, Sendable, Identifiable {
        let id: String
        let label: String
        let title: String
        var kind: Kind = .count
        enum Kind: Sendable { case count, decimal, seconds, outs }
        func display(_ value: Double, average: Bool = false) -> String {
            switch kind {
            case .seconds:
                let seconds = Int(value.rounded()); return String(format: "%d:%02d", seconds / 60, seconds % 60)
            case .outs:
                if average { return String(format: "%.1f", value / 3) }
                return "\(Int(value) / 3).\(Int(value) % 3)"
            case .decimal: return String(format: "%.1f", value)
            case .count: return average ? String(format: "%.1f", value) : String(format: "%.0f", value)
            }
        }
    }
    struct Group: Sendable, Identifiable {
        let id: String
        let title: String
        let stats: [Stat]
    }
    struct Line: Sendable, Equatable, Identifiable {
        let playerID: String
        let name: String
        let group: String
        let values: [String: Double]
        var id: String { group + ":" + playerID }
    }
    struct Box: Sendable, Identifiable {
        let game: MatchupSimulation.Game
        let lines: [Line]
        var id: String { game.id }
    }
    struct History: Sendable {
        let team: InterestTeam
        let roster: [PlayerMatchupPlayer]
        let boxes: [Box]
        let partial: Bool
    }
    struct Trial: Sendable, Equatable { let away: Int; let home: Int }
    struct Range: Sendable {
        let average: Double
        let low: Double
        let high: Double
        let appearances: Int
        init(_ values: [(Double, Int)], appearances: Int) {
            let sorted = values.sorted { $0.0 < $1.0 }, count = values.reduce(0) { $0 + $1.1 }
            average = values.reduce(0) { $0 + $1.0 * Double($1.1) } / Double(count)
            func quantile(_ fraction: Double) -> Double {
                let index = Int(Double(count - 1) * fraction); var consumed = 0
                for (value, weight) in sorted { consumed += weight; if consumed > index { return value } }
                return sorted.last!.0
            }
            low = quantile(0.1); high = quantile(0.9)
            self.appearances = appearances
        }
    }
    let away: History
    let home: History
    let trials: [Trial]
    let scores: MatchupSimulation
    let awayDrawCounts: [Int]
    let homeDrawCounts: [Int]

    struct Current: Sendable {
        let teams: [SportPlaySimulation.Team]
        let templates: [[Line]]
        let columns: [[String: [Float]]]
        let ranges: [[String: Range]]
        var live: SimulationLiveSnapshot? = nil
        var sourceTrials: [Int] = []
        var forecast: SimulationForecast.Evidence? = nil
    }
    var current: Current? = nil
    func replay(_ trial: Int) throws -> SportPlaySimulation.Game {
        guard let current else { throw SportPlaySimulation.Failure.roster }
        return try SportPlaySimulation.run(away: current.teams[0], home: current.teams[1], seed: Self.trialSeed(scores.seed, current.sourceTrials.isEmpty ? trial : current.sourceTrials[min(max(0, trial), current.sourceTrials.count - 1)]), live: current.live)
    }
    /// Hash each trial into a distinct stream instead of shifting the same RNG
    /// sequence by one draw between successive games.
    static func trialSeed(_ seed: UInt64, _ index: Int) -> UInt64 {
        var value = seed &+ UInt64(index) &* 0x9e3779b97f4a7c15
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }
    static func currentRun(away: History, home: History, availability: [SimulationAvailability], now: Date, seed: UInt64, count: Int = 5_000, live: SimulationLiveSnapshot? = nil, market: SimulationForecast.MarketInput? = nil) throws -> Self {
        guard availability.count == 2, (100...10_000).contains(count), away.team != home.team, away.team.sport == home.team.sport else { throw MatchupSimulation.Failure.selection }
        for (index, history) in [away, home].enumerated() {
            guard (5...20).contains(history.boxes.count), Set(history.boxes.map(\.id)).count == history.boxes.count,
                  history.boxes.allSatisfy({ $0.game.date < now && $0.game.date > now.addingTimeInterval(-540 * 86400) }),
                  let latest = history.boxes.map(\.game.date).max(), now.timeIntervalSince(latest) < 240 * 86400,
                  (0...120).contains(now.timeIntervalSince(availability[index].checked)),
                  availability[index].roster.allSatisfy({ $0.teamID == history.team.providerID }) else { throw MatchupSimulation.Failure.history }
        }
        if let live {
            guard !live.isStale(at: now), live.teamIDs == [away.team.providerID, home.team.providerID],
                  (away.boxes + home.boxes).allSatisfy({ $0.game.date < live.start && $0.id != live.eventID }) else { throw MatchupSimulation.Failure.history }
        }
        let teams = try [SportPlaySimulation.team(away, availability: availability[0]), SportPlaySimulation.team(home, availability: availability[1])]
        var scores: [MatchupSimulation.Score] = [], columns: [[String: [Float]]] = [[:], [:]], templates: [[String: Line]] = [[:], [:]]
        for index in 0..<count {
            try Task.checkCancellation()
            let game = try SportPlaySimulation.run(away: teams[0], home: teams[1], seed: trialSeed(seed, index), recordPlays: false, live: live)
            scores.append(.init(away: game.awayScore, home: game.homeScore))
            for (side, lines) in [game.away, game.home].enumerated() { for line in lines {
                templates[side][line.id] = line
                for (stat, value) in line.values {
                    let key = line.id + "/" + stat
                    if columns[side][key] == nil { columns[side][key] = Array(repeating: .nan, count: count) }
                    columns[side][key]![index] = Float(value)
                }
            } }
        }
        let selection = SimulationForecast.select(scores: scores, histories: [away, home], now: now, seed: seed, market: market, live: live)
        scores = selection.indices.map { scores[$0] }
        columns = columns.map { side in side.mapValues { values in selection.indices.map { values[$0] } } }
        var ranges: [[String: Range]] = [[:], [:]]
        for side in 0...1 { for (key, values) in columns[side] {
            let components = key.split(separator: "/"), lineID = String(components[0]), stat = String(components[1])
            let history = side == 0 ? away : home
            let appearances = history.boxes.filter { $0.lines.first(where: { $0.id == lineID })?.values[stat] != nil }.count
            let known = values.filter(\.isFinite)
            if appearances >= 3, !known.isEmpty { ranges[side][key] = Range(known.map { (Double($0), 1) }, appearances: appearances) }
        } }
        let summary = MatchupSimulation(away: .init(team: away.team, games: away.boxes.map(\.game), partial: away.partial), home: .init(team: home.team, games: home.boxes.map(\.game), partial: home.partial), scores: scores, generated: now, seed: seed)
        return Self(away: away, home: home, trials: (0..<count).map { .init(away: $0, home: $0) }, scores: summary, awayDrawCounts: [], homeDrawCounts: [], current: .init(teams: teams, templates: templates.map { $0.values.sorted { $0.id < $1.id } }, columns: columns, ranges: ranges, live: live, sourceTrials: selection.indices, forecast: selection.evidence))
    }
    func box(homeSide: Bool, trial: Int) -> Box {
        if let current {
            let index = min(max(trial, 0), trials.count - 1), side = homeSide ? 1 : 0
            let score = scores.scores[index]
            let lines = current.templates[side].compactMap { template -> Line? in
                let values = template.values.keys.reduce(into: [String: Double]()) { values, stat in
                    if let value = current.columns[side][template.id + "/" + stat]?[index], value.isFinite { values[stat] = Double(value) }
                }
                return values.isEmpty ? nil : Line(playerID: template.playerID, name: template.name, group: template.group, values: values)
            }
            return Box(game: .init(id: "simulation-" + String(index), date: scores.generated, season: 0, scored: homeSide ? score.home : score.away, allowed: homeSide ? score.away : score.home, home: homeSide), lines: lines)
        }
        let draw = trials[min(max(trial, 0), trials.count - 1)]
        return homeSide ? home.boxes[draw.home] : away.boxes[draw.away]
    }
    func range(homeSide: Bool, lineID: String, stat: String) -> Range? {
        if let current { return current.ranges[homeSide ? 1 : 0][lineID + "/" + stat] }
        let history = homeSide ? home : away
        let observations = history.boxes.map { $0.lines.first { $0.id == lineID }?.values[stat] }
        let appearances = observations.compactMap { $0 }.count
        // A missing appearance is not a zero. At least three distinct games,
        // not thousands of repeated draws, are needed for a displayed range.
        guard appearances >= 3 else { return nil }
        let weights = homeSide ? homeDrawCounts : awayDrawCounts
        let values = observations.enumerated().compactMap { index, value -> (Double, Int)? in
            guard let value, weights[index] > 0 else { return nil }; return (value, weights[index])
        }
        guard !values.isEmpty else { return nil }
        return Range(values, appearances: appearances)
    }
    func rangeUnavailableReason(homeSide: Bool, lineID: String, stat: String) -> String {
        let history = homeSide ? home : away
        if let current, current.columns[homeSide ? 1 : 0][lineID + "/" + stat]?.contains(where: \.isFinite) != true {
            return "This stat isn’t modeled for this player."
        }
        let count = history.boxes.filter { $0.lines.first { $0.id == lineID }?.values[stat] != nil }.count
        return count < 3 ? "\(count) recorded games · 3 needed for a range." : "No simulated values are available for this stat."
    }
    static func run(away: History, home: History, now: Date, seed: UInt64, count: Int = 5_000) throws -> Self {
        guard away.team != home.team, away.team.sport == home.team.sport, (100...10_000).contains(count) else { throw MatchupSimulation.Failure.selection }
        for history in [away, home] {
            guard history.team.isValid, (5...20).contains(history.boxes.count), !history.roster.isEmpty,
                  Set(history.boxes.map(\.id)).count == history.boxes.count,
                  history.boxes.allSatisfy({ !$0.lines.isEmpty && $0.game.date < now && $0.game.date > now.addingTimeInterval(-540 * 86400) }),
                  let recent = history.boxes.map(\.game.date).max(), now.timeIntervalSince(recent) < 240 * 86400 else { throw MatchupSimulation.Failure.history }
        }
        var state = seed
        func draw(_ count: Int) -> Int {
            state &+= 0x9e3779b97f4a7c15
            var z = state; z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            return Int((z ^ (z >> 31)) % UInt64(count))
        }
        let trials = (0..<count).map { _ in Trial(away: draw(away.boxes.count), home: draw(home.boxes.count)) }
        let scores = MatchupSimulation(
            away: .init(team: away.team, games: away.boxes.map(\.game), partial: away.partial),
            home: .init(team: home.team, games: home.boxes.map(\.game), partial: home.partial),
            scores: trials.map { .init(away: away.boxes[$0.away].game.scored, home: home.boxes[$0.home].game.scored) }, generated: now, seed: seed)
        var aCounts = Array(repeating: 0, count: away.boxes.count), hCounts = Array(repeating: 0, count: home.boxes.count)
        for trial in trials { aCounts[trial.away] += 1; hCounts[trial.home] += 1 }
        return .init(away: away, home: home, trials: trials, scores: scores, awayDrawCounts: aCounts, homeDrawCounts: hCounts)
    }

    static func groups(_ sport: InterestSport) -> [Group] {
        func s(_ key: String, _ label: String, _ title: String, _ kind: Stat.Kind = .count) -> Stat { .init(id: key, label: label, title: title, kind: kind) }
        switch sport {
        case .nba:
            return [.init(id: "players", title: "Players", stats: [s("points", "PTS", "Points"), s("rebounds", "REB", "Rebounds"), s("assists", "AST", "Assists"), s("steals", "STL", "Steals"), s("blocks", "BLK", "Blocks"), s("threePointFieldGoalsMade", "3PM", "Three-pointers made"), s("turnovers", "TO", "Turnovers"), s("minutes", "MIN", "Minutes", .decimal), s("fieldGoalsMade", "FGM", "Field goals made"), s("fieldGoalsAttempted", "FGA", "Field goal attempts"), s("freeThrowsMade", "FTM", "Free throws made"), s("freeThrowsAttempted", "FTA", "Free throw attempts"), s("threePointFieldGoalsAttempted", "3PA", "Three-point attempts")])]
        case .nfl:
            return [
                .init(id: "passing", title: "Passing", stats: [s("passingYards", "YDS", "Passing yards"), s("passingTouchdowns", "TD", "Passing touchdowns"), s("interceptions", "INT", "Interceptions thrown"), s("completions", "CMP", "Completions"), s("passingAttempts", "ATT", "Passing attempts")]),
                .init(id: "rushing", title: "Rushing", stats: [s("rushingYards", "YDS", "Rushing yards"), s("rushingAttempts", "CAR", "Carries"), s("rushingTouchdowns", "TD", "Rushing touchdowns"), s("longRushing", "LNG", "Longest rush")]),
                .init(id: "receiving", title: "Receiving", stats: [s("receivingYards", "YDS", "Receiving yards"), s("receptions", "REC", "Receptions"), s("receivingTouchdowns", "TD", "Receiving touchdowns"), s("receivingTargets", "TGT", "Targets"), s("longReception", "LNG", "Longest reception")]),
                .init(id: "defensive", title: "Defense", stats: [s("totalTackles", "TKL", "Total tackles", .decimal), s("soloTackles", "SOLO", "Solo tackles"), s("sacks", "SACK", "Sacks", .decimal), s("tacklesForLoss", "TFL", "Tackles for loss", .decimal), s("passesDefended", "PD", "Passes defended"), s("defensiveTouchdowns", "TD", "Defensive touchdowns")]),
                .init(id: "kicking", title: "Kicking", stats: [s("totalKickingPoints", "PTS", "Kicking points"), s("fieldGoalsMade", "FGM", "Field goals made"), s("fieldGoalAttempts", "FGA", "Field goal attempts"), s("extraPointsMade", "XPM", "Extra points made"), s("extraPointAttempts", "XPA", "Extra point attempts")]),
                .init(id: "interceptions", title: "Interceptions", stats: [s("interceptions", "INT", "Defensive interceptions"), s("interceptionYards", "YDS", "Interception return yards"), s("interceptionTouchdowns", "TD", "Interception return touchdowns")]),
                .init(id: "kickReturns", title: "Kick returns", stats: [s("kickReturns", "RET", "Kick returns"), s("kickReturnYards", "YDS", "Kick return yards"), s("kickReturnTouchdowns", "TD", "Kick return touchdowns")]),
                .init(id: "puntReturns", title: "Punt returns", stats: [s("puntReturns", "RET", "Punt returns"), s("puntReturnYards", "YDS", "Punt return yards"), s("puntReturnTouchdowns", "TD", "Punt return touchdowns")]),
                .init(id: "punting", title: "Punting", stats: [s("punts", "PUNT", "Punts"), s("puntYards", "YDS", "Punt yards"), s("puntsInside20", "IN20", "Punts inside the 20")])]
        case .mlb:
            return [.init(id: "batting", title: "Batting", stats: [s("hits", "H", "Hits"), s("runs", "R", "Runs"), s("RBIs", "RBI", "Runs batted in"), s("homeRuns", "HR", "Home runs"), s("doubles", "2B", "Doubles"), s("triples", "3B", "Triples"), s("atBats", "AB", "At bats"), s("walks", "BB", "Walks"), s("strikeouts", "K", "Strikeouts")]),
                    .init(id: "pitching", title: "Pitching", stats: [s("strikeouts", "K", "Strikeouts"), s("fullInnings.partInnings", "IP", "Innings pitched", .outs), s("earnedRuns", "ER", "Earned runs"), s("hits", "H", "Hits allowed"), s("walks", "BB", "Walks allowed"), s("pitches", "PC", "Pitch count"), s("homeRuns", "HR", "Home runs allowed")])]
        case .nhl:
            return [.init(id: "skaters", title: "Skaters", stats: [s("goals", "G", "Goals"), s("assists", "A", "Assists"), s("shotsTotal", "SOG", "Shots on goal"), s("hits", "HIT", "Hits"), s("blockedShots", "BLK", "Blocked shots"), s("timeOnIce", "TOI", "Time on ice", .seconds), s("penaltyMinutes", "PIM", "Penalty minutes")]),
                    .init(id: "goalies", title: "Goalies", stats: [s("saves", "SV", "Saves"), s("goalsAgainst", "GA", "Goals against"), s("shotsAgainst", "SA", "Shots against"), s("timeOnIce", "TOI", "Time on ice", .seconds)])]
        }
    }
}

struct PlayerSimulationSource: Sendable {
    typealias Reader = MatchupSimulationSource.Reader
    var read: Reader = { try await PlayerHistoryService.shared.read($0) }

    func history(_ history: MatchupSimulation.History, cutoff: Date) async throws -> PlayerSimulation.History {
        let team = history.team
        let rosterURL = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(team.sport.leaguePath)/teams/\(team.providerID)/roster")!
        let roster = try PlayerMatchupSource.players(await read(rosterURL), teamID: team.providerID, teamName: team.name)
        guard !roster.isEmpty else { throw MatchupSimulation.Failure.source }
        let games = Array(history.games.filter { $0.date < cutoff && $0.date > cutoff.addingTimeInterval(-540 * 86400) }
            .sorted { $0.date > $1.date }.prefix(10))
        var boxes: [PlayerSimulation.Box] = []
        // Four bounded reads per team, shared HTTP cache. No paid odds calls.
        for start in stride(from: 0, to: games.count, by: 4) {
            try Task.checkCancellation()
            let batch = Array(games[start..<min(start + 4, games.count)])
            let loaded = await withTaskGroup(of: PlayerSimulation.Box?.self, returning: [PlayerSimulation.Box].self) { group in
                for game in batch {
                    group.addTask {
                        let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(team.sport.leaguePath)/summary?event=\(game.id)")!
                        return try? Self.parse(await read(url), team: team, game: game, cutoff: cutoff)
                    }
                }
                var values: [PlayerSimulation.Box] = []
                for await value in group { if let value { values.append(value) } }
                return values
            }
            boxes += loaded
        }
        try Task.checkCancellation()
        guard boxes.count >= 5 else { throw MatchupSimulation.Failure.history }
        return .init(team: team, roster: roster, boxes: boxes.sorted { $0.game.date > $1.game.date }, partial: history.partial || boxes.count != games.count)
    }

    static func parse(_ data: Data, team: InterestTeam, game: MatchupSimulation.Game, cutoff: Date, allowInProgress: Bool = false) throws -> PlayerSimulation.Box {
        guard data.count <= 4 * 1024 * 1024, team.isValid,
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let header = root["header"] as? [String: Any], header["id"] as? String == game.id,
              let league = header["league"] as? [String: Any], (league["abbreviation"] as? String)?.uppercased() == team.sport.rawValue,
              let season = header["season"] as? [String: Any], season["year"] as? Int == game.season,
              [2, 3].contains(season["type"] as? Int ?? 0),
              let competitions = header["competitions"] as? [[String: Any]], competitions.count == 1,
              let competition = competitions.first, competition["id"] as? String == game.id,
              let date = (competition["date"] as? String).flatMap(sportsDate), abs(date.timeIntervalSince(game.date)) < 60, date < cutoff,
              let status = competition["status"] as? [String: Any], let type = status["type"] as? [String: Any],
              ((type["completed"] as? Bool == true && type["state"] as? String == "post") || (allowInProgress && type["state"] as? String == "in")),
              let competitors = competition["competitors"] as? [[String: Any]], competitors.count == 2,
              let own = competitors.first(where: { $0["id"] as? String == team.providerID }),
              let other = competitors.first(where: { $0["id"] as? String != team.providerID }),
              own["score"] as? String == String(game.scored), other["score"] as? String == String(game.allowed),
              own["homeAway"] as? String == (game.home ? "home" : "away"),
              let boxscore = root["boxscore"] as? [String: Any], let sides = boxscore["players"] as? [[String: Any]], sides.count == 2,
              sides.filter({ ($0["team"] as? [String: Any])?["id"] as? String == team.providerID }).count == 1,
              let side = sides.first(where: { ($0["team"] as? [String: Any])?["id"] as? String == team.providerID }),
              let categories = side["statistics"] as? [[String: Any]], categories.count <= 20 else { throw MatchupSimulation.Failure.source }
        var lines: [String: PlayerSimulation.Line] = [:]
        for category in categories {
            guard let keys = category["keys"] as? [String], keys.count <= 60, Set(keys).count == keys.count,
                  let athletes = category["athletes"] as? [[String: Any]], athletes.count <= 150 else { throw MatchupSimulation.Failure.source }
            let groupID: String
            switch team.sport {
            case .nba: groupID = "players"
            case .mlb: groupID = keys.contains("fullInnings.partInnings") ? "pitching" : "batting"
            case .nhl: groupID = category["name"] as? String == "goalies" ? "goalies" : "skaters"
            case .nfl: groupID = category["name"] as? String ?? ""
            }
            guard let group = PlayerSimulation.groups(team.sport).first(where: { $0.id == groupID }) else { continue }
            for row in athletes {
                // `active` means currently on the field/court, not participation.
                // Substituted-out players still belong in a completed box score.
                if row["didNotPlay"] as? Bool == true { continue }
                guard let athlete = row["athlete"] as? [String: Any], let id = athlete["id"] as? String, InterestTeam.validProviderID(id),
                      let name = athlete["displayName"] as? String, (1...120).contains(name.count),
                      let values = row["stats"] as? [String], values.count == keys.count else { throw MatchupSimulation.Failure.source }
                var raw = Dictionary(uniqueKeysWithValues: zip(keys, values))
                // Expand made/attempted pairs before selecting supported stats.
                for key in keys where key.contains("-") || key.contains("/") {
                    let separator: Character = key.contains("/") ? "/" : "-"
                    let names = key.split(separator: separator), parts = (raw[key] ?? "").split(separator: separator)
                    if names.count == 2, parts.count == 2 {
                        raw[String(names[0])] = String(parts[0]); raw[String(names[1])] = String(parts[1])
                    }
                }
                var parsed: [String: Double] = [:]
                for stat in group.stats {
                    guard let text = raw[stat.id], !["", "--", "-", "DNP"].contains(text) else { continue }
                    guard let number = number(text, kind: stat.kind), number.isFinite, (-100...20_000).contains(number),
                          stat.kind != .count || number.rounded() == number else { throw MatchupSimulation.Failure.source }
                    if number < 0 && !["passingYards", "rushingYards", "receivingYards", "interceptionYards", "kickReturnYards", "puntReturnYards", "longRushing", "longReception"].contains(stat.id) { throw MatchupSimulation.Failure.source }
                    parsed[stat.id] = number
                }
                if team.sport == .nba, let rawValue = raw["offensiveRebounds"], let value = Double(rawValue), value.isFinite, (0...50).contains(value), value.rounded() == value { parsed["offensiveRebounds"] = value }
                guard !parsed.isEmpty else { continue }
                for (made, attempts) in [("fieldGoalsMade", "fieldGoalsAttempted"), ("threePointFieldGoalsMade", "threePointFieldGoalsAttempted"), ("freeThrowsMade", "freeThrowsAttempted"), ("completions", "passingAttempts"), ("hits", "atBats")] {
                    if let m = parsed[made], let a = parsed[attempts], m > a { throw MatchupSimulation.Failure.source }
                }
                if team.sport == .nba, let pts = parsed["points"], let fg = parsed["fieldGoalsMade"], let threes = parsed["threePointFieldGoalsMade"], let ft = parsed["freeThrowsMade"], pts != 2 * fg + threes + ft { throw MatchupSimulation.Failure.source }
                let line = PlayerSimulation.Line(playerID: id, name: name, group: groupID, values: parsed)
                if let previous = lines[line.id], previous != line { throw MatchupSimulation.Failure.source }
                lines[line.id] = line
            }
        }
        if team.sport == .mlb, let plays = root["plays"] as? [[String: Any]], plays.count <= 5000 {
            var hits: [String: [String: Int]] = [:], seen = Set<String>()
            for play in plays {
                guard (play["type"] as? [String: Any])?["id"] as? String == "57",
                      let atBat = play["atBatId"] as? String, seen.insert(atBat).inserted,
                      let outcome = (play["alternativeType"] as? [String: Any])?["type"] as? String,
                      ["single", "double", "triple", "home-run"].contains(outcome),
                      let participants = play["participants"] as? [[String: Any]],
                      let batter = participants.first(where: { $0["type"] as? String == "batter" }),
                      let id = (batter["athlete"] as? [String: Any])?["id"] as? String else { continue }
                hits[id, default: [:]][outcome, default: 0] += 1
            }
            for (key, line) in lines where line.group == "batting" {
                let counts = hits[line.playerID] ?? [:]
                // Only add extra-base-hit rates when the play ledger agrees
                // with the verified box score; absent play data isn't zero.
                if Double(counts.values.reduce(0, +)) == line.values["hits"] {
                    var values = line.values; values["doubles"] = Double(counts["double", default: 0]); values["triples"] = Double(counts["triple", default: 0])
                    lines[key] = .init(playerID: line.playerID, name: line.name, group: line.group, values: values)
                }
            }
        }
        guard !lines.isEmpty else { throw MatchupSimulation.Failure.source }
        if team.sport == .nba || team.sport == .mlb || team.sport == .nhl {
            let group = team.sport == .nba ? "players" : team.sport == .mlb ? "batting" : "skaters"
            let key = team.sport == .nba ? "points" : team.sport == .mlb ? "runs" : "goals"
            let rows = lines.values.filter { $0.group == group }
            guard !rows.isEmpty, rows.allSatisfy({ $0.values[key] != nil }) else { throw MatchupSimulation.Failure.source }
            let total = rows.reduce(0.0) { $0 + ($1.values[key] ?? 0) }
            let difference = Double(game.scored) - total
            // NHL shootout winners receive one team goal, not a player goal.
            guard difference == 0 || (team.sport == .nhl && difference == 1 && type["name"] as? String == "STATUS_FINAL_SO") else { throw MatchupSimulation.Failure.source }
        }
        return .init(game: game, lines: lines.values.sorted { $0.id < $1.id })
    }
    static func number(_ text: String, kind: PlayerSimulation.Stat.Kind) -> Double? {
        switch kind {
        case .seconds:
            let parts = text.split(separator: ":")
            guard parts.count == 2, let minutes = Int(parts[0]), let seconds = Int(parts[1]), minutes >= 0, (0...59).contains(seconds) else { return nil }
            return Double(minutes * 60 + seconds)
        case .outs:
            let parts = text.split(separator: ".")
            guard (1...2).contains(parts.count), let innings = Int(parts[0]), innings >= 0 else { return nil }
            let outs = parts.count == 2 ? Int(parts[1]) : 0
            guard let outs, (0...2).contains(outs) else { return nil }; return Double(innings * 3 + outs)
        case .count, .decimal: return Double(text)
        }
    }
}
