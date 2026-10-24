import Foundation

/// Presentation only: deterministic, illustrative paths from the play ledger.
/// This never changes an outcome, player statistic, roster or probability.
enum SportReplayMotion {
    struct Point: Equatable {
        var x: Double
        var y: Double
        func toward(_ end: Self, _ amount: Double) -> Self {
            let t = min(1, max(0, amount))
            return .init(x: x + (end.x - x) * t, y: y + (end.y - y) * t)
        }
        var bounded: Self { .init(x: min(0.96, max(0.04, x)), y: min(0.94, max(0.06, y))) }
    }
    struct Marker: Equatable {
        let id: String
        let playerID: String?
        let team: Int
        var point: Point
        var emphasized: Bool
    }
    struct Scene: Equatable {
        var markers: [Marker]
        var ball: Point?
        var ballHeight: Double = 0
    }
    private static func random(_ text: String) -> Double {
        var hash: UInt32 = 2166136261
        for byte in text.utf8 { hash = (hash ^ UInt32(byte)) &* 16777619 }
        return Double(hash) / Double(UInt32.max)
    }
    private static func eased(_ t: Double) -> Double {
        let t = min(1, max(0, t)); return t * t * (3 - 2 * t)
    }
    private static func attackingRight(_ sport: InterestSport, _ play: SportPlaySimulation.Play) -> Bool {
        let firstDirection = !play.home
        if sport == .nba || sport == .nfl { return play.period <= 2 ? firstDirection : !firstDirection }
        if sport == .nhl { return play.period % 2 == 1 ? firstDirection : !firstDirection }
        return true
    }
    private static func oriented(_ x: Double, _ y: Double, sport: InterestSport, play: SportPlaySimulation.Play) -> Point {
        .init(x: attackingRight(sport, play) ? x : 1 - x, y: y)
    }
    private static func layout(sport: InterestSport, teams: [SportPlaySimulation.Team], play: SportPlaySimulation.Play) -> [Marker] {
        guard teams.count == 2 else { return [] }
        let offense = play.home ? 1 : 0
        func point(_ x: Double, _ y: Double) -> Point { oriented(x, y, sport: sport, play: play).bounded }
        var result: [Marker] = []
        func add(_ id: String?, team: Int, at p: Point, slot: Int) {
            let verified = id.flatMap { id in teams[team].profiles.contains { $0.id == id } ? id : nil }
            result.append(.init(id: verified ?? "formation-\(team)-\(slot)", playerID: verified, team: team, point: p.bounded, emphasized: verified != nil && (verified == play.actorID || verified == play.secondaryID)))
        }
        if sport == .nba {
            // Minute deltas identify the engine's actual five-player rotation.
            let lineups = (0...1).map { side in
                Array(Set(play.deltas.filter { $0.home == (side == 1) && $0.group == "players" && $0.stat == "minutes" && $0.amount >= 0 }.map(\.playerID))).sorted()
            }
            let formation: [(Double, Double)] = [(0.70, 0.24), (0.65, 0.76), (0.77, 0.36), (0.83, 0.70), (0.61, 0.5)]
            for (i, id) in lineups[offense].prefix(5).enumerated() {
                let base = formation[i]
                var p = point(base.0 + (random("\(play.id)x" + id) - 0.5) * 0.034, base.1 + (random("\(play.id)y" + id) - 0.5) * 0.043)
                if id == play.actorID {
                    let distance = play.kind == "three" ? 0.26 : play.kind == "freeThrow" ? 0.145 : play.kind == "turnover" ? 0.25 : 0.10
                    p = point(0.926 - distance, play.kind == "freeThrow" ? 0.5 : 0.5 + (random("\(play.id)shot") - 0.5) * (play.kind == "three" ? 0.5 : 0.16))
                }
                add(id, team: offense, at: p, slot: i)
            }
            let attackers = result
            for (i, id) in lineups[1 - offense].prefix(5).enumerated() {
                let marked = i < attackers.count ? attackers[i].point : point(0.75, 0.5)
                let sign = attackingRight(sport, play) ? 1.0 : -1.0
                add(id, team: 1 - offense, at: .init(x: marked.x + sign * 0.032, y: marked.y + (i % 2 == 0 ? 0.032 : -0.032)), slot: i)
            }
        } else if sport == .nhl {
            for side in 0...1 {
                let involved = play.deltas.filter { $0.home == (side == 1) && $0.group == "skaters" && ["goals", "assists", "shotsTotal"].contains($0.stat) }.map(\.playerID)
                let ordered = involved + teams[side].profiles.filter { $0.mean("skaters", "timeOnIce") > 0 }.sorted { $0.mean("skaters", "timeOnIce") > $1.mean("skaters", "timeOnIce") }.map(\.id)
                var seen = Set<String>()
                let skaters = ordered.filter { seen.insert($0).inserted }.prefix(5)
                for (i, id) in skaters.enumerated() {
                    let x = side == offense ? 0.52 + Double(i % 3) * 0.10 : 0.68 + Double(i % 2) * 0.11
                    add(id, team: side, at: point(id == play.actorID ? 0.76 : x, id == play.actorID ? 0.42 : 0.18 + Double(i) * 0.15), slot: i)
                }
                let goalie = play.deltas.first { $0.home == (side == 1) && $0.group == "goalies" }?.playerID
                add(goalie, team: side, at: point(side == offense ? 0.07 : 0.93, 0.5), slot: 5)
            }
        } else if sport == .nfl {
            // Blocking/defending assignments are not supplied: these markers
            // deliberately have no player identity attached.
            for side in 0...1 {
                for i in 0..<9 {
                    let x = side == offense ? (i < 5 ? 0.36 : 0.29) : (i < 5 ? 0.42 : 0.58)
                    add(nil, team: side, at: point(x, 0.17 + Double(i % 5) * 0.16), slot: i)
                }
            }
            add(play.actorID, team: offense, at: point(0.28, 0.5), slot: 9)
            if let receiver = play.secondaryID { add(receiver, team: offense, at: point(0.35, 0.18), slot: 10) }
        } else {
            add(play.actorID, team: offense, at: .init(x: 0.5, y: 0.86), slot: 0)
            add(play.secondaryID, team: 1 - offense, at: .init(x: 0.5, y: 0.49), slot: 0)
            let field: [(Double, Double)] = [(0.5, 0.93), (0.73, 0.51), (0.58, 0.31), (0.28, 0.51), (0.40, 0.33), (0.20, 0.20), (0.50, 0.12), (0.80, 0.20)]
            for (i, p) in field.enumerated() { add(nil, team: 1 - offense, at: .init(x: p.0, y: p.1), slot: i + 1) }
        }
        return result
    }
    private static func action(sport: InterestSport, play: SportPlaySimulation.Play, markers: [Marker], progress: Double) -> Scene {
        let t = min(1, max(0, progress)), side = play.home ? 1 : 0
        var markers = markers
        func location(_ id: String?) -> Point? { markers.first { id != nil && $0.playerID == id }?.point }
        guard let origin = location(play.actorID) else { return .init(markers: markers, ball: nil) }
        func point(_ x: Double, _ y: Double) -> Point { oriented(x, y, sport: sport, play: play).bounded }
        func move(_ id: String?, to p: Point) { if let i = markers.firstIndex(where: { id != nil && $0.playerID == id }) { markers[i].point = p.bounded } }
        var ball = origin, height = 0.0
        let flight = min(1, max(0, (t - 0.55) / 0.31))
        if sport == .nba || sport == .nhl {
            let helper = sport == .nba ? play.secondaryID : play.deltas.first { $0.home == play.home && $0.stat == "assists" && $0.amount > 0 }?.playerID
            if let passer = location(helper), t < 0.55 {
                ball = passer.toward(origin, eased((t - 0.25) / 0.23))
            }
            if t >= 0.55 && play.kind != "turnover" {
                let target = point(0.926, 0.5)
                ball = origin.toward(target, flight)
                height = sport == .nba ? sin(flight * .pi) * 0.15 : 0
                if t >= 0.86 {
                    let rebound = play.deltas.first { $0.stat == "rebounds" && $0.amount > 0 }?.playerID
                    let recipient = sport == .nba ? rebound : (play.points == 0 ? play.secondaryID : nil)
                    if let landing = location(recipient) { ball = target.toward(landing, eased((t - 0.86) / 0.14)) }
                }
            } else if play.kind == "turnover", t >= 0.7 {
                // Only identify a stealing defender when the engine recorded one.
                let thief = play.deltas.first { $0.stat == "steals" && $0.amount > 0 }?.playerID
                if let receiver = location(thief) { ball = origin.toward(receiver, eased((t - 0.7) / 0.3)) }
            }
        } else if sport == .nfl {
            let travel = eased((t - 0.18) / 0.72)
            let destination = point(min(0.92, max(0.18, 0.36 + Double(play.yards) / 100)), 0.30)
            if play.kind == "run" {
                let runner = origin.toward(destination, travel); move(play.actorID, to: runner); ball = runner
            } else if let receiver = location(play.secondaryID) {
                let end = play.kind == "incomplete" ? point(0.62, 0.23) : destination
                let runner = receiver.toward(end, travel); move(play.secondaryID, to: runner)
                ball = origin.toward(runner, flight); height = sin(flight * .pi) * 0.10
            } else if ["fieldGoal", "extraPoint", "punt", "interception"].contains(play.kind) {
                ball = origin.toward(point(0.94, 0.5), flight); height = sin(flight * .pi) * 0.16
            }
            for i in markers.indices where markers[i].playerID == nil {
                let p = markers[i].point
                markers[i].point = p.toward(destination, travel * (markers[i].team == side ? 0.12 : 0.28)).bounded
            }
        } else {
            let home = Point(x: 0.5, y: 0.86), mound = Point(x: 0.5, y: 0.49)
            ball = mound.toward(home, eased(t / 0.4))
            let bases = play.kind == "homeRun" ? 4 : play.kind == "triple" ? 3 : play.kind == "double" ? 2 : ["single", "walk"].contains(play.kind) ? 1 : 0
            if t >= 0.4 {
                let hit = bases > 0 && play.kind != "walk"
                let end = hit ? Point(x: 0.65 + random("hit\(play.id)") * 0.15, y: play.kind == "homeRun" ? 0.08 : 0.25) : home
                ball = home.toward(end, eased((t - 0.4) / 0.6))
                height = hit ? sin((t - 0.4) / 0.6 * .pi) * 0.08 : 0
                if bases > 0 {
                    let route = [home, Point(x: 0.72, y: 0.52), Point(x: 0.5, y: 0.18), Point(x: 0.28, y: 0.52), home]
                    let distance = min(Double(bases), (t - 0.4) / 0.6 * Double(bases)), segment = min(3, Int(distance))
                    move(play.actorID, to: route[segment].toward(route[segment + 1], distance - Double(segment)))
                }
            }
        }
        return .init(markers: markers, ball: ball, ballHeight: height)
    }
    static func scene(sport: InterestSport, teams: [SportPlaySimulation.Team], play: SportPlaySimulation.Play?, previous: SportPlaySimulation.Play?, progress: Double, reduceMotion: Bool = false) -> Scene {
        guard let play, teams.count == 2 else { return .init(markers: [], ball: nil) }
        let targets = layout(sport: sport, teams: teams, play: play)
        let t = progress.isFinite ? min(1, max(0, progress)) : 0
        if reduceMotion { return .init(markers: targets, ball: targets.first { $0.playerID == play.actorID }?.point) }
        let prior = previous.map { action(sport: sport, play: $0, markers: layout(sport: sport, teams: teams, play: $0), progress: 1).markers } ?? []
        let transition = eased(t / (sport == .nba || sport == .nhl ? 0.55 : 0.18))
        let moving = targets.map { target -> Marker in
            let start = prior.first { $0.id == target.id }?.point ?? (previous == nil ? target.point : Point(x: target.point.x, y: target.team == 0 ? 0.06 : 0.94))
            var marker = target; marker.point = start.toward(target.point, transition); return marker
        }
        var scene = action(sport: sport, play: play, markers: moving, progress: t)
        // Blend possession changes instead of snapping the ball across the surface.
        if t < 0.18, let previous, let end = scene.ball,
           let start = action(sport: sport, play: previous, markers: layout(sport: sport, teams: teams, play: previous), progress: 1).ball {
            scene.ball = start.toward(end, eased(t / 0.18))
        }
        return scene
    }
}

/// Sport-specific, experimental event generators fitted to recent player rates.
/// Whole paths are subsequently conditioned on team/opponent and market priors,
/// independently of the historical replay payload. Every actor must belong to the checked roster.
enum SportPlaySimulation {
    struct Profile: Sendable, Identifiable {
        let player: PlayerMatchupPlayer
        let means: [String: Double]
        let totals: [String: Double]
        let samples: [String: Int]
        private let groupedMeans: [String: [String: Double]]
        private let groupedTotals: [String: [String: Double]]
        init(player: PlayerMatchupPlayer, means: [String: Double], totals: [String: Double], samples: [String: Int]) {
            self.player = player; self.means = means; self.totals = totals; self.samples = samples
            func grouped(_ values: [String: Double]) -> [String: [String: Double]] {
                values.reduce(into: [:]) { result, entry in
                    guard let dot = entry.key.firstIndex(of: ".") else { return }
                    result[String(entry.key[..<dot]), default: [:]][String(entry.key[entry.key.index(after: dot)...])] = entry.value
                }
            }
            groupedMeans = grouped(means); groupedTotals = grouped(totals)
        }
        var id: String { player.id }
        func mean(_ group: String, _ key: String) -> Double { groupedMeans[group]?[key] ?? 0 }
        func total(_ group: String, _ key: String) -> Double { groupedTotals[group]?[key] ?? 0 }
        func rate(_ group: String, _ made: String, _ attempted: String) -> Double {
            let attempts = total(group, attempted)
            return attempts > 0 ? min(1, max(0, total(group, made) / attempts)) : 0
        }
    }
    struct Team: Sendable {
        let identity: InterestTeam
        let profiles: [Profile]
        let availability: SimulationAvailability
        let missingHistory: [PlayerMatchupPlayer]
        let possessions: Double?
    }
    struct Delta: Sendable, Equatable {
        let home: Bool
        let playerID: String
        let group: String
        let stat: String
        let amount: Double
    }
    struct Play: Identifiable, Sendable, Equatable {
        let id: Int
        let home: Bool
        let actorID: String
        let secondaryID: String?
        let kind: String
        let title: String
        let period: Int
        let clock: String
        let points: Int
        let yards: Int
        let deltas: [Delta]
    }
    struct Game: Sendable {
        let away: [PlayerSimulation.Line]
        let home: [PlayerSimulation.Line]
        let awayScore: Int
        let homeScore: Int
        let plays: [Play]
        var initialLines: [[PlayerSimulation.Line]] = [[], []]
        var initialScores: [Int] = [0, 0]
    }
    enum Failure: Error { case roster, history, lineup, incomplete }

    static func team(_ history: PlayerSimulation.History, availability: SimulationAvailability) throws -> Team {
        guard history.team.isValid, !availability.roster.isEmpty else { throw Failure.roster }
        let profiles = availability.eligible.compactMap { player -> Profile? in
            var sums: [String: Double] = [:], weights: [String: Double] = [:], counts: [String: Int] = [:]
            for box in history.boxes {
                let w = SimulationForecast.weight(date: box.game.date, asOf: availability.checked, sport: history.team.sport)
                for line in box.lines where line.playerID == player.id {
                    for (key, value) in line.values {
                        let field = line.group + "." + key
                        sums[field, default: 0] += value * w; weights[field, default: 0] += w; counts[field, default: 0] += 1
                    }
                }
            }
            guard !sums.isEmpty else { return nil }
            return Profile(player: player, means: sums.mapValues { $0 }.reduce(into: [:]) { $0[$1.key] = $1.value / max(1e-9, weights[$1.key] ?? 0) }, totals: sums, samples: counts)
        }
        guard !profiles.isEmpty else { throw Failure.history }
        let paces = history.team.sport == .nba ? history.boxes.compactMap { box -> Double? in
            let lines = box.lines.filter { $0.group == "players" }
            guard lines.allSatisfy({ $0.values["offensiveRebounds"] != nil }) else { return nil }
            func total(_ key: String) -> Double { lines.reduce(0) { $0 + ($1.values[key] ?? 0) } }
            let minutes = total("minutes")
            guard minutes >= 230 else { return nil }
            return (total("fieldGoalsAttempted") + 0.44 * total("freeThrowsAttempted") + total("turnovers") - total("offensiveRebounds")) * 240 / minutes
        } : []
        return Team(identity: history.team, profiles: profiles, availability: availability, missingHistory: availability.eligible.filter { player in !profiles.contains { $0.id == player.id } }, possessions: paces.count >= 3 ? paces.reduce(0, +) / Double(paces.count) : nil)
    }
    static func run(away: Team, home: Team, seed: UInt64, recordPlays: Bool = true, live: SimulationLiveSnapshot? = nil) throws -> Game {
        guard away.identity.sport == home.identity.sport, away.identity.id != home.identity.id,
              Set(away.availability.roster.map(\.id)).isDisjoint(with: Set(home.availability.roster.map(\.id))) else { throw Failure.roster }
        if let live {
            guard live.sport == away.identity.sport, live.teamIDs == [away.identity.providerID, home.identity.providerID], live.forecastIssue == nil else { throw Failure.incomplete }
        }
        var engine = Engine(teams: [away, home], random: Random(state: seed), record: recordPlays, observed: live)
        engine.scores = live?.scores ?? [0, 0]
        for (side, lines) in (live?.lines ?? [[], []]).enumerated() {
            for line in lines { engine.values[side][line.id] = line.values }
        }
        if live?.phase == .final { return engine.result() }
        switch away.identity.sport {
        case .nba: try engine.basketball()
        case .nfl: try engine.football()
        case .mlb: try engine.baseball()
        case .nhl: try engine.hockey()
        }
        return engine.result()
    }
    struct Random {
        var state: UInt64
        mutating func uniform() -> Double {
            state &+= 0x9e3779b97f4a7c15; var z = state
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            return Double((z ^ (z >> 31)) >> 11) / 9007199254740992
        }
        mutating func yes(_ p: Double) -> Bool { uniform() < min(1, max(0, p)) }
        mutating func normal(_ mean: Double, _ sd: Double) -> Double {
            mean + sd * sqrt(-2 * log(max(1e-12, uniform()))) * cos(2 * .pi * uniform())
        }
        mutating func count(_ mean: Double) -> Int {
            let mean = min(150, max(0, mean))
            if mean > 25 { return max(0, Int(normal(mean, sqrt(mean)).rounded())) }
            let limit = exp(-mean); var product = 1.0, count = 0
            repeat { count += 1; product *= uniform() } while product > limit && count < 500
            return count - 1
        }
        mutating func gamma(_ shape: Double) -> Double {
            if shape < 1 { return gamma(shape + 1) * pow(max(1e-12, uniform()), 1 / shape) }
            let d = shape - 1 / 3.0, c = 1 / sqrt(9 * d)
            for _ in 0..<100 {
                let x = normal(0, 1), v = pow(1 + c * x, 3)
                guard v > 0 else { continue }
                let u = max(1e-12, uniform())
                if u < 1 - 0.0331 * pow(x, 4) || log(u) < 0.5 * x * x + d * (1 - v + log(v)) { return d * v }
            }
            return shape
        }
        mutating func beta(made: Double, attempts: Double, prior: Double, strength: Double) -> Double {
            let a = gamma(max(0, min(attempts, made)) + prior * strength)
            let b = gamma(max(0, attempts - made) + (1 - prior) * strength)
            return a / max(1e-12, a + b)
        }
        mutating func choose(_ players: [Profile], _ weight: (Profile) -> Double) -> Profile {
            let weights = players.map { max(0, weight($0)) }, total = weights.reduce(0, +)
            var remainder = uniform() * (total > 0 ? total : Double(players.count))
            for (i, player) in players.enumerated() { remainder -= total > 0 ? weights[i] : 1; if remainder < 0 { return player } }
            return players.last!
        }
    }
    private struct Engine {
        let teams: [Team]
        var random: Random
        let record: Bool
        let observed: SimulationLiveSnapshot?
        var values: [[String: [String: Double]]] = [[:], [:]]
        var scores = [0, 0]
        var plays: [Play] = []
        var pending: [Delta] = []
        var sequence = 0
        var drawnRates: [String: Double] = [:]
        mutating func rate(_ actor: Profile, _ key: String, made: Double, attempts: Double, prior: Double, strength: Double) -> Double {
            let id = actor.id + "/" + key
            if let value = drawnRates[id] { return value }
            let value = random.beta(made: made, attempts: max(0, attempts), prior: prior, strength: strength)
            drawnRates[id] = value
            return value
        }
        mutating func add(_ side: Int, _ player: Profile, _ group: String, _ stat: String, _ amount: Double) {
            let id = group + ":" + player.id
            values[side][id, default: [:]][stat, default: 0] += amount
            if record { pending.append(.init(home: side == 1, playerID: player.id, group: group, stat: stat, amount: amount)) }
        }
        mutating func maximum(_ side: Int, _ player: Profile, _ group: String, _ stat: String, _ value: Double) {
            let previous = values[side][group + ":" + player.id]?[stat] ?? 0
            if value > previous { add(side, player, group, stat, value - previous) }
        }
        mutating func initialize(_ side: Int, _ players: [Profile], _ group: String) {
            guard let schema = PlayerSimulation.groups(teams[side].identity.sport).first(where: { $0.id == group }) else { return }
            if group == "players" { for player in players where values[side][group + ":" + player.id]?["offensiveRebounds"] == nil { values[side][group + ":" + player.id, default: [:]]["offensiveRebounds"] = 0 } }
            for player in players { for stat in schema.stats where !(group == "skaters" && ["hits", "blockedShots", "penaltyMinutes"].contains(stat.id)) { if values[side][group + ":" + player.id]?[stat.id] == nil { values[side][group + ":" + player.id, default: [:]][stat.id] = 0 } } }
        }
        mutating func emit(_ side: Int, _ actor: Profile, kind: String, title: String, period: Int, clock: String, points: Int = 0, yards: Int = 0, secondary: String? = nil) {
            scores[side] += points
            if record { plays.append(.init(id: sequence, home: side == 1, actorID: actor.id, secondaryID: secondary, kind: kind, title: title, period: period, clock: clock, points: points, yards: yards, deltas: pending)); pending.removeAll(keepingCapacity: true) }
            sequence += 1
        }
        func clock(_ remaining: Int) -> String { String(format: "%d:%02d", max(0, remaining) / 60, max(0, remaining) % 60) }
        func result() -> Game {
            func lines(_ side: Int) -> [PlayerSimulation.Line] {
                values[side].compactMap { id, stats in
                    let parts = id.split(separator: ":")
                    guard parts.count == 2 else { return nil }
                    let name = teams[side].profiles.first(where: { $0.id == parts[1] })?.player.name
                        ?? observed?.lines[side].first(where: { $0.id == id })?.name
                    guard let name else { return nil }
                    return .init(playerID: String(parts[1]), name: name, group: String(parts[0]), values: stats)
                }.sorted { $0.id < $1.id }
            }
            return Game(away: lines(0), home: lines(1), awayScore: scores[0], homeScore: scores[1], plays: plays, initialLines: observed?.lines ?? [[], []], initialScores: observed?.scores ?? [0, 0])
        }

        mutating func basketball() throws {
            let rotations = teams.map { Array($0.profiles.filter { $0.mean("players", "minutes") > 0 }.sorted { $0.mean("players", "minutes") > $1.mean("players", "minutes") }.prefix(10)) }
            guard rotations.allSatisfy({ $0.count >= 5 && $0.reduce(0, { $0 + $1.mean("players", "fieldGoalsAttempted") }) > 0 }) else { throw Failure.lineup }
            for side in 0...1 { initialize(side, rotations[side], "players") }
            guard let awayPace = teams[0].possessions, let homePace = teams[1].possessions else { throw Failure.history }
            let startElapsed = observed?.elapsed ?? 0
            let minutes = Double(max(0, 2880 - startElapsed)) / 60
            guard minutes > 0 else { return }
            let fullPossessions = min(260, max(140, Int(random.normal(awayPace + homePace, 12).rounded())))
            let possessions = max(1, Int((Double(fullPossessions) * minutes / 48).rounded()))
            var side = observed?.possession ?? (random.yes(0.5) ? 0 : 1)
            var onCourt: [[Profile]] = [[], []]
            var allocated: [[String: Double]] = [[:], [:]]
            let targets = rotations.map { roster in
                let total = roster.reduce(0.0) { $0 + $1.mean("players", "minutes") }
                return Dictionary(uniqueKeysWithValues: roster.map { ($0.id, $0.mean("players", "minutes") * minutes * 5 / max(1, total)) })
            }
            for index in 0..<possessions {
                // Allocate court time by outstanding minute targets instead of
                // sampling five names repeatedly (which overplayed the bench).
                if index % 6 == 0 {
                    for team in 0...1 {
                        let progress = min(1, Double(index + 6) / Double(possessions))
                        let ranked = rotations[team].map { player in
                            (player, (targets[team][player.id] ?? 0) * progress - (allocated[team][player.id] ?? 0) + random.uniform() * 0.05)
                        }.sorted { $0.1 > $1.1 }
                        onCourt[team] = ranked.prefix(5).map(\.0)
                    }
                }
                let other = 1 - side, offense = onCourt[side], defense = onCourt[other]
                let elapsed = startElapsed + index * (2880 - startElapsed) / possessions, period = min(4, elapsed / 720 + 1), display = clock(720 - elapsed % 720)
                for team in 0...1 { for active in onCourt[team] { allocated[team][active.id, default: 0] += minutes / Double(possessions) } }
                var retained = true, attempt = 0
                while retained && attempt < 20 {
                    retained = false
                    let actor = random.choose(offense) { ($0.mean("players", "fieldGoalsAttempted") + 0.5 * $0.mean("players", "freeThrowsAttempted") + $0.mean("players", "turnovers")) / max(1, $0.mean("players", "minutes")) }
                    // Only the first action consumes this possession's court time.
                    // Zero minute deltas retain the on-court identities for replay.
                    for team in 0...1 { for active in onCourt[team] { add(team, active, "players", "minutes", attempt == 0 ? minutes / Double(possessions) : 0) } }
                    let shots = actor.mean("players", "fieldGoalsAttempted"), trips = 0.5 * actor.mean("players", "freeThrowsAttempted"), turnovers = actor.mean("players", "turnovers")
                    let choice = random.uniform() * max(1, shots + trips + turnovers)
                    if choice < turnovers {
                        add(side, actor, "players", "turnovers", 1)
                        let thief = random.choose(defense) { $0.mean("players", "steals") }
                        let steals = defense.reduce(0.0) { $0 + $1.mean("players", "steals") }, lost = offense.reduce(0.0) { $0 + $1.mean("players", "turnovers") }
                        if random.yes(steals / max(1, lost)) { add(other, thief, "players", "steals", 1) }
                        emit(side, actor, kind: "turnover", title: actor.player.name + " turns it over", period: period, clock: display)
                    } else if choice < turnovers + trips {
                        let probability = rate(actor, "ft", made: actor.total("players", "freeThrowsMade"), attempts: actor.total("players", "freeThrowsAttempted"), prior: 0.78, strength: 20)
                        let made = (0..<2).reduce(0) { sum, _ in sum + (random.yes(probability) ? 1 : 0) }
                        add(side, actor, "players", "freeThrowsAttempted", 2); add(side, actor, "players", "freeThrowsMade", Double(made)); add(side, actor, "players", "points", Double(made))
                        emit(side, actor, kind: "freeThrow", title: "\(actor.player.name) · \(made)/2 at the line", period: period, clock: display, points: made)
                    } else {
                        let three = random.yes(actor.rate("players", "threePointFieldGoalsAttempted", "fieldGoalsAttempted"))
                        let tries = three ? actor.total("players", "threePointFieldGoalsAttempted") : actor.total("players", "fieldGoalsAttempted") - actor.total("players", "threePointFieldGoalsAttempted")
                        let makes = three ? actor.total("players", "threePointFieldGoalsMade") : actor.total("players", "fieldGoalsMade") - actor.total("players", "threePointFieldGoalsMade")
                        let probability = rate(actor, three ? "3p" : "2p", made: makes, attempts: tries, prior: three ? 0.36 : 0.54, strength: 40)
                        let made = random.yes(probability)
                        add(side, actor, "players", "fieldGoalsAttempted", 1); if three { add(side, actor, "players", "threePointFieldGoalsAttempted", 1) }
                        var assistID: String?
                        if made {
                            add(side, actor, "players", "fieldGoalsMade", 1); if three { add(side, actor, "players", "threePointFieldGoalsMade", 1) }; add(side, actor, "players", "points", three ? 3 : 2)
                            let helpers = offense.filter { $0.id != actor.id }, assists = offense.reduce(0.0) { $0 + $1.mean("players", "assists") }, madeHistory = offense.reduce(0.0) { $0 + $1.mean("players", "fieldGoalsMade") }
                            if random.yes(assists / max(1, madeHistory)) { let helper = random.choose(helpers) { $0.mean("players", "assists") }; add(side, helper, "players", "assists", 1); assistID = helper.id }
                        } else {
                            let blocker = random.choose(defense) { $0.mean("players", "blocks") }
                            let blocks = defense.reduce(0.0) { $0 + $1.mean("players", "blocks") }, opposingShots = offense.reduce(0.0) { $0 + $1.mean("players", "fieldGoalsAttempted") }
                            if random.yes(blocks / max(1, opposingShots)) { add(other, blocker, "players", "blocks", 1) }
                            let offensiveBoards = offense.reduce(0.0) { $0 + $1.mean("players", "offensiveRebounds") }
                            let defensiveBoards = defense.reduce(0.0) { $0 + max(0, $1.mean("players", "rebounds") - $1.mean("players", "offensiveRebounds")) }
                            retained = random.yes(offensiveBoards / max(1, offensiveBoards + defensiveBoards)) && attempt < 19
                            let rebounder = random.choose(retained ? offense : defense) { retained ? $0.mean("players", "offensiveRebounds") : max(0, $0.mean("players", "rebounds") - $0.mean("players", "offensiveRebounds")) }
                            add(retained ? side : other, rebounder, "players", "rebounds", 1)
                            if retained { add(side, rebounder, "players", "offensiveRebounds", 1) }
                        }
                        emit(side, actor, kind: three ? "three" : "shot", title: actor.player.name + (made ? (three ? " makes a three" : " scores") : retained ? " misses · offensive rebound" : " misses"), period: period, clock: display, points: made ? (three ? 3 : 2) : 0, secondary: assistID)
                    }
                    attempt += 1
                }
                side = other
            }
        }

        mutating func hockey() throws {
            let skaters = teams.map { Array($0.profiles.filter { $0.mean("skaters", "timeOnIce") > 0 }.sorted { $0.mean("skaters", "timeOnIce") > $1.mean("skaters", "timeOnIce") }.prefix(18)) }
            let goalies = teams.enumerated().map { side, team -> Profile? in
                if let id = observed?.activeGoalies[side] { return team.profiles.first { $0.id == id && $0.mean("goalies", "timeOnIce") > 0 } }
                return team.profiles.filter { $0.mean("goalies", "timeOnIce") > 0 }.max { $0.mean("goalies", "timeOnIce") < $1.mean("goalies", "timeOnIce") }
            }
            guard skaters.allSatisfy({ $0.count >= 5 }), let awayGoalie = goalies[0], let homeGoalie = goalies[1] else { throw Failure.lineup }
            let selectedGoalies = [awayGoalie, homeGoalie]
            for side in 0...1 { initialize(side, skaters[side], "skaters"); initialize(side, [selectedGoalies[side]], "goalies") }
            let startElapsed = observed?.elapsed ?? 0, remaining = max(0, 3600 - (observed?.elapsed ?? 0))
            guard remaining > 0 else { return }
            var attempts: [(Int, Profile)] = []
            for side in 0...1 { for player in skaters[side] { for _ in 0..<min(20, random.count(player.mean("skaters", "shotsTotal") * Double(remaining) / 3600)) { attempts.append((side, player)) } } }
            guard !attempts.isEmpty else { return }
            // Shuffle ownership and actors without changing their shot totals.
            for index in attempts.indices.reversed() where index > 0 { let other = Int(random.uniform() * Double(index + 1)); attempts.swapAt(index, other) }
            for (index, entry) in attempts.enumerated() {
                let (side, actor) = entry, goalie = selectedGoalies[1 - side]
                let elapsed = startElapsed + index * remaining / attempts.count, period = min(3, elapsed / 1200 + 1), display = clock(1200 - elapsed % 1200)
                let conversion = rate(actor, "goals", made: actor.total("skaters", "goals"), attempts: actor.total("skaters", "shotsTotal"), prior: 0.1, strength: 30), allowed = rate(goalie, "goalsAllowed", made: goalie.total("goalies", "goalsAgainst"), attempts: goalie.total("goalies", "shotsAgainst"), prior: 0.1, strength: 100)
                let goal = random.yes((conversion + allowed) / 2)
                add(side, actor, "skaters", "shotsTotal", 1); add(1 - side, goalie, "goalies", "shotsAgainst", 1)
                if goal {
                    add(side, actor, "skaters", "goals", 1); add(1 - side, goalie, "goalies", "goalsAgainst", 1)
                    var helpers = skaters[side].filter { $0.id != actor.id }
                    let assists = skaters[side].reduce(0.0) { $0 + $1.total("skaters", "assists") }, goals = skaters[side].reduce(0.0) { $0 + $1.total("skaters", "goals") }
                    for _ in 0..<2 where random.yes(assists / max(1, 2 * goals)) {
                        let helper = random.choose(helpers) { $0.mean("skaters", "assists") }; add(side, helper, "skaters", "assists", 1); helpers.removeAll { $0.id == helper.id }
                    }
                } else { add(1 - side, goalie, "goalies", "saves", 1) }
                for team in 0...1 {
                    add(team, selectedGoalies[team], "goalies", "timeOnIce", Double(remaining) / Double(attempts.count))
                    let totalTime = skaters[team].reduce(0.0) { $0 + $1.mean("skaters", "timeOnIce") }
                    for player in skaters[team] { add(team, player, "skaters", "timeOnIce", Double(remaining * 5) * player.mean("skaters", "timeOnIce") / max(1, totalTime) / Double(attempts.count)) }
                }
                emit(side, actor, kind: "hockeyShot", title: actor.player.name + (goal ? " scores" : " · saved by " + goalie.player.name), period: period, clock: display, points: goal ? 1 : 0, secondary: goalie.id)
            }
        }

        mutating func football() throws {
            let quarterbacks = teams.map { $0.profiles.filter { $0.mean("passing", "passingAttempts") > 0 }.max { $0.mean("passing", "passingAttempts") < $1.mean("passing", "passingAttempts") } }
            let runners = teams.map { $0.profiles.filter { $0.mean("rushing", "rushingAttempts") > 0 } }
            let receivers = teams.map { $0.profiles.filter { $0.mean("receiving", "receivingTargets") > 0 || $0.mean("receiving", "receptions") > 0 } }
            let kickers = teams.map { $0.profiles.filter { $0.mean("kicking", "fieldGoalAttempts") > 0 }.max { $0.mean("kicking", "fieldGoalAttempts") < $1.mean("kicking", "fieldGoalAttempts") } }
            guard let aq = quarterbacks[0], let hq = quarterbacks[1], runners.allSatisfy({ !$0.isEmpty }), receivers.allSatisfy({ !$0.isEmpty }) else { throw Failure.lineup }
            let qbs = [aq, hq]
            for side in 0...1 { initialize(side, [qbs[side]], "passing"); initialize(side, runners[side], "rushing"); initialize(side, receivers[side], "receiving"); if let kicker = kickers[side] { initialize(side, [kicker], "kicking") } }
            var side = observed?.possession ?? (random.yes(0.5) ? 0 : 1), yard = observed?.yard ?? 25, down = observed?.down ?? 1, needed = observed?.distance ?? 10, elapsed = observed?.elapsed ?? 0
            while elapsed < 3600 {
                let qb = qbs[side], period = min(4, elapsed / 900 + 1), display = clock(900 - elapsed % 900)
                let rushes = runners[side].reduce(0.0) { $0 + $1.mean("rushing", "rushingAttempts") }
                var possessionEnds = false
                if down == 4 {
                    if yard >= 60, let kicker = kickers[side] {
                        let good = random.yes(kicker.rate("kicking", "fieldGoalsMade", "fieldGoalAttempts"))
                        add(side, kicker, "kicking", "fieldGoalAttempts", 1); add(side, kicker, "kicking", "fieldGoalsMade", good ? 1 : 0); add(side, kicker, "kicking", "totalKickingPoints", good ? 3 : 0)
                        emit(side, kicker, kind: "fieldGoal", title: "\(kicker.player.name) · \(117 - yard)-yard field goal \(good ? "good" : "missed")", period: period, clock: display, points: good ? 3 : 0, yards: 117 - yard)
                    } else { emit(side, qb, kind: "punt", title: teams[side].identity.abbreviation + " punts", period: period, clock: display) }
                    possessionEnds = true
                } else if random.yes(qb.mean("passing", "passingAttempts") / max(1, qb.mean("passing", "passingAttempts") + rushes)) {
                    let receiver = random.choose(receivers[side]) { max($0.mean("receiving", "receivingTargets"), $0.mean("receiving", "receptions")) }
                    add(side, qb, "passing", "passingAttempts", 1); add(side, receiver, "receiving", "receivingTargets", 1)
                    let interception = rate(qb, "int", made: qb.total("passing", "interceptions"), attempts: qb.total("passing", "passingAttempts"), prior: 0.025, strength: 75)
                    let completion = rate(qb, "completion", made: qb.total("passing", "completions"), attempts: qb.total("passing", "passingAttempts"), prior: 0.65, strength: 75)
                    if random.yes(interception) {
                        add(side, qb, "passing", "interceptions", 1)
                        emit(side, qb, kind: "interception", title: qb.player.name + " · intercepted", period: period, clock: display); possessionEnds = true
                    } else if random.yes(completion) {
                        let mean = receiver.total("receiving", "receivingYards") / max(1, receiver.total("receiving", "receptions"))
                        let gain = min(100 - yard, max(1 - yard, max(-5, Int(random.normal(mean, max(3, abs(mean) * 0.65)).rounded()))))
                        let td = yard + gain >= 100
                        add(side, qb, "passing", "completions", 1); add(side, qb, "passing", "passingYards", Double(gain)); add(side, receiver, "receiving", "receptions", 1); add(side, receiver, "receiving", "receivingYards", Double(gain))
                        maximum(side, receiver, "receiving", "longReception", Double(gain))
                        if td { add(side, qb, "passing", "passingTouchdowns", 1); add(side, receiver, "receiving", "receivingTouchdowns", 1) }
                        emit(side, qb, kind: "pass", title: "\(qb.player.name) → \(receiver.player.name), \(gain) yards" + (td ? " · TD" : ""), period: period, clock: display, points: td ? 6 : 0, yards: gain, secondary: receiver.id)
                        yard += gain; needed -= gain; possessionEnds = td
                        if td { extraPoint(side, qb: qb, kicker: kickers[side], period: period, clock: display) }
                    } else { emit(side, qb, kind: "incomplete", title: qb.player.name + " · incomplete", period: period, clock: display, secondary: receiver.id) }
                    down += 1
                } else {
                    let runner = random.choose(runners[side]) { $0.mean("rushing", "rushingAttempts") }
                    let mean = runner.total("rushing", "rushingYards") / max(1, runner.total("rushing", "rushingAttempts"))
                    let gain = min(100 - yard, max(1 - yard, max(-8, Int(random.normal(mean, max(3, abs(mean))).rounded()))))
                    let td = yard + gain >= 100
                    add(side, runner, "rushing", "rushingAttempts", 1); add(side, runner, "rushing", "rushingYards", Double(gain)); if td { add(side, runner, "rushing", "rushingTouchdowns", 1) }
                    maximum(side, runner, "rushing", "longRushing", Double(gain))
                    emit(side, runner, kind: "run", title: "\(runner.player.name) runs for \(gain) yards" + (td ? " · TD" : ""), period: period, clock: display, points: td ? 6 : 0, yards: gain)
                    yard += gain; needed -= gain; down += 1; possessionEnds = td
                    if td { extraPoint(side, qb: qb, kicker: kickers[side], period: period, clock: display) }
                }
                if possessionEnds { side = 1 - side; yard = 25; down = 1; needed = 10 }
                else if needed <= 0 { down = 1; needed = min(10, 100 - yard) }
                elapsed += 20 + Int(random.uniform() * 20)
            }
        }
        mutating func extraPoint(_ side: Int, qb: Profile, kicker: Profile?, period: Int, clock: String) {
            if let kicker {
                let made = random.yes(kicker.rate("kicking", "extraPointsMade", "extraPointAttempts"))
                add(side, kicker, "kicking", "extraPointAttempts", 1); add(side, kicker, "kicking", "extraPointsMade", made ? 1 : 0); add(side, kicker, "kicking", "totalKickingPoints", made ? 1 : 0)
                emit(side, kicker, kind: "extraPoint", title: kicker.player.name + (made ? " · extra point good" : " · extra point missed"), period: period, clock: clock, points: made ? 1 : 0)
            } else {
                let good = random.yes(qb.rate("passing", "completions", "passingAttempts"))
                emit(side, qb, kind: "twoPoint", title: teams[side].identity.abbreviation + (good ? " · two-point conversion" : " · conversion fails"), period: period, clock: clock, points: good ? 2 : 0)
            }
        }

        mutating func baseball() throws {
            var lineups = teams.map { Array($0.profiles.filter { $0.mean("batting", "atBats") > 0 }.sorted { $0.mean("batting", "atBats") > $1.mean("batting", "atBats") }.prefix(9)) }
            var staff = teams.map { $0.profiles.filter { $0.mean("pitching", "fullInnings.partInnings") > 0 }.sorted { $0.mean("pitching", "fullInnings.partInnings") > $1.mean("pitching", "fullInnings.partInnings") } }
            if let live = observed {
                for side in 0...1 {
                    lineups[side] = live.battingOrders[side].compactMap { id in teams[side].profiles.first { $0.id == id } }
                    guard let current = live.activePitchers[side] else { throw Failure.lineup }
                    let used = Set(live.lines[side].filter { $0.group == "pitching" && $0.playerID != current }.map(\.playerID))
                    staff[side] = staff[side].filter { !used.contains($0.id) }.sorted { if ($0.id == current) != ($1.id == current) { return $0.id == current }; return $0.mean("pitching", "fullInnings.partInnings") > $1.mean("pitching", "fullInnings.partInnings") }
                    guard staff[side].first?.id == current else { throw Failure.lineup }
                }
            }
            guard lineups.allSatisfy({ $0.count == 9 }), staff.allSatisfy({ !$0.isEmpty }) else { throw Failure.lineup }
            for side in 0...1 { initialize(side, lineups[side], "batting") }
            var battingIndex = observed?.nextBatters ?? [0, 0], pitcherIndex = [0, 0], pitcherOuts = [0, 0]
            if let live = observed { for side in 0...1 { pitcherOuts[side] = Int(live.lines[side].first { $0.group == "pitching" && $0.playerID == staff[side][0].id }?.values["fullInnings.partInnings"] ?? 0) } }
            let firstInning = observed?.period ?? 1
            for inning in firstInning...9 {
                for side in 0...1 {
                    if inning == firstInning && side < (observed?.battingSide ?? 0) { continue }
                    if inning == 9 && side == 1 && scores[1] > scores[0] { continue }
                    let checkpoint = observed != nil && inning == firstInning && side == observed?.battingSide
                    var outs = checkpoint ? observed?.outs ?? 0 : 0, bases: [(runner: Profile, pitcher: Profile)?] = [nil, nil, nil], appearances = 0
                    if checkpoint, let live = observed, outs < 3 {
                        for base in 0..<3 {
                            guard let id = live.bases[base] else { continue }
                            guard let runner = teams[side].profiles.first(where: { $0.id == id }), let responsibleID = live.basePitchers[base], let responsible = teams[1 - side].profiles.first(where: { $0.id == responsibleID }) else { throw Failure.lineup }
                            bases[base] = (runner, responsible)
                        }
                    }
                    while outs < 3 && appearances < 60 {
                        let other = 1 - side, batter = lineups[side][battingIndex[side] % 9]
                        let pitcher = staff[other][min(pitcherIndex[other], staff[other].count - 1)]
                        if values[other]["pitching:" + pitcher.id] == nil { initialize(other, [pitcher], "pitching") }
                        guard batter.samples["batting.doubles"] != nil, batter.samples["batting.triples"] != nil else { throw Failure.history }
                        let ab = batter.total("batting", "atBats"), walks = batter.total("batting", "walks"), pa = max(1, ab + walks)
                        let batterWalk = rate(batter, "bb", made: walks, attempts: pa, prior: 0.085, strength: 40)
                        let batterHit = rate(batter, "hit", made: batter.total("batting", "hits"), attempts: ab, prior: 0.245, strength: 40)
                        let pitcherAB = pitcher.total("pitching", "fullInnings.partInnings") + pitcher.total("pitching", "hits")
                        let pitcherHit = rate(pitcher, "hitAllowed", made: pitcher.total("pitching", "hits"), attempts: pitcherAB, prior: 0.245, strength: 60)
                        let pitcherWalk = rate(pitcher, "bbAllowed", made: pitcher.total("pitching", "walks"), attempts: pitcherAB + pitcher.total("pitching", "walks"), prior: 0.085, strength: 60)
                        let walk = random.yes((batterWalk + pitcherWalk) / 2), hit = !walk && random.yes((batterHit + pitcherHit) / 2)
                        let hitDraw = random.uniform() * max(1, batter.total("batting", "hits"))
                        let basesHit = !hit ? 0 : hitDraw < batter.total("batting", "homeRuns") ? 4 : hitDraw < batter.total("batting", "homeRuns") + batter.total("batting", "triples") ? 3 : hitDraw < batter.total("batting", "homeRuns") + batter.total("batting", "triples") + batter.total("batting", "doubles") ? 2 : 1
                        let homer = basesHit == 4
                        let strikeout = !walk && !hit && random.yes(batter.total("batting", "strikeouts") / max(1, ab - batter.total("batting", "hits")))
                        let pitchCount = 3 + Int(random.uniform() * 4)
                        add(other, pitcher, "pitching", "pitches", Double(pitchCount))
                        var runs = 0
                        if walk {
                            add(side, batter, "batting", "walks", 1); add(other, pitcher, "pitching", "walks", 1)
                            if bases[0] != nil {
                                if bases[1] != nil { if let runner = bases[2] { add(side, runner.runner, "batting", "runs", 1); add(other, runner.pitcher, "pitching", "earnedRuns", 1); runs += 1 }; bases[2] = bases[1] }
                                bases[1] = bases[0]
                            }
                            bases[0] = (batter, pitcher)
                        } else {
                            add(side, batter, "batting", "atBats", 1)
                            if hit {
                                add(side, batter, "batting", "hits", 1); add(other, pitcher, "pitching", "hits", 1)
                                if homer {
                                    for runner in bases.compactMap({ $0 }) { add(side, runner.runner, "batting", "runs", 1); add(other, runner.pitcher, "pitching", "earnedRuns", 1); runs += 1 }
                                    add(side, batter, "batting", "runs", 1); add(other, pitcher, "pitching", "earnedRuns", 1); runs += 1; bases = [nil, nil, nil]
                                    add(side, batter, "batting", "homeRuns", 1); add(other, pitcher, "pitching", "homeRuns", 1)
                                } else {
                                    for base in (0..<3).reversed() {
                                        if let runner = bases[base] {
                                            if base + basesHit >= 3 { add(side, runner.runner, "batting", "runs", 1); add(other, runner.pitcher, "pitching", "earnedRuns", 1); runs += 1 }
                                            else { bases[base + basesHit] = runner }
                                            bases[base] = nil
                                        }
                                    }
                                    bases[basesHit - 1] = (batter, pitcher)
                                    if basesHit == 2 { add(side, batter, "batting", "doubles", 1) }
                                    if basesHit == 3 { add(side, batter, "batting", "triples", 1) }
                                }
                            } else {
                                outs += 1; pitcherOuts[other] += 1; add(other, pitcher, "pitching", "fullInnings.partInnings", 1)
                                if strikeout { add(side, batter, "batting", "strikeouts", 1); add(other, pitcher, "pitching", "strikeouts", 1) }
                            }
                        }
                        if runs > 0 { add(side, batter, "batting", "RBIs", Double(runs)) }
                        let kind = homer ? "homeRun" : walk ? "walk" : hit ? (basesHit == 3 ? "triple" : basesHit == 2 ? "double" : "single") : strikeout ? "strikeout" : "out"
                        let action = homer ? "homers" : walk ? "walks" : hit ? (basesHit == 3 ? "triples" : basesHit == 2 ? "doubles" : "singles") : strikeout ? "strikes out" : "is out"
                        emit(side, batter, kind: kind, title: batter.player.name + " " + action + (runs > 0 ? " · \(runs) run\(runs == 1 ? "" : "s")" : ""), period: inning, clock: (side == 0 ? "Top" : "Bottom") + " · \(outs) out\(outs == 1 ? "" : "s")", points: runs, secondary: pitcher.id)
                        battingIndex[side] += 1; appearances += 1
                        let limit = max(3, Int(pitcher.mean("pitching", "fullInnings.partInnings").rounded()))
                        if pitcherOuts[other] >= limit && pitcherIndex[other] + 1 < staff[other].count { pitcherIndex[other] += 1; pitcherOuts[other] = 0 }
                        if inning == 9 && side == 1 && scores[1] > scores[0] { break }
                    }
                    guard appearances < 60 else { throw Failure.incomplete }
                }
            }
        }
    }
}


extension SportPlaySimulation.Game {
    func through(_ completed: Int) -> Self {
        let completed = min(max(0, completed), plays.count)
        var values: [[String: [String: Double]]] = [[:], [:]], scores = initialScores
        for (side, lines) in [away, home].enumerated() { for line in lines {
            values[side][line.id] = line.values.mapValues { _ in 0 }
        } }
        for (side, lines) in initialLines.enumerated() { for line in lines { values[side][line.id] = line.values } }
        for play in plays.prefix(completed) {
            scores[play.home ? 1 : 0] += play.points
            for delta in play.deltas { values[delta.home ? 1 : 0][delta.group + ":" + delta.playerID, default: [:]][delta.stat, default: 0] += delta.amount }
        }
        func lines(_ templates: [PlayerSimulation.Line], side: Int) -> [PlayerSimulation.Line] {
            templates.map { .init(playerID: $0.playerID, name: $0.name, group: $0.group, values: values[side][$0.id] ?? [:]) }
        }
        return .init(away: lines(away, side: 0), home: lines(home, side: 1), awayScore: scores[0], homeScore: scores[1], plays: Array(plays.prefix(completed)), initialLines: initialLines, initialScores: initialScores)
    }
}
