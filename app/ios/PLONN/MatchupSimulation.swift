import Foundation
import CoreFoundation

/// A score-scenario explorer, not a trained win-probability or betting model.
/// Each draw preserves the observed integer score distribution of its sport.
struct MatchupSimulation: Sendable {
    struct Game: Equatable, Sendable, Identifiable {
        let id: String
        let date: Date
        let season: Int
        let scored: Int
        let allowed: Int
        let home: Bool
    }
    struct History: Sendable {
        let team: InterestTeam
        let games: [Game]
        let partial: Bool
    }
    struct Score: Equatable, Sendable {
        let away: Int
        let home: Int
        var total: Int { away + home }
        var margin: Int { home - away }
    }
    enum Outcome: Int, CaseIterable, Sendable {
        case away, home, level
        init(_ score: Score) { self = score.away > score.home ? .away : score.home > score.away ? .home : .level }
    }
    struct RepresentativeScenario: Sendable {
        let index: Int
        let outcome: Outcome
        let share: Double
        let tiedForTop: Bool
    }
    /// A representative replay from the most frequent outcome, never a probability
    /// assigned to an individual score or play sequence. Tied outcomes imply no edge.
    static func representative(in scores: [Score], candidates: [Score]? = nil) -> RepresentativeScenario? {
        guard !scores.isEmpty else { return nil }
        let candidates = candidates ?? scores
        let counts = Dictionary(grouping: scores, by: Outcome.init).mapValues(\.count)
        let maximum = counts.values.max() ?? 0
        let leading = Set(Outcome.allCases.filter { counts[$0] == maximum })
        let pool = scores.filter { leading.contains(Outcome($0)) }
        let awayMean = pool.reduce(0.0) { $0 + Double($1.away) } / Double(pool.count)
        let homeMean = pool.reduce(0.0) { $0 + Double($1.home) } / Double(pool.count)
        let eligible = candidates.indices.filter { leading.contains(Outcome(candidates[$0])) }
        func distance(_ index: Int) -> Double {
            abs(Double(candidates[index].away) - awayMean) + abs(Double(candidates[index].home) - homeMean)
        }
        guard let index = eligible.min(by: { distance($0) < distance($1) }) else { return nil }
        return .init(index: index, outcome: Outcome(candidates[index]), share: Double(maximum) / Double(scores.count), tiedForTop: leading.count > 1)
    }
    struct Summary: Sendable {
        let average: Double
        let low: Int
        let high: Int
        init(_ values: [Int]) {
            let sorted = values.sorted()
            average = Double(values.reduce(0, +)) / Double(values.count)
            low = sorted[Int(Double(sorted.count - 1) * 0.1)]
            high = sorted[Int(Double(sorted.count - 1) * 0.9)]
        }
    }
    /// A bounded number of equal-width columns, preserving every sampled score.
    struct MarginHistogram {
        struct Bin: Identifiable {
            let center: Int
            let count: Int
            var id: Int { center }
        }
        let bins: [Bin]
        let width: Int
        let extent: Int
        let median: Double
        init(_ margins: [Int]) {
            let sorted = margins.sorted()
            let maximum = max(1, sorted.map { abs($0) }.max() ?? 1)
            width = max(1, Int(ceil(Double(maximum * 2) / 25)))
            let width = self.width
            let counts = Dictionary(grouping: sorted) { Int((Double($0) / Double(width)).rounded()) * width }.mapValues(\.count)
            bins = counts.keys.sorted().map { Bin(center: $0, count: counts[$0] ?? 0) }
            extent = max(width, (bins.map { abs($0.center) }.max() ?? width) + width)
            if sorted.isEmpty { median = 0 }
            else { median = (Double(sorted[(sorted.count - 1) / 2]) + Double(sorted[sorted.count / 2])) / 2 }
        }
    }
    enum Failure: Error { case selection, source, history }
    let away: History
    let home: History
    let scores: [Score]
    let generated: Date
    let seed: UInt64
    var awaySummary: Summary { Summary(scores.map(\.away)) }
    var homeSummary: Summary { Summary(scores.map(\.home)) }
    var totalSummary: Summary { Summary(scores.map(\.total)) }
    var homeAhead: Double { Double(scores.filter { $0.home > $0.away }.count) / Double(scores.count) }
    var awayAhead: Double { Double(scores.filter { $0.away > $0.home }.count) / Double(scores.count) }
    var level: Double { Double(scores.filter { $0.home == $0.away }.count) / Double(scores.count) }
    var unit: String { Self.unit(away.team.sport) }
    static func unit(_ sport: InterestSport) -> String { sport == .nhl ? "goals" : sport == .mlb ? "runs" : "points" }
    static func season(_ sport: InterestSport, at date: Date) -> Int {
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let year = calendar.component(.year, from: date)
        return [.nba, .nhl].contains(sport) && calendar.component(.month, from: date) >= 7 ? year + 1 : year
    }
    static func run(away: History, home: History, now: Date = Date(), seed: UInt64, count: Int = 5_000) throws -> Self {
        guard away.team.isValid, home.team.isValid, away.team.sport == home.team.sport,
              away.team.id != home.team.id, (100...10_000).contains(count) else { throw Failure.selection }
        func clean(_ history: History) throws -> History {
            guard Set(history.games.map(\.id)).count == history.games.count,
                  history.games.allSatisfy({ $0.scored >= 0 && $0.allowed >= 0 && $0.scored <= 300 && $0.allowed <= 300 }) else { throw Failure.source }
            let games = Array(history.games.filter { $0.date < now && $0.date > now.addingTimeInterval(-540 * 86400) }
                .sorted { $0.date > $1.date }.prefix(20))
            guard games.count >= 5, let last = games.first, now.timeIntervalSince(last.date) < 240 * 86400 else { throw Failure.history }
            return History(team: history.team, games: games, partial: history.partial)
        }
        let a = try clean(away), h = try clean(home)
        // Equal-weight offense/defense mixture. Do not average two individual
        // score draws: that would artificially halve their variance.
        // No fabricated home advantage, roster adjustment, overtime or prices.
        let aFor = a.games.map(\.scored), aAgainst = a.games.map(\.allowed)
        let hFor = h.games.map(\.scored), hAgainst = h.games.map(\.allowed)
        var random = Generator(state: seed)
        var scores: [Score] = []; scores.reserveCapacity(count)
        for _ in 0..<count {
            let aPool = random.next() & 1 == 0 ? aFor : hAgainst
            let hPool = random.next() & 1 == 0 ? hFor : aAgainst
            scores.append(Score(away: aPool[random.index(aPool.count)], home: hPool[random.index(hPool.count)]))
        }
        return Self(away: a, home: h, scores: scores, generated: now, seed: seed)
    }
    private struct Generator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9e3779b97f4a7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            return z ^ (z >> 31)
        }
        mutating func index(_ count: Int) -> Int { Int(next() % UInt64(count)) }
    }
}

struct MatchupSimulationSource: Sendable {
    typealias Reader = @Sendable (URL) async throws -> Data
    var read: Reader = { try await PlayerHistoryService.shared.read($0) }

    func history(_ team: InterestTeam, cutoff: Date = Date()) async throws -> MatchupSimulation.History {
        guard team.isValid else { throw MatchupSimulation.Failure.selection }
        let year = MatchupSimulation.season(team.sport, at: cutoff)
        let parts = await withTaskGroup(of: [MatchupSimulation.Game]?.self, returning: [[MatchupSimulation.Game]?].self) { group in
            for season in [year, year - 1] {
                for phase in [2, 3] {
                    group.addTask {
                        let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(team.sport.leaguePath)/teams/\(team.providerID)/schedule?season=\(season)&seasontype=\(phase)")!
                        return try? Self.parse(await read(url), team: team, season: season, phase: phase, cutoff: cutoff)
                    }
                }
            }
            var result: [[MatchupSimulation.Game]?] = []
            for await part in group { result.append(part) }
            return result
        }
        try Task.checkCancellation()
        guard parts.contains(where: { $0 != nil }) else { throw MatchupSimulation.Failure.source }
        var unique: [String: MatchupSimulation.Game] = [:]
        for game in parts.compactMap({ $0 }).flatMap({ $0 }) {
            if let previous = unique[game.id], previous != game { throw MatchupSimulation.Failure.source }
            unique[game.id] = game
        }
        return .init(team: team, games: unique.values.sorted { $0.date > $1.date }, partial: parts.contains(where: { $0 == nil }))
    }

    static func parse(_ data: Data, team: InterestTeam, season: Int, phase: Int, cutoff: Date) throws -> [MatchupSimulation.Game] {
        guard team.isValid, data.count <= 4 * 1024 * 1024,
              let body = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let identity = body["team"] as? [String: Any], identity["id"] as? String == team.providerID,
              let events = body["events"] as? [[String: Any]], events.count <= 300 else { throw MatchupSimulation.Failure.source }
        // ESPN omits requestedSeason when the specified postseason has no games.
        if events.isEmpty { return [] }
        guard let requested = body["requestedSeason"] as? [String: Any],
              requested["year"] as? Int == season, requested["type"] as? Int == phase else { throw MatchupSimulation.Failure.source }
        var games: [MatchupSimulation.Game] = [], seen = Set<String>()
        for event in events {
            guard let id = event["id"] as? String, InterestTeam.validProviderID(id), seen.insert(id).inserted,
                  let text = event["date"] as? String, let date = sportsDate(text),
                  let competitions = event["competitions"] as? [[String: Any]], competitions.count == 1,
                  let game = competitions.first, game["id"] as? String == id,
                  let status = game["status"] as? [String: Any], let type = status["type"] as? [String: Any] else { throw MatchupSimulation.Failure.source }
            guard date < cutoff, date > cutoff.addingTimeInterval(-540 * 86400),
                  type["completed"] as? Bool == true, type["state"] as? String == "post" else { continue }
            guard let eventPhase = event["seasonType"] as? [String: Any], eventPhase["type"] as? Int == phase,
                  let sides = game["competitors"] as? [[String: Any]], sides.count == 2,
                  Set(sides.compactMap { $0["homeAway"] as? String }) == ["home", "away"],
                  let own = sides.first(where: { $0["id"] as? String == team.providerID }),
                  let other = sides.first(where: { $0["id"] as? String != team.providerID }),
                  let ownTeam = own["team"] as? [String: Any], ownTeam["id"] as? String == team.providerID,
                  let otherTeam = other["team"] as? [String: Any], let otherID = otherTeam["id"] as? String,
                  InterestTeam.validProviderID(otherID), other["id"] as? String == otherID,
                  let scored = score(own["score"]), let allowed = score(other["score"]) else { throw MatchupSimulation.Failure.source }
            games.append(.init(id: id, date: date, season: season, scored: scored, allowed: allowed, home: own["homeAway"] as? String == "home"))
        }
        return games
    }
    private static func score(_ value: Any?) -> Int? {
        guard let body = value as? [String: Any], let number = body["value"] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite,
              (0...300).contains(number.doubleValue), number.doubleValue.rounded() == number.doubleValue else { return nil }
        return number.intValue
    }
}

/// Centralized invalidation prevents a slow request from replacing a newer selection.
struct SimulationSelection: Equatable {
    var sport: InterestSport = .nfl
    var away: InterestTeam?
    var home: InterestTeam?
    var event: PlayerMatchupEvent?
    var revision = UUID()
    var ready: Bool { away != nil && home != nil && away?.id != home?.id && away?.sport == sport && home?.sport == sport }
    mutating func changeSport(_ value: InterestSport) {
        sport = value; away = nil; home = nil; event = nil; revision = UUID()
    }
    mutating func choose(_ team: InterestTeam, homeSide: Bool) {
        guard team.sport == sport, team.isValid else { return }
        if homeSide { if away?.id == team.id { away = home }; home = team }
        else { if home?.id == team.id { home = away }; away = team }
        event = nil; revision = UUID()
    }
    mutating func swap() { let previous = home; home = away; away = previous; event = nil; revision = UUID() }
    mutating func choose(_ game: PlayerMatchupEvent, teams: [InterestTeam]) {
        guard game.game.league == sport.rawValue,
              let a = teams.first(where: { $0.sport == sport && $0.providerID == game.awayID }),
              let h = teams.first(where: { $0.sport == sport && $0.providerID == game.homeID }), a.id != h.id else { return }
        away = a; home = h; event = game; revision = UUID()
    }
}
