import Foundation
import CoreFoundation

struct InsightGameObservation {
    let id: String
    let date: Date
    let scored: Double
    let allowed: Double
}

/// Converts provider observations into season rates. Missing metrics remain missing.
enum MatchupInsightSources {
    static func nbaScheduleAllowance(teamID: String, payload: [String: Any], sample: Int,
                                     cutoff: Date, excludingEventID: String) -> Double? {
        guard !teamID.isEmpty, sample > 0, cutoff.timeIntervalSince1970.isFinite,
              let events = payload["events"] as? [[String: Any]] else { return nil }
        let formatter = ISO8601DateFormatter()
        func date(_ value: Any?) -> Date? {
            guard let text = value as? String else { return nil }
            formatter.formatOptions = [.withInternetDateTime]
            if let parsed = formatter.date(from: text) { return parsed }
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return formatter.date(from: text)
        }
        var games: [String: InsightGameObservation] = [:]
        for event in events {
            if event["id"] as? String == excludingEventID { continue }
            guard let competitions = event["competitions"] as? [[String: Any]] else { return nil }
            let completed = competitions.filter {
                (($0["status"] as? [String: Any])?["type"] as? [String: Any])?["completed"] as? Bool == true
            }
            guard !completed.isEmpty else { continue }
            guard completed.count == 1, let competition = completed.first,
                  let eventDate = date(event["date"] ?? competition["date"]) else { return nil }
            guard eventDate < cutoff else { continue }
            guard let id = event["id"] as? String, !id.isEmpty,
                  let competitors = competition["competitors"] as? [[String: Any]], competitors.count == 2 else { return nil }
            func identity(_ competitor: [String: Any]) -> String? {
                let direct = competitor["id"] as? String
                let nested = (competitor["team"] as? [String: Any])?["id"] as? String
                if let direct, let nested, direct != nested { return nil }
                return nested ?? direct
            }
            let ids = competitors.compactMap(identity)
            guard ids.count == 2, Set(ids).count == 2, ids.filter({ $0 == teamID }).count == 1 else { return nil }
            let ownIndex = ids[0] == teamID ? 0 : 1
            func score(_ competitor: [String: Any]) -> Double? {
                if let object = competitor["score"] as? [String: Any] { return number(object["value"]) }
                return number(competitor["score"])
            }
            guard let scored = score(competitors[ownIndex]), let allowed = score(competitors[1 - ownIndex]) else { return nil }
            let game = InsightGameObservation(id: id, date: eventDate, scored: scored, allowed: allowed)
            if let previous = games[id], previous.date != game.date || previous.scored != scored || previous.allowed != allowed { return nil }
            games[id] = game
        }
        guard games.count == sample else { return nil }
        let average = games.values.reduce(0) { $0 + $1.allowed } / Double(sample)
        return average.isFinite ? average : nil
    }

    static func comparisonGroups(sport: String) -> [String] {
        switch sport {
        case "NFL": return ["Scoring", "Passing", "Rushing", "Disruption"]
        case "NBA": return ["Efficiency", "Shooting", "Possession", "Defense"]
        case "NHL": return ["Goals", "Shots", "Goal prevention"]
        case "MLB": return ["Batting", "Power", "Discipline", "Pitching"]
        default: return []
        }
    }

    static func espn(team: String, sport: String, requestedYear: Int, payload: [String: Any], seasonType: Int = 2) -> InsightTeamProfile? {
        guard ["NBA", "NFL", "NHL"].contains(sport),
              let period = payload["requestedSeason"] as? [String: Any],
              [1,2].contains(seasonType), number(period["year"]) == Double(requestedYear), number(period["type"]) == Double(seasonType),
              let results = payload["results"] as? [String: Any],
              let stats = results["stats"] as? [String: Any] else { return nil }
        let own = metrics(stats["categories"])
        let opponent = metrics(results["opponent"])
        let sampleKeys = sport == "NFL" ? ["teamGamesPlayed"] : sport == "NHL" ? ["gamesPlayed", "games"] : ["gamesPlayed"]
        let sampleValues = sampleKeys.compactMap { own[$0] }
        guard let games = sampleValues.first, sampleValues.allSatisfy({ $0 == games }),
              games > 0, games.rounded() == games, games < Double(Int.max),
              !sampleKeys.contains(where: { hasInvalidMetric($0, categories: stats["categories"]) }) else { return nil }
        var offense: [String: Double] = [:], allowed: [String: Double] = [:]
        switch sport {
        case "NFL":
            for key in ["rushingYardsPerGame", "passingYardsPerGame", "totalPointsPerGame"] {
                offense[key] = own[key]
                if opponent["teamGamesPlayed"] == games { allowed[key] = opponent[key] }
            }
            offense["rushingAttemptsPerGame"] = own["rushingAttempts"].map { $0 / games }
            offense["passingAttemptsPerGame"] = own["passingAttempts"].map { $0 / games }
            if opponent["teamGamesPlayed"] == games {
                allowed["rushingAttemptsPerGame"] = opponent["rushingAttempts"].map { $0 / games }
                allowed["passingAttemptsPerGame"] = opponent["passingAttempts"].map { $0 / games }
            }
        case "NBA":
            for key in ["avgPoints", "avgAssists", "avgRebounds", "avgThreePointFieldGoalsMade"] {
                offense[key] = own[key]; allowed[key] = opponent[key]
            }
        case "NHL":
            offense["avgGoals"] = own["goals"].map { $0 / games }
            offense["avgShots"] = own["shotsTotal"].map { $0 / games }
            allowed["avgGoals"] = own["goalsAgainst"].map { $0 / games }
            allowed["avgShots"] = own["shotsAgainst"].map { $0 / games }
        default: break
        }
        guard !offense.isEmpty || !allowed.isEmpty else { return nil }
        let display = (period["displayName"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return InsightTeamProfile(team: team, sample: Int(games), offense: offense, allowed: allowed,
            source: "\((display?.isEmpty == false ? display : nil) ?? String(requestedYear)) ESPN team statistics",
            periodKey: "\(sport)|\(requestedYear)|\(seasonType == 1 ? "preseason":"regular")")
    }

    /// Each dictionary is the MLB Stats API season split `stat`, plus `season` from its split.
    /// Callers must select a regular-season request; dictionaries with another season are rejected.
    static func mlb(team: String, requestedYear: Int, batting: [String: Any]?, pitching: [String: Any]?) -> InsightTeamProfile? {
        let sides = [batting, pitching].compactMap { $0 }
        guard !sides.isEmpty, sides.allSatisfy({ String(describing: $0["season"] ?? "") == String(requestedYear) }) else { return nil }
        let games = sides.compactMap { number($0["gamesPlayed"]) }
        guard games.count == sides.count, let sample = games.first, sample > 0,
              sample.rounded() == sample, sample < Double(Int.max), games.allSatisfy({ $0 == sample }) else { return nil }
        func rates(_ stats: [String: Any]?) -> [String: Double] {
            guard let stats else { return [:] }
            var values: [String: Double] = [:]
            for key in ["avg", "slg"] { values[key] = number(stats[key]) }
            values["homeRunsPerGame"] = number(stats["homeRuns"]).map { $0 / sample }
            for (input,output) in [("runs","runsPerGame"),("hits","hitsPerGame"),("strikeOuts","strikeoutsPerGame")] {
                values[output] = number(stats[input]).map { $0 / sample }
            }
            return values
        }
        let offense = rates(batting), allowed = rates(pitching)
        guard !offense.isEmpty || !allowed.isEmpty else { return nil }
        return InsightTeamProfile(team: team, sample: Int(sample), offense: offense, allowed: allowed,
            source: "\(requestedYear) MLB season statistics", periodKey: "MLB|\(requestedYear)|regular")
    }

    static func recent(team: String, sport: String, games: [InsightGameObservation], cutoff: Date) -> InsightTeamProfile? {
        guard sport == "NBA" || sport == "NHL", cutoff.timeIntervalSince1970.isFinite else { return nil }
        var unique: [String: InsightGameObservation] = [:]
        for game in games {
            guard game.date.timeIntervalSince1970.isFinite else { return nil }
            guard game.date < cutoff else { continue }
            guard !game.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  game.scored.isFinite, game.scored >= 0, game.allowed.isFinite, game.allowed >= 0 else { return nil }
            if let prior = unique[game.id],
               prior.date != game.date || prior.scored != game.scored || prior.allowed != game.allowed { return nil }
            unique[game.id] = game
        }
        let observations = Array(unique.values)
        guard !observations.isEmpty else { return nil }
        let sample = Double(observations.count)
        let offense = observations.reduce(0) { $0 + $1.scored } / sample
        let allowed = observations.reduce(0) { $0 + $1.allowed } / sample
        guard offense.isFinite, allowed.isFinite else { return nil }
        let key = sport == "NBA" ? "avgPoints" : "avgGoals"
        return InsightTeamProfile(team: team, sample: observations.count, offense: [key: offense], allowed: [key: allowed],
            source: "ESPN recent completed games", periodKey: "\(sport)|recent|\(cutoff.timeIntervalSince1970)",
            through: observations.map(\.date).max())
    }

    static func uniqueRosterTeam(player: String, rosters: [String: [String]], expectedTeams: [String]) -> String? {
        guard expectedTeams.count == 2, Set(expectedTeams).count == 2,
              Set(rosters.keys) == Set(expectedTeams) else { return nil }
        func normalized(_ name: String) -> String {
            name.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
                .unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(String.init).joined()
        }
        let target = normalized(player)
        guard !target.isEmpty else { return nil }
        let matches = expectedTeams.flatMap { team in
            (rosters[team] ?? []).filter { normalized($0) == target }.map { _ in team }
        }
        return matches.count == 1 ? matches[0] : nil
    }

    private static func number(_ raw: Any?) -> Double? {
        if let numeric = raw as? NSNumber, CFGetTypeID(numeric) == CFBooleanGetTypeID() { return nil }
        let value: Double?
        if let text = raw as? String { value = Double(text) }
        else if let numeric = raw as? NSNumber { value = numeric.doubleValue }
        else { value = nil }
        guard let value, value.isFinite, value >= 0 else { return nil }
        return value
    }

    private static func entries(_ categories: Any?) -> [[String: Any]] {
        let list: [[String: Any]]
        if let wrapper = categories as? [String: Any] { list = wrapper["categories"] as? [[String: Any]] ?? [] }
        else { list = categories as? [[String: Any]] ?? [] }
        return list.flatMap { $0["stats"] as? [[String: Any]] ?? [] }
    }

    private static func metrics(_ categories: Any?) -> [String: Double] {
        var values: [String: Double] = [:], invalid = Set<String>()
        for entry in entries(categories) {
            guard let key = entry["name"] as? String else { continue }
            guard let value = number(entry["value"]) else { invalid.insert(key); continue }
            if let previous = values[key], previous != value { invalid.insert(key) }
            values[key] = value
        }
        for key in invalid { values.removeValue(forKey: key) }
        return values
    }

    private static func hasInvalidMetric(_ key: String, categories: Any?) -> Bool {
        let matching = entries(categories).filter { $0["name"] as? String == key }
        return !matching.isEmpty && metrics(categories)[key] == nil
    }
}
