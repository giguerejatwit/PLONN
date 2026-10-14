import Foundation

struct PlayerGame: Identifiable, Sendable {
    let id: String
    let date: Date
    let opponent: String
    let value: Double
    let workload: String
    var sport: String? = nil
    var playerID: String? = nil
    var playerTeamID: String? = nil
    var opponentID: String? = nil
    var marketKey: String? = nil
    var season: Int? = nil
    var homeAway: String? = nil
}

struct OpponentHistoryContext: Sendable {
    let sport: String
    let playerID: String
    let playerName: String
    let marketKey: String
    let opponentID: String
    let opponentName: String
    let targetStart: Date
    let cutoff: Date
    var targetEventID: String? = nil
    let source: String
    let requestedSeasons: [Int]
    let loadedSeasons: [Int]
}

struct OpponentHistorySummary: Sendable {
    let games: [PlayerGame]
    let n: Int
    let average: Double?
    let overs: Int?
    let unders: Int?
    let pushes: Int?
    let text: String
    let evidence: String
    let unavailableReason: String?
}

enum OpponentPropHistoryEngine {
    static func insightMarkets(markets: [Market], matchup: InsightMatchup, verifiedPlayerTeams: [String: String], limit: Int = 3) -> [Market] {
        func normalized(_ text: String) -> String {
            text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").lowercased()
        }
        func date(_ text: String?) -> Date? {
            guard let text else { return nil }
            let formatter = ISO8601DateFormatter()
            if let parsed = formatter.date(from: text) { return parsed }
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return formatter.date(from: text)
        }
        let home = normalized(matchup.homeTeam), away = normalized(matchup.awayTeam)
        guard !home.isEmpty, !away.isEmpty, home != away,
              let start = date(matchup.commence), supportedMarkets[matchup.sport] != nil else { return [] }
        let ids = Dictionary(grouping: markets, by: \.id)
        let labels: Set<String> = ["\(away) at \(home)", "\(home) at \(away)", "\(away) @ \(home)", "\(home) @ \(away)"]
        let eligible = markets.filter { market in
            guard !normalized(market.id).isEmpty, ids[market.id]?.count == 1,
                  market.kind == "prop", market.mode == "live", market.sport == matchup.sport,
                  supportedMarkets[matchup.sport]?.contains(market.market) == true,
                  !normalized(market.name).isEmpty, !normalized(market.book).isEmpty,
                  let team = verifiedPlayerTeams[market.id], [matchup.homeTeam, matchup.awayTeam].contains(team),
                  !["post", "final", "completed", "canceled", "cancelled", "postponed", "suspended"].contains(normalized(market.eventState ?? "")),
                  market.price.isFinite, abs(market.price) >= 100,
                  let line = market.line, line.isFinite,
                  line >= 0 || (matchup.sport == "NFL" && signedYards.contains(market.market)),
                  let side = market.side, ["over", "under"].contains(normalized(side)),
                  let label = market.eventLabel, labels.contains(normalized(label)),
                  let commence = date(market.commence), abs(commence.timeIntervalSince(start)) <= 60 else { return false }
            return true
        }.sorted { $0.id < $1.id }
        var keys = Set<String>()
        let distinct = eligible.filter { keys.insert(normalized($0.name) + "|" + $0.market).inserted }
        let count = min(max(limit, 1), 3)
        var selected: [Market] = []
        for team in [matchup.homeTeam, matchup.awayTeam] {
            if selected.count < count, let market = distinct.first(where: { verifiedPlayerTeams[$0.id] == team }) { selected.append(market) }
        }
        for market in distinct where selected.count < count && !selected.contains(where: { $0.id == market.id }) { selected.append(market) }
        return selected
    }

    static let supportedMarkets: [String: Set<String>] = [
        "NBA": ["player_points", "player_rebounds", "player_assists", "player_threes", "player_blocks", "player_steals", "player_points_rebounds_assists", "player_points_rebounds", "player_points_assists", "player_rebounds_assists"],
        "NFL": ["player_pass_yds", "player_pass_tds", "player_pass_attempts", "player_pass_completions", "player_pass_interceptions", "player_rush_yds", "player_rush_attempts", "player_reception_yds", "player_receptions", "player_rush_reception_yds"],
        "MLB": ["batter_hits", "batter_rbis", "batter_stolen_bases", "batter_home_runs", "batter_total_bases", "batter_runs_scored", "pitcher_strikeouts"],
        "NHL": ["player_goals", "player_assists", "player_points", "player_shots_on_goal", "player_blocked_shots", "player_total_saves", "player_power_play_points"]
    ]
    private static let signedYards: Set<String> = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_rush_reception_yds"]

    static func summarize(games: [PlayerGame], context: OpponentHistoryContext, line: Double?, displayCount: Int = 10) -> OpponentHistorySummary {
        let label = statLabel(context.marketKey)
        let requested = Set(context.requestedSeasons), loaded = Set(context.loadedSeasons)
        let seasons = requested.sorted().map(String.init).joined(separator: ", ")
        let missing = requested.subtracting(loaded).sorted()
        let cutoff = min(context.targetStart, context.cutoff)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York")!
        let cutoffDay = calendar.startOfDay(for: cutoff)
        let formatter = DateFormatter()
        formatter.calendar = calendar; formatter.timeZone = calendar.timeZone
        formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.dateFormat = "MMM d, yyyy"
        let window = "\(context.source) · seasons \(seasons) · completed games before \(formatter.string(from: cutoffDay)) (Eastern)"
        func unavailable(_ reason: String) -> OpponentHistorySummary {
            OpponentHistorySummary(games: [], n: 0, average: nil, overs: nil, unders: nil, pushes: nil,
                text: "Opponent history is unavailable. \(reason)", evidence: window, unavailableReason: reason)
        }
        guard supportedMarkets[context.sport]?.contains(context.marketKey) == true else {
            return unavailable("This stat is not supported for this sport.")
        }
        func nonempty(_ value: String) -> Bool { !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let targetYear = calendar.component(.year, from: context.targetStart)
        guard nonempty(context.playerID), nonempty(context.opponentID), nonempty(context.playerName),
              nonempty(context.opponentName), nonempty(context.source),
              requested.allSatisfy({ (2000...(max(2000, targetYear) + 1)).contains($0) }),
              !requested.isEmpty, context.targetStart.timeIntervalSince1970.isFinite,
              context.cutoff.timeIntervalSince1970.isFinite else {
            return unavailable("Verified player, opponent or source information is missing.")
        }
        // Quarantine conflicting copies before filtering. A wrong-identity copy
        // cannot silently make a same-event right-identity copy look trustworthy.
        let groups = Dictionary(grouping: games, by: \.id)
        var cleaned: [PlayerGame] = []
        for (id, copies) in groups {
            guard !id.isEmpty, id != context.targetEventID, let game = copies.first else { continue }
            guard copies.allSatisfy({
                $0.date == game.date && $0.value == game.value && $0.sport == game.sport &&
                $0.playerID == game.playerID && $0.playerTeamID == game.playerTeamID &&
                $0.opponentID == game.opponentID && $0.marketKey == game.marketKey && $0.season == game.season && $0.homeAway == game.homeAway
            }) else { continue }
            guard game.sport == context.sport, game.playerID == context.playerID,
                  game.opponentID == context.opponentID, game.marketKey == context.marketKey,
                  let ownTeam = game.playerTeamID, nonempty(ownTeam), ownTeam != context.opponentID,
                  let season = game.season, requested.contains(season), loaded.contains(season),
                  game.date.timeIntervalSince1970.isFinite, game.date < cutoffDay,
                  game.value.isFinite, game.value.rounded(.towardZero) == game.value,
                  game.value >= 0 || (context.sport == "NFL" && signedYards.contains(context.marketKey)) else { continue }
            cleaned.append(game)
        }
        let limit = displayCount == 5 ? 5 : 10
        cleaned = Array(cleaned.sorted { $0.date == $1.date ? $0.id < $1.id : $0.date > $1.date }.prefix(limit))
        guard !cleaned.isEmpty else {
            return unavailable(missing.isEmpty ? "No verified prior games against this opponent were found in the selected seasons." : "No verified prior games were found; some requested seasons could not be loaded.")
        }
        // Divide each value first to avoid overflow from summing finite outcomes.
        let average = cleaned.reduce(0.0) { $0 + $1.value / Double(cleaned.count) }
        guard average.isFinite else { return unavailable("The reported game values could not be summarized safely.") }
        let signedMarket = context.sport == "NFL" && signedYards.contains(context.marketKey)
        let validLine = line.flatMap { $0.isFinite && ($0 >= 0 || signedMarket) ? $0 : nil }
        let overs = validLine.map { threshold in cleaned.filter { $0.value > threshold }.count }
        let unders = validLine.map { threshold in cleaned.filter { $0.value < threshold }.count }
        let pushes = validLine.map { threshold in cleaned.filter { $0.value == threshold }.count }
        let avg = String(format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), average)
        var text = "\(context.playerName) averaged \(avg) \(label) in \(cleaned.count) prior \(cleaned.count == 1 ? "game" : "games") against \(context.opponentName)."
        if let threshold = validLine, let overs, let unders, let pushes {
            let quoted = String(format: "%g", locale: Locale(identifier: "en_US_POSIX"), threshold)
            text += " Against the current \(quoted) line: \(overs) above, \(unders) below, \(pushes) equal."
        }
        var evidence = window + " · \(cleaned.count) verified \(cleaned.count == 1 ? "game" : "games")"
        if cleaned.count < 5 { evidence += " · small sample" }
        if !missing.isEmpty { evidence += " · unavailable seasons: " + missing.map(String.init).joined(separator: ", ") }
        evidence += ". Actual game results compared with today’s line; not historical bet results or a forecast."
        return OpponentHistorySummary(games: cleaned, n: cleaned.count, average: average, overs: overs, unders: unders,
            pushes: pushes, text: text, evidence: evidence, unavailableReason: nil)
    }

    static func statLabel(_ key: String) -> String {
        let labels: [String: String] = [
            "player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists", "player_threes": "made threes",
            "player_blocks": "blocks", "player_steals": "steals", "player_points_rebounds_assists": "points + rebounds + assists",
            "player_points_rebounds": "points + rebounds", "player_points_assists": "points + assists", "player_rebounds_assists": "rebounds + assists",
            "player_pass_yds": "passing yards", "player_pass_tds": "passing touchdowns", "player_pass_attempts": "pass attempts",
            "player_pass_completions": "pass completions", "player_pass_interceptions": "interceptions thrown", "player_rush_yds": "rushing yards",
            "player_rush_attempts": "rush attempts", "player_reception_yds": "receiving yards", "player_receptions": "receptions", "player_rush_reception_yds": "rushing + receiving yards",
            "batter_hits": "hits", "batter_rbis": "RBI", "batter_stolen_bases": "stolen bases", "batter_home_runs": "home runs",
            "batter_total_bases": "total bases", "batter_runs_scored": "runs", "pitcher_strikeouts": "strikeouts",
            "player_goals": "goals", "player_shots_on_goal": "shots on goal", "player_blocked_shots": "blocked shots",
            "player_total_saves": "saves", "player_power_play_points": "power-play points"
        ]
        return labels[key] ?? key
    }
}
