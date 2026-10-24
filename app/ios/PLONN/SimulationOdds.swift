import Foundation

/// Recorded contracts are matched to a verified scheduled matchup. Simulation
/// frequencies are descriptive; they never become modelValidated or betting EV.
enum SimulationOdds {
    enum MissingEstimate: Equatable, Sendable {
        case limitedHistory(Int), notModeled, outsideRotation, noScoredScenarios
        var title: String {
            switch self {
            case .limitedHistory: return "Limited history"
            case .notModeled: return "Not modeled"
            case .outsideRotation: return "Not simulated"
            case .noScoredScenarios: return "No estimate"
            }
        }
        var detail: String {
            switch self {
            case .limitedHistory(let count): return "\(count) recorded games · 5 needed for a comparison."
            case .notModeled: return "This stat isn’t included in the simulation."
            case .outsideRotation: return "This player has no simulated role for this stat."
            case .noScoredScenarios: return "No scenarios could be scored against this line."
            }
        }
    }
    struct Comparison: Identifiable, Sendable {
        let quote: Market
        let hit: Int
        let miss: Int
        let push: Int
        let unavailable: Int
        let playerAppearances: Int?
        var missingEstimate: MissingEstimate? = nil
        var id: String { quote.id }
        var counted: Int { hit + miss + push }
        var hitRate: Double? { counted > 0 ? Double(hit) / Double(counted) : nil }
        var implied: Double? { BuilderLogic.decimalOdds(quote.price).map { 1 / $0 } }
    }
    static func ranked(_ comparisons: [Comparison]) -> [Comparison] {
        comparisons.sorted { a, b in
            switch (a.hitRate, b.hitRate) {
            case let (x?, y?) where x != y: return x > y
            case (_?, nil): return true
            case (nil, _?): return false
            default:
                // Stable ties keep the list from jumping when quotes refresh.
                let x = [a.quote.name, a.quote.label, a.quote.book, a.id].map(BuilderLogic.normalized)
                let y = [b.quote.name, b.quote.label, b.quote.book, b.id].map(BuilderLogic.normalized)
                return x == y ? a.id < b.id : x.lexicographicallyPrecedes(y)
            }
        }
    }
    static func compare(_ result: PlayerSimulation, event: PlayerMatchupEvent?, markets: [Market], now: Date) -> [Comparison] {
        guard let event, event.homeID == result.home.team.providerID, event.awayID == result.away.team.providerID,
              event.game.league == result.home.team.sport.rawValue, let start = event.game.start, start > now,
              now.timeIntervalSince(result.scores.generated) <= 900 else { return [] }
        let proof = AnalyticsEvent(sport: event.game.league, providerID: event.game.eventID, home: event.game.home, away: event.game.away, start: start,
                                   homeID: event.homeID, awayID: event.awayID, seasonType: event.seasonType)
        let roster = result.current?.teams.flatMap { $0.availability.eligible } ?? (result.away.roster + result.home.roster)
        var accepted: [String: Market] = [:]
        for team in [result.away.team, result.home.team] {
            for key in ["h2h", "spreads", "totals", "team_totals"] {
                let target = AnalyticsStatsTarget(sport: event.game.league, kind: "team", name: team.name, marketKey: key, teamName: team.name, teamID: team.providerID)
                for quote in AnalyticsOddsMatcher.matchCached(target: target, event: proof, markets: markets, now: now, recordedSnapshot: true).markets { accepted[quote.id] = quote }
            }
        }
        // Only quote-referenced players are evaluated; no extra prop requests.
        for player in roster {
            let keys = Set(markets.filter { $0.kind == "prop" && AnalyticsOddsMatcher.normalized($0.name) == AnalyticsOddsMatcher.normalized(player.name) && !$0.market.hasSuffix("_alternate") }.map(\.market))
            for key in keys {
                guard let stat = PlayerMatchupSetup.stat(sport: event.game.league, marketKey: key) else { continue }
                for side in ["over", "under"] {
                    for quote in PlayerMatchupLineMatching.quotes(event: event, playerID: player.id, players: roster, stat: stat, side: side, markets: markets, now: now) where !quote.market.hasSuffix("_alternate") { accepted[quote.id] = quote }
                }
            }
        }
        return accepted.values.sorted { $0.id < $1.id }.compactMap { quote in
            guard let recorded = quote.updated.flatMap(sportsDate), (0...86400).contains(now.timeIntervalSince(recorded)) else { return nil }
            let player = quote.kind == "prop" ? roster.first { AnalyticsOddsMatcher.normalized($0.name) == AnalyticsOddsMatcher.normalized(quote.name) } : nil
            let homeSide = player.map { $0.teamID == event.homeID } ?? RecordedTeamIdentity.matches(quote.name, event.game.home)
            var appearances: Int?
            if let player {
                appearances = (homeSide ? result.home : result.away).boxes.filter { value(box: $0, playerID: player.id, market: quote.market, sport: quote.sport) != nil }.count
            }
            let statFields = fields(market: quote.market, sport: quote.sport)
            let samples: [[Float]]? = player.flatMap { player in
                guard let current = result.current, let statFields else { return nil }
                let columns = statFields.compactMap { group, stat in current.columns[homeSide ? 1 : 0][group + ":" + player.id + "/" + stat] }
                return columns.count == statFields.count ? columns : nil
            }
            var hit = 0, miss = 0, push = 0, unknown = 0
            for index in result.trials.indices {
                let margin: Double?
                if let player {
                    let stat: Double?
                    if result.current != nil {
                        let values = samples?.map { Double($0[index]) }
                        stat = values?.allSatisfy(\.isFinite) == true ? values?.reduce(0, +) : nil
                    } else { stat = value(box: result.box(homeSide: homeSide, trial: index), playerID: player.id, market: quote.market, sport: quote.sport) }
                    if (appearances ?? 0) >= 5, let line = quote.line, let stat {
                        margin = (stat - line) * (quote.side?.lowercased() == "under" ? -1 : 1)
                    } else { margin = nil }
                } else {
                    let score = result.scores.scores[index]
                    let own = Double(homeSide ? score.home : score.away), other = Double(homeSide ? score.away : score.home)
                    switch quote.market {
                    case "h2h": margin = own == other ? nil : own - other // No invented overtime winner or moneyline push.
                    case "spreads": margin = quote.line.map { own - other + $0 }
                    case "totals": margin = quote.line.map { (Double(score.total) - $0) * (quote.side?.lowercased() == "under" ? -1 : 1) }
                    case "team_totals": margin = quote.line.map { (own - $0) * (quote.side?.lowercased() == "under" ? -1 : 1) }
                    default: margin = nil
                    }
                }
                if let margin { if abs(margin) < 1e-9 { push += 1 } else if margin > 0 { hit += 1 } else { miss += 1 } }
                else { unknown += 1 }
            }
            let missing: MissingEstimate?
            if hit + miss + push > 0 { missing = nil }
            else if quote.kind == "prop", statFields == nil { missing = .notModeled }
            else if let appearances, appearances < 5 { missing = .limitedHistory(appearances) }
            else if player != nil, result.current != nil, samples == nil { missing = .outsideRotation }
            else { missing = .noScoredScenarios }
            return Comparison(quote: quote, hit: hit, miss: miss, push: push, unavailable: unknown, playerAppearances: appearances, missingEstimate: missing)
        }
    }
    /// A recorded pick can be saved without a simulation estimate. Recheck time
    /// at the tap, and preserve the original contract rather than attaching a
    /// scenario frequency as a validated model probability.
    static func builderRejection(for quote: Market, event: PlayerMatchupEvent?, existing: [Market], now: Date) -> String? {
        guard let event, let start = event.game.start, start > now,
              quote.sport == event.game.league,
              !["in", "post"].contains(quote.eventState ?? "") else { return "This game has started. Pregame picks are no longer available." }
        guard let recorded = quote.updated.flatMap(sportsDate), (0...86400).contains(now.timeIntervalSince(recorded)),
              BuilderLogic.decimalOdds(quote.price) != nil else { return "This recorded price has expired. Refresh the matchup." }
        return BuilderLogic.rejection(for: quote, existing: existing)
    }
    static func value(box: PlayerSimulation.Box, playerID: String, market: String, sport: String) -> Double? {
        guard let fields = fields(market: market, sport: sport) else { return nil }
        let values = fields.compactMap { group, key in box.lines.first { $0.playerID == playerID && $0.group == group }?.values[key] }
        return values.count == fields.count ? values.reduce(0, +) : nil
    }
    static func fields(market: String, sport: String) -> [(String, String)]? {
        let mapping: [String: [(String, String)]]
        switch sport {
        case "NBA":
            mapping = ["player_points":[("players","points")], "player_rebounds":[("players","rebounds")], "player_assists":[("players","assists")], "player_steals":[("players","steals")], "player_blocks":[("players","blocks")], "player_threes":[("players","threePointFieldGoalsMade")], "player_turnovers":[("players","turnovers")], "player_field_goals":[("players","fieldGoalsMade")], "player_frees_made":[("players","freeThrowsMade")], "player_frees_attempts":[("players","freeThrowsAttempted")], "player_points_rebounds_assists":[("players","points"),("players","rebounds"),("players","assists")], "player_points_rebounds":[("players","points"),("players","rebounds")], "player_points_assists":[("players","points"),("players","assists")], "player_rebounds_assists":[("players","rebounds"),("players","assists")], "player_blocks_steals":[("players","blocks"),("players","steals")]]
        case "NFL":
            mapping = ["player_pass_yds":[("passing","passingYards")], "player_pass_tds":[("passing","passingTouchdowns")], "player_pass_attempts":[("passing","passingAttempts")], "player_pass_completions":[("passing","completions")], "player_pass_interceptions":[("passing","interceptions")], "player_rush_yds":[("rushing","rushingYards")], "player_rush_attempts":[("rushing","rushingAttempts")], "player_rush_tds":[("rushing","rushingTouchdowns")], "player_reception_yds":[("receiving","receivingYards")], "player_receptions":[("receiving","receptions")], "player_reception_tds":[("receiving","receivingTouchdowns")], "player_rush_reception_yds":[("rushing","rushingYards"),("receiving","receivingYards")], "player_pass_rush_yds":[("passing","passingYards"),("rushing","rushingYards")], "player_sacks":[("defensive","sacks")], "player_tackles_assists":[("defensive","totalTackles")], "player_field_goals":[("kicking","fieldGoalsMade")], "player_kicking_points":[("kicking","totalKickingPoints")]]
        case "MLB":
            mapping = ["batter_hits":[("batting","hits")], "batter_rbis":[("batting","RBIs")], "batter_home_runs":[("batting","homeRuns")], "batter_runs_scored":[("batting","runs")], "batter_walks":[("batting","walks")], "batter_strikeouts":[("batting","strikeouts")], "batter_hits_runs_rbis":[("batting","hits"),("batting","runs"),("batting","RBIs")], "pitcher_strikeouts":[("pitching","strikeouts")], "pitcher_outs":[("pitching","fullInnings.partInnings")], "pitcher_earned_runs":[("pitching","earnedRuns")], "pitcher_hits_allowed":[("pitching","hits")], "pitcher_walks":[("pitching","walks")]]
        case "NHL":
            mapping = ["player_goals":[("skaters","goals")], "player_assists":[("skaters","assists")], "player_points":[("skaters","goals"),("skaters","assists")], "player_shots_on_goal":[("skaters","shotsTotal")], "player_blocked_shots":[("skaters","blockedShots")], "player_total_saves":[("goalies","saves")]]
        default: return nil
        }
        return mapping[market]
    }
}
