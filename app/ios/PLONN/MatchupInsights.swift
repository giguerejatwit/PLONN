import Foundation

/// Values are observed season rates, never predictions for an individual player.
struct InsightTeamProfile {
    let team: String
    let sample: Int
    let offense: [String: Double]
    var allowed: [String: Double]
    let source: String
    var periodKey: String? = nil
    var through: Date? = nil
}

struct InsightMatchup {
    let sport: String
    let eventID: String
    let homeTeam: String
    let awayTeam: String
    let commence: String?
}

/// The caller must resolve player membership from an authoritative roster for this matchup.
/// Market.opponent, presentation order and player names are not membership evidence.
struct InsightPlayerProp {
    let market: Market
    let verifiedPlayerTeam: String
}

struct MatchupInsight: Identifiable, Equatable, Sendable {
    let id: String
    let text: String
    let evidence: String
    let marketID: String?
    let topic: String
    let team: String
    var hasPairedEvidence: Bool = false
}

enum MatchupInsightEngine {
    private struct Metric {
        let key: String
        let topic: String
        let unit: String
        let markets: Set<String>
        var decimals: Int = 1
    }

    static func generate(matchup: InsightMatchup, profiles: [InsightTeamProfile], props: [InsightPlayerProp], recentProfiles: [InsightTeamProfile] = []) -> [MatchupInsight] {
        guard !matchup.eventID.isEmpty, !matchup.homeTeam.isEmpty, !matchup.awayTeam.isEmpty,
              matchup.homeTeam != matchup.awayTeam else { return [] }
        let metrics: [Metric]
        switch matchup.sport {
        case "NFL": metrics = [
            Metric(key: "rushingYardsPerGame", topic: "Rushing", unit: "rushing yards per game", markets: ["player_rush_yds"]),
            Metric(key: "passingYardsPerGame", topic: "Passing", unit: "passing yards per game", markets: ["player_pass_yds", "player_reception_yds"])]
        case "NBA": metrics = [
            Metric(key: "avgPoints", topic: "Scoring", unit: "points per game", markets: ["player_points"]),
            Metric(key: "avgAssists", topic: "Assists", unit: "assists per game", markets: ["player_assists"]),
            Metric(key: "avgRebounds", topic: "Rebounds", unit: "rebounds per game", markets: ["player_rebounds"]),
            Metric(key: "avgThreePointFieldGoalsMade", topic: "Three-pointers", unit: "made threes per game", markets: ["player_threes"])]
        case "MLB": metrics = [
            Metric(key: "avg", topic: "Hitting", unit: "batting average", markets: ["batter_hits"], decimals: 3),
            Metric(key: "slg", topic: "Total bases", unit: "slugging percentage", markets: ["batter_total_bases"], decimals: 3),
            Metric(key: "homeRunsPerGame", topic: "Home runs", unit: "home runs per game", markets: ["batter_home_runs"])]
        case "NHL": metrics = [
            Metric(key: "avgGoals", topic: "Goals", unit: "goals per game", markets: ["player_goals"]),
            Metric(key: "avgShots", topic: "Shots", unit: "shots on goal per game", markets: ["player_shots_on_goal"])]
        default: return []
        }
        // Duplicate profiles are ambiguous even when their numbers happen to agree.
        let grouped = Dictionary(grouping: profiles, by: \.team)
        func profile(_ name: String) -> InsightTeamProfile? {
            guard let values = grouped[name], values.count == 1, let value = values.first,
                  value.sample > 0, !value.source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            if let through = value.through, let commence = matchup.commence.flatMap(date), through >= commence { return nil }
            return value
        }
        func observed(_ value: Double?) -> Double? {
            guard let value, value.isFinite, value >= 0 else { return nil }
            return value
        }
        let propIDs = Dictionary(grouping: props, by: { $0.market.id })
        var result: [MatchupInsight] = []
        for team in [matchup.homeTeam, matchup.awayTeam] {
            let opponent = team == matchup.homeTeam ? matchup.awayTeam : matchup.homeTeam
            let own = profile(team), defense = profile(opponent)
            if let ownPeriod = own?.periodKey, let opposingPeriod = defense?.periodKey,
               ownPeriod != opposingPeriod { continue }
            for metric in metrics {
                let offenseValue = observed(own?.offense[metric.key])
                let allowedValue = observed(defense?.allowed[metric.key])
                guard offenseValue != nil || allowedValue != nil else { continue }
                func formatted(_ value: Double) -> String { String(format: "%.*f", locale: Locale(identifier: "en_US_POSIX"), metric.decimals, value) }
                var clauses: [String] = []
                if let value = offenseValue {
                    if metric.decimals == 3 { clauses.append("\(team) has a \(formatted(value)) \(metric.unit)") }
                    else if matchup.sport == "NFL",
                            let attempts = observed(own?.offense[metric.key == "rushingYardsPerGame" ? "rushingAttemptsPerGame" : "passingAttemptsPerGame"]) {
                        let action = metric.key == "rushingYardsPerGame" ? "runs" : "passes"
                        clauses.append("\(team) \(action) \(formatted(attempts)) times per game for \(formatted(value)) \(metric.unit)")
                    } else { clauses.append("\(team) averages \(formatted(value)) \(metric.unit)") }
                }
                if let value = allowedValue {
                    if metric.decimals == 3 { clauses.append("\(opponent) allows a \(formatted(value)) opponent \(metric.unit)") }
                    else { clauses.append("\(opponent) allows \(formatted(value)) \(metric.unit)") }
                }
                let eligible = props.filter { prop in
                    let market = prop.market
                    guard offenseValue != nil, allowedValue != nil,
                          prop.verifiedPlayerTeam == team, !market.id.isEmpty, propIDs[market.id]?.count == 1,
                          market.sport == matchup.sport, matchesEvent(market, matchup: matchup),
                          market.kind == "prop", metric.markets.contains(market.market),
                          !market.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                          !market.book.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                          market.mode == "live",
                          !["post", "final", "completed", "canceled", "cancelled", "postponed", "suspended"].contains(market.eventState?.lowercased() ?? ""),
                          let line = market.line,
                          line.isFinite, line >= 0, market.price.isFinite, abs(market.price) >= 100,
                          let side = market.side, ["over", "under"].contains(side.lowercased()) else { return false }
                    return true
                }.sorted { $0.market.id < $1.market.id }.first
                var text = clauses.joined(separator: "; ") + "."
                if let prop = eligible, let line = prop.market.line, let side = prop.market.side {
                    let market = prop.market
                    let propLine = String(format: "%g", locale: Locale(identifier: "en_US_POSIX"), line)
                    text += " Related prop: \(market.name), \(side.lowercased()) \(propLine) \(propLabel(market.market)) (\(market.book))."
                } else if offenseValue == nil {
                    text += " \(team)’s matching offensive rate is unavailable."
                } else if allowedValue == nil {
                    text += " \(opponent)’s matching defensive rate is unavailable."
                }
                var evidence: [String] = []
                if offenseValue != nil, let own { evidence.append("\(team): \(own.source) · \(own.sample) games") }
                if allowedValue != nil, let defense { evidence.append("\(opponent): \(defense.source) · \(defense.sample) games") }
                result.append(MatchupInsight(id: "\(matchup.sport)|\(matchup.eventID)|\(team)|\(metric.key)", text: text,
                    evidence: evidence.joined(separator: "; ") + ". Team history, not a player forecast.",
                    marketID: eligible?.market.id, topic: metric.topic, team: team,
                    hasPairedEvidence: offenseValue != nil && allowedValue != nil))
            }
        }
        if !recentProfiles.isEmpty {
            // Generate each period independently: a recent offense must never be
            // paired with a season defensive rate merely because the key agrees.
            let fallback = generate(matchup: matchup, profiles: recentProfiles, props: props)
            for insight in fallback {
                if let index = result.firstIndex(where: { $0.id == insight.id }) {
                    if !result[index].hasPairedEvidence && insight.hasPairedEvidence {
                        result[index] = insight
                    }
                } else {
                    result.append(insight)
                }
            }
        }
        return result
    }

    private static func date(_ value: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: value) { return date }
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: value)
    }

    private static func matchesEvent(_ market: Market, matchup: InsightMatchup) -> Bool {
        guard let expectedStart = matchup.commence.flatMap(date),
              let actualStart = market.commence.flatMap(date),
              abs(expectedStart.timeIntervalSince(actualStart)) <= 60 else { return false }
        if market.event == matchup.eventID { return true }
        guard let label = market.eventLabel else { return false }
        func normalized(_ text: String) -> String {
            text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").lowercased()
        }
        let value = normalized(label)
        let home = normalized(matchup.homeTeam), away = normalized(matchup.awayTeam)
        guard !home.isEmpty, !away.isEmpty, home != away else { return false }
        return ["\(away) at \(home)", "\(home) at \(away)", "\(away) @ \(home)", "\(home) @ \(away)"].contains(value)
    }

    private static func propLabel(_ key: String) -> String {
        switch key {
        case "player_rush_yds": return "rushing yards"
        case "player_pass_yds": return "passing yards"
        case "player_reception_yds": return "receiving yards"
        case "player_points": return "points"
        case "player_assists": return "assists"
        case "player_rebounds": return "rebounds"
        case "player_threes": return "made threes"
        case "batter_hits": return "hits"
        case "batter_total_bases": return "total bases"
        case "batter_home_runs": return "home runs"
        case "player_goals": return "goals"
        case "player_shots_on_goal": return "shots on goal"
        default: return key
        }
    }
}
