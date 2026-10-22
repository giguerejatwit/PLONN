import Foundation

enum AnalyticsTrendInsightKind: String, Sendable, CaseIterable {
    case recentBaseline, consistency, homeAway, seasonBaseline, opponent
}
struct AnalyticsTrendInsight: Identifiable, Sendable {
    let id: String
    let kind: AnalyticsTrendInsightKind
    let title: String
    let headline: String
    let detail: String
    let sample: Int
    let startDate: Date?
    let endDate: Date?
    let source: String
    var availability: String? = nil
}

/// Labels describe the measured result, including its sport-specific unit.
enum AnalyticsCardCopy {
    static func isScoringMargin(_ market: Market) -> Bool { ["h2h", "spreads"].contains(market.market) }
    static func scoringUnit(_ market: Market) -> String {
        switch market.sport { case "MLB": return "runs"; case "NHL": return "goals"; default: return "points" }
    }
    static func unit(_ market: Market) -> String {
        guard market.kind == "prop" else { return scoringUnit(market) }
        let key = AnalyticsHistoryEngine.statisticKey(sport: market.sport, marketKey: market.market) ?? market.market
        if key == "batter_rbis" { return "runs batted in" }
        return OpponentPropHistoryEngine.statLabel(key).replacingOccurrences(of: "_", with: " ")
    }
    static func title(_ market: Market) -> String {
        switch market.market {
        case "h2h", "spreads": return "Wins & losses"
        case "totals": return "Combined scoring"
        case "team_totals": return "Team scoring"
        default: return unit(market).prefix(1).uppercased() + unit(market).dropFirst()
        }
    }
    static func number(_ value: Double) -> String {
        String(format: abs(value) >= 1_000_000 ? "%.4g" : "%.1f", locale: Locale(identifier: "en_US_POSIX"), value)
    }
    /// Keep the sport/stat qualifier where an abbreviation alone is ambiguous.
    static func perGameUnit(_ market: Market) -> String {
        guard market.kind == "prop" else {
            return ["MLB": "R/GM", "NHL": "G/GM"][market.sport] ?? "PTS/GM"
        }
        let key = AnalyticsHistoryEngine.statisticKey(sport: market.sport, marketKey: market.market) ?? market.market
        let labels = [
            "player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST", "player_threes": "3PM",
            "player_blocks": "BLK", "player_steals": "STL", "player_turnovers": "TO",
            "player_points_rebounds_assists": "PTS+REB+AST", "player_points_rebounds": "PTS+REB",
            "player_points_assists": "PTS+AST", "player_rebounds_assists": "REB+AST",
            "player_pass_yds": "PASS YDS", "player_pass_tds": "PASS TD", "player_pass_attempts": "PASS ATT",
            "player_pass_completions": "CMP", "player_pass_interceptions": "INT THROWN",
            "player_rush_yds": "RUSH YDS", "player_rush_attempts": "RUSH ATT",
            "player_reception_yds": "REC YDS", "player_receptions": "REC", "player_rush_reception_yds": "RUSH+REC YDS",
            "batter_hits": "H", "batter_rbis": "RBI", "batter_stolen_bases": "SB", "batter_home_runs": "HR",
            "batter_total_bases": "TB", "batter_runs_scored": "R", "pitcher_strikeouts": "K",
            "player_goals": "G", "player_shots_on_goal": "SOG", "player_blocked_shots": "BLK",
            "player_total_saves": "SV", "player_power_play_points": "PPP"
        ]
        return (labels[key] ?? unit(market)) + "/GM"
    }
    static func average(_ value: Double, market: Market, abbreviated: Bool = true) -> String {
        let units = abbreviated ? perGameUnit(market) : unit(market) + " per game"
        if isScoringMargin(market) {
            if value == 0 { return "Even with opponents on average" }
            return (value > 0 ? "Outscored opponents by " : "Outscored by ") + number(abs(value)) + " " + units
        }
        return number(value) + " " + units + (market.market == "totals" ? " (both teams)" : "")
    }
    static func result(_ value: Double, market: Market) -> String {
        if isScoringMargin(market) {
            if value == 0 { return "Tied" }
            let units = abs(value) == 1 ? String(scoringUnit(market).dropLast()) : scoringUnit(market)
            return (value > 0 ? "Won by " : "Lost by ") + String(format: "%g", value.magnitude) + " " + units
        }
        return String(format: "%g", value) + " " + unit(market)
    }
    static func range(low: Double, high: Double, market: Market) -> String {
        if isScoringMargin(market) { return result(low, market: market) + " · " + result(high, market: market) }
        return String(format: "%g–%g", low, high) + " " + unit(market)
    }
    static func comparison(_ summary: AnalyticsHistorySummary, market: Market) -> String {
        let counted = summary.hits + summary.misses
        guard counted > 0 else { return "No games to compare" }
        return (market.market == "h2h" ? "Won " : "Met this line in ") + "\(summary.hits) of \(counted) games"
    }
    static func exclusions(_ summary: AnalyticsHistorySummary, market: Market) -> String? {
        var notes: [String] = []
        if summary.pushes > 0 {
            let games = "\(summary.pushes) " + (summary.pushes == 1 ? "game" : "games")
            notes.append(games + (market.market == "h2h" ? " ended in a tie" : " landed exactly on the line"))
        }
        let unavailable = summary.voids + summary.missing
        if unavailable > 0 {
            if market.market == "h2h" {
                notes.append("\(unavailable) " + (unavailable == 1 ? "game was tied or had" : "games were tied or had") + " no usable result")
            } else {
                notes.append("\(unavailable) " + (unavailable == 1 ? "game had" : "games had") + " no usable result")
            }
        }
        guard !notes.isEmpty else { return nil }
        return "Not counted: " + notes.joined(separator: "; ") + "."
    }
}

/// Descriptive evidence only. Uses the already-returned observations; no provider or model calls.
enum AnalyticsTrendInsights {
    static func make(candidate: AnalyticsFeedCandidate, history: AnalyticsHistoryResult, now: Date = Date()) -> [AnalyticsTrendInsight] {
        guard history.observations.count <= 2_000, !history.source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        if candidate.subject == .player {
            if history.source == "MLB Stats API", candidate.market.sport != "MLB" { return [] }
            guard let playerID = history.verifiedPlayerID, !playerID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
            if let proof = candidate.verifiedSubject {
                guard proof.subject == .player, proof.sport == candidate.market.sport,
                      AnalyticsOddsMatcher.normalized(proof.name) == AnalyticsOddsMatcher.normalized(candidate.market.name),
                      proof.providerID == playerID, let url = URL(string: proof.sourceURL), PlayerHistoryService.allowedURL(url) else { return [] }
                let mlbProof = url.host == "statsapi.mlb.com"
                if history.source == "MLB Stats API", !mlbProof { return [] }
                if history.source == "ESPN", mlbProof { return [] }
            }
        }
        let rows = AnalyticsHistoryEngine.validObservations(market: candidate.market, observations: history.observations, now: now)
        guard !rows.isEmpty else { return [] }
        var insights: [AnalyticsTrendInsight] = []
        func average(_ value: Double) -> String { AnalyticsCardCopy.average(value, market: candidate.market) }
        func card(_ kind: AnalyticsTrendInsightKind, title: String, headline: String, detail: String, sample: [AnalyticsObservation]) -> AnalyticsTrendInsight {
            AnalyticsTrendInsight(id: candidate.id + "|" + kind.rawValue, kind: kind, title: title, headline: headline, detail: detail, sample: sample.count, startDate: sample.map(\.date).min(), endDate: sample.map(\.date).max(), source: history.source)
        }
        let recent = Array(rows.prefix(5)), baseline = Array(rows.dropFirst(5).prefix(20))
        if recent.count == 5, baseline.count >= 5, let recentMean = mean(recent.compactMap(\.value)), let baselineMean = mean(baseline.compactMap(\.value)) {
            insights.append(card(.recentBaseline, title: "Last 5 games", headline: average(recentMean), detail: "Previously: \(average(baselineMean)) in the \(baseline.count) games immediately before them.", sample: recent + baseline))
        }
        if rows.count >= 5 {
            let values = rows.compactMap(\.value).sorted()
            if let low = values.first, let high = values.last {
                insights.append(card(.consistency, title: "Recent highs & lows", headline: AnalyticsCardCopy.range(low: low, high: high, market: candidate.market), detail: "Lowest and highest results in these \(values.count) games with usable statistics.", sample: rows))
            }
        }
        // Unknown/neutral venue metadata is never inferred from a matchup name.
        let home = rows.filter { $0.homeAway == "home" }, away = rows.filter { $0.homeAway == "away" }
        if home.count >= 3, away.count >= 3, let homeMean = mean(home.compactMap(\.value)), let awayMean = mean(away.compactMap(\.value)) {
            insights.append(card(.homeAway, title: "At home", headline: average(homeMean), detail: "Away: \(average(awayMean)). Based on \(home.count) home games and \(away.count) away games; neutral or unknown venues aren't counted.", sample: home + away))
        }
        // Only source-attested season metadata may define a season; calendar dates cannot.
        if recent.count == 5, let season = recent.first?.season, (2000...2100).contains(season), recent.allSatisfy({ $0.season == season }) {
            let earlierSeason = rows.dropFirst(5).filter { $0.season == season }
            if earlierSeason.count >= 10, let recentMean = mean(recent.compactMap(\.value)), let seasonMean = mean(earlierSeason.compactMap(\.value)) {
                insights.append(card(.seasonBaseline, title: "Last 5 games this season", headline: average(recentMean), detail: "Previously: \(average(seasonMean)) in \(earlierSeason.count) earlier recorded games from the same \(season) season. Available results may not cover the whole season.", sample: recent + earlierSeason))
            }
        }
        if let opponent = verifiedOpponent(candidate: candidate, history: history, now: now) {
            let opposing = rows.filter { AnalyticsOddsMatcher.normalized($0.opponent) == AnalyticsOddsMatcher.normalized(opponent) }
            if opposing.count >= 3, let opponentMean = mean(opposing.compactMap(\.value)) {
                insights.append(card(.opponent, title: "Against \(opponent)", headline: average(opponentMean), detail: "Based on the last \(opposing.count) recorded games against this opponent.", sample: opposing))
            }
        }
        return insights
    }
    static func verifiedOpponent(candidate: AnalyticsFeedCandidate, history: AnalyticsHistoryResult, now: Date) -> String? {
        guard let proof = candidate.verifiedSubject, let event = proof.event,
              let url = URL(string: proof.sourceURL), PlayerHistoryService.allowedURL(url),
              proof.sport == candidate.market.sport, proof.subject == candidate.subject,
              AnalyticsOddsMatcher.normalized(proof.name) == AnalyticsOddsMatcher.normalized(candidate.market.name) else { return nil }
        let teamName = proof.subject == .team ? proof.name : proof.teamName ?? ""
        let teamID = proof.subject == .team ? proof.providerID : proof.teamID
        if proof.subject == .player {
            guard history.verifiedPlayerID == proof.providerID,
                  let teamID, url.path.hasSuffix("/teams/\(teamID)/roster") else { return nil }
            if history.source == "MLB Stats API", url.host != "statsapi.mlb.com" { return nil }
        }
        let target = AnalyticsStatsTarget(sport: proof.sport, kind: proof.subject == .player ? "prop" : "team", name: proof.name, marketKey: candidate.market.market, teamName: teamName, teamID: teamID, playerID: proof.subject == .player ? proof.providerID : nil, rosterVerifiedAt: proof.subject == .player ? proof.verifiedAt : nil)
        guard proof.verifiedAt <= now, now.timeIntervalSince(proof.verifiedAt) <= 900, AnalyticsOddsMatcher.eligible(target: target, event: event, now: now, recordedSnapshot: candidate.recordedSnapshot, recordedArchive: candidate.recordedArchive) else { return nil }
        return AnalyticsOddsMatcher.normalized(teamName) == AnalyticsOddsMatcher.normalized(event.home) ? event.away : event.home
    }
    private static func mean(_ values: [Double]) -> Double? {
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { return nil }
        let scale = values.map(abs).max() ?? 0
        guard scale > 0 else { return 0 }
        let normalized = values.reduce(0) { $0 + ($1 / scale) / Double(values.count) }
        let result = min(max(normalized, -1), 1) * scale
        return result.isFinite ? result : nil
    }
}


/// Presentation windows share the history engine's identity, cutoff and exclusion rules.
/// The wider window is opt-in; existing last-ten callers retain their original bounds.
enum AnalyticsResearchWindow: String, CaseIterable, Identifiable {
    case last5 = "L5", last10 = "L10", last20 = "L20", season = "Season", opponent = "H2H"
    var id: String { rawValue }
    var limit: Int { switch self { case .last5: return 5; case .last10: return 10; case .last20: return 20; default: return 2_000 } }
    var accessibilityLabel: String {
        switch self { case .last5: return "Last 5 games"; case .last10: return "Last 10 games"; case .last20: return "Last 20 games"; case .season: return "Latest recorded season"; case .opponent: return "Against this opponent" }
    }
}
struct AnalyticsResearchSample {
    let statistics: AnalyticsHistoryResult
    let comparison: AnalyticsHistoryResult?
    let label: String
    let note: String?
}
enum AnalyticsResearch {
    static func subjectID(_ candidate: AnalyticsFeedCandidate) -> String {
        [candidate.market.sport, candidate.subject.rawValue, candidate.verifiedSubject?.providerID ?? "",
         candidate.market.name, candidate.verifiedSubject?.teamID ?? ""].map { "\($0.utf8.count):" + $0 }.joined(separator: "|")
    }
    static func statLabel(_ market: Market) -> String {
        if market.market == "h2h" || market.market == "spreads" { return "Win margin" }
        if market.market == "team_totals" { return "Team scoring" }
        if market.market == "totals" { return "Game total" }
        return AnalyticsCardCopy.perGameUnit(market).replacingOccurrences(of: "/GM", with: "")
    }
    static func chartLine(_ quote: Market) -> Double? {
        if quote.market == "h2h" { return 0 }
        guard let line = quote.line, line.isFinite else { return nil }
        return quote.market == "spreads" ? -line : line
    }
    static func sample(candidate: AnalyticsFeedCandidate, history: AnalyticsHistoryResult, window: AnalyticsResearchWindow,
                       quote: Market?, now: Date = Date()) -> AnalyticsResearchSample {
        var market = candidate.market
        if let event = candidate.verifiedSubject?.event {
            market.event = event.providerID; market.commence = ISO8601DateFormatter().string(from: event.start)
        }
        let identityMatches = candidate.subject != .player || (history.verifiedPlayerID != nil && history.verifiedPlayerID == candidate.verifiedSubject?.providerID)
        let sourceMatches = history.source != "MLB Stats API" || candidate.verifiedSubject.flatMap { URL(string: $0.sourceURL)?.host } == "statsapi.mlb.com"
        let input = identityMatches && sourceMatches && history.observations.count <= 2_000 ? history.observations : []
        // Deduplicate before filtering metadata, so conflicting copies cannot be rescued by a filter.
        let all = AnalyticsHistoryEngine.statistics(market: market, observations: input, limit: 2_000, now: now, maximumWindow: 2_000)
        var rows = all.statsRows
        var label = window.accessibilityLabel
        var note: String?
        switch window {
        case .season:
            if let season = rows.compactMap(\.season).first(where: { (2000...2100).contains($0) }) {
                rows = rows.filter { $0.season == season }; label = "\(season) season · Available games"
            } else { rows = []; note = "Season information isn’t available for these results." }
        case .opponent:
            if let opponent = AnalyticsTrendInsights.verifiedOpponent(candidate: candidate, history: history, now: now) {
                rows = rows.filter { AnalyticsOddsMatcher.normalized($0.opponent) == AnalyticsOddsMatcher.normalized(opponent) }
                label = "Against " + opponent
            } else { rows = []; note = "A verified opponent is needed for this comparison." }
        default: break
        }
        rows = Array(rows.prefix(window.limit))
        let stats = AnalyticsHistoryEngine.statistics(market: market, observations: rows, limit: window.limit, now: now, maximumWindow: 2_000)
        // Only a currently eligible, exact recorded contract may color the chart.
        let eligible = AnalyticsFeedCore.cardOdds(candidate: candidate, history: history, now: now)
        let selected = quote.flatMap { proposed in eligible.first(where: { $0 == proposed }) }
        let comparison = selected.map { AnalyticsHistoryEngine.evaluate(market: $0, observations: rows, limit: window.limit, now: now, maximumWindow: 2_000) }
        if stats.statsRows.isEmpty, note == nil { note = "No recorded games match this view yet." }
        return AnalyticsResearchSample(statistics: stats, comparison: comparison, label: label, note: note)
    }
}

// Uses the same usable appearances as the selected historical window. This is a
// sample standard deviation, never a projected interval or probability.
extension AnalyticsResearch {
 static func standardDeviation(_ values: [Double]) -> Double? {
  guard values.count >= 2, values.allSatisfy(\.isFinite) else { return nil }
  var mean = 0.0, squaredDifferences = 0.0
  for (index, value) in values.enumerated() {
   let delta = value - mean
   mean += delta / Double(index + 1)
   squaredDifferences += delta * (value - mean)
  }
  guard mean.isFinite, squaredDifferences.isFinite else { return nil }
  let deviation = sqrt(max(squaredDifferences / Double(values.count - 1), 0))
  return deviation.isFinite ? deviation : nil
 }
}

// Descriptive least-squares fit over the displayed completed games. No extrapolation.
extension AnalyticsResearch {
 struct TrendPoint: Identifiable, Equatable {
  let index: Int
  let value: Double
  var id: Int { index }
 }
 static func trendLine(_ chronologicalRows: [AnalyticsObservation]) -> [TrendPoint] {
  let points = chronologicalRows.enumerated().compactMap { index, row -> TrendPoint? in
   guard row.status == .completed, let value = row.value, value.isFinite else { return nil }
   return TrendPoint(index: index, value: value)
  }
  guard points.count >= 3, let first = points.first, let last = points.last else { return [] }
  let meanX = points.reduce(0.0) { $0 + Double($1.index) } / Double(points.count)
  let meanY = points.reduce(0.0) { $0 + $1.value } / Double(points.count)
  let xx = points.reduce(0.0) { $0 + pow(Double($1.index) - meanX, 2) }
  let xy = points.reduce(0.0) { $0 + (Double($1.index) - meanX) * ($1.value - meanY) }
  guard xx > 0, xx.isFinite, xy.isFinite, meanY.isFinite else { return [] }
  let slope = xy / xx
  let fit = [first.index, last.index].map { TrendPoint(index: $0, value: meanY + slope * (Double($0) - meanX)) }
  return fit.allSatisfy { $0.value.isFinite } ? fit : []
 }
}
