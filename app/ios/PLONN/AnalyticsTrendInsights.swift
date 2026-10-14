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
        let metric = metricLabel(candidate)
        let definition = candidate.subject == .team && ["h2h", "spreads"].contains(candidate.market.market)
            ? " Score difference is this team's score minus its opponent's; positive means a win and negative means a loss." : ""
        func card(_ kind: AnalyticsTrendInsightKind, title: String, headline: String, detail: String, sample: [AnalyticsObservation]) -> AnalyticsTrendInsight {
            AnalyticsTrendInsight(id: candidate.id + "|" + kind.rawValue, kind: kind, title: title, headline: headline, detail: detail, sample: sample.count, startDate: sample.map(\.date).min(), endDate: sample.map(\.date).max(), source: history.source)
        }
        let recent = Array(rows.prefix(5)), baseline = Array(rows.dropFirst(5).prefix(20))
        if recent.count == 5, baseline.count >= 5, let recentMean = mean(recent.compactMap(\.value)), let baselineMean = mean(baseline.compactMap(\.value)) {
            insights.append(card(.recentBaseline, title: "Last 5 vs earlier games", headline: "\(number(recentMean)) vs \(number(baselineMean)) \(metric)", detail: "Average in the last 5 games with usable statistics, compared with the \(baseline.count) games immediately before them. Each game appears in only one group." + definition, sample: recent + baseline))
        }
        if rows.count >= 5 {
            let values = rows.compactMap(\.value).sorted()
            if let median = quantile(values, fraction: 0.5), let q1 = quantile(values, fraction: 0.25), let q3 = quantile(values, fraction: 0.75), populationSD(values) != nil, (q3 - q1).isFinite {
                insights.append(card(.consistency, title: "How results vary", headline: "Middle result \(number(median))", detail: "Last \(values.count) games with usable \(metric) statistics. The middle result is halfway through the ordered values, averaging the two center values when needed. The middle-half range is \(number(q1)) to \(number(q3)), with boundaries estimated between recorded values." + definition, sample: rows))
            }
        }
        // Unknown/neutral venue metadata is never inferred from a matchup name.
        let home = rows.filter { $0.homeAway == "home" }, away = rows.filter { $0.homeAway == "away" }
        if home.count >= 3, away.count >= 3, let homeMean = mean(home.compactMap(\.value)), let awayMean = mean(away.compactMap(\.value)) {
            insights.append(card(.homeAway, title: "Home and away games", headline: "Home \(number(homeMean)) · Away \(number(awayMean))", detail: "Average \(metric) in the last \(home.count) recorded home games and \(away.count) recorded away games. Neutral sites and games without a verified venue are excluded." + definition, sample: home + away))
        }
        // Only source-attested season metadata may define a season; calendar dates cannot.
        if recent.count == 5, let season = recent.first?.season, (2000...2100).contains(season), recent.allSatisfy({ $0.season == season }) {
            let earlierSeason = rows.dropFirst(5).filter { $0.season == season }
            if earlierSeason.count >= 10, let recentMean = mean(recent.compactMap(\.value)), let seasonMean = mean(earlierSeason.compactMap(\.value)) {
                insights.append(card(.seasonBaseline, title: "This season's recorded games", headline: "Last 5 \(number(recentMean)) · Earlier \(number(seasonMean))", detail: "Average \(metric) in the last 5 games versus the \(earlierSeason.count) earlier recorded games from the same \(season) season. Each game appears in only one group; available records may not cover the whole season." + definition, sample: recent + earlierSeason))
            }
        }
        if let opponent = verifiedOpponent(candidate: candidate, history: history, now: now) {
            let opposing = rows.filter { AnalyticsOddsMatcher.normalized($0.opponent) == AnalyticsOddsMatcher.normalized(opponent) }
            if opposing.count >= 3, let opponentMean = mean(opposing.compactMap(\.value)) {
                insights.append(card(.opponent, title: "Against this opponent", headline: "\(number(opponentMean)) \(metric) vs \(opponent)", detail: "Average in the last \(opposing.count) recorded games against \(opponent). The matchup and team membership are verified." + definition, sample: opposing))
            }
        }
        return insights
    }
    private static func verifiedOpponent(candidate: AnalyticsFeedCandidate, history: AnalyticsHistoryResult, now: Date) -> String? {
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
    private static func metricLabel(_ candidate: AnalyticsFeedCandidate) -> String {
        if candidate.subject == .player { return OpponentPropHistoryEngine.statLabel(AnalyticsHistoryEngine.statisticKey(sport: candidate.market.sport, marketKey: candidate.market.market) ?? candidate.market.market) }
        switch candidate.market.market { case "team_totals": return "team score"; case "totals": return "combined score"; default: return "score difference" }
    }
    private static func mean(_ values: [Double]) -> Double? {
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { return nil }
        let scale = values.map(abs).max() ?? 0
        guard scale > 0 else { return 0 }
        let normalized = values.reduce(0) { $0 + ($1 / scale) / Double(values.count) }
        let result = min(max(normalized, -1), 1) * scale
        return result.isFinite ? result : nil
    }
    private static func quantile(_ sorted: [Double], fraction: Double) -> Double? {
        guard !sorted.isEmpty else { return nil }
        let position = Double(sorted.count - 1) * fraction
        let lower = Int(position.rounded(.down)), upper = Int(position.rounded(.up)), weight = position - Double(lower)
        let result = sorted[lower] * (1 - weight) + sorted[upper] * weight
        return result.isFinite ? result : nil
    }
    private static func populationSD(_ values: [Double]) -> Double? {
        guard let average = mean(values), let scale = values.map(abs).max() else { return nil }
        guard scale > 0 else { return 0 }
        let scaledAverage = average / scale
        let variance = values.reduce(0) { partial, value in
            let difference = value / scale - scaledAverage
            return partial + difference * difference / Double(values.count)
        }
        let result = sqrt(variance) * scale
        return result.isFinite ? result : nil
    }
    private static func number(_ value: Double) -> String { String(format: abs(value) >= 1_000_000 ? "%.4g" : "%.1f", locale: Locale(identifier: "en_US_POSIX"), value) }
}
