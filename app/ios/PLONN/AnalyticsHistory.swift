import Foundation
import CoreFoundation

enum AnalyticsObservationStatus: String, Sendable { case completed, void, missing }
enum AnalyticsOutcome: String, Sendable { case hit, miss, push, void, missing }
struct AnalyticsObservation: Identifiable, Sendable, Equatable {
    let id: String
    let date: Date
    let opponent: String
    var value: Double? = nil
    var scored: Double? = nil
    var allowed: Double? = nil
    var status: AnalyticsObservationStatus = .completed
    var homeAway: String? = nil
    var season: Int? = nil
}
struct AnalyticsHistoryRow: Identifiable, Sendable {
    let id: String
    let date: Date
    let opponent: String
    let value: Double?
    let outcome: AnalyticsOutcome
}
struct AnalyticsHistorySummary: Sendable {
    let n: Int
    let hits: Int
    let misses: Int
    let pushes: Int
    let voids: Int
    let missing: Int
    let hitFrequency: Double?
    let average: Double?
    let startDate: Date?
    let endDate: Date?
    let smallSample: Bool
    let line: Double?
    let evidence: String
}
struct AnalyticsStatisticsSummary: Sendable {
    let n: Int
    let valid: Int
    let missing: Int
    let voids: Int
    let startDate: Date?
    let endDate: Date?
    let mean: Double?
    let minimum: Double?
    let maximum: Double?
    let teamWins: Int?
    let teamLosses: Int?
    let teamTies: Int?
    let evidence: String
}
struct AnalyticsHistoryResult: Sendable {
    let summary: AnalyticsHistorySummary?
    let rows: [AnalyticsHistoryRow]
    let observations: [AnalyticsObservation]
    let source: String
    let warnings: [String]
    let availability: String
    var sourceURLs: [String] = []
    var statistics: AnalyticsStatisticsSummary? = nil
    var statsRows: [AnalyticsObservation] = []
    var verifiedPlayerID: String? = nil
    /// Historical roster candidate only; current membership must be separately verified.
    var candidateTeamID: String? = nil
}
enum AnalyticsHistoryEngine {
    /// Alternate full-game lines measure the same recorded statistic. Keep the
    /// original quote market, side and line; map only the historical stat request.
    static func statisticKey(sport: String, marketKey: String) -> String? {
        if PlayerHistoryService.statKeys(sport: sport, marketKey: marketKey) != nil { return marketKey }
        guard marketKey.hasSuffix("_alternate") else { return nil }
        let base = String(marketKey.dropLast("_alternate".count))
        return PlayerHistoryService.statKeys(sport: sport, marketKey: base) != nil ? base : nil
    }

    /// Full-game team totals use only the selected team's recorded score.
    private static func teamValue(marketKey: String, scored: Double?, allowed: Double?) -> Double? {
        guard let own = scored, own.isFinite, own >= 0, own.rounded() == own else { return nil }
        if marketKey == "team_totals" { return own }
        guard let other = allowed, other.isFinite, other >= 0, other.rounded() == other else { return nil }
        return marketKey == "totals" ? own + other : own - other
    }
    static func statsUnavailableReason(market: Market, now: Date = Date()) -> String? {
        let prop = market.kind == "prop"
        guard ["NBA", "NFL", "MLB", "NHL"].contains(market.sport) else { return "History is unavailable for this sport." }
        guard !market.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "Choose a player or full team name." }
        if market.market == "totals", ["over", "under"].contains(market.name.lowercased()), market.opponent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Choose the full team name for this total comparison." }
        if market.market == "team_totals", market.kind != "team" || ["over", "under"].contains(market.name.lowercased()) { return "Choose the verified full team name for this team total." }
        let supported = prop ? statisticKey(sport: market.sport, marketKey: market.market) != nil : ["h2h", "spreads", "totals", "team_totals"].contains(market.market)
        guard supported else { return "History is unavailable for this market; period and three-way contracts are unsupported." }
        guard now.timeIntervalSince1970.isFinite else { return "The history cutoff is invalid." }
        if let commence = market.commence, sportsDate(commence) == nil { return "The selected event date is invalid." }
        return nil
    }
    /// Shared input eligibility; nil means a supported comparison, without any provider read.
    static func unavailableReason(market: Market, line: Double? = nil, now: Date = Date()) -> String? {
        let selectedLine = line ?? market.line
        let side = (market.side ?? market.name).lowercased()
        let prop = market.kind == "prop"
        if let reason = statsUnavailableReason(market: market, now: now) { return reason }
        if prop || ["totals", "team_totals"].contains(market.market) {
            guard ["over", "under"].contains(side) else { return "Choose an explicit Over or Under side." }
        }
        if market.market != "h2h" {
            guard let selectedLine, selectedLine.isFinite else { return "Choose a finite comparison line." }
            let signedYards = market.sport == "NFL" && ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_rush_reception_yds"].contains(statisticKey(sport: market.sport, marketKey: market.market) ?? market.market)
            guard market.market == "spreads" || signedYards || selectedLine >= 0 else { return "This comparison requires a nonnegative line." }
        }
        guard now.timeIntervalSince1970.isFinite else { return "The history cutoff is invalid." }
        if let commence = market.commence, sportsDate(commence) == nil { return "The selected event date is invalid." }
        return nil
    }
    private static func observationValue(market: Market, observation: AnalyticsObservation) -> Double? {
        guard observation.status == .completed else { return nil }
        if market.kind != "prop" { return teamValue(marketKey: market.market, scored: observation.scored, allowed: observation.allowed) }
        let signed = market.sport == "NFL" && ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_rush_reception_yds"].contains(statisticKey(sport: market.sport, marketKey: market.market) ?? market.market)
        return observation.value.flatMap { $0.isFinite && $0.rounded() == $0 && ($0 >= 0 || signed) ? $0 : nil }
    }
    /// All valid prior observations, without the presentation window's five/ten-row cap.
    /// Metadata stays explicit: absent venue or season is never reconstructed from dates.
    static func validObservations(market: Market, observations: [AnalyticsObservation], now: Date = Date()) -> [AnalyticsObservation] {
        guard statsUnavailableReason(market: market, now: now) == nil else { return [] }
        let cutoff = min(market.commence.flatMap(sportsDate) ?? now, now)
        return Dictionary(grouping: observations, by: \.id).compactMap { id, copies -> AnalyticsObservation? in
            guard !id.isEmpty, id != market.event, let first = copies.first,
                  copies.allSatisfy({ $0 == first }), first.date.timeIntervalSince1970.isFinite,
                  first.date < cutoff, let value = observationValue(market: market, observation: first), value.isFinite else { return nil }
            var row = first
            row.value = value
            if !["home", "away"].contains(row.homeAway ?? "") { row.homeAway = nil }
            if let season = row.season, !(2000...2100).contains(season) { row.season = nil }
            return row
        }.sorted { $0.date == $1.date ? $0.id < $1.id : $0.date > $1.date }
    }
    static func statistics(market: Market, observations: [AnalyticsObservation], limit: Int = 10, now: Date = Date()) -> AnalyticsHistoryResult {
        if let reason = statsUnavailableReason(market: market, now: now) {
            return AnalyticsHistoryResult(summary: nil, rows: [], observations: observations, source: "Recorded game results", warnings: [], availability: reason)
        }
        let cutoff = min(market.commence.flatMap(sportsDate) ?? now, now)
        var warnings: [String] = []
        var selected: [AnalyticsObservation] = []
        for (id, copies) in Dictionary(grouping: observations, by: \.id) {
            guard !id.isEmpty, id != market.event, let first = copies.first else { continue }
            guard copies.allSatisfy({ $0 == first }) else { warnings.append("Conflicting copies of event \(id) were excluded."); continue }
            guard first.date.timeIntervalSince1970.isFinite, first.date < cutoff else { continue }
            selected.append(first)
        }
        selected = Array(selected.sorted { $0.date == $1.date ? $0.id < $1.id : $0.date > $1.date }.prefix(min(max(limit, 1), 10)))
        let prop = market.kind == "prop"
        var values: [Double] = [], wins = 0, losses = 0, ties = 0, missing = 0, voids = 0
        let rows = selected.map { original -> AnalyticsObservation in
            var row = original
            if row.status == .void { voids += 1; return row }
            var value: Double?
            if row.status == .completed {
                value = observationValue(market: market, observation: row)
                if !prop {
                    if value?.isFinite == true, let own = row.scored, let other = row.allowed, other.isFinite, other >= 0, other.rounded() == other {
                        if own > other { wins += 1 } else if own < other { losses += 1 } else { ties += 1 }
                    }
                }
            }
            if let value, value.isFinite { row.value = value; values.append(value) }
            else { row.value = nil; row.status = .missing; missing += 1 }
            return row
        }
        let average = values.isEmpty ? nil : values.reduce(0) { $0 + $1 / Double(values.count) }
        let evidence = "Last \(rows.count) recorded observations before the selected event and now. \(values.count) valid, \(missing) missing and \(voids) observed exclusions. Actual statistics only; no sportsbook line, forecast, probability or settled-bet inference."
        let summary = AnalyticsStatisticsSummary(n: rows.count, valid: values.count, missing: missing, voids: voids, startDate: rows.last?.date, endDate: rows.first?.date, mean: average.flatMap { $0.isFinite ? $0 : nil }, minimum: values.min(), maximum: values.max(), teamWins: prop ? nil : wins, teamLosses: prop ? nil : losses, teamTies: prop ? nil : ties, evidence: evidence)
        return AnalyticsHistoryResult(summary: nil, rows: [], observations: observations, source: "Recorded game results", warnings: warnings, availability: rows.isEmpty ? "No verified prior observations are available." : "Recorded historical statistics.", statistics: rows.isEmpty ? nil : summary, statsRows: rows)
    }
    static func evaluate(market: Market, observations: [AnalyticsObservation], line: Double? = nil, limit: Int = 10, now: Date = Date()) -> AnalyticsHistoryResult {
        let selectedLine = line ?? market.line
        let side = (market.side ?? market.name).lowercased()
        let prop = market.kind == "prop"
        func unavailable(_ reason: String) -> AnalyticsHistoryResult {
            AnalyticsHistoryResult(summary: nil, rows: [], observations: observations, source: "Recorded game results", warnings: [], availability: reason)
        }
        if let reason = unavailableReason(market: market, line: line, now: now) { return unavailable(reason) }
        let cutoff = min(market.commence.flatMap(sportsDate) ?? now, now)
        var warnings: [String] = []
        let groups = Dictionary(grouping: observations, by: \.id)
        var cleaned: [AnalyticsObservation] = []
        for (id, copies) in groups {
            guard !id.isEmpty, id != market.event, let first = copies.first else { continue }
            guard copies.allSatisfy({ $0 == first }) else { warnings.append("Conflicting copies of event \(id) were excluded."); continue }
            guard first.date.timeIntervalSince1970.isFinite, first.date < cutoff else { continue }
            cleaned.append(first)
        }
        cleaned = Array(cleaned.sorted { $0.date == $1.date ? $0.id < $1.id : $0.date > $1.date }.prefix(min(max(limit, 1), 10)))
        let rows = cleaned.map { observation -> AnalyticsHistoryRow in
            var value: Double?
            var outcome: AnalyticsOutcome = .missing
            if observation.status == .void { outcome = .void }
            else if observation.status == .completed {
                value = observationValue(market: market, observation: observation)
                if let actual = value, actual.isFinite {
                    let comparison = market.market == "h2h" ? actual : (market.market == "spreads" ? actual + selectedLine! : actual - selectedLine!)
                    if comparison.isFinite {
                        if comparison == 0 { outcome = market.market == "h2h" ? .missing : .push }
                        else { outcome = (side == "under" && (prop || ["totals", "team_totals"].contains(market.market)) ? comparison < 0 : comparison > 0) ? .hit : .miss }
                    }
                }
            }
            return AnalyticsHistoryRow(id: observation.id, date: observation.date, opponent: observation.opponent, value: value, outcome: outcome)
        }
        let hits = rows.filter { $0.outcome == .hit }.count, misses = rows.filter { $0.outcome == .miss }.count
        let pushes = rows.filter { $0.outcome == .push }.count, voids = rows.filter { $0.outcome == .void }.count, missing = rows.filter { $0.outcome == .missing }.count
        let values = rows.filter { $0.outcome != .void }.compactMap(\.value).filter(\.isFinite)
        let average = values.isEmpty ? nil : values.reduce(0) { $0 + $1 / Double(values.count) }
        if market.market == "h2h", missing > 0 { warnings.append("Draws have no verified settlement contract and are excluded from win frequency.") }
        let evidence = "Last \(rows.count) observed games before the selected event and now. Frequency = hits / (hits + misses); \(pushes) equal-line, \(voids) void and \(missing) missing/ambiguous observations excluded. Compared with the current selected line; historical sportsbook lines and bet settlements are unavailable."
        let summary = AnalyticsHistorySummary(n: rows.count, hits: hits, misses: misses, pushes: pushes, voids: voids, missing: missing, hitFrequency: hits + misses > 0 ? Double(hits) / Double(hits + misses) : nil, average: average.flatMap { $0.isFinite ? $0 : nil }, startDate: rows.last?.date, endDate: rows.first?.date, smallSample: hits + misses < 5, line: market.market == "h2h" ? nil : selectedLine, evidence: evidence)
        return AnalyticsHistoryResult(summary: rows.isEmpty ? nil : summary, rows: rows, observations: observations, source: "Recorded game results", warnings: warnings, availability: rows.isEmpty ? "No verified prior observations are available." : "Historical observations against the current selected line.")
    }
}

/// At most five public ESPN reads per team load: directory and two season types for two seasons.
actor AnalyticsHistoryService {
    typealias Reader = @Sendable (URL) async throws -> Data
    static let shared = AnalyticsHistoryService()
    private let reader: Reader
    init(reader: @escaping Reader = { try await PlayerHistoryService.shared.read($0) }) { self.reader = reader }
    func load(market: Market, limit: Int = 10, now: Date = Date()) async -> AnalyticsHistoryResult {
        if let reason = AnalyticsHistoryEngine.unavailableReason(market: market, now: now) {
            return AnalyticsHistoryResult(summary: nil, rows: [], observations: [], source: "Recorded game results", warnings: [], availability: reason)
        }
        let stats = await loadStats(market: market, limit: limit, now: now)
        let comparison = AnalyticsHistoryEngine.evaluate(market: market, observations: stats.observations, limit: limit, now: now)
        return AnalyticsHistoryResult(summary: comparison.summary, rows: comparison.rows, observations: stats.observations, source: stats.source, warnings: stats.warnings + comparison.warnings, availability: stats.observations.isEmpty ? stats.availability : comparison.availability, sourceURLs: stats.sourceURLs, statistics: stats.statistics, statsRows: stats.statsRows, verifiedPlayerID: stats.verifiedPlayerID, candidateTeamID: stats.candidateTeamID)
    }
    func loadStats(market: Market, limit: Int = 10, now: Date = Date()) async -> AnalyticsHistoryResult {
        if let reason = AnalyticsHistoryEngine.statsUnavailableReason(market: market, now: now) {
            return AnalyticsHistoryResult(summary: nil, rows: [], observations: [], source: "Recorded game results", warnings: [], availability: reason)
        }
        if market.kind == "prop" {
            let history = await PlayerHistoryService.shared.load(market: market, includeOpponent: false, seasonCount: 2, now: now)
            let observations = history.games.map { AnalyticsObservation(id: $0.id, date: $0.date, opponent: $0.opponent, value: $0.value, homeAway: $0.homeAway, season: $0.season) }
            let evaluated = AnalyticsHistoryEngine.statistics(market: market, observations: observations, limit: limit, now: now)
            return AnalyticsHistoryResult(summary: evaluated.summary, rows: evaluated.rows, observations: observations, source: history.source, warnings: history.warnings + evaluated.warnings + ["Player DNP, void and missing-stat coverage is unavailable; only verified recorded appearances are shown."], availability: observations.isEmpty ? history.availability : evaluated.availability, sourceURLs: history.sourceURLs, statistics: evaluated.statistics, statsRows: evaluated.statsRows, verifiedPlayerID: history.playerID, candidateTeamID: market.sport == "MLB" ? nil : history.games.first?.playerTeamID)
        }
        var urls: [String] = [], warnings: [String] = [], observations: [AnalyticsObservation] = []
        func empty(_ reason: String) -> AnalyticsHistoryResult { AnalyticsHistoryResult(summary: nil, rows: [], observations: [], source: "ESPN", warnings: warnings, availability: reason, sourceURLs: urls) }
        guard ["NBA", "NFL", "MLB", "NHL"].contains(market.sport), ["h2h", "spreads", "totals", "team_totals"].contains(market.market) else { return empty("Team history is unavailable for this sport or contract.") }
        do {
            try Task.checkCancellation()
            let league = PlayerHistoryService.leaguePath(market.sport)
            let directoryURL = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(league)/teams?limit=1000")!
            urls.append(directoryURL.absoluteString)
            let directory = try await json(directoryURL)
            let teamName = market.market == "totals" && ["over", "under"].contains(market.name.lowercased()) ? market.opponent : market.name
            guard let teamID = Self.teamID(directory: directory, name: teamName, sport: market.sport) else { return empty("Selected team could not be uniquely matched to ESPN’s full team name.") }
            var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(secondsFromGMT: 0)!
            let anchor = min(market.commence.flatMap(sportsDate) ?? now, now)
            let season = PlayerHistoryService.firstSeason(sport: market.sport, year: calendar.component(.year, from: anchor), month: calendar.component(.month, from: anchor))
            for year in [season, season - 1] {
                for type in [2, 3] {
                    try Task.checkCancellation()
                    let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(league)/teams/\(teamID)/schedule?season=\(year)&seasontype=\(type)")!
                    urls.append(url.absoluteString)
                    do { observations += try Self.schedule(payload: await json(url), teamID: teamID, season: year) }
                    catch is CancellationError { throw CancellationError() }
                    catch { warnings.append("\(year) \(type == 2 ? "regular season" : "postseason") could not be verified.") }
                }
            }
            try Task.checkCancellation()
            let result = AnalyticsHistoryEngine.statistics(market: market, observations: observations, limit: limit, now: now)
            return AnalyticsHistoryResult(summary: result.summary, rows: result.rows, observations: observations, source: "ESPN", warnings: warnings + result.warnings + ["Provider coverage of unrecorded missing games and void/excluded games is unavailable; counts describe only returned verified final-game observations."], availability: result.availability, sourceURLs: urls, statistics: result.statistics, statsRows: result.statsRows)
        } catch is CancellationError { return empty("History request cancelled.") }
        catch { return empty("Couldn’t load verified team history. Try again.") }
    }
    func json(_ url: URL) async throws -> [String: Any] {
        guard PlayerHistoryService.allowedURL(url) else { throw PlayerHistoryError.invalidURL }
        let data = try await reader(url)
        try Task.checkCancellation()
        guard data.count <= 4 * 1024 * 1024, let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw PlayerHistoryError.invalidPayload }
        return payload
    }
    static func teamID(directory: [String: Any], name: String, sport: String? = nil) -> String? {
        guard !name.isEmpty, let sports = directory["sports"] as? [[String: Any]] else { return nil }
        let leagues = sports.flatMap { $0["leagues"] as? [[String: Any]] ?? [] }
        let relevant = leagues.filter { league in
            guard let sport else { return true }
            if let abbreviation = league["abbreviation"] as? String, abbreviation.uppercased() != sport { return false }
            if let slug = league["slug"] as? String, slug.lowercased() != sport.lowercased() { return false }
            return true
        }
        let teams = relevant.flatMap { $0["teams"] as? [[String: Any]] ?? [] }.compactMap { $0["team"] as? [String: Any] }
        let matching = teams.filter { ($0["displayName"] as? String).map { AnalyticsOddsMatcher.normalized($0) == AnalyticsOddsMatcher.normalized(name) } == true }
        guard matching.count == 1, let id = matching[0]["id"] as? String, !id.isEmpty, id.allSatisfy(\.isNumber) else { return nil }
        return id
    }
    static func schedule(payload: [String: Any], teamID: String, season: Int? = nil) throws -> [AnalyticsObservation] {
        guard let events = payload["events"] as? [[String: Any]], events.count <= 500 else { throw PlayerHistoryError.invalidPayload }
        func score(_ value: Any?) -> Double? {
            if let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() { return nil }
            let raw = (value as? [String: Any])?["value"] ?? value
            if let number = raw as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() { return nil }
            let value = (raw as? NSNumber)?.doubleValue ?? (raw as? String).flatMap(Double.init)
            return value.flatMap { $0.isFinite && $0 >= 0 && $0.rounded() == $0 ? $0 : nil }
        }
        func seasonYear(_ value: Any?) -> Int? {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
                  (2000.0...2100.0).contains(number.doubleValue) else { return nil }
            return number.intValue
        }
        let requestedSeason = season.flatMap { (2000...2100).contains($0) ? $0 : nil }
        var result: [AnalyticsObservation] = []
        let copies = Dictionary(grouping: events, by: { $0["id"] as? String ?? "" })
        let conflicting = Set(copies.compactMap { id, rows -> String? in
            guard let first = rows.first else { return nil }
            return rows.allSatisfy { NSDictionary(dictionary: $0).isEqual(to: first) } ? nil : id
        })
        for event in events {
            guard let id = event["id"] as? String, !id.isEmpty, !conflicting.contains(id), let stamp = event["date"] as? String, let date = sportsDate(stamp),
                  let competitions = event["competitions"] as? [[String: Any]], competitions.count == 1,
                  let competition = competitions.first, let status = competition["status"] as? [String: Any], let type = status["type"] as? [String: Any],
                  type["completed"] as? Bool == true, type["state"] as? String == "post",
                  let competitors = competition["competitors"] as? [[String: Any]], competitors.count == 2 else { continue }
            let own = competitors.filter { (($0["team"] as? [String: Any])?["id"] as? String) == teamID }
            let opposing = competitors.filter { (($0["team"] as? [String: Any])?["id"] as? String) != teamID }
            guard own.count == 1, opposing.count == 1, let otherID = (opposing[0]["team"] as? [String: Any])?["id"] as? String, !otherID.isEmpty,
                  let name = (opposing[0]["team"] as? [String: Any])?["displayName"] as? String, !name.isEmpty else { continue }
            let scored = score(own[0]["score"]), allowed = score(opposing[0]["score"])
            let venue = own[0]["homeAway"] as? String
            let venueAllowed = [event, competition].allSatisfy { source in
                guard let flag = source["neutralSite"] else { return true }
                guard let number = flag as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return false }
                return !number.boolValue
            }
            let eventSeason = event["season"] as? [String: Any]
            let explicitSeason = eventSeason.flatMap { seasonYear($0["year"]) }
            let verifiedSeason: Int? = event["season"] == nil ? requestedSeason : explicitSeason.flatMap { year in
                requestedSeason == nil || requestedSeason == year ? year : nil
            }
            result.append(AnalyticsObservation(id: id, date: date, opponent: name, scored: scored, allowed: allowed,
                status: scored == nil || allowed == nil ? .missing : .completed,
                homeAway: venueAllowed && ["home", "away"].contains(venue ?? "") ? venue : nil, season: verifiedSeason))
        }
        return result
    }
}
