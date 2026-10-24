import Foundation

enum AnalyticsEventLifecycle: String, Sendable {
    case scheduled, inProgress, final, cancelled, postponed, suspended, unknown
    var label: String {
        switch self {
        case .scheduled: return "Scheduled"
        case .inProgress: return "In progress · Recorded pregame prices"
        case .final: return "Final"
        case .cancelled: return "Cancelled"
        case .postponed: return "Postponed · Status confirmation needed"
        case .suspended: return "Suspended · Status confirmation needed"
        case .unknown: return "Event status unavailable"
        }
    }
    static func espn(_ type: [String: Any]?) -> Self {
        guard let type, let name = type["name"] as? String, let state = type["state"] as? String,
              let completed = type["completed"] as? Bool else { return .unknown }
        if ["STATUS_CANCELED", "STATUS_CANCELLED"].contains(name) { return .cancelled }
        if name == "STATUS_POSTPONED" { return .postponed }
        if name == "STATUS_SUSPENDED" { return .suspended }
        if state == "post", completed, ["STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_SO", "STATUS_FULL_TIME"].contains(name) { return .final }
        if state == "pre", !completed, name == "STATUS_SCHEDULED" { return .scheduled }
        if state == "in", !completed, ["STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_END_PERIOD", "STATUS_END_QUARTER", "STATUS_OVERTIME", "STATUS_SHOOTOUT"].contains(name) { return .inProgress }
        return .unknown
    }
    static func mlb(_ status: [String: Any]?) -> Self {
        guard let status, let code = status["codedGameState"] as? String,
              let abstract = status["abstractGameState"] as? String else { return .unknown }
        if code == "C" { return .cancelled }
        if code == "D" { return .postponed }
        if ["U", "T"].contains(code) { return .suspended }
        if abstract == "Final", ["F", "O"].contains(code) { return .final }
        if abstract == "Live", code == "I" { return .inProgress }
        if abstract == "Preview", code == "S", status["startTimeTBD"] as? Bool == false { return .scheduled }
        return .unknown
    }
}
struct AnalyticsEvent: Sendable, Equatable {
    let sport: String
    let providerID: String
    let home: String
    let away: String
    let start: Date
    var homeID: String? = nil
    var awayID: String? = nil
    var seasonType: Int? = nil
    var lifecycle: AnalyticsEventLifecycle = .scheduled
    var statusVerifiedAt: Date? = nil
    func displayLifecycle(now: Date) -> AnalyticsEventLifecycle {
        if lifecycle == .scheduled, start <= now { return .unknown }
        if lifecycle == .inProgress {
            guard let verified = statusVerifiedAt, verified <= now, now.timeIntervalSince(verified) <= 60 else { return .unknown }
        }
        return lifecycle
    }
    /// Started events require a fresh, independently verified status and archive opt-in.
    func allowsRecorded(now: Date, archive: Bool) -> Bool {
        guard now.timeIntervalSince1970.isFinite, start.timeIntervalSince1970.isFinite else { return false }
        if lifecycle == .scheduled { return AnalyticsOddsMatcher.recordedWindowContains(start, now: now) }
        guard archive, lifecycle == .inProgress,
              (start <= now || (sport == "NHL" && start.timeIntervalSince(now) <= 600)),
              let verified = statusVerifiedAt, verified <= now, now.timeIntervalSince(verified) <= 60 else { return false }
        return true
    }
}
struct AnalyticsStatsTarget: Sendable, Equatable {
    let sport: String
    let kind: String
    let name: String
    let marketKey: String
    let teamName: String
    var teamID: String? = nil
    var playerID: String? = nil
    var rosterVerifiedAt: Date? = nil
}
struct AnalyticsContextResult: Sendable {
    let event: AnalyticsEvent?
    let target: AnalyticsStatsTarget?
    let availability: String
    var sourceURLs: [String] = []
}
struct AnalyticsScopedEvent: Codable, Sendable, Equatable {
    let home: String
    let away: String
    let commence: String
}
struct AnalyticsScopedOddsRequest: Codable, Sendable {
    var schemaVersion: Int = 1
    let sport: String
    let event: AnalyticsScopedEvent
    let markets: [String]
    enum CodingKeys: String, CodingKey { case schemaVersion = "schema_version", sport, event, markets }
}
struct AnalyticsOddsCoverage: Codable, Sendable {
    let sport: String
    let eventID: String
    let home: String
    let away: String
    let commence: String
    let markets: [String]
    enum CodingKeys: String, CodingKey { case sport, eventID = "event_id", home, away, commence, markets }
}
struct AnalyticsScopedOddsResponse: Codable, Sendable {
    let schemaVersion: Int
    let coverage: AnalyticsOddsCoverage
    let markets: [Market]
    let fetchedAt: String
    let expiresAt: String
    let stale: Bool
    enum CodingKeys: String, CodingKey { case schemaVersion = "schema_version", coverage, markets, fetchedAt = "fetched_at", expiresAt = "expires_at", stale }
}
enum AnalyticsOddsFreshness: String, Sendable { case fresh, stale, unavailable }
struct AnalyticsOddsMatchResult: Sendable {
    let markets: [Market]
    let freshness: AnalyticsOddsFreshness
    let fetchedAt: Date?
    let expiresAt: Date?
    let availability: String
}

/// Pure identity and scope checks. Never ranks subjects or requests additional markets.
enum AnalyticsOddsMatcher {
    static func normalized(_ value: String) -> String {
        value.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }
    static func sameDay(_ lhs: Date, _ rhs: Date) -> Bool {
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        return calendar.isDate(lhs, inSameDayAs: rhs)
    }
    /// Five Eastern calendar dates, matching the bounded shared snapshot discovery window.
    /// Scoped live requests retain their original same-day restriction.
    static func recordedWindowContains(_ start: Date, now: Date) -> Bool {
        guard start.timeIntervalSince1970.isFinite, now.timeIntervalSince1970.isFinite else { return false }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York")!
        let end = calendar.date(byAdding: .day, value: 5, to: calendar.startOfDay(for: now))!
        return start > now && start < end
    }
    static func eligible(target: AnalyticsStatsTarget, event: AnalyticsEvent, now: Date, recordedSnapshot: Bool = false, recordedArchive: Bool = false) -> Bool {
        guard ["NBA", "NFL", "MLB", "NHL"].contains(target.sport), target.sport == event.sport,
              !event.providerID.isEmpty, !normalized(event.home).isEmpty, !normalized(event.away).isEmpty,
              normalized(event.home) != normalized(event.away), event.start.timeIntervalSince1970.isFinite,
              now.timeIntervalSince1970.isFinite, (recordedSnapshot ? event.allowsRecorded(now: now, archive: recordedArchive) : event.start > now && sameDay(event.start, now)),
              [event.home, event.away].map(normalized).contains(normalized(target.teamName)) else { return false }
        if target.kind == "prop" {
            guard (recordedSnapshot ? AnalyticsHistoryEngine.statisticKey(sport: target.sport, marketKey: target.marketKey) != nil : PlayerHistoryService.statKeys(sport: target.sport, marketKey: target.marketKey) != nil),
                  let player = target.playerID, !player.isEmpty, let team = target.teamID, !team.isEmpty,
                  [event.homeID, event.awayID].compactMap({ $0 }).contains(team),
                  let verified = target.rosterVerifiedAt, verified <= now, now.timeIntervalSince(verified) <= 900,
                  !normalized(target.name).isEmpty else { return false }
        } else {
            guard target.kind == "team", ["h2h", "spreads", "totals", "team_totals"].contains(target.marketKey), normalized(target.name) == normalized(target.teamName) else { return false }
        }
        return true
    }
    static func request(target: AnalyticsStatsTarget, event: AnalyticsEvent, now: Date = Date()) -> AnalyticsScopedOddsRequest? {
        guard eligible(target: target, event: event, now: now) else { return nil }
        return AnalyticsScopedOddsRequest(sport: target.sport, event: AnalyticsScopedEvent(home: event.home, away: event.away, commence: ISO8601DateFormatter().string(from: event.start)), markets: [target.marketKey])
    }
    /// Existing snapshots only; no fetched-at time is inferred from individual quote timestamps.
    static func matchCached(target: AnalyticsStatsTarget, event: AnalyticsEvent, markets: [Market], now: Date = Date(), recordedSnapshot: Bool = false, recordedArchive: Bool = false) -> AnalyticsOddsMatchResult {
        if recordedSnapshot, !eligible(target: target, event: event, now: now, recordedSnapshot: true, recordedArchive: recordedArchive) {
            return AnalyticsOddsMatchResult(markets: [], freshness: .unavailable, fetchedAt: nil, expiresAt: nil, availability: "A verified upcoming event and subject are required for recorded comparisons.")
        }
        let labels = Set(["\(normalized(event.away)) at \(normalized(event.home))", "\(normalized(event.away)) @ \(normalized(event.home))"])
        let pair = markets.filter { $0.sport == target.sport && $0.eventLabel.map { labels.contains(normalized($0)) } == true }
        // Thousands of player contracts usually share a single provider start.
        // Preserve every distinct event/time identity without parsing and checking
        // that same instant once for every contract against every other contract.
        let dates = Dictionary(uniqueKeysWithValues: Set(pair.compactMap(\.commence)).compactMap { text -> (String, Date)? in sportsDate(text).map { (text, $0) } })
        let starts = Dictionary(grouping: pair, by: \.event).flatMap { id, rows in
            Set(rows.compactMap(\.commence)).compactMap { text -> (String, Date)? in dates[text].map { (id, $0) } }
        }
        func matchesStart(_ quote: Market) -> Bool {
            guard let text = quote.commence, let start = dates[text] else { return false }
            if recordedSnapshot { return RecordedStartPolicy.accepts(eventID: quote.event, quoteStart: start, start: event.start, sport: event.sport, candidates: starts, now: now, includeStarted: recordedArchive && event.allowsRecorded(now: now, archive: true)) }
            return abs(start.timeIntervalSince(event.start)) <= 60
        }
        let candidates = markets.filter { $0.sport == target.sport && $0.market == target.marketKey && $0.eventLabel.map { labels.contains(normalized($0)) } == true && matchesStart($0) }
        let ids = Set(candidates.map(\.event))
        guard ids.count == 1, let id = ids.first, !id.isEmpty else {
            return AnalyticsOddsMatchResult(markets: [], freshness: .unavailable, fetchedAt: nil, expiresAt: nil, availability: "No unambiguous cached event coverage matches this selected stat.")
        }
        let sameIDs = markets.filter { $0.event == id && $0.market == target.marketKey }
        let stamp = ISO8601DateFormatter().string(from: now)
        // For recorded prices, use the unique provider start time already
        // validated by RecordedStartPolicy. Retain the ESPN event/subject identity.
        var matchingEvent = event
        if recordedSnapshot, let text = candidates.first?.commence, let start = dates[text] {
            matchingEvent = AnalyticsEvent(sport: event.sport, providerID: event.providerID, home: event.home, away: event.away, start: start, homeID: event.homeID, awayID: event.awayID, seasonType: event.seasonType, lifecycle: event.lifecycle, statusVerifiedAt: event.statusVerifiedAt)
        }
        let response = AnalyticsScopedOddsResponse(schemaVersion: 1, coverage: AnalyticsOddsCoverage(sport: target.sport, eventID: id, home: event.home, away: event.away, commence: ISO8601DateFormatter().string(from: matchingEvent.start), markets: [target.marketKey]), markets: sameIDs, fetchedAt: stamp, expiresAt: stamp, stale: true)
        let result = match(target: target, event: matchingEvent, response: response, now: now, recordedSnapshot: recordedSnapshot, recordedArchive: recordedArchive)
        return AnalyticsOddsMatchResult(markets: result.markets, freshness: result.markets.isEmpty ? .unavailable : .stale, fetchedAt: nil, expiresAt: nil, availability: result.markets.isEmpty ? result.availability : "Cached matching lines; snapshot freshness is unverified. Each quote retains its own update time.")
    }
    static func match(target: AnalyticsStatsTarget, event: AnalyticsEvent, response: AnalyticsScopedOddsResponse, now: Date = Date(), recordedSnapshot: Bool = false, recordedArchive: Bool = false) -> AnalyticsOddsMatchResult {
        var parsedDates: [String: Date] = [:]
        func date(_ text: String) -> Date? {
            if let cached = parsedDates[text] { return cached }
            guard let parsed = sportsDate(text) else { return nil }
            parsedDates[text] = parsed; return parsed
        }
        let fetched = date(response.fetchedAt), expires = date(response.expiresAt)
        func empty(_ message: String) -> AnalyticsOddsMatchResult { AnalyticsOddsMatchResult(markets: [], freshness: .unavailable, fetchedAt: fetched, expiresAt: expires, availability: message) }
        guard eligible(target: target, event: event, now: now, recordedSnapshot: recordedSnapshot, recordedArchive: recordedArchive) else { return empty("A unique same-day open event and verified subject membership are required.") }
        let scope = response.coverage
        guard response.schemaVersion == 1, scope.sport == target.sport, !scope.eventID.isEmpty,
              normalized(scope.home) == normalized(event.home), normalized(scope.away) == normalized(event.away),
              let start = date(scope.commence), abs(start.timeIntervalSince(event.start)) <= 60,
              scope.markets == [target.marketKey], response.markets.count <= 500,
              let fetched, let expires, fetched <= now, expires >= fetched else { return empty("Scoped coverage or freshness metadata does not match this event and stat.") }
        // Reject cross-event rows and reused/conflicting event IDs before subject filtering.
        let labels = Set(["\(normalized(event.away)) at \(normalized(event.home))", "\(normalized(event.away)) @ \(normalized(event.home))"])
        guard response.markets.allSatisfy({ quote in
            quote.event == scope.eventID && quote.sport == scope.sport && quote.market == target.marketKey &&
            quote.commence.flatMap(date).map { abs($0.timeIntervalSince(event.start)) <= 60 } == true &&
            quote.eventLabel.map { labels.contains(normalized($0)) } == true
        }) else { return empty("Conflicting or unrelated event rows were excluded.") }
        let groups = Dictionary(grouping: response.markets, by: \.id)
        guard groups.allSatisfy({ entry in !entry.key.isEmpty && entry.value.dropFirst().allSatisfy { $0 == entry.value.first } }) else { return empty("Conflicting quote identities were excluded.") }
        var selected: [Market] = []
        var stale = response.stale || expires <= now || now.timeIntervalSince(fetched) > 900
        for copies in groups.values {
            guard let quote = copies.first, quote.mode == "live", quote.line.map({ $0.isFinite }) != false, !quote.book.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  quote.price.isFinite, abs(quote.price) >= 100,
                  let updated = quote.updated.flatMap(date), updated <= now,
                  (!recordedArchive || updated <= event.start),
                  !["post", "final", "completed", "canceled", "cancelled", "postponed", "suspended", "in", "live"].contains(normalized(quote.eventState ?? "")) else { continue }
            let label = normalized(quote.label)
            guard !["first half", "second half", "1st half", "2nd half", "quarter", "period", "inning", "regulation", "1h", "2h"].contains(where: { label.contains($0) }) else { continue }
            let side = normalized(quote.side ?? "")
            if target.kind == "prop" {
                guard quote.kind == "prop", normalized(quote.name) == normalized(target.name), ["over", "under"].contains(side) else { continue }
            } else if target.marketKey == "team_totals" {
                guard quote.kind == "team", ["over", "under"].contains(side),
                      normalized(quote.name) == normalized(target.teamName),
                      [event.home, event.away].map(normalized).contains(normalized(quote.name)) else { continue }
            } else if target.marketKey == "totals" {
                guard quote.kind != "prop", ["over", "under"].contains(side), normalized(quote.opponent) == normalized(event.home) else { continue }
            } else {
                guard quote.kind != "prop", normalized(quote.name) == normalized(target.teamName) else { continue }
            }
            if target.marketKey != "h2h" {
                guard let line = quote.line, line.isFinite else { continue }
                let signed = target.marketKey == "spreads" || (target.sport == "NFL" && ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_rush_reception_yds"].contains(AnalyticsHistoryEngine.statisticKey(sport: target.sport, marketKey: target.marketKey) ?? target.marketKey))
                guard signed || line >= 0 else { continue }
            }
            if now.timeIntervalSince(updated) > 900 { stale = true }
            selected.append(quote)
        }
        selected.sort { $0.id < $1.id }
        return AnalyticsOddsMatchResult(markets: selected, freshness: selected.isEmpty ? .unavailable : (stale ? .stale : .fresh), fetchedAt: fetched, expiresAt: expires, availability: selected.isEmpty ? "No verified book lines match this selected subject and stat." : (stale ? "Stale scoped book lines; not current live odds." : "Scoped book lines with independent freshness timestamps."))
    }
}

extension AnalyticsHistoryService {
    /// Explicit follow-up only: at most directory + scoreboard for teams, scoreboard + one roster for players.
    func resolveContext(market: Market, history: AnalyticsHistoryResult? = nil, now: Date = Date()) async -> AnalyticsContextResult {
        var urls: [String] = []
        func empty(_ message: String) -> AnalyticsContextResult { AnalyticsContextResult(event: nil, target: nil, availability: message, sourceURLs: urls) }
        if let reason = AnalyticsHistoryEngine.statsUnavailableReason(market: market, now: now) { return empty(reason) }
        do {
            try Task.checkCancellation()
            let league = PlayerHistoryService.leaguePath(market.sport)
            let prop = market.kind == "prop"
            let teamID: String
            let suppliedName = market.market == "totals" && ["over", "under"].contains(market.name.lowercased()) ? market.opponent : market.name
            if prop {
                guard let candidate = history?.candidateTeamID, !candidate.isEmpty, candidate.allSatisfy({ $0 >= "0" && $0 <= "9" }), history?.verifiedPlayerID != nil else { return empty("Current player-team membership is unverified; manual comparison remains available.") }
                teamID = candidate
            } else {
                let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(league)/teams?limit=1000")!
                urls.append(url.absoluteString)
                guard let verified = Self.teamID(directory: try await json(url), name: suppliedName, sport: market.sport) else { return empty("Team identity is unverified.") }
                teamID = verified
            }
            let formatter = DateFormatter(); formatter.dateFormat = "yyyyMMdd"; formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = TimeZone(secondsFromGMT: 0)
            let scoreboard = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(league)/scoreboard?dates=\(formatter.string(from: now))&limit=100")!
            urls.append(scoreboard.absoluteString)
            let events = try Self.upcomingEvents(payload: await json(scoreboard), sport: market.sport, teamID: teamID, now: now)
            guard events.count == 1, let event = events.first else { return empty("No unique same-day upcoming event was verified; ambiguous events and doubleheaders need an explicit event selection.") }
            let teamName = event.homeID == teamID ? event.home : event.away
            var playerID: String?, rosterTime: Date?
            if prop {
                let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(league)/teams/\(teamID)/roster")!
                urls.append(url.absoluteString)
                let roster = try await json(url)
                guard Self.rosterPlayerID(payload: roster, teamID: teamID, name: market.name) == history?.verifiedPlayerID else { return empty("Player identity and current roster membership could not both be verified.") }
                playerID = history?.verifiedPlayerID; rosterTime = now
            }
            let target = AnalyticsStatsTarget(sport: market.sport, kind: prop ? "prop" : "team", name: prop ? market.name : teamName, marketKey: market.market, teamName: teamName, teamID: teamID, playerID: playerID, rosterVerifiedAt: rosterTime)
            try Task.checkCancellation()
            return AnalyticsContextResult(event: event, target: target, availability: "Verified same-day upcoming event and subject context.", sourceURLs: urls)
        } catch is CancellationError { return empty("Context request cancelled.") }
        catch { return empty("Upcoming event context could not be verified; manual comparison remains available.") }
    }
    static func upcomingEvents(payload: [String: Any], sport: String, teamID: String, now: Date) throws -> [AnalyticsEvent] {
        guard let events = payload["events"] as? [[String: Any]], events.count <= 100 else { throw PlayerHistoryError.invalidPayload }
        var result: [AnalyticsEvent] = []
        var fingerprints: [String: NSDictionary] = [:], conflict = Set<String>()
        for event in events {
            guard let id = event["id"] as? String, !id.isEmpty else { continue }
            let fingerprint = NSDictionary(dictionary: event)
            if let existing = fingerprints[id], existing != fingerprint { conflict.insert(id) }
            fingerprints[id] = fingerprint
            guard let stamp = event["date"] as? String, let start = sportsDate(stamp), start > now, AnalyticsOddsMatcher.sameDay(start, now),
                  let competitions = event["competitions"] as? [[String: Any]], competitions.count == 1,
                  let status = competitions[0]["status"] as? [String: Any], let type = status["type"] as? [String: Any],
                  type["state"] as? String == "pre", type["completed"] as? Bool == false, type["name"] as? String == "STATUS_SCHEDULED",
                  let competitors = competitions[0]["competitors"] as? [[String: Any]], competitors.count == 2 else { continue }
            let home = competitors.filter { $0["homeAway"] as? String == "home" }, away = competitors.filter { $0["homeAway"] as? String == "away" }
            guard home.count == 1, away.count == 1,
                  let h = home[0]["team"] as? [String: Any], let a = away[0]["team"] as? [String: Any],
                  let homeID = h["id"] as? String, let awayID = a["id"] as? String, homeID != awayID,
                  [homeID, awayID].contains(teamID), let homeName = h["displayName"] as? String, let awayName = a["displayName"] as? String,
                  !homeName.isEmpty, !awayName.isEmpty else { continue }
            result.append(AnalyticsEvent(sport: sport, providerID: id, home: homeName, away: awayName, start: start, homeID: homeID, awayID: awayID))
        }
        return Dictionary(grouping: result.filter { !conflict.contains($0.providerID) }, by: \.providerID).compactMap { $0.value.first }.sorted { $0.start < $1.start }
    }
    static func rosterPlayerID(payload: [String: Any], teamID: String, name: String) -> String? {
        guard let team = payload["team"] as? [String: Any], team["id"] as? String == teamID,
              let athletes = payload["athletes"] as? [[String: Any]], athletes.count <= 200 else { return nil }
        let rows = athletes.flatMap { group -> [[String: Any]] in
            if let items = group["items"] as? [[String: Any]] { return items }
            return [group]
        }
        guard rows.count <= 200 else { return nil }
        let matches = rows.filter { AnalyticsOddsMatcher.normalized($0["fullName"] as? String ?? $0["displayName"] as? String ?? "") == AnalyticsOddsMatcher.normalized(name) }
        guard matches.count == 1, let id = matches[0]["id"] as? String, !id.isEmpty else { return nil }
        return id
    }
}
