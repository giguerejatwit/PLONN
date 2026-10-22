import Foundation
import CoreFoundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct PlayerHistoryResult: Sendable {
    let games: [PlayerGame]
    let opponentGames: [PlayerGame]
    let context: OpponentHistoryContext?
    let source: String
    let warnings: [String]
    let availability: String
    var playerName: String? = nil
    var playerID: String? = nil
    var sourceURLs: [String] = []
}

enum PlayerHistoryError: Error {
    case invalidURL, oversizedResponse, invalidPayload, ambiguousIdentity, unsupportedMarket, conflictingEvent, capacity
}

/// Public provider reads only. Cache keys contain provider URLs, never account credentials.
actor PlayerHistoryService {
    typealias Transport = @Sendable (URL) async throws -> Data
    static let shared = PlayerHistoryService()
    private struct Cached { let data: Data; let expires: Date; let touched: Date }
    private struct Pending { let id: UUID; let task: Task<Data, Error> }
    private let transport: Transport
    private let clock: @Sendable () -> Date
    private var cache: [URL: Cached] = [:]
    private var pending: [URL: Pending] = [:]
    private static let limit = 4 * 1024 * 1024

    init(transport: @escaping Transport = { url in try await PlayerHistoryService.publicRead(url) }, clock: @escaping @Sendable () -> Date = { Date() }) {
        self.transport = transport; self.clock = clock
    }

    static func allowedURL(_ url: URL) -> Bool {
        guard url.scheme == "https", url.user == nil, url.password == nil,
              url.port == nil || url.port == 443, url.fragment == nil,
              let host = url.host else { return false }
        return ["site.api.espn.com", "site.web.api.espn.com", "statsapi.mlb.com"].contains(host)
            && (host != "statsapi.mlb.com" || url.path.hasPrefix("/api/v1/"))
    }

    private static func publicRead(_ url: URL) async throws -> Data {
        guard allowedURL(url) else { throw PlayerHistoryError.invalidURL }
        let session = URLSession(configuration: .ephemeral, delegate: PublicHistoryRedirects(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url); request.timeoutInterval = 25
        #if canImport(FoundationNetworking)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
              let finalURL = http.url, allowedURL(finalURL) else { throw URLError(.badServerResponse) }
        guard data.count <= limit else { throw PlayerHistoryError.oversizedResponse }
        return data
        #else
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
              let finalURL = http.url, allowedURL(finalURL) else { throw URLError(.badServerResponse) }
        guard response.expectedContentLength <= Int64(limit) else { throw PlayerHistoryError.oversizedResponse }
        var data = Data()
        for try await byte in bytes {
            guard data.count < limit else { throw PlayerHistoryError.oversizedResponse }
            data.append(byte)
        }
        return data
        #endif
    }

    func read(_ url: URL) async throws -> Data {
        try Task.checkCancellation()
        guard Self.allowedURL(url) else { throw PlayerHistoryError.invalidURL }
        let timestamp = clock()
        cache = cache.filter { $0.value.expires > timestamp }
        if let hit = cache[url] {
            cache[url] = Cached(data: hit.data, expires: hit.expires, touched: timestamp)
            return hit.data
        }
        let request: Pending
        if let existing = pending[url] { request = existing }
        else {
            guard pending.count < 64 else { throw PlayerHistoryError.capacity }
            while cache.count + pending.count >= 64, let oldest = cache.min(by: { $0.value.touched < $1.value.touched })?.key { cache.removeValue(forKey: oldest) }
            let transport = self.transport
            request = Pending(id: UUID(), task: Task {
                let data = try await transport(url)
                guard data.count <= Self.limit else { throw PlayerHistoryError.oversizedResponse }
                return data
            })
            pending[url] = request
        }
        do {
            let data = try await request.task.value
            if pending[url]?.id == request.id {
                pending.removeValue(forKey: url)
                while cache.count + pending.count >= 64, let oldest = cache.min(by: { $0.value.touched < $1.value.touched })?.key { cache.removeValue(forKey: oldest) }
                let completed = clock()
                cache[url] = Cached(data: data, expires: completed.addingTimeInterval(900), touched: completed)
            }
            try Task.checkCancellation()
            return data
        } catch {
            if pending[url]?.id == request.id { pending.removeValue(forKey: url) }
            throw error
        }
    }

    private func json(_ path: String) async throws -> [String: Any] {
        guard let url = URL(string: path) else { throw PlayerHistoryError.invalidURL }
        let data = try await read(url)
        guard let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw PlayerHistoryError.invalidPayload }
        return payload
    }

    private func verifyMLBFinals(_ games: [PlayerGame]) async throws -> (games: [PlayerGame], sourceURLs: [String]) {
        let ids = Array(Set(games.map(\.id))).sorted()
        guard ids.count <= 250 else { throw PlayerHistoryError.capacity }
        var scheduleGames: [[String: Any]] = [], sourceURLs: [String] = []
        for start in stride(from: 0, to: ids.count, by: 50) {
            try Task.checkCancellation()
            let batch = Array(ids[start..<min(start + 50, ids.count)])
            var components = URLComponents(string: "https://statsapi.mlb.com/api/v1/schedule")!
            components.queryItems = [URLQueryItem(name: "sportId", value: "1"), URLQueryItem(name: "gamePks", value: batch.joined(separator: ","))]
            let path = components.url!.absoluteString
            let payload = try await json(path)
            sourceURLs.append(path)
            guard let dates = payload["dates"] as? [[String: Any]], dates.allSatisfy({ $0["games"] is [[String: Any]] }) else { throw PlayerHistoryError.invalidPayload }
            scheduleGames += dates.flatMap { $0["games"] as? [[String: Any]] ?? [] }
        }
        return (Self.finalizedMLBGames(games: games, payload: ["dates": [["games": scheduleGames]]]), sourceURLs)
    }

    func load(market: Market, includeOpponent: Bool = false, seasonCount: Int = 3, now: Date = Date(), maximumGames: Int = 10) async -> PlayerHistoryResult {
        let source = market.sport == "MLB" ? "MLB Stats API" : "ESPN"
        let gameLimit = min(max(maximumGames, 1), 2_000)
        // Request-local state: concurrent loads cannot inherit each other's provenance.
        var sourceURLs: [String] = []
        func record(_ path: String) { if !sourceURLs.contains(path) { sourceURLs.append(path) } }
        func citations() -> [String] {
            let evidence = sourceURLs.filter { $0.contains("/gamelog?") || $0.contains("/stats?") }
            let proofs = sourceURLs.filter { $0.contains("/schedule?") }
            let support = sourceURLs.filter { !evidence.contains($0) && !proofs.contains($0) }
            return Array((evidence + support + proofs).prefix(12))
        }
        func empty(_ message: String) -> PlayerHistoryResult {
            PlayerHistoryResult(games: [], opponentGames: [], context: nil, source: source, warnings: [], availability: message, sourceURLs: citations())
        }
        guard Self.statKeys(sport: market.sport, marketKey: market.market) != nil else { return empty("History is unavailable for this sport and prop type.") }
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(identifier: "America/New_York")!
        let target = market.commence.flatMap(sportsDate)
        let cutoff = calendar.startOfDay(for: min(target ?? now, now))
        let year = calendar.component(.year, from: target ?? now)
        let month = calendar.component(.month, from: target ?? now)
        let firstSeason = Self.firstSeason(sport: market.sport, year: year, month: month)
        let requested = Array((0..<min(max(seasonCount, 1), 3)).map { firstSeason - $0 })
        var warnings: [String] = [], loaded: [Int] = [], allGames: [PlayerGame] = []
        do {
            try Task.checkCancellation()
            let league = Self.leaguePath(market.sport)
            var teams: [HistoryTeam] = [], rosterTeamID: String?, rosterPlayerID: String?
            do {
                let directoryPath = market.sport == "MLB" ? "https://statsapi.mlb.com/api/v1/teams?sportId=1" : "https://site.api.espn.com/apis/site/v2/sports/\(league)/teams?limit=1000"
                let directory = try await json(directoryPath)
                record(directoryPath)
                teams = Self.matchupTeams(payload: directory, sport: market.sport, eventLabel: market.eventLabel)
                if teams.count == 2 {
                    var matches: [(String, String)] = []
                    for team in teams {
                        let rosterPath = market.sport == "MLB" ? "https://statsapi.mlb.com/api/v1/teams/\(team.id)/roster" : "https://site.api.espn.com/apis/site/v2/sports/\(league)/teams/\(team.id)/roster"
                        let roster = try await json(rosterPath)
                        record(rosterPath)
                        guard Self.validRoster(payload: roster, sport: market.sport),
                              (roster["team"] as? [String: Any]).flatMap({ Self.identifier($0["id"]) }).map({ $0 == team.id }) != false else { throw PlayerHistoryError.invalidPayload }
                        matches += Self.rosterMatches(payload: roster, sport: market.sport, playerName: market.name).map { (team.id, $0) }
                    }
                    if matches.count == 1 { rosterTeamID = matches[0].0; rosterPlayerID = matches[0].1 }
                }
            } catch is CancellationError { throw CancellationError() }
            catch { warnings.append("Matchup rosters could not be verified.") }
            let playerID: String
            if market.sport == "MLB" {
                let directoryPath = "https://statsapi.mlb.com/api/v1/sports/1/players?season=\(year)"
                let directory = try await json(directoryPath)
                record(directoryPath)
                guard let verified = Self.mlbPlayerID(payload: directory, playerName: market.name) else { return empty("Player could not be uniquely matched to MLB’s season directory.") }
                guard rosterPlayerID == nil || rosterPlayerID == verified else { return empty("Player identity conflicts with the matchup roster.") }
                playerID = verified
            } else if let verified = rosterPlayerID { playerID = verified }
            else {
                var components = URLComponents(string: "https://site.api.espn.com/apis/common/v3/search")!
                components.queryItems = [URLQueryItem(name: "query", value: market.name), URLQueryItem(name: "limit", value: "50"), URLQueryItem(name: "type", value: "player")]
                let searchPath = components.url!.absoluteString
                let search = try await json(searchPath)
                record(searchPath)
                guard let verified = Self.espnSearchPlayerID(payload: search, sport: market.sport, playerName: market.name) else { return empty("Player could not be uniquely verified in this sport’s public directory.") }
                playerID = verified
            }
            var opponent: HistoryTeam?
            let isCurrent = target.map { calendar.startOfDay(for: $0) >= calendar.startOfDay(for: now) } ?? false
            if includeOpponent, isCurrent, let own = rosterTeamID, teams.count == 2 { opponent = teams.first { $0.id != own } }
            for season in requested {
                try Task.checkCancellation()
                do {
                    let path: String
                    if market.sport == "MLB" {
                        let group = market.market.hasPrefix("pitcher_") ? "pitching" : "hitting"
                        path = "https://statsapi.mlb.com/api/v1/people/\(playerID)/stats?stats=gameLog&group=\(group)&season=\(season)&gameType=R,F,D,L,W"
                    } else { path = "https://site.web.api.espn.com/apis/common/v3/sports/\(league)/athletes/\(playerID)/gamelog?season=\(season)" }
                    let payload = try await json(path)
                    record(path)
                    var verifiedMLB: [PlayerGame]?
                    if market.sport == "MLB" {
                        let appearanceCutoff = includeOpponent && !isCurrent && target != nil
                            ? calendar.startOfDay(for: target!).addingTimeInterval(2 * 86400) : cutoff
                        let normalized = try Self.mlbGames(payload: payload, playerID: playerID, marketKey: market.market,
                            season: season, cutoff: appearanceCutoff)
                        let completed = try await verifyMLBFinals(normalized)
                        verifiedMLB = completed.games
                        completed.sourceURLs.forEach(record)
                        if verifiedMLB!.count < normalized.count { warnings.append("\(season) appearances without matching final game status were excluded.") }
                    }
                    // Parse actual appearances on the target date separately for historical membership.
                    if includeOpponent, !isCurrent, let target, teams.count == 2 {
                        let targetCutoff = calendar.startOfDay(for: target).addingTimeInterval(2 * 86400)
                        let appearances = try market.sport == "MLB"
                            ? (verifiedMLB ?? []).filter { $0.date < targetCutoff }
                            : Self.espnGames(payload: payload, sport: market.sport, playerID: playerID, marketKey: market.market, season: season, cutoff: targetCutoff)
                        let matches = appearances.filter { game in
                            calendar.isDate(game.date, inSameDayAs: target) && Set([game.playerTeamID ?? "", game.opponentID ?? ""]) == Set(teams.map(\.id))
                        }
                        let opposingIDs = Set(matches.compactMap(\.opponentID))
                        if opposingIDs.count == 1, let id = opposingIDs.first { opponent = teams.first { $0.id == id } }
                    }
                    let parsed = try market.sport == "MLB"
                        ? (verifiedMLB ?? []).filter { $0.date < cutoff }
                        : Self.espnGames(payload: payload, sport: market.sport, playerID: playerID, marketKey: market.market, season: season, cutoff: cutoff)
                    allGames += parsed; loaded.append(season)
                } catch is CancellationError { throw CancellationError() }
                catch { warnings.append("\(season) season could not be loaded or verified; available seasons are shown.") }
                let unique = try Self.uniqueGames(allGames)
                if !includeOpponent, unique.count >= gameLimit { break }
            }
            let sorted = try Self.uniqueGames(allGames)
            let games = Array(sorted.prefix(gameLimit))
            var context: OpponentHistoryContext?
            var opponentGames: [PlayerGame] = []
            if includeOpponent, let opponent, let target {
                opponentGames = Array(sorted.filter { $0.opponentID == opponent.id }.prefix(10))
                context = OpponentHistoryContext(sport: market.sport, playerID: playerID, playerName: market.name,
                    marketKey: market.market, opponentID: opponent.id, opponentName: opponent.name,
                    targetStart: target, cutoff: cutoff, targetEventID: market.event, source: source,
                    requestedSeasons: requested, loadedSeasons: loaded)
            } else if includeOpponent { warnings.append("Opponent is unverified for this matchup; recent games are shown without an opponent comparison.") }
            try Task.checkCancellation()
            return PlayerHistoryResult(games: games, opponentGames: opponentGames, context: context, source: source,
                warnings: warnings, availability: games.isEmpty ? "No verified earlier appearances are available for this player and prop." : "Regular season and postseason · Earlier calendar dates only.", playerName: market.name, playerID: playerID, sourceURLs: citations())
        } catch is CancellationError { return empty("History request cancelled.") }
        catch { return empty("Couldn’t load verified player history. Try again.") }
    }
}

private final class PublicHistoryRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(request.url.map { PlayerHistoryService.allowedURL($0) } == true ? request : nil)
    }
}

struct HistoryTeam: Sendable, Equatable { let id: String; let name: String }

extension PlayerHistoryService {
    private static func rows(_ value: Any?) -> [[String: Any]] { value as? [[String: Any]] ?? [] }
    private static func key(_ value: String) -> String {
        value.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .filter { $0.isLetter || $0.isNumber }
    }
    private static func identifier(_ value: Any?) -> String? {
        if let text = value as? String, !text.isEmpty, text.allSatisfy({ $0 >= "0" && $0 <= "9" }), let integer = Int(text), integer > 0 { return text }
        guard let number = numeric(value), number > 0, number.rounded() == number, number < Double(Int.max) else { return nil }
        return String(Int(number))
    }
    private static func numeric(_ value: Any?, signed: Bool = false) -> Double? {
        if let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() { return nil }
        let number: Double?
        if let text = value as? String { number = Double(text) }
        else if let value = value as? NSNumber { number = value.doubleValue }
        else { number = nil }
        guard let number, number.isFinite, signed || number >= 0 else { return nil }
        return number
    }
    private static func count(_ value: Any?, signed: Bool = false) -> Double? {
        guard let value = numeric(value, signed: signed), value.rounded() == value else { return nil }
        return value
    }
    static func firstSeason(sport: String, year: Int, month: Int) -> Int {
        if ["NBA", "NHL"].contains(sport), month >= 9 { return year + 1 }
        if sport == "NFL", month <= 2 { return year - 1 }
        return year
    }

    static func leaguePath(_ sport: String) -> String {
        switch sport { case "NBA": return "basketball/nba"; case "NFL": return "football/nfl"; case "NHL": return "hockey/nhl"; default: return "baseball/mlb" }
    }
    static func statKeys(sport: String, marketKey: String) -> [String]? {
        let mapping: [String: [String]]
        switch sport {
        case "NBA": mapping = ["player_points": ["points"], "player_rebounds": ["totalRebounds"], "player_assists": ["assists"],
            "player_steals": ["steals"], "player_blocks": ["blocks"], "player_threes": ["threePointFieldGoalsMade-threePointFieldGoalsAttempted"],
            "player_points_rebounds_assists": ["points", "totalRebounds", "assists"], "player_points_rebounds": ["points", "totalRebounds"],
            "player_points_assists": ["points", "assists"], "player_rebounds_assists": ["totalRebounds", "assists"]]
        case "NFL": mapping = ["player_pass_yds": ["passingYards"], "player_pass_tds": ["passingTouchdowns"], "player_pass_attempts": ["passingAttempts"], "player_pass_completions": ["completions"], "player_pass_interceptions": ["interceptions"], "player_rush_attempts": ["rushingAttempts"], "player_rush_yds": ["rushingYards"],
            "player_reception_yds": ["receivingYards"], "player_receptions": ["receptions"], "player_rush_reception_yds": ["rushingYards", "receivingYards"]]
        case "NHL": mapping = ["player_goals": ["goals"], "player_assists": ["assists"], "player_points": ["points"], "player_shots_on_goal": ["shotsTotal"],
            "player_power_play_points": ["powerPlayGoals", "powerPlayAssists"], "player_total_saves": ["saves"], "player_blocked_shots": ["blockedShots"]]
        case "MLB": mapping = ["batter_hits": ["hits"], "batter_rbis": ["rbi"], "batter_stolen_bases": ["stolenBases"], "batter_home_runs": ["homeRuns"],
            "batter_total_bases": ["totalBases"], "batter_runs_scored": ["runs"], "pitcher_strikeouts": ["strikeOuts"]]
        default: return nil
        }
        return mapping[marketKey]
    }

    static func espnSearchPlayerID(payload: [String: Any], sport: String, playerName: String) -> String? {
        guard ["NBA", "NFL", "NHL"].contains(sport), !key(playerName).isEmpty,
              count(payload["pageCount"]) == 1, let items = payload["items"] as? [[String: Any]],
              let total = count(payload["count"]), total == Double(items.count) else { return nil }
        let expectedSport = leaguePath(sport).components(separatedBy: "/")[0]
        let matches = items.filter {
            key($0["displayName"] as? String ?? "") == key(playerName)
                && $0["type"] as? String == "player" && $0["league"] as? String == sport.lowercased()
                && $0["sport"] as? String == expectedSport
        }
        guard matches.count == 1 else { return nil }
        return identifier(matches[0]["id"])
    }

    static func mlbPlayerID(payload: [String: Any], playerName: String) -> String? {
        let matches = rows(payload["people"]).filter { key($0["fullName"] as? String ?? "") == key(playerName) }
        guard !key(playerName).isEmpty, matches.count == 1 else { return nil }
        return identifier(matches[0]["id"])
    }

    static func matchupTeams(payload: [String: Any], sport: String, eventLabel: String?) -> [HistoryTeam] {
        guard let label = eventLabel else { return [] }
        let pieces: [String]
        if label.components(separatedBy: " at ").count == 2 { pieces = label.components(separatedBy: " at ") }
        else if label.components(separatedBy: " @ ").count == 2 { pieces = label.components(separatedBy: " @ ") }
        else { return [] }
        guard pieces.count == 2, key(pieces[0]) != key(pieces[1]) else { return [] }
        var directory: [[String: Any]] = []
        if sport == "MLB" { directory = rows(payload["teams"]).filter { identifier(($0["sport"] as? [String: Any])?["id"]) == "1" } }
        else {
            let expected = leaguePath(sport).components(separatedBy: "/")
            for item in rows(payload["sports"]) {
                if let slug = item["slug"] as? String, slug != expected[0] { continue }
                for league in rows(item["leagues"]) {
                    if let slug = league["slug"] as? String, slug != expected[1] { continue }
                    directory += rows(league["teams"]).compactMap { $0["team"] as? [String: Any] }
                }
            }
        }
        var result: [HistoryTeam] = []
        for piece in pieces {
            let matches = directory.filter { key(($0[sport == "MLB" ? "name" : "displayName"] as? String) ?? "") == key(piece) }
            guard matches.count == 1, let id = identifier(matches[0]["id"]), let name = matches[0][sport == "MLB" ? "name" : "displayName"] as? String else { return [] }
            result.append(HistoryTeam(id: id, name: name))
        }
        return Set(result.map(\.id)).count == 2 ? result : []
    }

    private static func validRoster(payload: [String: Any], sport: String) -> Bool {
        if sport == "MLB" {
            guard let roster = payload["roster"] as? [[String: Any]] else { return false }
            return roster.allSatisfy {
                guard let person = $0["person"] as? [String: Any], let name = person["fullName"] as? String else { return false }
                return !key(name).isEmpty && identifier(person["id"]) != nil
            }
        }
        guard let groups = payload["athletes"] as? [[String: Any]] else { return false }
        let players = groups.flatMap { $0["items"] as? [[String: Any]] ?? [$0] }
        return players.allSatisfy {
            guard let name = $0["displayName"] as? String else { return false }
            return !key(name).isEmpty && identifier($0["id"]) != nil
        }
    }

    static func rosterMatches(payload: [String: Any], sport: String, playerName: String) -> [String] {
        let athletes: [[String: Any]]
        if sport == "MLB" { athletes = rows(payload["roster"]).compactMap { $0["person"] as? [String: Any] } }
        else { athletes = rows(payload["athletes"]).flatMap { $0["items"] != nil ? rows($0["items"]) : [$0] } }
        return athletes.filter { key($0[sport == "MLB" ? "fullName" : "displayName"] as? String ?? "") == key(playerName) }.compactMap { identifier($0["id"]) }
    }

    static func espnGames(payload: [String: Any], sport: String, playerID: String, marketKey: String, season: Int, cutoff: Date) throws -> [PlayerGame] {
        guard let keys = statKeys(sport: sport, marketKey: marketKey), ["NBA", "NFL", "NHL"].contains(sport),
              identifier(playerID) != nil, let names = payload["names"] as? [String], Set(names).count == names.count,
              keys.allSatisfy({ names.contains($0) }), let events = payload["events"] as? [String: [String: Any]],
              let groups = payload["seasonTypes"] as? [[String: Any]] else { throw PlayerHistoryError.invalidPayload }
        let periods = rows(payload["filters"]).filter { $0["name"] as? String == "season" }
        guard periods.count == 1, numeric(periods[0]["value"]) == Double(season) else { throw PlayerHistoryError.invalidPayload }
        let leagues = rows(payload["filters"]).filter { $0["name"] as? String == "league" }
        guard leagues.isEmpty || (leagues.count == 1 && leagues[0]["value"] as? String == sport.lowercased()) else { throw PlayerHistoryError.invalidPayload }
        let prefix = sport == "NFL" ? String(season) : "\(season - 1)-\(String(season).suffix(2))"
        // Quarantine raw conflicts before validation, including a malformed copy of an otherwise valid row.
        var rawStatistics: [String: [String]] = [:], quarantined = Set<String>()
        for group in groups where ["\(prefix) Regular Season", "\(prefix) Postseason"].contains(group["displayName"] as? String ?? "") {
            for category in rows(group["categories"]) where !excludedPhase(category["displayName"] as? String ?? "") {
                for row in rows(category["events"]) {
                    guard let id = identifier(row["eventId"]) else { continue }
                    guard let stats = row["stats"] as? [String], stats.count == names.count else { quarantined.insert(id); continue }
                    if let previous = rawStatistics[id], previous != stats { quarantined.insert(id) }
                    rawStatistics[id] = stats
                }
            }
        }
        var games: [PlayerGame] = []
        for group in groups {
            guard let label = group["displayName"] as? String, ["\(prefix) Regular Season", "\(prefix) Postseason"].contains(label) else { continue }
            for category in rows(group["categories"]) {
                if excludedPhase(category["displayName"] as? String ?? "") { continue }
                for row in rows(category["events"]) {
                    guard let id = identifier(row["eventId"]), !quarantined.contains(id), let event = events[id], identifier(event["id"]) == id,
                          event["leagueAbbreviation"] as? String == sport,
                          let day = event["gameDate"] as? String, let date = sportsDate(day), date < cutoff,
                          !excludedPhase(event["eventNote"] as? String ?? ""),
                          !excludedPhase((event["type"] as? [String: Any])?["text"] as? String ?? ""),
                          !excludedPhase((row["type"] as? [String: Any])?["text"] as? String ?? ""),
                          let own = event["team"] as? [String: Any], let ownID = identifier(own["id"]),
                          (own["isAllStar"] as? Bool) != true,
                          let opponent = event["opponent"] as? [String: Any], let opponentID = identifier(opponent["id"]), ownID != opponentID,
                          let opponentName = opponent["displayName"] as? String, !opponentName.isEmpty,
                          let home = identifier(event["homeTeamId"]), let away = identifier(event["awayTeamId"]), home != away,
                          Set([home, away]) == Set([ownID, opponentID]),
                          let homeScore = count(event["homeTeamScore"]), let awayScore = count(event["awayTeamScore"]),
                          let result = event["gameResult"] as? String, ["W", "L", "T"].contains(result),
                          let stats = row["stats"] as? [String], stats.count == names.count else { continue }
                    let ownScore = ownID == home ? homeScore : awayScore, opposingScore = ownID == home ? awayScore : homeScore
                    guard (result == "W" && ownScore > opposingScore) || (result == "L" && ownScore < opposingScore) || (result == "T" && sport == "NFL" && ownScore == opposingScore) else { continue }
                    func raw(_ key: String) -> String? { names.firstIndex(of: key).map { stats[$0] } }
                    func value(_ key: String) -> Double? {
                        guard let text = raw(key) else { return nil }
                        if key == "threePointFieldGoalsMade-threePointFieldGoalsAttempted" {
                            let parts = text.components(separatedBy: "-")
                            guard parts.count == 2, let made = count(parts[0]), let attempted = count(parts[1]), made <= attempted else { return nil }
                            return made
                        }
                        return count(text, signed: sport == "NFL" && ["passingYards", "rushingYards", "receivingYards"].contains(key))
                    }
                    let values = keys.compactMap(value)
                    guard values.count == keys.count else { continue }
                    let workload: String
                    if sport == "NBA" {
                        guard let minutes = raw("minutes").flatMap({ numeric($0) }), minutes > 0 else { continue }
                        workload = "\(minutes.formatted()) MIN"
                    } else if sport == "NHL" {
                        guard let time = raw("timeOnIcePerGame"), let seconds = iceSeconds(time), seconds > 0 else { continue }
                        workload = "\(time) TOI"
                    } else {
                        let workKeys = ["passingAttempts", "rushingAttempts", "receptions", "receivingTargets"]
                        let work = workKeys.compactMap { raw($0).flatMap { count($0) } }
                        guard !work.isEmpty, work.contains(where: { $0 > 0 }) else { continue }
                        workload = "Game · verified activity"
                    }
                    let total = values.reduce(0, +)
                    guard total.isFinite else { continue }
                    games.append(PlayerGame(id: id, date: date, opponent: opponentName, value: total, workload: workload,
                        sport: sport, playerID: playerID, playerTeamID: ownID, opponentID: opponentID, marketKey: marketKey, season: season, homeAway: explicitESPNVenue(event: event, ownID: ownID)))
                }
            }
        }
        return try uniqueGames(games)
    }

    /// Explicit source team IDs locate the already verified player's own team.
    /// Contradictory direct venue metadata is never replaced with an inferred label.
    private static func explicitESPNVenue(event: [String: Any], ownID: String) -> String? {
        guard allowsVenueMetadata(event) else { return nil }
        guard let home = identifier(event["homeTeamId"]), let away = identifier(event["awayTeamId"]),
              home != away, [home, away].contains(ownID) else { return nil }
        let mapped = ownID == home ? "home" : "away"
        let competitors = rows(event["competitions"]).flatMap { rows($0["competitors"]) }
        let own = competitors.filter { identifier(($0["team"] as? [String: Any])?["id"]) == ownID }
        guard own.count <= 1 else { return nil }
        let team = event["team"] as? [String: Any]
        let direct = [own.first?["homeAway"], team?["homeAway"]].compactMap { $0 }
        guard direct.allSatisfy({ ($0 as? String) == mapped }) else { return nil }
        return mapped
    }
    private static func allowsVenueMetadata(_ event: [String: Any]) -> Bool {
        let sources = [event] + rows(event["competitions"])
        return sources.allSatisfy { source in
            guard let flag = source["neutralSite"] else { return true }
            guard let number = flag as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return false }
            return !number.boolValue
        }
    }
    private static func explicitMLBVenue(_ value: Any?) -> String? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue ? "home" : "away"
    }
    private static func excludedPhase(_ text: String) -> Bool {
        let normalized = key(text)
        return ["preseason", "allstar", "playin", "exhibition"].contains { normalized.contains($0) }
    }
    private static func iceSeconds(_ value: String) -> Double? {
        let parts = value.components(separatedBy: ":")
        guard parts.count == 2, let minutes = count(parts[0]), let seconds = count(parts[1]), seconds < 60 else { return nil }
        return minutes * 60 + seconds
    }

    static func mlbGames(payload: [String: Any], playerID: String, marketKey: String, season: Int, cutoff: Date) throws -> [PlayerGame] {
        guard let keys = statKeys(sport: "MLB", marketKey: marketKey), identifier(playerID) != nil,
              let groups = payload["stats"] as? [[String: Any]] else { throw PlayerHistoryError.invalidPayload }
        let pitching = marketKey.hasPrefix("pitcher_")
        var games: [PlayerGame] = []
        for group in groups {
            guard (group["group"] as? [String: Any])?["displayName"] as? String == (pitching ? "pitching" : "hitting"),
                  (group["type"] as? [String: Any])?["displayName"] as? String == "gameLog" else { continue }
            for row in rows(group["splits"]) {
                guard numeric(row["season"]) == Double(season), identifier((row["sport"] as? [String: Any])?["id"]) == "1",
                      identifier((row["player"] as? [String: Any])?["id"]) == playerID,
                      let type = row["gameType"] as? String, ["R", "F", "D", "L", "W"].contains(type),
                      let day = row["date"] as? String, let date = sportsDate(day + "T12:00:00Z"), date < cutoff,
                      let stat = row["stat"] as? [String: Any], let own = row["team"] as? [String: Any], let ownID = identifier(own["id"]),
                      let opponent = row["opponent"] as? [String: Any], let opponentID = identifier(opponent["id"]), ownID != opponentID,
                      let name = opponent["name"] as? String, !name.isEmpty,
                      let id = identifier((row["game"] as? [String: Any])?["gamePk"]) else { continue }
                let values = keys.compactMap { count(stat[$0]) }
                guard values.count == keys.count else { continue }
                let workload: String
                if pitching {
                    guard let innings = stat["inningsPitched"] as? String, let outs = baseballOuts(innings),
                          let faced = count(stat["battersFaced"]), outs > 0 || faced > 0 else { continue }
                    workload = "\(innings) IP · \(Int(faced)) BF"
                } else {
                    guard let appearances = count(stat["plateAppearances"]), appearances > 0 else { continue }
                    workload = "\(Int(appearances)) PA"
                }
                games.append(PlayerGame(id: id, date: date, opponent: name, value: values.reduce(0, +), workload: workload,
                    sport: "MLB", playerID: playerID, playerTeamID: ownID, opponentID: opponentID, marketKey: marketKey, season: season, homeAway: allowsVenueMetadata(row) ? explicitMLBVenue(row["isHome"]) : nil))
            }
        }
        return try uniqueGames(games)
    }
    /// The schedule supplies final status absent from Stats API appearance logs.
    static func finalizedMLBGames(games: [PlayerGame], payload: [String: Any]) -> [PlayerGame] {
        struct Proof: Equatable {
            let date: Date; let home: String; let away: String; let season: Int; let type: String
            let state: String; let code: String; let venueAllowed: Bool
        }
        var proofs: [String: Proof] = [:], quarantined = Set<String>()
        for row in rows(payload["dates"]).flatMap({ rows($0["games"]) }) {
            guard let id = identifier(row["gamePk"]) else { continue }
            guard let official = row["officialDate"] as? String, let date = sportsDate(official + "T12:00:00Z"),
                  let year = count(row["season"]), year < Double(Int.max),
                  let teams = row["teams"] as? [String: Any],
                  let home = identifier(((teams["home"] as? [String: Any])?["team"] as? [String: Any])?["id"]),
                  let away = identifier(((teams["away"] as? [String: Any])?["team"] as? [String: Any])?["id"]), home != away,
                  let status = row["status"] as? [String: Any], let state = status["abstractGameState"] as? String,
                  let code = status["codedGameState"] as? String, let type = row["gameType"] as? String else { quarantined.insert(id); continue }
            let proof = Proof(date: date, home: home, away: away, season: Int(year), type: type, state: state, code: code, venueAllowed: allowsVenueMetadata(row))
            if let previous = proofs[id], previous != proof { quarantined.insert(id) }
            proofs[id] = proof
        }
        return games.compactMap { game -> PlayerGame? in
            guard game.sport == "MLB", !quarantined.contains(game.id), let proof = proofs[game.id],
                  proof.state == "Final", proof.code == "F", ["R", "F", "D", "L", "W"].contains(proof.type),
                  proof.date == game.date, proof.season == game.season,
                  let own = game.playerTeamID, let opposing = game.opponentID,
                  Set([own, opposing]) == Set([proof.home, proof.away]) else { return nil }
            var verified = game
            if !proof.venueAllowed { verified.homeAway = nil }
            return verified
        }
    }

    private static func baseballOuts(_ text: String) -> Double? {
        let parts = text.components(separatedBy: ".")
        guard parts.count == 2, let innings = count(parts[0]), let outs = count(parts[1]), outs <= 2 else { return nil }
        return innings * 3 + outs
    }
    static func uniqueGames(_ games: [PlayerGame]) throws -> [PlayerGame] {
        var unique: [String: PlayerGame] = [:], quarantined = Set<String>()
        for game in games {
            guard !quarantined.contains(game.id) else { continue }
            if let previous = unique[game.id] {
                guard previous.date == game.date, previous.value == game.value, previous.playerID == game.playerID,
                      previous.playerTeamID == game.playerTeamID, previous.opponentID == game.opponentID,
                      previous.sport == game.sport, previous.marketKey == game.marketKey,
                      previous.season == game.season, previous.homeAway == game.homeAway, previous.workload == game.workload else {
                    quarantined.insert(game.id); unique.removeValue(forKey: game.id); continue
                }
            }
            unique[game.id] = game
        }
        return unique.values.sorted { $0.date == $1.date ? $0.id > $1.id : $0.date > $1.date }
    }
}
