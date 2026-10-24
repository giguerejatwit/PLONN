import Foundation

/// A provider-verified scheduled event and current roster. No sportsbook names or
/// historical team membership are promoted into current identities.
struct PlayerMatchupEvent: Equatable, Identifiable {
    let game: UpcomingGameSetup
    let homeID: String
    let awayID: String
    var seasonType: Int? = nil
    var isLive = false
    var id: String { game.id }
}
struct PlayerMatchupPlayer: Equatable, Identifiable {
    let id: String
    let name: String
    let teamID: String
    let team: String
    let position: String
}
struct PlayerMatchupDiscovery {
    let events: [PlayerMatchupEvent]
    let incomplete: Bool
}
struct PlayerMatchupSource {
    typealias Reader = @Sendable (URL) async throws -> Data
    let read: Reader
    init(read: @escaping Reader = { try await PlayerHistoryService.shared.read($0) }) { self.read = read }
    static func path(_ league: String) -> String? {
        ["NFL": "football/nfl", "NBA": "basketball/nba", "MLB": "baseball/mlb", "NHL": "hockey/nhl"][league]
    }
    private static func numericID(_ value: Any?) -> String? {
        guard let text = value as? String, (1...15).contains(text.count), text.allSatisfy({ $0.isASCII && $0.isNumber }), UInt64(text).map({ $0 > 0 }) == true else { return nil }
        return text
    }
    static func events(_ data: Data, league: String, now: Date) throws -> [PlayerMatchupEvent] {
        let accepted = try UpcomingGameSetup.scheduledGames(data, league: league, now: now)
        guard let body = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = body["events"] as? [[String: Any]], raw.count <= 1000 else { throw PlayerHistoryError.invalidPayload }
        // Reject duplicate provider events rather than arbitrarily accepting one.
        var seen = Set<String>(), duplicates = Set<String>()
        for event in raw { if let id = event["id"] as? String, !seen.insert(id).inserted { duplicates.insert(id) } }
        return accepted.compactMap { setup in
            guard !duplicates.contains(setup.eventID), (setup.start ?? .distantFuture) <= now.addingTimeInterval(7 * 86400),
                  let event = raw.first(where: { $0["id"] as? String == setup.eventID }),
                  let competition = (event["competitions"] as? [[String: Any]])?.first,
                  let competitors = competition["competitors"] as? [[String: Any]],
                  let home = competitors.first(where: { $0["homeAway"] as? String == "home" })?["team"] as? [String: Any],
                  let away = competitors.first(where: { $0["homeAway"] as? String == "away" })?["team"] as? [String: Any],
                  let homeID = numericID(home["id"]), let awayID = numericID(away["id"]), homeID != awayID else { return nil }
            return PlayerMatchupEvent(game: setup, homeID: homeID, awayID: awayID, seasonType: NBARecordedOddsRouting.seasonType(event: event))
        }
    }
    func discover(league: String, now: Date = Date()) async throws -> PlayerMatchupDiscovery {
        guard let path = Self.path(league) else { throw PlayerHistoryError.unsupportedMarket }
        let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0); formatter.dateFormat = "yyyyMMdd"
        let urls = (0...7).map { offset in
            URL(string: "https://site.api.espn.com/apis/site/v2/sports/" + path + "/scoreboard?limit=1000&dates=" + formatter.string(from: now.addingTimeInterval(Double(offset) * 86400)))!
        }
        let days = await withTaskGroup(of: [PlayerMatchupEvent]?.self, returning: [[PlayerMatchupEvent]?].self) { group in
            for url in urls { group.addTask { try? await Self.events(read(url), league: league, now: now) } }
            var result: [[PlayerMatchupEvent]?] = []
            for await day in group { result.append(day) }
            return result
        }
        try Task.checkCancellation()
        guard days.contains(where: { $0 != nil }) else { throw PlayerHistoryError.invalidPayload }
        var unique: [String: PlayerMatchupEvent] = [:], conflicts = Set<String>()
        for event in days.compactMap({ $0 }).flatMap({ $0 }) {
            if let old = unique[event.id], old != event { conflicts.insert(event.id) }
            unique[event.id] = event
        }
        return PlayerMatchupDiscovery(events: unique.values.filter { !conflicts.contains($0.id) }.sorted { ($0.game.start ?? .distantFuture) < ($1.game.start ?? .distantFuture) }, incomplete: days.contains(where: { $0 == nil }) || !conflicts.isEmpty)
    }
    static func players(_ data: Data, teamID: String, teamName: String) throws -> [PlayerMatchupPlayer] {
        guard numericID(teamID) != nil,
              let body = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let team = body["team"] as? [String: Any], numericID(team["id"]) == teamID,
              let returnedName = team["displayName"] as? String, LabNameSearch.normalized(returnedName) == LabNameSearch.normalized(teamName),
              let groups = body["athletes"] as? [[String: Any]], groups.count <= 500 else { throw PlayerHistoryError.invalidPayload }
        let rows = groups.flatMap { $0["items"] as? [[String: Any]] ?? [$0] }
        guard rows.count <= 500 else { throw PlayerHistoryError.invalidPayload }
        var result: [PlayerMatchupPlayer] = [], seen = Set<String>()
        for player in rows {
            guard let id = numericID(player["id"]), let name = player["displayName"] as? String, (1...120).contains(name.count),
                  seen.insert(id).inserted else { throw PlayerHistoryError.invalidPayload }
            let position = (player["position"] as? [String: Any])?["abbreviation"] as? String ?? ""
            result.append(PlayerMatchupPlayer(id: id, name: name, teamID: teamID, team: teamName, position: String(position.prefix(30))))
        }
        return result.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }
    func roster(event: PlayerMatchupEvent) async throws -> [PlayerMatchupPlayer] {
        guard let path = Self.path(event.game.league) else { throw PlayerHistoryError.unsupportedMarket }
        @Sendable func load(_ id: String, _ team: String) async throws -> [PlayerMatchupPlayer] {
            let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/" + path + "/teams/" + id + "/roster")!
            return try Self.players(await read(url), teamID: id, teamName: team)
        }
        async let home = load(event.homeID, event.game.home)
        async let away = load(event.awayID, event.game.away)
        let players = try await home + away
        guard Set(players.map(\.id)).count == players.count else { throw PlayerHistoryError.ambiguousIdentity }
        return players
    }
    static func matches(_ proof: AnalyticsEvent, event: PlayerMatchupEvent, within events: [PlayerMatchupEvent]) -> Bool {
        guard proof.sport == event.game.league, proof.start == event.game.start,
              LabNameSearch.normalized(proof.home) == LabNameSearch.normalized(event.game.home),
              LabNameSearch.normalized(proof.away) == LabNameSearch.normalized(event.game.away) else { return false }
        if proof.sport != "MLB" { return proof.providerID == event.game.eventID }
        // MLB-native event IDs and ESPN IDs are separate namespaces. The full
        // home/away/start tuple must identify exactly one verified ESPN event.
        return events.filter { $0.game.league == proof.sport && $0.game.start == proof.start &&
            LabNameSearch.normalized($0.game.home) == LabNameSearch.normalized(proof.home) &&
            LabNameSearch.normalized($0.game.away) == LabNameSearch.normalized(proof.away) }.count == 1
    }
    static func selection(for candidate: AnalyticsFeedCandidate, players: [PlayerMatchupPlayer]) -> PlayerMatchupPlayer? {
        let matches = players.filter { player in
            guard LabNameSearch.normalized(player.name) == LabNameSearch.normalized(candidate.market.name) else { return false }
            if let proof = candidate.verifiedSubject {
                if let team = proof.teamName, LabNameSearch.normalized(team) != LabNameSearch.normalized(player.team) { return false }
                // MLB analytics uses MLB-native IDs. Verify its name AND team in
                // the ESPN roster, never treat its numeric ID as an ESPN ID.
                if candidate.market.sport == "MLB" { return proof.teamName != nil }
                return proof.providerID == player.id && (proof.teamID == nil || proof.teamID == player.teamID)
            }
            return true
        }
        return matches.count == 1 ? matches[0] : nil
    }
}


/// Uses recorded sportsbook contracts only. Roster identities and the scheduled
/// event must be verified independently before a name can match a quote.
enum PlayerMatchupLineMatching {
    static func quotes(event: PlayerMatchupEvent, playerID: String, players: [PlayerMatchupPlayer],
                       stat: String, side: String, markets: [Market], now: Date = Date()) -> [Market] {
        guard let start = event.game.start, start > now,
              let player = players.first(where: { $0.id == playerID }),
              players.filter({ $0.id == playerID }).count == 1,
              players.filter({ AnalyticsOddsMatcher.normalized($0.name) == AnalyticsOddsMatcher.normalized(player.name) }).count == 1,
              (player.teamID == event.homeID && RecordedTeamIdentity.matches(player.team, event.game.home)) ||
              (player.teamID == event.awayID && RecordedTeamIdentity.matches(player.team, event.game.away)),
              ["over", "under"].contains(side.lowercased()) else { return [] }
        let eventProof = AnalyticsEvent(sport: event.game.league, providerID: event.game.eventID,
            home: event.game.home, away: event.game.away, start: start,
            homeID: event.homeID, awayID: event.awayID, seasonType: event.seasonType)
        let keys = Set(markets.filter { $0.sport == event.game.league &&
            PlayerMatchupSetup.stat(sport: $0.sport, marketKey: $0.market) == stat }.map(\.market))
        return keys.flatMap { key -> [Market] in
            let target = AnalyticsStatsTarget(sport: event.game.league, kind: "prop", name: player.name,
                marketKey: key, teamName: player.team, teamID: player.teamID,
                playerID: player.id, rosterVerifiedAt: now)
            return AnalyticsOddsMatcher.matchCached(target: target, event: eventProof, markets: markets,
                now: now, recordedSnapshot: true).markets.filter { quote in
                    guard quote.side?.lowercased() == side.lowercased(),
                          let timestamp = quote.updated.flatMap(sportsDate),
                          now.timeIntervalSince(timestamp) <= 86400 else { return false }
                    return true
                }
        }.sorted { $0.id < $1.id }
    }

    /// Rechecks a quote already selected through the verified roster/event matcher
    /// when its saved result is displayed. No quote price is inferred from the line.
    static func matches(_ quote: Market, setup: PlayerMatchupSetup, now: Date = Date()) -> Bool {
        guard let start = setup.start, start > now, quote.sport == setup.league, quote.kind == "prop",
              quote.mode == "live", !quote.event.isEmpty,
              RecordedTeamIdentity.matches(quote.name, setup.player),
              PlayerMatchupSetup.stat(sport: quote.sport, marketKey: quote.market) == setup.stat,
              quote.line == setup.line, quote.side?.lowercased() == setup.side.lowercased(), quote.book == setup.book,
              let label = quote.eventLabel.map(AnalyticsOddsMatcher.normalized),
              [setup.away + " at " + setup.home, setup.away + " @ " + setup.home].map(AnalyticsOddsMatcher.normalized).contains(label),
              let quoteStart = quote.commence.flatMap(sportsDate),
              RecordedStartPolicy.accepts(eventID: quote.event, quoteStart: quoteStart, start: start,
                sport: setup.league, candidates: [(quote.event, quoteStart)], now: now),
              let recorded = quote.updated.flatMap(sportsDate), (0...86400).contains(now.timeIntervalSince(recorded)),
              !["post", "final", "completed", "canceled", "cancelled", "postponed", "suspended", "in", "live"].contains(AnalyticsOddsMatcher.normalized(quote.eventState ?? "")),
              BuilderLogic.decimalOdds(quote.price) != nil else { return false }
        return true
    }

    /// Main lines only. Prefer the opening quote's book, otherwise use the newest
    /// recorded contract. Never choose a favorable alternate as a default line.
    static func preferred(_ quotes: [Market], book: String?, line: Double? = nil) -> Market? {
        let candidates = quotes.filter { quote in
            if let line { return quote.line == line }
            return !quote.market.hasSuffix("_alternate")
        }.sorted { left, right in
            let leftPreferred = left.book == book, rightPreferred = right.book == book
            if leftPreferred != rightPreferred { return leftPreferred }
            let leftTime = left.updated.flatMap(sportsDate) ?? .distantPast
            let rightTime = right.updated.flatMap(sportsDate) ?? .distantPast
            if leftTime != rightTime { return leftTime > rightTime }
            if left.book != right.book { return left.book < right.book }
            return left.id < right.id
        }
        guard let chosen = candidates.first else { return nil }
        // A book publishing conflicting primary lines at the same timestamp is
        // ambiguous; leave the comparison editable rather than guessing.
        guard candidates.filter({ $0.book == chosen.book && $0.market == chosen.market && $0.updated == chosen.updated })
            .allSatisfy({ $0.line == chosen.line && $0.price == chosen.price }) else { return nil }
        return chosen
    }
}

/// A delayed cache result can fill an untouched field, but never erase an edit
/// (including deliberately clearing it). Player/prop/event changes start afresh.
struct PlayerMatchupLineDraft: Equatable {
    private(set) var text = ""
    private(set) var edited = false
    private(set) var selection = ""
    mutating func select(_ key: String, binary: Bool) {
        guard selection != key else { return }
        selection = key; edited = false; text = binary ? "0.5" : ""
    }
    mutating func edit(_ value: String) { text = value; edited = true }
    mutating func fill(_ value: Double?, for key: String) {
        guard key == selection, !edited, let value, value.isFinite else { return }
        text = value.formatted(.number.grouping(.never).precision(.fractionLength(0...2)).locale(Locale(identifier: "en_US_POSIX")))
    }
}
