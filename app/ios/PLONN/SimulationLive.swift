import Foundation
import CoreFoundation
import CryptoKit

/// A provider snapshot is observed history. It is never assigned a simulated probability.
struct SimulationLiveSnapshot: Sendable {
    enum Phase: String, Sendable { case scheduled, live, final, interrupted }
    struct Event: Identifiable, Sendable {
        let id: String
        let text: String
        let period: Int
        let clock: String
        let awayScore: Int
        let homeScore: Int
    }
    let eventID: String
    let sport: InterestSport
    let teamIDs: [String]
    let start: Date
    let checked: Date
    let phase: Phase
    let status: String
    let scores: [Int]
    let period: Int
    let remaining: Int?
    let possession: Int?
    let down: Int?
    let distance: Int?
    let yard: Int?
    let battingSide: Int?
    let outs: Int?
    let bases: [String?]
    let batter: String?
    let pitcher: String?
    let battingOrders: [[String]]
    let lines: [[PlayerSimulation.Line]]
    let events: [Event]
    let fingerprint: String
    var nextBatters: [Int] = [0, 0]
    var activePitchers: [String?] = [nil, nil]
    var basePitchers: [String?] = [nil, nil, nil]
    var baseStateVerified = true
    var activeGoalies: [String?] = [nil, nil]
    var balls = 0
    var strikes = 0
    var elapsed: Int {
        let length = sport == .nba ? 720 : sport == .nhl ? 1200 : 900
        return max(0, (period - 1) * length + length - (remaining ?? length))
    }
    var regulationSeconds: Int { sport == .nba ? 2880 : 3600 }
    var forecastIssue: String? {
        if phase == .final { return nil }
        guard phase == .live else { return "Waiting for play to resume." }
        if sport == .mlb {
            guard (1...9).contains(period) else { return "Extra innings: projections paused." }
            guard let battingSide, (0...1).contains(battingSide), let outs, (0...3).contains(outs), baseStateVerified, battingOrders.count == 2, battingOrders.allSatisfy({ $0.count == 9 }) else { return "Waiting for inning and baserunner details." }
            if balls > 0 || strikes > 0 { return "At-bat in progress · projections update after this plate appearance." }
        } else {
            guard period <= (sport == .nhl ? 3 : 4) else { return "Overtime: projections paused." }
            guard remaining != nil else { return "Waiting for the game clock." }
            if remaining == 0 && (period == (sport == .nhl ? 3 : 4) || (sport == .nfl && period == 2)) {
                return "Waiting for the next period or confirmed final score."
            }
            if sport == .nfl, possession == nil || down == nil || distance == nil || yard == nil { return "Waiting for possession and field position." }
        }
        guard lines.count == 2, lines.allSatisfy({ !$0.isEmpty }) else { return "Waiting for player stats." }
        return nil
    }
    /// Accept corrections to the latest play, but reject an older feed revision
    /// that would rewind the clock, inning or completed outs.
    func supersedes(_ previous: Self) -> Bool {
        guard eventID == previous.eventID, sport == previous.sport, teamIDs == previous.teamIDs, checked >= previous.checked else { return false }
        if phase == .final { return true }
        if previous.phase == .final || (phase == .scheduled && previous.phase == .live) { return false }
        let samePlay = events.last?.id != nil && events.last?.id == previous.events.last?.id
        if samePlay { return true }
        if sport == .mlb {
            let half = period * 2 + (battingSide ?? 0), oldHalf = previous.period * 2 + (previous.battingSide ?? 0)
            return half > oldHalf || (half == oldHalf && (outs ?? 0) >= (previous.outs ?? 0))
        }
        return elapsed >= previous.elapsed
    }
    func isStale(at now: Date) -> Bool { phase != .final && (now < checked || now.timeIntervalSince(checked) > 90) }
}

struct SimulationLiveSource: Sendable {
    typealias Reader = MatchupSimulationSource.Reader
    var read: Reader = { url in
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        request.setValue("no-cache", forHTTPHeaderField: "Cache-Control")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200, data.count <= 8_000_000 else { throw PlayerHistoryError.invalidPayload }
        if let age = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Age").flatMap(Double.init), age > 90 { throw PlayerHistoryError.invalidPayload }
        return data
    }
    func fetch(_ event: PlayerMatchupEvent) async throws -> SimulationLiveSnapshot {
        guard let path = PlayerMatchupSource.path(event.game.league) else { throw PlayerHistoryError.unsupportedMarket }
        let data = try await read(URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(path)/summary?event=\(event.game.eventID)")!)
        try Task.checkCancellation()
        return try Self.parse(data, event: event, now: Date())
    }
    static func parse(_ data: Data, event: PlayerMatchupEvent, now: Date) throws -> SimulationLiveSnapshot {
        guard data.count <= 8_000_000, let sport = InterestSport(rawValue: event.game.league),
              let start = event.game.start,
              let body = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let header = body["header"] as? [String: Any], header["id"] as? String == event.game.eventID,
              let league = header["league"] as? [String: Any], (league["abbreviation"] as? String)?.uppercased() == sport.rawValue,
              let competitions = header["competitions"] as? [[String: Any]], competitions.count == 1,
              let competition = competitions.first, competition["id"] as? String == event.game.eventID,
              let date = (competition["date"] as? String).flatMap(sportsDate), abs(date.timeIntervalSince(start)) < 60,
              let sides = competition["competitors"] as? [[String: Any]], sides.count == 2,
              let away = sides.first(where: { $0["homeAway"] as? String == "away" }), away["id"] as? String == event.awayID,
              let home = sides.first(where: { $0["homeAway"] as? String == "home" }), home["id"] as? String == event.homeID,
              let status = competition["status"] as? [String: Any], let type = status["type"] as? [String: Any],
              let season = header["season"] as? [String: Any], let year = season["year"] as? Int else { throw PlayerHistoryError.invalidPayload }
        func integer(_ value: Any?) -> Int? {
            if let string = value as? String { return Int(string) }
            if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue { return number.intValue }
            return nil
        }
        let name = type["name"] as? String ?? ""
        let interrupted = ["DELAY", "POSTPON", "SUSPEND", "CANCEL", "ABANDON"].contains { name.contains($0) }
        let phase: SimulationLiveSnapshot.Phase
        if interrupted { phase = .interrupted }
        else if type["completed"] as? Bool == true && type["state"] as? String == "post" { phase = .final }
        else if type["state"] as? String == "in" { phase = .live }
        else if type["state"] as? String == "pre" { phase = .scheduled }
        else { throw PlayerHistoryError.invalidPayload }
        guard let a = integer(away["score"]), let h = integer(home["score"]), (0...300).contains(a), (0...300).contains(h) else { throw PlayerHistoryError.invalidPayload }
        var raw = body["plays"] as? [[String: Any]] ?? []
        if let drives = body["drives"] as? [String: Any] {
            raw += (drives["previous"] as? [[String: Any]] ?? []).flatMap { $0["plays"] as? [[String: Any]] ?? [] }
            raw += (drives["current"] as? [String: Any])?["plays"] as? [[String: Any]] ?? []
        }
        guard raw.count <= 15_000 else { throw PlayerHistoryError.invalidPayload }
        var byID: [String: [String: Any]] = [:]
        for play in raw {
            guard let id = play["id"] as? String, id.count < 100 else { continue }
            byID[id] = play // The current full response is authoritative for corrections.
        }
        let ordered = byID.values.sorted {
            // MLB sequenceNumber restarts every at-bat. Event-prefixed IDs carry
            // inning/at-bat order; never sort an entire baseball game by pitch number.
            if sport == .mlb { return ($0["id"] as? String ?? "").compare($1["id"] as? String ?? "", options: .numeric) == .orderedAscending }
            let left = integer($0["sequenceNumber"]) ?? 0, right = integer($1["sequenceNumber"]) ?? 0
            return left == right ? ($0["id"] as? String ?? "") < ($1["id"] as? String ?? "") : left < right
        }
        let events: [SimulationLiveSnapshot.Event] = ordered.compactMap { play in
            guard let id = play["id"] as? String, let text = play["text"] as? String, text.count <= 2000,
                  let av = integer(play["awayScore"]), let hv = integer(play["homeScore"]), (0...300).contains(av), (0...300).contains(hv),
                  let period = (play["period"] as? [String: Any])?["number"] as? Int, (1...30).contains(period) else { return nil }
            return .init(id: id, text: text, period: period, clock: (play["clock"] as? [String: Any])?["displayValue"] as? String ?? (play["period"] as? [String: Any])?["displayValue"] as? String ?? "", awayScore: av, homeScore: hv)
        }
        if phase == .live, let last = events.last, [last.awayScore, last.homeScore] != [a, h] { throw PlayerHistoryError.invalidPayload }
        let last = ordered.last ?? [:], situation = body["situation"] as? [String: Any] ?? [:]
        let period = integer(status["period"]) ?? (last["period"] as? [String: Any])?["number"] as? Int ?? 1
        guard (1...30).contains(period) else { throw PlayerHistoryError.invalidPayload }
        var remaining = integer(status["clock"])
        if remaining == nil, let display = status["displayClock"] as? String {
            let parts = display.split(separator: ":").compactMap { Int($0) }
            if parts.count == 2 && (0...59).contains(parts[1]) { remaining = parts[0] * 60 + parts[1] }
        }
        if let seconds = remaining, !(0...(sport == .nba ? 720 : sport == .nhl ? 1200 : 900)).contains(seconds) { throw PlayerHistoryError.invalidPayload }
        let possessionID = situation["possession"] as? String
        let possession = possessionID == event.awayID ? 0 : possessionID == event.homeID ? 1 : nil
        let down = integer(situation["down"]).flatMap { (1...4).contains($0) ? $0 : nil }
        let distance = integer(situation["distance"]).flatMap { (1...100).contains($0) ? $0 : nil }
        let yard = integer(situation["yardsToEndzone"]).flatMap { (1...99).contains($0) ? 100 - $0 : nil }
        let half = (last["period"] as? [String: Any])?["type"] as? String ?? ""
        let battingSide = half.lowercased() == "top" ? 0 : half.lowercased() == "bottom" ? 1 : nil
        let outs = integer(situation["outs"]) ?? integer(last["outs"])
        func athleteID(_ value: Any?) -> String? {
            guard let row = value as? [String: Any] else { return nil }
            return (row["athlete"] as? [String: Any])?["id"] as? String ?? row["id"] as? String
        }
        var basesVerified = true
        let bases: [String?] = ["onFirst", "onSecond", "onThird"].map { key in
            if let current = situation[key] {
                if let flag = current as? Bool { if flag { basesVerified = false }; return nil }
                let id = athleteID(current); if id == nil { basesVerified = false }; return id
            }
            return athleteID(last[key])
        }
        let participants = last["participants"] as? [[String: Any]] ?? []
        let batter = athleteID(situation["batter"]) ?? participants.first(where: { $0["type"] as? String == "batter" }).flatMap { athleteID($0) }
        let pitcher = athleteID(situation["pitcher"]) ?? participants.first(where: { $0["type"] as? String == "pitcher" }).flatMap { athleteID($0) }
        var orders: [[Int: String]] = [[:], [:]], nextBatters = [0, 0]
        var responsible: [String: String] = [:]
        var activePitchers: [String?] = [nil, nil], activeGoalies: [String?] = [nil, nil]
        var lastHalf = ""

        for play in ordered {
            guard let order = integer(play["batOrder"]), (1...9).contains(order),
                  let teamID = (play["team"] as? [String: Any])?["id"] as? String,
                  let player = (play["participants"] as? [[String: Any]])?.first(where: { $0["type"] as? String == "batter" }).flatMap({ athleteID($0) }) else { continue }
            let side = teamID == event.awayID ? 0 : teamID == event.homeID ? 1 : -1
            guard side >= 0 else { continue }
            orders[side][order] = player
            let kind = (play["type"] as? [String: Any])?["id"] as? String
            nextBatters[side] = (order - 1 + (["57", "99"].contains(kind ?? "") ? 1 : 0)) % 9
            let halfKey = String((play["period"] as? [String: Any])?["number"] as? Int ?? 0) + String(side)
            if halfKey != lastHalf { responsible = [:]; lastHalf = halfKey }
            if let pitcherID = (play["participants"] as? [[String: Any]])?.first(where: { $0["type"] as? String == "pitcher" }).flatMap({ athleteID($0) }) {
                // Remember who put each inherited runner on base.
                for key in ["onFirst", "onSecond", "onThird"] {
                    if let runner = athleteID(play[key]), responsible[runner] == nil { responsible[runner] = pitcherID }
                }
            }

        }
        // Actual active batting slots override historical appearance order,
        // including substitutions. Pitchers already removed cannot pitch again.
        for team in (body["boxscore"] as? [String: Any])?["players"] as? [[String: Any]] ?? [] {
            let teamID = (team["team"] as? [String: Any])?["id"] as? String
            let side = teamID == event.awayID ? 0 : teamID == event.homeID ? 1 : -1
            guard side >= 0 else { continue }
            for group in team["statistics"] as? [[String: Any]] ?? [] {
                for row in group["athletes"] as? [[String: Any]] ?? [] where row["active"] as? Bool == true {
                    guard let id = athleteID(row) else { continue }
                    if let order = integer(row["batOrder"]), (1...9).contains(order) { orders[side][order] = id }
                    if group["type"] as? String == "pitching" { activePitchers[side] = id }
                    if group["name"] as? String == "goalies" { activeGoalies[side] = id }
                }
            }
        }
        var lines: [[PlayerSimulation.Line]] = [[], []]
        if phase == .live || phase == .final {
            for (side, rawTeam) in [away, home].enumerated() {
                let id = side == 0 ? event.awayID : event.homeID
                let info = rawTeam["team"] as? [String: Any] ?? [:]
                let team = InterestTeam(sport: sport, providerID: id, name: side == 0 ? event.game.away : event.game.home, abbreviation: info["abbreviation"] as? String ?? id)
                let game = MatchupSimulation.Game(id: event.game.eventID, date: start, season: year, scored: side == 0 ? a : h, allowed: side == 0 ? h : a, home: side == 1)
                lines[side] = (try? PlayerSimulationSource.parse(data, team: team, game: game, cutoff: now.addingTimeInterval(1), allowInProgress: true).lines) ?? []
            }
        }
        // Ignore transport timestamps in the revision: unchanged responses must
        // not create fresh random outcomes or restart a user's replay.
        let revision: [String: Any] = ["phase":phase.rawValue, "scores":[a,h], "period":period, "clock":remaining ?? -1, "situation":situation, "plays":ordered, "boxscore":body["boxscore"] ?? [:]]
        let fingerprint = SHA256.hash(data: try JSONSerialization.data(withJSONObject: revision, options: [.sortedKeys])).map { String(format: "%02x", $0) }.joined()
        var snapshot = SimulationLiveSnapshot(eventID: event.game.eventID, sport: sport, teamIDs: [event.awayID, event.homeID], start: start, checked: now, phase: phase, status: type["detail"] as? String ?? type["description"] as? String ?? "", scores: [a,h], period: period, remaining: remaining, possession: possession, down: down, distance: distance, yard: yard, battingSide: battingSide, outs: outs, bases: bases, batter: batter, pitcher: pitcher, battingOrders: orders.map { value in value.keys.sorted().compactMap { value[$0] } }, lines: lines, events: events, fingerprint: fingerprint)
        snapshot.nextBatters = nextBatters
        snapshot.activePitchers = activePitchers
        snapshot.activeGoalies = activeGoalies
        snapshot.basePitchers = bases.map { $0.flatMap { responsible[$0] } }
        snapshot.baseStateVerified = basesVerified && zip(bases, snapshot.basePitchers).allSatisfy { $0.0 == nil || $0.1 != nil }
        let lastKind = (last["type"] as? [String: Any])?["id"] as? String ?? ""
        if !["57", "99"].contains(lastKind) {
            let count = situation["count"] as? [String: Any] ?? last["resultCount"] as? [String: Any] ?? [:]
            snapshot.balls = min(3, max(0, integer(count["balls"]) ?? integer(situation["balls"]) ?? 0))
            snapshot.strikes = min(2, max(0, integer(count["strikes"]) ?? integer(situation["strikes"]) ?? 0))
        }
        return snapshot
    }

    func discover(league: String, now: Date = Date()) async throws -> PlayerMatchupDiscovery {
        async let scheduled = PlayerMatchupSource().discover(league: league, now: now)
        guard let path = PlayerMatchupSource.path(league) else { throw PlayerHistoryError.unsupportedMarket }
        let formatter = DateFormatter(); formatter.dateFormat = "yyyyMMdd"; formatter.timeZone = TimeZone(identifier: "America/New_York"); formatter.locale = Locale(identifier: "en_US_POSIX")
        var live: [PlayerMatchupEvent] = [], failed = false
        for offset in [-1, 0] {
            let day = formatter.string(from: now.addingTimeInterval(Double(offset) * 86400))
            do {
                let data = try await read(URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(path)/scoreboard?dates=\(day)&limit=1000")!)
                live += try Self.liveEvents(data, league: league, now: now)
            } catch { failed = true }
        }
        let future = (try? await scheduled) ?? .init(events: [], incomplete: true)
        var unique: [String: PlayerMatchupEvent] = [:]
        for event in future.events + live { unique[event.id] = event }
        return .init(events: unique.values.sorted { if $0.isLive != $1.isLive { return $0.isLive }; return ($0.game.start ?? .distantFuture) < ($1.game.start ?? .distantFuture) }, incomplete: failed || future.incomplete)
    }
    static func liveEvents(_ data: Data, league: String, now: Date) throws -> [PlayerMatchupEvent] {
        guard let body = try JSONSerialization.jsonObject(with: data) as? [String: Any], let rows = body["events"] as? [[String: Any]], rows.count <= 1000 else { throw PlayerHistoryError.invalidPayload }
        var seen = Set<String>()
        return rows.compactMap { row in
            guard let id = row["id"] as? String, InterestTeam.validProviderID(id), seen.insert(id).inserted,
                  let dateString = row["date"] as? String, let date = sportsDate(dateString), (0...36 * 3600).contains(now.timeIntervalSince(date)),
                  let competition = (row["competitions"] as? [[String: Any]])?.first,
                  let status = (competition["status"] as? [String: Any])?["type"] as? [String: Any], status["state"] as? String == "in", status["completed"] as? Bool != true,
                  let sides = competition["competitors"] as? [[String: Any]], sides.count == 2,
                  let a = sides.first(where: { $0["homeAway"] as? String == "away" })?["team"] as? [String: Any],
                  let h = sides.first(where: { $0["homeAway"] as? String == "home" })?["team"] as? [String: Any],
                  let aid = a["id"] as? String, let hid = h["id"] as? String, aid != hid,
                  let an = a["displayName"] as? String, let hn = h["displayName"] as? String else { return nil }
            return .init(game: .init(league: league, eventID: id, home: hn, away: an, startsAt: dateString), homeID: hid, awayID: aid, seasonType: NBARecordedOddsRouting.seasonType(event: row), isLive: true)
        }
    }
}
