import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Observed game facts only. This feed does not supply in-play prices or model probabilities.
struct LiveMatchupSnapshot: Equatable, Sendable {
    enum Phase: String, Sendable { case scheduled, live, final, interrupted }
    struct Team: Equatable, Sendable {
        let id, name, abbreviation: String
        let score: String?
    }
    struct Stat: Identifiable, Equatable, Sendable {
        let id, label: String
        let away, home: String?
    }
    let event: SharedMatchupEvent
    let phase: Phase
    let status: String
    let away, home: Team
    let stats: [Stat]
    let checkedAt: Date

    func isStale(at now: Date) -> Bool {
        phase != .final && (now < checkedAt || now.timeIntervalSince(checkedAt) >= 90)
    }
    var scoreSummary: String? {
        guard phase == .live || phase == .final,
              let a = away.score, let h = home.score, let av = Int(a), let hv = Int(h) else { return nil }
        if av == hv { return phase == .final ? "Finished level at \(a)–\(h)." : "Tied at \(a)–\(h)." }
        let leader = av > hv ? away : home
        return "\(leader.name) \(phase == .final ? "won" : "leads") \(max(av,hv))–\(min(av,hv))."
    }
}

enum LiveMatchupPolicy {
    static func showsGameContext(event: SharedMatchupEvent, state: String, snapshot: LiveMatchupSnapshot?, now: Date) -> Bool {
        if let snapshot, snapshot.event == event, snapshot.phase != .scheduled { return true }
        if ["in", "post", "final", "completed"].contains(state.lowercased()) { return true }
        return sportsDate(event.startISO).map { $0 <= now } ?? false
    }
    /// Keep the scheduled start boundary even if the user leaves the screen open before kickoff.
    static func nextRefresh(event: SharedMatchupEvent, snapshot: LiveMatchupSnapshot?, now: Date, failed: Bool) -> TimeInterval? {
        let snapshot = snapshot.flatMap { $0.event == event ? $0 : nil }
        if snapshot?.phase == .final { return nil }
        if failed { return 30 }
        if snapshot?.phase == .live { return 30 }
        if let start = sportsDate(event.startISO), start > now { return min(60, max(1, start.timeIntervalSince(now))) }
        return 30
    }
}

struct LiveMatchupClient: Sendable {
    enum Failure: Error { case unsupportedSport, invalidEvent, invalidResponse }
    typealias Transport = @Sendable (URLRequest) async throws -> (Data, URLResponse)
    var transport: Transport = { try await URLSession.shared.data(for: $0) }
    var now: @Sendable () -> Date = { Date() }

    static func feedPath(_ sport: String) -> String? {
        let paths = ["NFL":"football/nfl", "NCAAF":"football/college-football", "NBA":"basketball/nba",
                     "WNBA":"basketball/wnba", "NCAAM":"basketball/mens-college-basketball",
                     "NCAAW":"basketball/womens-college-basketball", "MLB":"baseball/mlb", "NHL":"hockey/nhl",
                     "Soccer":"soccer/uefa.nations"]
        if let path = paths[sport] { return path }
        if sport.hasPrefix("Soccer:") {
            let league = String(sport.dropFirst(7))
            guard !league.isEmpty, league.range(of: "^[a-z0-9.]+$", options: .regularExpression) != nil else { return nil }
            return "soccer/" + league
        }
        return nil
    }

    func fetch(event: SharedMatchupEvent) async throws -> LiveMatchupSnapshot {
        guard let path = Self.feedPath(event.sport) else { throw Failure.unsupportedSport }
        let id = event.eventID.hasPrefix("espn:") ? String(event.eventID.dropFirst(5)) : event.eventID
        guard !id.isEmpty, id.allSatisfy({ $0.isASCII && $0.isNumber }) else { throw Failure.invalidEvent }
        let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(path)/summary?event=\(id)")!
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        request.setValue("no-cache", forHTTPHeaderField: "Cache-Control")
        let (data, response) = try await transport(request)
        try Task.checkCancellation()
        guard (response as? HTTPURLResponse)?.statusCode == 200, data.count <= 8_000_000 else { throw Failure.invalidResponse }
        let checkedAt = now()
        let result = try await Task.detached(priority: .utility) {
            try Self.parse(data, event: event, checkedAt: checkedAt)
        }.value
        try Task.checkCancellation()
        return result
    }

    static func parse(_ data: Data, event: SharedMatchupEvent, checkedAt: Date) throws -> LiveMatchupSnapshot {
        guard data.count <= 8_000_000,
              let root = try JSONSerialization.jsonObject(with: data) as? [String:Any],
              let header = root["header"] as? [String:Any] else { throw Failure.invalidResponse }
        let id = event.eventID.hasPrefix("espn:") ? String(event.eventID.dropFirst(5)) : event.eventID
        guard text(header["id"]) == id,
              let competition = rows(header["competitions"]).first(where: { text($0["id"]) == id }),
              let status = competition["status"] as? [String:Any], let type = status["type"] as? [String:Any] else { throw Failure.invalidResponse }
        let competitors = rows(competition["competitors"])
        func team(_ side: String, expected: String) throws -> LiveMatchupSnapshot.Team {
            let matches = competitors.filter { text($0["homeAway"]) == side }
            guard matches.count == 1, let row = matches.first, let info = row["team"] as? [String:Any],
                  let name = text(info["displayName"]), let teamID = text(info["id"]),
                  name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == expected.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else { throw Failure.invalidResponse }
            let rawScore = text(row["score"]) ?? text((row["score"] as? [String:Any])?["displayValue"])
            let score = rawScore.flatMap { Int($0) }.flatMap { $0 >= 0 ? String($0) : nil }
            return .init(id: teamID, name: name, abbreviation: text(info["abbreviation"]) ?? name, score: score)
        }
        let home = try team("home", expected: event.homeTeam), away = try team("away", expected: event.awayTeam)
        guard home.id != away.id else { throw Failure.invalidResponse }
        let state = text(type["state"]), statusName = text(type["name"]) ?? ""
        let interrupted = ["DELAYED", "POSTPONED", "SUSPENDED", "CANCELED", "CANCELLED", "ABANDONED"].contains { statusName.contains($0) }
        let phase: LiveMatchupSnapshot.Phase
        if interrupted { phase = .interrupted }
        else if type["completed"] as? Bool == true && state == "post" { phase = .final }
        else if state == "in" { phase = .live }
        else if state == "pre" { phase = .scheduled }
        else { throw Failure.invalidResponse }
        let detail = text(type["detail"]) ?? text(type["description"]) ?? (phase == .live ? "In progress" : "Awaiting game update")
        let box = root["boxscore"] as? [String:Any] ?? [:]
        func stats(_ team: LiveMatchupSnapshot.Team) -> [String:String] {
            let matches = rows(box["teams"]).filter { text(($0["team"] as? [String:Any])?["id"]) == team.id }
            guard matches.count == 1, let row = matches.first else { return [:] }
            var result: [String:String] = [:]
            for stat in rows(row["statistics"]) {
                guard let name = text(stat["name"]) else { continue }
                if let value = usable(stat["displayValue"]) { result[name] = value }
                // Baseball repeats hits/walks under both batting and pitching. Keep their namespaces.
                for child in rows(stat["stats"]) {
                    if let key = text(child["name"]), let value = usable(child["displayValue"]) { result[name + "." + key] = value }
                }
            }
            return result
        }
        let hs = stats(home), aws = stats(away)
        let comparisons: [LiveMatchupSnapshot.Stat] = metrics(event.sport).compactMap { key,label in
            guard hs[key] != nil || aws[key] != nil else { return nil }
            return .init(id: key, label: label, away: aws[key], home: hs[key])
        }
        return .init(event: event, phase: phase, status: detail, away: away, home: home,
                     stats: phase == .scheduled ? [] : comparisons, checkedAt: checkedAt)
    }

    static func metrics(_ sport: String) -> [(String,String)] {
        switch sport {
        case "NFL", "NCAAF": return [("totalYards","Total yards"),("yardsPerPlay","Yards per play"),("thirdDownEff","Third downs"),("turnovers","Turnovers"),("rushingYards","Rushing yards"),("possessionTime","Possession")]
        case "NBA", "WNBA", "NCAAM", "NCAAW": return [("fieldGoalPct","Field goal %"),("threePointFieldGoalsMade-threePointFieldGoalsAttempted","Three-pointers"),("totalRebounds","Rebounds"),("totalTurnovers","Turnovers"),("pointsInPaint","Points in the paint"),("assists","Assists")]
        case "NHL": return [("shotsTotal","Shots on goal"),("powerPlayGoals","Power-play goals"),("powerPlayOpportunities","Power-play chances"),("faceoffPercent","Faceoff win %"),("blockedShots","Blocked shots"),("penaltyMinutes","Penalty minutes")]
        case "MLB": return [("batting.hits","Hits"),("batting.homeRuns","Home runs"),("batting.walks","Walks drawn"),("batting.strikeouts","Batting strikeouts"),("fielding.errors","Errors")]
        default: return [("shotsOnTarget","Shots on target"),("totalShots","Total shots"),("possessionPct","Possession %"),("wonCorners","Corners"),("yellowCards","Yellow cards"),("redCards","Red cards")]
        }
    }
    private static func rows(_ value: Any?) -> [[String:Any]] { value as? [[String:Any]] ?? [] }
    private static func text(_ value: Any?) -> String? {
        if let value = value as? String, !value.isEmpty { return value }
        if let value = value as? NSNumber { return value.stringValue }
        return nil
    }
    private static func usable(_ value: Any?) -> String? {
        guard let value = text(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !["", "--", "-", "—", "N/A"].contains(value) else { return nil }
        return value
    }
}
