import Foundation

/// Provider identifiers stay in the request model; the UI presents teams and dates.
struct LabHistoricalGame: Identifiable, Equatable, Sendable {
    struct Team: Equatable, Sendable {
        let name: String
        let requestName: String
    }
    let id: String
    let date: Date
    let home: Team
    let away: Team
    let homeScore: Int
    let awayScore: Int
    let detail: String
    var title: String { "\(away.name) at \(home.name)" }
    func opponent(for team: Team) -> Team? {
        team == home ? away : team == away ? home : nil
    }
}

struct LabHistoricalGames {
    enum Source: String {
        case espn, nflverse, mlb
        static func source(league: String, fullGame: Bool, propSource: String? = nil) -> Self {
            if !fullGame, let propSource, let source=Self(rawValue:propSource) {return source}
            return fullGame ? .espn : league == "NFL" ? .nflverse : league == "MLB" ? .mlb : .espn
        }
    }
    static var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York")!
        return calendar
    }
    static func day(_ date: Date, compact: Bool = false) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = compact ? "yyyyMMdd" : "yyyy-MM-dd"
        return formatter.string(from: date)
    }
    static func latestDate(now: Date = Date()) -> Date {
        calendar.date(byAdding: .day, value: -2, to: calendar.startOfDay(for: now))!
    }
    static func isEligibleForHistoricalTest(_ start: Date, now: Date = Date()) -> Bool {
        var utc = Calendar(identifier: .gregorian)
        utc.timeZone = TimeZone(secondsFromGMT: 0)!
        let cutoff = utc.date(byAdding: .day, value: -1, to: utc.startOfDay(for: now))!
        return start < cutoff
    }
    static func displayDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeZone = calendar.timeZone
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
    private static func timestamp(_ raw: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        // ESPN omits seconds in scoreboard timestamps.
        return formatter.date(from: raw) ?? formatter.date(from: raw.replacingOccurrences(of: "Z", with: ":00Z"))
    }
    static func load(league: String, source: Source, date: Date) async throws -> [LabHistoricalGame] {
        guard date <= latestDate().addingTimeInterval(86_399) else { return [] }
        if source == .mlb {
            let data = try await fetch("https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=\(day(date))")
            return try parseMLB(data, date: date).filter { isEligibleForHistoricalTest($0.date) }
        }
        let paths = ["NFL":"football/nfl", "NBA":"basketball/nba", "NHL":"hockey/nhl", "MLB":"baseball/mlb"]
        guard let path = paths[league] else { throw URLError(.unsupportedURL) }
        let data = try await fetch("https://site.api.espn.com/apis/site/v2/sports/\(path)/scoreboard?limit=1000&dates=\(day(date, compact: true))")
        let games = try parseESPN(data, date: date).filter { isEligibleForHistoricalTest($0.date) }
        guard source == .nflverse, !games.isEmpty else { return games }
        let csv = try await fetch("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
        return try nflGames(games, csv: csv, date: date)
    }
    private static func fetch(_ address: String) async throws -> Data {
        var request = URLRequest(url: URL(string: address)!)
        request.timeoutInterval = 20
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200, data.count <= 4_000_000 else {
            throw URLError(.badServerResponse)
        }
        try Task.checkCancellation()
        return data
    }
    static func parseESPN(_ data: Data, date: Date) throws -> [LabHistoricalGame] {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String:Any],
              let events = root["events"] as? [[String:Any]] else { throw URLError(.cannotParseResponse) }
        return events.compactMap { event in
            guard let id = event["id"] as? String, !id.isEmpty, id.allSatisfy(\.isNumber),
                  (event["season"] as? [String:Any])?["type"] as? Int == 2,
                  let raw = event["date"] as? String, let start = timestamp(raw), day(start) == day(date),
                  let competition = (event["competitions"] as? [[String:Any]])?.first,
                  let status = competition["status"] as? [String:Any],
                  (status["type"] as? [String:Any])?["completed"] as? Bool == true,
                  (status["type"] as? [String:Any])?["state"] as? String == "post",
                  (status["type"] as? [String:Any])?["name"] as? String == "STATUS_FINAL",
                  let entries = competition["competitors"] as? [[String:Any]], entries.count == 2,
                  let home = entries.first(where: { $0["homeAway"] as? String == "home" }),
                  let away = entries.first(where: { $0["homeAway"] as? String == "away" }),
                  let homeName = (home["team"] as? [String:Any])?["displayName"] as? String,
                  let awayName = (away["team"] as? [String:Any])?["displayName"] as? String,
                  !homeName.isEmpty, !awayName.isEmpty,
                  let hs = home["score"] as? String, let homeScore = Int(hs),
                  let ascore = away["score"] as? String, let awayScore = Int(ascore) else { return nil }
            if let slug = (competition["type"] as? [String:Any])?["slug"] as? String, slug != "standard" { return nil }
            if (competition["notes"] as? [[String:Any]] ?? []).contains(where: {
                ($0["headline"] as? String ?? "").localizedCaseInsensitiveContains("championship")
            }) { return nil }
            return LabHistoricalGame(id: id, date: start,
                home: .init(name: homeName, requestName: homeName), away: .init(name: awayName, requestName: awayName),
                homeScore: homeScore, awayScore: awayScore, detail: "Final")
        }
    }
    static func parseMLB(_ data: Data, date: Date) throws -> [LabHistoricalGame] {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String:Any],
              let dates = root["dates"] as? [[String:Any]] else { throw URLError(.cannotParseResponse) }
        return dates.flatMap { $0["games"] as? [[String:Any]] ?? [] }.compactMap { game in
            guard game["gameType"] as? String == "R", game["officialDate"] as? String == day(date),
                  (game["status"] as? [String:Any])?["abstractGameState"] as? String == "Final",
                  let id = game["gamePk"] as? Int, id > 0,
                  let raw = game["gameDate"] as? String, let start = timestamp(raw),
                  let teams = game["teams"] as? [String:[String:Any]], let home = teams["home"], let away = teams["away"],
                  let homeName = (home["team"] as? [String:Any])?["name"] as? String,
                  let awayName = (away["team"] as? [String:Any])?["name"] as? String,
                  let homeScore = home["score"] as? Int, let awayScore = away["score"] as? Int else { return nil }
            let number = game["gameNumber"] as? Int ?? 1
            let doubleheader = game["doubleHeader"] as? String != "N" && game["doubleHeader"] != nil
            return LabHistoricalGame(id: String(id), date: start,
                home: .init(name: homeName, requestName: homeName), away: .init(name: awayName, requestName: awayName),
                homeScore: homeScore, awayScore: awayScore, detail: doubleheader ? "Final · Game \(number)" : "Final")
        }
    }
    static func nflGames(_ games: [LabHistoricalGame], csv: Data, date: Date) throws -> [LabHistoricalGame] {
        guard csv.count <= 4_000_000, let text = String(data: csv, encoding: .utf8) else { throw URLError(.cannotParseResponse) }
        let rows = try csvRows(text)
        guard let header = rows.first else { throw URLError(.cannotParseResponse) }
        let fields = ["espn", "game_id", "game_type", "gameday", "home_team", "away_team", "home_score", "away_score"]
        guard fields.allSatisfy(header.contains) else { throw URLError(.cannotParseResponse) }
        var matches: [String:[String:String]] = [:]
        for row in rows.dropFirst() where row.count == header.count {
            let values = Dictionary(uniqueKeysWithValues: zip(header, row))
            guard values["game_type"] == "REG", values["gameday"] == day(date),
                  let homeScore = values["home_score"], Double(homeScore) != nil,
                  let awayScore = values["away_score"], Double(awayScore) != nil,
                  let espn = values["espn"], !espn.isEmpty else { continue }
            matches[espn] = values
        }
        return games.compactMap { game in
            guard let row = matches[game.id], let id = row["game_id"], !id.isEmpty,
                  let home = row["home_team"], !home.isEmpty, let away = row["away_team"], !away.isEmpty,
                  Double(row["home_score"] ?? "") == Double(game.homeScore),
                  Double(row["away_score"] ?? "") == Double(game.awayScore) else { return nil }
            return LabHistoricalGame(id: id, date: game.date,
                home: .init(name: game.home.name, requestName: home), away: .init(name: game.away.name, requestName: away),
                homeScore: game.homeScore, awayScore: game.awayScore, detail: game.detail)
        }
    }
    private static func csvRows(_ text: String) throws -> [[String]] {
        var rows: [[String]] = [], row: [String] = [], field = "", quoted = false
        var chars = text.makeIterator()
        while let char = chars.next() {
            if char == "\"" { quoted.toggle() }
            else if char == "," && !quoted { row.append(field); field = "" }
            else if (char == "\n" || char == "\r\n") && !quoted {
                row.append(field); rows.append(row); row = []; field = ""
                guard rows.count <= 30_000 else { throw URLError(.cannotParseResponse) }
            } else { field.append(char) }
        }
        guard !quoted else { throw URLError(.cannotParseResponse) }
        if !field.isEmpty || !row.isEmpty { row.append(field); rows.append(row) }
        guard let header = rows.first, Set(header).count == header.count else { throw URLError(.cannotParseResponse) }
        return rows
    }
}
