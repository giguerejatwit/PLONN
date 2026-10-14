import Foundation

/// Discovery identifies a request, never establishes a statistic or a prediction.
enum AskPLONNContextTarget: Sendable, Equatable, Identifiable {
    case game(SharedMatchupEvent)
    case team(sport: String, name: String)
    case player(sport: String, name: String, stat: String)

    var id: String {
        switch self {
        case .game(let event): return [event.sport, event.eventID, event.homeTeam, event.awayTeam, event.startISO].joined(separator: "|")
        case .team(let sport, let name): return "team|\(sport)|\(name)"
        case .player(let sport, let name, let stat): return "player|\(sport)|\(name)|\(stat)"
        }
    }
    var sport: String {
        switch self { case .game(let event): return event.sport; case .team(let sport, _), .player(let sport, _, _): return sport }
    }
    var label: String {
        switch self {
        case .game(let event):
            let start = sportsDate(event.startISO)?.formatted(date: .abbreviated, time: .shortened) ?? event.startISO
            return "\(event.awayTeam) at \(event.homeTeam) · \(event.sport) · \(start)"
        case .team(let sport, let name): return "\(name) · \(sport)"
        case .player(let sport, let name, let stat): return "\(name) · \(sport) · \(OpponentPropHistoryEngine.statLabel(stat))"
        }
    }
}

struct AskPLONNContextCatalog: Sendable {
    var events: [SharedMatchupEvent] = []
    var teams: [AnalyticsFeedVerifiedSubject] = []
    var completeSports: Set<String> = []
}
enum AskPLONNContextResolution: Sendable {
    case resolved(AskPLONNContextTarget)
    case clarification(String, [AskPLONNContextTarget])
}

enum AskPLONNContextResolver {
    static let sports = ["NFL", "NBA", "NHL", "MLB"]
    static func normalized(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .components(separatedBy: CharacterSet.alphanumerics.inverted).filter { !$0.isEmpty }.joined(separator: " ")
    }
    static func contains(_ text: String, _ phrase: String) -> Bool {
        (" " + normalized(text) + " ").contains(" " + normalized(phrase) + " ")
    }
    static func explicitSports(_ question: String) -> [String] {
        let aliases = ["NFL": ["nfl", "football"], "NBA": ["nba", "basketball"], "NHL": ["nhl", "hockey"], "MLB": ["mlb", "baseball"]]
        return sports.filter { sport in aliases[sport]!.contains { contains(question, $0) } }
    }
    static func isFollowup(_ question: String) -> Bool {
        let text = normalized(question)
        return text.hasPrefix("what about ") || ["that game", "this game", "those teams", "their", "his", "tell me more"].contains { contains(text, $0) }
    }
    static func shouldDiscover(_ question: String, previous: AskPLONNContextTarget? = nil, markets: [Market] = []) -> Bool {
        markets.contains(where: { contains(question, $0.name) }) || AskPLONNCore.requiresResearchForQuestion(question) || !statusDimensions(question).isEmpty || contains(question, "stats") || contains(question, "expect") || contains(question, "matchup") ||
            (explicitSports(question).contains { !requestedStats(question, sport: $0).isEmpty }) || (previous != nil && isFollowup(question))
    }
    static func statusDimensions(_ question: String) -> Set<String> {
        var dimensions = Set<String>()
        if ["injury", "injuries", "injured", "availability", "available", "status", "out", "questionable", "doubtful", "playing", "healthy"].contains(where: { contains(question, $0) }) { dimensions.insert("availability") }
        if ["starter", "starters", "starting", "lineup", "lineups"].contains(where: { contains(question, $0) }) { dimensions.insert("starters") }
        return dimensions
    }
    /// Calendar supplied by the caller; today's games follow the user's date, including DST.
    static func day(_ question: String, now: Date, calendar: Calendar) -> Date? {
        let text = question.lowercased()
        let iso = try! NSRegularExpression(pattern: #"\b\d{4}-\d{2}-\d{2}\b"#)
        let matches = iso.matches(in: text, range: NSRange(text.startIndex..., in: text))
        if matches.count > 1 { return nil }
        if let match = matches.first, let range = Range(match.range, in: text) {
            let formatter = DateFormatter(); formatter.calendar = calendar; formatter.timeZone = calendar.timeZone
            formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.dateFormat = "yyyy-MM-dd"; formatter.isLenient = false
            return formatter.date(from: String(text[range]))
        }
        let relative = ["today", "tonight", "tomorrow", "yesterday"].filter { contains(text, $0) }
        if Set(relative.map { $0 == "tonight" ? "today" : $0 }).count > 1 { return nil }
        if ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "next week", "last week", "october", "november", "december", "january", "february", "march", "april", "may", "june", "july", "august", "september"].contains(where: { contains(text, $0) }) { return nil }
        return calendar.date(byAdding: .day, value: contains(text, "tomorrow") ? 1 : contains(text, "yesterday") ? -1 : 0, to: calendar.startOfDay(for: now))
    }
    static func requestedStats(_ question: String, sport: String) -> [String] {
        let mappings: [String: [(String, [String])]] = [
            "MLB": [("pitcher_strikeouts", ["strikeout", "strikeouts", "ks"]), ("batter_hits", ["hit", "hits"]), ("batter_home_runs", ["home run", "home runs"])],
            "NBA": [("player_points", ["points", "scoring"]), ("player_rebounds", ["rebounds"]), ("player_assists", ["assists"])],
            "NFL": [("player_pass_yds", ["passing", "pass yards"]), ("player_rush_yds", ["rushing", "rush yards"]), ("player_reception_yds", ["receiving", "receiving yards"]), ("player_receptions", ["receptions", "catches"])],
            "NHL": [("player_shots_on_goal", ["shots", "shots on goal"]), ("player_total_saves", ["saves"]), ("player_points", ["points"]), ("player_goals", ["goals"])]]
        return (mappings[sport] ?? []).filter { entry in entry.1.contains { contains(question, $0) } && PlayerHistoryService.statKeys(sport: sport, marketKey: entry.0) != nil }.map(\.0)
    }
    static func matchesTeam(_ question: String, name: String) -> Bool {
        let words = normalized(name).split(separator: " ")
        // Sourced full names and nicknames only; city-only matching is too ambiguous.
        let nickname = words.suffix(words.last == "sox" || words.last == "blazers" || words.last == "jackets" || words.last == "knights" || words.last == "wings" || words.last == "leafs" ? 2 : 1).joined(separator: " ")
        return contains(question, name) || (nickname.count >= 4 && contains(question, nickname))
    }
    /// Full-name extraction is only a search query. PlayerHistoryService must verify identity.
    static func namedPlayer(_ question: String, markets: [Market]) -> AskPLONNContextResolution? {
        let scope = explicitSports(question)
        let known = markets.filter { $0.kind == "prop" && sports.contains($0.sport) && (scope.isEmpty || scope.contains($0.sport)) && contains(question, $0.name) }
        var targets: [AskPLONNContextTarget] = []
        for market in known {
            let stats = requestedStats(question, sport: market.sport)
            for stat in stats.isEmpty ? [market.market] : stats {
                if PlayerHistoryService.statKeys(sport: market.sport, marketKey: stat) != nil { targets.append(.player(sport: market.sport, name: market.name, stat: stat)) }
            }
        }
        if targets.isEmpty, scope.count == 1 {
            var text = question.trimmingCharacters(in: .whitespacesAndNewlines)
            for word in [scope[0], "baseball", "football", "basketball", "hockey"] { text = text.replacingOccurrences(of: "\\b" + word + "\\b", with: "", options: [.regularExpression, .caseInsensitive]) }
            let pattern = #"^\s*(?:(?:show|give|tell)\s+(?:me\s+)?|(?:what (?:are|were)|how (?:has|did))\s+)?([\p{L}][\p{L}'’.-]+\s+[\p{L}][\p{L}'’.-]+(?:\s+(?:Jr\.?|Sr\.?|III|II))?)(?:['’]s)?\s+(?:recent\s+)?(?:hits|strikeouts|points|rebounds|assists|passing|rushing|receiving|receptions|shots|saves|goals|performed|played)\b"#
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive), let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)), let range = Range(match.range(at: 1), in: text) {
                let name = String(text[range])
                if !["the game", "the team", "high strikeouts", "this game"].contains(normalized(name)) {
                    targets = requestedStats(question, sport: scope[0]).map { .player(sport: scope[0], name: name, stat: $0) }
                }
            }
        }
        let unique = Dictionary(grouping: targets, by: \.id).compactMap { $0.value.first }.sorted { $0.id < $1.id }
        if unique.count == 1 { return .resolved(unique[0]) }
        if !unique.isEmpty { return .clarification("Which player and stat would you like to research?", Array(unique.prefix(8))) }
        return nil
    }
    static func isGenericGameQuestion(_ question: String) -> Bool {
        let vocabulary = Set("what can could should do does we i you expect from for the a an in on of at to about tell me show give us is are there any game games matchup matchups today tonight tomorrow yesterday mlb nba nfl nhl baseball basketball football hockey high low higher lower scoring score scores runs hits strikeouts ks goals shots points passing rushing yards pace rebounds assists saves outlook look like please and with recorded recent performance who injury injuries injured availability available status out questionable doubtful playing healthy starter starters starting lineup lineups probable confirmed current updated".split(separator: " ").map(String.init))
        let withoutDate = question.replacingOccurrences(of: #"\b\d{4}-\d{2}-\d{2}\b"#, with: "", options: .regularExpression)
        return normalized(withoutDate).split(separator: " ").allSatisfy { vocabulary.contains(String($0)) }
    }
    static func resolve(question: String, catalog: AskPLONNContextCatalog, now: Date, calendar: Calendar = .current) -> AskPLONNContextResolution {
        guard let day = day(question, now: now, calendar: calendar) else { return .clarification("Which date do you mean? Use today, tomorrow, or a date such as 2026-10-14.", []) }
        let explicit = explicitSports(question), scope = explicit.isEmpty ? sports : explicit
        let teams = catalog.teams.filter { scope.contains($0.sport) && matchesTeam(question, name: $0.name) }
        let uniqueTeams = Dictionary(grouping: teams, by: { $0.sport + "|" + $0.name }).compactMap { $0.value.first }
        let gameQuestion = !statusDimensions(question).isEmpty || ["game", "games", "matchup", "expect", "today", "tonight", "tomorrow", "yesterday", "vs", "versus", "against"].contains { contains(question, $0) } || uniqueTeams.count > 1
        if !gameQuestion, !uniqueTeams.isEmpty, Set(scope).isSubset(of: catalog.completeSports) {
            let choices = uniqueTeams.map { AskPLONNContextTarget.team(sport: $0.sport, name: $0.name) }.sorted { $0.id < $1.id }
            return choices.count == 1 ? .resolved(choices[0]) : .clarification("Which team do you mean?", Array(choices.prefix(8)))
        }
        if uniqueTeams.isEmpty && !isGenericGameQuestion(question) {
            return .clarification("Which teams or player do you mean? Include the league and full name so I can match verified research.", [])
        }
        let candidates = catalog.events.filter { event in
            guard scope.contains(event.sport), let start = sportsDate(event.startISO), calendar.isDate(start, inSameDayAs: day) else { return false }
            let named = uniqueTeams.filter { $0.sport == event.sport }
            return uniqueTeams.isEmpty || (!named.isEmpty && named.allSatisfy { [event.homeTeam, event.awayTeam].contains($0.name) })
        }
        let choices = Dictionary(grouping: candidates, by: { $0.sport + "|" + $0.eventID }).values.compactMap { copies -> SharedMatchupEvent? in
            guard let first = copies.first, copies.allSatisfy({ $0 == first }) else { return nil }; return first
        }.sorted { $0.startISO == $1.startISO ? $0.eventID < $1.eventID : $0.startISO < $1.startISO }.map(AskPLONNContextTarget.game)
        if ["vs", "versus", "against"].contains(where: { contains(question, $0) }), uniqueTeams.count < 2 {
            return .clarification("Please name both teams and the league so I can verify the exact matchup.", [])
        }
        if !Set(scope).isSubset(of: catalog.completeSports) {
            return .clarification("I couldn’t verify the complete schedule for that request. Specify the league and both teams, or retry when the schedule is available.", [])
        }
        if choices.count == 1, gameQuestion || !explicit.isEmpty { return .resolved(choices[0]) }
        if choices.count > 1 { return .clarification("Which game do you mean?" + (choices.count > 8 ? " Name both teams if your game is not listed." : ""), Array(choices.prefix(8))) }
        return .clarification("I couldn’t find a verified game for that date and league. Name both teams, or ask about a player’s full name, league, and stat.", [])
    }
}
