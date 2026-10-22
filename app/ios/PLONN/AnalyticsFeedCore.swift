import Foundation

enum AnalyticsFeedSubject: String, Sendable { case team, player }

/// These records must come from a verified public directory/scoreboard/roster read,
/// not a suggested name, forecast, popularity list or sportsbook description.
struct AnalyticsFeedVerifiedSubject: Sendable, Equatable {
    let sport: String
    let subject: AnalyticsFeedSubject
    let name: String
    let providerID: String
    var teamName: String? = nil
    var teamID: String? = nil
    var imageURL: String? = nil
    var teamLogoURL: String? = nil
    let marketKeys: [String]
    var event: AnalyticsEvent? = nil
    let sourceURL: String
    let verifiedAt: Date
}
struct AnalyticsFeedCandidate: Identifiable, Sendable {
    let id: String
    let subject: AnalyticsFeedSubject
    /// A line-independent statistics request; quotes retain their original scope separately.
    let market: Market
    let quotes: [Market]
    let verifiedSubject: AnalyticsFeedVerifiedSubject?
    /// True only for an authenticated cache-only response validated by the feed service.
    var recordedSnapshot: Bool = false
    var recordedArchive: Bool = false
}
struct AnalyticsFeedBuildResult: Sendable {
    let candidates: [AnalyticsFeedCandidate]
    let warnings: [String]
}

enum AnalyticsFeedCore {
    static let browsingCapacity = 10_000
    /// Only source-returned ESPN image URLs; no name-based image discovery.
    static func imageURL(_ raw: String?) -> URL? {
        guard let raw, let url = URL(string: raw), url.scheme == "https",
              url.host?.lowercased() == "a.espncdn.com", url.user == nil, url.password == nil,
              url.port == nil, url.path.hasPrefix("/i/") else { return nil }
        return url
    }
    private static func normalized(_ value: String) -> String { AnalyticsOddsMatcher.normalized(value) }
    static func supports(sport: String, subject: AnalyticsFeedSubject, marketKey: String) -> Bool {
        guard ["NBA", "NFL", "MLB", "NHL"].contains(sport) else { return false }
        if subject == .player { return AnalyticsHistoryEngine.statisticKey(sport: sport, marketKey: marketKey) != nil }
        return ["h2h", "spreads", "totals", "team_totals"].contains(marketKey)
    }
    static func candidateID(sport: String, subject: AnalyticsFeedSubject, name: String, marketKey: String) -> String {
        [sport, subject.rawValue, normalized(name), marketKey].map { value in
            "\(value.utf8.count):\(value)"
        }.joined(separator: "|")
    }
    private static func statisticsMarket(sport: String, subject: AnalyticsFeedSubject, name: String, key: String, id: String) -> Market {
        Market(id: id, event: "", sport: sport, kind: subject == .player ? "prop" : "team", name: name, opponent: "", label: subject == .player ? OpponentPropHistoryEngine.statLabel(AnalyticsHistoryEngine.statisticKey(sport: sport, marketKey: key) ?? key) : (key == "team_totals" ? "Team score" : key), aliases: "", p: nil, price: 0, market: key, book: "", quotes: [], updated: nil, mode: "analytics", logs: nil, line: nil, eventLabel: nil, modelValidated: nil, modelUpdated: nil, modelVersion: nil, eventState: nil, commence: nil, side: nil)
    }
    private static func validProof(_ proof: AnalyticsFeedVerifiedSubject, now: Date) -> Bool {
        guard !normalized(proof.name).isEmpty, !proof.providerID.isEmpty,
              proof.verifiedAt.timeIntervalSince1970.isFinite, proof.verifiedAt <= now, now.timeIntervalSince(proof.verifiedAt) <= 900,
              let url = URL(string: proof.sourceURL), PlayerHistoryService.allowedURL(url) else { return false }
        if proof.subject == .player {
            guard url.path.hasSuffix("/roster"), let team = proof.teamID, !team.isEmpty,
                  let name = proof.teamName, !normalized(name).isEmpty, url.path.hasSuffix("/teams/\(team)/roster") else { return false }
        } else {
            guard url.path.hasSuffix("/teams") || url.path.hasSuffix("/scoreboard") || url.path.contains("/teams/") || (proof.sport == "MLB" && url.host == "statsapi.mlb.com" && url.path == "/api/v1/schedule") else { return false }
        }
        if let event = proof.event {
            guard event.sport == proof.sport, event.start.timeIntervalSince1970.isFinite, !event.providerID.isEmpty,
                  !normalized(event.home).isEmpty, !normalized(event.away).isEmpty, normalized(event.home) != normalized(event.away) else { return false }
            let name = proof.subject == .team ? proof.name : proof.teamName ?? ""
            guard [event.home, event.away].map(normalized).contains(normalized(name)) else { return false }
            if proof.subject == .player {
                guard [event.homeID, event.awayID].compactMap({ $0 }).contains(proof.teamID ?? "") else { return false }
            }
        }
        return true
    }
    static func build(snapshots: [Market], verifiedSubjects: [AnalyticsFeedVerifiedSubject] = [], now: Date = Date(), limit: Int = 12, recordedSnapshot: Bool = false) -> AnalyticsFeedBuildResult {
        guard now.timeIntervalSince1970.isFinite else { return AnalyticsFeedBuildResult(candidates: [], warnings: ["The feed cutoff is invalid."]) }
        // Input ceilings protect the local feed independently of source response limits.
        guard snapshots.count <= 10_000, verifiedSubjects.count <= browsingCapacity else { return AnalyticsFeedBuildResult(candidates: [], warnings: ["Feed source capacity exceeded; no partial source was silently selected."]) }
        var warnings: [String] = []
        var names: [String: (String, AnalyticsFeedSubject, String, String)] = [:]
        var quotes: [String: [Market]] = [:]
        var proofs: [String: AnalyticsFeedVerifiedSubject] = [:]
        var conflictingProofs = Set<String>()
        var subjectProofs: [String: AnalyticsFeedVerifiedSubject] = [:]
        var conflictingSubjects = Set<String>()
        let snapshotGroups = Dictionary(grouping: snapshots, by: \.id)
        var cleanSnapshots: [Market] = []
        for (id, copies) in snapshotGroups.sorted(by: { $0.key < $1.key }) {
            guard !id.isEmpty, let first = copies.first else { continue }
            guard copies.allSatisfy({ $0 == first }) else { warnings.append("Conflicting copies of snapshot \(id) were excluded."); continue }
            guard first.mode == "live" else { continue }
            cleanSnapshots.append(first)
        }
        var eventScopes: [String: (String, String, Date)] = [:], conflictingEvents = Set<String>()
        for snapshot in cleanSnapshots {
            guard !snapshot.event.isEmpty, let label = snapshot.eventLabel, !normalized(label).isEmpty, let start = snapshot.commence.flatMap(sportsDate) else { continue }
            let scope = (snapshot.sport, normalized(label), start)
            if let prior = eventScopes[snapshot.event], prior.0 != scope.0 || prior.1 != scope.1 || abs(prior.2.timeIntervalSince(scope.2)) > 60 { conflictingEvents.insert(snapshot.event) }
            eventScopes[snapshot.event] = scope
        }
        for snapshot in cleanSnapshots {
            let subject: AnalyticsFeedSubject = snapshot.kind == "prop" ? .player : .team
            guard snapshot.kind == "prop" || ["team", "game"].contains(snapshot.kind), supports(sport: snapshot.sport, subject: subject, marketKey: snapshot.market) else { continue }
            // Game totals remain full-game totals for this genuine team context.
            // outcome.description never turns a team total into a player prop.
            let name = subject == .team && snapshot.market == "totals" && ["over", "under"].contains(normalized(snapshot.name)) ? snapshot.opponent : snapshot.name
            guard !normalized(name).isEmpty else { continue }
            if snapshot.market == "team_totals" {
                guard subject == .team, snapshot.kind == "team",
                      let label = snapshot.eventLabel,
                      declaredTeams(label).contains(normalized(name)) else { continue }
            }
            let id = candidateID(sport: snapshot.sport, subject: subject, name: name, marketKey: snapshot.market)
            if names[id] == nil { names[id] = (snapshot.sport, subject, name, snapshot.market) }
            if !snapshot.event.isEmpty, !conflictingEvents.contains(snapshot.event), snapshot.eventLabel?.isEmpty == false,
               snapshot.commence.flatMap(sportsDate) != nil, !snapshot.book.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               snapshot.price.isFinite, abs(snapshot.price) >= 100, snapshot.line.map({ $0.isFinite }) != false,
               let updated = snapshot.updated.flatMap(sportsDate), updated <= now {
                quotes[id, default: []].append(snapshot)
            }
        }
        if !conflictingEvents.isEmpty { warnings.append("Conflicting event scopes were excluded from cached comparisons.") }
        for proof in verifiedSubjects.sorted(by: { $0.sourceURL == $1.sourceURL ? $0.providerID < $1.providerID : $0.sourceURL < $1.sourceURL }) {
            guard validProof(proof, now: now) else { warnings.append("Unverified or expired public subject context was excluded."); continue }
            let subjectID = candidateID(sport: proof.sport, subject: proof.subject, name: proof.name, marketKey: "")
            if let existing = subjectProofs[subjectID], existing.providerID != proof.providerID || existing.teamID != proof.teamID || existing.teamName.map(normalized) != proof.teamName.map(normalized) || existing.event != proof.event { conflictingSubjects.insert(subjectID) }
            if subjectProofs[subjectID] == nil { subjectProofs[subjectID] = proof }
            for key in Set(proof.marketKeys).sorted() where supports(sport: proof.sport, subject: proof.subject, marketKey: key) {
                let id = candidateID(sport: proof.sport, subject: proof.subject, name: proof.name, marketKey: key)
                if let existing = proofs[id], existing.providerID != proof.providerID || existing.teamID != proof.teamID || existing.teamName.map(normalized) != proof.teamName.map(normalized) || existing.event != proof.event {
                    conflictingProofs.insert(id)
                }
                if proofs[id] == nil { proofs[id] = proof }
                if names[id] == nil { names[id] = (proof.sport, proof.subject, proof.name, key) }
            }
        }
        if !conflictingProofs.isEmpty || !conflictingSubjects.isEmpty { warnings.append("Conflicting public subject identities were quarantined.") }
        let candidates = names.keys.sorted().compactMap { id -> AnalyticsFeedCandidate? in
            guard !conflictingProofs.contains(id), let (sport, subject, name, key) = names[id] else { return nil }
            let subjectID = candidateID(sport: sport, subject: subject, name: name, marketKey: "")
            guard !conflictingSubjects.contains(subjectID) else { return nil }
            return AnalyticsFeedCandidate(id: id, subject: subject, market: statisticsMarket(sport: sport, subject: subject, name: name, key: key, id: id), quotes: ((quotes[id] ?? []) + (subject == .team && key == "h2h" ? (quotes[candidateID(sport: sport, subject: subject, name: name, marketKey: "spreads")] ?? []) : [])).sorted { $0.id < $1.id }, verifiedSubject: proofs[id] ?? subjectProofs[subjectID], recordedSnapshot: recordedSnapshot)
        }
        // Neutral deterministic interleaving preserves both subject types without ranking performance.
        let teams = candidates.filter { $0.subject == .team }, players = candidates.filter { $0.subject == .player }
        var mixed: [AnalyticsFeedCandidate] = []
        for index in 0..<max(teams.count, players.count) {
            if index < teams.count { mixed.append(teams[index]) }
            if index < players.count { mixed.append(players[index]) }
        }
        return AnalyticsFeedBuildResult(candidates: Array(mixed.prefix(min(max(limit, 1), recordedSnapshot ? 30_000 : browsingCapacity))), warnings: Array(Set(warnings)).sorted())
    }
    /// Public statistics remain visible while history loads or matching odds are unavailable.
    /// Quote eligibility is checked separately inside each card.
    static func visibleCandidates(_ candidates: [AnalyticsFeedCandidate], sport: String, query: String,
                                  histories: [String: AnalyticsHistoryResult], now: Date) -> [AnalyticsFeedCandidate] {
        guard ["NFL", "NBA", "MLB", "NHL"].contains(sport) else { return [] }
        let league = candidates.filter { $0.market.sport == sport }
        if !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return search(candidates: league, query: query) }
        return league.filter { candidate in
            candidate.verifiedSubject.map { validProof($0, now: now) } == true
        }
    }
    private static func declaredTeams(_ label: String) -> [String] {
        let label = normalized(label)
        for separator in [" at ", " @ "] {
            let parts = label.components(separatedBy: separator)
            if parts.count == 2, parts.allSatisfy({ !$0.isEmpty }), parts[0] != parts[1] { return parts }
        }
        return []
    }
    /// Comparison eligibility remains separate from candidate discovery and never guesses a roster.
    static func eligibleComparisons(candidate: AnalyticsFeedCandidate, history: AnalyticsHistoryResult? = nil, now: Date = Date()) -> [Market] {
        guard let proof = candidate.verifiedSubject, validProof(proof, now: now), let event = proof.event else { return [] }
        if candidate.subject == .player {
            guard let history, history.verifiedPlayerID == proof.providerID, (history.statistics?.valid ?? 0) > 0 else { return [] }
            // ESPN roster IDs and MLB Stats API athlete IDs are different namespaces.
            if history.source == "MLB Stats API", URL(string: proof.sourceURL)?.host != "statsapi.mlb.com" { return [] }
        }
        let teamName = proof.subject == .team ? proof.name : proof.teamName ?? ""
        let teamID = proof.subject == .team ? proof.providerID : proof.teamID
        let target = AnalyticsStatsTarget(sport: proof.sport, kind: proof.subject == .player ? "prop" : "team", name: proof.name, marketKey: candidate.market.market, teamName: teamName, teamID: teamID, playerID: proof.subject == .player ? proof.providerID : nil, rosterVerifiedAt: proof.subject == .player ? proof.verifiedAt : nil)
        return AnalyticsOddsMatcher.matchCached(target: target, event: event, markets: candidate.quotes, now: now, recordedSnapshot: candidate.recordedSnapshot, recordedArchive: candidate.recordedArchive).markets
    }
    /// Default betting-feed eligibility is stricter than general statistics discovery.
    /// Authenticated recorded snapshots keep original prices/times, even after 15 minutes.
    /// Unattributed retained selections still require the original 15-minute window.
    /// Neither path represents a cached price as live.
    /// Preserve each returned quote's update time; no fetched/expires time is invented.
    static func bettingFeedQuotes(candidate: AnalyticsFeedCandidate, history: AnalyticsHistoryResult?, now: Date = Date()) -> [Market] {
        guard now.timeIntervalSince1970.isFinite,
              let history, let stats = history.statistics, stats.valid > 0, stats.n >= stats.valid,
              let mean = stats.mean, mean.isFinite,
              let start = stats.startDate, let end = stats.endDate,
              start.timeIntervalSince1970.isFinite, end.timeIntervalSince1970.isFinite,
              start <= end, end <= now,
              let proof = candidate.verifiedSubject, proof.subject == candidate.subject,
              proof.sport == candidate.market.sport, normalized(proof.name) == normalized(candidate.market.name),
              let event = proof.event else { return [] }
        let teamName = proof.subject == .team ? proof.name : proof.teamName ?? ""
        let teamID = proof.subject == .team ? proof.providerID : proof.teamID
        let matchesHome = normalized(teamName) == normalized(event.home) && teamID == event.homeID && event.homeID != nil
        let matchesAway = normalized(teamName) == normalized(event.away) && teamID == event.awayID && event.awayID != nil
        guard matchesHome != matchesAway else { return [] }
        return eligibleComparisons(candidate: candidate, history: history, now: now).filter { quote in
            guard let updated = quote.updated.flatMap(sportsDate) else { return false }
            let age = now.timeIntervalSince(updated)
            return age.isFinite && age >= 0 && (candidate.recordedSnapshot || age <= 900)
        }
    }
    /// Score difference measures the margin, so show spreads even when discovery
    /// selected the team's moneyline seed. Quotes themselves are never rewritten.
    static func cardOdds(candidate: AnalyticsFeedCandidate, history: AnalyticsHistoryResult?, now: Date = Date()) -> [Market] {
        var request = candidate.market
        if candidate.subject == .team, request.market == "h2h" { request.market = "spreads" }
        let scoped = AnalyticsFeedCandidate(id: candidate.id, subject: candidate.subject, market: request,
            quotes: candidate.quotes, verifiedSubject: candidate.verifiedSubject,
            recordedSnapshot: candidate.recordedSnapshot, recordedArchive: candidate.recordedArchive)
        return bettingFeedQuotes(candidate: scoped, history: history, now: now)
    }
    static func cardOddsTitle(_ candidate: AnalyticsFeedCandidate) -> String {
        switch candidate.market.market {
        case "h2h", "spreads": return "Spread odds"
        case "totals": return "Game total odds"
        case "team_totals": return "Team total odds"
        default: return "Player prop odds"
        }
    }
    static func cardOddsContract(_ quote: Market) -> String {
        let side = quote.market == "spreads" ? quote.name : (quote.side?.capitalized ?? quote.name)
        let line = quote.line.map { String(format: quote.market == "spreads" ? "%+g" : "%g", $0) } ?? ""
        return [side, line].filter { !$0.isEmpty }.joined(separator: " ")
    }
    static func search(candidates: [AnalyticsFeedCandidate], query: String) -> [AnalyticsFeedCandidate] {
        let tokens = normalized(query).split(separator: " ").map(String.init)
        guard !tokens.isEmpty else { return candidates }
        return candidates.filter { candidate in
            let market = candidate.market
            let quoteTerms = candidate.quotes.flatMap { [$0.name, $0.opponent, $0.eventLabel ?? "", $0.aliases] }.joined(separator: " ")
            let searchable = normalized([quoteTerms, market.name, market.sport, market.market, market.market.replacingOccurrences(of: "_", with: " "), market.label, candidate.subject.rawValue, candidate.verifiedSubject?.teamName ?? "", candidate.verifiedSubject?.event?.home ?? "", candidate.verifiedSubject?.event?.away ?? ""] .joined(separator: " "))
            return tokens.allSatisfy { searchable.contains($0) }
        }
    }
}

