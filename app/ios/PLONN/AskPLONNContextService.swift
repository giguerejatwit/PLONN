import Foundation

struct AskPLONNAutomaticResearch: Sendable {
    let target: AskPLONNContextTarget?
    let evidence: [AskPLONNEvidence]
    let guidance: String?
    let choices: [AskPLONNContextTarget]
}

/// Public discovery runs only after Send. The report reader is supplied by the
/// authenticated account; this service cannot enable an endpoint or generate reports.
struct AskPLONNContextService: Sendable {
    typealias Reader = @Sendable (URL) async throws -> Data
    typealias ReportReader = @Sendable (SharedMatchupEvent) async throws -> SharedMatchupReportResponse
    /// The account client validates original bytes, pins, source freshness and
    /// authorization on every read. This seam cannot generate or refresh reports.
    typealias NarrativeReader = @Sendable (SharedMatchupEvent) async throws -> MatchupNarrativeV4.Result?
    typealias PlayerReader = @Sendable (Market, Date) async -> PlayerHistoryResult
    typealias TeamReader = @Sendable (Market, Date) async -> AnalyticsHistoryResult
    let read: Reader
    let player: PlayerReader
    let team: TeamReader
    init(read: @escaping Reader = { try await PlayerHistoryService.shared.read($0) },
         player: @escaping PlayerReader = { await PlayerHistoryService.shared.load(market: $0, includeOpponent: false, seasonCount: 2, now: $1) },
         team: @escaping TeamReader = { await AnalyticsHistoryService.shared.loadStats(market: $0, now: $1) }) {
        self.read = read; self.player = player; self.team = team
    }
    func discover(question: String, markets: [Market], previous: AskPLONNContextTarget? = nil,
                  chosen: AskPLONNContextTarget? = nil, report: ReportReader? = nil, narrative: NarrativeReader? = nil,
                  now: Date = Date(), calendar: Calendar = .current) async throws -> AskPLONNAutomaticResearch {
        try Task.checkCancellation()
        let resolution: AskPLONNContextResolution
        if let chosen { resolution = .resolved(chosen) }
        else if let previous, case .game = previous,
                ["any injuries", "who is starting", "who is out", "any lineup updates"].contains(AskPLONNContextResolver.normalized(question)) {
            resolution = .resolved(previous)
        }
        else if markets.contains(where: { $0.kind == "prop" && AskPLONNContextResolver.contains(question, $0.name) }),
                let named = AskPLONNContextResolver.namedPlayer(question, markets: markets) { resolution = named }
        else if let previous, AskPLONNContextResolver.isFollowup(question),
                AskPLONNContextResolver.explicitSports(question).isEmpty,
                !["today", "tonight", "tomorrow", "yesterday", "vs", "versus"].contains(where: { AskPLONNContextResolver.contains(question, $0) }),
                Self.safeFollowup(question) {
            switch previous {
            case .player(let sport, let name, let stat):
                let stats = AskPLONNContextResolver.requestedStats(question, sport: sport)
                resolution = stats.count > 1 ? .clarification("Which stat would you like to research?", stats.map { .player(sport: sport, name: name, stat: $0) }) : .resolved(.player(sport: sport, name: name, stat: stats.first ?? stat))
            default: resolution = .resolved(previous)
            }
        } else {
            let catalog = try await catalog(question: question, now: now, calendar: calendar)
            if !catalog.teams.contains(where: { AskPLONNContextResolver.matchesTeam(question, name: $0.name) }),
               let named = AskPLONNContextResolver.namedPlayer(question, markets: markets) { resolution = named }
            else { resolution = AskPLONNContextResolver.resolve(question: question, catalog: catalog, now: now, calendar: calendar) }
        }
        try Task.checkCancellation()
        switch resolution {
        case .clarification(let text, let choices): return .init(target: nil, evidence: [], guidance: text, choices: choices)
        case .resolved(let target):
            let evidence = try await evidence(target: target, question: question, report: report, narrative: narrative, now: now)
            try Task.checkCancellation()
            return .init(target: target, evidence: evidence, guidance: nil, choices: [])
        }
    }
    /// Referential follow-ups may only reuse identity when they contain no new subject.
    static func safeFollowup(_ question: String) -> Bool {
        let allowed = Set("what about how their his this that game those teams tell me more and the a an of for in on do does can we expect high low higher lower recent form scoring points runs hits strikeouts ks shots goals saves rebounds assists passing rushing receiving yards receptions catches pace defense offense why who is are any injury injuries injured availability available status out questionable doubtful playing healthy starter starters starting lineup lineups probable confirmed current updated".split(separator: " ").map(String.init))
        return AskPLONNContextResolver.normalized(question).split(separator: " ").allSatisfy { allowed.contains(String($0)) }
    }
    private func json(_ url: URL) async throws -> [String: Any] {
        try Task.checkCancellation()
        guard PlayerHistoryService.allowedURL(url) else { throw PlayerHistoryError.invalidURL }
        let bytes = try await read(url)
        guard bytes.count <= 4 * 1024 * 1024, let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else { throw PlayerHistoryError.invalidPayload }
        return object
    }
    func catalog(question: String, now: Date, calendar: Calendar) async throws -> AskPLONNContextCatalog {
        guard let day = AskPLONNContextResolver.day(question, now: now, calendar: calendar),
              let end = calendar.date(byAdding: .day, value: 1, to: day) else { return .init() }
        let requested = AskPLONNContextResolver.explicitSports(question)
        let sports = requested.isEmpty ? AskPLONNContextResolver.sports : requested
        let format = DateFormatter(); format.locale = Locale(identifier: "en_US_POSIX"); format.timeZone = TimeZone(identifier: "America/New_York"); format.dateFormat = "yyyyMMdd"
        // ESPN date buckets use Eastern time; cover both buckets intersecting the user's day.
        let days = Set([format.string(from: day), format.string(from: end.addingTimeInterval(-1))]).sorted()
        var result = AskPLONNContextCatalog()
        for sport in sports {
            try Task.checkCancellation()
            let base = "https://site.api.espn.com/apis/site/v2/sports/\(PlayerHistoryService.leaguePath(sport))"
            var complete = true
            for date in days {
                let url = URL(string: base + "/scoreboard?dates=\(date)&limit=100")!
                do {
                    let parsed = try Self.scoreboard(await json(url), sport: sport, source: url.absoluteString, now: now)
                    result.events += parsed.events; result.teams += parsed.teams
                } catch is CancellationError { throw CancellationError() }
                catch { complete = false }
            }
            // Full directory allows team requests on off days and avoids guessing city aliases.
            let directory = URL(string: base + "/teams?limit=1000")!
            do { result.teams += try AnalyticsFeedService.directorySubjects(payload: await json(directory), sport: sport, sourceURL: directory.absoluteString, now: now) }
            catch is CancellationError { throw CancellationError() }
            catch { complete = false }
            if complete { result.completeSports.insert(sport) }
        }
        return result
    }
    static func scoreboard(_ payload: [String: Any], sport: String, source: String, now: Date) throws -> AskPLONNContextCatalog {
        guard AskPLONNContextResolver.sports.contains(sport), let events = payload["events"] as? [[String: Any]], events.count < 100 else { throw PlayerHistoryError.invalidPayload }
        var result = AskPLONNContextCatalog()
        for raw in events {
            guard let id = raw["id"] as? String, !id.isEmpty, id.allSatisfy(\.isNumber), let start = raw["date"] as? String,
                  let competitions = raw["competitions"] as? [[String: Any]], competitions.count == 1,
                  let competitors = competitions[0]["competitors"] as? [[String: Any]], competitors.count == 2,
                  let home = competitors.first(where: { $0["homeAway"] as? String == "home" })?["team"] as? [String: Any],
                  let away = competitors.first(where: { $0["homeAway"] as? String == "away" })?["team"] as? [String: Any],
                  let hn = home["displayName"] as? String, let an = away["displayName"] as? String,
                  let hi = home["id"] as? String, let ai = away["id"] as? String, hi != ai, !hi.isEmpty, !ai.isEmpty,
                  hi.allSatisfy(\.isNumber), ai.allSatisfy(\.isNumber) else { throw PlayerHistoryError.invalidPayload }
            let event = try SharedMatchupReportValidator.normalizedEvent(.init(sport: sport, eventID: "espn:" + id, homeTeam: hn, awayTeam: an, startISO: SharedMatchupEvent.scoreboardStart(start)))
            let status = ((competitions[0]["status"] as? [String: Any])?["type"] as? [String: Any]) ?? ((raw["status"] as? [String: Any])?["type"] as? [String: Any])
            guard let state = status?["state"] as? String, ["pre", "in", "post"].contains(state), let name = status?["name"] as? String else { throw PlayerHistoryError.invalidPayload }
            if ["STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED", "STATUS_SUSPENDED"].contains(name) { continue }
            result.events.append(event)
            for (name, id) in [(hn, hi), (an, ai)] {
                result.teams.append(.init(sport: sport, subject: .team, name: name, providerID: id, marketKeys: ["team_totals"], sourceURL: source, verifiedAt: now))
            }
        }
        return result
    }
    static func market(sport: String, name: String, stat: String = "team_totals", event: SharedMatchupEvent? = nil, player: Bool = false) -> Market {
        Market(id: "chat-research|\(sport)|\(name)|\(stat)", event: event?.eventID ?? "", sport: sport, kind: player ? "prop" : "team", name: name,
               opponent: event.map { $0.homeTeam == name ? $0.awayTeam : $0.homeTeam } ?? "", label: player ? OpponentPropHistoryEngine.statLabel(stat) : "Team score", aliases: "", p: nil, price: 0, market: stat, book: "", quotes: [], updated: nil, mode: "research", logs: nil, line: nil, eventLabel: nil, modelValidated: nil, modelUpdated: nil, modelVersion: nil, eventState: nil, commence: event?.startISO, side: nil)
    }
    static func gap(_ text: String, id: String = "automatic-context-gap", stale: Bool = false) -> AskPLONNEvidence {
        .init(id: id, kind: .recentForm, title: "Research availability", text: text, source: "PLONN verified research", availability: stale ? .stale : .unavailable)
    }
    private func evidence(target: AskPLONNContextTarget, question: String, report: ReportReader?, narrative: NarrativeReader?, now: Date) async throws -> [AskPLONNEvidence] {
        switch target {
        case .player(let sport, let name, let stat):
            if !AskPLONNContextResolver.statusDimensions(question).isEmpty {
                return [Self.gap("Current status for \(name) requires a verified game-specific injury or lineup report. Name the league and matchup to check that report; recorded player performance does not establish availability.", id: "player-status-gap")]
            }
            let market = Self.market(sport: sport, name: name, stat: stat, player: true)
            let history = await player(market, now)
            return [AskPLONNResearch.recentForm(market: market, history: history, now: now)]
        case .team(let sport, let name):
            var facts = [await teamEvidence(Self.market(sport: sport, name: name), now: now)]
            if !AskPLONNContextResolver.requestedStats(question, sport: sport).isEmpty {
                facts.append(Self.gap("Only recorded team scores were verified here. They do not establish other team statistics or a player projection.", id: "requested-stat-gap"))
            }
            return facts
        case .game(let event):
            let requested = Self.narrativeDimensions(question, sport: event.sport)
            let statusOnly = !requested.isEmpty && requested.isSubset(of: ["availability", "starters"])
            var narrativeGaps: [AskPLONNEvidence] = []
            if let narrative {
                do {
                    if let recorded = try await narrative(event) {
                        try Task.checkCancellation()
                        let richer = try Self.narrativeEvidence(recorded, event: event, question: question, now: now)
                        if statusOnly || richer.contains(where: { $0.availability == .available }) { return richer }
                        narrativeGaps = richer
                    }
                } catch is CancellationError { throw CancellationError() }
                catch {
                    narrativeGaps = [Self.gap("The richer recorded matchup research could not be verified. Any legacy research below is checked separately.", id: "narrative-unavailable")]
                }
            }
            try Task.checkCancellation()
            if statusOnly {
                return narrativeGaps + [Self.gap("Current player status or starter designations could not be verified. Missing status does not mean available; no lineup or matchup impact has been inferred.", id: "narrative-status-gap")]
            }
            var facts: [AskPLONNEvidence] = []
            if let report {
                do { facts = try Self.reportEvidence(await report(event), event: event, question: question, now: now) }
                catch is CancellationError { throw CancellationError() }
                catch { facts = [Self.gap("The verified matchup report could not be loaded. No outlook has been inferred.")] }
            } else { facts = [Self.gap("The verified matchup report is unavailable. Sign in to access saved matchup research.")] }
            if !facts.contains(where: { $0.availability == .available }) {
                for name in [event.awayTeam, event.homeTeam] {
                    try Task.checkCancellation()
                    facts.append(await teamEvidence(Self.market(sport: event.sport, name: name, event: event), now: now))
                }
                if !AskPLONNContextResolver.requestedStats(question, sport: event.sport).isEmpty {
                    facts.append(Self.gap("These team scores do not establish player stats or a high/low expectation for the requested stat. Name a player to check recorded performance.", id: "requested-stat-gap"))
                }
            }
            return narrativeGaps + facts
        }
    }
    /// Project only independently validated claims and availability observations.
    /// Reported status, scope and exact-game confirmation remain separate fields.
    static func narrativeEvidence(_ result: MatchupNarrativeV4.Result, event: SharedMatchupEvent,
                                  question: String, now: Date) throws -> [AskPLONNEvidence] {
        guard try SharedMatchupReportValidator.normalizedEvent(result.event) == SharedMatchupReportValidator.normalizedEvent(event),
              ["fresh", "partial"].contains(result.contextState), !result.providers.isEmpty,
              (1...6).contains(result.claims.count), Set(result.claims.map(\.id)).count == result.claims.count,
              try SharedMatchupReportValidator.date(result.generatedAt) <= now,
              try SharedMatchupReportValidator.date(event.startISO) > now else { throw SharedMatchupReportError.invalidResponse }
        let requested = narrativeDimensions(question, sport: event.sport)
        let selected = result.claims.filter { requested.isEmpty || requested.contains(requestDimension($0.dimension)) }
        let generated = try SharedMatchupReportValidator.date(result.generatedAt)
        var evidence = selected.map { claim in
            AskPLONNEvidence(id: "narrative|\(event.sport)|\(event.eventID)|\(claim.id)", kind: .recentForm,
                title: claim.label, text: claim.text + "\n\n" + claim.uncertainty,
                source: result.providers.joined(separator: ", ") + " · Recorded model-selected rule wording",
                capturedAt: generated)
        }
        let includeInjuries = requested.isEmpty || requested.contains("availability")
        let includeStarters = requested.isEmpty || requested.contains("starters")
        let injuries = try injuryEvidence(result.injuryContext, additionalObservations: result.roleStatusEvidence.array,
            includeObservations: includeInjuries, generated: generated)
        evidence += injuries
        evidence += try roleLimitations(result, generated: generated)
        let designations = includeStarters ? try designationEvidence(result.designations, sources: result.context["sources"], generated: generated) : []
        evidence += designations
        // Preserve all supplied gaps while grouping them within chat's evidence limits.
        var gapGroups: [String] = []
        for text in result.missingContext {
            guard text.count <= 2000 else { throw AskPLONNError.invalidEvidence }
            if let last = gapGroups.last, last.count + text.count + 1 <= 2000 {
                gapGroups[gapGroups.count - 1] = last + "\n" + text
            } else { gapGroups.append(text) }
        }
        for (index, text) in gapGroups.enumerated() {
            evidence.append(gap(text, id: "narrative-gap|\(index)"))
        }
        var supplied = Set(selected.map { requestDimension($0.dimension) })
        if includeInjuries && injuries.contains(where: { $0.availability == .available }) { supplied.insert("availability") }
        if includeStarters && designations.contains(where: { $0.availability == .available }) { supplied.insert("starters") }
        let missing = requested.subtracting(supplied)
        if !missing.isEmpty {
            evidence.append(gap("The recorded report does not verify every requested dimension (\(missing.sorted().joined(separator: ", "))). No missing expectation has been inferred.", id: "narrative-requested-gap"))
        }
        if selected.isEmpty && requested.isEmpty {
            evidence.append(gap("No verified claims are available in the richer recorded report.", id: "narrative-empty"))
        }
        _ = try AskPLONNCore.boundedRequest(.init(question: question, intent: .recentForm, evidence: evidence))
        return evidence
    }
    private static func requestDimension(_ dimension: String) -> String {
        ["home_availability", "away_availability"].contains(dimension) ? "availability" : dimension
    }
    /// Both inputs are validated projections. Role premises can sit outside the
    /// general six-row summary; merge by claim identity without truncating them.
    static func injuryEvidence(_ injury: RichContextJSON, additionalObservations: [RichContextJSON] = [],
                               includeObservations: Bool = true,
                               generated: Date) throws -> [AskPLONNEvidence] {
        guard injury != .null else {
            return [gap("Verified current injury context is not supplied; missing status does not mean available.", id: "narrative-injury-missing")]
        }
        let sources = injury["sources"].array
        let teams = Dictionary(uniqueKeysWithValues: injury["teams"].array.map { ($0["id"].string, $0["name"].string) })
        var seen = Set<String>()
        let rows = (additionalObservations + injury["observations"].array).filter { seen.insert($0["claimID"].string).inserted }
        var evidence: [AskPLONNEvidence] = []
        if includeObservations {
            let lines = rows.map { row -> (text: String, providers: [String]) in
                let confirmation = row["eventStatusConfirmed"] == .bool(true) ? "yes" : "no"
                let updated = row["updatedAt"] == .null ? "not supplied" : row["updatedAt"].string
                let estimated = row["estimated"] == .bool(true) ? " This practice report is estimated." : ""
                let team = teams[row["teamID"].string] ?? "Team"
                let text = "\(team) · \(row["playerName"].string): reported status \(row["reportedStatus"].string). Scope: \(row["scope"].string). Confirmed for this game: \(confirmation). Updated: \(updated).\(estimated)"
                return (text, sources.filter { $0["id"] == row["sourceID"] }.map { $0["provider"].string })
            }
            evidence += try metadataEvidence(lines, id: "narrative-injury", title: "Reported player status", generated: generated)
        }
        var limits = [injury["interpretation"].string]
        if rows.isEmpty {
            limits.insert(injury["state"].string == "stale"
                ? "Current player availability is unverified: earlier injury reports are stale and excluded."
                : "No eligible current injury observations are supplied. Missing status does not mean available.", at: 0)
        }
        for coverage in injury["coverage"].array {
            let name = teams[coverage["teamID"].string] ?? "Team"
            let reason = coverage["reason"].string.replacingOccurrences(of: "_", with: " ")
            limits.append("\(name): injury coverage is \(coverage["state"].string) (\(reason)); \(coverage["excludedRows"].integer ?? 0) unsupported source rows excluded.")
        }
        for exclusion in injury["exclusions"].array {
            let name = teams[exclusion["teamID"].string] ?? "Team"
            let excluded = exclusion["excludedCount"].integer ?? 0, omitted = exclusion["promptOmittedCount"].integer ?? 0
            if excluded > 0 {
                let reasons = exclusion["reasons"].array.map { $0.string.replacingOccurrences(of: "_", with: " ") }.joined(separator: ", ")
                limits.append("\(name): \(excluded) injury observations excluded (\(reasons)).")
            }
            if omitted > 0 { limits.append("\(name): \(omitted) eligible injury observations omitted from the general injury summary; this is not a complete player list.") }
        }
        let providers = Array(Set(sources.map { $0["provider"].string })).sorted()
        evidence += try metadataEvidence(limits.filter { !$0.isEmpty }.map { ($0, providers) }, id: "narrative-injury-limit", title: "Player status limitations", generated: generated,
            availability: injury["state"].string == "stale" ? .stale : .unavailable)
        return evidence
    }
    /// Gaps remain limitations, including when fresh sports claims are present.
    /// Counts preserve every excluded/omitted item without exposing diagnostic IDs.
    static func roleLimitations(_ result: MatchupNarrativeV4.Result, generated: Date) throws -> [AskPLONNEvidence] {
        let role = result.injuryRoleContext
        guard role != .null else { return [] }
        var limits = ["Player-role evidence is \(role["state"].string). Missing role or status evidence does not establish health, availability, a replacement, or a matchup effect."]
        for coverage in role["coverage"].array {
            limits.append("\(coverage["teamName"].string): role coverage is \(coverage["state"].string) (\(coverage["reason"].string.replacingOccurrences(of: "_", with: " "))).")
        }
        let gaps = Dictionary(grouping: result.injuryRoleGaps.array, by: { $0["reason"].string })
        for reason in gaps.keys.sorted() {
            limits.append("\(gaps[reason]!.count) player-status observations cannot support a role claim: \(roleGapReason(reason)).")
        }
        let excluded = Dictionary(grouping: role["excluded"].array, by: { $0["teamName"].string })
        for team in excluded.keys.sorted() {
            let reasons = Dictionary(grouping: excluded[team]!, by: { $0["reason"].string })
            for reason in reasons.keys.sorted() {
                limits.append("\(team): \(reasons[reason]!.count) role observations excluded: \(roleGapReason(reason)).")
            }
        }
        if let omitted = role["omissions"]["count"].integer, omitted > 0 {
            limits.append("\(omitted) eligible role observations omitted from the bounded role summary; role coverage is incomplete.")
        }
        return try metadataEvidence(limits.map { ($0, result.providers) }, id: "narrative-role-limit", title: "Player role limitations",
            generated: generated, availability: role["state"].string == "stale" ? .stale : .unavailable)
    }
    private static func roleGapReason(_ reason: String) -> String {
        ["no_eligible_exact_role": "no matching current role evidence",
         "observation_stale": "the player-status report is stale", "source_stale": "the source is stale",
         "role_stale": "the role report is stale", "usage_stale": "the historical usage is stale",
         "role_update_unknown": "the role update time is unknown", "update_unknown": "the player-status update time is unknown",
         "different_event": "the report belongs to a different game", "event_started": "the game has started",
         "identity_unresolved": "the player identity is unresolved", "conflicting_status": "the player-status reports conflict",
         "conflicting_role": "the role reports conflict"][reason] ?? reason.replacingOccurrences(of: "_", with: " ")
    }
    static func designationEvidence(_ designations: RichContextJSON, sources: RichContextJSON,
                                    generated: Date) throws -> [AskPLONNEvidence] {
        var known: [(text: String, providers: [String])] = [], unknown: [(text: String, providers: [String])] = []
        for row in designations.array {
            let state = row["state"].string
            let subject = row["subject"] == .null ? "not reported" : row["subject"]["name"].string
            let reported = row["reportedAt"] == .null ? "not supplied" : row["reportedAt"].string
            let kind = row["kind"].string.replacingOccurrences(of: "_", with: " ")
            let providers = sources.array.filter { row["sourceIDs"].array.contains($0["id"]) }.map { $0["provider"].string }
            let text = "\(row["team"].string): \(kind): \(subject). Designation: \(state). Reported: \(reported). " +
                (state == "probable" ? "Probable is not confirmed." : state == "unknown" ? "An unknown designation does not establish a starter or player availability." : "No matchup impact is inferred.")
            if state == "unknown" { unknown.append((text, providers)) } else { known.append((text, providers)) }
        }
        return try metadataEvidence(known, id: "narrative-designation", title: "Reported starters and lineups", generated: generated) +
            metadataEvidence(unknown, id: "narrative-designation-unknown", title: "Starter limitations", generated: generated, availability: .unavailable)
    }
    /// Pack related observations without truncating statuses, limitations or attribution.
    private static func metadataEvidence(_ rows: [(text: String, providers: [String])], id: String, title: String,
                                         generated: Date, availability: AskPLONNAvailability = .available) throws -> [AskPLONNEvidence] {
        var groups: [(text: String, providers: Set<String>)] = []
        for row in rows {
            let providers = Set(row.providers.filter { !$0.isEmpty })
            guard row.text.count <= 2000, providers.sorted().joined(separator: ", ").count <= 500 else { throw AskPLONNError.invalidEvidence }
            if let last = groups.last, last.text.count + row.text.count + 2 <= 2000,
               last.providers.union(providers).sorted().joined(separator: ", ").count <= 500 {
                groups[groups.count - 1] = (last.text + "\n\n" + row.text, last.providers.union(providers))
            } else { groups.append((row.text, providers)) }
        }
        return groups.enumerated().map { index, group in
            .init(id: "\(id)|\(index)", kind: .recentForm, title: title, text: group.text,
                source: group.providers.isEmpty ? "PLONN verified research" : group.providers.sorted().joined(separator: ", "),
                capturedAt: generated, availability: availability)
        }
    }
    static func narrativeDimensions(_ question: String, sport: String) -> Set<String> {
        let mappings: [String: [(String, [String])]] = [
            "NFL": [("passing", ["passing", "pass", "completions"]), ("rushing", ["rushing", "rush"]), ("play_volume", ["pace", "tempo", "plays", "play volume"]), ("scoring", ["scoring", "points"])],
            "NBA": [("pace", ["pace", "tempo", "possessions"]), ("efficiency", ["efficiency", "efficient"]), ("shooting", ["shooting", "shots", "field goals", "three pointers"]), ("ball_movement", ["assists", "ball movement"]), ("rebounds", ["rebounds", "rebounding"]), ("turnovers", ["turnovers"]), ("scoring", ["scoring", "points"])],
            "MLB": [("hits", ["hits", "hitting"]), ("strikeouts", ["strikeout", "strikeouts", "ks"]), ("run_scoring", ["runs", "scoring"])],
            "NHL": [("shots", ["shots", "shooting"]), ("special_teams", ["power play", "penalty kill", "special teams"]), ("goaltending", ["saves", "goalie", "goaltender", "goaltending"]), ("scoring", ["goals", "scoring"])]]
        var dimensions = Set((mappings[sport] ?? []).filter { mapping in mapping.1.contains { AskPLONNContextResolver.contains(question, $0) } }.map(\.0))
        // Status and starter projections are separate from numerical claim dimensions.
        dimensions.formUnion(AskPLONNContextResolver.statusDimensions(question))
        return dimensions
    }
    private func teamEvidence(_ market: Market, now: Date) async -> AskPLONNEvidence {
        let history = await team(market, now)
        let checked = AnalyticsHistoryEngine.statistics(market: market, observations: history.observations, now: now)
        guard let summary = checked.statistics, let mean = summary.mean, summary.valid > 0,
              let start = summary.startDate, let end = summary.endDate,
              !history.sourceURLs.isEmpty, history.sourceURLs.allSatisfy(AskPLONNCore.validSourceURL) else {
            return Self.gap("Verified recorded team scores are unavailable for \(market.name).", id: "team|\(market.sport)|\(market.name)")
        }
        let unit = market.sport == "MLB" ? "runs" : market.sport == "NHL" ? "goals" : "points"
        let date = DateFormatter(); date.locale = Locale(identifier: "en_US_POSIX"); date.timeZone = TimeZone(secondsFromGMT: 0); date.dateFormat = "yyyy-MM-dd"
        let text = "\(market.name) averaged \(String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), mean)) \(unit) across \(summary.valid) recorded completed games from \(date.string(from: start)) to \(date.string(from: end)). Historical scoring is not a forecast or player projection. Coverage may be incomplete."
        return .init(id: "team|\(market.sport)|\(market.name)", kind: .recentForm, title: "\(market.name) recorded scoring", text: text, source: history.source, capturedAt: now, sampleSize: summary.valid, sourceURLs: Array(history.sourceURLs.prefix(12)))
    }
    /// Accept only the existing validated, exact-event client result. Reapply age
    /// and started-event guards at use time; never promote a stale report to fresh.
    static func reportEvidence(_ response: SharedMatchupReportResponse, event: SharedMatchupEvent, question: String, now: Date) throws -> [AskPLONNEvidence] {
        guard try SharedMatchupReportValidator.normalizedEvent(response.event) == SharedMatchupReportValidator.normalizedEvent(event) else { throw SharedMatchupReportError.invalidResponse }
        guard try SharedMatchupReportValidator.date(event.startISO) > now else { return [gap("This game has started; its pregame outlook is no longer current.")] }
        guard response.status == .ready, let generated = response.generatedAt, let checked = response.sourceCheckedAt,
              let snapshot = response.snapshot, let report = response.report else {
            return [gap(response.status == .stale ? "The saved matchup research is stale and has been excluded from the answer." : "Verified matchup research is not available for this game.", stale: response.status == .stale)]
        }
        let generatedAt = try SharedMatchupReportValidator.date(generated), checkedAt = try SharedMatchupReportValidator.date(checked)
        guard generatedAt <= now, checkedAt <= now, now.timeIntervalSince(generatedAt) < 86400, now.timeIntervalSince(checkedAt) < 86400 else { return [gap("The saved matchup research has expired and has been excluded from the answer.", stale: true)] }
        // The authenticated client validates the original bytes/fingerprint. A typed
        // re-encoding changes decimal lexemes and cannot revalidate that byte contract.
        // Recheck the deterministic rendering and identity without rewriting evidence.
        guard try SharedMatchupReportValidator.normalizedEvent(snapshot.event) == SharedMatchupReportValidator.normalizedEvent(event),
              report == (try SharedMatchupReportValidator.render(selection: report.selection, snapshot: snapshot, rules: response.versions.rules)) else {
            throw SharedMatchupReportError.invalidResponse
        }
        let requested = AskPLONNContextResolver.requestedStats(question, sport: event.sport)
        let terms = requested.map { $0.replacingOccurrences(of: "pitcher_", with: "").replacingOccurrences(of: "batter_", with: "").replacingOccurrences(of: "player_", with: "").replacingOccurrences(of: "_", with: " ") }
        let observations = report.observations.filter { observation in
            terms.isEmpty || terms.contains { term in observation.text.localizedCaseInsensitiveContains(term) || snapshot.facts.first(where: { $0.id == observation.factID })?.metric.replacingOccurrences(of: "_", with: " ").localizedCaseInsensitiveContains(term) == true }
        }
        var result = observations.prefix(12).compactMap { observation -> AskPLONNEvidence? in
            guard let fact = snapshot.facts.first(where: { $0.id == observation.factID }) else { return nil }
            let sources = snapshot.sources.filter { observation.sourceIDs.contains($0.id) }
            return .init(id: "game|\(event.sport)|\(event.eventID)|\(observation.factID)", kind: .recentForm, title: "\(fact.team) · \(fact.metric)", text: observation.text + " Recorded window: \(fact.windowStart) to \(fact.windowEnd). Historical evidence is not a player projection.", source: Array(Set(sources.map(\.provider))).sorted().joined(separator: ", "), capturedAt: checkedAt, sampleSize: fact.sample, sourceURLs: Array(Set(sources.map(\.url)).sorted().prefix(12)))
        }
        if terms.isEmpty {
            for outlook in report.outlook.prefix(4) {
                let sources = snapshot.sources.filter { outlook.sourceIDs.contains($0.id) }
                result.append(.init(id: "outlook|\(event.sport)|\(event.eventID)|\(outlook.id)", kind: .recentForm,
                    title: outlook.label, text: outlook.text + " " + outlook.uncertainty,
                    source: Array(Set(sources.map(\.provider))).sorted().joined(separator: ", "), capturedAt: checkedAt,
                    sourceURLs: Array(Set(sources.map(\.url)).sorted().prefix(12))))
            }
        }
        if !report.missingContext.isEmpty { result.append(gap(report.missingContext.joined(separator: " "), id: "report-missing-context")) }
        if !result.contains(where: { $0.availability == .available }) { result.append(gap("The available matchup report does not verify the requested stat. No high/low expectation has been inferred.")) }
        return result
    }
}
