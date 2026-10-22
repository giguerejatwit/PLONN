import Foundation
import CoreFoundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Public statistics only. This transport has no odds endpoint or auth headers.
actor AnalyticsFeedPublicReader {
    typealias Transport = @Sendable (URL) async throws -> Data
    private struct Cached { let data: Data; let expires: Date }
    private let transport: Transport
    private var cache: [URL: Cached] = [:]
    private var pending: [URL: Task<Data, Error>] = [:]
    private var active = 0
    private var waiting: [CheckedContinuation<Void, Never>] = []
    init(transport: @escaping Transport = { try await AnalyticsFeedPublicReader.publicRead($0) }) { self.transport = transport }
    private func acquire() async {
        if active < 2 { active += 1; return }
        await withCheckedContinuation { waiting.append($0) }
    }
    private func release() {
        if !waiting.isEmpty { waiting.removeFirst().resume() }
        else { active -= 1 }
    }
    func read(_ url: URL, useCache: Bool = true) async throws -> Data {
        try Task.checkCancellation()
        guard PlayerHistoryService.allowedURL(url) else { throw PlayerHistoryError.invalidURL }
        cache = cache.filter { $0.value.expires > Date() }
        if useCache, let hit = cache[url] { return hit.data }
        if let task = pending[url] { let data = try await task.value; try Task.checkCancellation(); return data }
        guard pending.count < 32 else { throw PlayerHistoryError.capacity }
        let task = Task { try await self.perform(url) }
        pending[url] = task
        do {
            let data = try await task.value
            pending.removeValue(forKey: url)
            if cache.count >= 96, let oldest = cache.min(by: { $0.value.expires < $1.value.expires })?.key { cache.removeValue(forKey: oldest) }
            cache[url] = Cached(data: data, expires: Date().addingTimeInterval(900))
            try Task.checkCancellation()
            return data
        } catch { pending.removeValue(forKey: url); throw error }
    }
    private func perform(_ url: URL) async throws -> Data {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        let data = try await transport(url)
        guard data.count <= 4 * 1024 * 1024 else { throw PlayerHistoryError.oversizedResponse }
        return data
    }
    static func publicRead(_ url: URL) async throws -> Data {
        guard PlayerHistoryService.allowedURL(url) else { throw PlayerHistoryError.invalidURL }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil; configuration.urlCache = nil
        let session = URLSession(configuration: configuration, delegate: AnalyticsFeedRedirects(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url); request.timeoutInterval = 20
        #if canImport(FoundationNetworking)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200, let final = http.url, PlayerHistoryService.allowedURL(final) else { throw URLError(.badServerResponse) }
        guard data.count <= 4 * 1024 * 1024 else { throw PlayerHistoryError.oversizedResponse }
        return data
        #else
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200, let final = http.url, PlayerHistoryService.allowedURL(final), response.expectedContentLength <= 4 * 1024 * 1024 else { throw URLError(.badServerResponse) }
        var data = Data()
        for try await byte in bytes {
            guard data.count < 4 * 1024 * 1024 else { throw PlayerHistoryError.oversizedResponse }
            data.append(byte)
        }
        return data
        #endif
    }
}
private final class AnalyticsFeedRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(request.url.map(PlayerHistoryService.allowedURL) == true ? request : nil)
    }
}

/// A small neutral feed from source-returned subjects, never a popularity ranking.
actor AnalyticsFeedService {
    static let shared = AnalyticsFeedService()
    private let reader: AnalyticsFeedPublicReader
    private let teamHistory: AnalyticsHistoryService
    private let playerHistory: PlayerHistoryService
    init(reader: AnalyticsFeedPublicReader = AnalyticsFeedPublicReader()) {
        self.reader = reader
        self.teamHistory = AnalyticsHistoryService(reader: { try await reader.read($0) })
        self.playerHistory = PlayerHistoryService(transport: { try await reader.read($0) })
    }
    private func json(_ url: URL, statusRead: Bool = false) async throws -> [String: Any] {
        let data = try await reader.read(url, useCache: !statusRead)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw PlayerHistoryError.invalidPayload }
        return object
    }
    func load(snapshots: [Market], now: Date = Date(), recordedRead: RecordedReader? = nil, sport selectedSport: String? = nil, publicStatisticsFallback: Bool = false) async -> AnalyticsFeedBuildResult {
        #if DEBUG && targetEnvironment(simulator)
        if ProcessInfo.processInfo.environment["PLONN_ANALYTICS_MARKETS_FIXTURE"] != nil {
            return AnalyticsFeedCore.build(snapshots: snapshots.filter { selectedSport == nil || $0.sport == selectedSport }, verifiedSubjects: fixtureSubjects(now: now).filter { selectedSport == nil || $0.sport == selectedSport }, now: now)
        }
        #endif
        if let recordedRead {
            if publicStatisticsFallback {
                async let recorded = loadRecorded(read: recordedRead, now: now, sport: selectedSport, includeStatistics: true)
                async let statistics = load(snapshots: [], now: now, sport: selectedSport, publicStatisticsFallback: true)
                return await Self.mergeBrowsing(statistics: statistics, recorded: recorded)
            }
            return await loadRecorded(read: recordedRead, now: now, sport: selectedSport)
        }
        var subjects: [AnalyticsFeedVerifiedSubject] = [], warnings: [String] = []
        var rosterTeams: [AnalyticsFeedVerifiedSubject] = []
        let sports: [String] = ["NFL", "NBA", "MLB", "NHL"].filter { selectedSport == nil || $0 == selectedSport }
        let date = DateFormatter(); date.locale = Locale(identifier: "en_US_POSIX"); date.timeZone = TimeZone(identifier: "America/New_York")
        // Discover the entire current slate independently of odds coverage.
        // Empty days use the verified league directory; roster reads are bounded below.
        for sport in sports {
            guard !Task.isCancelled else { break }
            do {
                var teams: [AnalyticsFeedVerifiedSubject]
                if sport == "MLB" {
                    date.dateFormat = "yyyy-MM-dd"
                    let url = URL(string: "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=\(date.string(from: now))")!
                    teams = try Self.mlbScheduleSubjects(payload: await json(url), sourceURL: url.absoluteString, now: now, includeInactive: true)
                    if teams.isEmpty {
                        let directory = URL(string: "https://statsapi.mlb.com/api/v1/teams?sportId=1")!
                        teams = try Self.mlbDirectorySubjects(payload: await json(directory), sourceURL: directory.absoluteString, now: now)
                    }
                } else {
                    date.dateFormat = "yyyyMMdd"
                    let league = PlayerHistoryService.leaguePath(sport)
                    let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(league)/scoreboard?dates=\(date.string(from: now))&limit=100")!
                    teams = try Self.scoreboardSubjects(payload: await json(url), sport: sport, sourceURL: url.absoluteString, now: now, includeInactive: publicStatisticsFallback)
                    if teams.isEmpty {
                        let directory = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(league)/teams?limit=1000")!
                        teams = try Self.directorySubjects(payload: await json(directory), sport: sport, sourceURL: directory.absoluteString, now: now)
                    }
                }
                subjects += publicStatisticsFallback ? Self.currentTeamContexts(teams, now: now) : Array(teams.prefix(2))
                if publicStatisticsFallback { rosterTeams += Self.currentTeamContexts(teams, now: now) }
                else if let team = teams.first { rosterTeams.append(team) }
            } catch is CancellationError { break }
            catch { warnings.append("\(sport): public team statistics could not be loaded. Please try again.") }
        }
        // Two roster transports at a time; discovering more teams never loads their histories.
        var iterator = rosterTeams.makeIterator()
        await withTaskGroup(of: (AnalyticsFeedVerifiedSubject, [AnalyticsFeedVerifiedSubject]?).self) { group in
            for _ in 0..<2 {
                if let team = iterator.next() { group.addTask { (team, try? await self.roster(team: team, now: now, browsing: publicStatisticsFallback)) } }
            }
            while let (team, players) = await group.next() {
                if let players { subjects += players }
                else { warnings.append("\(team.sport): player statistics could not be loaded for \(team.name).") }
                if !Task.isCancelled, let next = iterator.next() { group.addTask { (next, try? await self.roster(team: next, now: now, browsing: publicStatisticsFallback)) } }
            }
        }
        // Without an authenticated reader, only public identities enter the feed.
        // Retained selections never grant a guest access to account-scoped odds.
        let built = AnalyticsFeedCore.build(snapshots: [], verifiedSubjects: subjects, now: now, limit: publicStatisticsFallback ? AnalyticsFeedCore.browsingCapacity : 24)
        let candidates = sports.flatMap { sport in built.candidates.filter { $0.market.sport == sport } }
        return AnalyticsFeedBuildResult(candidates: candidates, warnings: warnings + built.warnings)
    }
    private func roster(team: AnalyticsFeedVerifiedSubject, now: Date, browsing: Bool) async throws -> [AnalyticsFeedVerifiedSubject] {
        let url = URL(string: team.sport == "MLB" ? "https://statsapi.mlb.com/api/v1/teams/\(team.providerID)/roster" : "https://site.api.espn.com/apis/site/v2/sports/\(PlayerHistoryService.leaguePath(team.sport))/teams/\(team.providerID)/roster")!
        let payload = try await json(url)
        let players = try team.sport == "MLB" ? Self.mlbRosterSubjects(payload: payload, team: team, sourceURL: url.absoluteString, now: now) : Self.rosterSubjects(payload: payload, sport: team.sport, teamName: team.name, teamID: team.providerID, event: team.event, sourceURL: url.absoluteString, now: now, limit: browsing ? 200 : 2)
        return browsing ? players.map(Self.browsingSubject) : Array(players.prefix(2))
    }
    /// One research context per team: live first, then next scheduled game, then latest final.
    /// This preserves both teams in doubleheaders without mixing different games' prices.
    static func currentTeamContexts(_ teams: [AnalyticsFeedVerifiedSubject], now: Date) -> [AnalyticsFeedVerifiedSubject] {
        func rank(_ team: AnalyticsFeedVerifiedSubject) -> Int {
            guard let event = team.event else { return 3 }
            if event.lifecycle == .inProgress { return 0 }
            return event.lifecycle == .scheduled && event.start > now ? 1 : 2
        }
        var seen = Set<String>()
        return teams.sorted {
            if rank($0) != rank($1) { return rank($0) < rank($1) }
            if let a = $0.event, let b = $1.event, a.start != b.start { return rank($0) == 2 ? a.start > b.start : a.start < b.start }
            return $0.name < $1.name
        }.filter { seen.insert($0.sport + "|" + $0.providerID).inserted }
    }
    static func mergeBrowsing(statistics: AnalyticsFeedBuildResult, recorded: AnalyticsFeedBuildResult) -> AnalyticsFeedBuildResult {
        var byID = Dictionary(statistics.candidates.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        func identity(_ proof: AnalyticsFeedVerifiedSubject) -> String { proof.sport + "|" + proof.subject.rawValue + "|" + proof.providerID }
        let contexts = Dictionary(statistics.candidates.compactMap { $0.verifiedSubject }.map { (identity($0), $0) }, uniquingKeysWith: { first, _ in first })
        for candidate in recorded.candidates {
            guard let proof = candidate.verifiedSubject else { continue }
            if let current = contexts[identity(proof)] {
                guard proof.teamID == current.teamID, AnalyticsOddsMatcher.normalized(proof.name) == AnalyticsOddsMatcher.normalized(current.name) else { continue }
                if let event = current.event {
                    guard let recordedEvent = proof.event,
                          recordedEvent.sport == event.sport, recordedEvent.providerID == event.providerID,
                          recordedEvent.homeID == event.homeID, recordedEvent.awayID == event.awayID,
                          recordedEvent.start == event.start else { continue }
                }
            }
            byID[candidate.id] = candidate
        }
        return AnalyticsFeedBuildResult(candidates: byID.values.sorted { $0.id < $1.id }, warnings: Array(Set(statistics.warnings + recorded.warnings)).sorted())
    }
    /// One authenticated, read-only cache response; the Store adapter cannot request paid odds.
    struct RecordedSnapshot: Sendable {
        let markets: [Market]
        let events: [RecordedEventMetadata]
        let cacheOnly: Bool
        let fetchedAt: String?
        let servedAt: String?
        var recordedArchive: Bool = false
    }
    typealias RecordedReader = @Sendable (String, NBARecordedOddsPhase?, String?) async throws -> RecordedSnapshot

    /// Resolve the selected research subject against an exact existing game cache.
    /// This never requests additional provider coverage or changes the public identity.
    static func recordedSelection(_ candidate: AnalyticsFeedCandidate, read: RecordedReader, now: Date = Date()) async throws -> AnalyticsFeedCandidate? {
        guard let proof = candidate.verifiedSubject, let event = proof.event, event.start > now else { return nil }
        let phase = NBARecordedOddsRouting.phase(sport: proof.sport, seasonTypes: [event.seasonType])
        guard phase != .unknown else { return nil }
        let slate = try validateRecorded(try await read(proof.sport, phase, nil), sport: proof.sport, now: now)
        try Task.checkCancellation()
        guard let selected = RecordedEventSelection.select(slate.events, sport: proof.sport, phase: phase,
            home: event.home, away: event.away, start: event.start, now: now) else { return nil }
        let detail: RecordedSnapshot
        if candidate.subject == .player && !slate.markets.contains(where: { $0.event == selected.id && $0.kind == "prop" }) {
            detail = try validateRecorded(try await read(proof.sport, phase, selected.id), sport: proof.sport, now: now)
        } else { detail = slate }
        try Task.checkCancellation()
        guard let verified = RecordedEventSelection.select(detail.events, sport: proof.sport, phase: phase,
            home: event.home, away: event.away, start: event.start, now: now), verified.id == selected.id else { return nil }
        let built = AnalyticsFeedCore.build(snapshots: detail.markets, verifiedSubjects: [proof], now: now,
            limit: AnalyticsFeedCore.browsingCapacity, recordedSnapshot: true)
        return built.candidates.first { $0.id == candidate.id && $0.verifiedSubject == proof }
    }

    /// Read the cache, verify relevant schedule dates, then read at most one exact event's props per sport.
    /// The first matching event is deterministic; no odds, outcome, or forecast ranking.
    func loadRecorded(read: @escaping RecordedReader, now: Date = Date(), sport selectedSport: String? = nil, includeStatistics: Bool = false) async -> AnalyticsFeedBuildResult {
        guard now.timeIntervalSince1970.isFinite else { return AnalyticsFeedBuildResult(candidates: [], warnings: ["The feed cutoff is invalid."]) }
        var subjects: [AnalyticsFeedVerifiedSubject] = [], snapshots: [Market] = [], warnings: [String] = []
        var archiveQuoteIDs = Set<String>()
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(identifier: "America/New_York")!
        let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = calendar.timeZone
        let sports: [String] = ["NFL", "NBA", "NHL", "MLB"].filter { selectedSport == nil || $0 == selectedSport }
        let today = calendar.startOfDay(for: now)
        let windowEnd = calendar.date(byAdding: .day, value: 5, to: today)!
        for sport in sports {
            guard !Task.isCancelled else { break }
            do {
                var teams: [AnalyticsFeedVerifiedSubject] = []
                var slates: [String: RecordedSnapshot] = [:]
                let phases: [NBARecordedOddsPhase?] = sport == "NBA" ? [.regular, .preseason] : [nil]
                for phase in phases {
                    // A missing phase is not evidence about another phase. Every read is cache-only.
                    if let slate = try? Self.validateRecorded(try await read(sport, phase, nil), sport: sport, now: now) { slates[phase?.rawValue ?? sport] = slate }
                }
                // Only dates represented in the cache can produce a matching card.
                // Still verify every eligible cached event against the public schedule.
                let statusDays = Array(Set(slates.values.flatMap { slate in
                    // Match RecordedStartPolicy across Eastern midnight too: NHL
                    // puck-drop estimates may differ by ten minutes, others by one second.
                    let tolerance: TimeInterval = sport == "NHL" ? 600 : 1
                    return slate.events.compactMap { sportsDate($0.commence) }.flatMap { start in
                        [start.addingTimeInterval(-tolerance), start, start.addingTimeInterval(tolerance)]
                    }.filter {
                        ($0 >= today && $0 < windowEnd) || (slate.recordedArchive && $0 < now)
                    }.map { calendar.startOfDay(for: $0) }
                })).sorted()
                guard statusDays.count <= 32 else { throw PlayerHistoryError.invalidPayload }
                guard !statusDays.isEmpty else {
                    warnings.append("\(sport): no unique scheduled event matches the recorded cache.")
                    continue
                }
                if sport == "MLB" {
                    formatter.dateFormat = "yyyy-MM-dd"
                    let url = URL(string: "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=\(formatter.string(from: statusDays[0]))&endDate=\(formatter.string(from: statusDays.last!))")!
                    teams = try Self.mlbScheduleSubjects(payload: await json(url, statusRead: true), sourceURL: url.absoluteString, now: now)
                } else {
                    formatter.dateFormat = "yyyyMMdd"
                    // ESPN's scoreboard rejects multi-date ranges for these leagues.
                    for day in statusDays {
                        let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(PlayerHistoryService.leaguePath(sport))/scoreboard?dates=\(formatter.string(from: day))&limit=100")!
                        teams += try Self.scoreboardSubjects(payload: await json(url, statusRead: true), sport: sport, sourceURL: url.absoluteString, now: now, recordedWindow: true)
                    }
                }
                let uniqueEvents = Dictionary(grouping: teams.compactMap(\.event), by: \.providerID)
                    .values.compactMap { events -> AnalyticsEvent? in
                        guard let first = events.first, events.allSatisfy({ $0 == first }) else { return nil }; return first
                    }.sorted { $0.start == $1.start ? $0.providerID < $1.providerID : $0.start < $1.start }
                var selected: [(AnalyticsEvent, RecordedEventMetadata, RecordedSnapshot, NBARecordedOddsPhase?)] = []
                for event in uniqueEvents {
                    let phase = NBARecordedOddsRouting.phase(sport: sport, seasonTypes: [event.seasonType])
                    if phase == .unknown { continue }
                    let key = phase?.rawValue ?? sport
                    if slates[key] == nil { slates[key] = try Self.validateRecorded(try await read(sport, phase, nil), sport: sport, now: now) }
                    guard let slate = slates[key], let metadata = RecordedEventSelection.select(slate.events, sport: sport, phase: phase, home: event.home, away: event.away, start: event.start, now: now, includeStarted: slate.recordedArchive && event.allowsRecorded(now: now, archive: true)) else { continue }
                    selected.append((event, metadata, slate, phase))
                    if !includeStatistics { break }
                }
                guard !selected.isEmpty else {
                    warnings.append("\(sport): no unique scheduled event matches the recorded cache.")
                    continue
                }
                var seenTeams = Set<String>()
                for (event, metadata, slate, phase) in selected {
                // Keep the complete selected event scope so contradictory rows are quarantined.
                let eventMarkets = slate.markets.filter { $0.event == metadata.id }
                snapshots += eventMarkets
                if slate.recordedArchive { archiveQuoteIDs.formUnion(eventMarkets.map(\.id)) }
                let selectedTeams = Dictionary(grouping: teams.filter { $0.event == event }, by: \.providerID).values.compactMap { $0.first }.sorted { $0.providerID < $1.providerID }.filter { seenTeams.insert($0.providerID).inserted }
                guard !selectedTeams.isEmpty else { continue }
                subjects += selectedTeams
                var props: [Market] = []
                do {
                    let detail = try Self.validateRecorded(try await read(sport, phase, metadata.id), sport: sport, now: now)
                    let returned = detail.markets.filter { $0.kind == "prop" }
                    guard returned.allSatisfy({ $0.event == metadata.id }) else { throw PlayerHistoryError.invalidPayload }
                    if !returned.isEmpty {
                        // Prices still require an exact event match; roster browsing does not.
                        guard let verified = RecordedEventSelection.select(detail.events, sport: sport, phase: phase, home: event.home, away: event.away, start: event.start, now: now, includeStarted: detail.recordedArchive && event.allowsRecorded(now: now, archive: true)), verified.id == metadata.id, verified.commence == metadata.commence else { throw PlayerHistoryError.invalidPayload }
                        props = returned
                        snapshots += props
                        if detail.recordedArchive { archiveQuoteIDs.formUnion(props.map(\.id)) }
                    }
                } catch { warnings.append("\(sport): recorded player props are unavailable; verified statistics remain.") }
                if includeStatistics || !props.isEmpty {
                    for team in selectedTeams {
                        let url = URL(string: sport == "MLB" ? "https://statsapi.mlb.com/api/v1/teams/\(team.providerID)/roster" : "https://site.api.espn.com/apis/site/v2/sports/\(PlayerHistoryService.leaguePath(sport))/teams/\(team.providerID)/roster")!
                        do {
                            let payload = try await json(url)
                            let roster = try sport == "MLB"
                                ? Self.mlbRosterSubjects(payload: payload, team: team, sourceURL: url.absoluteString, now: now)
                                : Self.rosterSubjects(payload: payload, sport: sport, teamName: team.name, teamID: team.providerID, event: event, sourceURL: url.absoluteString, now: now, limit: 200)
                            subjects += roster.compactMap { player in
                                var keys = Set(props.filter { AnalyticsOddsMatcher.normalized($0.name) == AnalyticsOddsMatcher.normalized(player.name) && AnalyticsFeedCore.supports(sport: sport, subject: .player, marketKey: $0.market) }.map(\.market))
                                if includeStatistics { keys.formUnion(Self.browsingSubject(player).marketKeys) }
                                guard !keys.isEmpty else { return nil }
                                return AnalyticsFeedVerifiedSubject(sport: player.sport, subject: .player, name: player.name, providerID: player.providerID, teamName: player.teamName, teamID: player.teamID, imageURL: player.imageURL, teamLogoURL: player.teamLogoURL, marketKeys: keys.sorted(), event: player.event, sourceURL: player.sourceURL, verifiedAt: player.verifiedAt)
                            }
                        } catch { warnings.append("\(sport): player membership could not be verified for \(team.name).") }
                    }
                }
                }
            } catch is CancellationError { break }
            catch { warnings.append("\(sport): recorded cache or public schedule is unavailable.") }
        }
        let built = AnalyticsFeedCore.build(snapshots: snapshots, verifiedSubjects: subjects, now: now, limit: 30_000, recordedSnapshot: true)
        if includeStatistics {
            // Browsing is independent of quote coverage. Card-level matching still gates every price.
            let candidates = built.candidates.filter { $0.verifiedSubject != nil }.map { original in
                var candidate = original
                candidate.recordedArchive = !candidate.quotes.isEmpty && candidate.quotes.allSatisfy { archiveQuoteIDs.contains($0.id) }
                return candidate
            }
            return AnalyticsFeedBuildResult(candidates: candidates, warnings: Array(Set(warnings + built.warnings)).sorted())
        }
        // Select only verified exact quote matches before applying the bounded history budget.
        // Four leagues and both subject types get independent slots; quote-less alphabetical
        // seeds cannot consume those slots. One stat per subject keeps the feed varied.
        var selected: [AnalyticsFeedCandidate] = []
        for sport in sports {
            for subject in [AnalyticsFeedSubject.team, .player] {
                var names = Set<String>()
                let candidates = built.candidates.filter { $0.market.sport == sport && $0.subject == subject }.map { original in
                    var candidate = original
                    candidate.recordedArchive = !candidate.quotes.isEmpty && candidate.quotes.allSatisfy { archiveQuoteIDs.contains($0.id) }
                    return candidate
                }.filter { candidate in
                    guard let proof = candidate.verifiedSubject, let event = proof.event else { return false }
                    let target = AnalyticsStatsTarget(sport: sport, kind: subject == .player ? "prop" : "team", name: proof.name, marketKey: candidate.market.market, teamName: subject == .team ? proof.name : proof.teamName ?? "", teamID: subject == .team ? proof.providerID : proof.teamID, playerID: subject == .player ? proof.providerID : nil, rosterVerifiedAt: proof.verifiedAt)
                    return !AnalyticsOddsMatcher.matchCached(target: target, event: event, markets: candidate.quotes, now: now, recordedSnapshot: true, recordedArchive: candidate.recordedArchive).markets.isEmpty
                }
                selected += candidates.filter { names.insert(AnalyticsOddsMatcher.normalized($0.market.name)).inserted }.prefix(subject == .team ? 2 : 4)
            }
        }
        return AnalyticsFeedBuildResult(candidates: selected, warnings: Array(Set(warnings + built.warnings)).sorted())
    }
    /// Public status checks only; no odds, roster or history refetch and no proof TTL extension.
    func refreshStatuses(candidates: [AnalyticsFeedCandidate], now: Date = Date()) async -> AnalyticsFeedBuildResult {
        let events = candidates.filter(\.recordedSnapshot).compactMap { $0.verifiedSubject?.event }
        let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = TimeZone(identifier: "America/New_York"); formatter.dateFormat = "yyyyMMdd"
        let groups = Dictionary(grouping: events, by: { $0.sport + "|" + formatter.string(from: $0.start) })
        var verified: [String: AnalyticsEvent] = [:], warnings: [String] = []
        for key in groups.keys.sorted() {
            guard !Task.isCancelled, let group = groups[key], let event = group.first else { continue }
            do {
                let url: URL
                let proofs: [AnalyticsFeedVerifiedSubject]
                if event.sport == "MLB" {
                    let ids = Set(group.map(\.providerID)).sorted().joined(separator: ",")
                    guard group.allSatisfy({ !$0.providerID.isEmpty && $0.providerID.allSatisfy(\.isNumber) }) else { continue }
                    url = URL(string: "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePks=" + ids)!
                    proofs = try Self.mlbScheduleSubjects(payload: await json(url, statusRead: true), sourceURL: url.absoluteString, now: now, includeInactive: true)
                } else {
                    url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(PlayerHistoryService.leaguePath(event.sport))/scoreboard?dates=\(formatter.string(from: event.start))&limit=100")!
                    proofs = try Self.scoreboardSubjects(payload: await json(url, statusRead: true), sport: event.sport, sourceURL: url.absoluteString, now: now, recordedWindow: true, includeInactive: true)
                }
                for proof in proofs {
                    guard let updated = proof.event else { continue }
                    if group.contains(where: { $0.providerID == updated.providerID && $0.sport == updated.sport && $0.homeID == updated.homeID && $0.awayID == updated.awayID && $0.home == updated.home && $0.away == updated.away && $0.start == updated.start && $0.seasonType == updated.seasonType }) {
                        verified[updated.sport + "|" + updated.providerID] = updated
                    }
                }
            } catch { warnings.append("\(event.sport): event status could not be verified; started-game prices are hidden.") }
        }
        let updated = candidates.map { candidate -> AnalyticsFeedCandidate in
            guard candidate.recordedSnapshot, var proof = candidate.verifiedSubject, var old = proof.event else { return candidate }
            if let fresh = verified[old.sport + "|" + old.providerID] { old = fresh }
            else { old.lifecycle = .unknown; old.statusVerifiedAt = nil }
            if [.postponed, .suspended, .unknown].contains(old.lifecycle) { warnings.append("\(old.sport): \(old.lifecycle.label). Recorded prices remain archived.") }
            proof.event = old
            return AnalyticsFeedCandidate(id: candidate.id, subject: candidate.subject, market: candidate.market, quotes: candidate.quotes, verifiedSubject: proof, recordedSnapshot: true, recordedArchive: candidate.recordedArchive)
        }
        return AnalyticsFeedBuildResult(candidates: updated, warnings: Array(Set(warnings)).sorted())
    }
    static func validateRecorded(_ response: RecordedSnapshot, sport: String, now: Date) throws -> RecordedSnapshot {
        guard response.cacheOnly, response.markets.count <= 10_000, response.events.count <= 512,
              response.markets.allSatisfy({ $0.sport == sport && $0.mode == "live" }),
              let fetched = response.fetchedAt.flatMap(sportsDate), fetched <= now,
              let served = response.servedAt.flatMap(sportsDate), served >= fetched, served <= now.addingTimeInterval(60) else { throw PlayerHistoryError.invalidPayload }
        return response
    }
    private static func mlbIdentifier(_ value: Any?) -> String? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, number.doubleValue > 0,
              number.doubleValue <= 9_007_199_254_740_991, number.doubleValue.rounded() == number.doubleValue else { return nil }
        return number.stringValue
    }
    static func mlbScheduleSubjects(payload: [String: Any], sourceURL: String, now: Date, includeInactive: Bool = false) throws -> [AnalyticsFeedVerifiedSubject] {
        guard let dates = payload["dates"] as? [[String: Any]], dates.count <= 32 else { throw PlayerHistoryError.invalidPayload }
        let games = dates.flatMap { $0["games"] as? [[String: Any]] ?? [] }
        guard games.count <= 100 else { throw PlayerHistoryError.invalidPayload }
        let groups = Dictionary(grouping: games, by: { mlbIdentifier($0["gamePk"]) ?? "" })
        return groups.keys.sorted().flatMap { id -> [AnalyticsFeedVerifiedSubject] in
            guard !id.isEmpty, let copies = groups[id], let game = copies.first, copies.allSatisfy({ NSDictionary(dictionary: $0).isEqual(to: game) }),
                  let start = (game["gameDate"] as? String).flatMap(sportsDate),
                  let status = game["status"] as? [String: Any],
                  let teams = game["teams"] as? [String: Any], let home = (teams["home"] as? [String: Any])?["team"] as? [String: Any], let away = (teams["away"] as? [String: Any])?["team"] as? [String: Any],
                  let hn = home["name"] as? String, let an = away["name"] as? String, let hi = mlbIdentifier(home["id"]), let ai = mlbIdentifier(away["id"]), hi != ai, !hn.isEmpty, !an.isEmpty else { return [] }
            let event = AnalyticsEvent(sport: "MLB", providerID: id, home: hn, away: an, start: start, homeID: hi, awayID: ai, lifecycle: AnalyticsEventLifecycle.mlb(status), statusVerifiedAt: now)
            guard includeInactive || event.allowsRecorded(now: now, archive: true) else { return [] }
            return [(hn, hi), (an, ai)].map { name, teamID in AnalyticsFeedVerifiedSubject(sport: "MLB", subject: .team, name: name, providerID: teamID, marketKeys: ["team_totals"], event: event, sourceURL: sourceURL, verifiedAt: now) }
        }
    }
    static func mlbDirectorySubjects(payload: [String: Any], sourceURL: String, now: Date) throws -> [AnalyticsFeedVerifiedSubject] {
        guard let teams = payload["teams"] as? [[String: Any]], teams.count <= 100 else { throw PlayerHistoryError.invalidPayload }
        return teams.compactMap { team in
            guard let id = mlbIdentifier(team["id"]), let name = team["name"] as? String, !name.isEmpty,
                  let sport = team["sport"] as? [String: Any], mlbIdentifier(sport["id"]) == "1" else { return nil }
            return AnalyticsFeedVerifiedSubject(sport: "MLB", subject: .team, name: name, providerID: id,
                marketKeys: ["team_totals"], sourceURL: sourceURL, verifiedAt: now)
        }.sorted { $0.name < $1.name }
    }
    static func mlbRosterSubjects(payload: [String: Any], team: AnalyticsFeedVerifiedSubject, sourceURL: String, now: Date) throws -> [AnalyticsFeedVerifiedSubject] {
        guard let rows = payload["roster"] as? [[String: Any]], rows.count <= 200,
              (payload["teamId"] == nil || mlbIdentifier(payload["teamId"]) == team.providerID) else { throw PlayerHistoryError.invalidPayload }
        let athletes = rows.compactMap { row -> [String: Any]? in
            guard let person = row["person"] as? [String: Any], let id = mlbIdentifier(person["id"]), let name = person["fullName"] as? String else { return nil }
            return ["id": id, "fullName": name, "position": row["position"] as? [String: Any] ?? [:]]
        }
        return try rosterSubjects(payload: ["team": ["id": team.providerID, "displayName": team.name], "athletes": athletes], sport: "MLB", teamName: team.name, teamID: team.providerID, event: team.event, sourceURL: sourceURL, now: now, limit: 200)
    }
    func history(candidate: AnalyticsFeedCandidate, limit: Int = 10, now: Date = Date()) async -> AnalyticsHistoryResult {
        #if DEBUG && targetEnvironment(simulator)
        if ProcessInfo.processInfo.environment["PLONN_ANALYTICS_MARKETS_FIXTURE"] != nil {
            if let raw = ProcessInfo.processInfo.environment["PLONN_ANALYTICS_HISTORY_DELAY_MS"], let delay = Int(raw), delay > 0, delay <= 5000 { try? await Task.sleep(for: .milliseconds(delay)) }
            return fixtureHistory(candidate: candidate, limit: limit, now: now)
        }
        #endif
        if candidate.subject == .team { return await teamHistory.loadStats(market: candidate.market, limit: limit, now: now) }
        var request = candidate.market
        request.market = AnalyticsHistoryEngine.statisticKey(sport: request.sport, marketKey: request.market) ?? request.market
        if let event = candidate.verifiedSubject?.event {
            request.event = event.providerID; request.eventLabel = event.away + " at " + event.home
            request.commence = ISO8601DateFormatter().string(from: event.start)
        }
        let history = await playerHistory.load(market: request, includeOpponent: false, seasonCount: 2, now: now, maximumGames: 2_000)
        if !history.games.isEmpty, !Self.acceptsPlayerHistory(candidate: candidate, history: history) {
            return AnalyticsHistoryResult(summary: nil, rows: [], observations: [], source: history.source, warnings: ["Current roster and historical player identities could not both be verified."], availability: "Recorded history is unavailable for this roster entry until its player identity is verified.", sourceURLs: history.sourceURLs)
        }
        let observations = history.games.map { AnalyticsObservation(id: $0.id, date: $0.date, opponent: $0.opponent, value: $0.value, homeAway: $0.homeAway, season: $0.season) }
        let result = AnalyticsHistoryEngine.statistics(market: candidate.market, observations: observations, limit: limit, now: now)
        return AnalyticsHistoryResult(summary: nil, rows: [], observations: observations, source: history.source, warnings: history.warnings + result.warnings + ["Only recorded appearances are shown; unrecorded DNP and missing-stat coverage are unavailable."], availability: observations.isEmpty ? history.availability : result.availability, sourceURLs: history.sourceURLs, statistics: result.statistics, statsRows: result.statsRows, verifiedPlayerID: history.playerID, candidateTeamID: history.games.first?.playerTeamID)
    }
    static func acceptsPlayerHistory(candidate: AnalyticsFeedCandidate, history: PlayerHistoryResult) -> Bool {
        guard candidate.subject == .player, let id = history.playerID, !id.isEmpty,
              let name = history.playerName, AnalyticsOddsMatcher.normalized(name) == AnalyticsOddsMatcher.normalized(candidate.market.name) else { return false }
        if history.source == "MLB Stats API", candidate.market.sport != "MLB" { return false }
        guard let proof = candidate.verifiedSubject else { return true }
        guard proof.subject == .player, proof.sport == candidate.market.sport, proof.providerID == id,
              AnalyticsOddsMatcher.normalized(proof.name) == AnalyticsOddsMatcher.normalized(name) else { return false }
        if history.source == "MLB Stats API", URL(string: proof.sourceURL)?.host != "statsapi.mlb.com" { return false }
        return true
    }
    private static func logo(_ team: [String: Any]) -> String? {
        let values = [team["logo"] as? String] + ((team["logos"] as? [[String: Any]]) ?? []).map { $0["href"] as? String }
        return values.compactMap { AnalyticsFeedCore.imageURL($0)?.absoluteString }.first
    }
    static func scoreboardSubjects(payload: [String: Any], sport: String, sourceURL: String, now: Date, recordedWindow: Bool = false, includeInactive: Bool = false) throws -> [AnalyticsFeedVerifiedSubject] {
        guard let events = payload["events"] as? [[String: Any]], events.count <= 100 else { throw PlayerHistoryError.invalidPayload }
        var result: [AnalyticsFeedVerifiedSubject] = []
        let groups = Dictionary(grouping: events, by: { $0["id"] as? String ?? "" })
        let conflicts = Set(groups.compactMap { id, rows -> String? in
            guard let first = rows.first else { return nil }
            return rows.allSatisfy { NSDictionary(dictionary: $0).isEqual(to: first) } ? nil : id
        })
        for event in events where !conflicts.contains(event["id"] as? String ?? "") {
            guard let id = event["id"] as? String, !id.isEmpty, let start = (event["date"] as? String).flatMap(sportsDate), (recordedWindow || includeInactive || AnalyticsOddsMatcher.sameDay(start, now)), let competitions = event["competitions"] as? [[String: Any]], competitions.count == 1, let competitors = competitions[0]["competitors"] as? [[String: Any]], competitors.count == 2 else { continue }
            let home = competitors.filter { $0["homeAway"] as? String == "home" }, away = competitors.filter { $0["homeAway"] as? String == "away" }
            guard home.count == 1, away.count == 1, let h = home[0]["team"] as? [String: Any], let a = away[0]["team"] as? [String: Any], let hn = h["displayName"] as? String, let an = a["displayName"] as? String, let hi = h["id"] as? String, let ai = a["id"] as? String, hi != ai, !hn.isEmpty, !an.isEmpty, hi.allSatisfy(\.isNumber), ai.allSatisfy(\.isNumber), !hi.isEmpty, !ai.isEmpty else { continue }
            let type = ((competitions[0]["status"] as? [String: Any])?["type"] as? [String: Any]) ?? ((event["status"] as? [String: Any])?["type"] as? [String: Any])
            let eventType = (event["status"] as? [String: Any])?["type"] as? [String: Any]
            let competitionType = (competitions[0]["status"] as? [String: Any])?["type"] as? [String: Any]
            var lifecycle = AnalyticsEventLifecycle.espn(type)
            if let eventType, let competitionType, AnalyticsEventLifecycle.espn(eventType) != AnalyticsEventLifecycle.espn(competitionType) { lifecycle = .unknown }
            let verified = AnalyticsEvent(sport: sport, providerID: id, home: hn, away: an, start: start, homeID: hi, awayID: ai, seasonType: NBARecordedOddsRouting.seasonType(event: event), lifecycle: lifecycle, statusVerifiedAt: now)
            let eligible = recordedWindow ? verified.allowsRecorded(now: now, archive: true) : lifecycle == .scheduled && start > now
            let context = eligible || includeInactive ? verified : nil
            result.append(AnalyticsFeedVerifiedSubject(sport: sport, subject: .team, name: hn, providerID: hi, imageURL: logo(h), marketKeys: ["team_totals"], event: context, sourceURL: sourceURL, verifiedAt: now))
            result.append(AnalyticsFeedVerifiedSubject(sport: sport, subject: .team, name: an, providerID: ai, imageURL: logo(a), marketKeys: ["team_totals"], event: context, sourceURL: sourceURL, verifiedAt: now))
        }
        // Conflicting copies are kept for Core's identity quarantine, never chosen silently.
        return result.sorted { $0.name < $1.name }
    }
    static func directorySubjects(payload: [String: Any], sport: String, sourceURL: String, now: Date) throws -> [AnalyticsFeedVerifiedSubject] {
        guard let sports = payload["sports"] as? [[String: Any]], sports.count <= 8 else { throw PlayerHistoryError.invalidPayload }
        let leagues = sports.flatMap { $0["leagues"] as? [[String: Any]] ?? [] }.filter { ($0["abbreviation"] as? String)?.uppercased() == sport || ($0["slug"] as? String)?.lowercased() == sport.lowercased() }
        guard leagues.count == 1, let entries = leagues[0]["teams"] as? [[String: Any]], entries.count <= 100 else { throw PlayerHistoryError.invalidPayload }
        return entries.compactMap { entry in
            guard let team = entry["team"] as? [String: Any], let name = team["displayName"] as? String, !name.isEmpty, let id = team["id"] as? String, !id.isEmpty, id.allSatisfy(\.isNumber) else { return nil }
            return AnalyticsFeedVerifiedSubject(sport: sport, subject: .team, name: name, providerID: id, imageURL: logo(team), marketKeys: ["team_totals"], sourceURL: sourceURL, verifiedAt: now)
        }.sorted { $0.name < $1.name }
    }
    /// Expand only statistics supported by this history reader and the verified playing role.
    /// The roster parser's primary key identifies that role without inventing membership.
    static func browsingSubject(_ player: AnalyticsFeedVerifiedSubject) -> AnalyticsFeedVerifiedSubject {
        let primary = player.marketKeys.first ?? ""
        let supported = OpponentPropHistoryEngine.supportedMarkets[player.sport] ?? []
        let keys = supported.filter { key in
            switch player.sport {
            case "NFL":
                if primary == "player_pass_yds" { return key.hasPrefix("player_pass_") || key.hasPrefix("player_rush_") && key != "player_rush_reception_yds" }
                return !key.hasPrefix("player_pass_")
            case "MLB": return key.hasPrefix(primary.hasPrefix("pitcher_") ? "pitcher_" : "batter_")
            case "NHL": return primary == "player_total_saves" ? key == "player_total_saves" : key != "player_total_saves"
            default: return true
            }
        }.filter { AnalyticsFeedCore.supports(sport: player.sport, subject: .player, marketKey: $0) }
        return AnalyticsFeedVerifiedSubject(sport: player.sport, subject: player.subject, name: player.name, providerID: player.providerID, teamName: player.teamName, teamID: player.teamID, imageURL: player.imageURL, teamLogoURL: player.teamLogoURL, marketKeys: keys.sorted(), event: player.event, sourceURL: player.sourceURL, verifiedAt: player.verifiedAt)
    }
    static func rosterSubjects(payload: [String: Any], sport: String, teamName: String, teamID: String, event: AnalyticsEvent?, sourceURL: String, now: Date, limit: Int = 2) throws -> [AnalyticsFeedVerifiedSubject] {
        guard let team = payload["team"] as? [String: Any], team["id"] as? String == teamID, AnalyticsOddsMatcher.normalized(team["displayName"] as? String ?? "") == AnalyticsOddsMatcher.normalized(teamName), let groups = payload["athletes"] as? [[String: Any]], groups.count <= 200 else { throw PlayerHistoryError.invalidPayload }
        let athletes = groups.flatMap { $0["items"] as? [[String: Any]] ?? [$0] }
        guard athletes.count <= 200 else { throw PlayerHistoryError.invalidPayload }
        let byID = Dictionary(grouping: athletes, by: { $0["id"] as? String ?? "" })
        let byName = Dictionary(grouping: athletes, by: { AnalyticsOddsMatcher.normalized($0["fullName"] as? String ?? $0["displayName"] as? String ?? "") })
        let conflictingIDs = Set(byID.compactMap { id, rows -> String? in
            guard let first = rows.first else { return nil }
            return rows.allSatisfy { NSDictionary(dictionary: $0).isEqual(to: first) } ? nil : id
        })
        let conflictingNames = Set(byName.compactMap { name, rows -> String? in
            Set(rows.compactMap { $0["id"] as? String }).count > 1 ? name : nil
        })
        return athletes.filter {
            !conflictingIDs.contains($0["id"] as? String ?? "") && !conflictingNames.contains(AnalyticsOddsMatcher.normalized($0["fullName"] as? String ?? $0["displayName"] as? String ?? ""))
        }.compactMap { athlete in
            guard let id = athlete["id"] as? String, !id.isEmpty, id.allSatisfy(\.isNumber), let name = athlete["fullName"] as? String ?? athlete["displayName"] as? String, !name.isEmpty else { return nil }
            let position = (athlete["position"] as? [String: Any])?["abbreviation"] as? String ?? ""
            let key: String
            switch sport {
            case "NBA": key = "player_points"
            case "NHL": key = position == "G" ? "player_total_saves" : "player_shots_on_goal"
            case "MLB": key = ["P", "SP", "RP"].contains(position) ? "pitcher_strikeouts" : "batter_hits"
            case "NFL":
                switch position { case "QB": key = "player_pass_yds"; case "RB", "FB": key = "player_rush_yds"; case "WR", "TE": key = "player_reception_yds"; default: return nil }
            default: return nil
            }
            return AnalyticsFeedVerifiedSubject(sport: sport, subject: .player, name: name, providerID: id, teamName: teamName, teamID: teamID, imageURL: AnalyticsFeedCore.imageURL((athlete["headshot"] as? [String: Any])?["href"] as? String)?.absoluteString, teamLogoURL: logo(team), marketKeys: [key], event: event, sourceURL: sourceURL, verifiedAt: now)
        }.sorted { $0.name < $1.name }.prefix(min(max(limit, 0), 200)).map { $0 }
    }
}

#if DEBUG && targetEnvironment(simulator)
private extension AnalyticsFeedService {
    func fixtureSubjects(now: Date) -> [AnalyticsFeedVerifiedSubject] {
        guard let raw = ProcessInfo.processInfo.environment["PLONN_ANALYTICS_SUBJECTS_FIXTURE"], let data = raw.data(using: .utf8), data.count <= 128 * 1024, let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]], rows.count <= 24 else { return [] }
        return rows.compactMap { row in
            guard let sport = row["sport"] as? String, let subject = (row["subject"] as? String).flatMap(AnalyticsFeedSubject.init(rawValue:)), let name = row["name"] as? String, let id = row["providerID"] as? String, let keys = row["marketKeys"] as? [String], let source = row["sourceURL"] as? String else { return nil }
            var event: AnalyticsEvent? = nil
            if let raw = row["event"] as? [String: Any], let es = raw["sport"] as? String, let eid = raw["providerID"] as? String, let home = raw["home"] as? String, let away = raw["away"] as? String, let start = (raw["start"] as? String).flatMap(sportsDate) {
                event = AnalyticsEvent(sport: es, providerID: eid, home: home, away: away, start: start, homeID: raw["homeID"] as? String, awayID: raw["awayID"] as? String)
            }
            return AnalyticsFeedVerifiedSubject(sport: sport, subject: subject, name: name, providerID: id, teamName: row["teamName"] as? String, teamID: row["teamID"] as? String, imageURL: AnalyticsFeedCore.imageURL(row["imageURL"] as? String)?.absoluteString, teamLogoURL: AnalyticsFeedCore.imageURL(row["teamLogoURL"] as? String)?.absoluteString, marketKeys: keys, event: event, sourceURL: source, verifiedAt: now)
        }
    }
    func fixtureHistory(candidate: AnalyticsFeedCandidate, limit: Int, now: Date) -> AnalyticsHistoryResult {
        let raw = ProcessInfo.processInfo.environment["PLONN_ANALYTICS_HISTORY_FIXTURE"] ?? "{}"
        let object = raw.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: [[String: Any]]] } ?? [:]
        let rows = object[candidate.market.name] ?? object[candidate.market.id] ?? candidate.quotes.first.flatMap { object[$0.id] } ?? []
        let observations = rows.prefix(128).compactMap { row -> AnalyticsObservation? in
            guard let id = row["id"] as? String, let date = (row["date"] as? String).flatMap(sportsDate) else { return nil }
            return AnalyticsObservation(id: id, date: date, opponent: row["opponent"] as? String ?? "Fixture opponent", value: row["value"] as? Double, scored: row["scored"] as? Double, allowed: row["allowed"] as? Double, status: AnalyticsObservationStatus(rawValue: row["status"] as? String ?? "completed") ?? .missing, homeAway: ["home", "away"].contains(row["homeAway"] as? String ?? "") ? row["homeAway"] as? String : nil, season: row["season"] as? Int)
        }
        let result = AnalyticsHistoryEngine.statistics(market: candidate.market, observations: observations, limit: limit, now: now)
        return AnalyticsHistoryResult(summary: nil, rows: [], observations: observations, source: "Offline fixture · preview only", warnings: result.warnings, availability: result.availability, statistics: result.statistics, statsRows: result.statsRows, verifiedPlayerID: candidate.subject == .player ? candidate.verifiedSubject?.providerID : nil, candidateTeamID: candidate.verifiedSubject?.teamID)
    }
}
#endif
