import Foundation
import CryptoKit
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct SharedMatchupEvent: Codable, Equatable, Hashable, Sendable {
    let sport: String
    let eventID: String
    let homeTeam: String
    let awayTeam: String
    let startISO: String
    /// ESPN scoreboard timestamps may omit seconds; normalize only that exact UTC form.
    static func scoreboardStart(_ value:String) -> String {
        guard value.range(of:#"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}Z$"#,options:.regularExpression) != nil else {return value}
        return String(value.dropLast()) + ":00Z"
    }
}
enum SharedMatchupReportStatus: String, Codable, Sendable { case ready, missing, stale }
struct SharedMatchupVersions: Codable, Equatable, Sendable {
    let contract, rules, model, modelDigest, prompt, validator: String
    static let current = SharedMatchupVersions(contract: "shared-outlook-v1", rules: "paired-reference-v2", model: "gemma4:26b", modelDigest: "08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68", prompt: "6c2ba97b58fdbcdcf557ff452dd217b6ba56ddea263c6d8e39f08e7d89b17483", validator: "238d994410d8033840a87dc9b8c258e0b4afc40a8010ab5f1cde3c3698112f35")
}
struct SharedMatchupSource: Codable, Equatable, Sendable {
    let id, provider, url, retrievedAt, checkedAt: String
    let sourceAsOf: String?
}
struct SharedMatchupFact: Codable, Equatable, Sendable {
    let id, team, role, metric, unit, period, windowStart, windowEnd: String
    let value: Double
    let sample: Int
    let sourceIDs: [String]
}
struct SharedMatchupSnapshot: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let event: SharedMatchupEvent
    let sources: [SharedMatchupSource]
    let facts: [SharedMatchupFact]
}
struct SharedMatchupSelection: Codable, Equatable, Sendable { let observationIDs, outlookIDs: [String] }
struct SharedMatchupObservation: Codable, Equatable, Sendable {
    let factID, text, period, windowStart, windowEnd: String
    let sourceIDs: [String]
}
struct SharedMatchupOutlook: Codable, Equatable, Sendable {
    let id, ruleID, label, text, uncertainty: String
    let factIDs, sourceIDs: [String]
}
struct SharedMatchupReport: Codable, Equatable, Sendable {
    let contractVersion: String
    let selection: SharedMatchupSelection
    let observations: [SharedMatchupObservation]
    let outlook: [SharedMatchupOutlook]
    let missingContext: [String]
    let provenance: String
}
struct SharedMatchupReportResponse: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let status: SharedMatchupReportStatus
    let reason: String?
    let event: SharedMatchupEvent
    let generatedAt, sourceAsOf, sourceRetrievedAt, sourceCheckedAt, sourceFingerprint: String?
    let versions: SharedMatchupVersions
    let refreshing: Bool
    let report: SharedMatchupReport?
    let snapshot: SharedMatchupSnapshot?
}
enum SharedMatchupReportError: Error, Equatable { case invalidEndpoint, authenticationRequired, unavailable, invalidResponse, oversizedPayload }

/// Ages an already validated display value without re-encoding its exact numeric premises.
/// This is not a substitute for wire validation or an authorization check.
enum SharedMatchupDisplayClock {
    static func update(_ value: SharedMatchupReportResponse, now: Date) throws -> SharedMatchupReportResponse {
        let start = try SharedMatchupReportValidator.date(value.event.startISO)
        if start <= now {
            return .init(schemaVersion:value.schemaVersion,status:.missing,reason:"event_started",event:value.event,
                generatedAt:nil,sourceAsOf:nil,sourceRetrievedAt:nil,sourceCheckedAt:nil,sourceFingerprint:nil,
                versions:value.versions,refreshing:false,report:nil,snapshot:nil)
        }
        guard value.report != nil else { return value }
        guard let generated=value.generatedAt, let checked=value.sourceCheckedAt else {throw SharedMatchupReportError.invalidResponse}
        let generation=try SharedMatchupReportValidator.date(generated), source=try SharedMatchupReportValidator.date(checked)
        guard generation<=now, source<=now else {throw SharedMatchupReportError.invalidResponse}
        let reason = now.timeIntervalSince(generation)>=86400 ? "generation_expired" : now.timeIntervalSince(source)>=86400 ? "source_expired" : value.reason
        return .init(schemaVersion:value.schemaVersion,status:reason == nil ? .ready:.stale,reason:reason,event:value.event,
            generatedAt:value.generatedAt,sourceAsOf:value.sourceAsOf,sourceRetrievedAt:value.sourceRetrievedAt,
            sourceCheckedAt:value.sourceCheckedAt,sourceFingerprint:value.sourceFingerprint,versions:value.versions,
            refreshing:value.refreshing,report:value.report,snapshot:value.snapshot)
    }
    static func nextCheck(_ value: SharedMatchupReportResponse, now: Date) throws -> Date {
        _ = try update(value,now:now)
        var deadlines=[now.addingTimeInterval(60),try SharedMatchupReportValidator.date(value.event.startISO)]
        for stamp in [value.generatedAt,value.sourceCheckedAt].compactMap({$0}) {
            deadlines.append(try SharedMatchupReportValidator.date(stamp).addingTimeInterval(86400))
        }
        return deadlines.filter {$0>now}.min() ?? now.addingTimeInterval(60)
    }
}

/// Public cache reads only: no prompt, user evidence, job creation, inference, or quota API.
struct SharedMatchupReportClient: Sendable {
    typealias Transport = @Sendable (URLRequest) async throws -> (Data, HTTPURLResponse)
    let endpoint: URL
    let tokenProvider: @Sendable () async throws -> String
    let transport: Transport
    let now: @Sendable () -> Date
    let expectedVersions: SharedMatchupVersions
    let localReader: (@Sendable (SharedMatchupEvent,Date) throws -> SharedMatchupReportResponse)?
    init(endpoint: URL, tokenProvider: @escaping @Sendable () async throws -> String,
         transport: Transport? = nil, now: @escaping @Sendable () -> Date = { Date() },
         expectedVersions: SharedMatchupVersions = .current,
         localReader: (@Sendable (SharedMatchupEvent,Date) throws -> SharedMatchupReportResponse)? = nil) {
        self.endpoint = endpoint; self.tokenProvider = tokenProvider
        self.transport = transport ?? { request in try await Self.read(request) }; self.now = now; self.expectedVersions = expectedVersions; self.localReader=localReader
    }
    func fetch(event: SharedMatchupEvent) async throws -> SharedMatchupReportResponse {
        guard var url = URLComponents(url: endpoint, resolvingAgainstBaseURL: false), url.scheme == "https",
              url.host != nil, url.user == nil, url.password == nil, url.port == nil,
              url.query == nil, url.fragment == nil, url.path == "/plonn/v1/insights/report" else { throw SharedMatchupReportError.invalidEndpoint }
        let event = try SharedMatchupReportValidator.normalizedEvent(event)
        url.queryItems = ["sport": event.sport, "eventID": event.eventID, "homeTeam": event.homeTeam, "awayTeam": event.awayTeam, "startISO": event.startISO].sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) }
        guard let target = url.url else { throw SharedMatchupReportError.invalidEndpoint }
        try Task.checkCancellation()
        let token = try await tokenProvider()
        try Task.checkCancellation()
        guard !token.isEmpty, token.utf8.count <= 8192, token.utf8.allSatisfy({ (33...126).contains($0) }) else { throw SharedMatchupReportError.authenticationRequired }
        if let localReader {return try localReader(event,now())}
        var request = URLRequest(url: target, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 20)
        request.httpMethod = "GET"; request.httpShouldHandleCookies = false
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await transport(request)
        try Task.checkCancellation()
        guard response.url == target else { throw SharedMatchupReportError.invalidResponse }
        guard data.count <= SharedMatchupReportValidator.maximumBytes else { throw SharedMatchupReportError.oversizedPayload }
        guard response.statusCode == 200 else {
            throw [401, 403].contains(response.statusCode) ? SharedMatchupReportError.authenticationRequired : SharedMatchupReportError.unavailable
        }
        return try SharedMatchupReportValidator.validate(data, event: event, now: now(), expectedVersions: expectedVersions)
    }
    private static func read(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let delegate = SharedMatchupReadDelegate(expectedURL: request.url!)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil; configuration.urlCredentialStorage = nil; configuration.urlCache = nil
        configuration.httpShouldSetCookies = false; configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 20; configuration.timeoutIntervalForResource = 25
        configuration.connectionProxyDictionary = [:]
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        return try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { delegate.start(session: session, request: request, continuation: $0) }
        }, onCancel: { delegate.cancel() })
    }
}

/// Rebuilds the report from typed premises and approved templates, never trusts server prose.
enum SharedMatchupReportValidator {
    static let maximumBytes = 65_536
    private static let metrics: [String: [String: (String, Double)]] = [
        "NFL": ["passingAttemptsPerGame": ("pass attempts per game",100), "rushingAttemptsPerGame": ("rush attempts per game",100), "pointsPerGame": ("points per game",100)],
        "NBA": ["pointsPerGame": ("points per game",200), "pace": ("possessions per game",200), "offensiveEfficiency": ("points per 100 possessions",200)],
        "MLB": ["runsPerGame": ("runs per game",30)],
        "NHL": ["goalsPerGame": ("goals per game",20), "shotsPerGame": ("shots on goal per game",100)]]
    private static let gaps = [
        "NFL": "Verified injuries, weather, and current lineups are not included.",
        "NBA": "Verified availability, rotations, and rest adjustments are not included.",
        "MLB": "Confirmed starting pitchers, bullpen availability, park, and weather are not included. Team runs allowed describe the recorded pitching and defense together.",
        "NHL": "Confirmed starting goaltenders, goalie-specific form, and special teams are not included. Goals allowed describe team defense and goaltending together."]
    private static func require(_ condition: Bool) throws { if !condition { throw SharedMatchupReportError.invalidResponse } }
    private static func matches(_ text: String, _ pattern: String) -> Bool { text.range(of: pattern, options: .regularExpression) != nil }
    private static func text(_ value: String, limit: Int) -> Bool {
        !value.isEmpty && value.unicodeScalars.count <= limit && value == value.trimmingCharacters(in: .whitespacesAndNewlines)
        && value == value.precomposedStringWithCanonicalMapping
        && value.unicodeScalars.allSatisfy { !CharacterSet.controlCharacters.contains($0) && !CharacterSet.illegalCharacters.contains($0) && !CharacterSet.symbols.contains($0) && $0.properties.generalCategory != .unassigned && $0.properties.generalCategory != .privateUse }
    }
    static func validateSourceURL(_ value:String) throws {
            let hosts = Set(["site.api.espn.com","site.web.api.espn.com","sports.core.api.espn.com","statsapi.mlb.com","api-web.nhle.com","api.nhle.com","stats.nba.com"])

        try require(value.unicodeScalars.count <= 1000 && text(String(value.split(separator:"?",maxSplits:1,omittingEmptySubsequences:false)[0]),limit:1000) && !value.unicodeScalars.contains {CharacterSet.whitespacesAndNewlines.contains($0) || CharacterSet.controlCharacters.contains($0)})
        guard let url=URLComponents(string:value) else {throw SharedMatchupReportError.invalidResponse}
        try require(url.scheme == "https" && hosts.contains(url.host ?? "") && url.user == nil && url.password == nil && url.port == nil && !url.path.isEmpty && url.fragment == nil && value.hasPrefix("https://"+(url.host ?? "")+"/"))
        let rules=["season":"20[0-9]{2}","seasontype":"2","limit":"1000","dates":"[0-9]{8}(?:-[0-9]{8})?","stats":"season","group":"hitting|pitching","sportIds":"1","gameType":"R"]
        let items=url.queryItems ?? []
        try require(items.count <= 8 && Set(items.map(\.name)).count == items.count)
        for item in items {
            guard let pattern=rules[item.name],let value=item.value else {throw SharedMatchupReportError.invalidResponse}
            try require(matches(value,"\\A(?:"+pattern+")\\z"))
        }
    }
    static func date(_ value: String) throws -> Date {
        try require(value.count <= 40 && matches(value, #"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"#))
        let format = ISO8601DateFormatter(); format.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let result = format.date(from: value) { return result }
        format.formatOptions = [.withInternetDateTime]
        guard let result = format.date(from: value) else { throw SharedMatchupReportError.invalidResponse }; return result
    }
    private static func iso(_ value: Date) -> String {
        let format = ISO8601DateFormatter(); format.formatOptions = [.withInternetDateTime, .withFractionalSeconds]; return format.string(from: value)
    }
    static func normalizedEvent(_ event: SharedMatchupEvent) throws -> SharedMatchupEvent {
        try require(metrics[event.sport] != nil && matches(event.eventID, #"^espn:[0-9]{1,15}$"#)
                    && text(event.homeTeam, limit: 100) && text(event.awayTeam, limit: 100)
                    && event.homeTeam.folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX")) != event.awayTeam.folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX")))
        return SharedMatchupEvent(sport: event.sport, eventID: event.eventID, homeTeam: event.homeTeam, awayTeam: event.awayTeam, startISO: iso(try date(event.startISO)))
    }
    static func validate(_ data: Data, event: SharedMatchupEvent, now: Date,
                         expectedVersions: SharedMatchupVersions = .current) throws -> SharedMatchupReportResponse {
        guard data.count <= maximumBytes else { throw SharedMatchupReportError.oversizedPayload }
        do {
            var parser = SharedReportJSONParser(data)
            let root = try parser.parse()
            try root.keys(["schemaVersion","status","reason","event","generatedAt","sourceAsOf","sourceRetrievedAt","sourceCheckedAt","sourceFingerprint","versions","refreshing","report","snapshot"])
            try root["event"].keys(["sport","eventID","homeTeam","awayTeam","startISO"])
            try root["versions"].keys(["contract","rules","model","modelDigest","prompt","validator"])
            try require(root["schemaVersion"].integer == 1)
            let response = try JSONDecoder().decode(SharedMatchupReportResponse.self, from: data)
            let identity = try normalizedEvent(event)
            try require(try normalizedEvent(response.event) == identity)
            try require(response.versions.contract == expectedVersions.contract && [expectedVersions.rules,"paired-reference-v3"].contains(response.versions.rules)
                && response.versions.model == expectedVersions.model && response.versions.modelDigest == expectedVersions.modelDigest
                && response.versions.prompt == expectedVersions.prompt && matches(response.versions.validator, #"^[0-9a-f]{64}$"#)
                && now.timeIntervalSince1970.isFinite)
            if response.status == .missing {
                try require(response.report == nil && response.snapshot == nil && response.generatedAt == nil)
                try require(["cache_miss","source_changed","version_changed","invalid_cached_report","source_unavailable","generation_failed","runtime_blocked","event_started"].contains(response.reason ?? ""))
                let started = try date(identity.startISO) <= now
                try require(!started || response.reason == "event_started")
                try require(response.reason != "event_started" || started)
                if let fingerprint = response.sourceFingerprint {
                    try require(matches(fingerprint, #"^[0-9a-f]{64}$"#))
                    guard let retrieved = response.sourceRetrievedAt, let checked = response.sourceCheckedAt else { throw SharedMatchupReportError.invalidResponse }
                    try require(date(retrieved) <= date(checked) && date(checked) <= now)
                    if let asOf = response.sourceAsOf { try require(date(asOf) <= date(retrieved)) }
                } else {
                    try require(response.sourceAsOf == nil && response.sourceRetrievedAt == nil && response.sourceCheckedAt == nil)
                }
                return response
            }
            guard let snapshot = response.snapshot, let report = response.report, let generated = response.generatedAt,
                  let fingerprint = response.sourceFingerprint else { throw SharedMatchupReportError.invalidResponse }
            try require(try date(identity.startISO) > now)
            try require(response.status == .ready ? response.reason == nil : ["generation_expired","source_expired"].contains(response.reason ?? ""))
            var rawSnapshot = root["snapshot"]
            try rawSnapshot.keys(["schemaVersion","event","sources","facts"])
            try rawSnapshot["event"].keys(["sport","eventID","homeTeam","awayTeam","startISO"])
            try require(rawSnapshot["schemaVersion"].integer == 1 && rawSnapshot.canonical.utf8.count <= 32768)
            try require(try normalizedEvent(snapshot.event) == identity)
            try require((1...40).contains(snapshot.sources.count) && (1...24).contains(snapshot.facts.count))
            var sourceIDs = Set<String>(); var sourceDates: [String: Date] = [:]
            for (index, source) in snapshot.sources.enumerated() {
                try rawSnapshot["sources"].array[index].keys(["id","provider","url","retrievedAt","checkedAt","sourceAsOf"])
                try require(text(source.id, limit: 80) && matches(source.id, #"^[a-zA-Z0-9._:-]+$"#) && sourceIDs.insert(source.id).inserted)
                try require(["ESPN","MLB Stats API","NHL","NBA"].contains(source.provider))
                try validateSourceURL(source.url)
                let retrieved = try date(source.retrievedAt), checked = try date(source.checkedAt)
                try require(retrieved <= checked && checked <= now)
                if let asOf = source.sourceAsOf { try require(try date(asOf) <= retrieved) }
                sourceDates[source.id] = retrieved
            }
            var ids = Set<String>(), dimensions = Set<String>()
            for (index, fact) in snapshot.facts.enumerated() {
                let raw = rawSnapshot["facts"].array[index]
                try raw.keys(["id","team","role","metric","value","unit","sample","period","windowStart","windowEnd","sourceIDs"])
                try require(text(fact.id, limit: 80) && matches(fact.id, #"^[a-zA-Z0-9._:-]+$"#) && ids.insert(fact.id).inserted)
                try require(["offense","defense","reference"].contains(fact.role))
                try require(fact.role == "reference" ? fact.team == "league" : [identity.homeTeam,identity.awayTeam].contains(fact.team))
                guard let metric = metrics[identity.sport]?[fact.metric] else { throw SharedMatchupReportError.invalidResponse }
                try require(raw["value"].number != nil && fact.value.isFinite && fact.value >= 0 && fact.value <= metric.1 && fact.unit == metric.0)
                try require(fact.role != "reference" || fact.value > 0)
                try require(raw["sample"].integer == fact.sample && (1...100000).contains(fact.sample) && text(fact.period, limit: 80))
                try require((1...(fact.role == "reference" ? 32 : 4)).contains(fact.sourceIDs.count) && Set(fact.sourceIDs).count == fact.sourceIDs.count && Set(fact.sourceIDs).isSubset(of: sourceIDs))
                let start = try date(fact.windowStart), end = try date(fact.windowEnd)
                try require(start < end && end < date(identity.startISO) && end <= fact.sourceIDs.compactMap { sourceDates[$0] }.min()!)
                try require(dimensions.insert([fact.team,fact.role,fact.metric].joined(separator: "\u{0}")).inserted)
            }
            // Python canonical snapshot normalizes only event.startISO and collection order.
            rawSnapshot = rawSnapshot.setting("event", rawSnapshot["event"].setting("startISO", .string(identity.startISO)))
            rawSnapshot = rawSnapshot.setting("facts", .list(rawSnapshot["facts"].array.sorted { $0["id"].string < $1["id"].string }))
            rawSnapshot = rawSnapshot.setting("sources", .list(rawSnapshot["sources"].array.sorted { $0["id"].string < $1["id"].string }))
            let content = rawSnapshot.setting("sources", .list(rawSnapshot["sources"].array.map { $0.removing(["retrievedAt","checkedAt"]) }))
            let actualFingerprint = SHA256.hash(data: Data(content.canonical.utf8)).map { String(format: "%02x", $0) }.joined()
            try require(fingerprint == actualFingerprint)
            let sources = snapshot.sources
            let retrieved = try sources.map { try date($0.retrievedAt) }.min()!, checked = try sources.map { try date($0.checkedAt) }.min()!
            let allAsOf = sources.compactMap(\.sourceAsOf)
            let asOf = allAsOf.count == sources.count ? try allAsOf.sorted { try date($0) < date($1) }.first : nil
            try require(response.sourceAsOf == asOf && response.sourceRetrievedAt == iso(retrieved) && response.sourceCheckedAt == iso(checked))
            let generatedDate = try date(generated)
            try require(generatedDate <= now)
            let fresh = now.timeIntervalSince(generatedDate) < 86400 && now.timeIntervalSince(checked) < 86400
            try require(response.status == (fresh ? .ready : .stale))
            if response.reason == "generation_expired" { try require(now.timeIntervalSince(generatedDate) >= 86400) }
            if response.reason == "source_expired" { try require(now.timeIntervalSince(checked) >= 86400) }
            try validateReportShape(root["report"])
            let expected = try render(selection: report.selection, snapshot: snapshot, rules: response.versions.rules)
            try require(report == expected)
            return response
        } catch let error as SharedMatchupReportError { throw error }
        catch { throw SharedMatchupReportError.invalidResponse }
    }
    private static func validateReportShape(_ raw: SharedReportJSON) throws {
        try raw.keys(["contractVersion","selection","observations","outlook","missingContext","provenance"])
        try raw["selection"].keys(["observationIDs","outlookIDs"])
        for item in raw["observations"].array { try item.keys(["factID","text","sourceIDs","period","windowStart","windowEnd"]) }
        for item in raw["outlook"].array { try item.keys(["id","ruleID","label","text","factIDs","sourceIDs","uncertainty"]) }
    }
    static func render(selection: SharedMatchupSelection, snapshot: SharedMatchupSnapshot, rules: String = "paired-reference-v2") throws -> SharedMatchupReport {
        let event = snapshot.event, facts = snapshot.facts.sorted { $0.id < $1.id }
        var outlooks: [SharedMatchupOutlook] = []
        let minimum = ["NFL":3,"NBA":5,"MLB":10,"NHL":5][event.sport]!
        func fact(_ team: String, _ role: String, _ metric: String) -> SharedMatchupFact? { facts.first { $0.team == team && $0.role == role && $0.metric == metric } }
        func compatible(_ items: [SharedMatchupFact?]) -> [SharedMatchupFact]? {
            let values = items.compactMap { $0 }; guard values.count == items.count, let first = values.first else { return nil }
            guard values.allSatisfy({ $0.sample >= minimum && $0.unit == first.unit && $0.period == first.period && (try? date($0.windowStart)) == (try? date(first.windowStart)) && (try? date($0.windowEnd)) == (try? date(first.windowEnd)) }) else { return nil }
            return values
        }
        func relative(_ value: SharedMatchupFact, _ reference: SharedMatchupFact, high: Bool, margin: Int = 10) -> Bool {
            // Compare decimal integers and exponents, matching Decimal(str(value))
            // without binary rounding at thresholds or Decimal's exponent limit.
            func scaled(_ value: Double, by multiplier: Int) -> (String, Int) {
                let parts = String(value).lowercased().split(separator: "e")
                let mantissa = String(parts[0]), exponent = parts.count == 2 ? Int(parts[1])! : 0
                let decimals = mantissa.split(separator: ".", omittingEmptySubsequences: false)
                let scale = exponent - (decimals.count == 2 ? decimals[1].count : 0)
                var digits = mantissa.filter { $0.isNumber }.map { Int(String($0))! }
                var carry = 0
                for index in digits.indices.reversed() { let product = digits[index] * multiplier + carry; digits[index] = product % 10; carry = product / 10 }
                while carry > 0 { digits.insert(carry % 10, at: 0); carry /= 10 }
                while digits.count > 1 && digits.first == 0 { digits.removeFirst() }
                return (digits.map(String.init).joined(), scale)
            }
            let left = scaled(value.value, by: 100), right = scaled(reference.value, by: high ? 100 + margin : 100 - margin)
            let exponent = min(left.1, right.1)
            let lhs = left.0 == "0" ? "0" : left.0 + String(repeating: "0", count: left.1 - exponent)
            let rhs = right.0 == "0" ? "0" : right.0 + String(repeating: "0", count: right.1 - exponent)
            let order = lhs.count == rhs.count ? lhs.compare(rhs) : (lhs.count < rhs.count ? ComparisonResult.orderedAscending : .orderedDescending)
            return high ? order != .orderedAscending : order != .orderedDescending
        }
        func add(_ id: String, _ label: String, _ sentence: String, _ items: [SharedMatchupFact]) {
            outlooks.append(SharedMatchupOutlook(id: id, ruleID: rules + ":" + id, label: label, text: sentence,
                uncertainty: "Conditional team profile from the cited sample, not a calibrated prediction. Game context and small samples can change the pattern.", factIDs: items.map(\.id), sourceIDs: Set(items.flatMap(\.sourceIDs)).sorted()))
        }
        let phrases = ["passingAttemptsPerGame":("Passing volume","a higher passing-volume profile"), "rushingAttemptsPerGame":("Rushing volume","a higher rushing-volume profile"), "pointsPerGame":("Scoring","a higher-scoring profile"), "offensiveEfficiency":("Scoring efficiency","efficient scoring"), "runsPerGame":("Offense versus pitching","a productive run-scoring profile against the recorded team pitching and defense"), "goalsPerGame":("Scoring","a higher-scoring profile"), "shotsPerGame":("Shot volume","a higher shot-volume profile")]
        for (side, team, opponent) in [("home",event.homeTeam,event.awayTeam),("away",event.awayTeam,event.homeTeam)] {
            for metric in metrics[event.sport]!.keys.sorted() where metric != "pace" {
                guard let values = compatible([fact(team,"offense",metric),fact(opponent,"defense",metric),fact("league","reference",metric)]) else { continue }
                let own = values[0], defense = values[1], reference = values[2]
                if relative(own,reference,high:true) && relative(defense,reference,high:true) {
                    let phrase = phrases[metric]!
                    add(side+":"+metric+":high",phrase.0,"If the recorded patterns persist, \(team) could show \(phrase.1) against \(opponent). Its offense and the opposing allowance are both above the cited league reference.",values)
                } else if relative(defense,reference,high:false) && own.value >= reference.value {
                    if ["passingAttemptsPerGame","rushingAttemptsPerGame","shotsPerGame"].contains(metric) {
                        add(side+":"+metric+":resistance","Contrasting volume patterns","\(team) faces contrasting recorded volume against \(opponent): its offensive volume is at or above the league reference, while the opposing allowance is below it. These samples do not establish which pattern will persist.",values)
                    } else {
                    add(side+":"+metric+":resistance","Opposing defensive resistance","\(team) faces a contrasting profile against \(opponent): its recorded offense is at or above the league reference, while the opposing allowance is below it. That could constrain this part of the offense if those patterns persist.",values)
                    }
                }
            }
        }
        let score = ["NFL":"pointsPerGame","NBA":"offensiveEfficiency","MLB":"runsPerGame","NHL":"goalsPerGame"][event.sport]!
        let scoring = [event.homeTeam,event.awayTeam].flatMap { team in ["offense","defense"].map { fact(team,$0,score) } } + [fact("league","reference",score)]
        if let values = compatible(scoring), values.dropLast().allSatisfy({ relative($0,values.last!,high:false) }) {
            if event.sport == "NBA" {
                add("game:lower-efficiency","Lower scoring efficiency","Lower scoring efficiency is plausible if the recorded patterns persist: both offenses and both opposing allowances sit below the league efficiency reference. Efficiency alone does not establish a low game total.",values)
            } else {
                add("game:lower-scoring","Lower-scoring profile","A lower-scoring matchup is plausible if the recorded patterns persist: both offenses and both opposing allowances sit below the same league scoring reference.",values)
            }
        }
        if event.sport == "NFL", let values=compatible(scoring) {
            let reference=values[4]
            let near=[values[1],values[3]].allSatisfy {relative($0,reference,high:true,margin:-10) && relative($0,reference,high:false,margin:-10)}
            for (high,low) in [(values[0],values[2]),(values[2],values[0])] where near && relative(high,reference,high:true) && relative(low,reference,high:false) {
                add("game:mixed-scoring","Mixed scoring signals","\(high.team) has scored above the league reference, while \(low.team) has scored below it. Both opposing allowances are within ten percent of the same reference. These mixed signals do not establish a clear scoring direction for the matchup; player availability and game context could change the pattern.",values)
            }
        }
        if event.sport == "NBA", let values = compatible([fact(event.homeTeam,"offense","pace"),fact(event.awayTeam,"offense","pace"),fact("league","reference","pace")]) {
            for (sign,label) in [("low","Slower tempo"),("high","Faster tempo")] where values.dropLast().allSatisfy({ relative($0,values.last!,high:sign == "high",margin:5) }) {
                add("game:pace:"+sign,label,"\(label) is plausible if both teams retain their recorded possession patterns relative to the cited league pace. Pace alone does not establish scoring efficiency or a game total.",values)
            }
        }
        try require(["paired-reference-v2","paired-reference-v3"].contains(rules))
        if rules == "paired-reference-v3", outlooks.isEmpty {
            let metric=["NFL":"pointsPerGame","NBA":"pointsPerGame","MLB":"runsPerGame","NHL":"goalsPerGame"][event.sport]!
            var values=[event.homeTeam,event.awayTeam].flatMap {team in ["offense","defense"].compactMap {fact(team,$0,metric)}}
            if values.count == 4, let first=values.first,
               values.allSatisfy({$0.unit == first.unit && $0.period == first.period && (try? date($0.windowStart)) == (try? date(first.windowStart)) && (try? date($0.windowEnd)) == (try? date(first.windowEnd))}),
               values[0].sample == values[1].sample, values[2].sample == values[3].sample {
                let short=[values[0],values[2]].filter {$0.sample < minimum}.map(\.team)
                var id="",label="",sentence="",uncertainty=""
                if !short.isEmpty {
                    id="game:limited-sample";label="Limited sample"
                    sentence="The recorded sample for \(short.joined(separator:" and ")) is below the \(minimum)-game minimum for a directional scoring outlook. These observations describe a limited sample; they do not establish how this matchup will play."
                    uncertainty="Sample limitation only, not a directional prediction. Additional games and verified game context are needed."
                } else if let ref=fact("league","reference",metric), compatible(values.map {Optional($0)}+[ref]) != nil {
                    func band(_ value:SharedMatchupFact)->String {
                        if relative(value,ref,high:true) {return "at least ten percent above"}
                        if relative(value,ref,high:false) {return "at least ten percent below"}
                        return "within ten percent of"
                    }
                    if values.allSatisfy({band($0) == "within ten percent of"}) {
                        id="game:near-reference";label="No strong scoring tilt"
                        sentence="Both offenses and both opposing allowances are within ten percent of the same league scoring reference. This paired sample does not establish a strong scoring tilt for the matchup."
                    } else {
                        id="game:paired-context";label="Paired scoring context"
                        sentence="\(event.homeTeam)'s recorded scoring is \(band(values[0])) the league reference, while \(event.awayTeam)'s opposing allowance is \(band(values[3])) it. \(event.awayTeam)'s recorded scoring is \(band(values[2])) the same reference, while \(event.homeTeam)'s opposing allowance is \(band(values[1])) it. These paired rates do not supply a supported directional matchup signal under the scoring rules."
                    }
                    values.append(ref)
                    if event.sport == "NBA" {sentence += " Scoring rates alone do not establish pace, efficiency, or a game total."}
                    uncertainty="Context from the cited sample, not a calibrated prediction. Player availability and game conditions can change the pattern."
                }
                if !id.isEmpty {outlooks.append(SharedMatchupOutlook(id:id,ruleID:rules+":"+id,label:label,text:sentence,uncertainty:uncertainty,factIDs:values.map(\.id),sourceIDs:Set(values.flatMap(\.sourceIDs)).sorted()))}
            }
        }
        let observationIDs = Set(selection.observationIDs), outlookIDs = Set(selection.outlookIDs)
        try require((1...6).contains(selection.observationIDs.count) && selection.outlookIDs.count <= 3
                    && observationIDs.count == selection.observationIDs.count && outlookIDs.count == selection.outlookIDs.count
                    && observationIDs.isSubset(of: Set(facts.map(\.id))) && outlookIDs.isSubset(of: Set(outlooks.map(\.id))))
        let selected = selection.outlookIDs.map { id in outlooks.first { $0.id == id }! }
        let required = observationIDs.union(selected.flatMap(\.factIDs))
        let observations = facts.filter { required.contains($0.id) }.map { value in
            SharedMatchupObservation(factID:value.id,text:"\(value.role == "reference" ? "League reference" : value.team) \(value.role == "defense" ? "allows" : "records") \(String(format:"%.6g",locale:Locale(identifier:"en_US_POSIX"),value.value)) \(value.unit) across \(value.sample) team games.",period:value.period,windowStart:value.windowStart,windowEnd:value.windowEnd,sourceIDs:value.sourceIDs)
        }
        var missing = [gaps[event.sport]!]
        if event.sport == "NBA" {
            if !facts.contains(where: {$0.metric == "pace"}) {missing.append("Verified pace data are not supplied in this snapshot.")}
            if !facts.contains(where: {$0.metric == "offensiveEfficiency"}) {missing.append("Verified scoring efficiency data are not supplied in this snapshot.")}
        }
        if selected.isEmpty { missing.insert("No supported qualitative outlook was selected. Matching offense, opposing defense, league reference, sample, or timeframe may be missing or inconclusive.",at:0) }
        if rules == "paired-reference-v3", facts.contains(where: {$0.period.lowercased().contains("preseason")}) {missing.insert("Preseason observations are a limited exhibition sample; rotations and player availability can differ from regular-season games.",at:0)}
        return SharedMatchupReport(contractVersion:"shared-outlook-v1",selection:selection,observations:observations,outlook:selected,missingContext:missing,provenance:"Gemma4 selected public observations and eligible outlook rules; conditional wording is rule-based, not free-form model prose.")
    }
}

// Preserve JSON numeric spelling for Python-canonical fingerprints (40 vs 40.0),
// reject duplicate keys before Codable, and bound nesting before recursion.
private indirect enum SharedReportJSON {
    case object([String:SharedReportJSON]), list([SharedReportJSON]), string(String), number(String), bool(Bool), null
    subscript(_ key: String) -> SharedReportJSON { if case .object(let fields) = self { return fields[key] ?? .null }; return .null }
    var array: [SharedReportJSON] { if case .list(let items) = self {return items}; return [] }
    var string: String { if case .string(let value) = self {return value}; return "" }
    var number: String? { if case .number(let value) = self {return value}; return nil }
    var integer: Int? { guard let number, number.range(of:#"^-?(0|[1-9][0-9]*)$"#,options:.regularExpression) != nil else {return nil}; return Int(number) }
    func keys(_ names: Set<String>) throws { guard case .object(let fields) = self, Set(fields.keys) == names else {throw SharedMatchupReportError.invalidResponse} }
    func setting(_ key: String, _ value: SharedReportJSON) -> SharedReportJSON { guard case .object(var fields) = self else {return self};fields[key]=value;return .object(fields) }
    func removing(_ keys: Set<String>) -> SharedReportJSON { guard case .object(var fields) = self else {return self};for key in keys {fields.removeValue(forKey:key)};return .object(fields) }
    private func quote(_ string: String) -> String {
        let encoded = try! JSONSerialization.data(withJSONObject:string,options:[.fragmentsAllowed,.withoutEscapingSlashes])
        return String(data:encoded,encoding:.utf8)!
    }
    var canonical: String {
        switch self {
        case .object(let fields):return "{"+fields.keys.sorted().map {quote($0)+":"+fields[$0]!.canonical}.joined(separator:",")+"}"
        case .list(let values):return "["+values.map(\.canonical).joined(separator:",")+"]"
        case .string(let value):return quote(value)
        case .number(let value):return value
        case .bool(let value):return value ? "true":"false"
        case .null:return "null"
        }
    }
}
private struct SharedReportJSONParser {
    let bytes: [UInt8]; var index=0
    init(_ data:Data){bytes=Array(data)}
    mutating func parse() throws -> SharedReportJSON {let result=try value(0);space();guard index==bytes.count else{throw SharedMatchupReportError.invalidResponse};return result}
    mutating func space(){while index<bytes.count && [9,10,13,32].contains(bytes[index]){index+=1}}
    mutating func take(_ byte:UInt8)->Bool{space();if index<bytes.count && bytes[index]==byte{index+=1;return true};return false}
    mutating func string() throws -> String {
        space();let start=index;guard index<bytes.count && bytes[index]==34 else{throw SharedMatchupReportError.invalidResponse};index+=1
        while index<bytes.count {
            if bytes[index]==92 {index+=2;continue}
            if bytes[index]==34 {index+=1;return try JSONDecoder().decode(String.self,from:Data(bytes[start..<index]))}
            index+=1
        }
        throw SharedMatchupReportError.invalidResponse
    }
    mutating func value(_ depth:Int)throws->SharedReportJSON {
        guard depth<=16 else{throw SharedMatchupReportError.invalidResponse};space()
        guard index<bytes.count else{throw SharedMatchupReportError.invalidResponse}
        if bytes[index]==34{return .string(try string())}
        if take(123){var fields:[String:SharedReportJSON]=[:];if take(125){return .object(fields)}
            repeat {let key=try string();guard fields[key]==nil,take(58) else{throw SharedMatchupReportError.invalidResponse};fields[key]=try value(depth+1);if take(125){return .object(fields)}}while take(44)
            throw SharedMatchupReportError.invalidResponse}
        if take(91){var items:[SharedReportJSON]=[];if take(93){return .list(items)}
            repeat {items.append(try value(depth+1));if take(93){return .list(items)}}while take(44)
            throw SharedMatchupReportError.invalidResponse}
        for (literal,result) in [("null",SharedReportJSON.null),("true",.bool(true)),("false",.bool(false))] {
            let token=Array(literal.utf8);if index+token.count<=bytes.count && Array(bytes[index..<index+token.count])==token{index+=token.count;return result}
        }
        let start=index;while index<bytes.count && ![9,10,13,32,44,93,125].contains(bytes[index]){index+=1}
        let token=String(decoding:bytes[start..<index],as:UTF8.self)
        guard token.range(of:#"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"#,options:.regularExpression) != nil else{throw SharedMatchupReportError.invalidResponse}
        return .number(token)
    }
}

final class SharedMatchupReadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let expectedURL:URL
    private let lock=NSLock()
    private var data=Data(), response:HTTPURLResponse?, task:URLSessionTask?
    private var continuation:CheckedContinuation<(Data,HTTPURLResponse),Error>?
    private var terminal:Result<(Data,HTTPURLResponse),Error>?
    init(expectedURL:URL){self.expectedURL=expectedURL}
    func start(session:URLSession,request:URLRequest,continuation:CheckedContinuation<(Data,HTTPURLResponse),Error>){
        lock.lock();if let terminal{lock.unlock();continuation.resume(with:terminal);return}
        self.continuation=continuation;let task=session.dataTask(with:request);self.task=task;lock.unlock();task.resume()
    }
    func cancel(){finish(.failure(CancellationError()))}
    private func finish(_ result:Result<(Data,HTTPURLResponse),Error>){lock.lock();guard terminal==nil else{lock.unlock();return};terminal=result;let callback=continuation;continuation=nil;let task=task;self.task=nil;lock.unlock();task?.cancel();callback?.resume(with:result)}
    func urlSession(_ session:URLSession,dataTask:URLSessionDataTask,didReceive response:URLResponse,completionHandler:@escaping(URLSession.ResponseDisposition)->Void){
        guard let http=response as? HTTPURLResponse,http.url==expectedURL else{completionHandler(.cancel);finish(.failure(SharedMatchupReportError.invalidResponse));return}
        guard http.statusCode==200 else{completionHandler(.cancel);finish(.failure([401,403].contains(http.statusCode) ? SharedMatchupReportError.authenticationRequired:SharedMatchupReportError.unavailable));return}
        guard response.expectedContentLength<=SharedMatchupReportValidator.maximumBytes else{completionHandler(.cancel);finish(.failure(SharedMatchupReportError.oversizedPayload));return}
        lock.lock();guard terminal==nil else{lock.unlock();completionHandler(.cancel);return};self.response=http;lock.unlock();completionHandler(.allow)
    }
    func urlSession(_ session:URLSession,dataTask:URLSessionDataTask,didReceive chunk:Data){lock.lock();guard terminal==nil else{lock.unlock();return};guard chunk.count<=SharedMatchupReportValidator.maximumBytes-data.count else{lock.unlock();finish(.failure(SharedMatchupReportError.oversizedPayload));return};data.append(chunk);lock.unlock()}
    func urlSession(_ session:URLSession,task:URLSessionTask,didCompleteWithError error:Error?){if let error{finish(.failure(error));return};lock.lock();let response=self.response,data=self.data;lock.unlock();guard let response else{finish(.failure(SharedMatchupReportError.invalidResponse));return};finish(.success((data,response)))}
    func urlSession(_ session:URLSession,task:URLSessionTask,willPerformHTTPRedirection response:HTTPURLResponse,newRequest request:URLRequest,completionHandler:@escaping(URLRequest?)->Void){completionHandler(nil);finish(.failure(SharedMatchupReportError.invalidResponse))}
}

/// Local preview cache: exact matching bytes only; an absent event is never a generated report.
enum SharedMatchupLocalCache {
 static let maximumBundleBytes = 8_300_000
 /// Base64 retains each original UTF-8 envelope without numeric reserialization.
 static func records(bundle:Data,now:Date) throws -> [Data] {
  guard bundle.count <= maximumBundleBytes else {throw SharedMatchupReportError.oversizedPayload}
  let encoded=try JSONDecoder().decode([String].self,from:bundle)
  guard !encoded.isEmpty,encoded.count <= 94 else {throw SharedMatchupReportError.invalidResponse}
  var records=[Data](),identities=[SharedMatchupEvent]()
  for value in encoded {
   guard let raw=Data(base64Encoded:value),raw.count <= SharedMatchupReportValidator.maximumBytes else {throw SharedMatchupReportError.oversizedPayload}
   let envelope=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
   let identity=try SharedMatchupReportValidator.normalizedEvent(envelope.event)
   guard !identities.contains(identity) else {throw SharedMatchupReportError.invalidResponse}
   identities.append(identity)
   if try SharedMatchupReportValidator.date(identity.startISO) <= now {continue}
   let validated=try SharedMatchupReportValidator.validate(raw,event:identity,now:now)
   guard validated.status == .ready || validated.status == .stale else {throw SharedMatchupReportError.invalidResponse}
   records.append(raw)
  }
  return records
 }
 static func response(records:[Data],event:SharedMatchupEvent,now:Date) throws -> Data {
  let requested=try SharedMatchupReportValidator.normalizedEvent(event)
  if try SharedMatchupReportValidator.date(requested.startISO) > now {
   for raw in records {
    let saved=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
    if try SharedMatchupReportValidator.normalizedEvent(saved.event) == requested {return raw}
   }
  }
  return try missing(event:requested,now:now)
 }
 static func response(recorded:Data,event:SharedMatchupEvent,now:Date) throws -> Data {
  let requested=try SharedMatchupReportValidator.normalizedEvent(event)
  let saved=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:recorded)
  if try SharedMatchupReportValidator.normalizedEvent(saved.event) == requested {return recorded}
  return try missing(event:requested,now:now)
 }
 private static func missing(event requested:SharedMatchupEvent,now:Date) throws -> Data {
  let missing=SharedMatchupReportResponse(schemaVersion:1,status:.missing,reason:try SharedMatchupReportValidator.date(requested.startISO) <= now ? "event_started":"cache_miss",event:requested,generatedAt:nil,sourceAsOf:nil,sourceRetrievedAt:nil,sourceCheckedAt:nil,sourceFingerprint:nil,versions:.current,refreshing:false,report:nil,snapshot:nil)
  let encoder=JSONEncoder()
  // Wire requires explicit nulls for missing metadata.
  var object=try JSONSerialization.jsonObject(with:encoder.encode(missing)) as! [String:Any]
  for key in ["generatedAt","sourceAsOf","sourceRetrievedAt","sourceCheckedAt","sourceFingerprint","report","snapshot"] {object[key]=NSNull()}
  return try JSONSerialization.data(withJSONObject:object)
 }
}

/// Display-only projection. Call only with a response accepted by the wire validator.
struct SharedMatchupPresentationSection: Codable, Equatable {
 let id, title: String
 let evidenceLines: [String]
 let comparisonText, periodText: String?
 let takeaway, uncertainty: String
 let factIDs, sourceIDs: [String]
 let sampleWindow: Window?
 let mapped: Bool
 struct Window: Codable, Equatable {let startISO,endISO:String}
}
struct SharedMatchupPresentation: Codable, Equatable {
 let presentationVersion, originalSHA256: String
 let event: SharedMatchupEvent
 let generatedAt, sourceCheckedAt: String?
 let sections: [SharedMatchupPresentationSection]
 let provenance: String
}
enum SharedMatchupPresentationCopy {
 struct Proof:Decodable {let sport,method:String;let referenceFact:SharedMatchupFact;let teamCount,teamGames,total:Int}
 static let metrics:[String:[String]] = [
  "pointsPerGame":["points scored per game","points allowed per game","points per team per game","points"],
  "runsPerGame":["runs scored per game","runs allowed per game","runs per team per game","runs"],
  "goalsPerGame":["goals scored per game","goals allowed per game","goals per team per game","goals"],
  "shotsPerGame":["shots on goal per game","opponent shots on goal per game","shots on goal per team per game","shots on goal"],
  "passingAttemptsPerGame":["pass attempts per game","opponent pass attempts per game","pass attempts per team per game","pass attempts"],
  "rushingAttemptsPerGame":["rush attempts per game","opponent rush attempts per game","rush attempts per team per game","rush attempts"],
  "offensiveEfficiency":["points scored per 100 possessions","points allowed per 100 possessions","points per 100 possessions","points per 100 possessions"],
  "pace":["possessions per game","opponent possessions per game","possessions per team per game","possessions"]]
 static func number(_ value:Double)->String {
  var decimal=Decimal(string:String(value),locale:Locale(identifier:"en_US_POSIX")) ?? Decimal(value)
  var rounded=Decimal();NSDecimalRound(&rounded,&decimal,2,.plain)
  return NSDecimalNumber(decimal:rounded).stringValue
 }
 static func localDate(_ value:String,minute:Bool=false)->String {
  guard let date=try? SharedMatchupReportValidator.date(value) else{return "Unavailable"}
  let f=DateFormatter();f.locale=Locale(identifier:"en_US_POSIX");f.timeZone=TimeZone(identifier:"America/New_York")
  f.dateFormat=minute ? "MMM d, yyyy 'at' h:mm a zzz":"MMM d, yyyy"
  return f.string(from:date)
 }
 static func period(_ fact:SharedMatchupFact)->String {
  let text=fact.period
  if let r=text.range(of:#"^(NBA|NHL) 20[0-9]{2} regular season$"#,options:.regularExpression) {
   let p=String(text[r]).split(separator:" ");let year=Int(p[1])!
   return "\(p[0]) \(year-1)-\(String(year).suffix(2)) regular season"
  }
  if text.range(of:#"^NBA 20[0-9]{2} preseason, .+, NBA opponents$"#,options:.regularExpression) != nil {
   let year=Int(text.dropFirst(4).prefix(4))!
   let middle=String(text.dropFirst("NBA 2027 preseason, ".count).dropLast(", NBA opponents".count))
   return "NBA preseason, \(middle), \(year-1); games against NBA teams"
  }
  return text
 }
 static func comparison(_ fact:SharedMatchupFact,sport:String,proofs:[Proof])->String {
  let value=number(fact.value),unit=metrics[fact.metric]![2]
  let expected=["NFL":32,"NBA":30,"MLB":30,"NHL":32][sport]!
  if proofs.contains(where: {$0.sport==sport && $0.referenceFact==fact && $0.method=="sum_totals_divided_by_sum_team_games" && $0.teamCount==expected && $0.total>=0 && $0.teamGames==fact.sample && $0.teamGames>0 && abs(Double($0.total)/Double($0.teamGames)-fact.value)<1e-12}) {
   return "\(sport) sample average: \(value) \(unit) across all \(expected) teams (\(fact.sample) team-game observations)."
  }
  return "Recorded comparison: \(value) \(unit); \(fact.sample) team-game observations. Full-league coverage is not verified."
 }
 static func section(_ outlook:SharedMatchupOutlook,snapshot:SharedMatchupSnapshot,checkedAt:String,proofs:[Proof]=proofs)->SharedMatchupPresentationSection {
  let items=outlook.factIDs.compactMap {id in snapshot.facts.first {$0.id==id}}
  let sport=snapshot.event.sport,rule=outlook.id,metric=items.first?.metric ?? ""
  let parts=rule.split(separator:":").map(String.init)
  let side=parts.count==3 && ["home","away"].contains(parts[0]) && parts[1]==metric && ["high","resistance"].contains(parts[2])
  let known=["game:mixed-scoring","game:paired-context","game:near-reference","game:limited-sample","game:lower-scoring","game:lower-efficiency","game:pace:low","game:pace:high"]
  func fallback()->SharedMatchupPresentationSection { .init(id:rule,title:outlook.label,evidenceLines:[],comparisonText:nil,periodText:nil,takeaway:outlook.text,uncertainty:outlook.uncertainty,factIDs:outlook.factIDs,sourceIDs:outlook.sourceIDs,sampleWindow:nil,mapped:false) }
  guard let names=metrics[metric],side || known.contains(rule),items.count==outlook.factIDs.count,let first=items.first,items.allSatisfy({$0.windowStart==first.windowStart && $0.windowEnd==first.windowEnd}) else{return fallback()}
  var lines=[String]()
  for team in [snapshot.event.awayTeam,snapshot.event.homeTeam] {
   let facts=items.filter {$0.team==team && $0.role != "reference"}
   if facts.isEmpty {continue}
   let phrases=facts.map {number($0.value)+" "+metrics[$0.metric]![$0.role=="offense" ? 0:1]}
   let samples=Set(facts.map(\.sample)).sorted().map(String.init).joined(separator:"/")
   lines.append("\(team): \(phrases.joined(separator:"; ")) (\(samples) recorded games).")
  }
  let comparisonText=items.first(where:{$0.role=="reference"}).map {comparison($0,sport:sport,proofs:proofs)}
  let label=names[3]
  var title:String,takeaway:String
  if side {
   let team=parts[0]=="home" ? snapshot.event.homeTeam:snapshot.event.awayTeam
   let opponent=parts[0]=="home" ? snapshot.event.awayTeam:snapshot.event.homeTeam
   if parts[2]=="high" {
    title="More \(label) possible";takeaway="These rates could support more \(label) for \(team) if the recorded patterns continue."
   } else if ["passingAttemptsPerGame","rushingAttemptsPerGame","shotsPerGame"].contains(metric) {
    title="Different \(label) patterns";takeaway="\(team)'s \(label) and the \(label) recorded against \(opponent) point in different directions. These games do not show which rate will carry into this matchup."
   } else {
    title=metric=="offensiveEfficiency" ? "Scoring efficiency may be lower":"\(label.prefix(1).uppercased()+label.dropFirst()) may be harder to score"
    takeaway="The low rate of \(label) allowed by \(opponent) could make scoring harder for \(team) if that pattern continues."
   }
  } else if rule=="game:limited-sample" {
   title="Too few games to judge scoring";let minimum=["NFL":3,"NBA":5,"MLB":10,"NHL":5][sport]!
   var short=[String]();for f in items where f.sample<minimum {if !short.contains(f.team){short.append(f.team)}}
   takeaway="With fewer than \(minimum) recorded games for \(short.joined(separator:" and ")), it is too early to draw a scoring expectation from these results."
  } else if rule=="game:mixed-scoring" {
   title="Scoring rates point in different directions";takeaway="The scoring rates point in different directions, so this sample does not support a clear high- or low-scoring expectation."
  } else if ["game:paired-context","game:near-reference"].contains(rule) {
   title=["runsPerGame":"No clear run-scoring direction","goalsPerGame":"No clear goal-scoring direction"][metric] ?? "No clear scoring direction"
   takeaway=rule=="game:near-reference" ? "The \(label) scored and allowed are close to the comparison number, so this sample does not support a strong scoring expectation.":"The \(label) scored and allowed do not line up into one clear expectation for this matchup."
  } else if rule=="game:lower-scoring" {
   title="Fewer \(label) possible";takeaway="Both teams' \(label) scored and allowed are low in this sample. Fewer \(label) are possible if those rates continue."
  } else if rule=="game:lower-efficiency" {
   title="Fewer points per possession possible";takeaway="Both teams have scored and allowed fewer points per 100 possessions in this sample. That could mean less efficient scoring; it does not establish a low game total."
  } else {
   let speed=rule.hasSuffix(":low") ? "slower":"faster"
   title=rule.hasSuffix(":low") ? "Slower game possible":"Faster game possible"
   takeaway="The possession rates point to a \(speed) game if they continue. Pace alone does not establish scoring efficiency or the game total."
  }
  var uncertainty=[
   "NFL":"Injuries, current lineups and weather are not included. These are conditional readings of recorded games, not winner or score predictions.",
   "NBA":"Player availability, rotations and rest are not included. Points per game do not establish pace or scoring efficiency.",
   "MLB":"Confirmed starting pitchers, bullpen availability, park and weather are not included. Runs allowed reflect team pitching and fielding together.",
   "NHL":"Starting goalies and special teams are not included. Goals allowed reflect team defense and goaltending together."][sport]!
  if sport=="NBA" && items.contains(where:{$0.period.contains("preseason")}) {uncertainty="These are preseason exhibition games, when lineups and minutes can change. "+uncertainty}
  var periods=[String]();for f in items {let p=period(f);if !periods.contains(p){periods.append(p)}}
  let periodText="Sample: "+periods.joined(separator:"; ")+". Sample window: \(localDate(first.windowStart,minute:true)) to \(localDate(first.windowEnd,minute:true)). Sources checked \(localDate(checkedAt))."
  return .init(id:rule,title:title,evidenceLines:lines,comparisonText:comparisonText,periodText:periodText,takeaway:takeaway,uncertainty:uncertainty,factIDs:outlook.factIDs,sourceIDs:outlook.sourceIDs,sampleWindow:.init(startISO:first.windowStart,endISO:first.windowEnd),mapped:true)
 }
 static func sections(_ response:SharedMatchupReportResponse)->[SharedMatchupPresentationSection] {
  guard let report=response.report,let snapshot=response.snapshot,let checked=response.sourceCheckedAt else{return []}
  return report.outlook.map {section($0,snapshot:snapshot,checkedAt:checked)}
 }
 static func project(_ raw:Data,now:Date)throws->SharedMatchupPresentation {
  let envelope=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
  let response=try SharedMatchupReportValidator.validate(raw,event:envelope.event,now:now)
  guard response.report != nil else {throw SharedMatchupReportError.invalidResponse}
  return .init(presentationVersion:"matchup-card-copy-v1",originalSHA256:SHA256.hash(data:raw).map {String(format:"%02x",$0)}.joined(),event:response.event,generatedAt:response.generatedAt,sourceCheckedAt:response.sourceCheckedAt,sections:sections(response),provenance:"Display wording derived from the original validated facts and selected rules. The frozen model response is unchanged.")
 }
 // Exact audited comparison catalog; full fact equality is required.
 static let proofs:[Proof] = (try? JSONDecoder().decode([Proof].self,from:Data(#"""
[{"method":"sum_totals_divided_by_sum_team_games","referenceFact":{"id":"nfl:league:passingAttemptsPerGame","metric":"passingAttemptsPerGame","period":"NFL 2026 regular season","role":"reference","sample":158,"sourceIDs":["public-023247816fbd8a0fd6d5","public-0300b79af29915bd75bc","public-0b014c0633f03b1c558a","public-14184efb0d789f4168ce","public-2086862fce59edc1c21e","public-23b98c580c92ce3fcaa1","public-426247c27d4205cb1de6","public-49468d65a7be1ceaf964","public-5030423c1416bcd94459","public-5b90492aeafab289a5f8","public-60f3c9fe7b198cd6a49f","public-70d656dd201338d74d8a","public-73ed412e2868b11dbd5b","public-7b42aaae46d9540c53f7","public-7df48581d83f5ce67e44","public-7dff48ecf9c6b7c70e21","public-88416228e65d05571d39","public-9e0f170d375a692ec7e5","public-9ff22e4e2f84194a2cc0","public-a3be97415586adb694b5","public-a66ca5b9ae9ca14da305","public-aacfea257fde7990ceb8","public-aeba0e67f77cc6cc9c5a","public-af391d2a8f91643bccb0","public-b277ecd6976a2a73ff65","public-c238a25c7f699e5b211d","public-cd64715e2eefc4bf4ac0","public-ce77603b871c82c36ce2","public-df4ae8c8e295dd206587","public-ee96b89008319213dbb1","public-efbfc3687016625c011b","public-ffb9dcbe0e4004c068d8"],"team":"league","unit":"pass attempts per game","value":33.107594936708864,"windowEnd":"2026-10-13T20:35:26.851Z","windowStart":"2026-09-10T00:20:00.000Z"},"sport":"NFL","teamCount":32,"teamGames":158,"total":5231},{"method":"sum_totals_divided_by_sum_team_games","referenceFact":{"id":"nfl:league:pointsPerGame","metric":"pointsPerGame","period":"NFL 2026 regular season","role":"reference","sample":158,"sourceIDs":["public-023247816fbd8a0fd6d5","public-0300b79af29915bd75bc","public-0b014c0633f03b1c558a","public-14184efb0d789f4168ce","public-2086862fce59edc1c21e","public-23b98c580c92ce3fcaa1","public-426247c27d4205cb1de6","public-49468d65a7be1ceaf964","public-5030423c1416bcd94459","public-5b90492aeafab289a5f8","public-60f3c9fe7b198cd6a49f","public-70d656dd201338d74d8a","public-73ed412e2868b11dbd5b","public-7b42aaae46d9540c53f7","public-7df48581d83f5ce67e44","public-7dff48ecf9c6b7c70e21","public-88416228e65d05571d39","public-9e0f170d375a692ec7e5","public-9ff22e4e2f84194a2cc0","public-a3be97415586adb694b5","public-a66ca5b9ae9ca14da305","public-aacfea257fde7990ceb8","public-aeba0e67f77cc6cc9c5a","public-af391d2a8f91643bccb0","public-b277ecd6976a2a73ff65","public-c238a25c7f699e5b211d","public-cd64715e2eefc4bf4ac0","public-ce77603b871c82c36ce2","public-df4ae8c8e295dd206587","public-ee96b89008319213dbb1","public-efbfc3687016625c011b","public-ffb9dcbe0e4004c068d8"],"team":"league","unit":"points per game","value":23.31645569620253,"windowEnd":"2026-10-13T20:35:26.851Z","windowStart":"2026-09-10T00:20:00.000Z"},"sport":"NFL","teamCount":32,"teamGames":158,"total":3684},{"method":"sum_totals_divided_by_sum_team_games","referenceFact":{"id":"nfl:league:rushingAttemptsPerGame","metric":"rushingAttemptsPerGame","period":"NFL 2026 regular season","role":"reference","sample":158,"sourceIDs":["public-023247816fbd8a0fd6d5","public-0300b79af29915bd75bc","public-0b014c0633f03b1c558a","public-14184efb0d789f4168ce","public-2086862fce59edc1c21e","public-23b98c580c92ce3fcaa1","public-426247c27d4205cb1de6","public-49468d65a7be1ceaf964","public-5030423c1416bcd94459","public-5b90492aeafab289a5f8","public-60f3c9fe7b198cd6a49f","public-70d656dd201338d74d8a","public-73ed412e2868b11dbd5b","public-7b42aaae46d9540c53f7","public-7df48581d83f5ce67e44","public-7dff48ecf9c6b7c70e21","public-88416228e65d05571d39","public-9e0f170d375a692ec7e5","public-9ff22e4e2f84194a2cc0","public-a3be97415586adb694b5","public-a66ca5b9ae9ca14da305","public-aacfea257fde7990ceb8","public-aeba0e67f77cc6cc9c5a","public-af391d2a8f91643bccb0","public-b277ecd6976a2a73ff65","public-c238a25c7f699e5b211d","public-cd64715e2eefc4bf4ac0","public-ce77603b871c82c36ce2","public-df4ae8c8e295dd206587","public-ee96b89008319213dbb1","public-efbfc3687016625c011b","public-ffb9dcbe0e4004c068d8"],"team":"league","unit":"rush attempts per game","value":26.70253164556962,"windowEnd":"2026-10-13T20:35:26.851Z","windowStart":"2026-09-10T00:20:00.000Z"},"sport":"NFL","teamCount":32,"teamGames":158,"total":4219},{"method":"sum_totals_divided_by_sum_team_games","referenceFact":{"id":"nhl:league:goalsPerGame","metric":"goalsPerGame","period":"NHL 2027 regular season","role":"reference","sample":178,"sourceIDs":["public-0013fe0dfb7806b1e3c0","public-072bb8e2312c65aa389c","public-10441890a64e88583aee","public-2fb7fede1b61a97da3e0","public-35fd51feb84f49f023f8","public-3d23e9a0a946505f2ef5","public-4520f3fab0f00134ee89","public-4e7e681a9ef8d92735bb","public-5cfa539b5b1805aa5e41","public-5da602f2abbdc2923ee4","public-64f852c1c829771c87eb","public-65188dcee4b61b1a3be4","public-6a47a91aa82bb900a75f","public-706c1783d2b59948e15f","public-8b7be95e2e2ffb1eeb38","public-94147922eba996e81af4","public-96e2db28ba1a5a73a093","public-a147bbab85d9df3e1ba9","public-ba88e2f8e8d9ae195858","public-ca8d8a727432039bf45f","public-d83de94d9c2542b251bd","public-daa4acea82cf4bcd0a0b","public-dca3f25250223aaa048a","public-dcac47c42a37db1fe4ad","public-dcb3e0dd87c0cc47a87c","public-dd24d896c2374fb37dcb","public-ee2260c1abec17303411","public-ef5b3592357d64fe941f","public-ef6c52e8ccac3c0992db","public-efaca6fc2c6220c0ffb7","public-fbe02c37a2feaf7df50b","public-fec726d7b2462df5058e"],"team":"league","unit":"goals per game","value":3.033707865168539,"windowEnd":"2026-10-13T20:35:27.690Z","windowStart":"2026-09-29T21:00:00.000Z"},"sport":"NHL","teamCount":32,"teamGames":178,"total":540},{"method":"sum_totals_divided_by_sum_team_games","referenceFact":{"id":"nhl:league:shotsPerGame","metric":"shotsPerGame","period":"NHL 2027 regular season","role":"reference","sample":178,"sourceIDs":["public-0013fe0dfb7806b1e3c0","public-072bb8e2312c65aa389c","public-10441890a64e88583aee","public-2fb7fede1b61a97da3e0","public-35fd51feb84f49f023f8","public-3d23e9a0a946505f2ef5","public-4520f3fab0f00134ee89","public-4e7e681a9ef8d92735bb","public-5cfa539b5b1805aa5e41","public-5da602f2abbdc2923ee4","public-64f852c1c829771c87eb","public-65188dcee4b61b1a3be4","public-6a47a91aa82bb900a75f","public-706c1783d2b59948e15f","public-8b7be95e2e2ffb1eeb38","public-94147922eba996e81af4","public-96e2db28ba1a5a73a093","public-a147bbab85d9df3e1ba9","public-ba88e2f8e8d9ae195858","public-ca8d8a727432039bf45f","public-d83de94d9c2542b251bd","public-daa4acea82cf4bcd0a0b","public-dca3f25250223aaa048a","public-dcac47c42a37db1fe4ad","public-dcb3e0dd87c0cc47a87c","public-dd24d896c2374fb37dcb","public-ee2260c1abec17303411","public-ef5b3592357d64fe941f","public-ef6c52e8ccac3c0992db","public-efaca6fc2c6220c0ffb7","public-fbe02c37a2feaf7df50b","public-fec726d7b2462df5058e"],"team":"league","unit":"shots on goal per game","value":26.853932584269664,"windowEnd":"2026-10-13T20:35:27.690Z","windowStart":"2026-09-29T21:00:00.000Z"},"sport":"NHL","teamCount":32,"teamGames":178,"total":4780},{"method":"sum_totals_divided_by_sum_team_games","referenceFact":{"id":"mlb:league:runsPerGame","metric":"runsPerGame","period":"MLB 2026 regular season","role":"reference","sample":4858,"sourceIDs":["public-b3aa4ebca5723e8db13a"],"team":"league","unit":"runs per game","value":4.481062165500206,"windowEnd":"2026-10-13T20:35:28.516Z","windowStart":"2026-03-26T00:05:00.000Z"},"sport":"MLB","teamCount":30,"teamGames":4858,"total":21769}]
"""#.utf8))) ?? []
}

/// Immutable public reports recorded October 13. Authenticated reads derive age/start state;
/// original envelopes and the network contract are never rewritten.
enum SharedMatchupFrozenArchive {
 static let resourceName="FrozenMatchupReports-20261013"
 static let expectedSHA="be7c932a32c4f8658b2933c777f740c7a4801370523a849e41c827d09057d321"
 static var bundled:Bool {Bundle.main.url(forResource:resourceName,withExtension:"json") != nil}
 static let records:Result<[Data],Error> = Result {
  guard let url=Bundle.main.url(forResource:resourceName,withExtension:"json") else {throw SharedMatchupReportError.unavailable}
  let handle=try FileHandle(forReadingFrom:url);defer {try? handle.close()}
  return try load(try handle.read(upToCount:8_300_001) ?? Data())
 }
 static func load(_ bundle:Data)throws->[Data] {
  guard bundle.count<=8_300_000 else {throw SharedMatchupReportError.oversizedPayload}
  guard SHA256.hash(data:bundle).map({String(format:"%02x",$0)}).joined()==expectedSHA else {throw SharedMatchupReportError.invalidResponse}
  let encoded=try JSONDecoder().decode([String].self,from:bundle)
  guard encoded.count==91 else {throw SharedMatchupReportError.invalidResponse}
  var result=[Data](),identities=Set<SharedMatchupEvent>()
  for value in encoded {
   guard let raw=Data(base64Encoded:value),raw.count<=SharedMatchupReportValidator.maximumBytes else {throw SharedMatchupReportError.oversizedPayload}
   let saved=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
   guard let generated=saved.generatedAt else {throw SharedMatchupReportError.invalidResponse}
   let validated=try SharedMatchupReportValidator.validate(raw,event:saved.event,now:SharedMatchupReportValidator.date(generated))
   guard validated.status == .ready,identities.insert(try SharedMatchupReportValidator.normalizedEvent(saved.event)).inserted else {throw SharedMatchupReportError.invalidResponse}
   result.append(raw)
  }
  return result
 }
 static func read(_ records:[Data],event:SharedMatchupEvent,now:Date)throws->SharedMatchupReportResponse {
  let requested=try SharedMatchupReportValidator.normalizedEvent(event)
  let started=try SharedMatchupReportValidator.date(requested.startISO)<=now
  if !started {
   for raw in records {
    let saved=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
    guard try SharedMatchupReportValidator.normalizedEvent(saved.event)==requested else {continue}
    guard let generated=saved.generatedAt,let checked=saved.sourceCheckedAt else {throw SharedMatchupReportError.invalidResponse}
    let generatedDate=try SharedMatchupReportValidator.date(generated),checkedDate=try SharedMatchupReportValidator.date(checked)
    guard generatedDate<=now,checkedDate<=now else {throw SharedMatchupReportError.invalidResponse}
    let valid=try SharedMatchupReportValidator.validate(raw,event:requested,now:generatedDate)
    let reason=now.timeIntervalSince(generatedDate)>=86400 ? "generation_expired":now.timeIntervalSince(checkedDate)>=86400 ? "source_expired":nil
    return SharedMatchupReportResponse(schemaVersion:valid.schemaVersion,status:reason == nil ? .ready:.stale,reason:reason,event:valid.event,generatedAt:valid.generatedAt,sourceAsOf:valid.sourceAsOf,sourceRetrievedAt:valid.sourceRetrievedAt,sourceCheckedAt:valid.sourceCheckedAt,sourceFingerprint:valid.sourceFingerprint,versions:valid.versions,refreshing:false,report:valid.report,snapshot:valid.snapshot)
   }
  }
  return .init(schemaVersion:1,status:.missing,reason:started ? "event_started":"cache_miss",event:requested,generatedAt:nil,sourceAsOf:nil,sourceRetrievedAt:nil,sourceCheckedAt:nil,sourceFingerprint:nil,versions:.current,refreshing:false,report:nil,snapshot:nil)
 }
}
