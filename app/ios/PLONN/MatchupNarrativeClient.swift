import Foundation
import CryptoKit

/// Authenticated reads of separately pinned public reports. This client has no
/// network transport, refresh operation, or inference capability.
struct MatchupNarrativeClient: Sendable {
    typealias Reader = @Sendable (SharedMatchupEvent, Date) throws -> MatchupNarrativeV4.Result?
    let tokenProvider: @Sendable () async throws -> String
    let reader: Reader
    let now: @Sendable () -> Date
    init(tokenProvider: @escaping @Sendable () async throws -> String,
         reader: @escaping Reader, now: @escaping @Sendable () -> Date = { Date() }) {
        self.tokenProvider = tokenProvider; self.reader = reader; self.now = now
    }
    func fetch(event: SharedMatchupEvent) async throws -> MatchupNarrativeV4.Result? {
        let event = try SharedMatchupReportValidator.normalizedEvent(event)
        try Task.checkCancellation()
        let token = try await tokenProvider()
        try Task.checkCancellation()
        guard !token.isEmpty, token.utf8.count <= 8192,
              token.utf8.allSatisfy({ (33...126).contains($0) }) else {
            throw SharedMatchupReportError.authenticationRequired
        }
        let result = try reader(event, now())
        guard result == nil || result?.event == event else { throw SharedMatchupReportError.invalidResponse }
        return result
    }
}

/// Separate from the immutable v2/v3 archive. Activation requires a reviewed
/// real-output bundle hash and trusted version pins; synthetic corpus is never loaded.
enum MatchupNarrativeArchive {
    static let resourceName = "MatchupNarrativeReports-v4"
    static let maximumBytes = 18 * 1024 * 1024
    // Populated only after independently verifying the generated artifact receipt.
    static let trustedBundleSHA256: String? = "3db2a6d6752f3dc1414ea692fd2302c1ab9caad9f52bcb81dc35d10198bf79f8"
    static let trustedVersions: [String: String]? = [
        "contract": "matchup-narrative-v4",
        "rules": "sport-dimensions-v1",
        "model": "gemma4:26b",
        "modelDigest": "08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68",
        "prompt": "16309bb514d5a6dd20c3daa2c4c319e9a4dbadd55bf964e5cef8cd8310175ebc",
        "validator": "4e84204a0e968646fa202bd2b0d737156b06aa7668622fd6e77de014aa7748ed"
    ]
    static var bundled: Bool {
        trustedBundleSHA256 != nil && trustedVersions != nil &&
        Bundle.main.url(forResource: resourceName, withExtension: "json") != nil
    }
    static func read(event: SharedMatchupEvent, now: Date) throws -> MatchupNarrativeV4.Result? {
        guard let pin = trustedBundleSHA256, let versions = trustedVersions,
              let url = Bundle.main.url(forResource: resourceName, withExtension: "json") else { return nil }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let bytes = try handle.read(upToCount: maximumBytes + 1) ?? Data()
        guard bytes.count <= maximumBytes,
              SHA256.hash(data: bytes).map({ String(format: "%02x", $0) }).joined() == pin else {
            throw SharedMatchupReportError.invalidResponse
        }
        return try read(bundle: bytes, event: event, now: now, versions: versions)
    }
    static func read(bundle: Data, event: SharedMatchupEvent, now: Date,
                     versions: [String: String], version5: Bool = false) throws -> MatchupNarrativeV4.Result? {
        guard bundle.count <= maximumBytes else {throw SharedMatchupReportError.oversizedPayload}
        var parser=RichContextJSONParser(bundle)
        let root=try parser.parse()
        try root.keys(["schemaVersion","reports"])
        guard root["schemaVersion"] == .number("1"), case .list(let rows)=root["reports"], rows.count<=94 else {
            throw SharedMatchupReportError.invalidResponse
        }
        let requested = try SharedMatchupReportValidator.normalizedEvent(event)
        var selected: Data?
        var identities = Set<SharedMatchupEvent>()
        for entry in rows {
            guard case .string(let encoded)=entry, let raw=Data(base64Encoded:encoded), raw.count<=131072 else {
                throw SharedMatchupReportError.invalidResponse
            }
            var rawParser=RichContextJSONParser(raw)
            let object=try rawParser.parse()
            let identity=try SharedMatchupReportValidator.normalizedEvent(
                JSONDecoder().decode(SharedMatchupEvent.self,from:Data(object["event"].canonical.utf8)))
            guard identities.insert(identity).inserted else { throw SharedMatchupReportError.invalidResponse }
            if identity == requested { selected = raw }
        }
        guard let selected else { return nil }
        let result = try version5 ? MatchupNarrativeV5.validate(selected, now: now, versions: versions) : MatchupNarrativeV4.validate(selected, now: now, versions: versions)
        guard result.event == requested else { throw SharedMatchupReportError.invalidResponse }
        return result
    }
}


/// Additive v5 publication gate. No synthetic fixture or unpinned report can activate it.
enum MatchupNarrativeV5Archive {
    static let resourceName = "MatchupNarrativeReports-v5"
    static let trustedBundleSHA256: String? = "52c38edaaed4b44727385cff61f15dfc77ffb24f9994fe0075f2ea7ed14baf05"
    static let trustedVersions: [String: String] = [
        "contract": "matchup-narrative-v5", "rules": "sport-and-availability-v1",
        "model": "gemma4:26b",
        "modelDigest": "08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68",
        "prompt": "1402c4da3bb661e0e7f7bda110477f170ed16b715016ddeb1684352c1773dc7a",
        "validator": "6a61690930a4849b00b40e42bc02ccc415ae559636b41b78389a9c99863bb9c7"
    ]
    static var bundled: Bool {
        trustedBundleSHA256 != nil && Bundle.main.url(forResource: resourceName, withExtension: "json") != nil
    }
    static func read(event: SharedMatchupEvent, now: Date) throws -> MatchupNarrativeV4.Result? {
        guard let pin = trustedBundleSHA256,
              let url = Bundle.main.url(forResource: resourceName, withExtension: "json") else { return nil }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let bytes = try handle.read(upToCount: MatchupNarrativeArchive.maximumBytes + 1) ?? Data()
        guard bytes.count <= MatchupNarrativeArchive.maximumBytes,
              SHA256.hash(data: bytes).map({ String(format: "%02x", $0) }).joined() == pin else {
            throw SharedMatchupReportError.invalidResponse
        }
        return try MatchupNarrativeArchive.read(bundle: bytes, event: event, now: now,
                                               versions: trustedVersions, version5: true)
    }
}


/// Rechecks a visible report without transport, authentication mutation, or generation.
enum MatchupNarrativeCurrentState {
    /// Wake at any possible current policy transition, capped at one minute for clock changes.
    /// Extra transitions are harmless; actual eligibility is always decided by the validator.
    static func nextCheck(_ result: MatchupNarrativeV4.Result, now: Date) throws -> Date {
        var parser = RichContextJSONParser(result.rawData)
        let wire = try parser.parse()
        var next = now.addingTimeInterval(60)
        let clocks: Set<String> = ["startISO", "eventStartISO", "retrievedAt", "checkedAt", "updatedAt", "reportedAt", "windowEnd"]
        func visit(_ value: RichContextJSON) {
            switch value {
            case .object(let fields):
                for (key, field) in fields {
                    if clocks.contains(key), let time = try? NarrativeInjuryContext.time(field) {
                        for offset in [0.0, 21600, 86400, 604800, 1209600] {
                            let boundary = Date(timeIntervalSince1970: time + offset)
                            if boundary > now && boundary < next { next = boundary }
                        }
                    }
                    visit(field)
                }
            case .list(let values): values.forEach(visit)
            default: break
            }
        }
        visit(wire)
        return next
    }
    static func validate(_ result: MatchupNarrativeV4.Result, now: Date) throws {
        var parser = RichContextJSONParser(result.rawData)
        let wire = try parser.parse()
        if wire["contractVersion"].string == "matchup-narrative-v5" {
            _ = try MatchupNarrativeV5.validate(result.rawData, now: now, versions: MatchupNarrativeV5Archive.trustedVersions)
        } else {
            guard let pins = MatchupNarrativeArchive.trustedVersions else { throw SharedMatchupReportError.invalidResponse }
            _ = try MatchupNarrativeV4.validate(result.rawData, now: now, versions: pins)
        }
    }
}
