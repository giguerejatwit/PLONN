import Foundation

/// Stable provider identities, rather than team names, are stored on the account.
enum InterestSport: String, Codable, CaseIterable, Identifiable, Sendable {
    case nfl = "NFL", nba = "NBA", mlb = "MLB", nhl = "NHL"
    var id: String { rawValue }
    var leaguePath: String {
        switch self {
        case .nfl: return "football/nfl"
        case .nba: return "basketball/nba"
        case .mlb: return "baseball/mlb"
        case .nhl: return "hockey/nhl"
        }
    }
}

struct InterestTeam: Codable, Equatable, Hashable, Identifiable, Sendable {
    let sport: InterestSport
    let providerID: String
    let name: String
    let abbreviation: String
    var id: String { sport.rawValue + ":espn:" + providerID }
    var isValid: Bool {
        Self.validProviderID(providerID) && !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        && name.count <= 100 && abbreviation.count <= 12
        && !name.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains)
        && !abbreviation.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains)
    }
    static func validProviderID(_ value: String) -> Bool {
        !value.isEmpty && value.count <= 10 && value.first != "0"
        && value.utf8.allSatisfy { (48...57).contains($0) }
    }
}

struct InterestNotifications: Codable, Equatable, Sendable {
    var modelReady = false
    var gameReminders = false
    var injuryLineup = false
    var dailyDigest = false
    var digestMinute = 18 * 60
    var quietHoursEnabled = true
    var quietStartMinute = 22 * 60
    var quietEndMinute = 8 * 60
    var timeZoneID = TimeZone.current.identifier
    var wantsAlerts: Bool { modelReady || gameReminders || injuryLineup || dailyDigest }
    var isValid: Bool {
        (0..<1440).contains(digestMinute) && (0..<1440).contains(quietStartMinute)
        && (0..<1440).contains(quietEndMinute) && (!quietHoursEnabled || quietStartMinute != quietEndMinute)
        && TimeZone(identifier: timeZoneID) != nil
    }

    /// Quiet hours apply in the account's explicit time zone, including DST changes.
    /// The future sender must drop an expired game reminder, not deliver it after kickoff.
    func isQuiet(at date: Date) -> Bool {
        guard quietHoursEnabled, isValid, let zone = TimeZone(identifier: timeZoneID) else { return false }
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = zone
        let minute = calendar.component(.hour, from: date) * 60 + calendar.component(.minute, from: date)
        if quietStartMinute < quietEndMinute { return minute >= quietStartMinute && minute < quietEndMinute }
        return minute >= quietStartMinute || minute < quietEndMinute
    }
}

struct UserInterests: Codable, Equatable, Sendable {
    static let metadataKey = "plonn_interests_v1"
    static let onboardingKey = "plonn_interests_onboarding"
    static let onboardingPending = "pending_v1"
    static let maximumBytes = 32_768
    var version = 1
    var onboardingCompleted = false
    var sports: [InterestSport] = []
    var teams: [InterestTeam] = []
    var notifications = InterestNotifications()

    var isValid: Bool {
        version == 1 && sports.count <= InterestSport.allCases.count
        && Set(sports).count == sports.count && teams.count <= 160
        && Set(teams.map(\.id)).count == teams.count && teams.allSatisfy(\.isValid)
        && notifications.isValid
    }
    func encoded() throws -> String {
        guard isValid else { throw InterestsError.invalidPreferences }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(self)
        guard bytes.count <= Self.maximumBytes, let value = String(data: bytes, encoding: .utf8) else { throw InterestsError.invalidPreferences }
        return value
    }
    static func decode(_ value: String?) -> UserInterests? {
        guard let value, value.utf8.count <= maximumBytes, let data = value.data(using: .utf8),
              let preferences = try? JSONDecoder().decode(Self.self, from: data), preferences.isValid else { return nil }
        return preferences
    }
    mutating func toggle(_ sport: InterestSport) {
        if sports.contains(sport) { sports.removeAll { $0 == sport } } else { sports.append(sport) }
        sports.sort { $0.rawValue < $1.rawValue }
    }
    mutating func toggle(_ team: InterestTeam) {
        guard team.isValid else { return }
        if teams.contains(where: { $0.id == team.id }) { teams.removeAll { $0.id == team.id } }
        else if teams.count < 160 { teams.append(team) }
        teams.sort { $0.id < $1.id }
    }
}

enum InterestsError: Error { case invalidPreferences, invalidDirectory, wrongAccount }

/// Cancels stale saves and clears all private view state when the authenticated owner changes.
struct InterestsAccountState: Equatable {
    struct Operation: Equatable { let owner: UUID; let generation: UUID }
    private(set) var owner: UUID?
    private(set) var generation = UUID()
    private(set) var preferences = UserInterests()
    private(set) var invalidMetadata = false
    private(set) var onboardingDeferred = false
    mutating func restore(owner: UUID?, isAnonymous: Bool?, metadata: String?) {
        let next = isAnonymous == false ? owner : nil
        if next != self.owner {
            self.owner = next; generation = UUID(); onboardingDeferred = false
        }
        guard next != nil else { preferences = UserInterests(); invalidMetadata = false; return }
        preferences = UserInterests.decode(metadata) ?? UserInterests()
        invalidMetadata = metadata != nil && UserInterests.decode(metadata) == nil
    }
    func operation() -> Operation? { owner.map { Operation(owner: $0, generation: generation) } }
    func accepts(_ operation: Operation) -> Bool { owner == operation.owner && generation == operation.generation }
    mutating func deferOnboarding() { onboardingDeferred = true }
    func shouldOnboard(pending: String?, createdAt: Date?, now: Date = Date()) -> Bool {
        guard owner != nil, !invalidMetadata, !onboardingDeferred, !preferences.onboardingCompleted else { return false }
        if pending == UserInterests.onboardingPending { return true }
        guard let createdAt else { return false }
        // Social sign-in creates an account without passing our email-signup metadata.
        return createdAt <= now && now.timeIntervalSince(createdAt) <= 10 * 60
    }
}

/// Uses the same cached public reader as Analytics. No odds request, image fetch or account credential.
struct InterestsTeamDirectory: Sendable {
    typealias Reader = @Sendable (URL) async throws -> Data
    var reader: Reader = { try await PlayerHistoryService.shared.read($0) }
    func load(_ sport: InterestSport) async throws -> [InterestTeam] {
        let url = URL(string: "https://site.api.espn.com/apis/site/v2/sports/\(sport.leaguePath)/teams?limit=1000")!
        let data = try await reader(url)
        try Task.checkCancellation()
        return try Self.parse(data, sport: sport)
    }
    static func parse(_ data: Data, sport: InterestSport) throws -> [InterestTeam] {
        guard data.count <= 2 * 1024 * 1024,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sports = object["sports"] as? [[String: Any]] else { throw InterestsError.invalidDirectory }
        let leagues = sports.flatMap { $0["leagues"] as? [[String: Any]] ?? [] }.filter {
            ($0["abbreviation"] as? String)?.uppercased() == sport.rawValue
            && ($0["slug"] as? String)?.lowercased() == sport.rawValue.lowercased()
        }
        guard leagues.count == 1, let rows = leagues[0]["teams"] as? [[String: Any]], rows.count <= 160 else { throw InterestsError.invalidDirectory }
        var byID: [String: InterestTeam] = [:]
        for row in rows {
            guard let team = row["team"] as? [String: Any], let id = team["id"] as? String,
                  let name = team["displayName"] as? String, let abbreviation = team["abbreviation"] as? String else { throw InterestsError.invalidDirectory }
            if team["isActive"] as? Bool == false || team["isAllStar"] as? Bool == true { continue }
            let item = InterestTeam(sport: sport, providerID: id, name: name, abbreviation: abbreviation)
            guard item.isValid else { throw InterestsError.invalidDirectory }
            if let previous = byID[item.id], previous != item { throw InterestsError.invalidDirectory }
            byID[item.id] = item
        }
        guard !byID.isEmpty else { throw InterestsError.invalidDirectory }
        return byID.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }
}
