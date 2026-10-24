import Foundation

struct SimulationAvailability: Equatable, Sendable {
    struct Observation: Equatable, Sendable, Identifiable {
        let playerID: String
        let name: String
        let status: String
        let updated: Date?
        let excluded: Bool
        var id: String { playerID }
    }
    let roster: [PlayerMatchupPlayer]
    let observations: [Observation]
    let checked: Date
    let injuryFeedUpdated: Date?
    let injuriesVerified: Bool
    var excludedIDs: Set<String> { Set(observations.filter(\.excluded).map(\.playerID)) }
    var eligible: [PlayerMatchupPlayer] { roster.filter { !excludedIDs.contains($0.id) } }
    var fingerprint: String {
        (roster.map { "\($0.id):\($0.teamID):\($0.position)" }.sorted() + observations.map { "\($0.playerID):\($0.status):\($0.excluded)" }.sorted() + [String(injuriesVerified)]).joined(separator: "|")
    }
    static func excluded(_ status: String) -> Bool {
        let lower = status.lowercased().replacingOccurrences(of: "-", with: " ")
        return ["out", "inactive", "suspension", "suspended", "injured reserve", "ir", "ltir", "pup", "reserve ret", "reserve retired", "retired", "practice squad", "waived", "released"].contains(lower)
            || lower.contains("day il") || lower.contains("injured/") || lower.contains("injured reserve") || lower.contains("injuredreserve")
    }
}

struct SimulationAvailabilitySource: Sendable {
    // Force a fresh bounded public read; do not relabel a 15-minute cached roster
    // as newly checked. Concurrent requests still share the HTTP flight.
    var read: MatchupSimulationSource.Reader = { try await PlayerHistoryService.shared.read($0, refresh: true) }
    func load(_ team: InterestTeam, now: Date? = nil) async throws -> SimulationAvailability {
        let base = "https://site.api.espn.com/apis/site/v2/sports/" + team.sport.leaguePath
        async let injuries = try? read(URL(string: base + "/injuries")!)
        let rosterData = try await read(URL(string: base + "/teams/\(team.providerID)/roster")!)
        let result = try Self.parse(rosterData, injuries: await injuries, team: team, now: now ?? Date())
        try Task.checkCancellation(); return result
    }
    static func parse(_ rosterData: Data, injuries: Data?, team: InterestTeam, now: Date) throws -> SimulationAvailability {
        guard rosterData.count <= 4 * 1024 * 1024,
              let root = try JSONSerialization.jsonObject(with: rosterData) as? [String: Any],
              let identity = root["team"] as? [String: Any], identity["id"] as? String == team.providerID,
              let rosterStamp = (root["timestamp"] as? String).flatMap(sportsDate), (-30...900).contains(now.timeIntervalSince(rosterStamp)),
              let groups = root["athletes"] as? [[String: Any]] else { throw MatchupSimulation.Failure.source }
        let roster = try PlayerMatchupSource.players(rosterData, teamID: team.providerID, teamName: team.name)
        guard !roster.isEmpty else { throw MatchupSimulation.Failure.source }
        let rows = groups.flatMap { $0["items"] as? [[String: Any]] ?? [$0] }
        var observations: [String: SimulationAvailability.Observation] = [:]
        for row in rows {
            guard let id = row["id"] as? String, let player = roster.first(where: { $0.id == id }) else { continue }
            let status = (row["status"] as? [String: Any])?["name"] as? String ?? "Unknown"
            if status.lowercased() != "active" {
                observations[id] = .init(playerID: id, name: player.name, status: status == "News" ? "Availability unconfirmed" : status, updated: nil, excluded: SimulationAvailability.excluded(status))
            }
            // The freshly fetched roster also carries availability reports.
            // Keep them when the league injury feed fails or lacks this team.
            if let reports = row["injuries"] as? [[String: Any]] {
                for report in reports {
                    guard let reported = report["status"] as? String, reported.count <= 80,
                          let date = (report["date"] as? String).flatMap(sportsDate), date <= now else { continue }
                    let blocked = SimulationAvailability.excluded(reported) || observations[id]?.excluded == true
                    if !["active", "available"].contains(reported.lowercased()) || blocked {
                        observations[id] = .init(playerID: id, name: player.name, status: reported, updated: date, excluded: blocked)
                    }
                }
            }
        }
        var stamp: Date?, verified = false
        if let injuries, injuries.count <= 4 * 1024 * 1024,
           let body = try? JSONSerialization.jsonObject(with: injuries) as? [String: Any],
           let date = (body["timestamp"] as? String).flatMap(sportsDate), (-30...900).contains(now.timeIntervalSince(date)),
           let reports = body["injuries"] as? [[String: Any]], reports.count <= 100,
           reports.filter({ String(describing: $0["id"] ?? "") == team.providerID }).count == 1,
           let report = reports.first(where: { String(describing: $0["id"] ?? "") == team.providerID }),
           let entries = report["injuries"] as? [[String: Any]], entries.count <= 150 {
            stamp = date; verified = true
            var seen = Set<String>()
            for entry in entries {
                guard let athlete = entry["athlete"] as? [String: Any], let name = athlete["displayName"] as? String,
                      let status = entry["status"] as? String, status.count <= 80,
                      let links = athlete["links"] as? [[String: Any]] else { verified = false; continue }
                let ids = Set(links.compactMap { link -> String? in
                    guard let raw = link["href"] as? String, let url = URL(string: raw), url.scheme == "https", url.host == "www.espn.com",
                          url.path.hasPrefix("/\(team.sport.rawValue.lowercased())/player/") else { return nil }
                    let parts = url.path.split(separator: "/")
                    guard let index = parts.firstIndex(of: "id"), parts.indices.contains(index + 1), InterestTeam.validProviderID(String(parts[index + 1])) else { return nil }
                    return String(parts[index + 1])
                })
                guard ids.count == 1, let id = ids.first else { verified = false; continue }
                guard let player = roster.first(where: { $0.id == id }) else { continue } // departed players never enter the simulation
                guard AnalyticsOddsMatcher.normalized(player.name) == AnalyticsOddsMatcher.normalized(name), seen.insert(id).inserted else { verified = false; continue }
                let updated = (entry["date"] as? String).flatMap(sportsDate)
                guard updated.map({ $0 <= now }) != false else { verified = false; continue }
                let blocked = SimulationAvailability.excluded(status) || observations[id]?.excluded == true
                // A current feed may retain an older injury onset date. Preserve
                // both clocks; a new fetch never changes the report's own date.
                if !["active", "available"].contains(status.lowercased()) || blocked {
                    observations[id] = .init(playerID: id, name: player.name, status: status, updated: updated, excluded: blocked)
                }
            }
        }
        return .init(roster: roster, observations: observations.values.sorted { $0.name < $1.name }, checked: now,
                     injuryFeedUpdated: stamp, injuriesVerified: verified)
    }
}

/// A run is usable only while both roster checks remain recent and unchanged.
struct SimulationAvailabilityWatch: Sendable {
    let original: [SimulationAvailability]
    var latest: [SimulationAvailability]
    var failed = false
    var changed: Bool { zip(original, latest).contains { $0.fingerprint != $1.fingerprint } }
    func usable(at now: Date) -> Bool {
        !failed && !changed && latest.count == 2 && latest.allSatisfy { (0...120).contains(now.timeIntervalSince($0.checked)) }
    }
    mutating func update(_ values: [SimulationAvailability]) { latest = values; failed = false }
}
