import XCTest
@testable import PLONNCore

private actor FeedTransportGate {
    private var continuations: [URL: CheckedContinuation<Data, Error>] = [:]
    private var startWaiters: [(Int, CheckedContinuation<Void, Never>)] = []
    private(set) var urls: [URL] = []
    private(set) var active = 0
    private(set) var maximumActive = 0
    func read(_ url: URL) async throws -> Data {
        urls.append(url); active += 1; maximumActive = max(maximumActive, active)
        return try await withCheckedThrowingContinuation { continuation in
            continuations[url] = continuation
            let ready = startWaiters.filter { urls.count >= $0.0 }
            startWaiters.removeAll { urls.count >= $0.0 }
            ready.forEach { $0.1.resume() }
        }
    }
    func started(_ count: Int) async {
        guard urls.count < count else { return }
        await withCheckedContinuation { startWaiters.append((count, $0)) }
    }
    func finish(_ url: URL) {
        guard let continuation = continuations.removeValue(forKey: url) else { return }
        active -= 1; continuation.resume(returning: Data("{}".utf8))
    }
}

private actor FeedRequestRecorder {
    private(set) var urls: [URL] = []
    func record(_ url: URL) { urls.append(url) }
}

final class AnalyticsFeedServiceTests: XCTestCase {
    func testSourceArtworkStaysBoundToVerifiedRosterIdentity() throws {
        let photo = "https://a.espncdn.com/i/headshots/nba/players/full/123.png"
        let logo = "https://a.espncdn.com/i/teamlogos/nba/500/atl.png"
        let payload: [String: Any] = ["team": ["id": "1", "displayName": "Fixture Home", "logos": [["href": logo]]], "athletes": [["id": "123", "fullName": "Verified Player", "headshot": ["href": photo]]]]
        let rows = try AnalyticsFeedService.rosterSubjects(payload: payload, sport: "NBA", teamName: "Fixture Home", teamID: "1", event: nil, sourceURL: rosterURL, now: now)
        XCTAssertEqual(rows.first?.providerID, "123")
        XCTAssertEqual(rows.first?.imageURL, photo)
        XCTAssertEqual(rows.first?.teamLogoURL, logo)
        XCTAssertThrowsError(try AnalyticsFeedService.rosterSubjects(payload: payload, sport: "NBA", teamName: "Other Team", teamID: "2", event: nil, sourceURL: rosterURL, now: now))
    }
    func testDirectoryArtworkAndMissingOrUntrustedImages() throws {
        let logo = "https://a.espncdn.com/i/teamlogos/nba/500/atl.png"
        let payload: [String: Any] = ["sports": [["leagues": [["abbreviation": "NBA", "teams": [
            ["team": ["id": "1", "displayName": "A", "logos": [["href": logo]]]],
            ["team": ["id": "2", "displayName": "B", "logo": "https://untrusted.example/image.png"]],
            ["team": ["id": "3", "displayName": "C"]]
        ]]]]]]
        let rows = try AnalyticsFeedService.directorySubjects(payload: payload, sport: "NBA", sourceURL: "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams", now: now)
        XCTAssertEqual(rows.map(\.providerID), ["1", "2", "3"])
        XCTAssertEqual(rows[0].imageURL, logo)
        XCTAssertNil(rows[1].imageURL)
        XCTAssertNil(rows[2].imageURL)
        for raw in ["http://a.espncdn.com/i/a.png", "https://a.espncdn.com.evil.test/i/a.png", "https://user@a.espncdn.com/i/a.png", "https://a.espncdn.com/not-images/a.png"] { XCTAssertNil(AnalyticsFeedCore.imageURL(raw)) }
    }
    private let now = ISO8601DateFormatter().date(from: "2026-10-13T12:00:00Z")!
    private let scoreboardURL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    private let rosterURL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/1/roster"
    private func url(_ suffix: String) -> URL { URL(string: "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/" + suffix)! }
    private func event(_ id: String = "event-1", home: String = "Fixture Home", start: String = "2026-10-13T18:00:00Z") -> [String: Any] {
        ["id": id, "date": start, "competitions": [[
            "competitors": [["homeAway": "home", "team": ["id": "1", "displayName": home]],
                            ["homeAway": "away", "team": ["id": "2", "displayName": "Fixture Away"]]],
            "status": ["type": ["state": "pre", "completed": false, "name": "STATUS_SCHEDULED"]]
        ]]]
    }
    private func roster(_ athletes: [[String: Any]], id: String = "1", name: String = "Fixture Home") -> [String: Any] {
        ["team": ["id": id, "displayName": name], "athletes": [["items": athletes]]]
    }
    private func athlete(_ id: String, _ name: String, position: String = "PG") -> [String: Any] {
        ["id": id, "fullName": name, "position": ["abbreviation": position]]
    }

    func testTransportKeepsCancelledPermitUntilCompletionAndNeverExceedsTwo() async throws {
        let gate = FeedTransportGate()
        let reader = AnalyticsFeedPublicReader(transport: { try await gate.read($0) })
        let firstURL = url("teams/1"), secondURL = url("teams/2"), thirdURL = url("teams/3")
        let first = Task { try await reader.read(firstURL) }
        let second = Task { try await reader.read(secondURL) }
        await gate.started(2)
        first.cancel()
        let third = Task { try await reader.read(thirdURL) }
        // Let the queued actor call run while both controlled transports stay open.
        for _ in 0..<100 { await Task.yield() }
        let before = await gate.urls
        XCTAssertEqual(before.count, 2)
        await gate.finish(firstURL)
        await gate.started(3)
        let maximum = await gate.maximumActive
        XCTAssertEqual(maximum, 2)
        await gate.finish(secondURL); await gate.finish(thirdURL)
        do { _ = try await first.value; XCTFail("Cancelled caller received a result") }
        catch { XCTAssertTrue(error is CancellationError) }
        _ = try await second.value; _ = try await third.value
    }

    func testConcurrentDuplicateAndCompletedCacheUseOneActualTransport() async throws {
        let gate = FeedTransportGate()
        let reader = AnalyticsFeedPublicReader(transport: { try await gate.read($0) })
        let requested = url("teams/1")
        let first = Task { try await reader.read(requested) }
        await gate.started(1)
        let duplicate = Task { try await reader.read(requested) }
        for _ in 0..<100 { await Task.yield() }
        let pendingCalls = await gate.urls
        XCTAssertEqual(pendingCalls, [requested])
        await gate.finish(requested)
        let a = try await first.value, b = try await duplicate.value
        XCTAssertEqual(a, b)
        let cached = try await reader.read(requested)
        XCTAssertEqual(cached, a)
        let completedCalls = await gate.urls
        XCTAssertEqual(completedCalls, [requested])
    }

    func testBadHostsCredentialsAndPaidEndpointsFailBeforeTransport() async {
        let recorder = FeedRequestRecorder()
        let reader = AnalyticsFeedPublicReader(transport: { await recorder.record($0); return Data() })
        for value in ["https://api.oakas.ai/plonn/v1/analytics/odds", "https://api.the-odds-api.com/v4/sports", "https://site.api.espn.com.evil.test/teams", "http://site.api.espn.com/teams", "https://secret@site.api.espn.com/teams"] {
            do { _ = try await reader.read(URL(string: value)!); XCTFail("Unapproved URL reached transport") }
            catch { XCTAssertEqual(error as? PlayerHistoryError, .invalidURL) }
        }
        let calls = await recorder.urls
        XCTAssertTrue(calls.isEmpty)
    }

    func testScoreboardQuarantinesConflictingEventCopiesInsteadOfChoosingFirst() throws {
        for altered in [event(home: "Other Home"), event(start: "2026-10-13T19:00:00Z")] {
            for events in [[event(), altered], [altered, event()]] {
                let subjects = try AnalyticsFeedService.scoreboardSubjects(payload: ["events": events], sport: "NBA", sourceURL: scoreboardURL, now: now)
                XCTAssertTrue(subjects.isEmpty)
            }
        }
        let subjects = try AnalyticsFeedService.scoreboardSubjects(payload: ["events": [event(), event()]], sport: "NBA", sourceURL: scoreboardURL, now: now)
        XCTAssertEqual(Set(subjects.map(\.name)), Set(["Fixture Home", "Fixture Away"]))
        XCTAssertTrue(subjects.allSatisfy { $0.event?.providerID == "event-1" })
    }

    func testRosterConflictsAreQuarantinedBeforeTwoSubjectPrefix() throws {
        let rows = [athlete("10", "A First"), athlete("11", "B Second"), athlete("10", "Z Different"),
                    athlete("12", "C Duplicate"), athlete("13", "C Duplicate"), athlete("14", "D Verified"), athlete("15", "E Verified")]
        for ordered in [rows, Array(rows.reversed())] {
            let subjects = try AnalyticsFeedService.rosterSubjects(payload: roster(ordered), sport: "NBA", teamName: "Fixture Home", teamID: "1", event: nil, sourceURL: rosterURL, now: now)
            XCTAssertEqual(subjects.map(\.name), ["B Second", "D Verified"])
            XCTAssertEqual(subjects.map(\.providerID), ["11", "14"])
        }
    }

    func testWrongTeamRosterRejectsIdentityAndUnsupportedPositionsInventNoNames() throws {
        for payload in [roster([athlete("10", "Fixture Player")], id: "9"), roster([athlete("10", "Fixture Player")], name: "Other Team")] {
            XCTAssertThrowsError(try AnalyticsFeedService.rosterSubjects(payload: payload, sport: "NBA", teamName: "Fixture Home", teamID: "1", event: nil, sourceURL: rosterURL, now: now)) {
                XCTAssertEqual($0 as? PlayerHistoryError, .invalidPayload)
            }
        }
        let subjects = try AnalyticsFeedService.rosterSubjects(payload: roster([athlete("10", "Returned Defender", position: "LB")]), sport: "NFL", teamName: "Fixture Home", teamID: "1", event: nil, sourceURL: rosterURL, now: now)
        XCTAssertTrue(subjects.isEmpty)
    }

    func testInitialDiscoveryUsesAtMostTenPublicReadsAndOnlyReturnedSubjects() async throws {
        let recorder = FeedRequestRecorder()
        let reader = AnalyticsFeedPublicReader(transport: { requested in
            await recorder.record(requested)
            let components = requested.path.split(separator: "/").map(String.init)
            let slug = components[components.count - (requested.path.hasSuffix("/roster") ? 4 : 2)]
            let sport = ["nba": "NBA", "mlb": "MLB", "nfl": "NFL", "nhl": "NHL"][slug] ?? ""
            let payload: [String: Any]
            if requested.path.hasSuffix("/scoreboard") { payload = ["events": []] }
            else if requested.path.hasSuffix("/roster") {
                payload = ["team": ["id": "1", "displayName": sport + " Returned Team"], "athletes": []]
            } else {
                payload = ["sports": [["leagues": [["abbreviation": sport, "teams": [["team": ["id": "1", "displayName": sport + " Returned Team"]]]]]]]]
            }
            return try JSONSerialization.data(withJSONObject: payload)
        })
        let service = AnalyticsFeedService(reader: reader)
        let result = await service.load(snapshots: [], now: now)
        let calls = await recorder.urls
        XCTAssertEqual(calls.count, 10)
        XCTAssertEqual(calls.filter { $0.path.hasSuffix("/scoreboard") }.count, 4)
        XCTAssertEqual(calls.filter { $0.path.hasSuffix("/teams") }.count, 4)
        XCTAssertEqual(calls.filter { $0.path.hasSuffix("/roster") }.count, 2)
        XCTAssertTrue(calls.allSatisfy { $0.host == "site.api.espn.com" })
        XCTAssertEqual(Set(result.candidates.map { $0.market.name }), Set(["NBA Returned Team", "MLB Returned Team", "NFL Returned Team", "NHL Returned Team"]))
        XCTAssertTrue(result.candidates.allSatisfy { $0.subject == .team && $0.quotes.isEmpty })
        _ = await service.load(snapshots: [], now: now)
        let cachedCalls = await recorder.urls
        XCTAssertEqual(cachedCalls.count, 10)
    }
    private func identityCandidate(sport: String = "NBA", proof: Bool = true) throws -> AnalyticsFeedCandidate {
        let subject = AnalyticsFeedVerifiedSubject(sport: sport, subject: .player, name: "Fixture Player", providerID: "77",
            teamName: "Fixture Home", teamID: "1", marketKeys: [sport == "MLB" ? "batter_hits" : "player_points"],
            sourceURL: "https://site.api.espn.com/apis/site/v2/sports/" + (sport == "MLB" ? "baseball/mlb" : "basketball/nba") + "/teams/1/roster", verifiedAt: now)
        let candidate = try XCTUnwrap(AnalyticsFeedCore.build(snapshots: [], verifiedSubjects: [subject], now: now).candidates.first)
        return AnalyticsFeedCandidate(id: candidate.id, subject: candidate.subject, market: candidate.market, quotes: candidate.quotes, verifiedSubject: proof ? candidate.verifiedSubject : nil)
    }
    private func identityHistory(source: String = "ESPN", name: String? = "Fixture Player", id: String? = "77") -> PlayerHistoryResult {
        PlayerHistoryResult(games: [], opponentGames: [], context: nil, source: source, warnings: [], availability: "Offline identity fixture", playerName: name, playerID: id)
    }

    func testHistoricalPlayerIdentityRejectsMismatchedOrMissingIDAndName() throws {
        let candidate = try identityCandidate()
        XCTAssertTrue(AnalyticsFeedService.acceptsPlayerHistory(candidate: candidate, history: identityHistory(name: " fixture PLAYER ")))
        for history in [identityHistory(id: "78"), identityHistory(id: nil), identityHistory(id: ""),
                        identityHistory(name: "Other Player"), identityHistory(name: nil)] {
            XCTAssertFalse(AnalyticsFeedService.acceptsPlayerHistory(candidate: candidate, history: history))
        }
    }

    func testSameNumericIDAcrossProvidersDoesNotAuthorizeRosterOrCurrentContext() throws {
        let mlbRosterCandidate = try identityCandidate(sport: "MLB")
        let mlbHistory = identityHistory(source: "MLB Stats API")
        XCTAssertFalse(AnalyticsFeedService.acceptsPlayerHistory(candidate: mlbRosterCandidate, history: mlbHistory),
            "ESPN roster ID 77 is not MLB Stats API athlete ID 77")
        XCTAssertFalse(AnalyticsFeedService.acceptsPlayerHistory(candidate: try identityCandidate(), history: mlbHistory),
            "MLB history cannot authorize another sport even when numeric IDs match")
        let historicalOnly = try identityCandidate(sport: "MLB", proof: false)
        XCTAssertTrue(AnalyticsFeedService.acceptsPlayerHistory(candidate: historicalOnly, history: mlbHistory),
            "A verified historical identity may populate statistics without current roster proof")
        let statistics = AnalyticsHistoryResult(summary: nil, rows: [], observations: [], source: mlbHistory.source,
            warnings: [], availability: "Historical identity only", verifiedPlayerID: mlbHistory.playerID)
        XCTAssertTrue(AnalyticsFeedCore.eligibleComparisons(candidate: historicalOnly, history: statistics, now: now).isEmpty,
            "Historical identity alone must not establish current roster/event comparisons")
    }

}
