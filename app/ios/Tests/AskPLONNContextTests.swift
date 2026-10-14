import XCTest
@testable import PLONNCore

final class AskPLONNContextTests: XCTestCase {
    let now = sportsDate("2026-10-14T12:00:00Z")!
    var calendar: Calendar { var value = Calendar(identifier: .gregorian); value.timeZone = TimeZone(secondsFromGMT: 0)!; return value }
    let names = ["NFL": ("New England Patriots", "New York Jets"), "NBA": ("Boston Celtics", "New York Knicks"), "NHL": ("Boston Bruins", "New York Rangers"), "MLB": ("New York Yankees", "Los Angeles Dodgers")]
    func event(_ sport: String, id: String = "1", start: String = "2026-10-14T20:00:00Z") -> SharedMatchupEvent {
        .init(sport: sport, eventID: "espn:" + id, homeTeam: names[sport]!.0, awayTeam: names[sport]!.1, startISO: start)
    }
    func catalog(_ events: [SharedMatchupEvent], complete: Set<String> = Set(AskPLONNContextResolver.sports)) -> AskPLONNContextCatalog {
        .init(events: events, teams: events.flatMap { event in [event.homeTeam, event.awayTeam].enumerated().map { index, name in
            .init(sport: event.sport, subject: .team, name: name, providerID: String(index + 1), marketKeys: ["team_totals"], sourceURL: "https://site.api.espn.com/apis/site/v2/sports/teams", verifiedAt: now)
        } }, completeSports: complete)
    }
    func resolved(_ result: AskPLONNContextResolution, file: StaticString = #filePath, line: UInt = #line) -> AskPLONNContextTarget? {
        guard case .resolved(let target) = result else { XCTFail("Expected resolved context", file: file, line: line); return nil }; return target
    }
    func choices(_ result: AskPLONNContextResolution, file: StaticString = #filePath, line: UInt = #line) -> [AskPLONNContextTarget] {
        guard case .clarification(_, let targets) = result else { XCTFail("Expected clarification", file: file, line: line); return [] }; return targets
    }
    func testBuild9QuestionDiscoversSingleGameAcrossAllFourSports() {
        let all = AskPLONNContextResolver.sports.map { event($0) }
        for sport in AskPLONNContextResolver.sports {
            let question = "what can we expect from the \(sport) game today? High strikeouts?"
            XCTAssertTrue(AskPLONNContextResolver.shouldDiscover(question))
            XCTAssertEqual(resolved(AskPLONNContextResolver.resolve(question: question, catalog: catalog(all), now: now, calendar: calendar)), .game(event(sport)))
        }
    }
    func testDoubleheaderRequiresChoiceEvenWithBothTeams() {
        let games = [event("MLB", id: "1"), event("MLB", id: "2", start: "2026-10-14T23:00:00Z")]
        let result = AskPLONNContextResolver.resolve(question: "Yankees vs Dodgers MLB today", catalog: catalog(games), now: now, calendar: calendar)
        XCTAssertEqual(choices(result), games.map(AskPLONNContextTarget.game))
    }
    func testGenericSeveralGamesDoesNotChooseFirst() {
        let other = SharedMatchupEvent(sport: "MLB", eventID: "espn:2", homeTeam: "Seattle Mariners", awayTeam: "Cleveland Guardians", startISO: event("MLB").startISO)
        XCTAssertEqual(choices(AskPLONNContextResolver.resolve(question: "MLB game today", catalog: catalog([event("MLB"), other]), now: now, calendar: calendar)).count, 2)
    }
    func testUnknownTeamAndOneVerifiedOpponentNeverSelectUnrelatedGame() {
        for question in ["Mars MLB game today", "Yankees vs Mars MLB today", "what about Seattle MLB game today"] {
            XCTAssertTrue(choices(AskPLONNContextResolver.resolve(question: question, catalog: catalog([event("MLB")]), now: now, calendar: calendar)).isEmpty)
        }
    }
    func testIncompleteScheduleNeverClaimsUniqueGameOrTeam() {
        let partial = catalog([event("NFL")], complete: [])
        XCTAssertTrue(choices(AskPLONNContextResolver.resolve(question: "NFL game today", catalog: partial, now: now, calendar: calendar)).isEmpty)
        XCTAssertTrue(choices(AskPLONNContextResolver.resolve(question: "Jets recent stats", catalog: partial, now: now, calendar: calendar)).isEmpty)
    }
    func testSharedNicknameAcrossSportsNeedsClarification() {
        let hockey = SharedMatchupEvent(sport: "NHL", eventID: "espn:2", homeTeam: "Winnipeg Jets", awayTeam: "Boston Bruins", startISO: event("NFL").startISO)
        let result = AskPLONNContextResolver.resolve(question: "Jets game today", catalog: catalog([event("NFL"), hockey]), now: now, calendar: calendar)
        XCTAssertEqual(choices(result).count, 2)
    }
    func testTeamResearchAvailableWithoutGameAttachment() {
        for sport in AskPLONNContextResolver.sports {
            let result = AskPLONNContextResolver.resolve(question: "\(names[sport]!.0) recent stats \(sport)", catalog: catalog([event(sport)]), now: now, calendar: calendar)
            XCTAssertEqual(resolved(result), .team(sport: sport, name: names[sport]!.0))
        }
    }
    func testConflictingEventCopiesAreQuarantined() {
        let games = [event("MLB"), event("MLB", start: "2026-10-14T21:00:00Z")]
        XCTAssertTrue(choices(AskPLONNContextResolver.resolve(question: "MLB game today", catalog: catalog(games), now: now, calendar: calendar)).isEmpty)
    }
    func testDatesUseUserCalendarAndNeverGuessUnsupportedDates() {
        var pacific = calendar; pacific.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        let late = event("MLB", start: "2026-10-15T02:00:00Z")
        XCTAssertEqual(resolved(AskPLONNContextResolver.resolve(question: "MLB tonight", catalog: catalog([late]), now: now, calendar: pacific)), .game(late))
        XCTAssertTrue(choices(AskPLONNContextResolver.resolve(question: "MLB tonight", catalog: catalog([late]), now: now, calendar: calendar)).isEmpty)
        for query in ["MLB next Friday", "MLB October 20", "MLB tomorrow and today", "MLB 2026-99-42"] { XCTAssertNil(AskPLONNContextResolver.day(query, now: now, calendar: calendar)) }
    }
    func testGeneralConceptsDoNotStartScheduleDiscovery() {
        for question in ["How do odds work?", "What is MLB?", "Explain NFL implied probability"] { XCTAssertFalse(AskPLONNContextResolver.shouldDiscover(question), question) }
    }
    func testNamedPlayersAcrossFourSportsAreSearchQueriesThenVerifiedByHistory() {
        for (sport, name, stat, words) in [("MLB", "Aaron Judge", "batter_hits", "hits"), ("NBA", "Jayson Tatum", "player_rebounds", "rebounds"), ("NFL", "Josh Allen", "player_pass_yds", "passing"), ("NHL", "Auston Matthews", "player_shots_on_goal", "shots")] {
            let result = AskPLONNContextResolver.namedPlayer("Show me \(name) \(words) \(sport)", markets: [])
            XCTAssertEqual(result.flatMap { resolved($0) }, .player(sport: sport, name: name, stat: stat))
        }
    }
    func testAvailablePlayerContextUsesFullNameAndDoesNotGuessSport() {
        let market = AskPLONNContextService.market(sport: "MLB", name: "Aaron Judge", stat: "batter_hits", player: true)
        XCTAssertEqual(AskPLONNContextResolver.namedPlayer("Aaron Judge hits", markets: [market]).flatMap { resolved($0) }, .player(sport: "MLB", name: "Aaron Judge", stat: "batter_hits"))
        XCTAssertNil(AskPLONNContextResolver.namedPlayer("Judge hits", markets: [market]))
        XCTAssertNil(AskPLONNContextResolver.namedPlayer("Unknown Person hits", markets: []))
    }
    func testFollowupDoesNotReuseContextForNewSubject() {
        XCTAssertTrue(AskPLONNContextService.safeFollowup("What about strikeouts?"))
        XCTAssertFalse(AskPLONNContextService.safeFollowup("What about the Dodgers?"))
        XCTAssertFalse(AskPLONNContextService.safeFollowup("What about NBA points?"))
    }
    func testScoreboardParserBindsESPNIdentityAndRejectsMalformedOrTruncatedSlate() throws {
        let payload: [String: Any] = ["events": [["id": "123", "date": "2026-10-14T20:00Z", "status": ["type": ["state": "pre", "name": "STATUS_SCHEDULED"]], "competitions": [["competitors": [["homeAway": "home", "team": ["id": "1", "displayName": "New York Yankees"]], ["homeAway": "away", "team": ["id": "2", "displayName": "Los Angeles Dodgers"]]]]]]]]
        let result = try AskPLONNContextService.scoreboard(payload, sport: "MLB", source: "https://site.api.espn.com/scoreboard", now: now)
        XCTAssertEqual(result.events.first?.eventID, "espn:123")
        XCTAssertEqual(result.events.first?.startISO, "2026-10-14T20:00:00.000Z")
        XCTAssertThrowsError(try AskPLONNContextService.scoreboard(["events": [[:]]], sport: "MLB", source: "", now: now))
        XCTAssertThrowsError(try AskPLONNContextService.scoreboard(["events": Array(repeating: [:], count: 100)], sport: "MLB", source: "", now: now))
    }
    func frozen() throws -> [Data] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("PLONN/FrozenMatchupReports-20261013.json")
        return try SharedMatchupFrozenArchive.load(Data(contentsOf: url))
    }
    func testAll91FrozenReportsProjectAtTheirOriginalValidTimeWithoutMutation() throws {
        let records = try frozen(), original = records
        var sports = Set<String>()
        for raw in records {
            let report = try JSONDecoder().decode(SharedMatchupReportResponse.self, from: raw)
            let now = try SharedMatchupReportValidator.date(XCTUnwrap(report.generatedAt))
            let facts = try AskPLONNContextService.reportEvidence(report, event: report.event, question: "What can we expect?", now: now)
            XCTAssertTrue(facts.contains { $0.availability == .available }, report.event.eventID)
            XCTAssertNoThrow(try AskPLONNCore.boundedRequest(.init(question: "What can we expect?", intent: .recentForm, evidence: facts)))
            sports.insert(report.event.sport)
        }
        XCTAssertEqual(sports, Set(AskPLONNContextResolver.sports)); XCTAssertEqual(records, original)
    }
    func testExpiredStartedAndMismatchedReportsCannotSupplyCurrentFacts() throws {
        let report = try JSONDecoder().decode(SharedMatchupReportResponse.self, from: XCTUnwrap(frozen().first))
        let generated = try SharedMatchupReportValidator.date(XCTUnwrap(report.generatedAt))
        let expired = try AskPLONNContextService.reportEvidence(report, event: report.event, question: "game", now: generated.addingTimeInterval(86400))
        XCTAssertFalse(expired.contains { $0.availability == .available })
        let started = try AskPLONNContextService.reportEvidence(report, event: report.event, question: "game", now: SharedMatchupReportValidator.date(report.event.startISO))
        XCTAssertFalse(started.contains { $0.availability == .available })
        XCTAssertThrowsError(try AskPLONNContextService.reportEvidence(report, event: event("MLB"), question: "game", now: generated))
    }
    func testMissingStrikeoutsNeverUsesRunsAsStrikeoutEvidence() throws {
        let report = try XCTUnwrap(frozen().compactMap { try? JSONDecoder().decode(SharedMatchupReportResponse.self, from: $0) }.first { $0.event.sport == "MLB" })
        let facts = try AskPLONNContextService.reportEvidence(report, event: report.event, question: "High strikeouts?", now: SharedMatchupReportValidator.date(XCTUnwrap(report.generatedAt)))
        XCTAssertFalse(facts.contains { $0.availability == .available })
        XCTAssertTrue(facts.contains { $0.text.contains("No high/low") })
    }
    func testChosenPlayerLoadsOnlyVerifiedPlayerHistoryWithoutScheduleOrReport() async throws {
        let timestamp = now
        let service = AskPLONNContextService(read: { _ in XCTFail("No schedule needed"); throw PlayerHistoryError.invalidPayload }, player: { market, _ in
            let game = PlayerGame(id: "prior", date: timestamp.addingTimeInterval(-86400 * 3), opponent: "Verified Opponent", value: 3, workload: "4 AB", sport: market.sport, playerID: "99", playerTeamID: "1", opponentID: "2", marketKey: market.market, season: 2026)
            return .init(games: [game], opponentGames: [], context: nil, source: "MLB Stats API", warnings: [], availability: "", playerName: market.name, playerID: "99", sourceURLs: ["https://statsapi.mlb.com/api/v1/people/99/stats"])
        }, team: { _, _ in XCTFail("No team history needed"); return .init(summary: nil, rows: [], observations: [], source: "", warnings: [], availability: "") })
        let result = try await service.discover(question: "Show me Aaron Judge hits MLB", markets: [AskPLONNContextService.market(sport: "MLB", name: "Aaron Judge", stat: "batter_hits", player: true)], now: now, calendar: calendar)
        XCTAssertEqual(result.target, .player(sport: "MLB", name: "Aaron Judge", stat: "batter_hits"))
        XCTAssertEqual(result.evidence.first?.availability, .available)
        XCTAssertTrue(result.evidence.first?.text.contains("3.0") == true)
    }
    func testUnavailablePlayerIdentityDoesNotBecomeEvidence() async throws {
        let service = AskPLONNContextService(read: { _ in throw PlayerHistoryError.invalidPayload }, player: { _, _ in .init(games: [], opponentGames: [], context: nil, source: "ESPN", warnings: [], availability: "Player identity is ambiguous.") })
        let result = try await service.discover(question: "Show me Alex Smith passing NFL", markets: [], now: now, calendar: calendar)
        XCTAssertFalse(result.evidence.contains { $0.availability == .available })
    }
    func testTwoWordTeamIsNotMisreadAsPlayerAndUnsupportedStatStaysMissing() async throws {
        let service = AskPLONNContextService(read: { url in
            if url.path.hasSuffix("/scoreboard") { return Data(#"{"events":[]}"#.utf8) }
            return Data(#"{"sports":[{"leagues":[{"abbreviation":"NHL","teams":[{"team":{"id":"1","displayName":"Boston Bruins"}}]}]}]}"#.utf8)
        }, player: { _, _ in
            XCTFail("A sourced team must not be searched as a player")
            return .init(games: [], opponentGames: [], context: nil, source: "", warnings: [], availability: "")
        }, team: { _, _ in
            .init(summary: nil, rows: [], observations: [], source: "ESPN", warnings: [], availability: "No records")
        })
        let result = try await service.discover(question: "Boston Bruins shots NHL", markets: [], now: now, calendar: calendar)
        XCTAssertEqual(result.target, .team(sport: "NHL", name: "Boston Bruins"))
        XCTAssertTrue(result.evidence.contains { $0.id == "requested-stat-gap" })
        XCTAssertFalse(result.evidence.contains { $0.availability == .available })
    }
    func testCancellationDoesNotReadPublicSources() async {
        let service = AskPLONNContextService(read: { _ in XCTFail("Cancelled request read a source"); throw PlayerHistoryError.invalidPayload })
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await service.discover(question: "MLB today", markets: [], now: now, calendar: calendar)
        }
        do { _ = try await task.value; XCTFail("Expected cancellation") } catch { XCTAssertTrue(error is CancellationError) }
    }
}
