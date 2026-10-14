import XCTest
@testable import PLONNCore

private actor RecordedFeedCalls {
    var reads: [(String, NBARecordedOddsPhase?, String?)] = []
    var urls: [URL] = []
    func read(_ sport: String, _ phase: NBARecordedOddsPhase?, _ event: String?) { reads.append((sport, phase, event)) }
    func publicRead(_ url: URL) { urls.append(url) }
}

final class AnalyticsRecordedFeedTests: XCTestCase {
    private let now = sportsDate("2026-10-14T17:00:00Z")!
    private func stamp(_ date: Date) -> String { ISO8601DateFormatter().string(from: date) }
    private func event(_ sport: String) -> AnalyticsEvent {
        AnalyticsEvent(sport: sport, providerID: "schedule-" + sport, home: sport + " Fixture Home", away: sport + " Fixture Away", start: sportsDate("2026-10-16T00:00:00Z")!, homeID: "1", awayID: "2", seasonType: sport == "NBA" ? 1 : nil)
    }
    private func quote(_ sport: String, player: Bool = false) -> Market {
        let key = player ? ["NFL":"player_pass_yds", "NBA":"player_points", "NHL":"player_shots_on_goal", "MLB":"batter_hits"][sport]! : "spreads"
        return Market(id: sport + (player ? "-player" : "-team"), event: "recorded-" + sport, sport: sport, kind: player ? "prop" : "team", name: player ? sport + " Fixture Player" : event(sport).home, opponent: event(sport).away, label: "Full game", aliases: "", p: nil, price: -110, market: key, book: "Fixture Book", quotes: [], updated: stamp(now.addingTimeInterval(-3600)), mode: "live", line: player ? 10.5 : -2.5, eventLabel: event(sport).away + " at " + event(sport).home, commence: stamp(event(sport).start.addingTimeInterval(sport == "NHL" ? 600 : 0)), side: player ? "Over" : event(sport).home)
    }
    private func proof(_ sport: String, player: Bool = false) -> AnalyticsFeedVerifiedSubject {
        let path = sport == "MLB" ? "https://statsapi.mlb.com/api/v1" : "https://site.api.espn.com/apis/site/v2/sports/" + PlayerHistoryService.leaguePath(sport)
        return AnalyticsFeedVerifiedSubject(sport: sport, subject: player ? .player : .team, name: quote(sport, player: player).name, providerID: player ? "77" : "1", teamName: player ? event(sport).home : nil, teamID: player ? "1" : nil, marketKeys: [quote(sport, player: player).market], event: event(sport), sourceURL: path + (player ? "/teams/1/roster" : sport == "MLB" ? "/schedule" : "/scoreboard"), verifiedAt: now)
    }
    private func candidate(_ sport: String, rows: [Market]? = nil, player: Bool = false, recorded: Bool = true) throws -> AnalyticsFeedCandidate {
        try XCTUnwrap(AnalyticsFeedCore.build(snapshots: rows ?? [quote(sport, player: player)], verifiedSubjects: [proof(sport, player: player)], now: now, recordedSnapshot: recorded).candidates.first { $0.market.market == (rows?.first?.market ?? quote(sport, player: player).market) })
    }
    private func history(_ candidate: AnalyticsFeedCandidate) -> AnalyticsHistoryResult {
        var value = AnalyticsHistoryEngine.statistics(market: candidate.market, observations: [AnalyticsObservation(id: "prior-final", date: now.addingTimeInterval(-86400), opponent: "Fixture Opponent", value: 20, scored: 100, allowed: 95)], now: now)
        value.verifiedPlayerID = candidate.subject == .player ? "77" : nil
        return value
    }
    func testDatedRecordedOddsRemainEligibleAcrossAllSportsWithoutBecomingLive() throws {
        for sport in ["NFL", "NBA", "NHL", "MLB"] {
            let selected = try candidate(sport)
            XCTAssertEqual(AnalyticsFeedCore.bettingFeedQuotes(candidate: selected, history: history(selected), now: now), [quote(sport)], sport)
            let target = AnalyticsStatsTarget(sport: sport, kind: "team", name: event(sport).home, marketKey: "spreads", teamName: event(sport).home)
            let match = AnalyticsOddsMatcher.matchCached(target: target, event: event(sport), markets: selected.quotes, now: now, recordedSnapshot: true)
            XCTAssertEqual(match.freshness, .stale); XCTAssertNil(match.fetchedAt); XCTAssertNil(match.expiresAt)
            XCTAssertNil(AnalyticsOddsMatcher.request(target: target, event: event(sport), now: now), "Scoped live requests retain same-day restriction")
            let retained = try candidate(sport, recorded: false)
            XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: retained, history: history(retained), now: now).isEmpty)
        }
    }
    func testRecordedPathStillRejectsInvalidQuoteHistoryAndSubjectEvidence() throws {
        for sport in ["NFL", "NBA", "NHL", "MLB"] {
            let valid = try candidate(sport, player: true)
            XCTAssertEqual(AnalyticsFeedCore.bettingFeedQuotes(candidate: valid, history: history(valid), now: now).count, 1)
            var wrongHistory = history(valid); wrongHistory.verifiedPlayerID = "999"
            XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: valid, history: wrongHistory, now: now).isEmpty)
            XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: valid, history: nil, now: now).isEmpty)
            for mutation in 0..<8 {
                var row = quote(sport, player: true)
                switch mutation {
                case 0: row.updated = stamp(now.addingTimeInterval(1))
                case 1: row.updated = nil
                case 2: row.line = nil
                case 3: row.eventState = "final"
                case 4: row.price = 0
                case 5: row.eventLabel = "Wrong Away at Wrong Home"
                case 6: row.commence = stamp(now.addingTimeInterval(-1))
                default: row.label = "First half"
                }
                let bad = try candidate(sport, rows: [row], player: true)
                XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: bad, history: history(bad), now: now).isEmpty, "\(sport) mutation \(mutation)")
            }
        }
    }
    func testEasternWindowIncludesOvernightGamesButExcludesStartedAndSixthDay() {
        XCTAssertTrue(AnalyticsOddsMatcher.recordedWindowContains(sportsDate("2026-10-15T02:30:00Z")!, now: now))
        XCTAssertTrue(AnalyticsOddsMatcher.recordedWindowContains(sportsDate("2026-10-19T03:59:59Z")!, now: now))
        XCTAssertFalse(AnalyticsOddsMatcher.recordedWindowContains(sportsDate("2026-10-19T04:00:00Z")!, now: now))
        XCTAssertFalse(AnalyticsOddsMatcher.recordedWindowContains(now, now: now))
    }
    func testAmbiguousNHLProviderEventsRemainExcluded() throws {
        var second = quote("NHL"); second.id = "second"; second.event = "other-provider-event"
        let ambiguous = try candidate("NHL", rows: [quote("NHL"), second])
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: ambiguous, history: history(ambiguous), now: now).isEmpty)
    }
    func testAlternateLinesUseSameRecordedStatisticWithoutRewritingContract() throws {
        for sport in ["NFL", "NBA", "NHL", "MLB"] {
            var alternate = quote(sport, player: true)
            alternate.market += "_alternate"
            let built = try candidate(sport, rows: [alternate], player: true)
            XCTAssertEqual(AnalyticsHistoryEngine.statisticKey(sport: sport, marketKey: alternate.market), quote(sport, player: true).market)
            XCTAssertEqual(AnalyticsFeedCore.bettingFeedQuotes(candidate: built, history: history(built), now: now), [alternate])
            XCTAssertNotNil(AnalyticsHistoryEngine.evaluate(market: alternate, observations: history(built).observations, now: now).summary)
        }
        for key in ["player_points_q1_alternate", "player_points_alternate_alternate", "player_unknown_alternate", "player_anytime_td"] {
            XCTAssertNil(AnalyticsHistoryEngine.statisticKey(sport: "NBA", marketKey: key))
        }
    }
    func testRecordedDiscoveryDoesNotDiscardLaterSportsBeforeEligibility() {
        let many = (0..<1050).map { index -> Market in
            var row = quote("NBA", player: true); row.id = "seed-\(index)"; row.name = "Unverified seed \(index)"; return row
        }
        let result = AnalyticsFeedCore.build(snapshots: many + [quote("NFL")], verifiedSubjects: [proof("NFL")], now: now, limit: 30_000, recordedSnapshot: true)
        XCTAssertEqual(result.candidates.count, 1051)
        XCTAssertNotNil(result.candidates.first { $0.market.sport == "NFL" && $0.verifiedSubject != nil })
    }
    func testTeamHistoryUsesExistingAccentNormalizationWithoutAliases() {
        let directory: [String: Any] = ["sports": [["leagues": [["abbreviation":"NHL", "slug":"nhl", "teams":[["team":["id":"10", "displayName":"Montreal Canadiens"]]]]]]]]
        XCTAssertEqual(AnalyticsHistoryService.teamID(directory: directory, name: "Montréal Canadiens", sport: "NHL"), "10")
        XCTAssertNil(AnalyticsHistoryService.teamID(directory: directory, name: "Montreal", sport: "NHL"))
        XCTAssertNil(AnalyticsHistoryService.teamID(directory: directory, name: "Montreal Canadiens", sport: "NBA"))
    }
    private func response(_ sport: String, player: Bool = false) -> AnalyticsFeedService.RecordedSnapshot {
        let row = quote(sport, player: player)
        return AnalyticsFeedService.RecordedSnapshot(markets: [row], events: [RecordedEventMetadata(id: row.event, code: RecordedEventSelection.providerCode(sport: sport, phase: sport == "NBA" ? .preseason : nil)!, homeTeam: event(sport).home, awayTeam: event(sport).away, commence: row.commence!)], cacheOnly: true, fetchedAt: stamp(now.addingTimeInterval(-3500)), servedAt: stamp(now))
    }
    func testRecordedResponseRequiresCacheOnlyAndRealReceiptTimes() throws {
        let value = response("NFL")
        XCTAssertNoThrow(try AnalyticsFeedService.validateRecorded(value, sport: "NFL", now: now))
        for bad in [
            AnalyticsFeedService.RecordedSnapshot(markets: value.markets, events: value.events, cacheOnly: false, fetchedAt: value.fetchedAt, servedAt: value.servedAt),
            AnalyticsFeedService.RecordedSnapshot(markets: value.markets, events: value.events, cacheOnly: true, fetchedAt: nil, servedAt: value.servedAt),
            AnalyticsFeedService.RecordedSnapshot(markets: value.markets, events: value.events, cacheOnly: true, fetchedAt: stamp(now.addingTimeInterval(1)), servedAt: value.servedAt),
            AnalyticsFeedService.RecordedSnapshot(markets: value.markets, events: value.events, cacheOnly: true, fetchedAt: value.fetchedAt, servedAt: stamp(now.addingTimeInterval(-4000)))
        ] { XCTAssertThrowsError(try AnalyticsFeedService.validateRecorded(bad, sport: "NFL", now: now)) }
        XCTAssertThrowsError(try AnalyticsFeedService.validateRecorded(value, sport: "NBA", now: now))
    }
    private func publicPayload(_ url: URL) -> [String: Any] {
        let sport = url.host == "statsapi.mlb.com" ? "MLB" : url.path.contains("/nfl/") ? "NFL" : url.path.contains("/nhl/") ? "NHL" : "NBA"
        let event = event(sport)
        if url.path.hasSuffix("/roster") {
            // Only the home team carries this athlete. Membership is never inferred from odds.
            let home = url.path.contains("/teams/1/")
            if sport == "MLB" { return ["teamId": home ? 1 : 2, "roster": home ? [["person": ["id": 77, "fullName": sport + " Fixture Player"], "position": ["abbreviation": "OF"]]] : []] }
            return ["team": ["id": home ? "1" : "2", "displayName": home ? event.home : event.away], "athletes": home ? [["id":"77", "fullName":sport + " Fixture Player", "position":["abbreviation":sport == "NFL" ? "QB" : "PG"]]] : []]
        }
        if sport == "MLB" { return ["dates": [["games": [["gamePk":123, "gameDate":stamp(event.start), "status":["abstractGameState":"Preview", "codedGameState":"S", "startTimeTBD":false], "teams":["home":["team":["id":1,"name":event.home]], "away":["team":["id":2,"name":event.away]]]]]]]] }
        guard url.query?.contains("20261015") == true else { return ["events": []] }
        return ["events": [["id":event.providerID,"date":stamp(event.start),"season":["type":sport == "NBA" ? 1 : 2],"competitions":[["status":["type":["state":"pre","completed":false,"name":"STATUS_SCHEDULED"]],"competitors":[["homeAway":"home","team":["id":"1","displayName":event.home]],["homeAway":"away","team":["id":"2","displayName":event.away]]]]]]]]
    }
    func testIndependentRecordedDiscoveryRoutesFourSportsAndPreseasonWithoutRetainedSelections() async throws {
        let calls = RecordedFeedCalls()
        let service = AnalyticsFeedService(reader: AnalyticsFeedPublicReader(transport: { url in
            await calls.publicRead(url); return try JSONSerialization.data(withJSONObject: self.publicPayload(url))
        }))
        let built = await service.loadRecorded(read: { sport, phase, event in
            await calls.read(sport, phase, event); return self.response(sport, player: event != nil)
        }, now: now)
        XCTAssertEqual(Set(built.candidates.map { $0.market.sport }), Set(["NFL","NBA","NHL","MLB"]))
        XCTAssertEqual(built.candidates.count, 8)
        XCTAssertTrue(built.candidates.allSatisfy(\.recordedSnapshot))
        XCTAssertTrue(built.warnings.isEmpty, built.warnings.joined(separator: "; "))
        let cacheCalls = await calls.reads
        XCTAssertEqual(cacheCalls.count, 9)
        XCTAssertEqual(Set(cacheCalls.filter { $0.0 == "NBA" && $0.2 == nil }.compactMap { $0.1 }), Set([.regular, .preseason]))
        XCTAssertTrue(cacheCalls.filter { $0.0 == "NBA" && $0.2 != nil }.allSatisfy { $0.1 == .preseason })
        XCTAssertTrue(cacheCalls.filter { $0.0 != "NBA" }.allSatisfy { $0.1 == nil })
        XCTAssertEqual(Set(cacheCalls.compactMap { $0.2 }), Set(["NFL","NBA","NHL","MLB"].map {"recorded-" + $0}))
        let urls = await calls.urls
        XCTAssertEqual(urls.count, 24)
        for candidate in built.candidates { XCTAssertFalse(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: history(candidate), now: now).isEmpty, candidate.id) }
        let mlb = try XCTUnwrap(built.candidates.first { $0.market.sport == "MLB" && $0.subject == .player })
        XCTAssertEqual(mlb.verifiedSubject?.providerID, "77")
        XCTAssertEqual(URL(string: mlb.verifiedSubject!.sourceURL)?.host, "statsapi.mlb.com")
    }
    func testAuthoritativeEmptyCacheDoesNotResurrectRetainedOdds() async {
        let service = AnalyticsFeedService(reader: AnalyticsFeedPublicReader(transport: { url in try JSONSerialization.data(withJSONObject: self.publicPayload(url)) }))
        let result = await service.load(snapshots: [quote("NFL")], now: now, recordedRead: { _,_,_ in AnalyticsFeedService.RecordedSnapshot(markets: [], events: [], cacheOnly: true, fetchedAt: self.stamp(self.now.addingTimeInterval(-60)), servedAt: self.stamp(self.now)) })
        XCTAssertTrue(result.candidates.isEmpty)
    }
    func testMLBNamespaceMismatchAndConflictingRosterIdentityStayRejected() throws {
        let player = try candidate("MLB", player: true)
        let proof = player.verifiedSubject!
        let espnProof = AnalyticsFeedVerifiedSubject(sport: "MLB", subject: .player, name: proof.name, providerID: proof.providerID, teamName: proof.teamName, teamID: proof.teamID, marketKeys: proof.marketKeys, event: proof.event, sourceURL: "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams/1/roster", verifiedAt: now)
        let mixed = AnalyticsFeedCandidate(id: player.id, subject: .player, market: player.market, quotes: player.quotes, verifiedSubject: espnProof, recordedSnapshot: true)
        let mlbHistory = PlayerHistoryResult(games: [], opponentGames: [], context: nil, source: "MLB Stats API", warnings: [], availability: "Fixture", playerName: proof.name, playerID: proof.providerID)
        XCTAssertTrue(AnalyticsFeedService.acceptsPlayerHistory(candidate: player, history: mlbHistory))
        XCTAssertFalse(AnalyticsFeedService.acceptsPlayerHistory(candidate: mixed, history: mlbHistory))
        let team = self.proof("MLB")
        let rows: [[String:Any]] = [["person":["id":77,"fullName":"Duplicate"]],["person":["id":78,"fullName":"Duplicate"]]]
        let roster = try AnalyticsFeedService.mlbRosterSubjects(payload:["teamId":1,"roster":rows],team:team,sourceURL:"https://statsapi.mlb.com/api/v1/teams/1/roster",now:now)
        XCTAssertTrue(roster.isEmpty)
        XCTAssertThrowsError(try AnalyticsFeedService.mlbRosterSubjects(payload:["teamId":99,"roster":rows],team:team,sourceURL:"https://statsapi.mlb.com/api/v1/teams/1/roster",now:now))
    }
}
