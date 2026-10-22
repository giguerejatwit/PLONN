import XCTest
@testable import PLONNCore

final class AnalyticsLifecycleTests: XCTestCase {
    private let now = sportsDate("2026-10-14T19:30:00Z")!
    private func stamp(_ date: Date) -> String { ISO8601DateFormatter().string(from: date) }
    private func subject(_ sport: String = "NFL", player: Bool = false, state: AnalyticsEventLifecycle = .inProgress,
                         statusAge: Double = 0, proofAge: Double = 0, archive: Bool = true) -> AnalyticsFeedCandidate {
        let event = AnalyticsEvent(sport: sport, providerID: "100", home: sport + " Home", away: sport + " Away", start: now.addingTimeInterval(-3600), homeID: "1", awayID: "2", seasonType: sport == "NBA" ? 1 : nil, lifecycle: state, statusVerifiedAt: now.addingTimeInterval(-statusAge))
        let key = player ? ["NFL":"player_receptions", "NBA":"player_points", "NHL":"player_shots_on_goal", "MLB":"batter_hits"][sport]! : "spreads"
        let name = player ? sport + " Player" : event.home
        let quote = Market(id: sport + (player ? "-player" : "-team"), event: "odds-" + sport, sport: sport, kind: player ? "prop" : "team", name: name, opponent: event.away, label: "Full game", aliases: "", p: nil, price: -110, market: key, book: "Recorded Book", quotes: [], updated: stamp(now.addingTimeInterval(-7200)), mode: "live", line: player ? 2.5 : -2.5, eventLabel: event.away + " at " + event.home, eventState: "pre", commence: stamp(event.start), side: player ? "over" : event.home)
        let base = sport == "MLB" ? "https://statsapi.mlb.com/api/v1" : "https://site.api.espn.com/apis/site/v2/sports/" + PlayerHistoryService.leaguePath(sport)
        let proof = AnalyticsFeedVerifiedSubject(sport: sport, subject: player ? .player : .team, name: name, providerID: player ? "77" : "1", teamName: player ? event.home : nil, teamID: player ? "1" : nil, marketKeys: [key], event: event, sourceURL: base + (player ? "/teams/1/roster" : sport == "MLB" ? "/schedule" : "/scoreboard"), verifiedAt: now.addingTimeInterval(-proofAge))
        return AnalyticsFeedCandidate(id: quote.id, subject: player ? .player : .team, market: quote, quotes: [quote], verifiedSubject: proof, recordedSnapshot: true, recordedArchive: archive)
    }
    private func history(_ candidate: AnalyticsFeedCandidate) -> AnalyticsHistoryResult {
        var result = AnalyticsHistoryEngine.statistics(market: candidate.market, observations: [AnalyticsObservation(id: "past-final", date: now.addingTimeInterval(-86400), opponent: "Earlier opponent", value: 3, scored: 20, allowed: 10)], now: now)
        result.verifiedPlayerID = candidate.subject == .player ? "77" : nil
        return result
    }
    func testAllFourSportsRetainExactPregameQuotesOnlyWithFreshLiveStatusAndArchiveEcho() {
        for sport in ["NFL", "NBA", "MLB", "NHL"] {
            for player in [false, true] {
                let candidate = subject(sport, player: player)
                XCTAssertEqual(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: history(candidate), now: now), candidate.quotes)
                XCTAssertEqual(candidate.quotes.first?.updated, stamp(now.addingTimeInterval(-7200)))
                XCTAssertNil(candidate.quotes.first?.p)
                let ordinary = subject(sport, player: player, archive: false)
                XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: ordinary, history: history(ordinary), now: now).isEmpty)
                let stale = subject(sport, player: player, statusAge: 61)
                XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: stale, history: history(stale), now: now).isEmpty)
            }
        }
    }
    func testTerminalUncertainAndExpiredRosterEvidenceNeverExtendsActiveEligibility() {
        for state in [AnalyticsEventLifecycle.final, .cancelled, .postponed, .suspended, .unknown, .scheduled] {
            let candidate = subject(state: state)
            XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: candidate, history: history(candidate), now: now).isEmpty, state.rawValue)
            XCTAssertEqual(candidate.quotes.count, 1, "Removing from active results must not erase the archive.")
        }
        let expired = subject(player: true, proofAge: 901)
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: expired, history: history(expired), now: now).isEmpty)
        XCTAssertEqual(subject(statusAge: 61).verifiedSubject?.event?.displayLifecycle(now: now), .unknown)
        XCTAssertEqual(subject(state: .scheduled).verifiedSubject?.event?.displayLifecycle(now: now), .unknown)
        let oldActive = AnalyticsEvent(sport:"MLB",providerID:"old",home:"Home",away:"Away",start:now.addingTimeInterval(-7*86400),lifecycle:.inProgress,statusVerifiedAt:now)
        XCTAssertTrue(oldActive.allowsRecorded(now:now,archive:true), "A verified active event must not expire from a guessed sport duration.")
        let futureStatus = subject(statusAge: -1)
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: futureStatus, history: history(futureStatus), now: now).isEmpty)
    }
    func testLiveOrPostStartPricesCannotBeRelabeledAsRecordedPregamePrices() {
        let candidate = subject()
        for postStart in [false, true] {
            var quote = candidate.quotes[0]
            if postStart { quote.updated = stamp(now.addingTimeInterval(-60)) } else { quote.eventState = "in" }
            let changed = AnalyticsFeedCandidate(id: candidate.id, subject: candidate.subject, market: candidate.market, quotes: [quote], verifiedSubject: candidate.verifiedSubject, recordedSnapshot: true, recordedArchive: true)
            XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: changed, history: history(changed), now: now).isEmpty)
        }
    }
    func testArchiveSelectionKeepsNHLUniquePairAndDefaultPregameBoundary() {
        let start = now.addingTimeInterval(-3600)
        let row = RecordedEventMetadata(id: "unique", code: "icehockey_nhl", homeTeam: "Home", awayTeam: "Away", commence: stamp(start.addingTimeInterval(420)))
        XCTAssertNil(RecordedEventSelection.select([row], sport: "NHL", phase: nil, home: "Home", away: "Away", start: start, now: now))
        XCTAssertEqual(RecordedEventSelection.select([row], sport: "NHL", phase: nil, home: "Home", away: "Away", start: start, now: now, includeStarted: true)?.id, "unique")
        let duplicate = RecordedEventMetadata(id: "other", code: row.code, homeTeam: row.homeTeam, awayTeam: row.awayTeam, commence: row.commence)
        XCTAssertNil(RecordedEventSelection.select([row, duplicate], sport: "NHL", phase: nil, home: "Home", away: "Away", start: start, now: now, includeStarted: true))
    }
    func testStatusParsersNeverInferLiveFromClockAndRecognizeTerminalAndUncertainStates() {
        for name in ["STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_END_PERIOD", "STATUS_END_QUARTER", "STATUS_OVERTIME"] {
            XCTAssertEqual(AnalyticsEventLifecycle.espn(["state":"in", "completed":false, "name":name]), .inProgress)
        }
        XCTAssertEqual(AnalyticsEventLifecycle.espn(["state":"post", "completed":true, "name":"STATUS_FINAL"]), .final)
        XCTAssertEqual(AnalyticsEventLifecycle.espn(["state":"pre", "completed":false, "name":"STATUS_POSTPONED"]), .postponed)
        XCTAssertEqual(AnalyticsEventLifecycle.espn(["state":"in", "completed":false, "name":"STATUS_SUSPENDED"]), .suspended)
        XCTAssertEqual(AnalyticsEventLifecycle.espn(["state":"in", "completed":false, "name":"STATUS_UNKNOWN"]), .unknown)
        XCTAssertEqual(AnalyticsEventLifecycle.espn(nil), .unknown)
        XCTAssertEqual(AnalyticsEventLifecycle.mlb(["abstractGameState":"Live", "codedGameState":"I"]), .inProgress)
        XCTAssertEqual(AnalyticsEventLifecycle.mlb(["abstractGameState":"Final", "codedGameState":"F"]), .final)
        XCTAssertEqual(AnalyticsEventLifecycle.mlb(["abstractGameState":"Preview", "codedGameState":"D"]), .postponed)
        XCTAssertEqual(AnalyticsEventLifecycle.mlb(["abstractGameState":"Live", "codedGameState":"T"]), .suspended)
        XCTAssertEqual(AnalyticsEventLifecycle.mlb(["abstractGameState":"Live", "codedGameState":"X"]), .unknown)
    }
    func testLeagueScopePrecedesSearchAndQuoteEligibility() {
        let candidates = ["NFL", "NBA", "MLB", "NHL"].flatMap { [subject($0), subject($0, player: true)] }
        let histories = Dictionary(uniqueKeysWithValues: candidates.map { ($0.id, history($0)) })
        for sport in ["NFL", "NBA", "MLB", "NHL"] {
            XCTAssertEqual(AnalyticsFeedCore.visibleCandidates(candidates, sport: sport, query: "", histories: histories, now: now).count, 2)
            XCTAssertEqual(AnalyticsFeedCore.visibleCandidates(candidates, sport: sport, query: "Player", histories: histories, now: now).map(\.market.sport), [sport])
            let other = sport == "NFL" ? "NBA" : "NFL"
            XCTAssertTrue(AnalyticsFeedCore.visibleCandidates(candidates, sport: sport, query: other, histories: histories, now: now).isEmpty)
        }
        XCTAssertTrue(AnalyticsFeedCore.visibleCandidates(candidates, sport: "All", query: "", histories: histories, now: now).isEmpty)
        XCTAssertEqual(AnalyticsFeedCore.visibleCandidates(candidates, sport: "NFL", query: "", histories: [:], now: now).count, 2, "Statistics cards stay visible while history loads, independently of odds.")
        XCTAssertEqual(AnalyticsFeedCore.visibleCandidates(candidates, sport: "NFL", query: "Player", histories: [:], now: now).count, 1, "Explicit search still permits general statistics.")
    }
    func testArchiveFlagIsOmittedForOrdinaryRequests() throws {
        let ordinary = OddsRequestBody(sport: "NFL", event: nil, cacheOnly: true)
        let archive = OddsRequestBody(sport: "NFL", event: "odds-NFL", cacheOnly: true, recordedArchive: true)
        let a = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(ordinary)) as? [String:Any])
        let b = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(archive)) as? [String:Any])
        XCTAssertNil(a["recordedArchive"]); XCTAssertEqual(b["recordedArchive"] as? Bool, true)
        XCTAssertEqual(b["event"] as? String, "odds-NFL"); XCTAssertEqual(b["cacheOnly"] as? Bool, true)
    }
    func testFreshStatusRefreshRemovesFinalWithoutChangingQuoteOrRosterReceipt() async throws {
        let candidate = subject(player: true, proofAge: 120)
        let event = try XCTUnwrap(candidate.verifiedSubject?.event)
        let payload: [String:Any] = ["events":[["id":event.providerID,"date":stamp(event.start),"competitions":[["status":["type":["state":"post","completed":true,"name":"STATUS_FINAL"]],"competitors":[["homeAway":"home","team":["id":"1","displayName":event.home]],["homeAway":"away","team":["id":"2","displayName":event.away]]]]]]]]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let service = AnalyticsFeedService(reader: AnalyticsFeedPublicReader(transport: { _ in data }))
        let refreshed = await service.refreshStatuses(candidates: [candidate], now: now)
        let final = try XCTUnwrap(refreshed.candidates.first)
        XCTAssertEqual(final.verifiedSubject?.event?.lifecycle, .final)
        XCTAssertEqual(final.verifiedSubject?.verifiedAt, candidate.verifiedSubject?.verifiedAt)
        XCTAssertEqual(final.quotes, candidate.quotes)
        XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: final, history: history(final), now: now).isEmpty)
    }
    func testAbsentOrFailedStatusReadCannotKeepAnEventActive() async throws {
        for fail in [false, true] {
            let service = AnalyticsFeedService(reader: AnalyticsFeedPublicReader(transport: { _ in
                if fail { throw URLError(.notConnectedToInternet) }
                return Data("{\"events\":[]}".utf8)
            }))
            let candidate = subject()
            let refreshed = await service.refreshStatuses(candidates: [candidate], now: now)
            let unknown = try XCTUnwrap(refreshed.candidates.first)
            XCTAssertEqual(unknown.verifiedSubject?.event?.lifecycle, .unknown)
            XCTAssertEqual(unknown.quotes, candidate.quotes)
            XCTAssertFalse(refreshed.warnings.isEmpty)
            XCTAssertTrue(AnalyticsFeedCore.bettingFeedQuotes(candidate: unknown, history: history(unknown), now: now).isEmpty)
        }
    }
    func testDiscoveryRequiresArchiveEchoBeforeRetainingAStartedEvent() async throws {
        let team = subject(), player = subject(player: true)
        let event = try XCTUnwrap(team.verifiedSubject?.event)
        let game: [String:Any] = ["id":event.providerID,"date":stamp(event.start),"competitions":[["status":["type":["state":"in","completed":false,"name":"STATUS_IN_PROGRESS"]],"competitors":[["homeAway":"home","team":["id":"1","displayName":event.home]],["homeAway":"away","team":["id":"2","displayName":event.away]]]]]]
        for archive in [false, true] {
            let service = AnalyticsFeedService(reader: AnalyticsFeedPublicReader(transport: { url in
                var payload: [String:Any] = ["events":[]]
                if url.host == "statsapi.mlb.com" { payload = ["dates":[]] }
                else if url.path.contains("/nfl/") {
                    if url.path.hasSuffix("/roster") {
                        let home = url.path.contains("/teams/1/")
                        payload = ["team":["id":home ? "1":"2","displayName":home ? event.home:event.away],"athletes":home ? [["id":"77","fullName":player.market.name,"position":["abbreviation":"WR"]]]:[]]
                    } else { payload = ["events":[game]] }
                }
                return try JSONSerialization.data(withJSONObject: payload)
            }))
            let result = await service.loadRecorded(read: { sport, _, detail in
                guard sport == "NFL" else { throw URLError(.resourceUnavailable) }
                return AnalyticsFeedService.RecordedSnapshot(markets: detail == nil ? team.quotes:player.quotes,
                    events:[RecordedEventMetadata(id:team.quotes[0].event,code:"americanfootball_nfl",homeTeam:event.home,awayTeam:event.away,commence:self.stamp(event.start))],
                    cacheOnly:true,fetchedAt:self.stamp(self.now.addingTimeInterval(-7000)),servedAt:self.stamp(self.now),recordedArchive:archive)
            }, now: now)
            XCTAssertEqual(result.candidates.count, archive ? 2:0)
            for candidate in result.candidates {
                XCTAssertTrue(candidate.recordedArchive)
                XCTAssertEqual(candidate.verifiedSubject?.event?.lifecycle, .inProgress)
                XCTAssertFalse(AnalyticsFeedCore.bettingFeedQuotes(candidate:candidate,history:history(candidate),now:now).isEmpty)
            }
        }
    }
    func testStatusReadBypassesTheOrdinaryPublicDataCache() async throws {
        actor Counter {
            var value = 0
            func next() -> Data { value += 1; return Data(String(value).utf8) }
        }
        let counter = Counter()
        let reader = AnalyticsFeedPublicReader(transport: { _ in await counter.next() })
        let url = URL(string:"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20261014&limit=100")!
        let first = try await reader.read(url)
        let cached = try await reader.read(url)
        let refreshed = try await reader.read(url,useCache:false)
        XCTAssertEqual(first,cached); XCTAssertNotEqual(first,refreshed)
    }

    func testConflictingPublicStatusCannotBeTreatedAsInProgress() throws {
        let candidate = subject()
        let event = try XCTUnwrap(candidate.verifiedSubject?.event)
        let payload: [String:Any] = ["events":[["id":event.providerID,"date":stamp(event.start),"status":["type":["state":"post","completed":true,"name":"STATUS_FINAL"]],"competitions":[["status":["type":["state":"in","completed":false,"name":"STATUS_IN_PROGRESS"]],"competitors":[["homeAway":"home","team":["id":"1","displayName":event.home]],["homeAway":"away","team":["id":"2","displayName":event.away]]]]]]]]
        let rows = try AnalyticsFeedService.scoreboardSubjects(payload:payload,sport:"NFL",sourceURL:candidate.verifiedSubject!.sourceURL,now:now,recordedWindow:true,includeInactive:true)
        XCTAssertEqual(rows.count,2)
        XCTAssertTrue(rows.allSatisfy { $0.event?.lifecycle == .unknown })
    }

}
