import XCTest
@testable import PLONNCore

final class PlayerMatchupSourceTests: XCTestCase {
    let now = ISO8601DateFormatter().date(from: "2026-10-19T12:00:00Z")!
    func data(_ body: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: body) }
    func event(_ state: String = "pre", date: String = "2026-10-20T12:00:00Z") -> [String: Any] {
        ["id": "401234567", "season": ["type": 2], "competitions": [["id": "401234567", "date": date,
            "status": ["type": ["state": state, "name": state == "pre" ? "STATUS_SCHEDULED" : "STATUS_IN_PROGRESS", "completed": false]],
            "competitors": [["homeAway": "home", "team": ["id": "17", "displayName": "New England Patriots"]], ["homeAway": "away", "team": ["id": "9", "displayName": "Green Bay Packers"]]]]]]
    }
    func roster(team: String = "17", name: String = "New England Patriots", duplicate: Bool = false) -> [String: Any] {
        let player: [String: Any] = ["id": "4569173", "displayName": "Rhamondre Stevenson", "position": ["abbreviation": "RB"]]
        return ["team": ["id": team, "displayName": name], "athletes": [["position": "offense", "items": duplicate ? [player, player] : [player]]]]
    }
    func testOnlyScheduledUpcomingEventsWithVerifiedTeamIDs() throws {
        let normal = try PlayerMatchupSource.events(data(["events": [event()]]), league: "NFL", now: now)
        XCTAssertEqual(normal.count, 1); XCTAssertEqual(normal.first?.homeID, "17")
        XCTAssertTrue(try PlayerMatchupSource.events(data(["events": [event("in")]]), league: "NFL", now: now).isEmpty)
        XCTAssertTrue(try PlayerMatchupSource.events(data(["events": [event(date: "2026-10-19T11:00:00Z")]]), league: "NFL", now: now).isEmpty)
        XCTAssertTrue(try PlayerMatchupSource.events(data(["events": [event(date: "2026-10-29T12:00:00Z")]]), league: "NFL", now: now).isEmpty)
        XCTAssertTrue(try PlayerMatchupSource.events(data(["events": [event(), event()]]), league: "NFL", now: now).isEmpty)
    }
    func testRosterRejectsWrongTeamAndDuplicateIDs() throws {
        let players = try PlayerMatchupSource.players(data(roster()), teamID: "17", teamName: "New England Patriots")
        XCTAssertEqual(players.first?.position, "RB")
        XCTAssertThrowsError(try PlayerMatchupSource.players(data(roster(team: "9")), teamID: "17", teamName: "New England Patriots"))
        XCTAssertThrowsError(try PlayerMatchupSource.players(data(roster(name: "Green Bay Packers")), teamID: "17", teamName: "New England Patriots"))
        XCTAssertThrowsError(try PlayerMatchupSource.players(data(roster(duplicate: true)), teamID: "17", teamName: "New England Patriots"))
    }
    func testFlatAndGroupedRostersAcrossSports() throws {
        for league in ["NFL", "NBA", "MLB", "NHL"] {
            XCTAssertNotNil(PlayerMatchupSource.path(league))
            let player: [String: Any] = ["id": "42", "displayName": "A Player", "position": ["abbreviation": "G"]]
            for athletes in [[player], [["items": [player]]]] {
                let result = try PlayerMatchupSource.players(data(["team": ["id": "17", "displayName": "Team"], "athletes": athletes]), teamID: "17", teamName: "Team")
                XCTAssertEqual(result.map(\.id), ["42"])
            }
        }
    }
    func candidate(sport: String, id: String, team: String?) -> AnalyticsFeedCandidate {
        let market = Market(id: "test", event: "", sport: sport, kind: "prop", name: "A Player", opponent: "", label: "", aliases: "", p: nil, price: 0, market: "player_points", book: "", quotes: [], updated: nil, mode: "analytics")
        let proof = AnalyticsFeedVerifiedSubject(sport: sport, subject: .player, name: "A Player", providerID: id, teamName: team, marketKeys: ["player_points"], sourceURL: "https://site.api.espn.com", verifiedAt: now)
        return AnalyticsFeedCandidate(id: "test", subject: .player, market: market, quotes: [], verifiedSubject: proof)
    }
    func testHistoricalVariationUsesSampleDeviationAndRequiresTwoFiniteResults() {
        XCTAssertNil(AnalyticsResearch.standardDeviation([]))
        XCTAssertNil(AnalyticsResearch.standardDeviation([5]))
        XCTAssertNil(AnalyticsResearch.standardDeviation([5, .nan]))
        XCTAssertNil(AnalyticsResearch.standardDeviation([1e308, -1e308]))
        XCTAssertEqual(AnalyticsResearch.standardDeviation([3, 7])!, sqrt(8), accuracy: 0.000001)
        XCTAssertEqual(AnalyticsResearch.standardDeviation([2, 4, 4, 4, 5, 5, 7, 9])!, sqrt(32.0 / 7), accuracy: 0.000001)
        XCTAssertEqual(AnalyticsResearch.standardDeviation([5, 5]), 0)
    }
    func testMLBEventMappingRequiresExactUniqueMatchup() {
        let game = UpcomingGameSetup(league: "MLB", eventID: "401234567", home: "Home", away: "Away", startsAt: "2026-10-20T12:00:00Z")
        let event = PlayerMatchupEvent(game: game, homeID: "2", awayID: "3")
        var proof = AnalyticsEvent(sport: "MLB", providerID: "777123", home: "Home", away: "Away", start: game.start!)
        XCTAssertTrue(PlayerMatchupSource.matches(proof, event: event, within: [event]))
        let other = PlayerMatchupEvent(game: UpcomingGameSetup(league: "MLB", eventID: "401999999", home: "Home", away: "Away", startsAt: game.startsAt), homeID: "2", awayID: "3")
        XCTAssertFalse(PlayerMatchupSource.matches(proof, event: event, within: [event, other]))
        proof = AnalyticsEvent(sport: "MLB", providerID: "777123", home: "Home", away: "Away", start: game.start!.addingTimeInterval(3600))
        XCTAssertFalse(PlayerMatchupSource.matches(proof, event: event, within: [event]))
        let nfl = PlayerMatchupEvent(game: UpcomingGameSetup(league: "NFL", eventID: "401234567", home: "Home", away: "Away", startsAt: game.startsAt), homeID: "2", awayID: "3")
        proof = AnalyticsEvent(sport: "NFL", providerID: "777123", home: "Home", away: "Away", start: game.start!)
        XCTAssertFalse(PlayerMatchupSource.matches(proof, event: nfl, within: [nfl]))
    }
    func testIdentityCannotJumpTeamsOrProviderNamespace() {
        let players = [PlayerMatchupPlayer(id: "42", name: "A Player", teamID: "17", team: "Team A", position: "RB")]
        XCTAssertNil(PlayerMatchupSource.selection(for: candidate(sport: "NFL", id: "43", team: "Team A"), players: players))
        XCTAssertNil(PlayerMatchupSource.selection(for: candidate(sport: "NFL", id: "42", team: "Team B"), players: players))
        XCTAssertEqual(PlayerMatchupSource.selection(for: candidate(sport: "NFL", id: "42", team: "Team A"), players: players)?.id, "42")
        XCTAssertEqual(PlayerMatchupSource.selection(for: candidate(sport: "MLB", id: "90001", team: "Team A"), players: players)?.id, "42")
        XCTAssertNil(PlayerMatchupSource.selection(for: candidate(sport: "MLB", id: "42", team: nil), players: players))
        XCTAssertNil(PlayerMatchupSource.selection(for: candidate(sport: "MLB", id: "90001", team: "Team B"), players: players))
    }
}
