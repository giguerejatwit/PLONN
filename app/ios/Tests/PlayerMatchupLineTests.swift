import XCTest
@testable import PLONNCore

final class PlayerMatchupLineTests: XCTestCase {
    let now = sportsDate("2026-10-22T20:00:00Z")!
    let start = "2026-10-23T00:15:00Z"
    let players = [PlayerMatchupPlayer(id: "42", name: "A Player", teamID: "17", team: "Away Team", position: "G")]
    func event(_ sport: String = "NFL") -> PlayerMatchupEvent {
        .init(game: .init(league: sport, eventID: "401234567", home: "Home Team", away: "Away Team", startsAt: start), homeID: "3", awayID: "17", seasonType: 2)
    }
    func quote(_ sport: String = "NFL", key: String = "player_pass_yds", line: Double = 249.5, side: String = "Over", book: String = "Book A") -> Market {
        Market(id: key + side + book, event: "provider-event", sport: sport, kind: "prop", name: "A Player", opponent: "Home Team", label: side + " " + String(line), aliases: "", p: nil, price: -110, market: key, book: book, quotes: [], updated: "2026-10-22T19:55:00Z", mode: "live", line: line, eventLabel: "Away Team at Home Team", eventState: "pre", commence: start, side: side)
    }
    func match(_ rows: [Market], sport: String = "NFL", stat: String = "Pass Yds", side: String = "Over", roster: [PlayerMatchupPlayer]? = nil) -> [Market] {
        PlayerMatchupLineMatching.quotes(event: event(sport), playerID: "42", players: roster ?? players, stat: stat, side: side, markets: rows, now: now)
    }
    func testCapturedDrakeMayeCacheMapsDifferentProviderIDsAndPrefillsRealLines() throws {
        guard let directory = ProcessInfo.processInfo.environment["PLONN_MATCHUP_LINE_FIXTURE_DIRECTORY"] else {
            throw XCTSkip("Set PLONN_MATCHUP_LINE_FIXTURE_DIRECTORY for the read-only captured-cache replay.")
        }
        func object(_ name: String) throws -> [String: Any] {
            let data = try Data(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent(name))
            return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        }
        let source = try object("captured-event-rosters.json")
        let capturedNow = try XCTUnwrap((source["checked_at"] as? String).flatMap(sportsDate))
        let scoreboard = try JSONSerialization.data(withJSONObject: XCTUnwrap(source["scoreboard"]))
        let events = try PlayerMatchupSource.events(scoreboard, league: "NFL", now: capturedNow)
        let event = try XCTUnwrap(events.first)
        XCTAssertEqual(event.game.eventID, "401873010")
        let rosters = try XCTUnwrap(source["rosters"] as? [[String: Any]])
        let players = try rosters.flatMap { roster -> [PlayerMatchupPlayer] in
            let team = try XCTUnwrap(roster["team"] as? [String: String])
            return try PlayerMatchupSource.players(JSONSerialization.data(withJSONObject: roster),
                teamID: XCTUnwrap(team["id"]), teamName: XCTUnwrap(team["displayName"]))
        }
        let player = try XCTUnwrap(players.first { $0.name == "Drake Maye" })
        XCTAssertEqual(player.id, "4431452"); XCTAssertEqual(player.teamID, "17")
        let cache = try object("captured-passing-yards-cache.json")
        let rows = try JSONDecoder().decode([Market].self, from: JSONSerialization.data(withJSONObject: XCTUnwrap(cache["markets"])))
        let matched = PlayerMatchupLineMatching.quotes(event: event, playerID: player.id, players: players,
            stat: "Pass Yds", side: "Over", markets: rows, now: capturedNow)
        XCTAssertFalse(matched.isEmpty)
        XCTAssertTrue(matched.allSatisfy { $0.event == "ffebea881c1d7b2dfb85b3e590c35985" && $0.event != event.game.eventID })
        let draftKings = try XCTUnwrap(PlayerMatchupLineMatching.preferred(matched, book: "DraftKings"))
        XCTAssertEqual(draftKings.line, 247.5); XCTAssertEqual(draftKings.price, -113); XCTAssertEqual(draftKings.book, "DraftKings")
        let newest = try XCTUnwrap(PlayerMatchupLineMatching.preferred(matched, book: nil))
        XCTAssertEqual(newest.line, 246.5); XCTAssertEqual(newest.price, -113); XCTAssertEqual(newest.book, "FanDuel")
        var draft = PlayerMatchupLineDraft(); draft.select("maye-pass", binary: false)
        draft.fill(newest.line, for: "maye-pass"); XCTAssertEqual(draft.text, "246.5")
        draft.edit("249.5"); draft.fill(draftKings.line, for: "maye-pass"); XCTAssertEqual(draft.text, "249.5")
        let alternate = try XCTUnwrap(PlayerMatchupLineMatching.preferred(matched, book: nil, line: 249.5))
        XCTAssertEqual(alternate.market, "player_pass_yds_alternate")
        XCTAssertEqual(alternate.book, "Fanatics"); XCTAssertEqual(alternate.price, 100)
    }
    func testPrimaryLinesAcrossAllFourSportsAndDifferentProps() {
        for (sport, key, stat, value) in [("NFL", "player_pass_yds", "Pass Yds", 249.5), ("NFL", "player_receptions", "Receptions", 4.5), ("NBA", "player_blocks", "Blocks", 1.5), ("NBA", "player_points", "Points", 22.5), ("MLB", "batter_hits", "Hits", 0.5), ("NHL", "player_shots_on_goal", "Shots on goal", 2.5)] {
            let row = quote(sport, key: key, line: value)
            XCTAssertEqual(PlayerMatchupLineMatching.preferred(match([row], sport: sport, stat: stat), book: row.book)?.line, value, sport + " " + stat)
        }
    }
    func testChangingPlayerOrPropMatchesOnlySelectedContract() {
        var other = quote(key: "player_receptions", line: 3.5); other.name = "Another Player"; other.id = "other-player"
        let roster = players + [.init(id: "43", name: "Another Player", teamID: "17", team: "Away Team", position: "WR")]
        let rows = [quote(), quote(key: "player_receptions", line: 4.5), other]
        XCTAssertEqual(match(rows, stat: "Receptions", roster: roster).map(\.line), [4.5])
        XCTAssertEqual(PlayerMatchupLineMatching.quotes(event: event(), playerID: "43", players: roster, stat: "Receptions", side: "Over", markets: rows, now: now).map(\.line), [3.5])
        XCTAssertTrue(match(rows, stat: "Rush Yds").isEmpty)
    }
    func testRejectsWrongEventStartSideRosterAndAmbiguousName() {
        let valid = quote()
        for change: (inout Market) -> Void in [
            { $0.eventLabel = "Home Team at Away Team" }, { $0.commence = "2026-10-24T00:15:00Z" },
            { $0.side = "Under" }, { $0.mode = "demo" }, { $0.eventState = "in" },
            { $0.updated = "2026-10-21T19:59:00Z" }, { $0.updated = "2026-10-22T20:01:00Z" },
            { $0.book = "" }, { $0.price = 0 }, { $0.line = .nan }
        ] {
            var bad = valid; change(&bad); XCTAssertTrue(match([bad]).isEmpty)
        }
        XCTAssertTrue(match([valid], roster: [.init(id: "42", name: "A Player", teamID: "8", team: "Other Team", position: "WR")]).isEmpty)
        XCTAssertTrue(match([valid], roster: players + [.init(id: "43", name: "A Player", teamID: "3", team: "Home Team", position: "WR")]).isEmpty)
    }
    func testRejectsAmbiguousProviderEventsAndConflictingQuoteIDs() {
        var other = quote(); other.event = "another-provider-event"; other.id = "other"
        XCTAssertTrue(match([quote(), other]).isEmpty)
        var conflict = quote(); conflict.line = 240.5
        XCTAssertTrue(match([quote(), conflict]).isEmpty)
    }
    func testSelectionUsesActualBookAndSideAndDoesNotDefaultToAlternate() {
        let row = quote(), second = quote(line: 250.5, book: "Book B")
        let under = quote(line: 251.5, side: "Under")
        let alternate = quote(key: "player_pass_yds_alternate", line: 200.5)
        let matched = match([row, second, under, alternate])
        XCTAssertEqual(PlayerMatchupLineMatching.preferred(matched, book: "Book B")?.line, 250.5)
        XCTAssertEqual(PlayerMatchupLineMatching.preferred(match([under], side: "Under"), book: "Book A")?.line, 251.5)
        XCTAssertNil(PlayerMatchupLineMatching.preferred(match([alternate]), book: "Book A"))
        XCTAssertEqual(PlayerMatchupLineMatching.preferred(matched, book: "Book A", line: 200.5)?.id, alternate.id)
        XCTAssertNil(PlayerMatchupLineMatching.preferred(matched, book: "Book A", line: 99))
    }
    func testConflictingPrimaryLinesAtSameBookAndTimestampAreNotGuessed() {
        var conflicting = quote(line: 240.5); conflicting.id = "different-line"
        XCTAssertNil(PlayerMatchupLineMatching.preferred(match([quote(), conflicting]), book: "Book A"))
    }
    func testNHLStartPolicyAcceptsUniquePuckDropAndRejectsAmbiguity() {
        var row = quote("NHL", key: "player_shots_on_goal", line: 2.5)
        row.commence = "2026-10-23T00:20:00Z"
        XCTAssertEqual(match([row], sport: "NHL", stat: "Shots on goal").count, 1)
        var other = row; other.event = "other"; other.id = "other"; other.commence = "2026-10-23T00:22:00Z"
        XCTAssertTrue(match([row, other], sport: "NHL", stat: "Shots on goal").isEmpty)
    }
    func testResultKeepsExactBookLineAndValidatedNHLStartConvention() {
        let setup = PlayerMatchupSetup(league: "NHL", eventID: "401234567", home: "Home Team", away: "Away Team", startsAt: start, player: "A Player", playerID: "42", teamID: "17", stat: "Shots on goal", line: 2.5, side: "Over", book: "Book A", propOptions: [:])
        var row = quote("NHL", key: "player_shots_on_goal", line: 2.5)
        row.commence = "2026-10-23T00:20:00Z"
        XCTAssertTrue(PlayerMatchupLineMatching.matches(row, setup: setup, now: now))
        for change: (inout Market) -> Void in [
            { $0.line = 3.5 }, { $0.book = "Book B" }, { $0.side = "Under" },
            { $0.name = "Another Player" }, { $0.eventLabel = "Different Team at Home Team" },
            { $0.eventState = "postponed" }, { $0.commence = "2026-10-23T00:30:00Z" }
        ] {
            var bad = row; change(&bad)
            XCTAssertFalse(PlayerMatchupLineMatching.matches(bad, setup: setup, now: now))
        }
    }
    func testDelayedAutofillNeverOverwritesManualValueOrIntentionalClear() {
        var draft = PlayerMatchupLineDraft()
        draft.select("event|player|stat", binary: false)
        draft.fill(249.5, for: "event|player|stat"); XCTAssertEqual(draft.text, "249.5")
        draft.edit("240.5"); draft.fill(250.5, for: "event|player|stat"); XCTAssertEqual(draft.text, "240.5")
        draft.edit(""); draft.fill(250.5, for: "event|player|stat"); XCTAssertEqual(draft.text, "")
        draft.select("event|other-player|stat", binary: false)
        draft.fill(100, for: "event|player|stat"); XCTAssertEqual(draft.text, "")
        draft.fill(3.5, for: "event|other-player|stat"); XCTAssertEqual(draft.text, "3.5")
    }
    func testBinaryContractStaysHalfAndSameSelectionDoesNotResetAnEdit() {
        var draft = PlayerMatchupLineDraft()
        draft.select("binary", binary: true); XCTAssertEqual(draft.text, "0.5")
        draft.select("count", binary: false); draft.edit("2.5")
        draft.select("count", binary: false); XCTAssertEqual(draft.text, "2.5")
        draft.fill(.infinity, for: "count"); XCTAssertEqual(draft.text, "2.5")
    }
}
