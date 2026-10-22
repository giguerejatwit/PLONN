import XCTest
@testable import PLONNCore

final class InterestsTests: XCTestCase {
    private func team(_ sport: InterestSport = .nfl, id: String = "17", name: String = "New England Patriots") -> InterestTeam {
        InterestTeam(sport: sport, providerID: id, name: name, abbreviation: "NE")
    }
    private func instant(_ value: String) -> Date { ISO8601DateFormatter().date(from: value)! }

    func testTypedMetadataRoundTripPreservesAllChoicesAndSkipCompletion() throws {
        var value = UserInterests()
        value.onboardingCompleted = true
        value.sports = [.nfl, .nba, .mlb, .nhl]
        value.teams = [team(), team(.nba, id: "2", name: "Boston Celtics"), team(.mlb, id: "2", name: "Boston Red Sox"), team(.nhl, id: "1", name: "Boston Bruins")]
        value.notifications.modelReady = true
        value.notifications.dailyDigest = true
        value.notifications.timeZoneID = "America/New_York"
        XCTAssertEqual(UserInterests.decode(try value.encoded()), value)
        XCTAssertLessThan(try value.encoded().utf8.count, UserInterests.maximumBytes)
        let skipped = UserInterests(onboardingCompleted: true)
        XCTAssertEqual(UserInterests.decode(try skipped.encoded())?.onboardingCompleted, true)
        XCTAssertFalse(skipped.notifications.wantsAlerts)
    }

    func testNoOptInIsImpliedBySelectingSportOrTeam() {
        var value = UserInterests()
        value.toggle(team())
        XCTAssertTrue(value.sports.isEmpty)
        XCTAssertEqual(value.teams.count, 1)
        value.toggle(.nfl)
        value.toggle(.nfl)
        XCTAssertTrue(value.sports.isEmpty)
        XCTAssertEqual(value.teams.count, 1)
        XCTAssertFalse(value.notifications.wantsAlerts)
        value.toggle(team())
        XCTAssertTrue(value.teams.isEmpty)
    }

    func testInvalidMetadataIDsDuplicatesVersionsAndOversizedDataAreRejected() throws {
        XCTAssertNil(UserInterests.decode("{}"))
        XCTAssertNil(UserInterests.decode("not json"))
        XCTAssertNil(UserInterests.decode(String(repeating: "x", count: UserInterests.maximumBytes + 1)))
        for id in ["", "0", "01", "-1", "1/roster", "１２", "12345678901"] {
            var value = UserInterests(); value.teams = [team(id: id)]
            XCTAssertThrowsError(try value.encoded(), id)
        }
        var value = UserInterests()
        value.version = 2; XCTAssertThrowsError(try value.encoded())
        value.version = 1; value.sports = [.nfl, .nfl]; XCTAssertThrowsError(try value.encoded())
        value.sports = []; value.teams = [team(), team()]; XCTAssertThrowsError(try value.encoded())
        value.teams = [team(name: "Bad\nName")]; XCTAssertThrowsError(try value.encoded())
        value.teams = []; value.notifications.timeZoneID = "Mars/Phobos"; XCTAssertThrowsError(try value.encoded())
        value.notifications.timeZoneID = "America/New_York"
        value.notifications.digestMinute = 1440; XCTAssertThrowsError(try value.encoded())
    }

    func testAccountChangeSignoutAnonymousAndABARejectOldSavesAndClearPrivateChoices() throws {
        let a = UUID(), b = UUID()
        var state = InterestsAccountState()
        var choices = UserInterests(); choices.teams = [team()]
        state.restore(owner: a, isAnonymous: false, metadata: try choices.encoded())
        let old = try XCTUnwrap(state.operation())
        XCTAssertTrue(state.accepts(old))
        state.deferOnboarding()
        state.restore(owner: b, isAnonymous: false, metadata: nil)
        XCTAssertFalse(state.accepts(old))
        XCTAssertTrue(state.preferences.teams.isEmpty)
        XCTAssertFalse(state.onboardingDeferred)
        state.restore(owner: a, isAnonymous: false, metadata: try choices.encoded())
        XCTAssertFalse(state.accepts(old), "Returning to the same account cannot revive a pending old operation")
        XCTAssertEqual(state.preferences.teams.count, 1)
        state.restore(owner: a, isAnonymous: true, metadata: try choices.encoded())
        XCTAssertNil(state.owner)
        XCTAssertTrue(state.preferences.teams.isEmpty)
        state.restore(owner: nil, isAnonymous: nil, metadata: nil)
        XCTAssertNil(state.operation())
    }

    func testRefreshKeepsCurrentOperationButMalformedMetadataCannotAutoOnboard() throws {
        var state = InterestsAccountState(); let owner = UUID()
        state.restore(owner: owner, isAnonymous: false, metadata: nil)
        let operation = try XCTUnwrap(state.operation())
        state.restore(owner: owner, isAnonymous: false, metadata: try UserInterests().encoded())
        XCTAssertTrue(state.accepts(operation))
        state.restore(owner: owner, isAnonymous: false, metadata: "corrupted")
        XCTAssertTrue(state.invalidMetadata)
        XCTAssertFalse(state.shouldOnboard(pending: UserInterests.onboardingPending, createdAt: Date()))
    }

    func testOnboardingNewSocialSignupDelayedEmailConfirmationExistingAccountAndSkip() throws {
        var state = InterestsAccountState(); let owner = UUID()
        let now = instant("2026-10-20T16:00:00Z")
        state.restore(owner: owner, isAnonymous: false, metadata: nil)
        XCTAssertTrue(state.shouldOnboard(pending: nil, createdAt: now.addingTimeInterval(-60), now: now))
        XCTAssertFalse(state.shouldOnboard(pending: nil, createdAt: now.addingTimeInterval(-86_400), now: now))
        XCTAssertFalse(state.shouldOnboard(pending: nil, createdAt: now.addingTimeInterval(60), now: now))
        XCTAssertTrue(state.shouldOnboard(pending: UserInterests.onboardingPending, createdAt: now.addingTimeInterval(-7 * 86_400), now: now))
        state.restore(owner: owner, isAnonymous: false, metadata: try UserInterests(onboardingCompleted: true).encoded())
        XCTAssertFalse(state.shouldOnboard(pending: UserInterests.onboardingPending, createdAt: now, now: now))
        state.restore(owner: owner, isAnonymous: false, metadata: nil)
        state.deferOnboarding()
        XCTAssertFalse(state.shouldOnboard(pending: UserInterests.onboardingPending, createdAt: now, now: now))
    }

    func testQuietHoursAcrossMidnightBoundariesDSTAndExplicitZone() {
        var prefs = InterestNotifications()
        prefs.timeZoneID = "America/New_York"
        XCTAssertFalse(prefs.isQuiet(at: instant("2026-10-20T01:59:00Z"))) // 21:59
        XCTAssertTrue(prefs.isQuiet(at: instant("2026-10-20T02:00:00Z")))  // 22:00
        XCTAssertTrue(prefs.isQuiet(at: instant("2026-10-20T11:59:00Z")))  // 07:59
        XCTAssertFalse(prefs.isQuiet(at: instant("2026-10-20T12:00:00Z"))) // 08:00
        XCTAssertTrue(prefs.isQuiet(at: instant("2026-11-01T05:30:00Z"))) // First 01:30, EDT
        XCTAssertTrue(prefs.isQuiet(at: instant("2026-11-01T06:30:00Z"))) // Second 01:30, EST
        prefs.quietStartMinute = 12 * 60; prefs.quietEndMinute = 14 * 60
        XCTAssertTrue(prefs.isQuiet(at: instant("2026-10-20T17:00:00Z"))) // 13:00
        XCTAssertFalse(prefs.isQuiet(at: instant("2026-10-20T18:00:00Z")))
        prefs.quietHoursEnabled = false
        XCTAssertFalse(prefs.isQuiet(at: instant("2026-10-20T17:00:00Z")))
        prefs.quietStartMinute = prefs.quietEndMinute
        XCTAssertTrue(prefs.isValid)
        prefs.quietHoursEnabled = true
        XCTAssertFalse(prefs.isValid)
    }

    private func payload(sport: InterestSport, teams: [[String: Any]]) throws -> Data {
        try JSONSerialization.data(withJSONObject: ["sports": [["leagues": [["abbreviation": sport.rawValue, "slug": sport.rawValue.lowercased(), "teams": teams.map { ["team": $0] }]]]]])
    }
    func testAllSportDirectoriesValidateLeagueNamespaceAndConflictingIdentity() throws {
        for sport in InterestSport.allCases {
            let data = try payload(sport: sport, teams: [["id": "17", "displayName": "Verified Team", "abbreviation": "VT"]])
            let teams = try InterestsTeamDirectory.parse(data, sport: sport)
            XCTAssertEqual(teams.count, 1)
            XCTAssertEqual(teams.first?.id, sport.rawValue + ":espn:17")
            let other: InterestSport = sport == .nfl ? .nba : .nfl
            XCTAssertThrowsError(try InterestsTeamDirectory.parse(data, sport: other))
        }
        let activeOnly = try payload(sport: .nba, teams: [
            ["id": "1", "displayName": "Active Team", "abbreviation": "ACT", "isActive": true],
            ["id": "2", "displayName": "Inactive Team", "abbreviation": "OLD", "isActive": false],
            ["id": "3", "displayName": "All Star Team", "abbreviation": "ALL", "isAllStar": true]
        ])
        XCTAssertEqual(try InterestsTeamDirectory.parse(activeOnly, sport: .nba).map(\.providerID), ["1"])
        let conflict = try payload(sport: .nfl, teams: [["id": "17", "displayName": "One Team", "abbreviation": "ONE"], ["id": "17", "displayName": "Another Team", "abbreviation": "TWO"]])
        XCTAssertThrowsError(try InterestsTeamDirectory.parse(conflict, sport: .nfl))
        XCTAssertThrowsError(try InterestsTeamDirectory.parse(try payload(sport: .nfl, teams: []), sport: .nfl))
        XCTAssertThrowsError(try InterestsTeamDirectory.parse(Data(repeating: 0, count: 2 * 1024 * 1024 + 1), sport: .nfl))
    }

    func testDirectoryUsesOnlyUncredentialedPublicTeamEndpoint() async throws {
        let fixture = try payload(sport: .mlb, teams: [["id": "2", "displayName": "Boston Red Sox", "abbreviation": "BOS"]])
        let service = InterestsTeamDirectory(reader: { url in
            XCTAssertEqual(url.absoluteString, "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams?limit=1000")
            XCTAssertNil(url.user); XCTAssertNil(url.password)
            return fixture
        })
        let teams = try await service.load(.mlb)
        XCTAssertEqual(teams.first?.providerID, "2")
    }
}
