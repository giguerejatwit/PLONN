import XCTest
@testable import PLONNCore

final class AskPLONNV5ContextTests: XCTestCase {
    typealias J = RichContextJSON

    func corpus() throws -> J {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var parser = RichContextJSONParser(try Data(contentsOf: root.appendingPathComponent("contracts/matchup-narrative-v5.json")))
        return try parser.parse()
    }
    func versions(_ root: J) -> [String: String] {
        guard case .object(let values) = root["versions"] else { return [:] }
        return values.mapValues(\.string)
    }
    func date(_ item: J) throws -> Date {
        Date(timeIntervalSince1970: try XCTUnwrap(Double(item["now"].number ?? "")))
    }
    func fixture(_ name: String) throws -> (MatchupNarrativeV4.Result, Date) {
        let root = try corpus(), item = try XCTUnwrap(root["cases"].array.first { $0["id"].string == name })
        let now = try date(item)
        return (try MatchupNarrativeV5.validate(Data(item["canonicalUTF8"].string.utf8), now: now, versions: versions(root)), now)
    }
    func projected(_ result: MatchupNarrativeV4.Result, _ now: Date, question: String = "Any injuries?") throws -> [AskPLONNEvidence] {
        try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: question, now: now)
    }
    func claimEvidence(_ evidence: [AskPLONNEvidence]) -> [AskPLONNEvidence] {
        evidence.filter { $0.id.hasPrefix("narrative|") }
    }
    func offlineService() -> AskPLONNContextService {
        .init(read: { _ in XCTFail("Chosen event must not discover"); throw PlayerHistoryError.invalidPayload },
              player: { _, _ in XCTFail("Status is not player history"); return .init(games: [], opponentGames: [], context: nil, source: "", warnings: [], availability: "") },
              team: { _, _ in XCTFail("Status is not historical team scoring"); return .init(summary: nil, rows: [], observations: [], source: "", warnings: [], availability: "") })
    }

    func testAll48CasesPreserveValidatedRoleClaimsStatusAndBoundsAcrossFourSports() throws {
        let root = try corpus()
        XCTAssertEqual(root["cases"].array.count, 48)
        var sports = Set<String>(), accepted = 0, rejected = 0
        for item in root["cases"].array {
            let now = try date(item)
            if item["expectedError"] != .null {
                XCTAssertThrowsError(try MatchupNarrativeV5.prepare(item["context"], now: now,
                    injuryContext: item["injuryContext"], injuryRoleContext: item["injuryRoleContext"]))
                rejected += 1
                continue
            }
            let result = try MatchupNarrativeV5.validate(Data(item["canonicalUTF8"].string.utf8), now: now, versions: versions(root))
            for question in ["Any injuries?", "What can we expect?"] {
                let evidence = try projected(result, now, question: question)
                let claims = question == "Any injuries?" ? result.claims.filter { $0.dimension.hasSuffix("_availability") } : result.claims
                XCTAssertEqual(claimEvidence(evidence).map(\.text), claims.map { $0.text + "\n\n" + $0.uncertainty }, item["id"].string)
                for fact in claimEvidence(evidence) {
                    for provider in result.providers { XCTAssertTrue(fact.source.contains(provider)) }
                }
                let statuses = evidence.filter { $0.id.hasPrefix("narrative-injury|") }.map(\.text).joined(separator: "\n")
                for row in result.roleStatusEvidence.array {
                    XCTAssertTrue(statuses.contains(row["playerName"].string + ": reported status " + row["reportedStatus"].string))
                    XCTAssertTrue(statuses.contains("Scope: " + row["scope"].string))
                    XCTAssertTrue(statuses.contains("Confirmed for this game: " + (row["eventStatusConfirmed"] == .bool(true) ? "yes" : "no")))
                }
                for gap in result.missingContext { XCTAssertTrue(evidence.contains { $0.availability != .available && $0.text.contains(gap) }) }
                let roleLimits = evidence.filter { $0.id.hasPrefix("narrative-role-limit|") }
                XCTAssertFalse(roleLimits.isEmpty)
                XCTAssertTrue(roleLimits.allSatisfy { $0.availability != .available })
                XCTAssertNoThrow(try AskPLONNCore.boundedRequest(.init(question: question, intent: .recentForm, evidence: evidence)))
            }
            sports.insert(result.event.sport); accepted += 1
        }
        XCTAssertEqual(sports, ["NFL", "NBA", "NHL", "MLB"])
        XCTAssertEqual(accepted, 44); XCTAssertEqual(rejected, 4)
    }

    func testHomeAndAwayAvailabilityBothMatchStatusQuestions() throws {
        let (result, now) = try fixture("recorded-nhl-offline-selection")
        let claims = result.claims.filter { $0.dimension.hasSuffix("_availability") }
        XCTAssertEqual(Set(claims.map(\.dimension)), ["home_availability", "away_availability"])
        for question in ["Any injuries?", "Who is out?", "Are they healthy?", "Player availability?"] {
            let evidence = try projected(result, now, question: question)
            XCTAssertEqual(claimEvidence(evidence).map(\.text), claims.map { $0.text + "\n\n" + $0.uncertainty })
            XCTAssertFalse(evidence.contains { $0.id == "narrative-requested-gap" })
        }
    }

    func testNumericalQuestionsDoNotReceiveAvailabilityClaimsOrStatusRows() throws {
        let (result, now) = try fixture("mlb-accepted_conditional")
        let evidence = try projected(result, now, question: "High strikeouts?")
        XCTAssertEqual(claimEvidence(evidence).map(\.text), result.claims.filter { $0.dimension == "strikeouts" }.map { $0.text + "\n\n" + $0.uncertainty })
        XCTAssertFalse(evidence.contains { $0.id.hasPrefix("narrative-injury|") })
        XCTAssertTrue(evidence.contains { $0.id.hasPrefix("narrative-role-limit|") && $0.availability == .unavailable })
    }

    func testReportedAndConfirmedOutRemainDistinctAcrossFourSports() throws {
        for sport in ["nfl", "nba", "mlb", "nhl"] {
            for confirmed in [false, true] {
                let (result, now) = try fixture(sport + (confirmed ? "-accepted_confirmed_out" : "-accepted_conditional"))
                let evidence = try projected(result, now)
                let text = claimEvidence(evidence).map(\.text).joined()
                XCTAssertEqual(text.contains("confirmed Out for this game"), confirmed)
                XCTAssertEqual(text.contains("This is not a confirmed game absence."), !confirmed)
                let statuses = evidence.filter { $0.id.hasPrefix("narrative-injury|") }.map(\.text).joined()
                XCTAssertTrue(statuses.contains("reported status Out"))
                XCTAssertTrue(statuses.contains(confirmed ? "Scope: game. Confirmed for this game: yes" : "Scope: general. Confirmed for this game: no"))
            }
        }
    }

    func testHistoricalUsageAndReplacementQualificationsStayExact() throws {
        for sport in ["nfl", "nba", "mlb", "nhl"] {
            for variant in ["historical_usage", "named_replacement", "cross_role_replacement", "starter_over_position"] {
                let (result, now) = try fixture(sport + "-" + variant)
                let claims = result.claims.filter { !$0.roleEvidenceIDs.isEmpty }
                let evidence = claimEvidence(try projected(result, now))
                XCTAssertEqual(evidence.map(\.text), claims.map { $0.text + "\n\n" + $0.uncertainty })
                if variant == "historical_usage" {
                    XCTAssertTrue(evidence.allSatisfy { $0.text.contains("Historical usage does not establish a current assignment") })
                }
                if variant == "cross_role_replacement" { XCTAssertFalse(evidence.contains { $0.text.contains("source-designated") }) }
            }
        }
    }

    func testRoleGapsExclusionsAndOmissionsRemainUnavailableEvidence() throws {
        for sport in ["nfl", "nba", "mlb", "nhl"] {
            for (variant, wording) in [("wrong_player", "no matching current role evidence"),
                                       ("wrong_event", "the report belongs to a different game"),
                                       ("stale_role_fresh_status", "the role report is stale"),
                                       ("fresh_role_stale_status", "the player-status report is stale")] {
                let (result, now) = try fixture(sport + "-" + variant)
                let evidence = try projected(result, now), limits = evidence.filter { $0.id.hasPrefix("narrative-role-limit|") }
                XCTAssertTrue(claimEvidence(evidence).isEmpty)
                XCTAssertTrue(limits.map(\.text).joined().contains(wording))
                XCTAssertTrue(limits.allSatisfy { $0.availability == (variant == "stale_role_fresh_status" ? .stale : .unavailable) })
                if variant == "fresh_role_stale_status" { XCTAssertFalse(evidence.contains { $0.availability == .available }) }
            }
        }
        let (result, now) = try fixture("recorded-nfl-offline-selection")
        let limits = try projected(result, now).filter { $0.id.hasPrefix("narrative-role-limit|") }.map(\.text).joined()
        XCTAssertTrue(limits.contains("\(result.injuryRoleGaps.array.count) player-status observations cannot support a role claim"))
        XCTAssertTrue(limits.contains("\(result.injuryRoleContext["omissions"]["count"].integer!) eligible role observations omitted"))
        for row in result.injuryRoleContext["coverage"].array {
            XCTAssertTrue(limits.contains(row["teamName"].string + ": role coverage is " + row["state"].string))
        }
        XCTAssertTrue(limits.contains("Missing role or status evidence does not establish health"))
    }

    func testRolePremiseOutsideSixGeneralInjuryRowsSurvivesWithoutDuplicates() throws {
        let root = try corpus(), item = try XCTUnwrap(root["cases"].array.first { $0["id"].string == "nfl-accepted_conditional" })
        let now = try date(item), original = item["injuryContext"]["observations"].array[0]
        // Synthetic additional players deliberately sort before the cited player.
        // No roles are supplied for these players, so they cannot create role claims.
        var extras: [J] = []
        for index in 1...100 where extras.count < 6 {
            var row = original.setting("playerID", .string("espn:nfl:player:\(5000 + index)"))
                .setting("playerName", .string("Additional Player \(index)"))
            row = row.setting("id", .string(NarrativeInjuryContext.id(row)))
            if row["id"].string < original["id"].string { extras.append(row) }
        }
        XCTAssertEqual(extras.count, 6)
        let injury = item["injuryContext"].setting("observations", .list(extras + [original]))
        let plan = try MatchupNarrativeV5.prepare(item["context"], now: now, injuryContext: injury, injuryRoleContext: item["injuryRoleContext"])
        let report = try MatchupNarrativeV5.render(item["selection"], plan: plan)
        XCTAssertEqual(report["injuryContext"]["observations"].array.count, 6)
        XCTAssertFalse(report["injuryContext"]["observations"].array.contains { $0["id"] == original["id"] })
        XCTAssertEqual(report["roleStatusEvidence"].array.map { $0["id"] }, [original["id"]])
        let fingerprint = RichSportsContext.digest(.object(["sports": plan["contextFingerprint"], "revision": plan["contextRevision"],
            "injury": MatchupNarrativeV4.injuryIdentity(plan["injuryContext"]), "role": MatchupNarrativeV5.roleIdentity(plan["injuryRoleContext"]),
            "roleStatusIDs": .list(plan["roleStatusEvidence"].array.map { $0["claimID"] })]))
        let wire = item["envelope"].setting("injuryContext", injury).setting("report", report).setting("sourceFingerprint", .string(fingerprint))
        let result = try MatchupNarrativeV5.validate(Data(wire.canonical.utf8), now: now, versions: versions(root))
        let evidence = try projected(result, now), statuses = evidence.filter { $0.id.hasPrefix("narrative-injury|") }.map(\.text).joined(separator: "\n")
        for row in extras + [original] {
            let label = row["playerName"].string + ": reported status " + row["reportedStatus"].string
            XCTAssertEqual(statuses.components(separatedBy: label).count - 1, 1)
        }
        XCTAssertTrue(statuses.contains("Scope: general. Confirmed for this game: no"))
        XCTAssertTrue(evidence.filter { $0.id.hasPrefix("narrative-injury|") }.allSatisfy { $0.source == "ESPN" })
        XCTAssertTrue(evidence.contains { $0.text.contains("1 eligible injury observations omitted from the general injury summary") })
        XCTAssertNoThrow(try AskPLONNCore.boundedRequest(.init(question: "Any injuries?", intent: .recentForm, evidence: evidence)))
        // The unmodified fixture has the same row in both projections; it appears once.
        let (ordinary, ordinaryNow) = try fixture("nfl-accepted_conditional")
        let ordinaryStatuses = try projected(ordinary, ordinaryNow).filter { $0.id.hasPrefix("narrative-injury|") }.map(\.text).joined()
        XCTAssertEqual(ordinaryStatuses.components(separatedBy: "Example Player: reported status Out").count - 1, 1)
    }

    func testV5StatusDiscoveryUsesReaderWithoutLegacyOrHistory() async throws {
        for name in ["recorded-nfl-offline-selection", "recorded-nba-offline-selection", "recorded-mlb-offline-selection", "recorded-nhl-offline-selection"] {
            let (result, now) = try fixture(name)
            let researched = try await offlineService().discover(question: "Any injuries?", markets: [], chosen: .game(result.event),
                report: { _ in XCTFail("Status does not fall back to numerical legacy reports"); throw SharedMatchupReportError.unavailable },
                narrative: { event in XCTAssertEqual(event, result.event); return result }, now: now)
            XCTAssertEqual(researched.target, .game(result.event))
            XCTAssertTrue(researched.evidence.contains { $0.id.hasPrefix("narrative-role-limit|") })
            XCTAssertFalse(researched.evidence.contains { $0.id.hasPrefix("team|") })
        }
    }

    func testExpiredV5ReaderCannotSupplyStatusOrRoleEvidence() async throws {
        let root = try corpus(), item = root["cases"].array[0], (result, now) = try fixture(item["id"].string)
        let expired = now.addingTimeInterval(21600), pins = versions(root)
        let researched = try await offlineService().discover(question: "Any injuries?", markets: [], chosen: .game(result.event),
            report: { _ in XCTFail("Rejected status does not fall back to scoring"); throw SharedMatchupReportError.unavailable },
            narrative: { _ in try MatchupNarrativeV5.validate(Data(item["canonicalUTF8"].string.utf8), now: expired, versions: pins) }, now: expired)
        XCTAssertFalse(researched.evidence.contains { $0.availability == .available })
        XCTAssertTrue(researched.evidence.contains { $0.id == "narrative-unavailable" })
        XCTAssertTrue(researched.evidence.contains { $0.id == "narrative-status-gap" })
    }
}
