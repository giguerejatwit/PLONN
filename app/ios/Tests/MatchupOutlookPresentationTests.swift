import XCTest
@testable import PLONNCore

final class MatchupOutlookPresentationTests: XCTestCase {
    typealias J = RichContextJSON
    private var ios: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent() }
    private func json(_ url: URL) throws -> J {
        var parser = RichContextJSONParser(try Data(contentsOf: url))
        return try parser.parse()
    }
    private func assertReadable(_ text: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertFalse(text.localizedCaseInsensitiveContains("sample"), text, file: file, line: line)
        XCTAssertFalse(text.contains("window"), text, file: file, line: line)
        XCTAssertFalse(text.contains("ESPN"), text, file: file, line: line)
        XCTAssertNil(text.range(of: #"\d{4}-\d{2}-\d{2}T"#, options: .regularExpression), text, file: file, line: line)
    }

    func testFourActualReportsKeepRatesConditionsInjuriesAndOriginalBytes() throws {
        let archive = try json(ios.appendingPathComponent("PLONN/MatchupNarrativeReports-v4.json"))
        let pins = try XCTUnwrap(MatchupNarrativeArchive.trustedVersions)
        var sports = Set<String>()
        for entry in archive["reports"].array {
            let raw = try XCTUnwrap(Data(base64Encoded: entry.string))
            var parser = RichContextJSONParser(raw)
            let envelope = try parser.parse()
            let now = try SharedMatchupReportValidator.date(envelope["generatedAt"].string)
            let result = try MatchupNarrativeV4.validate(raw, now: now, versions: pins)
            sports.insert(result.event.sport)
            let originalClaims = result.claims
            for claim in result.claims {
                let shown = MatchupOutlookPresentation.text(for: claim, sport: result.event.sport)
                assertReadable(shown)
                for fact in result.context["facts"].array where claim.factIDs.contains(fact["id"].string) {
                    let number = MatchupNarrativeV4.number(fact["value"])
                    if claim.text.contains(number) { XCTAssertTrue(shown.contains(number), "Lost rate \(number): \(shown)") }
                }
                if let condition = claim.text.range(of: "If ") {
                    XCTAssertTrue(shown.contains(String(claim.text[condition.lowerBound...])), "Changed conditional conclusion")
                }
                if result.event.sport == "NHL", claim.id == "home:shotsPerGame" {
                    XCTAssertEqual(shown, "Washington Capitals recorded 27.6 shots on goal per game; Montreal Canadiens allowed 28.2 shots on goal per game. Without a comparable same-period benchmark, these rates describe the shots on goal matchup but do not establish an advantage or forecast.")
                }
                assertReadable(MatchupOutlookPresentation.uncertainty(claim.uncertainty))
            }
            for note in MatchupOutlookPresentation.notes(for: result) { assertReadable(note) }
            XCTAssertEqual(result.missingContext, envelope["report"]["missingContext"].array.map(\.string))
            XCTAssertEqual(result.claims, originalClaims)
            XCTAssertEqual(result.rawData, raw)
            XCTAssertEqual(result.injuryContext, envelope["report"]["injuryContext"])
            XCTAssertEqual(result.designations, envelope["report"]["designations"])
            XCTAssertEqual(try MatchupNarrativeV4.validate(raw, now: now, versions: pins).report, result.report)
        }
        XCTAssertEqual(sports, ["NFL", "NBA", "MLB", "NHL"])
    }

    func testEverySyntheticClaimKeepsConditionalAndLimitedHistoryMeaning() throws {
        let fixture = try json(ios.deletingLastPathComponent().appendingPathComponent("contracts/matchup-narrative-v4.json"))
        var sawLimited = false, sawSpecialTeams = false, sawAttemptMix = false
        for row in fixture["cases"].array {
            let sport = row["envelope"]["event"]["sport"].string
            for raw in row["expectedClaims"].array {
                let claim = MatchupNarrativeV4.Claim(raw)
                let shown = MatchupOutlookPresentation.text(for: claim, sport: sport)
                assertReadable(shown)
                if claim.kind == "limited_sample" {
                    sawLimited = true
                    XCTAssertTrue(shown.contains("too little history"))
                    XCTAssertTrue(shown.contains("describe past games only"))
                    XCTAssertFalse(shown.contains("-game rule"))
                }
                if claim.id.contains("special-teams") {
                    sawSpecialTeams = true
                    XCTAssertTrue(shown.contains("different measures"))
                    XCTAssertTrue(shown.contains("do not predict penalty volume"))
                }
                if claim.id.contains("attempt-mix") {
                    sawAttemptMix = true
                    XCTAssertTrue(shown.contains("excludes sacks and does not establish tempo or yardage"))
                }
                XCTAssertEqual(claim.text, raw["text"].string)
                XCTAssertEqual(claim.samples, raw["samples"])
            }
        }
        XCTAssertTrue(sawLimited && sawSpecialTeams && sawAttemptMix)
    }

    func testAll91LegacyReportsKeepStatisticsWarningsAndInternalProjection() throws {
        let fixture = try json(ios.deletingLastPathComponent().appendingPathComponent("contracts/shared-insights-presentation-v1.json"))
        XCTAssertEqual(fixture["cases"].array.count, 91)
        var sports = Set<String>()
        for item in fixture["cases"].array {
            let raw = Data(item["rawUTF8"].string.utf8)
            let saved = try JSONDecoder().decode(SharedMatchupReportResponse.self, from: raw)
            let now = try SharedMatchupReportValidator.date(item["validatedAt"].string)
            let response = try SharedMatchupReportValidator.validate(raw, event: saved.event, now: now)
            let original = SharedMatchupPresentationCopy.sections(response)
            let shown = MatchupOutlookPresentation.sections(response)
            sports.insert(response.event.sport)
            XCTAssertEqual(shown.map(\.id), original.map(\.id))
            for (display, source) in zip(shown, original) {
                XCTAssertEqual(display.title, source.title)
                XCTAssertEqual(display.uncertainty, source.uncertainty)
                let text = ([display.takeaway] + display.evidenceLines + [display.comparison ?? ""]).joined(separator: " ")
                assertReadable(text)
                for fact in response.snapshot?.facts ?? [] where source.factIDs.contains(fact.id) {
                    XCTAssertTrue(text.contains(SharedMatchupPresentationCopy.number(fact.value)), "Lost rate for \(fact.id)")
                }
            }
            XCTAssertEqual(response.report, saved.report)
            XCTAssertEqual(response.snapshot, saved.snapshot)
            XCTAssertEqual(response.generatedAt, saved.generatedAt)
            XCTAssertEqual(SharedMatchupPresentationCopy.sections(response), original)
        }
        XCTAssertEqual(sports, ["NFL", "NBA", "MLB", "NHL"])
    }

    func testUnknownAndUnboundProseStaysUnchanged() {
        let text = "4.5 points in 7 games. Reported 2026-10-13T10:20:30Z. Unknown sample window."
        let claim = MatchupNarrativeV4.Claim(.object(["text": .string(text), "factIDs": .list([]), "samples": .list([
            .object(["factID": .string("unbound"), "sample": .object(["count": .number("7"), "unit": .string("games")])])
        ])]))
        XCTAssertEqual(MatchupOutlookPresentation.text(for: claim, sport: "NFL"), text)
        XCTAssertEqual(MatchupOutlookPresentation.uncertainty("A new uncertainty."), "A new uncertainty.")
    }

    func testInjuryRowsKeepReportedScopeAndDoNotImplyAvailability() {
        let context = J.object(["teams": .list([.object(["id": .string("team"), "name": .string("Home")])])])
        for scope in ["general", "practice", "roster"] {
            let row = J.object(["teamID": .string("team"), "playerName": .string("Player"), "reportedStatus": .string("Questionable"), "scope": .string(scope), "eventStatusConfirmed": .bool(false)])
            let text = MatchupOutlookPresentation.injuryText(row, context: context)
            XCTAssertEqual(text, "Home · Player: Questionable · \(scope) report; game status unconfirmed")
            XCTAssertFalse(text.contains("Confirmed"))
        }
        let confirmed = J.object(["playerName": .string("Player"), "reportedStatus": .string("Out"), "scope": .string("game"), "eventStatusConfirmed": .bool(true)])
        XCTAssertEqual(MatchupOutlookPresentation.injuryText(confirmed, context: context), "Player: Out · game report for this game")
    }

    func testOpponentHistoryRetainsSmallHistoryAndIncompleteCoverageWarnings() {
        let limited = MatchupOutlookPresentation.opponentCaution(count: 2, incomplete: true)
        XCTAssertTrue(limited.contains("little history"))
        XCTAssertTrue(limited.contains("Some past seasons could not be loaded"))
        XCTAssertTrue(limited.contains("not settled bet results or a prediction"))
        let complete = MatchupOutlookPresentation.opponentCaution(count: 10, incomplete: false)
        XCTAssertFalse(complete.contains("little history"))
        XCTAssertFalse(complete.contains("could not be loaded"))
        XCTAssertTrue(complete.contains("not settled bet results or a prediction"))
    }

    func testActualMLBStartersStayProbableAndGapFillerIsAbsent() throws {
        let raw = try Data(contentsOf: ios.appendingPathComponent("verification/narrative-v4-20261014/actual/mlb-response.json"))
        var parser = RichContextJSONParser(raw)
        let envelope = try parser.parse()
        let now = try SharedMatchupReportValidator.date(envelope["generatedAt"].string)
        let result = try MatchupNarrativeV4.validate(raw, now: now, versions: XCTUnwrap(MatchupNarrativeArchive.trustedVersions))
        let starters = MatchupOutlookPresentation.starters(result.designations, event: result.event)
        XCTAssertEqual(Set(starters.map(\.name)), ["Blake Snell", "Robert Gasser"])
        XCTAssertEqual(starters.map(\.status), ["Probable", "Probable"])
        XCTAssertTrue(MatchupOutlookPresentation.notes(for: result).isEmpty)
        XCTAssertEqual(result.missingContext.count, 8, "Gap evidence must remain internally")
        XCTAssertEqual(result.rawData, raw)
        XCTAssertEqual(result.designations, envelope["report"]["designations"])
    }

    func testOnlyBoundNamedProbableOrConfirmedStartersAreVisibleAcrossSports() {
        for (sport, kind) in ["MLB": "starting_pitcher", "NHL": "starting_goalie", "NFL": "starting_lineup", "NBA": "starting_lineup"] {
            let event = SharedMatchupEvent(sport: sport, eventID: "espn:1", homeTeam: "Home", awayTeam: "Away", startISO: "2026-10-15T20:00:00Z")
            let row = J.object(["id": .string("starter:1"), "kind": .string(kind), "eventID": .string(event.eventID), "team": .string("Home"), "state": .string("probable"), "subject": .object(["name": .string("Pitcher")])])
            XCTAssertEqual(MatchupOutlookPresentation.starters(.list([row]), event: event).first?.status, "Probable")
            XCTAssertEqual(MatchupOutlookPresentation.starters(.list([row.setting("state", .string("confirmed"))]), event: event).first?.status, "Confirmed")
            for invalid in [row.setting("state", .string("unknown")), row.setting("subject", .null), row.setting("eventID", .string("espn:2")), row.setting("team", .string("Other")), row.setting("kind", .string("unsupported"))] {
                XCTAssertTrue(MatchupOutlookPresentation.starters(.list([invalid]), event: event).isEmpty)
            }
        }
    }

    func testUnsupportedContextIsOmittedButPreseasonQualificationRemains() throws {
        let archive = try json(ios.appendingPathComponent("PLONN/MatchupNarrativeReports-v4.json"))
        for entry in archive["reports"].array {
            let raw = try XCTUnwrap(Data(base64Encoded: entry.string))
            var parser = RichContextJSONParser(raw)
            let envelope = try parser.parse()
            let now = try SharedMatchupReportValidator.date(envelope["generatedAt"].string)
            let result = try MatchupNarrativeV4.validate(raw, now: now, versions: XCTUnwrap(MatchupNarrativeArchive.trustedVersions))
            let notes = MatchupOutlookPresentation.notes(for: result)
            XCTAssertEqual(notes, result.event.sport == "NBA" ? ["These are preseason results; lineups and minutes can change."] : [])
            XCTAssertFalse(notes.contains { $0.contains("incomplete") || $0.contains("not available") })
            XCTAssertFalse(result.missingContext.isEmpty)
            XCTAssertEqual(result.rawData, raw)
        }
    }

    func testSupportingContextKeepsTwoPairedComparisonsWithoutGapFiller() {
        func insight(_ id: String, paired: Bool) -> MatchupInsight {
            .init(id: id, text: paired ? "Team averages 4.5; opponent allows 3.8." : "Matching defensive rate is unavailable.", evidence: "Original ESPN evidence and dates", marketID: nil, topic: "Scoring", team: id, hasPairedEvidence: paired)
        }
        let original = [insight("partial", paired: false), insight("home", paired: true), insight("away", paired: true), insight("extra", paired: true)]
        let shown = MatchupOutlookPresentation.supportingInsights(original)
        XCTAssertEqual(shown, Array(original[1...2]))
        XCTAssertTrue(shown.allSatisfy { !$0.text.contains("unavailable") })
        XCTAssertEqual(original.count, 4)
        XCTAssertTrue(MatchupOutlookPresentation.supportingInsights([original[0]]).isEmpty)
    }
}
