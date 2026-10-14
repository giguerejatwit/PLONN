import XCTest
@testable import PLONNCore

final class AskPLONNNarrativeContextTests: XCTestCase {
    typealias J = RichContextJSON
    func corpus() throws -> J {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("contracts/matchup-narrative-v4.json")
        var parser = RichContextJSONParser(try Data(contentsOf: url))
        return try parser.parse()
    }
    func versions(_ root: J) -> [String: String] {
        guard case .object(let values) = root["versions"] else { return [:] }
        return values.mapValues(\.string)
    }
    func fixture(_ name: String) throws -> (MatchupNarrativeV4.Result, Date) {
        let root = try corpus(), value = try XCTUnwrap(root["cases"].array.first { $0["id"].string == name })
        let now = Date(timeIntervalSince1970: try XCTUnwrap(Double(value["now"].number ?? "")))
        return (try MatchupNarrativeV4.validate(Data(value["envelope"].canonical.utf8), now: now, versions: versions(root)), now)
    }
    func offlineService() -> AskPLONNContextService {
        .init(read: { _ in XCTFail("A chosen game should not read discovery sources"); throw PlayerHistoryError.invalidPayload },
              player: { _, _ in XCTFail("A game request should not read player history"); return .init(games: [], opponentGames: [], context: nil, source: "", warnings: [], availability: "") },
              team: { _, _ in .init(summary: nil, rows: [], observations: [], source: "ESPN", warnings: [], availability: "No team history in fixture") })
    }
    func testAllFourSportsProjectExactValidatedClaimTextAndUncertainty() throws {
        let root = try corpus()
        var sports = Set<String>()
        for item in root["cases"].array {
            let now = Date(timeIntervalSince1970: try XCTUnwrap(Double(item["now"].number ?? "")))
            let result: MatchupNarrativeV4.Result
            do { result = try MatchupNarrativeV4.validate(Data(item["envelope"].canonical.utf8), now: now, versions: versions(root)) }
            catch { XCTFail("Corpus validation failed before chat projection: \(item["id"].string): \(error)"); continue }
            let evidence = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: "What can we expect?", now: now)
            let available = evidence.filter { $0.availability == .available }
            XCTAssertEqual(available.map(\.text), result.claims.map { $0.text + "\n\n" + $0.uncertainty })
            XCTAssertTrue(available.allSatisfy { $0.capturedAt == now && $0.source.contains("ESPN") })
            for gap in result.missingContext { XCTAssertTrue(evidence.contains { $0.availability == .unavailable && $0.text.contains(gap) }, gap) }
            XCTAssertNoThrow(try AskPLONNCore.boundedRequest(.init(question: "What can we expect?", intent: .recentForm, evidence: evidence)))
            sports.insert(result.event.sport)
        }
        XCTAssertEqual(sports, Set(["NFL", "NBA", "NHL", "MLB"]))
    }
    func testMLBHighStrikeoutsUsesStrikeoutClaimWithoutHitsOrRunsSubstitution() throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        let facts = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: "High strikeouts?", now: now)
        XCTAssertEqual(facts.filter { $0.availability == .available }.map(\.text), result.claims.filter { $0.dimension == "strikeouts" }.map { $0.text + "\n\n" + $0.uncertainty })
        XCTAssertEqual(facts.filter { $0.availability == .available }.count, 1)
    }
    func testRequestedDimensionsMapAcrossSportsAndKeepMissingRequestedDimension() throws {
        let cases = [("nfl_neutral_scoring_rich_dimensions", "Passing and pace", "passing"),
                     ("nhl_shots_and_special_teams", "Power play and saves", "special_teams"),
                     ("nba_recorded_pace_and_efficiency", "Pace and rebounds", "pace")]
        for (name, question, dimension) in cases {
            let (result, now) = try fixture(name)
            let facts = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: question, now: now)
            XCTAssertEqual(facts.filter { $0.availability == .available }.map(\.text), result.claims.filter { $0.dimension == dimension }.map { $0.text + "\n\n" + $0.uncertainty })
            XCTAssertTrue(facts.contains { $0.id == "narrative-requested-gap" })
        }
    }
    func testPreseasonAndSmallSampleWordingRemainExact() throws {
        let (result, now) = try fixture("nba_preseason_limited_no_pace")
        let facts = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: "Scoring?", now: now)
        XCTAssertTrue(facts.contains { $0.text.contains("descriptive only") })
        XCTAssertTrue(facts.contains { $0.text.contains("Preseason samples") && $0.availability == .unavailable })
        let pace = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: "Is the pace high?", now: now)
        XCTAssertFalse(pace.contains { $0.availability == .available })
    }
    func testInjuryAndStarterQuestionsDoNotBecomeScoringEvidence() throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        for question in ["Any injuries?", "Who is starting?", "Is the starter confirmed?"] {
            let facts = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: question, now: now)
            XCTAssertFalse(facts.contains { $0.availability == .available })
            XCTAssertTrue(facts.contains { $0.id == "narrative-requested-gap" })
        }
    }
    func testValidNarrativeWinsWithoutReadingLegacyOrHistory() async throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        let researched = try await offlineService().discover(question: "High strikeouts?", markets: [], chosen: .game(result.event),
            report: { _ in XCTFail("Valid matching v4 should not fetch legacy"); throw SharedMatchupReportError.unavailable },
            narrative: { event in XCTAssertEqual(event, result.event); return result }, now: now)
        XCTAssertTrue(researched.evidence.contains { $0.id.contains("strikeoutsPerGame") && $0.availability == .available })
        XCTAssertFalse(researched.evidence.contains { $0.id.hasPrefix("team|") })
    }
    func testNilNarrativeKeepsLegacyBehavior() async throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        let counter = NarrativeReadCounter()
        let researched = try await offlineService().discover(question: "High strikeouts?", markets: [], chosen: .game(result.event),
            report: { event in await counter.hit(); return Self.missing(event) }, narrative: { _ in nil }, now: now)
        let calls = await counter.value(); XCTAssertEqual(calls, 1)
        XCTAssertFalse(researched.evidence.contains { $0.id.hasPrefix("narrative") })
        XCTAssertFalse(researched.evidence.contains { $0.availability == .available })
    }
    func testRejectedNarrativeFallsBackWithVisibleUnverifiedNotice() async throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        let counter = NarrativeReadCounter()
        let researched = try await offlineService().discover(question: "What can we expect?", markets: [], chosen: .game(result.event),
            report: { event in await counter.hit(); return Self.missing(event) },
            narrative: { _ in throw SharedMatchupReportError.invalidResponse }, now: now)
        let calls = await counter.value(); XCTAssertEqual(calls, 1)
        XCTAssertTrue(researched.evidence.contains { $0.id == "narrative-unavailable" && $0.availability == .unavailable })
    }
    func testUncoveredDimensionFallsBackWithoutDiscardingV4Gaps() async throws {
        let (result, now) = try fixture("nba_preseason_limited_no_pace")
        let counter = NarrativeReadCounter()
        let researched = try await offlineService().discover(question: "Is pace high?", markets: [], chosen: .game(result.event),
            report: { event in await counter.hit(); return Self.missing(event) }, narrative: { _ in result }, now: now)
        let calls = await counter.value(); XCTAssertEqual(calls, 1)
        XCTAssertTrue(researched.evidence.contains { $0.id == "narrative-requested-gap" })
        XCTAssertTrue(researched.evidence.contains { $0.text.contains("Preseason samples") })
        XCTAssertFalse(researched.evidence.contains { $0.availability == .available })
    }
    func testCanceledNarrativeDoesNotFallBackOrSupplyEvidence() async throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        do {
            _ = try await offlineService().discover(question: "High strikeouts?", markets: [], chosen: .game(result.event),
                report: { _ in XCTFail("Cancellation must not fall back"); throw SharedMatchupReportError.unavailable },
                narrative: { _ in throw CancellationError() }, now: now)
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
    }
    func testUnresolvedDiscoveryDoesNotReadNarrativeOrLegacy() async throws {
        let service = AskPLONNContextService(read: { _ in throw PlayerHistoryError.invalidPayload })
        let result = try await service.discover(question: "MLB game today", markets: [],
            report: { _ in XCTFail("No exact event: legacy reader must not run"); throw SharedMatchupReportError.unavailable },
            narrative: { _ in XCTFail("No exact event: narrative reader must not run"); return nil })
        XCTAssertNotNil(result.guidance)
        XCTAssertNil(result.target)
        XCTAssertTrue(result.evidence.isEmpty)
    }
    func testOtherEventAndStartedEventCannotSupplyV4Facts() throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        let wrong = SharedMatchupEvent(sport: result.event.sport, eventID: "espn:9999", homeTeam: result.event.homeTeam, awayTeam: result.event.awayTeam, startISO: result.event.startISO)
        XCTAssertThrowsError(try AskPLONNContextService.narrativeEvidence(result, event: wrong, question: "Hits", now: now))
        XCTAssertThrowsError(try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: "Hits", now: SharedMatchupReportValidator.date(result.event.startISO)))
        XCTAssertThrowsError(try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: "Hits", now: now.addingTimeInterval(-1)))
    }
    func testReadTimeValidationRejectsExpiredSourceBeforeProjection() async throws {
        let root = try corpus(), c = try XCTUnwrap(root["cases"].array.first { $0["id"].string == "mlb_hits_and_batting_strikeouts" })
        let (result, now) = try fixture(c["id"].string), pins = versions(root)
        let expired = now.addingTimeInterval(2 * 86400)
        let researched = try await offlineService().discover(question: "High strikeouts?", markets: [], chosen: .game(result.event),
            report: { event in Self.missing(event) }, narrative: { _ in
                try MatchupNarrativeV4.validate(Data(c["envelope"].canonical.utf8), now: expired, versions: pins)
            }, now: expired)
        XCTAssertFalse(researched.evidence.contains { $0.availability == .available })
        XCTAssertTrue(researched.evidence.contains { $0.id == "narrative-unavailable" })
    }
    var appRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    func readJSON(_ path: String) throws -> J {
        var parser = RichContextJSONParser(try Data(contentsOf: appRoot.appendingPathComponent(path)))
        return try parser.parse()
    }
    func actualNarrative(_ sport: String) throws -> (MatchupNarrativeV4.Result, Date) {
        let raw = try Data(contentsOf: appRoot.appendingPathComponent("ios/verification/narrative-v4-20261014/actual/\(sport)-response.json"))
        var parser = RichContextJSONParser(raw)
        let wire = try parser.parse(), now = try SharedMatchupReportValidator.date(wire["generatedAt"].string)
        return (try MatchupNarrativeV4.validate(raw, now: now, versions: XCTUnwrap(MatchupNarrativeArchive.trustedVersions)), now)
    }
    func testActualNBAStaleInjuryContextIsVisibleForGeneralAndStatusQuestions() throws {
        let (result, now) = try actualNarrative("nba")
        XCTAssertEqual(result.injuryContext["state"].string, "stale")
        for question in ["What can we expect?", "Any injuries?"] {
            let evidence = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: question, now: now)
            let limitations = evidence.filter { $0.id.hasPrefix("narrative-injury-limit") }
            XCTAssertTrue(limitations.allSatisfy { $0.availability == .stale && $0.source == "ESPN" })
            let text = limitations.map(\.text).joined(separator: "\n")
            XCTAssertTrue(text.contains("earlier injury reports are stale and excluded"))
            XCTAssertTrue(text.contains(result.injuryContext["interpretation"].string))
            for exclusion in result.injuryContext["exclusions"].array {
                XCTAssertTrue(text.contains("\(exclusion["excludedCount"].integer!) injury observations excluded"))
            }
            XCTAssertFalse(evidence.contains { $0.id.hasPrefix("narrative-injury|") && $0.availability == .available })
            if question == "Any injuries?" { XCTAssertFalse(evidence.contains { $0.availability == .available }) }
        }
    }
    func testActualNFLStatusPreservesReportedScopeAndEveryCoverageLimit() throws {
        let (result, now) = try actualNarrative("nfl")
        let evidence = try AskPLONNContextService.narrativeEvidence(result, event: result.event, question: "Who is questionable or out?", now: now)
        let available = evidence.filter { $0.availability == .available }
        XCTAssertFalse(available.isEmpty)
        XCTAssertTrue(available.allSatisfy { $0.id.hasPrefix("narrative-injury|") && $0.source == "ESPN" })
        let text = available.map(\.text).joined(separator: "\n")
        for row in result.injuryContext["observations"].array {
            for value in [row["playerName"].string, row["reportedStatus"].string, "Scope: \(row["scope"].string)", row["updatedAt"].string] { XCTAssertTrue(text.contains(value), value) }
            let team = try XCTUnwrap(result.injuryContext["teams"].array.first { $0["id"] == row["teamID"] })
            XCTAssertTrue(text.contains(team["name"].string + " · " + row["playerName"].string))
            XCTAssertEqual(row["eventStatusConfirmed"], .bool(false))
        }
        XCTAssertTrue(text.contains("Confirmed for this game: no"))
        XCTAssertFalse(text.contains("Confirmed for this game: yes"))
        let limits = evidence.filter { $0.availability != .available }.map(\.text).joined(separator: "\n")
        XCTAssertTrue(limits.contains(result.injuryContext["interpretation"].string))
        for coverage in result.injuryContext["coverage"].array {
            XCTAssertTrue(limits.contains("\(coverage["excludedRows"].integer!) unsupported source rows excluded"))
        }
        for exclusion in result.injuryContext["exclusions"].array where exclusion["promptOmittedCount"].integer! > 0 {
            XCTAssertTrue(limits.contains("\(exclusion["promptOmittedCount"].integer!) eligible injury observations omitted"))
        }
        XCTAssertFalse(evidence.contains { $0.id == "narrative-requested-gap" })
        XCTAssertNoThrow(try AskPLONNCore.boundedRequest(.init(question: "Any injuries?", intent: .recentForm, evidence: evidence)))
    }
    func testActualStatusDiscoveryDoesNotReadLegacyEvenWhenAllInjuriesAreStale() async throws {
        for sport in ["nba", "nfl"] {
            let (result, now) = try actualNarrative(sport)
            let researched = try await offlineService().discover(question: "Any injuries?", markets: [], chosen: .game(result.event),
                report: { _ in XCTFail("Status-only requests must not fall back to numerical reports"); throw SharedMatchupReportError.unavailable },
                narrative: { _ in result }, now: now)
            XCTAssertFalse(researched.evidence.contains { $0.id.hasPrefix("team|") || $0.id.hasPrefix("narrative|\(result.event.sport)") })
            XCTAssertTrue(researched.evidence.contains { $0.id.hasPrefix("narrative-injury-limit") })
        }
    }
    func testAbsentOrRejectedStatusContextDoesNotFallBackToScoring() async throws {
        let (result, now) = try fixture("mlb_hits_and_batting_strikeouts")
        XCTAssertEqual(AskPLONNContextService.narrativeDimensions("Is the player playing?", sport: "MLB"), ["availability"])
        XCTAssertEqual(AskPLONNContextService.narrativeDimensions("Is he healthy?", sport: "NBA"), ["availability"])
        for reject in [false, true] {
            let researched = try await offlineService().discover(question: "Who is starting?", markets: [], chosen: .game(result.event),
                report: { _ in XCTFail("Numerical legacy reports cannot supply starters"); throw SharedMatchupReportError.unavailable },
                narrative: { _ in if reject { throw SharedMatchupReportError.invalidResponse }; return nil }, now: now)
            XCTAssertFalse(researched.evidence.contains { $0.availability == .available })
            XCTAssertTrue(researched.evidence.contains { $0.id == "narrative-status-gap" })
        }
    }
    func testFourSportsUseOnlyEligibleInjuryObservationsFromActualPublicPackets() throws {
        let base = "backend/local_lab/verification/injuries-20261014/"
        let receipt = try readJSON(base + "source-and-projection-receipt.json")
        let now = Date(timeIntervalSince1970: try NarrativeInjuryContext.time(receipt["evaluationClock"]))
        for name in ["nfl-401873001", "nba-401898404", "mlb-401908032", "nhl-401892504"] {
            let packet = try readJSON(base + name + "-packet.json")
            let projected = try NarrativeInjuryContext.model(packet, now: now)
            let evidence = try AskPLONNContextService.injuryEvidence(projected, generated: now)
            let text = evidence.filter { $0.availability == .available }.map(\.text).joined(separator: "\n")
            for row in projected["observations"].array {
                XCTAssertTrue(text.contains(row["playerName"].string))
                XCTAssertTrue(text.contains(row["reportedStatus"].string))
            }
            if projected["observations"].array.isEmpty { XCTAssertTrue(text.isEmpty) }
            let aged = try NarrativeInjuryContext.model(packet, now: now.addingTimeInterval(21600))
            XCTAssertTrue(aged["observations"].array.isEmpty)
            let stale = try AskPLONNContextService.injuryEvidence(aged, generated: now.addingTimeInterval(21600))
            XCTAssertFalse(stale.contains { $0.availability == .available })
            XCTAssertTrue(stale.contains { $0.availability == .stale })
        }
    }
    func testOfficialGameConfirmationAndEstimatedPracticeRemainDistinct() throws {
        let fixtures = try readJSON("backend/local_lab/verification/injuries-20261014/synthetic-contract-fixtures.json")
        let now = Date(timeIntervalSince1970: try NarrativeInjuryContext.time(fixtures["evaluationClock"]))
        for (sport, scope, status, kind, url) in [
            ("NBA", "game", "Questionable", "questionable", "https://official.nba.com/nba-injury-report-2026-27-season/"),
            ("NFL", "practice", "Limited participation", "practice", "https://www.nfl.com/injuries/")
        ] {
            var packet = try XCTUnwrap(fixtures["fixtures"].array.first { $0["sport"].string == sport })["packet"]
            let stamp = fixtures["evaluationClock"]
            let source = packet["sources"].array[0].setting("provider", .string("Official \(sport)"))
                .setting("url", .string(url)).setting("sourceIssuedAt", stamp).setting("feedUpdatedAt", .null)
                .setting("retrievedAt", stamp).setting("checkedAt", stamp)
            var row = packet["observations"].array[0].setting("scope", .string(scope)).setting("reportedStatus", .string(status))
                .setting("statusKind", .string(kind)).setting("timeBasis", .string("source_report"))
                .setting("playerID", .string(sport.lowercased() + ":example-player")).setting("updatedAt", stamp)
                .setting("estimated", .bool(scope == "practice"))
            if scope == "game" { row = row.setting("eventID", packet["event"]["eventID"]).setting("eventStartISO", packet["event"]["startISO"]) }
            row = row.setting("id", .string(NarrativeInjuryContext.id(row)))
            packet = packet.setting("sources", .list([source])).setting("observations", .list([row]))
            let view = try NarrativeInjuryContext.model(packet, now: now)
            let facts = try AskPLONNContextService.injuryEvidence(view, generated: now).filter { $0.availability == .available }
            let text = facts.map(\.text).joined()
            XCTAssertTrue(text.contains("Scope: \(scope)"))
            XCTAssertTrue(text.contains("Confirmed for this game: \(scope == "game" ? "yes" : "no")"))
            XCTAssertEqual(text.contains("practice report is estimated"), scope == "practice")
            XCTAssertTrue(facts.allSatisfy { $0.source == "Official \(sport)" })
        }
    }
    func testActualStarterDesignationsRetainProbableConfirmedAndUnknown() throws {
        var states = Set<String>()
        for name in ["recorded-mlb-context", "recorded-mlb-enriched-context", "recorded-nhl-context", "recorded-nhl-enriched-context"] {
            let context = try readJSON("backend/local_lab/verification/sports-context-20261014/\(name).json")
            let now = try XCTUnwrap(context["sources"].array.map { try RichSportsContext.time($0["checkedAt"]) }.max())
            let projected = try RichSportsContext.project(Data(context.canonical.utf8), now: now)
            let facts = try AskPLONNContextService.designationEvidence(projected["designations"], sources: projected["sources"], generated: now)
            for row in projected["designations"].array {
                let state = row["state"].string; states.insert(state)
                let matching = try XCTUnwrap(facts.first { $0.text.contains(row["team"].string) && $0.text.contains("Designation: \(state)") })
                XCTAssertEqual(matching.availability, state == "unknown" ? .unavailable : .available)
                if state == "probable" { XCTAssertTrue(matching.text.contains("Probable is not confirmed.")) }
                if row["subject"] != .null { XCTAssertTrue(matching.text.contains(row["subject"]["name"].string)) }
                for source in projected["sources"].array where row["sourceIDs"].array.contains(source["id"]) { XCTAssertTrue(matching.source.contains(source["provider"].string)) }
            }
            let aged = try RichSportsContext.project(Data(context.canonical.utf8), now: now.addingTimeInterval(21600))
            XCTAssertTrue(aged["designations"].array.isEmpty)
        }
        XCTAssertEqual(states, Set(["confirmed", "probable", "unknown"]))
    }
    func testAllFourActualEnvelopesSupplyBoundedGeneralStatusAndStarterResearch() async throws {
        for sport in ["nfl", "nba", "mlb", "nhl"] {
            let (result, now) = try actualNarrative(sport)
            for question in ["What can we expect?", "Any injuries?", "Who is starting?"] {
                let researched = try await offlineService().discover(question: question, markets: [], chosen: .game(result.event),
                    report: { _ in XCTFail("Validated v4 or its explicit status limitation should handle this request"); throw SharedMatchupReportError.unavailable },
                    narrative: { _ in result }, now: now)
                XCTAssertNoThrow(try AskPLONNCore.boundedRequest(.init(question: question, intent: .recentForm, evidence: researched.evidence)))
                XCTAssertTrue(researched.evidence.contains { $0.id.hasPrefix("narrative-injury-limit") })
                for gap in result.missingContext { XCTAssertTrue(researched.evidence.contains { $0.text.contains(gap) }) }
                if question != "What can we expect?" {
                    XCTAssertFalse(researched.evidence.contains { $0.id.hasPrefix("narrative|\(result.event.sport)") || $0.id.hasPrefix("team|") })
                }
            }
        }
    }
    func testStatusDiscoveryResolvesExactGamesAcrossFourSportsAndPreservesAmbiguity() throws {
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        for sport in ["nfl", "nba", "mlb", "nhl"] {
            let (result, now) = try actualNarrative(sport), event = result.event
            let date = String(event.startISO.prefix(10))
            for question in ["\(event.sport) injuries \(date)", "Who is starting in the \(event.sport) game \(date)?"] {
                XCTAssertTrue(AskPLONNContextResolver.shouldDiscover(question))
                let catalog = AskPLONNContextCatalog(events: [event], teams: [], completeSports: [event.sport])
                guard case .resolved(.game(let found)) = AskPLONNContextResolver.resolve(question: question, catalog: catalog, now: now, calendar: calendar) else {
                    XCTFail("Status question must reach available game research: \(question)"); continue
                }
                XCTAssertEqual(found, event)
                let second = SharedMatchupEvent(sport: event.sport, eventID: "espn:999999", homeTeam: "Example Home", awayTeam: "Example Away", startISO: event.startISO)
                let ambiguous = AskPLONNContextCatalog(events: [event, second], teams: [], completeSports: [event.sport])
                guard case .clarification(_, let choices) = AskPLONNContextResolver.resolve(question: question, catalog: ambiguous, now: now, calendar: calendar) else {
                    XCTFail("Multiple games must stay ambiguous"); continue
                }
                XCTAssertEqual(choices.count, 2)
            }
        }
    }
    func testStatusFollowupReusesExactPreviousGameWithoutDiscovery() async throws {
        let (result, now) = try actualNarrative("nfl")
        for question in ["Any injuries?", "What about their injuries?", "Who is starting?"] {
            XCTAssertTrue(AskPLONNContextResolver.shouldDiscover(question, previous: .game(result.event)))
            let researched = try await offlineService().discover(question: question, markets: [], previous: .game(result.event),
                report: { _ in XCTFail("Status follow-up must not read numerical legacy report"); throw SharedMatchupReportError.unavailable },
                narrative: { event in XCTAssertEqual(event, result.event); return result }, now: now)
            XCTAssertEqual(researched.target, .game(result.event))
            XCTAssertNil(researched.guidance)
        }
        XCTAssertFalse(AskPLONNContextService.safeFollowup("What about Yankees injuries?"))
    }
    func testPlayerHistoryDoesNotEstablishCurrentAvailability() async throws {
        let researched = try await offlineService().discover(question: "Is the player healthy?", markets: [],
            chosen: .player(sport: "NBA", name: "Example Player", stat: "player_points"))
        XCTAssertFalse(researched.evidence.contains { $0.availability == .available })
        XCTAssertEqual(researched.evidence.map(\.id), ["player-status-gap"])
    }
    static func missing(_ event: SharedMatchupEvent) -> SharedMatchupReportResponse {
        .init(schemaVersion: 1, status: .missing, reason: "cache_miss", event: event, generatedAt: nil, sourceAsOf: nil, sourceRetrievedAt: nil, sourceCheckedAt: nil, sourceFingerprint: nil, versions: .current, refreshing: false, report: nil, snapshot: nil)
    }
}

private actor NarrativeReadCounter {
    private var count = 0
    func hit() { count += 1 }
    func value() -> Int { count }
}
