import XCTest
@testable import PLONNCore

final class MatchupSummaryValidationTests: XCTestCase {
    private let teams = ["Boston Bruins", "New York Rangers"]
    private let summary = "Boston Bruins goal production is compared with what New York Rangers have allowed. These observations describe different opponents. This is not a prediction and not a player forecast."

    private func fact(_ id: String = "home-goals", title: String = "Goals", text: String = "Boston Bruins average 3.2 goals per game; New York Rangers allow 2.8 goals per game.", kind: AskPLONNIntent = .recentForm, source: String = "ESPN recent completed games", availability: AskPLONNAvailability = .available) -> AskPLONNEvidence {
        AskPLONNEvidence(id: id, kind: kind, title: title, text: text, source: source,
                         capturedAt: Date(timeIntervalSince1970: 1_791_000_000), sampleSize: 5,
                         availability: availability, sourceURLs: ["https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"])
    }

    private func request(_ facts: [AskPLONNEvidence]? = nil, intent: AskPLONNIntent = .recentForm,
                         purpose: AskPLONNResearchPurpose? = .matchupInsights,
                         matchupTeams: [String]? = ["Boston Bruins", "New York Rangers"],
                         history: [AskPLONNQuestionContext]? = nil) -> AskPLONNRequest {
        AskPLONNRequest(question: "Explain the observed matchup evidence.", intent: intent,
                        evidence: facts ?? [fact()], ownerID: "fixture-owner", priorTurns: history,
                        researchPurpose: purpose, matchupTeams: matchupTeams)
    }

    private func answer(_ facts: [AskPLONNEvidence]? = nil, summary: String? = nil) -> AskPLONNAnswer {
        let selected = facts ?? [fact()]
        return AskPLONNAnswer(text: selected.map(\.text).joined(separator: "\n\n"), evidence: selected,
                              mode: .provider, providerLabel: "Fixture Gemma", matchupSummary: summary)
    }

    private func assertRejected(_ response: AskPLONNAnswer, request supplied: AskPLONNRequest? = nil,
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try AskPLONNCore.validateProviderAnswer(response, for: supplied ?? request()), file: file, line: line) {
            XCTAssertEqual($0 as? AskPLONNError, .unsupportedAnswer, file: file, line: line)
        }
    }

    private func assertInvalidRequest(_ supplied: AskPLONNRequest, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try AskPLONNCore.boundedRequest(supplied), file: file, line: line) {
            XCTAssertEqual($0 as? AskPLONNError, .invalidEvidence, file: file, line: line)
        }
    }

    func testQualitativeNHLSummaryRetainsNegatedDisclaimersAndExactObservations() throws {
        let original = fact()
        let validated = try AskPLONNCore.validateProviderAnswer(answer([original], summary: summary), for: request([original]))
        XCTAssertEqual(validated.matchupSummary, summary)
        XCTAssertEqual(validated.evidence, [original])
        XCTAssertEqual(validated.text, "Verified data summary\n\n" + original.text)
        XCTAssertEqual(validated.mode, .provider)
        XCTAssertEqual(validated.providerLabel, "Fixture Gemma")
        XCTAssertFalse(validated.matchupSummary!.contains(where: \.isNumber))
        XCTAssertTrue(validated.text.contains("3.2"))
    }

    func testMatchupRequestRequiresPurposeRecentFormAndDistinctBoundedTeams() throws {
        _ = try AskPLONNCore.boundedRequest(request())
        _ = try AskPLONNCore.boundedRequest(request(history: []))
        for intent in AskPLONNIntent.allCases where intent != .recentForm {
            assertInvalidRequest(request(intent: intent))
        }
        assertInvalidRequest(request(purpose: nil))
        for invalidTeams: [String]? in [nil, [], ["Boston Bruins"], teams + ["Other Team"],
                                       ["Boston Bruins", " boston bruins "], ["Boston Bruins", " \n"],
                                       ["Boston Bruins", String(repeating: "a", count: 101)],
                                       ["Boston Bruins", "New\tYork Rangers"]] {
            assertInvalidRequest(request(matchupTeams: invalidTeams))
        }
    }

    func testMatchupRequestRejectsHistoryAndUnavailableOrMalformedEvidence() {
        assertInvalidRequest(request(history: [AskPLONNQuestionContext(question: "Earlier NHL question", intent: .recentForm)]))
        assertInvalidRequest(request([]))
        assertInvalidRequest(request((0..<5).map { fact("goals-\($0)") }))
        assertInvalidRequest(request([fact(), fact()]))
        for invalid in [fact(""), fact(text: " \n"), fact(kind: .opponentHistory), fact(source: " \t"),
                        fact(availability: .stale), fact(availability: .unavailable)] {
            assertInvalidRequest(request([invalid]))
        }
    }

    func testSummaryRejectsNumericUnicodeCurrencyPercentagesAndSymbols() {
        for addition in ["3", "３", "٣", "Ⅳ", "½", "$", "€", "%", "‰", "→", "🏒"] {
            assertRejected(answer(summary: "These goal observations describe team history. " + addition))
        }
    }

    func testDigitsInExactOfficialTeamNamesAreIdentityWhileOtherQuantitiesRemainRejected() throws {
        for names in [["Philadelphia 76ers", "Boston Celtics"], ["San Francisco 49ers", "Dallas Cowboys"]] {
            let original = fact(text: names[0] + " and " + names[1] + " have recorded scoring rates.")
            let supplied = request([original], matchupTeams: names)
            let explanation = names[0] + " and " + names[1] + " have contrasting recorded scoring production. These historical observations describe different opponents."
            XCTAssertEqual(try AskPLONNCore.validateProviderAnswer(answer([original], summary: explanation), for: supplied).matchupSummary, explanation)
            for addition in [" These teams scored 76 goals.", " The sample has 49 games.", " Toronto 76ers have stronger scoring production."] {
                assertRejected(answer([original], summary: explanation + addition), request: supplied)
            }
            for invalid in ["X" + explanation, explanation.replacingOccurrences(of: names[0], with: names[0] + "X"),
                            explanation.replacingOccurrences(of: "76", with: "７６").replacingOccurrences(of: "49", with: "４９")] {
                assertRejected(answer([original], summary: invalid), request: supplied)
            }
        }
    }

    func testRecordedLocalGemmaNHLSummaryAcceptsOrdinaryContrastWithoutOutsideNames() throws {
        let names = ["Calgary Flames", "Anaheim Ducks"]
        let original = fact(text: "Calgary Flames averages 1.5 goals per game; Anaheim Ducks allows 2.8 goals per game. Anaheim Ducks averages 3.5 goals per game; Calgary Flames allows 5.8 goals per game.")
        let generated = "The Calgary Flames average a low number of goals per game while allowing a high number of goals per game. Conversely, the Anaheim Ducks average a high number of goals per game while allowing a low number of goals per game. These observations are based on a limited sample size of a few completed games."
        let supplied = request([original], matchupTeams: names)
        XCTAssertEqual(try AskPLONNCore.validateProviderAnswer(answer([original], summary: generated), for: supplied).matchupSummary, generated)
        assertRejected(answer([original], summary: generated + " Toronto Maple Leafs show stronger goal production."), request: supplied)
    }

    func testSummaryRejectsNewLinksAndCitationSyntax() {
        for addition in ["https://example.com/news", "www.example.com", "example.com", "[source]", "[one](https://example.com)",
                         "According to ESPN", "Sources: ESPN", "Citations: ESPN", "References: ESPN"] {
            assertRejected(answer(summary: "These goal observations describe team history. " + addition))
        }
    }

    func testSummaryRejectsOutsidePlayersAndTeamsEvenWithoutNumbers() {
        for invalid in ["Connor McDavid drives goal production.", "Toronto Maple Leafs have stronger goal production.",
                        "McDavid shows strong goal production."] {
            assertRejected(answer(summary: invalid))
        }
    }

    func testSummaryRejectsForecastsBetsRosterGoalieAndUnsupportedAdvancedStatistics() {
        for invalid in ["Boston Bruins will score more goals.", "Boston Bruins are likely to score more goals.",
                        "Boston Bruins ｗｉｌｌ score more goals.",
                        "These rates predict the result.", "These rates are a player forecast.",
                        "The matchup offers a betting edge.", "The odds favor Boston Bruins.",
                        "The roster is healthy.", "The goalie strengthens goal prevention.",
                        "The power play is strong.", "The teams have strong expected goals rates.",
                        "The teams have strong corsi rates.", "The teams have strong fenwick rates."] {
            assertRejected(answer(summary: invalid))
        }
    }

    func testSummaryUsesFullSuppliedBasisWhileSelectedEvidenceRetainsPriority() throws {
        let goals = fact()
        let shots = fact("home-shots", title: "Shots", text: "Boston Bruins average 30.2 shots on goal per game; New York Rangers allow 28.4 shots on goal per game.")
        let supplied = request([goals, shots])
        let explanation = "Shots compare offensive volume with the opposing allowance. These observations describe team history."
        let prioritized = try AskPLONNCore.validateProviderAnswer(answer([goals], summary: explanation), for: supplied)
        XCTAssertEqual(prioritized.matchupSummary, explanation)
        XCTAssertEqual(prioritized.evidence, [goals])
        assertRejected(answer([goals], summary: explanation), request: request([goals]))
        let validated = try AskPLONNCore.validateProviderAnswer(answer([shots], summary: explanation), for: supplied)
        XCTAssertEqual(validated.matchupSummary, explanation)
        XCTAssertEqual(validated.evidence, [shots])
    }

    func testSummaryRejectsUnsupportedOtherSportMetrics() {
        for metric in ["assists", "rebounds", "rushing", "passing", "slugging", "batting", "threes"] {
            assertRejected(answer(summary: "These " + metric + " observations compare team history."))
        }
    }

    func testValidSummaryCannotAuthorizeTamperedFactsOrMetadata() {
        let original = fact()
        let supplied = request([original])
        var changedCapture = original; changedCapture.capturedAt = original.capturedAt!.addingTimeInterval(1)
        var changedSample = original; changedSample.sampleSize = 6
        var changedURLs = original; changedURLs.sourceURLs = ["https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams"]
        var changedAvailability = original; changedAvailability.availability = .stale
        for changed in [fact("outside-id"), fact(title: "Shots"), fact(text: "Boston Bruins average 9.9 goals per game."),
                        fact(source: "Another source"), fact(kind: .opponentHistory),
                        changedCapture, changedSample, changedURLs, changedAvailability] {
            assertRejected(answer([changed], summary: summary), request: supplied)
        }
        var changedText = answer(summary: summary)
        changedText = AskPLONNAnswer(text: original.text + " Additional claim.", evidence: [original], mode: .provider,
                                    providerLabel: changedText.providerLabel, matchupSummary: changedText.matchupSummary)
        assertRejected(changedText, request: supplied)
    }

    func testMatchupSummaryIsRequiredNonblankBoundedAndFreeOfControlCharacters() throws {
        for invalid in [nil, "", " \n\t", String(repeating: "a", count: 701), "These observations\u{0000} describe goal production.",
                        "These observations\u{202E} describe goal production.", "These observations\tdescribe goal production."] as [String?] {
            assertRejected(answer(summary: invalid))
        }
        let combiningMarks = "a" + String(repeating: "\u{0301}", count: 700)
        XCTAssertLessThan(combiningMarks.count, 700)
        XCTAssertGreaterThan(combiningMarks.unicodeScalars.count, 700)
        assertRejected(answer(summary: combiningMarks))
        let boundary = String(repeating: "a ", count: AskPLONNCore.maximumMatchupSummaryLength / 2)
        XCTAssertEqual(try AskPLONNCore.validateProviderAnswer(answer(summary: boundary), for: request()).matchupSummary, boundary)
    }

    func testMatchupAnswerRequiresProviderModeAndAtMostThreeSelectedFacts() {
        let four = (0..<4).map { fact("goals-\($0)") }
        assertRejected(answer(four, summary: summary), request: request(four))
        var local = answer(summary: summary); local.mode = .localResearch
        assertRejected(local)
        var general = answer(summary: summary); general.mode = .generalExplanation
        assertRejected(general)
        var needsContext = answer(summary: summary); needsContext.requiresResearchContext = true
        assertRejected(needsContext)
        assertRejected(answer([], summary: summary))
    }

    func testOrdinaryResearchAndGeneralChatRejectUnsolicitedMatchupSummary() throws {
        let ordinary = request(purpose: nil, matchupTeams: nil)
        assertRejected(answer(summary: summary), request: ordinary)
        XCTAssertNil(try AskPLONNCore.validateProviderAnswer(answer(), for: ordinary).matchupSummary)
        let general = request([], intent: .generalChat, purpose: nil, matchupTeams: nil)
        assertRejected(AskPLONNAnswer(text: "An average describes observations.", evidence: [], mode: .generalExplanation,
                                     matchupSummary: summary), request: general)
    }

    func testLegacyEncodedAnswersDecodeWithAbsentSummaryAndNewAnswersRoundTrip() throws {
        let minimal = Data(#"{"text":"Older local answer","evidence":[]}"#.utf8)
        let legacy = Data(#"{"text":"Older provider answer","evidence":[],"mode":"provider","providerLabel":"Fixture Gemma","requiresResearchContext":false}"#.utf8)
        let decodedMinimal = try JSONDecoder().decode(AskPLONNAnswer.self, from: minimal)
        let decodedLegacy = try JSONDecoder().decode(AskPLONNAnswer.self, from: legacy)
        XCTAssertNil(decodedMinimal.matchupSummary)
        XCTAssertNil(decodedLegacy.matchupSummary)
        XCTAssertEqual(decodedMinimal.mode, .localResearch)
        XCTAssertEqual(decodedLegacy.mode, .provider)
        let encoded = try JSONEncoder().encode(answer(summary: summary))
        let restored = try JSONDecoder().decode(AskPLONNAnswer.self, from: encoded)
        XCTAssertEqual(restored.matchupSummary, summary)
        XCTAssertEqual(restored.evidence, [fact()])
    }
}
