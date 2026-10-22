import XCTest
@testable import PLONNCore
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

final class HostedAskPLONNProviderTests: XCTestCase {
    func testModelCreationWireAndValidatedDraft() async throws {
        let draft = AskPLONNModelDraft(league: "NFL", target: "Player", subject: "Example Player", stat: "Rec Yds", line: 60.5, side: "Over")
        let previous = AskPLONNModelDraft(league: "NFL", target: "Player", subject: "Example Player", stat: "Rec Yds")
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(draft)) as? [String: Any])
        let expected = try transport(envelope(["draft": object]))
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { http in
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: http.httpBody!) as? [String: Any])
            XCTAssertEqual(parsed["purpose"] as? String, "modelCreation")
            XCTAssertNil(parsed["matchup_teams"])
            XCTAssertNil(parsed["ownerID"])
            XCTAssertEqual((parsed["evidence"] as? [Any])?.count, 0)
            XCTAssertEqual(Set((parsed["model_draft"] as? [String: Any] ?? [:]).keys), AskPLONNModelDraft.wireKeys)
            XCTAssertTrue((parsed["model_draft"] as? [String: Any])?["line"] is NSNull)
            XCTAssertEqual((parsed["prior_turns"] as? [[String: Any]])?.first?["question"] as? String, "Create an NFL receiving yards model for Example Player")
            return try await expected(http)
        })
        let answer = try await provider.answer(AskPLONNRequest(question: "Over 60.5", intent: .generalChat, evidence: [],
            ownerID: "private-owner", priorTurns: [.init(question: "Create an NFL receiving yards model for Example Player", intent: .generalChat)],
            researchPurpose: .modelCreation, modelDraft: previous))
        XCTAssertEqual(answer.modelDraft, draft)
        XCTAssertEqual(answer.text, draft.guidance)
        XCTAssertTrue(answer.evidence.isEmpty)
        XCTAssertNil(answer.matchupSummary)
    }

    func testModelCreationRejectsInvalidDraftAndUnexpectedClaims() async throws {
        let draft = AskPLONNModelDraft(league: "NBA", target: "Player", subject: "Example Player", stat: "Points", line: 25.5, side: "Over")
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(draft)) as? [String: Any])
        let request = AskPLONNRequest(question: "Create a model", intent: .generalChat, evidence: [], researchPurpose: .modelCreation)
        var invalidStat = object; invalidStat["stat"] = "Rec Yds"
        var boolLine = object; boolLine["line"] = true
        var gameID = object; gameID["game_id"] = "12345"
        for response: [String: Any] in [["draft": invalidStat], ["draft": boolLine], ["draft": gameID],
                                       ["draft": object, "prediction": 0.9], ["kind": "explanation", "message": "Created a model"]] {
            do { _ = try await provider(envelope(response)).answer(request); XCTFail("Accepted invalid model draft") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        }
    }
    private static let token = "fixture.access.token"
    private static let matchupSummary = "Bruins recorded lower scoring than Sabres despite higher shot volume in the supplied sample. The limited sample supports a cautious comparison of recorded patterns."
    private func matchupRequest() -> AskPLONNRequest {
        let texts = ["Bruins averaged 2 goals across 4 recorded games.", "Sabres averaged 3 goals across 4 recorded games.",
                     "Bruins averaged 31 shots and Sabres averaged 28 shots in the recorded sample.", "The sample contains 4 games and is not a forecast."]
        return AskPLONNRequest(question: "Explain the observed Bruins and Sabres matchup patterns.", intent: .recentForm,
            evidence: texts.enumerated().map { fact("matchup-\($0.offset)", text: $0.element) }, ownerID: "private-owner",
            researchPurpose: .matchupInsights, matchupTeams: ["Bruins", "Sabres"])
    }
    func testMatchupPurposeAndTeamsUseBoundedWireAndCarryGeneratedSummaryUnchanged() async throws {
        let request = matchupRequest()
        let expected = try transport(envelope(["supported": true, "evidence_ids": ["matchup-2", "matchup-0", "matchup-1"],
                                               "summary": Self.matchupSummary]))
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { http in
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: http.httpBody!) as? [String: Any])
            XCTAssertEqual(Set(parsed.keys), Set(["consent_version", "request_id", "schema_version", "question", "intent", "evidence", "prior_turns", "purpose", "matchup_teams"]))
            XCTAssertEqual(parsed["schema_version"] as? Int, 1)
            XCTAssertEqual(parsed["consent_version"] as? String, AccountAIConsentPolicy.version)
            XCTAssertNotNil(UUID(uuidString: parsed["request_id"] as? String ?? ""))
            XCTAssertEqual(parsed["purpose"] as? String, "matchupInsights")
            XCTAssertEqual(parsed["matchup_teams"] as? [String], ["Bruins", "Sabres"])
            XCTAssertEqual(parsed["intent"] as? String, "recentForm")
            XCTAssertEqual((parsed["prior_turns"] as? [Any])?.count, 0)
            XCTAssertLessThanOrEqual(http.httpBody!.count, HostedAskPLONNProvider.maximumRequestBytes)
            XCTAssertFalse(String(decoding: http.httpBody!, as: UTF8.self).contains("private-owner"))
            let facts = try XCTUnwrap(parsed["evidence"] as? [[String: Any]])
            XCTAssertEqual(facts.compactMap { $0["text"] as? String }, request.evidence.map(\.text))
            return try await expected(http)
        })
        let answer = try await provider.answer(request)
        let selected = [request.evidence[2], request.evidence[0], request.evidence[1]]
        XCTAssertEqual(answer.matchupSummary, Self.matchupSummary)
        XCTAssertEqual(answer.evidence, selected)
        XCTAssertEqual(answer.text, "Verified data summary\n\n" + selected.map(\.text).joined(separator: "\n\n"))
        XCTAssertEqual(answer.providerLabel, provider.displayName)
    }
    func testBackendCompatibleDescriptiveLimitationsRemainUnchanged() async throws {
        for summary in [
            "Overall, Bruins recorded lower scoring than Sabres in the supplied sample.",
            "Bruins and Sabres samples can support a cautious comparison of recorded scoring.",
            "Bruins and Sabres samples could only support a limited comparison of recorded scoring.",
            "Bruins and Sabres samples would support a descriptive comparison of recorded scoring."
        ] {
            let answer = try await provider(envelope(["supported": true, "evidence_ids": ["matchup-0"], "summary": summary])).answer(matchupRequest())
            XCTAssertEqual(answer.matchupSummary, summary)
            XCTAssertEqual(answer.evidence, [matchupRequest().evidence[0]])
        }
    }

    func testDescriptiveExceptionDoesNotPermitUnsupportedClaims() async throws {
        for summary in [
            "Bruins could win against Sabres.",
            "Bruins could score against Sabres.",
            "Bruins and Sabres samples can support a comparison and would win tomorrow.",
            "Bruins and Sabres samples can support a cautious comparison with Connor McDavid.",
            "Bruins and Sabres samples can support a cautious comparison of rushing.",
            "Bruins and Sabres samples can support a cautious comparison with three goals.",
            "Bruins are favored against Sabres.",
            "Bruins and Sabres samples establish calibration.",
            "Bruins shall outperform Sabres.",
            "Bruins and Sabres samples can support a cautious comparison [source]."
        ] {
            do {
                _ = try await provider(envelope(["supported": true, "evidence_ids": ["matchup-0"], "summary": summary])).answer(matchupRequest())
                XCTFail("Accepted unsupported commentary")
            } catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        }
    }

    func testUnsupportedContentWordsCannotHideBehindLowercaseOrOrdinaryNames() async throws {
        for summary in [
            "Bruins have superior turnover efficiency.",
            "Bruins and Sabres have contrasting turnover patterns.",
            "Bruins and Sabres have contrasting possession patterns.",
            "Bruins and Sabres have contrasting completion patterns.",
            "Bruins and Sabres have contrasting rebound patterns.",
            "Bruins and Sabres samples support connor mcdavid scoring.",
            "bruins and sabres compare with boston celtics.",
            "Bruins scoring causes Sabres scoring.",
            "Bruins have better recorded scoring than Sabres.",
            "Bruins and Sabres samples can support a cautious comparison of turnover efficiency."
        ] {
            do {
                _ = try await provider(envelope(["supported": true, "evidence_ids": ["matchup-0"], "summary": summary])).answer(matchupRequest())
                XCTFail("Accepted ungrounded content words")
            } catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        }
    }

    func testContentMetricWordsMustComeFromActualObservations() async throws {
        let evidence = fact("metric", text: "Bruins recorded 12 turnovers and Sabres recorded 14 turnovers in the sample.")
        let request = AskPLONNRequest(question: "Explain observed turnover counts.", intent: .recentForm,
            evidence: [evidence], researchPurpose: .matchupInsights, matchupTeams: ["Bruins", "Sabres"])
        let summary = "Bruins and Sabres have contrasting recorded turnover patterns. The sample limits the comparison."
        let answer = try await provider(envelope(["supported": true, "evidence_ids": ["metric"], "summary": summary])).answer(request)
        XCTAssertEqual(answer.matchupSummary, summary)
        XCTAssertEqual(answer.evidence, [evidence])
    }

    func testFailureDiagnosticsUseOnlyFixedCategories() {
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: AskPLONNError.unsupportedAnswer), "answer_contract")
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: HostedAskPLONNError.invalidResponse), "response_schema")
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: HostedAskPLONNError.unverifiedModel), "model_identity")
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: URLError(.notConnectedToInternet)), "network")
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: HostedAskPLONNError.authenticationRequired), "authentication")
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: NSError(domain: "secret response", code: 1)), "other")
    }

    func testMatchupMalformedSchemaCannotFallBackToLegacySelection() async throws {
        let malformed: [[String: Any]] = [
            ["supported": true, "evidence_ids": ["matchup-0"]],
            ["supported": 1, "evidence_ids": ["matchup-0"], "summary": Self.matchupSummary],
            ["supported": true, "evidence_ids": [1], "summary": Self.matchupSummary],
            ["supported": true, "evidence_ids": ["matchup-0"], "summary": 123],
            ["supported": true, "evidence_ids": ["matchup-0"], "summary": Self.matchupSummary, "source": "Invented"]]
        for response in malformed {
            do { _ = try await provider(envelope(response)).answer(matchupRequest()); XCTFail("Accepted malformed matchup response") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        }
    }
    func testMatchupUnsafeSummaryOrEvidenceIsRejected() async throws {
        let badSummaries = ["", " \n", String(repeating: "a", count: 701), String(repeating: "👩", count: 701),
                            "Bruins recorded 9 goals against Sabres.", "Bruins recorded ９ goals against Sabres.",
                            "Bruins recorded three goals against Sabres.", "Bruins will win against Sabres.",
                            "See https://example.com for Bruins and Sabres."]
        var responses = badSummaries.map { ["supported": true, "evidence_ids": ["matchup-0"], "summary": $0] as [String: Any] }
        responses += [["supported": true, "evidence_ids": ["invented"], "summary": Self.matchupSummary],
                      ["supported": true, "evidence_ids": ["matchup-0", "matchup-0"], "summary": Self.matchupSummary],
                      ["supported": true, "evidence_ids": ["matchup-0", "matchup-1", "matchup-2", "matchup-3"], "summary": Self.matchupSummary],
                      ["supported": true, "evidence_ids": [], "summary": Self.matchupSummary],
                      ["supported": false, "evidence_ids": ["matchup-0"], "summary": ""],
                      ["supported": false, "evidence_ids": [], "summary": Self.matchupSummary]]
        for response in responses {
            do { _ = try await provider(envelope(response)).answer(matchupRequest()); XCTFail("Accepted unsafe matchup response: \(response)") }
            catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        }
        do { _ = try await provider(envelope(["supported": false, "evidence_ids": [], "summary": ""])).answer(matchupRequest()); XCTFail("Accepted unsupported matchup") }
        catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedQuestion) }
    }
    private func fact(_ id: String = "fact", text: String = "The recorded result was 12 points.",
                      kind: AskPLONNIntent = .recentForm, availability: AskPLONNAvailability = .available) -> AskPLONNEvidence {
        AskPLONNEvidence(id: id, kind: kind, title: "Recorded form", text: text, source: "Fixture",
                         availability: availability, sourceURLs: ["https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=1"])
    }
    func testMatchupInvalidContextFailsBeforeCredentialsOrTransport() async {
        var noTeams = matchupRequest(); noTeams.matchupTeams = nil
        var noPurpose = matchupRequest(); noPurpose.researchPurpose = nil
        var duplicateTeams = matchupRequest(); duplicateTeams.matchupTeams = ["Bruins", " bruins "]
        var longTeam = matchupRequest(); longTeam.matchupTeams = [String(repeating: "界", count: 101), "Sabres"]
        var history = matchupRequest(); history.priorTurns = [AskPLONNQuestionContext(question: "Earlier question", intent: .recentForm)]
        let unavailable = AskPLONNRequest(question: "Explain", intent: .recentForm,
            evidence: [fact("stale", availability: .stale)], researchPurpose: .matchupInsights, matchupTeams: ["Bruins", "Sabres"])
        let provider = HostedAskPLONNProvider(tokenProvider: { XCTFail("Invalid matchup acquired credentials"); return Self.token },
            transport: { _ in XCTFail("Invalid matchup reached transport"); throw HostedAskPLONNError.invalidResponse })
        for request in [noTeams, noPurpose, duplicateTeams, longTeam, history, unavailable] {
            do { _ = try await provider.answer(request); XCTFail("Accepted invalid matchup context") }
            catch { XCTAssertEqual(error as? AskPLONNError, .invalidEvidence) }
        }
    }
    func testLegacyResearchRejectsSummaryAndRetainsFourEvidenceCapacity() async throws {
        do { _ = try await provider(envelope(["supported": true, "evidence_ids": ["fact"], "summary": Self.matchupSummary])).answer(research()); XCTFail("Legacy response accepted summary") }
        catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        let facts = (0..<4).map { fact("legacy-\($0)") }
        let legacy = try provider(envelope(["supported": true, "evidence_ids": facts.map(\.id)]))
        let answer = try await legacy.answer(research(facts))
        XCTAssertEqual(answer.evidence, facts)
        XCTAssertNil(answer.matchupSummary)
    }
    private func research(_ facts: [AskPLONNEvidence]? = nil) -> AskPLONNRequest {
        AskPLONNRequest(question: "What were the recorded points?", intent: .recentForm,
                        evidence: facts ?? [fact()], ownerID: "private-owner")
    }
    private func general(_ question: String = "What is implied probability?") -> AskPLONNRequest {
        AskPLONNRequest(question: question, intent: .generalChat, evidence: [], ownerID: "private-owner")
    }
    private func envelope(_ result: [String: Any], model: String = HostedAskPLONNProvider.modelName,
                          providerName: String = "ollama_cloud") -> [String: Any] {
        ["schema_version": 2, "model": model, "provider": providerName, "response": result]
    }
    private func transport(_ payload: [String: Any]? = nil, status: Int = 200, url: URL? = nil) throws -> HostedAskPLONNProvider.Transport {
        let data = try JSONSerialization.data(withJSONObject: payload ?? envelope(["supported": true, "evidence_ids": ["fact"]]))
        return { request in
            (data, HTTPURLResponse(url: url ?? request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!)
        }
    }
    private func provider(_ payload: [String: Any]? = nil) throws -> HostedAskPLONNProvider {
        HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: try transport(payload))
    }
    private func input(_ request: URLRequest) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody)) as? [String: Any])
    }

    func testEndpointMustMatchExactlyBeforeTokenOrTransport() async {
        for raw in ["http://api.oakas.ai/plonn/v1/chat", "https://other.invalid/plonn/v1/chat",
                    "https://api.oakas.ai:443/plonn/v1/chat", "https://user@api.oakas.ai/plonn/v1/chat",
                    "https://user:password@api.oakas.ai/plonn/v1/chat", "https://api.oakas.ai/plonn/v1/chat?",
                    "https://api.oakas.ai/plonn/v1/chat?x=1", "https://api.oakas.ai/plonn/v1/chat#",
                    "https://api.oakas.ai/plonn/v1/chat#fragment", "https://api.oakas.ai/plonn/v1/chat/",
                    "https://api.oakas.ai/plonn/v1/%63hat", "https://API.OAKAS.AI/plonn/v1/chat"] {
            let endpoint = URL(string: raw)!
            XCTAssertFalse(HostedAskPLONNProvider.allowedEndpoint(endpoint), raw)
            let provider = HostedAskPLONNProvider(endpoint: endpoint, tokenProvider: {
                XCTFail("Token requested for rejected endpoint"); return Self.token
            }, transport: { _ in XCTFail("Rejected endpoint reached transport"); throw HostedAskPLONNError.invalidResponse })
            do { _ = try await provider.answer(general()); XCTFail("Accepted endpoint: " + raw) }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidEndpoint) }
        }
        XCTAssertTrue(HostedAskPLONNProvider.allowedEndpoint(HostedAskPLONNProvider.endpointURL))
    }

    func testBearerTokensAreFreshHeaderOnlyAndInvalidTokensCannotReachTransport() async throws {
        let tokens = TokenSequence()
        let expected = try transport()
        let provider = HostedAskPLONNProvider(tokenProvider: { await tokens.next() }, transport: { request in
            let count = await tokens.count()
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer fixture.token.\(count)")
            XCTAssertFalse(String(decoding: request.httpBody!, as: UTF8.self).contains("fixture.token"))
            return try await expected(request)
        })
        _ = try await provider.answer(research())
        _ = try await provider.answer(research())
        let count = await tokens.count()
        XCTAssertEqual(count, 2)
        for token in ["", " ", "token\r\nInjected: yes", "token\n", "token\t", "é", String(repeating: "a", count: 8193)] {
            let invalid = HostedAskPLONNProvider(tokenProvider: { token }, transport: { _ in
                XCTFail("Invalid token reached transport"); throw HostedAskPLONNError.invalidResponse
            })
            do { _ = try await invalid.answer(research()); XCTFail("Invalid token accepted") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .authenticationRequired) }
        }
    }

    func testWireContainsOnlyCurrentAvailableMatchingFactsAndNoPrivateMetadata() async throws {
        let current = fact()
        let expected = try transport()
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
            XCTAssertEqual(request.url, HostedAskPLONNProvider.endpointURL)
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.timeoutInterval, 45)
            XCTAssertFalse(request.httpShouldHandleCookies)
            XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
            let body = try XCTUnwrap(request.httpBody)
            XCTAssertLessThanOrEqual(body.count, 16 * 1024)
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
            XCTAssertEqual(Set(parsed.keys), Set(["consent_version", "request_id", "schema_version", "question", "intent", "evidence", "prior_turns"]))
            XCTAssertEqual(parsed["schema_version"] as? Int, 1)
            XCTAssertEqual(parsed["consent_version"] as? String, AccountAIConsentPolicy.version)
            XCTAssertNotNil(UUID(uuidString: parsed["request_id"] as? String ?? ""))
            let evidence = try XCTUnwrap(parsed["evidence"] as? [[String: Any]])
            XCTAssertEqual(evidence.count, 1)
            XCTAssertEqual(Set(evidence[0].keys), Set(["id", "kind", "title", "text", "source", "source_urls"]))
            XCTAssertEqual(evidence[0]["text"] as? String, current.text)
            XCTAssertEqual(evidence[0]["source_urls"] as? [String], current.sourceURLs)
            XCTAssertEqual((parsed["prior_turns"] as? [Any])?.count, 0)
            let text = String(decoding: body, as: UTF8.self)
            for secret in ["private-owner", "capturedAt", "sampleSize", "availability", "options", "gemma4:26b", "api.oakas.ai", "STALE", "MISMATCH", "UNAVAILABLE"] {
                XCTAssertFalse(text.contains(secret), secret)
            }
            return try await expected(request)
        })
        let answer = try await provider.answer(research([current, fact("stale", text: "STALE", availability: .stale),
                                                        fact("other", text: "MISMATCH", kind: .opponentHistory),
                                                        fact("missing", text: "UNAVAILABLE", availability: .unavailable)]))
        XCTAssertEqual(answer.text, "Verified data summary\n\n" + current.text)
        XCTAssertEqual(answer.evidence, [current])
        XCTAssertEqual(answer.mode, .provider)
        XCTAssertEqual(answer.providerLabel, "Gemma 4 31B")
    }

    func testResearchSelectionPreservesExactOriginalTextAndRejectsUnsafeSelections() async throws {
        let first = fact("one", text: "The original result was 12 points; this is not a forecast.")
        let second = fact("two", text: "The original second result was 14 points.")
        let ordered = try provider(envelope(["supported": true, "evidence_ids": ["two", "one"]]))
        let answer = try await ordered.answer(research([first, second]))
        XCTAssertEqual(answer.text, "Verified data summary\n\n" + second.text + "\n\n" + first.text)
        XCTAssertEqual(answer.evidence, [second, first])
        let invalid: [[String: Any]] = [
            ["supported": 1, "evidence_ids": ["fact"]], ["supported": "true", "evidence_ids": ["fact"]],
            ["supported": true, "evidence_ids": ["fact"], "text": "999 points"],
            ["supported": true, "evidence_ids": [1]], ["supported": true, "evidence_ids": ["unknown"]],
            ["supported": true, "evidence_ids": ["fact", "fact"]], ["supported": false, "evidence_ids": ["fact"]],
            ["supported": true, "evidence_ids": []]
        ]
        for result in invalid {
            do { _ = try await provider(envelope(result)).answer(research()); XCTFail("Unsafe selection accepted") }
            catch {
                XCTAssertTrue(error as? HostedAskPLONNError == .invalidResponse || error as? AskPLONNError == .unsupportedAnswer)
            }
        }
        do { _ = try await provider(envelope(["supported": false, "evidence_ids": []])).answer(research()); XCTFail("Unsupported question accepted") }
        catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedQuestion) }
    }

    func testResponseIdentityAndEnvelopeKeysArePinned() async throws {
        let result: [String: Any] = ["supported": true, "evidence_ids": ["fact"]]
        for payload in [envelope(result, model: "gemma3:12b"), envelope(result, model: "gemma4:26b-cloud"), envelope(result, providerName: "wrong")] {
            do { _ = try await provider(payload).answer(research()); XCTFail("Unverified model accepted") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .unverifiedModel) }
        }
        var additional = envelope(result); additional["owner"] = "private-owner"
        var missing = envelope(result); missing.removeValue(forKey: "provider")
        for version: Any in [true, 1, "2", NSNull()] {
            var payload = envelope(result); payload["schema_version"] = version
            do { _ = try await provider(payload).answer(research()); XCTFail("Invalid schema version accepted") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        }
        for payload in [additional, missing] {
            do { _ = try await provider(payload).answer(research()); XCTFail("Invalid envelope accepted") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        }
    }

    func testMalformedJSONAndNonIntegerSchemaAreRejected() async {
        let prefix = "{\"schema_version\":1,\"model\":\"gemma4:26b\",\"digest\":\"\(HostedAskPLONNProvider.pinnedDigest)\","
        let validResponse = "\"response\":{\"supported\":true,\"evidence_ids\":[\"fact\"]}}"
        let duplicateEnvelope = prefix + "\"model\":\"gemma4:26b\"," + validResponse
        let escapedDuplicate = prefix + "\"mod\\u0065l\":\"gemma4:26b\"," + validResponse
        let duplicateSelection = prefix + "\"response\":{\"supported\":false,\"supported\":true,\"evidence_ids\":[\"fact\"]}}"
        let deep = String(repeating: "[", count: 100) + "0" + String(repeating: "]", count: 100)
        for raw in ["not-json", "[]", "{", duplicateEnvelope, escapedDuplicate, duplicateSelection, deep,
                    "{\"schema_version\":1.5,\"model\":\"gemma4:26b\",\"digest\":\"\(HostedAskPLONNProvider.pinnedDigest)\",\"response\":{\"supported\":true,\"evidence_ids\":[\"fact\"]}}"] {
            let data = Data(raw.utf8)
            let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
                (data, HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
            })
            do { _ = try await provider.answer(research()); XCTFail("Invalid JSON accepted") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        }
    }

    func testGeneralExplanationsAndStaticGuidanceDoNotExposeUnsupportedClaims() async throws {
        let explanation = "In a hypothetical example, decimal odds of 2.00 imply 50% before margin."
        let explained = try await provider(envelope(["kind": "explanation", "message": explanation])).answer(general())
        XCTAssertEqual(explained.text, explanation)
        XCTAssertEqual(explained.mode, .generalExplanation)
        XCTAssertEqual(explained.providerLabel, "Gemma 4 31B")
        XCTAssertTrue(explained.evidence.isEmpty)
        for kind in ["needs_research", "unsupported"] {
            let answer = try await provider(envelope(["kind": kind, "message": "The player scored 999 points."])).answer(general())
            XCTAssertFalse(answer.text.contains("999"))
            XCTAssertEqual(answer.text, kind == "needs_research" ? AskPLONNCore.generalResearchGuidance().text : AskPLONNCore.generalUnsupportedGuidance().text)
            XCTAssertEqual(answer.requiresResearchContext, kind == "needs_research")
        }
        for question in ["How many points did LeBron James score?", "how many hits did aaron judge have?"] {
            let answer = try await provider(envelope(["kind": "explanation", "message": "The player scored 999 points."])).answer(general(question))
            XCTAssertEqual(answer.text, AskPLONNCore.generalResearchGuidance().text)
            XCTAssertTrue(answer.requiresResearchContext)
        }
        for question in ["Give me a guaranteed winning bet", "Project a player's score from team averages"] {
            let answer = try await provider(envelope(["kind": "explanation", "message": "Guaranteed 999 points."])).answer(general(question))
            XCTAssertEqual(answer.text, AskPLONNCore.generalUnsupportedGuidance().text)
        }
    }

    func testMalformedAndUnsafeGeneralResponsesAreRejected() async throws {
        let malformed: [[String: Any]] = [
            ["kind": "other", "message": "Hello"], ["kind": true, "message": "Hello"],
            ["kind": "explanation", "message": "Hello", "source": "Invented"],
            ["kind": "explanation"], ["kind": "explanation", "message": 123],
            ["kind": "explanation", "message": " \n"], ["kind": "explanation", "message": String(repeating: "a", count: 2001)],
            ["kind": "explanation", "message": "See https://example.com"], ["kind": "explanation", "message": "Sources: ESPN"],
            ["kind": "needs_research", "message": "Read [source](https://example.com)"],
            ["kind": "unsupported", "message": "This is true [1]"], ["kind": "explanation", "message": "Bad\u{0001}control"]
        ]
        for result in malformed {
            do { _ = try await provider(envelope(result)).answer(general()); XCTFail("Malformed general response accepted") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        }
    }

    func testHistoryUsesSnakeCaseAndResearchSendsOnlyCurrentEvidence() async throws {
        let expected = try transport()
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: request.httpBody!) as? [String: Any])
            let prior = try XCTUnwrap(parsed["prior_turns"] as? [[String: String]])
            if parsed["question"] as? String == "What does that mean?" {
                XCTAssertEqual(prior, [["question": "What were the recorded points?", "intent": "recentForm"]])
                XCTAssertFalse(String(decoding: request.httpBody!, as: UTF8.self).contains("OLD ANSWER CLAIM"))
                let facts = try XCTUnwrap(parsed["evidence"] as? [[String: Any]])
                XCTAssertEqual(facts[0]["text"] as? String, "CURRENT CLAIM: 14 recorded points.")
            } else { XCTAssertTrue(prior.isEmpty) }
            return try await expected(request)
        })
        let conversation = AskPLONNConversation()
        _ = try await conversation.ask(question: "What were the recorded points?", intent: .recentForm,
                                        evidence: [fact(text: "OLD ANSWER CLAIM: 12 recorded points.")],
                                        ownerID: "private-owner", contextID: "private-context", provider: provider)
        let answer = try await conversation.ask(question: "What does that mean?", intent: .recentForm,
                                                evidence: [fact(text: "CURRENT CLAIM: 14 recorded points.")],
                                                ownerID: "private-owner", contextID: "private-context", provider: provider)
        XCTAssertEqual(answer.text, "Verified data summary\n\nCURRENT CLAIM: 14 recorded points.")
    }

    func testGeneralHistoryIsBoundedAndCannotCrossAccountOrContext() async throws {
        let message = "A model can overfit by learning noise rather than patterns."
        let expected = try transport(envelope(["kind": "explanation", "message": message]))
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: request.httpBody!) as? [String: Any])
            let prior = try XCTUnwrap(parsed["prior_turns"] as? [[String: String]])
            if parsed["question"] as? String == "Explain the second point." {
                XCTAssertEqual(prior, [["question": "How can models overfit?", "intent": "generalChat", "general_response": message]])
            } else { XCTAssertTrue(prior.isEmpty) }
            let wire = String(decoding: request.httpBody!, as: UTF8.self)
            for secret in ["private-owner", "other-owner", "private-context", "other-context", "generalResponse"] {
                XCTAssertFalse(wire.contains(secret))
            }
            XCTAssertEqual((parsed["evidence"] as? [Any])?.count, 0)
            return try await expected(request)
        })
        let conversation = AskPLONNConversation()
        for (question, owner, context) in [("How can models overfit?", "private-owner", "private-context"),
                                            ("Explain the second point.", "private-owner", "private-context"),
                                            ("What is variance?", "other-owner", "private-context"),
                                            ("What is calibration?", "other-owner", "other-context")] {
            _ = try await conversation.ask(question: question, intent: .generalChat, evidence: [], ownerID: owner, contextID: context, provider: provider)
        }
    }

    func testInvalidContextAndPayloadBoundsFailBeforeCredentials() async {
        let noCredentials = HostedAskPLONNProvider(tokenProvider: {
            XCTFail("Invalid request acquired credentials"); return Self.token
        }, transport: { _ in XCTFail("Invalid request reached transport"); throw HostedAskPLONNError.invalidResponse })
        var tooManyTurns = research()
        tooManyTurns.priorTurns = Array(repeating: AskPLONNQuestionContext(question: "q", intent: .recentForm), count: 7)
        var researchWithGeneralResponse = research()
        researchWithGeneralResponse.priorTurns = [AskPLONNQuestionContext(question: "q", intent: .generalChat, generalResponse: "Old response")]
        for request in [tooManyTurns, researchWithGeneralResponse] {
            do { _ = try await noCredentials.answer(request); XCTFail("Invalid history accepted") }
            catch { XCTAssertEqual(error as? AskPLONNError, .invalidHistory) }
        }
        do { _ = try await noCredentials.answer(AskPLONNRequest(question: "Explain", intent: .generalChat, evidence: [fact()])); XCTFail("General evidence accepted") }
        catch { XCTAssertEqual(error as? AskPLONNError, .invalidEvidence) }
        for request in [research([]), research((0..<5).map { fact("fact-\($0)") })] {
            do { _ = try await noCredentials.answer(request); XCTFail("Unsupported evidence accepted") }
            catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedQuestion) }
        }
        // Four valid multibyte facts are within character bounds but exceed the byte budget.
        let oversized = research((0..<4).map { fact("fact-\($0)", text: String(repeating: "界", count: 1800)) })
        do { _ = try await noCredentials.answer(oversized); XCTFail("Oversized request accepted") }
        catch { XCTAssertEqual(error as? HostedAskPLONNError, .oversizedPayload) }
    }

    func testReturnedURLResponseBoundsAnd429DoNotRetry() async throws {
        let counter = TokenSequence()
        let requests = TokenSequence()
        let limited = HostedAskPLONNProvider(tokenProvider: { await counter.next() }, transport: { request in
            _ = await requests.next()
            return (Data(), HTTPURLResponse(url: request.url!, statusCode: 429, httpVersion: nil, headerFields: nil)!)
        })
        do { _ = try await limited.answer(research()); XCTFail("429 accepted") }
        catch { XCTAssertEqual(error as? HostedAskPLONNError, .rateLimited) }
        let count = await counter.count()
        XCTAssertEqual(count, 1)
        let requestCount = await requests.count()
        XCTAssertEqual(requestCount, 1)
        let wrongURL = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: try transport(url: URL(string: "https://other.invalid/chat")!))
        do { _ = try await wrongURL.answer(research()); XCTFail("Changed response URL accepted") }
        catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        let oversized = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
            (Data(repeating: 65, count: 32769), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        })
        do { _ = try await oversized.answer(research()); XCTFail("Oversized reply accepted") }
        catch { XCTAssertEqual(error as? HostedAskPLONNError, .oversizedPayload) }
        for (status, expected) in [(401, HostedAskPLONNError.authenticationRequired), (403, .authenticationRequired),
                                   (409, .capacity), (429, .rateLimited), (502, .unavailable), (503, .unavailable),
                                   (504, .unavailable), (413, .oversizedPayload), (302, .invalidResponse)] {
            let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: try transport(status: status))
            do { _ = try await provider.answer(research()); XCTFail("HTTP error accepted") }
            catch { XCTAssertEqual(error as? HostedAskPLONNError, expected) }
        }
    }

    func testStructured502DistinguishesRejectedResponseWithoutExposingBody() async throws {
        for (raw, expected) in [
            (#"{"status":"unavailable","reason":"chat_response_invalid","message":"private response text"}"#, HostedAskPLONNError.invalidResponse),
            (#"{"status":"unavailable","reason":"model_response_invalid"}"#, .unavailable),
            (#"{"status":"unavailable","reason":"unknown"}"#, .unavailable),
            (#"{"status":"unavailable","reason":"chat_response_invalid","reason":"unknown"}"#, .unavailable),
            (#"{"reason":"chat_response_invalid"}"#, .unavailable),
            ("not JSON", .unavailable)
        ] {
            let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
                (Data(raw.utf8), HTTPURLResponse(url: request.url!, statusCode: 502, httpVersion: nil, headerFields: nil)!)
            })
            do { _ = try await provider.answer(research()); XCTFail("502 accepted") }
            catch {
                XCTAssertEqual(error as? HostedAskPLONNError, expected)
                XCTAssertFalse(AskPLONNFailureDiagnostic.matchupMessage(for: error).contains("private response text"))
            }
        }
    }

    func testStructured429SeparatesDailyAllowanceFromTemporaryRateLimitsWithoutRetry() async throws {
        for (raw, expected) in [
            (#"{"status":"unavailable","reason":"chat_quota_exceeded","message":"private account details"}"#, HostedAskPLONNError.dailyAllowance),
            (#"{"status":"unavailable","reason":"chat_provider_rate_limited"}"#, .rateLimited),
            (#"{"status":"unavailable","reason":"unknown"}"#, .rateLimited),
            (#"{"status":"unavailable","reason":"chat_quota_exceeded","reason":"unknown"}"#, .rateLimited),
            (#"{"reason":"chat_quota_exceeded"}"#, .rateLimited),
            ("<html>429 Too Many Requests</html>", .rateLimited)
        ] {
            let attempts = TokenSequence()
            let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
                _ = await attempts.next()
                return (Data(raw.utf8), HTTPURLResponse(url: request.url!, statusCode: 429, httpVersion: nil, headerFields: nil)!)
            })
            do { _ = try await provider.answer(general()); XCTFail("429 accepted") }
            catch {
                XCTAssertEqual(error as? HostedAskPLONNError, expected)
                XCTAssertFalse(AskPLONNFailureDiagnostic.matchupMessage(for: error).contains("private account details"))
            }
            let count = await attempts.count()
            XCTAssertEqual(count, 1)
        }
        XCTAssertFalse(HostedAskPLONNError.rateLimited.userFacingMessage.contains("daily"))
        XCTAssertFalse(HostedAskPLONNError.rateLimited.userFacingMessage.contains("beta"))
        XCTAssertTrue(HostedAskPLONNError.dailyAllowance.userFacingMessage.contains("midnight UTC"))
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: HostedAskPLONNError.dailyAllowance), "daily_allowance")
    }

    func testStreamingDelegateReadsQuota429ButStillBoundsErrorBodies() async throws {
        let raw = Data(#"{"status":"unavailable","reason":"chat_quota_exceeded"}"#.utf8)
        let (data, response) = try await streamed { delegate, session, task in
            let response = HTTPURLResponse(url: HostedAskPLONNProvider.endpointURL, statusCode: 429, httpVersion: nil, headerFields: nil)!
            delegate.urlSession(session, dataTask: task, didReceive: response) { XCTAssertEqual($0, .allow) }
            delegate.urlSession(session, dataTask: task, didReceive: raw)
            delegate.urlSession(session, task: task, didCompleteWithError: nil)
        }
        XCTAssertEqual(data, raw)
        XCTAssertThrowsError(try HostedAskPLONNProvider.validateStatus(response.statusCode, body: data)) {
            XCTAssertEqual($0 as? HostedAskPLONNError, .dailyAllowance)
        }
        do {
            _ = try await streamed { delegate, session, task in
                let response = HTTPURLResponse(url: HostedAskPLONNProvider.endpointURL, statusCode: 429, httpVersion: nil, headerFields: nil)!
                delegate.urlSession(session, dataTask: task, didReceive: response) { XCTAssertEqual($0, .allow) }
                delegate.urlSession(session, dataTask: task, didReceive: Data(repeating: 65, count: 32769))
            }
            XCTFail("Oversized rate-limit body accepted")
        } catch { XCTAssertEqual(error as? HostedAskPLONNError, .oversizedPayload) }
    }

    func testStreamingDelegateCollectsBounded502ForStructuredClassification() async throws {
        let raw = Data(#"{"status":"unavailable","reason":"chat_response_invalid"}"#.utf8)
        let (data, response) = try await streamed { delegate, session, task in
            let response = HTTPURLResponse(url: HostedAskPLONNProvider.endpointURL, statusCode: 502, httpVersion: nil, headerFields: nil)!
            delegate.urlSession(session, dataTask: task, didReceive: response) { XCTAssertEqual($0, .allow) }
            delegate.urlSession(session, dataTask: task, didReceive: raw)
            delegate.urlSession(session, task: task, didCompleteWithError: nil)
        }
        XCTAssertEqual(data, raw)
        XCTAssertThrowsError(try HostedAskPLONNProvider.validateStatus(response.statusCode, body: data)) {
            XCTAssertEqual($0 as? HostedAskPLONNError, .invalidResponse)
        }
    }

    func testBetaReadinessSeparatesSignInConsentAndReadyWithoutClaimingConnectivity() {
        XCTAssertEqual(HostedAskPLONNConnectionState.resolve(serviceAvailable: false, signedIn: true, consented: true), .unavailable)
        XCTAssertEqual(HostedAskPLONNConnectionState.resolve(serviceAvailable: true, signedIn: false, consented: true), .signInRequired)
        XCTAssertEqual(HostedAskPLONNConnectionState.resolve(serviceAvailable: true, signedIn: true, consented: false), .consentRequired)
        XCTAssertEqual(HostedAskPLONNConnectionState.resolve(serviceAvailable: true, signedIn: true, consented: true), .ready)
        XCTAssertTrue(HostedAskPLONNConnectionState.signInRequired.statusText.contains("Sign in"))
        XCTAssertTrue(HostedAskPLONNConnectionState.consentRequired.statusText.contains("Enable Gemma 4 31B"))
        XCTAssertTrue(HostedAskPLONNConnectionState.ready.statusText.contains("Hosted AI"))
        XCTAssertFalse(HostedAskPLONNConnectionState.ready.statusText.contains("connected"))
    }

    func testGatewayFailuresUseGroundedFallbackWithoutRetryOrExposingErrorBodies() async throws {
        for status in [401, 403, 409, 429, 502, 503, 504] {
            let attempts = TokenSequence()
            let errorBody = Data("{\"error\":\"private-owner fixture.access.token invented 999 points\"}".utf8)
            let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
                _ = await attempts.next()
                return (errorBody, HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!)
            })
            let conversation = AskPLONNConversation()
            let answer = try await conversation.ask(question: research().question, intent: .recentForm, evidence: [fact()],
                                                      ownerID: "private-owner", contextID: "fixture-research", provider: provider)
            XCTAssertEqual(answer.mode, .localResearch)
            XCTAssertNil(answer.providerLabel)
            XCTAssertTrue(answer.text.contains(fact().text))
            XCTAssertTrue(answer.text.contains("unavailable or could not be verified"))
            for secret in ["private-owner", Self.token, "999"] { XCTAssertFalse(answer.text.contains(secret)) }
            let calls = await attempts.count()
            XCTAssertEqual(calls, 1)
            do {
                _ = try await conversation.ask(question: general().question, intent: .generalChat, evidence: [],
                                               ownerID: "private-owner", contextID: "fixture-general", provider: provider)
                XCTFail("General chat failures require explicit retry")
            } catch {
                guard let failure = error as? HostedAskPLONNError else { return XCTFail("Wrong error type") }
                for secret in ["private-owner", Self.token, "999"] { XCTAssertFalse(failure.userFacingMessage.contains(secret)) }
            }
            let history = await conversation.history()
            XCTAssertTrue(history.isEmpty)
            let allCalls = await attempts.count()
            XCTAssertEqual(allCalls, 2)
        }
    }

    func testTransportDeadlineIsUnavailableAndFallsBackOnce() async throws {
        let attempts = TokenSequence()
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
            XCTAssertEqual(request.timeoutInterval, 45)
            _ = await attempts.next()
            throw URLError(.timedOut)
        })
        do { _ = try await provider.answer(research()); XCTFail("Deadline ignored") }
        catch { XCTAssertEqual(error as? HostedAskPLONNError, .unavailable) }
        let answer = try await AskPLONNConversation().ask(question: research().question, intent: .recentForm,
                                                         evidence: [fact()], ownerID: "fixture-owner", provider: provider)
        XCTAssertEqual(answer.mode, .localResearch)
        XCTAssertTrue(answer.text.contains(fact().text))
        let count = await attempts.count()
        XCTAssertEqual(count, 2, "Each manual operation makes one attempt.")
    }

    func testOwnerSwitchDuringCredentialRefreshPreventsTransport() async throws {
        let owner = UUID()
        let credentials = SuspendedHostedToken(owner: owner)
        let provider = HostedAskPLONNProvider(tokenProvider: { try await credentials.fresh(boundTo: owner) }, transport: { _ in
            XCTFail("An account switch during refresh reached transport")
            throw HostedAskPLONNError.invalidResponse
        })
        let request = research()
        let pending = Task { try await provider.answer(request) }
        await credentials.waitUntilRefreshing()
        await credentials.switchOwner(to: UUID())
        do { _ = try await pending.value; XCTFail("Old account's request survived refresh") }
        catch { XCTAssertTrue(error is CancellationError) }
    }

    func testLateResponseCannotEnterNewOwnersConversation() async throws {
        let delayed = SuspendedHostedTransport()
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { try await delayed.send($0) })
        let conversation = AskPLONNConversation()
        let request = research()
        let pending = Task {
            try await conversation.ask(question: request.question, intent: request.intent, evidence: request.evidence,
                                       ownerID: "old-owner", contextID: "fixture-context", provider: provider)
        }
        await delayed.waitUntilStarted()
        await conversation.reset(ownerID: "new-owner", contextID: "fixture-context")
        await delayed.finish(data: try JSONSerialization.data(withJSONObject: envelope(["supported": true, "evidence_ids": ["fact"]])))
        do { _ = try await pending.value; XCTFail("A late response entered another account") }
        catch { XCTAssertEqual(error as? AskPLONNError, .superseded) }
        let turns = await conversation.history()
        XCTAssertTrue(turns.isEmpty)
    }

    func testMatchupRequestAndResponseUseSharedChatContract() async throws {
        let home = fact("home", text: "Fixture Home records a season average of 12 points.")
        let away = fact("away", text: "Fixture Away records a season average of 14 points.")
        let summary = "Fixture Home has lower recorded scoring than Fixture Away. These historical samples limit the comparison, not a prediction."
        let expected = try transport(envelope(["supported": true, "evidence_ids": ["away", "home"], "summary": summary]))
        let provider = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
            XCTAssertEqual(request.url, HostedAskPLONNProvider.endpointURL)
            let wire = try XCTUnwrap(JSONSerialization.jsonObject(with: request.httpBody!) as? [String: Any])
            XCTAssertEqual(Set(wire.keys), Set(["consent_version", "request_id", "schema_version", "question", "intent", "evidence", "prior_turns", "purpose", "matchup_teams"]))
            XCTAssertEqual(wire["purpose"] as? String, "matchupInsights")
            XCTAssertEqual(wire["matchup_teams"] as? [String], ["Fixture Home", "Fixture Away"])
            XCTAssertEqual((wire["prior_turns"] as? [Any])?.count, 0)
            XCTAssertFalse(String(decoding: request.httpBody!, as: UTF8.self).contains("private-owner"))
            return try await expected(request)
        })
        var request = research([home, away])
        request.researchPurpose = .matchupInsights
        request.matchupTeams = ["Fixture Home", "Fixture Away"]
        let answer = try await provider.answer(request)
        XCTAssertEqual(answer.evidence, [away, home])
        XCTAssertEqual(answer.text, "Verified data summary\n\n" + away.text + "\n\n" + home.text)
        XCTAssertEqual(answer.matchupSummary, summary)
        for unsafe in ["Fixture Home will score 999 points.", "Fixture Away has 99 percent probability.", "See https://example.com", "An Unrelated Team has lower scoring."] {
            let unsafeProvider = try self.provider(envelope(["supported": true, "evidence_ids": ["home"], "summary": unsafe]))
            do { _ = try await unsafeProvider.answer(request); XCTFail("Unsafe matchup summary accepted") }
            catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        }
    }

    func testCancellationBeforeTokenDuringTokenAndFromTransport() async throws {
        let request = research()
        let early = HostedAskPLONNProvider(tokenProvider: { XCTFail("Canceled task requested token"); return Self.token }, transport: { _ in
            XCTFail("Canceled task reached transport"); throw HostedAskPLONNError.invalidResponse
        })
        let task = Task { withUnsafeCurrentTask { $0?.cancel() }; return try await early.answer(request) }
        do { _ = try await task.value; XCTFail("Early cancellation ignored") }
        catch { XCTAssertTrue(error is CancellationError) }
        let duringToken = HostedAskPLONNProvider(tokenProvider: {
            withUnsafeCurrentTask { $0?.cancel() }; return Self.token
        }, transport: { _ in XCTFail("Cancellation during token reached transport"); throw HostedAskPLONNError.invalidResponse })
        let tokenTask = Task { try await duringToken.answer(request) }
        do { _ = try await tokenTask.value; XCTFail("Token cancellation ignored") }
        catch { XCTAssertTrue(error is CancellationError) }
        let canceled = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { _ in throw URLError(.cancelled) })
        do { _ = try await canceled.answer(request); XCTFail("Transport cancellation ignored") }
        catch { XCTAssertTrue(error is CancellationError) }
        let successfulResponse = try transport()
        let canceledResult = HostedAskPLONNProvider(tokenProvider: { Self.token }, transport: { request in
            let result = try await successfulResponse(request)
            withUnsafeCurrentTask { $0?.cancel() }
            return result
        })
        let resultTask = Task { try await canceledResult.answer(request) }
        do { _ = try await resultTask.value; XCTFail("Canceled response exposed") }
        catch { XCTAssertTrue(error is CancellationError) }
    }

    func testDefaultSessionDisablesCredentialStorageCookiesCacheAndProxies() {
        let configuration = HostedAskPLONNProvider.sessionConfiguration()
        XCTAssertNil(configuration.httpCookieStorage)
        XCTAssertNil(configuration.urlCredentialStorage)
        XCTAssertNil(configuration.urlCache)
        XCTAssertFalse(configuration.httpShouldSetCookies)
        XCTAssertTrue(configuration.connectionProxyDictionary?.isEmpty == true)
        XCTAssertEqual(configuration.requestCachePolicy, .reloadIgnoringLocalCacheData)
        XCTAssertEqual(configuration.timeoutIntervalForResource, 45)
    }

    /// Exercises delegate callbacks directly. No task is resumed and no socket is opened.
    private func streamed(_ events: (BoundedHostedChatResponse, URLSession, URLSessionDataTask) -> Void) async throws -> (Data, HTTPURLResponse) {
        let session = URLSession(configuration: HostedAskPLONNProvider.sessionConfiguration())
        defer { session.invalidateAndCancel() }
        let task = session.dataTask(with: HostedAskPLONNProvider.endpointURL)
        let delegate = BoundedHostedChatResponse(expectedURL: HostedAskPLONNProvider.endpointURL)
        events(delegate, session, task)
        // Every fixture completes the delegate before start, so start only returns its terminal result.
        return try await withCheckedThrowingContinuation { continuation in
            delegate.start(session: session, request: URLRequest(url: HostedAskPLONNProvider.endpointURL), continuation: continuation)
        }
    }

    func testStreamingDelegateRejectsRedirectsAndOversizedHeadersBeforeReading() async throws {
        do {
            _ = try await streamed { delegate, session, task in
                let redirect = HTTPURLResponse(url: HostedAskPLONNProvider.endpointURL, statusCode: 302, httpVersion: nil, headerFields: nil)!
                delegate.urlSession(session, task: task, willPerformHTTPRedirection: redirect,
                                    newRequest: URLRequest(url: URL(string: "https://other.invalid")!)) { XCTAssertNil($0) }
            }
            XCTFail("Redirect accepted")
        } catch { XCTAssertEqual(error as? HostedAskPLONNError, .invalidResponse) }
        do {
            _ = try await streamed { delegate, session, task in
                let response = HTTPURLResponse(url: HostedAskPLONNProvider.endpointURL, statusCode: 200, httpVersion: nil,
                                               headerFields: ["Content-Length": "32769"])!
                delegate.urlSession(session, dataTask: task, didReceive: response) { XCTAssertEqual($0, .cancel) }
            }
            XCTFail("Oversized declared body accepted")
        } catch { XCTAssertEqual(error as? HostedAskPLONNError, .oversizedPayload) }
    }

    func testStreamingDelegateBoundsUnknownLengthChunksAndHandlesCancellationBeforeStart() async throws {
        let response = HTTPURLResponse(url: HostedAskPLONNProvider.endpointURL, statusCode: 200, httpVersion: nil, headerFields: nil)!
        do {
            _ = try await streamed { delegate, session, task in
                delegate.urlSession(session, dataTask: task, didReceive: response) { XCTAssertEqual($0, .allow) }
                delegate.urlSession(session, dataTask: task, didReceive: Data(repeating: 65, count: 32768))
                delegate.urlSession(session, dataTask: task, didReceive: Data([65]))
            }
            XCTFail("Chunked oversized body accepted")
        } catch { XCTAssertEqual(error as? HostedAskPLONNError, .oversizedPayload) }
        let (data, _) = try await streamed { delegate, session, task in
            delegate.urlSession(session, dataTask: task, didReceive: response) { XCTAssertEqual($0, .allow) }
            delegate.urlSession(session, dataTask: task, didReceive: Data(repeating: 65, count: 32768))
            delegate.urlSession(session, task: task, didCompleteWithError: nil)
        }
        XCTAssertEqual(data.count, 32768)
        do {
            _ = try await streamed { delegate, _, _ in delegate.cancel(); delegate.cancel() }
            XCTFail("Prestart cancellation ignored")
        } catch { XCTAssertTrue(error is CancellationError) }
    }
}

private actor TokenSequence {
    private var calls = 0
    func next() -> String { calls += 1; return "fixture.token.\(calls)" }
    func count() -> Int { calls }
}

private actor SuspendedHostedToken {
    private var owner: UUID
    private var refresh: CheckedContinuation<Void, Never>?
    private var observer: CheckedContinuation<Void, Never>?
    init(owner: UUID) { self.owner = owner }
    func fresh(boundTo expected: UUID) async throws -> String {
        guard owner == expected else { throw CancellationError() }
        await withCheckedContinuation { continuation in
            refresh = continuation
            observer?.resume(); observer = nil
        }
        guard owner == expected else { throw CancellationError() }
        return "fixture.refreshed.token"
    }
    func waitUntilRefreshing() async {
        if refresh != nil { return }
        await withCheckedContinuation { observer = $0 }
    }
    func switchOwner(to next: UUID) {
        owner = next; refresh?.resume(); refresh = nil
    }
}

private actor SuspendedHostedTransport {
    private var completion: CheckedContinuation<(Data, HTTPURLResponse), Error>?
    private var observer: CheckedContinuation<Void, Never>?
    private var url: URL?
    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        url = request.url
        return try await withCheckedThrowingContinuation { continuation in
            completion = continuation
            observer?.resume(); observer = nil
        }
    }
    func waitUntilStarted() async {
        if completion != nil { return }
        await withCheckedContinuation { observer = $0 }
    }
    func finish(data: Data) {
        guard let url else { return }
        completion?.resume(returning: (data, HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!))
        completion = nil
    }
}
