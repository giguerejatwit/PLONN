import XCTest
@testable import PLONNCore
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

final class LocalAskPLONNProviderTests: XCTestCase {
    private static let matchupSummary = "Bruins recorded lower scoring than Sabres despite higher shot volume in the supplied sample. The limited sample supports a cautious comparison of recorded patterns."
    private func matchupRequest() -> AskPLONNRequest {
        let texts = ["Bruins averaged 2 goals across 4 recorded games.", "Sabres averaged 3 goals across 4 recorded games.",
                     "Bruins averaged 31 shots and Sabres averaged 28 shots in the recorded sample.", "The sample contains 4 games and is not a forecast."]
        return AskPLONNRequest(question: "Explain the observed Bruins and Sabres matchup patterns.", intent: .recentForm,
            evidence: texts.enumerated().map { AskPLONNEvidence(id: "matchup-\($0.offset)", kind: .recentForm, title: "Recorded matchup", text: $0.element, source: "Fixture") },
            ownerID: "private-owner", researchPurpose: .matchupInsights, matchupTeams: ["Bruins", "Sabres"])
    }
    private func matchupTransport(_ response: [String: Any]) throws -> LocalAskPLONNProvider.Transport {
        let content = String(decoding: try JSONSerialization.data(withJSONObject: response), as: UTF8.self)
        let tags = transport()
        let expectedFacts = matchupRequest().evidence
        return { req in
            if req.url?.path == "/api/tags" { return try await tags(req) }
            let body = try XCTUnwrap(req.httpBody)
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
            let options = try XCTUnwrap(parsed["options"] as? [String: Any])
            XCTAssertEqual(options["num_predict"] as? Int, 512)
            XCTAssertEqual(options["temperature"] as? Int, 0)
            XCTAssertEqual(options["num_ctx"] as? Int, 4096)
            let messages = try XCTUnwrap(parsed["messages"] as? [[String: String]])
            XCTAssertEqual(messages.first?["content"], LocalAskPLONNProvider.matchupInstruction)
            let input = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(messages.last!["content"]!.utf8)) as? [String: Any])
            XCTAssertEqual(input["purpose"] as? String, "matchupInsights")
            XCTAssertEqual(input["matchup_teams"] as? [String], ["Bruins", "Sabres"])
            XCTAssertNil(input["priorTurns"])
            XCTAssertNil(input["ownerID"])
            let facts = try XCTUnwrap(input["facts"] as? [[String: String]])
            XCTAssertEqual(facts.compactMap { $0["text"] }, expectedFacts.map(\.text))
            let schema = try XCTUnwrap(parsed["format"] as? [String: Any])
            XCTAssertEqual(Set(schema["required"] as? [String] ?? []), Set(["supported", "evidence_ids", "summary"]))
            XCTAssertEqual(schema["additionalProperties"] as? Bool, false)
            let properties = try XCTUnwrap(schema["properties"] as? [String: [String: Any]])
            XCTAssertEqual(properties["evidence_ids"]?["maxItems"] as? Int, 3)
            XCTAssertEqual(properties["summary"]?["maxLength"] as? Int, 700)
            let result: [String: Any] = ["model": LocalAskPLONNProvider.modelName, "done": true,
                                        "message": ["role": "assistant", "content": content]]
            return (try JSONSerialization.data(withJSONObject: result), HTTPURLResponse(url: req.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
    }
    func testMatchupGeneratedSummaryIsCarriedUnchangedAlongsideExactFacts() async throws {
        let request = matchupRequest()
        let provider = LocalAskPLONNProvider(transport: try matchupTransport(["supported": true,
            "evidence_ids": ["matchup-2", "matchup-0", "matchup-1"], "summary": Self.matchupSummary]), gate: LocalAskPLONNGate())
        let answer = try await provider.answer(request)
        let selected = [request.evidence[2], request.evidence[0], request.evidence[1]]
        XCTAssertEqual(answer.matchupSummary, Self.matchupSummary)
        XCTAssertEqual(answer.evidence, selected)
        XCTAssertEqual(answer.text, "Verified data summary\n\n" + selected.map(\.text).joined(separator: "\n\n"))
        XCTAssertEqual(answer.mode, .provider)
    }
    func testMatchupMalformedSchemaCannotFallBackToLegacySelection() async throws {
        let malformed: [[String: Any]] = [
            ["supported": true, "evidence_ids": ["matchup-0"]],
            ["supported": 1, "evidence_ids": ["matchup-0"], "summary": Self.matchupSummary],
            ["supported": true, "evidence_ids": [1], "summary": Self.matchupSummary],
            ["supported": true, "evidence_ids": ["matchup-0"], "summary": 123],
            ["supported": true, "evidence_ids": ["matchup-0"], "summary": Self.matchupSummary, "source": "Invented"]]
        for response in malformed {
            do { _ = try await LocalAskPLONNProvider(transport: try matchupTransport(response), gate: LocalAskPLONNGate()).answer(matchupRequest()); XCTFail("Accepted malformed matchup response") }
            catch { XCTAssertEqual(error as? LocalAskPLONNError, .invalidResponse) }
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
            do { _ = try await LocalAskPLONNProvider(transport: try matchupTransport(response), gate: LocalAskPLONNGate()).answer(matchupRequest()); XCTFail("Accepted unsafe matchup response: \(response)") }
            catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        }
        do { _ = try await LocalAskPLONNProvider(transport: try matchupTransport(["supported": false, "evidence_ids": [], "summary": ""]), gate: LocalAskPLONNGate()).answer(matchupRequest()); XCTFail("Accepted unsupported matchup") }
        catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedQuestion) }
    }
    private func request() -> AskPLONNRequest {
        AskPLONNRequest(question: "What were the recorded points?", intent: .recentForm,
            evidence: [AskPLONNEvidence(id: "fact", kind: .recentForm, title: "Form", text: "The recorded result was 12 points.", source: "Fixture")], ownerID: "private-owner")
    }
    func testMatchupInvalidContextFailsBeforeLocalTransport() async {
        var noTeams = matchupRequest(); noTeams.matchupTeams = nil
        var noPurpose = matchupRequest(); noPurpose.researchPurpose = nil
        var duplicateTeams = matchupRequest(); duplicateTeams.matchupTeams = ["Bruins", " bruins "]
        var longTeam = matchupRequest(); longTeam.matchupTeams = [String(repeating: "界", count: 101), "Sabres"]
        var history = matchupRequest(); history.priorTurns = [AskPLONNQuestionContext(question: "Earlier question", intent: .recentForm)]
        let unavailable = AskPLONNRequest(question: "Explain", intent: .recentForm,
            evidence: [AskPLONNEvidence(id: "stale", kind: .recentForm, title: "Recorded", text: "Bruins sample", source: "Fixture", availability: .stale)],
            researchPurpose: .matchupInsights, matchupTeams: ["Bruins", "Sabres"])
        let provider = LocalAskPLONNProvider(transport: { _ in XCTFail("Invalid matchup reached transport"); throw LocalAskPLONNError.invalidResponse }, gate: LocalAskPLONNGate())
        for request in [noTeams, noPurpose, duplicateTeams, longTeam, history, unavailable] {
            do { _ = try await provider.answer(request); XCTFail("Accepted invalid matchup context") }
            catch { XCTAssertEqual(error as? AskPLONNError, .invalidEvidence) }
        }
    }
    func testLegacyResearchRejectsSummaryAndRetainsFourEvidenceCapacity() async throws {
        let invalid = LocalAskPLONNProvider(transport: transport(response: ["supported": true, "evidence_ids": ["fact"], "summary": Self.matchupSummary]), gate: LocalAskPLONNGate())
        do { _ = try await invalid.answer(request()); XCTFail("Legacy response accepted summary") }
        catch { XCTAssertEqual(error as? LocalAskPLONNError, .invalidResponse) }
        let original = request().evidence[0]
        let facts = (0..<4).map { AskPLONNEvidence(id: "legacy-\($0)", kind: original.kind, title: original.title, text: original.text, source: original.source) }
        let legacy = LocalAskPLONNProvider(transport: transport(ids: facts.map(\.id)), gate: LocalAskPLONNGate())
        let answer = try await legacy.answer(AskPLONNRequest(question: "Describe the recorded results", intent: .recentForm, evidence: facts))
        XCTAssertEqual(answer.evidence, facts)
        XCTAssertNil(answer.matchupSummary)
    }
    private func transport(ids: [String] = ["fact"], supported: Bool = true, digest: String = LocalAskPLONNProvider.localModelsDigest, remote: Bool = false, response: [String: Any]? = nil) -> LocalAskPLONNProvider.Transport {
        { req in
            let url = try XCTUnwrap(req.url)
            XCTAssertEqual(url.port, 11435)
            let json: [String: Any]
            if url.path == "/api/tags" {
                var model: [String: Any] = ["name": LocalAskPLONNProvider.modelName, "digest": digest, "size": 12345, "details": ["format": "gguf"]]
                if remote { model["remote_host"] = "https://remote.invalid" }
                json = ["models": [model]]
            } else {
                XCTAssertEqual(req.timeoutInterval, 35)
                XCTAssertEqual(req.httpMethod, "POST")
                let body = try XCTUnwrap(req.httpBody)
                XCTAssertFalse(String(decoding: body, as: UTF8.self).contains("private-owner"))
                let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
                XCTAssertEqual(parsed["stream"] as? Bool, false)
                XCTAssertEqual(parsed["model"] as? String, "gemma4:26b")
                XCTAssertEqual(parsed["think"] as? Bool, false)
                let options = try XCTUnwrap(parsed["options"] as? [String: Any])
                XCTAssertEqual(options["num_predict"] as? Int, 128)
                XCTAssertEqual(options["num_ctx"] as? Int, 4096)
                XCTAssertEqual(options["temperature"] as? Int, 0)
                let messages = try XCTUnwrap(parsed["messages"] as? [[String: String]])
                XCTAssertEqual(messages.first?["content"], LocalAskPLONNProvider.selectionInstruction)
                let input = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(messages.last!["content"]!.utf8)) as? [String: Any])
                XCTAssertNil(input["purpose"])
                XCTAssertNil(input["matchup_teams"])
                let schema = try XCTUnwrap(parsed["format"] as? [String: Any])
                XCTAssertEqual(Set(schema["required"] as? [String] ?? []), Set(["supported", "evidence_ids"]))
                let content = try JSONSerialization.data(withJSONObject: response ?? ["supported": supported, "evidence_ids": ids])
                json = ["model": LocalAskPLONNProvider.modelName, "done": true, "message": ["role": "assistant", "content": String(decoding: content, as: UTF8.self)]]
            }
            return (try JSONSerialization.data(withJSONObject: json), HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
    }
    private func generalRequest(_ question: String = "What is implied probability?") -> AskPLONNRequest {
        AskPLONNRequest(question: question, intent: .generalChat, evidence: [], ownerID: "private-owner")
    }
    private func generalTransport(_ response: [String: Any] = ["kind": "explanation", "message": "In a hypothetical example, decimal odds of 2.00 imply 50% before margin. These are illustrative, not current odds."]) throws -> LocalAskPLONNProvider.Transport {
        let content = String(decoding: try JSONSerialization.data(withJSONObject: response), as: UTF8.self)
        let tags = transport()
        return { req in
            if req.url?.path == "/api/tags" { return try await tags(req) }
            let body = try XCTUnwrap(req.httpBody)
            let bodyText = String(decoding: body, as: UTF8.self)
            XCTAssertFalse(bodyText.contains("private-owner"))
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
            XCTAssertEqual(parsed["model"] as? String, LocalAskPLONNProvider.modelName)
            XCTAssertEqual(parsed["stream"] as? Bool, false)
            XCTAssertEqual(parsed["think"] as? Bool, false)
            let options = try XCTUnwrap(parsed["options"] as? [String: Any])
            XCTAssertEqual(options["num_predict"] as? Int, 768)
            XCTAssertEqual(options["num_ctx"] as? Int, 4096)
            let messages = try XCTUnwrap(parsed["messages"] as? [[String: String]])
            XCTAssertEqual(messages.first?["content"], LocalAskPLONNProvider.generalInstruction)
            let inputData = try XCTUnwrap(messages.last?["content"]?.data(using: .utf8))
            let input = try XCTUnwrap(JSONSerialization.jsonObject(with: inputData) as? [String: Any])
            XCTAssertNil(input["facts"])
            XCTAssertNil(input["evidence"])
            XCTAssertNil(input["ownerID"])
            XCTAssertNil(input["purpose"])
            XCTAssertNil(input["matchup_teams"])
            let schema = try XCTUnwrap(parsed["format"] as? [String: Any])
            XCTAssertEqual(schema["additionalProperties"] as? Bool, false)
            let properties = try XCTUnwrap(schema["properties"] as? [String: [String: Any]])
            XCTAssertEqual(properties["message"]?["maxLength"] as? Int, 2000)
            XCTAssertEqual(properties["kind"]?["enum"] as? [String], ["explanation", "needs_research", "unsupported"])
            let result: [String: Any] = ["model": LocalAskPLONNProvider.modelName, "done": true, "message": ["role": "assistant", "content": content]]
            return (try JSONSerialization.data(withJSONObject: result), HTTPURLResponse(url: req.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
    }
    func testGeneralChatWithoutSelectionUsesLocalTransportAndLabelsHypotheticalOddsAsGeneral() async throws {
        let provider = LocalAskPLONNProvider(transport: try generalTransport(), gate: LocalAskPLONNGate())
        let answer = try await AskPLONNConversation().ask(question: "What is implied probability?", intent: .generalChat, evidence: [], ownerID: "private-owner", contextID: "general-chat", provider: provider)
        XCTAssertEqual(answer.mode, .generalExplanation)
        XCTAssertTrue(answer.text.contains("hypothetical"))
        XCTAssertTrue(answer.text.contains("2.00"))
        XCTAssertTrue(answer.text.contains("50%"))
        XCTAssertFalse(answer.text.contains("Verified"))
        XCTAssertFalse(answer.requiresResearchContext)
        XCTAssertTrue(answer.evidence.isEmpty)
        XCTAssertEqual(answer.providerLabel, provider.displayName)
    }
    func testGeneralFollowupTransportRetainsAssistantMemoryWithoutResearchFacts() async throws {
        let message = "First, models can learn noise. Second, validation uses unseen examples."
        let response = try generalTransport(["kind": "explanation", "message": message])
        let provider = LocalAskPLONNProvider(transport: { req in
            if req.url?.path == "/api/chat" {
                let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: req.httpBody!) as? [String: Any])
                let messages = try XCTUnwrap(parsed["messages"] as? [[String: String]])
                let input = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(messages.last!["content"]!.utf8)) as? [String: Any])
                if input["question"] as? String == "Explain the second point." {
                    let prior = try XCTUnwrap(input["priorTurns"] as? [[String: String]])
                    XCTAssertEqual(prior, [["question": "How can models overfit?", "intent": "generalChat", "generalResponse": message]])
                } else { XCTAssertNil(input["priorTurns"]) }
            }
            return try await response(req)
        }, gate: LocalAskPLONNGate())
        let conversation = AskPLONNConversation()
        _ = try await conversation.ask(question: "How can models overfit?", intent: .generalChat, evidence: [], ownerID: "private-owner", contextID: "general-chat", provider: provider)
        let answer = try await conversation.ask(question: "Explain the second point.", intent: .generalChat, evidence: [], ownerID: "private-owner", contextID: "general-chat", provider: provider)
        XCTAssertEqual(answer.mode, .generalExplanation)
    }
    func testGeneralDataRequestsClarifyAndCannotTurnUnverifiedStatsIntoAnExplanation() async throws {
        let clarification = "Which player or model do you mean? You can optionally attach research context, or ask about the concept generally."
        let provider = LocalAskPLONNProvider(transport: try generalTransport(["kind": "needs_research", "message": clarification]), gate: LocalAskPLONNGate())
        let answer = try await provider.answer(generalRequest("What does my model setup tell me?"))
        XCTAssertTrue(answer.requiresResearchContext)
        XCTAssertEqual(answer.text, AskPLONNCore.generalResearchGuidance().text)
        XCTAssertEqual(answer.mode, .generalExplanation)
        let unsafe = LocalAskPLONNProvider(transport: try generalTransport(["kind": "explanation", "message": "Aaron Judge scored 999 today."]), gate: LocalAskPLONNGate())
        let protected = try await unsafe.answer(generalRequest("How has Aaron Judge performed recently?"))
        XCTAssertTrue(protected.requiresResearchContext)
        XCTAssertFalse(protected.text.contains("999"))
        XCTAssertTrue(protected.text.contains("optionally attach"))
        let unsafeClarifier = LocalAskPLONNProvider(transport: try generalTransport(["kind": "needs_research", "message": "Aaron Judge scored 999 today. Attach research for more."]), gate: LocalAskPLONNGate())
        let clarified = try await unsafeClarifier.answer(generalRequest("How has Aaron Judge performed recently?"))
        XCTAssertTrue(clarified.requiresResearchContext)
        XCTAssertFalse(clarified.text.contains("999"))
        XCTAssertEqual(clarified.providerLabel, unsafeClarifier.displayName)
    }
    func testGeneralUnsupportedRequestsCannotBecomeBettingAdviceOrTeamAverageProjections() async throws {
        let provider = LocalAskPLONNProvider(transport: try generalTransport(["kind": "explanation", "message": "A guaranteed result is 999 points."]), gate: LocalAskPLONNGate())
        for question in ["Give me a guaranteed winning bet", "Project a player's score from team averages"] {
            let answer = try await provider.answer(generalRequest(question))
            XCTAssertFalse(answer.text.contains("999"))
            XCTAssertTrue(answer.text.contains("can’t recommend bets"))
            XCTAssertFalse(answer.requiresResearchContext)
        }
        let refusal = LocalAskPLONNProvider(transport: try generalTransport(["kind": "unsupported", "message": "I can explain probability instead. What would you like to understand?"]), gate: LocalAskPLONNGate())
        let refused = try await refusal.answer(generalRequest("Help me decide"))
        XCTAssertEqual(refused.mode, .generalExplanation)
    }
    func testStatQuestionsAndTypedNonExplanationsNeverExposeModelNumericClaims() async throws {
        for kind in ["explanation", "needs_research", "unsupported"] {
            let provider = LocalAskPLONNProvider(transport: try generalTransport(["kind": kind, "message": "The player scored 999 points."]), gate: LocalAskPLONNGate())
            for question in ["How many points did LeBron James score?", "how many hits did aaron judge have?"] {
                let answer = try await provider.answer(generalRequest(question))
                XCTAssertTrue(answer.requiresResearchContext, question)
                XCTAssertFalse(answer.text.contains("999"))
                XCTAssertEqual(answer.text, AskPLONNCore.generalResearchGuidance().text)
                XCTAssertEqual(answer.providerLabel, provider.displayName)
            }
        }
        for kind in ["needs_research", "unsupported"] {
            let provider = LocalAskPLONNProvider(transport: try generalTransport(["kind": kind, "message": "The Dodgers won 10 games. Attach a model to learn more."]), gate: LocalAskPLONNGate())
            let answer = try await provider.answer(generalRequest("Tell me about the Dodgers"))
            XCTAssertFalse(answer.text.contains("10 games"))
            XCTAssertEqual(answer.text, kind == "needs_research" ? AskPLONNCore.generalResearchGuidance().text : AskPLONNCore.generalUnsupportedGuidance().text)
            XCTAssertEqual(answer.requiresResearchContext, kind == "needs_research")
        }
    }
    func testGeneralResponseRejectsInvalidSchemaLinksCitationsAndOversizedText() async throws {
        let invalid: [[String: Any]] = [
            ["kind": "other", "message": "Hello"], ["kind": true, "message": "Hello"],
            ["kind": "explanation", "message": "Hello", "source": "invented"],
            ["kind": "explanation"], ["kind": "explanation", "message": 123],
            ["kind": "explanation", "message": " \n"], ["kind": "explanation", "message": String(repeating: "a", count: 2001)],
            ["kind": "explanation", "message": "See https://example.com"], ["kind": "explanation", "message": "Sources: ESPN"],
            ["kind": "needs_research", "message": "Read [source](https://example.com)"], ["kind": "explanation", "message": "This is true [1]"]
        ]
        for payload in invalid {
            let provider = LocalAskPLONNProvider(transport: try generalTransport(payload), gate: LocalAskPLONNGate())
            do { _ = try await provider.answer(generalRequest()); XCTFail("Accepted malformed general response") }
            catch { XCTAssertEqual(error as? LocalAskPLONNError, .invalidResponse) }
        }
    }
    func testGeneralRequestsKeepPinnedIdentityCancellationAndEvidenceBoundaries() async {
        for provider in [LocalAskPLONNProvider(transport: transport(digest: "wrong"), gate: LocalAskPLONNGate()), LocalAskPLONNProvider(transport: transport(remote: true), gate: LocalAskPLONNGate())] {
            do { _ = try await provider.answer(generalRequest()); XCTFail("Accepted unsafe general model") }
            catch { XCTAssertEqual(error as? LocalAskPLONNError, .unverifiedModel) }
        }
        let canceled = LocalAskPLONNProvider(transport: { _ in throw URLError(.cancelled) }, gate: LocalAskPLONNGate())
        do { _ = try await canceled.answer(generalRequest()); XCTFail("General cancellation ignored") }
        catch { XCTAssertTrue(error is CancellationError) }
        let invalid = AskPLONNRequest(question: "Explain", intent: .generalChat, evidence: request().evidence)
        let provider = LocalAskPLONNProvider(transport: { _ in XCTFail("General evidence reached transport"); throw LocalAskPLONNError.invalidResponse }, gate: LocalAskPLONNGate())
        do { _ = try await provider.answer(invalid); XCTFail("General evidence accepted") }
        catch { XCTAssertEqual(error as? AskPLONNError, .invalidEvidence) }
    }
    func testExactFactSelectionOmitsOwnerAndBoundsOptions() async throws {
        let provider = LocalAskPLONNProvider(transport: transport(), gate: LocalAskPLONNGate())
        let answer = try await provider.answer(request())
        XCTAssertEqual(answer.text, "Verified data summary\n\nThe recorded result was 12 points.")
        XCTAssertEqual(answer.evidence, request().evidence)
        XCTAssertEqual(answer.mode, .provider)
    }
    func testFollowupPromptContainsQuestionsAndTopicsButOnlyCurrentFacts() async throws {
        let response = transport()
        let provider = LocalAskPLONNProvider(transport: { req in
            if req.url?.path == "/api/chat" {
                let body = try XCTUnwrap(req.httpBody)
                let bodyText = String(decoding: body, as: UTF8.self)
                XCTAssertFalse(bodyText.contains("private-context"))
                let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
                let messages = try XCTUnwrap(parsed["messages"] as? [[String: String]])
                let inputData = try XCTUnwrap(messages.last?["content"]?.data(using: .utf8))
                let input = try XCTUnwrap(JSONSerialization.jsonObject(with: inputData) as? [String: Any])
                if input["question"] as? String == "What does that mean?" {
                    let prior = try XCTUnwrap(input["priorTurns"] as? [[String: String]])
                    XCTAssertEqual(prior, [["question": "How were the recent points?", "intent": "recentForm"]])
                    XCTAssertEqual(Set(prior[0].keys), Set(["question", "intent"]))
                    let facts = try XCTUnwrap(input["facts"] as? [[String: String]])
                    XCTAssertEqual(facts.map { $0["text"] }, ["CURRENT CLAIM: The recorded result was 14 points."])
                    XCTAssertFalse(bodyText.contains("OLD ANSWER CLAIM"))
                    XCTAssertTrue(messages.first?["content"]?.contains("Only current facts authorize an answer") == true)
                    XCTAssertTrue(messages.first?["content"]?.contains("Explicit context in the current question takes precedence") == true)
                } else {
                    XCTAssertNil(input["priorTurns"])
                }
            }
            return try await response(req)
        }, gate: LocalAskPLONNGate())
        let conversation = AskPLONNConversation()
        let oldFact = AskPLONNEvidence(id: "fact", kind: .recentForm, title: "Old", text: "OLD ANSWER CLAIM: The recorded result was 12 points.", source: "Old source")
        _ = try await conversation.ask(question: "How were the recent points?", intent: .recentForm, evidence: [oldFact], ownerID: "private-owner", contextID: "private-context", provider: provider)
        let currentFact = AskPLONNEvidence(id: "fact", kind: .recentForm, title: "Current", text: "CURRENT CLAIM: The recorded result was 14 points.", source: "Current source")
        let answer = try await conversation.ask(question: "What does that mean?", intent: .recentForm, evidence: [currentFact], ownerID: "private-owner", contextID: "private-context", provider: provider)
        XCTAssertEqual(answer.mode, .provider)
        XCTAssertEqual(answer.evidence, [currentFact])
        XCTAssertFalse(answer.text.contains("OLD ANSWER CLAIM"))
    }
    func testPriorQuestionCannotAuthorizeMissingEvidenceOrBypassPayloadBounds() async {
        var followup = request()
        followup.priorTurns = [AskPLONNQuestionContext(question: "Explain the previous 999 point claim", intent: .recentForm)]
        do {
            _ = try await LocalAskPLONNProvider(transport: transport(ids: ["previous-fact"]), gate: LocalAskPLONNGate()).answer(followup)
            XCTFail("A prior question authorized absent evidence")
        } catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        followup.priorTurns = Array(repeating: AskPLONNQuestionContext(question: "q", intent: .recentForm), count: 7)
        let provider = LocalAskPLONNProvider(transport: { _ in
            XCTFail("Unbounded history reached transport")
            throw LocalAskPLONNError.invalidResponse
        }, gate: LocalAskPLONNGate())
        do { _ = try await provider.answer(followup); XCTFail("Unbounded history accepted") }
        catch { XCTAssertEqual(error as? AskPLONNError, .invalidHistory) }
    }
    func testEndpointsAreLoopbackOnly() {
        for raw in ["https://127.0.0.1:11434", "http://example.com:11434", "http://localhost:11434", "http://127.0.0.1:11434/remote", "http://user@127.0.0.1:11434", "http://127.0.0.1:11434?x=1"] {
            XCTAssertFalse(LocalAskPLONNProvider.allowedEndpoint(URL(string: raw)!))
        }
        XCTAssertTrue(LocalAskPLONNProvider.allowedEndpoint(URL(string: "http://127.0.0.1:12345")!))
    }
    func testGemma3AndCloudSelectionsAreRejectedBeforeTransport() async {
        for model in ["gemma3:12b", "gemma4:31b-cloud"] {
            let provider = LocalAskPLONNProvider(model: model, transport: { _ in
                XCTFail("An unsupported model reached transport")
                throw LocalAskPLONNError.invalidResponse
            }, gate: LocalAskPLONNGate())
            do { _ = try await provider.answer(request()); XCTFail("Accepted an unsupported model") }
            catch { XCTAssertEqual(error as? LocalAskPLONNError, .invalidEndpoint) }
        }
    }
    func testUnknownAndDuplicateEvidenceIDsRejected() async {
        for ids in [["invented"], ["fact", "fact"]] {
            do { _ = try await LocalAskPLONNProvider(transport: transport(ids: ids), gate: LocalAskPLONNGate()).answer(request()); XCTFail("Accepted unsafe IDs") }
            catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer) }
        }
    }
    func testUnsupportedAndEmptySelectionAreTyped() async {
        for pair in [(false, [String]()), (true, [String]())] {
            do { _ = try await LocalAskPLONNProvider(transport: transport(ids: pair.1, supported: pair.0), gate: LocalAskPLONNGate()).answer(request()); XCTFail("Accepted unsupported question") }
            catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedQuestion) }
        }
    }
    func testContextMismatchRefusalPreservesQuestionAndOnlySuppliedFacts() async throws {
        for question in ["How has LeBron James performed recently?", "What were the recorded rebounds?"] {
            var mismatched = request()
            mismatched = AskPLONNRequest(question: question, intent: mismatched.intent, evidence: mismatched.evidence)
            let refusal = transport(ids: [], supported: false)
            let provider = LocalAskPLONNProvider(transport: { req in
                if req.url?.path == "/api/chat" {
                    let body = try XCTUnwrap(req.httpBody)
                    let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
                    let messages = try XCTUnwrap(parsed["messages"] as? [[String: String]])
                    let userText = try XCTUnwrap(messages.last?["content"]?.data(using: .utf8))
                    let input = try XCTUnwrap(JSONSerialization.jsonObject(with: userText) as? [String: Any])
                    XCTAssertEqual(input["question"] as? String, question)
                    let facts = try XCTUnwrap(input["facts"] as? [[String: String]])
                    XCTAssertEqual(facts.map { $0["text"] }, ["The recorded result was 12 points."])
                    XCTAssertEqual(messages.first?["content"], LocalAskPLONNProvider.selectionInstruction)
                }
                return try await refusal(req)
            }, gate: LocalAskPLONNGate())
            let answer = try await AskPLONNConversation().ask(question: question, intent: mismatched.intent,
                evidence: mismatched.evidence, ownerID: nil, provider: provider)
            XCTAssertEqual(answer.mode, .localResearch)
            XCTAssertNil(answer.providerLabel)
            XCTAssertTrue(answer.text.contains("cannot be answered"))
            XCTAssertFalse(answer.text.contains("LeBron"))
            XCTAssertFalse(answer.text.contains("rebounds"))
            XCTAssertEqual(answer.evidence, mismatched.evidence)
        }
    }
    func testRemoteAndDigestMismatchNeverChat() async {
        for provider in [LocalAskPLONNProvider(transport: transport(digest: "wrong"), gate: LocalAskPLONNGate()), LocalAskPLONNProvider(transport: transport(remote: true), gate: LocalAskPLONNGate())] {
            do { _ = try await provider.answer(request()); XCTFail("Accepted unverified model") }
            catch { XCTAssertEqual(error as? LocalAskPLONNError, .unverifiedModel) }
        }
    }
    func testGateSingleFlightAndRollingRateLimit() async throws {
        let gate = LocalAskPLONNGate(clock: { Date(timeIntervalSince1970: 1) })
        try await gate.acquire()
        do { try await gate.acquire(); XCTFail("Concurrent request allowed") } catch { XCTAssertEqual(error as? LocalAskPLONNError, .capacity) }
        await gate.release()
        for _ in 0..<11 { try await gate.acquire(); await gate.release() }
        do { try await gate.acquire(); XCTFail("Rate limit exceeded") } catch { XCTAssertEqual(error as? LocalAskPLONNError, .capacity) }
    }
    func testCancelledTransportBecomesCancellationError() async {
        let provider = LocalAskPLONNProvider(transport: { _ in throw URLError(.cancelled) }, gate: LocalAskPLONNGate())
        do { _ = try await provider.answer(request()); XCTFail("Cancellation ignored") } catch { XCTAssertTrue(error is CancellationError) }
    }
    func testResponseAndRequestBounds() async {
        let oversized = LocalAskPLONNProvider(transport: { req in
            (Data(repeating: 65, count: 32769), HTTPURLResponse(url: req.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }, gate: LocalAskPLONNGate())
        do { _ = try await oversized.answer(request()); XCTFail("Oversized response accepted") } catch { XCTAssertEqual(error as? LocalAskPLONNError, .oversizedPayload) }
        let noFacts = AskPLONNRequest(question: "Predict a winner", intent: .recentForm, evidence: [])
        do { _ = try await LocalAskPLONNProvider(transport: transport(), gate: LocalAskPLONNGate()).answer(noFacts); XCTFail("No evidence accepted") } catch { XCTAssertEqual(error as? AskPLONNError, .unsupportedQuestion) }
    }
}
