import XCTest
@testable import PLONNCore
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Synthetic offline fixtures, not captured phone prompts or model responses.
final class Build6MatchupContractReproductionTests: XCTestCase {
    func testGroundedParaphrasesSurviveHTTP200ContractValidation() async throws {
        let fact = AskPLONNEvidence(id: "fixture-passing", kind: .recentForm, title: "Passing",
            text: "Seattle Seahawks average 240 passing yards per game; Denver Broncos allow 210 passing yards per game.",
            source: "Synthetic offline fixture")
        let request = AskPLONNRequest(question: "Explain the observed matchup.", intent: .recentForm,
            evidence: [fact], researchPurpose: .matchupInsights,
            matchupTeams: ["Seattle Seahawks", "Denver Broncos"])
        for (summary, accepted) in [
            ("Seattle Seahawks passing production is compared with Denver Broncos allowance. These observations describe a limited sample.", true),
            ("Seattle Seahawks passing production contrasts with Denver Broncos allowance. These observations describe a limited sample.", true)
        ] {
            let body = try JSONSerialization.data(withJSONObject: [
                "schema_version": 2, "model": HostedAskPLONNProvider.modelName,
                "provider": "ollama_cloud",
                "response": ["supported": true, "evidence_ids": [fact.id], "summary": summary]
            ])
            let provider = HostedAskPLONNProvider(tokenProvider: { "offline.fixture.token" }, transport: { http in
                (body, HTTPURLResponse(url: http.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
            })
            do {
                let answer = try await provider.answer(request)
                XCTAssertTrue(accepted)
                XCTAssertEqual(answer.matchupSummary, summary)
            } catch {
                XCTAssertFalse(accepted)
                XCTAssertEqual(error as? AskPLONNError, .unsupportedAnswer)
                XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: error), "answer_contract")
            }
        }
    }
    func testSharedContractCorpus() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("contracts/matchup-summary-v1.json")
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let teams = try XCTUnwrap(corpus["teams"] as? [String])
        let facts = try XCTUnwrap(corpus["evidence"] as? [[String: String]]).map {
            AskPLONNEvidence(id: $0["id"]!, kind: .recentForm, title: $0["title"]!, text: $0["text"]!, source: "Offline fixture")
        }
        for item in try XCTUnwrap(corpus["cases"] as? [[String: Any]]) {
            let caseFacts = (item["evidence"] as? [[String: String]])?.map {
                AskPLONNEvidence(id: $0["id"]!, kind: .recentForm, title: $0["title"]!, text: $0["text"]!, source: "Offline fixture")
            } ?? facts
            XCTAssertEqual(AskPLONNCore.validMatchupSummary(item["summary"] as! String, evidence: caseFacts, teams: teams),
                           item["accepted"] as! Bool, item["id"] as! String)
        }
    }

    func testUnsupportedResponseAndFailureCategoriesRemainDistinctAndPayloadFree() async throws {
        let fact = AskPLONNEvidence(id: "fixture", kind: .recentForm, title: "Passing", text: "Seattle Seahawks passing production.", source: "Offline fixture")
        let request = AskPLONNRequest(question: "Explain observed matchup.", intent: .recentForm, evidence: [fact],
            researchPurpose: .matchupInsights, matchupTeams: ["Seattle Seahawks", "Denver Broncos"])
        let data = try JSONSerialization.data(withJSONObject: ["schema_version": 2, "model": HostedAskPLONNProvider.modelName,
            "provider": "ollama_cloud", "response": ["supported": false, "evidence_ids": [], "summary": ""]])
        let provider = HostedAskPLONNProvider(tokenProvider: { "offline.fixture" }, transport: { http in
            (data, HTTPURLResponse(url: http.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        })
        do { _ = try await provider.answer(request); XCTFail("Unsupported response must not become generated commentary") }
        catch {
            XCTAssertEqual(error as? AskPLONNError, .unsupportedQuestion)
            XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: error), "unsupported_question")
        }
        XCTAssertEqual(AskPLONNFailureDiagnostic.matchupMessage(for: AskPLONNError.unsupportedAnswer),
                       "The AI explanation did not match the recorded evidence. Observed data is available below.")
        let privateError = NSError(domain: "private-payload-must-not-appear", code: 1)
        XCTAssertEqual(AskPLONNFailureDiagnostic.category(for: privateError), "other")
        XCTAssertEqual(AskPLONNFailureDiagnostic.matchupMessage(for: privateError),
                       "The AI explanation could not be completed. Observed data is available below.")
    }

}
