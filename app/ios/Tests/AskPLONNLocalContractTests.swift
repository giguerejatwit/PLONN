import XCTest
@testable import PLONNCore

final class AskPLONNLocalContractTests: XCTestCase {
    private actor CountingProvider: AskPLONNProvider {
        var calls = 0
        func answer(_ request: AskPLONNRequest) async throws -> AskPLONNAnswer {
            calls += 1
            throw AskPLONNError.unsupportedQuestion
        }
        func count() -> Int { calls }
    }
    func testMissingAndStaleDataNeverInvokeModel() async throws {
        let provider = CountingProvider(), conversation = AskPLONNConversation()
        for status in [AskPLONNAvailability.unavailable, .stale] {
            let item = AskPLONNEvidence(id: status.rawValue, kind: .comparePrices, title: "Quote", text: "Old or missing quote", source: "Recorded source", availability: status)
            let answer = try await conversation.ask(question: "What are the best prices?", intent: .comparePrices, evidence: [item], ownerID: nil, provider: provider)
            XCTAssertEqual(answer.mode, .localResearch)
            XCTAssertTrue(answer.text.contains("unavailable"))
        }
        let calls = await provider.count()
        XCTAssertEqual(calls, 0)
    }
    func testUnsupportedQuestionDisclosesSourceLimitWithoutProviderClaims() async throws {
        let item = AskPLONNEvidence(id: "actual", kind: .recentForm, title: "Recent games", text: "Observed earlier games only, not a forecast.", source: "MLB Stats API")
        let provider = CountingProvider()
        let answer = try await AskPLONNConversation().ask(question: "What is a guaranteed winner?", intent: .recentForm, evidence: [item], ownerID: nil, provider: provider)
        XCTAssertTrue(answer.text.contains("cannot be answered"))
        XCTAssertEqual(answer.evidence, [item])
        XCTAssertEqual(answer.mode, .localResearch)
        XCTAssertNil(answer.providerLabel)
    }
    func testSourceLinksRequireOfficialHTTPSOriginAndOriginalEvidence() throws {
        let item = AskPLONNEvidence(id: "actual", kind: .recentForm, title: "Recent games", text: "Observed data", source: "MLB Stats API", sourceURLs: ["https://statsapi.mlb.com/api/v1/people/592450/stats?stats=gameLog"])
        let request = AskPLONNRequest(question: "What do these logs show?", intent: .recentForm, evidence: [item])
        XCTAssertNoThrow(try AskPLONNCore.boundedRequest(request))
        for url in ["http://statsapi.mlb.com/api/v1/stats", "https://statsapi.mlb.com.evil.invalid/api/v1/stats", "https://user@statsapi.mlb.com/api/v1/stats", "https://127.0.0.1/private"] {
            var changed = item; changed.sourceURLs = [url]
            XCTAssertThrowsError(try AskPLONNCore.boundedRequest(AskPLONNRequest(question: request.question, intent: request.intent, evidence: [changed])))
        }
        var changed = item; changed.sourceURLs = ["https://site.api.espn.com/apis/site/v2/sports"]
        XCTAssertThrowsError(try AskPLONNCore.validateProviderAnswer(AskPLONNAnswer(text: changed.text, evidence: [changed]), for: request))
    }
}
