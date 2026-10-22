import XCTest
@testable import PLONNCore

final class AskPLONNModelDraftTests: XCTestCase {
    func testMissingDetailsAskForClarificationInsteadOfDefaultingToAnotherSport() {
        let draft = AskPLONNModelDraft()
        XCTAssertTrue(draft.isValid); XCTAssertFalse(draft.canReview)
        XCTAssertTrue(draft.guidance.contains("sport (NFL, NBA, MLB or NHL)"))
        XCTAssertNil(draft.effectiveLine); XCTAssertNil(draft.effectiveSide)
    }
    func testAllFourSportsAndNativeTargetsCanBeDrafted() {
        for league in ["NFL", "NBA", "MLB", "NHL"] {
            for target in ["Player", "Team"] {
                for stat in LabPropCatalog.options(league: league, target: target) where stat != "Custom stat" {
                    let draft = AskPLONNModelDraft(league: league, target: target, stat: stat)
                    XCTAssertTrue(draft.isValid, league + " " + stat)
                    XCTAssertFalse(draft.canReview)
                }
            }
        }
    }
    func testDraftReviewRequiresLineDirectionAndFantasyScoring() {
        var draft = AskPLONNModelDraft(league: "NBA", target: "Player", subject: "Example Player", stat: "Fantasy points")
        XCTAssertFalse(draft.canReview)
        draft.line = 25.5; draft.side = "Over"
        XCTAssertFalse(draft.canReview)
        draft.scoringPreset = "fanduel_classic"
        XCTAssertTrue(draft.canReview)
        XCTAssertEqual(draft.thresholdDescription, "Over 25.5")
        draft.stat = "Points"
        XCTAssertFalse(draft.isValid, "Fantasy rules must not leak into another target")
    }
    func testBinaryAndFullGameSemantics() {
        let yes = AskPLONNModelDraft(league: "NHL", target: "Player", subject: "Example Player", stat: "Anytime goal scorer", side: "Over")
        XCTAssertTrue(yes.canReview); XCTAssertEqual(yes.effectiveLine, 0.5); XCTAssertEqual(yes.thresholdDescription, "Yes")
        let win = AskPLONNModelDraft(league: "NFL", target: "Team", subject: "San Francisco 49ers", stat: "Moneyline")
        XCTAssertTrue(win.canReview); XCTAssertEqual(win.effectiveLine, 0); XCTAssertEqual(win.thresholdDescription, "To win")
        XCTAssertTrue(win.guidance.contains("Nothing has been saved or tested"))
    }
    func testDraftCannotBypassResearchContractOrCarryPredictionClaims() throws {
        let draft = AskPLONNModelDraft(league: "MLB", target: "Player", subject: "Example Player", stat: "Hits", line: 1.5, side: "Over")
        let request = AskPLONNRequest(question: "Create a model", intent: .generalChat, evidence: [], researchPurpose: .modelCreation)
        var answer = AskPLONNAnswer(text: draft.guidance, evidence: [], mode: .generalExplanation, modelDraft: draft)
        XCTAssertNoThrow(try AskPLONNCore.validateProviderAnswer(answer, for: request))
        XCTAssertThrowsError(try AskPLONNCore.validateProviderAnswer(answer, for: AskPLONNRequest(question: "Explain models", intent: .generalChat, evidence: [])))
        answer.matchupSummary = "Will win"
        XCTAssertThrowsError(try AskPLONNCore.validateProviderAnswer(answer, for: request))
        let claim = AskPLONNAnswer(text: "Saved a winning model", evidence: [], mode: .generalExplanation, modelDraft: draft)
        XCTAssertThrowsError(try AskPLONNCore.validateProviderAnswer(claim, for: request))
    }
    func testWireNullsAndInvalidInputs() throws {
        let data = try JSONEncoder().encode(AskPLONNModelDraft())
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(object.keys), AskPLONNModelDraft.wireKeys)
        XCTAssertTrue(object.values.allSatisfy { $0 is NSNull })
        for draft in [AskPLONNModelDraft(league: "NFL", target: "Player", stat: "Rebounds"),
                      AskPLONNModelDraft(subject: "https://example.com"),
                      AskPLONNModelDraft(league: "NBA", target: "Player", stat: "Points", line: .infinity)] {
            XCTAssertFalse(draft.isValid)
        }
    }
    func testCreationRoutingKeepsExplanationsInOrdinaryChat() {
        for request in ["Create a model for the 49ers", "Can you build me a model?", "Make a rebounds model"] {
            XCTAssertTrue(AskPLONNModelDraft.isCreationRequest(request))
        }
        for question in ["Explain my model", "How accurate is this model?", "Compare these odds"] {
            XCTAssertFalse(AskPLONNModelDraft.isCreationRequest(question))
        }
    }
}
