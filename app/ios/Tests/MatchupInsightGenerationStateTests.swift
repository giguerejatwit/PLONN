import XCTest
@testable import PLONNCore

final class MatchupInsightGenerationStateTests: XCTestCase {
    private func context(owner: String? = "owner-a", contextID: String = "nhl-game-a", evidence: String = "Official source · 5 games",
                         ready: Bool = true, provider: Bool = true, local: Bool = false, active: Bool = true,
                         empty: Bool = false) -> MatchupInsightGenerationContext {
        let insight = MatchupInsight(id: "home-goals", text: "Home recorded goal production.", evidence: evidence,
                                    marketID: "original-player-prop", topic: "Goals", team: "Home")
        return MatchupInsightGenerationContext(snapshot: GemmaMatchupInsightSnapshot(contextID: contextID,
            homeTeam: "Home", awayTeam: "Away", ownerID: owner, insights: empty ? [] : [insight]),
            ready: ready, providerAvailable: provider, usesLocalProvider: local, active: active)
    }

    func testAppearanceAndEligibilityChangesNeverCreateRequestTickets() {
        let state = MatchupInsightGenerationState()
        // Reading the new eligibility after consent, sign-in, or scene reentry
        // cannot create a request. Only the button calls begin.
        for candidate in [context(provider: false), context(), context(owner: "owner-b"),
                          context(active: false), context(active: true), context(contextID: "nhl-game-b")] {
            _ = candidate.canGenerate
            XCTAssertNil(state.pending)
        }
    }

    func testHostedRequestsRequireOwnerProviderEvidenceReadyAndActiveScene() {
        var state = MatchupInsightGenerationState()
        for candidate in [context(owner: nil), context(owner: " \n"), context(provider: false),
                          context(ready: false), context(active: false), context(empty: true)] {
            XCTAssertNil(state.begin(in: candidate))
            XCTAssertNil(state.pending)
        }
        XCTAssertNotNil(state.begin(in: context()))
    }

    func testLocalPrototypeCanBeExplicitlyRequestedWithoutHostedAccount() throws {
        var state = MatchupInsightGenerationState()
        let candidate = context(owner: nil, local: true)
        let ticket = try XCTUnwrap(state.begin(in: candidate))
        XCTAssertTrue(state.accepts(ticket, in: candidate))
        XCTAssertTrue(state.finish(ticket, in: candidate))
        XCTAssertNil(state.pending)
    }

    func testRepeatedTapsCannotStartConcurrentRequestsAndRetryRequiresNewBegin() throws {
        var state = MatchupInsightGenerationState()
        let candidate = context()
        let first = try XCTUnwrap(state.begin(in: candidate))
        XCTAssertNil(state.begin(in: candidate))
        // Both success and failure finish the ticket; neither issues a retry.
        XCTAssertTrue(state.finish(first, in: candidate))
        XCTAssertNil(state.pending)
        XCTAssertFalse(state.finish(first, in: candidate))
        let explicitRetry = try XCTUnwrap(state.begin(in: candidate))
        XCTAssertNotEqual(first.id, explicitRetry.id)
        XCTAssertFalse(state.accepts(first, in: candidate))
    }

    func testOwnerMatchupEvidenceConsentAndSceneChangesRejectLateResponses() throws {
        var state = MatchupInsightGenerationState()
        let original = context()
        let ticket = try XCTUnwrap(state.begin(in: original))
        for changed in [context(owner: "owner-b"), context(owner: nil), context(contextID: "nhl-game-b"),
                        context(evidence: "Changed source · 6 games"), context(provider: false),
                        context(ready: false), context(active: false), context(local: true)] {
            XCTAssertFalse(state.accepts(ticket, in: changed))
            XCTAssertFalse(state.finish(ticket, in: changed))
        }
        state.invalidate()
        XCTAssertFalse(state.accepts(ticket, in: original), "Returning to the old owner/context cannot revive a canceled request.")
        XCTAssertNil(state.pending)
        let next = try XCTUnwrap(state.begin(in: original))
        XCTAssertFalse(state.accepts(ticket, in: original))
        XCTAssertTrue(state.accepts(next, in: original))
    }

    func testChangedOriginalPlayerLinkInvalidatesTicket() throws {
        var state = MatchupInsightGenerationState()
        let original = context()
        let ticket = try XCTUnwrap(state.begin(in: original))
        let row = original.snapshot.insights[0]
        let changedRow = MatchupInsight(id: row.id, text: row.text, evidence: row.evidence,
                                       marketID: "different-player-prop", topic: row.topic, team: row.team)
        let changed = MatchupInsightGenerationContext(snapshot: GemmaMatchupInsightSnapshot(
            contextID: original.snapshot.contextID, homeTeam: "Home", awayTeam: "Away", ownerID: "owner-a", insights: [changedRow]),
            ready: true, providerAvailable: true, usesLocalProvider: false, active: true)
        XCTAssertFalse(state.accepts(ticket, in: changed))
    }
}
