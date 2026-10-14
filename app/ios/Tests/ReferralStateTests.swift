import Foundation
import XCTest
#if REFERRAL_STANDALONE_TESTS
@testable import ReferralCore
#else
@testable import PLONNCore
#endif

final class ReferralStateTests: XCTestCase {
    let a = UUID(), b = UUID()
    func reply(_ owner: UUID, _ status: String, _ code: String? = nil) -> ReferralReply {
        ReferralReply(owner_id: owner, status: status, code: code)
    }
    func testNormalization() {
        XCTAssertEqual(ReferralState.normalized(" \tpartner-1\n"), "PARTNER-1")
        XCTAssertEqual(ReferralState.normalized(" \n"), "")
        XCTAssertEqual(ReferralState.normalized("A B"), "A B") // server rejects internal spaces
    }
    func testOnlyConfirmedServerReplyReportsApplied() {
        var state = ReferralState(); state.switchOwner(a)
        for status in ["none", "invalid", "needs_verification", "unknown"] {
            let token = state.begin(owner: a)!
            state.finish(reply(a, status), token: token)
            XCTAssertNil(state.appliedCode)
            XCTAssertFalse(state.busy)
        }
        let token = state.begin(owner: a)!
        state.finish(reply(a, "applied", "PARTNER"), token: token)
        XCTAssertEqual(state.appliedCode, "PARTNER")
    }
    func testDuplicateRequestsAndIdempotentReplies() {
        var state = ReferralState(); state.switchOwner(a)
        let token = state.begin(owner: a)!
        XCTAssertNil(state.begin(owner: a))
        state.finish(reply(a, "already_applied", "FIRST"), token: token)
        let next = state.begin(owner: a)!
        state.finish(reply(a, "already_attributed", "FIRST"), token: next)
        XCTAssertEqual(state.appliedCode, "FIRST")
    }
    func testSignoutAndAccountSwitchDiscardLateReplies() {
        var state = ReferralState(); state.switchOwner(a)
        let old = state.begin(owner: a)!
        state.switchOwner(nil)
        state.finish(reply(a, "applied", "SECRET"), token: old)
        XCTAssertNil(state.appliedCode)
        XCTAssertNil(state.begin(owner: a))
        state.switchOwner(b)
        XCTAssertNil(state.begin(owner: a))
        let current = state.begin(owner: b)!
        state.fail(token: old)
        state.finish(reply(a, "applied", "SECRET"), token: old)
        XCTAssertEqual(state.request, current)
        XCTAssertNil(state.appliedCode)
        state.finish(reply(b, "applied", "SECOND"), token: current)
        state.switchOwner(a)
        state.finish(reply(a, "applied", "SECRET"), token: old)
        XCTAssertNil(state.appliedCode) // also protects A -> B -> A
    }
    func testFailureRetryAndSameOwnerRefresh() {
        var state = ReferralState(); state.switchOwner(a)
        let token = state.begin(owner: a)!
        state.switchOwner(a)
        XCTAssertEqual(state.request, token)
        state.fail(token: token)
        XCTAssertFalse(state.busy)
        XCTAssertNil(state.appliedCode)
        let retry = state.begin(owner: a)!
        state.finish(reply(a, "applied", "OK"), token: retry)
        XCTAssertEqual(state.appliedCode, "OK")
    }
    func testWrongOwnerResponseFailsWithoutLeakingCode() {
        var state = ReferralState(); state.switchOwner(a)
        let token = state.begin(owner: a)!
        state.finish(reply(b, "applied", "PRIVATE"), token: token)
        XCTAssertNil(state.appliedCode)
        XCTAssertFalse(state.busy)
        XCTAssertFalse(state.message.contains("PRIVATE"))
    }
    func testMalformedSuccessDoesNotClaimApplied() throws {
        var state = ReferralState(); state.switchOwner(a)
        let token = state.begin(owner: a)!
        let data = Data("{\"owner_id\":\"\(a)\",\"status\":\"applied\"}".utf8)
        state.finish(try JSONDecoder().decode(ReferralReply.self, from: data), token: token)
        XCTAssertNil(state.appliedCode)
        XCTAssertFalse(state.message.contains("Referral code applied"))
    }
    func testConfirmationWithoutReferralMetadataDoesNotCreateARequestOrCodeError() {
        for metadata in [nil, "", " ", "\t\n"] as [String?] {
            var state = ReferralState(); state.switchOwner(a)
            XCTAssertNil(state.beginAutomatic(owner: a, pendingCode: metadata))
            XCTAssertNil(state.beginAutomatic(owner: a, pendingCode: metadata))
            XCTAssertFalse(state.busy)
            XCTAssertNil(state.request); XCTAssertNil(state.appliedCode)
            XCTAssertEqual(state.message, "Add a referral code if someone referred you.")
        }
    }
    func testExplicitEmptyOrWhitespaceCodeCannotBecomeAStatusRefreshRequest() {
        var state = ReferralState(); state.switchOwner(a)
        for code in ["", " ", "\t\n"] {
            XCTAssertNil(state.beginRequest(owner: a, code: code))
            XCTAssertFalse(state.busy)
        }
        let refresh = state.beginRequest(owner: a)
        XCTAssertNotNil(refresh); XCTAssertNil(refresh?.code)
        XCTAssertTrue(state.busy)
    }
    func testPendingCodeRequestIsNormalizedAndNeverAppliedBeforeServerConfirmation() {
        var state = ReferralState(); state.switchOwner(a)
        let operation = state.beginAutomatic(owner: a, pendingCode: " \tpartner-1\n")!
        XCTAssertEqual(operation.owner, a); XCTAssertEqual(operation.code, "PARTNER-1")
        XCTAssertTrue(state.isCurrent(operation)); XCTAssertTrue(state.busy)
        XCTAssertNil(state.appliedCode)
        XCTAssertNil(state.beginAutomatic(owner: a, pendingCode: "PARTNER-1"))
        state.finish(reply(a, "applied", "PARTNER-1"), token: operation.token)
        XCTAssertEqual(state.appliedCode, "PARTNER-1")
        XCTAssertNil(state.beginAutomatic(owner: a, pendingCode: "PARTNER-1"))
        XCTAssertNil(state.beginAutomatic(owner: a, pendingCode: "ANOTHER"))
    }
    func testRepeatedPendingMetadataDoesNotReplayAnInvalidCodeOrFailedAutomaticCall() {
        for fails in [false, true] {
            var state = ReferralState(); state.switchOwner(a)
            let operation = state.beginAutomatic(owner: a, pendingCode: "invalid-code")!
            if fails { state.fail(token: operation.token) }
            else { state.finish(reply(a, "invalid"), token: operation.token) }
            XCTAssertNil(state.appliedCode); XCTAssertFalse(state.busy)
            XCTAssertNil(state.beginAutomatic(owner: a, pendingCode: " INVALID-CODE\n"))
            let retry = state.beginRequest(owner: a, code: "CORRECTED")!
            state.finish(reply(a, "applied", "CORRECTED"), token: retry.token)
            XCTAssertEqual(state.appliedCode, "CORRECTED")
        }
    }
    func testQueuedAutomaticRequestsRemainBoundToOriginalOwnerAndToken() {
        var state = ReferralState(); state.switchOwner(a)
        XCTAssertNil(state.beginAutomatic(owner: b, pendingCode: "B-CODE"))
        let queued = state.beginAutomatic(owner: a, pendingCode: "A-CODE")!
        state.switchOwner(b)
        XCTAssertFalse(state.isCurrent(queued))
        let current = state.beginAutomatic(owner: b, pendingCode: "B-CODE")!
        state.finish(reply(a, "applied", "A-CODE"), token: queued.token)
        state.fail(token: queued.token)
        XCTAssertTrue(state.isCurrent(current)); XCTAssertNil(state.appliedCode)
        state.switchOwner(a)
        XCTAssertFalse(state.isCurrent(queued))
        let newA = state.beginAutomatic(owner: a, pendingCode: "A-CODE")!
        XCTAssertNotEqual(newA.token, queued.token)
        state.finish(reply(a, "applied", "A-CODE"), token: queued.token)
        XCTAssertTrue(state.isCurrent(newA)); XCTAssertNil(state.appliedCode)
        state.switchOwner(nil)
        XCTAssertFalse(state.isCurrent(newA)); XCTAssertFalse(state.busy)
    }
    func testStatusRefreshFailureHasDifferentMessageAndPreservesConfirmedAttribution() {
        var state = ReferralState(); state.switchOwner(a)
        let apply = state.beginRequest(owner: a, code: "SERVER-CODE")!
        state.finish(reply(a, "already_applied", "SERVER-CODE"), token: apply.token)
        let refresh = state.beginRequest(owner: a)!
        state.fail(token: refresh.token)
        XCTAssertEqual(state.appliedCode, "SERVER-CODE")
        XCTAssertEqual(state.message, "Couldn’t refresh your referral status. Try again shortly.")
        let submission = state.beginRequest(owner: a, code: "RETRY")!
        state.fail(token: submission.token)
        XCTAssertEqual(state.message, "Couldn’t check your code. Try again shortly.")
        XCTAssertEqual(state.appliedCode, "SERVER-CODE")
    }
    func testMalformedOrUnknownStatusResponseDoesNotClaimCodeSubmissionFailed() {
        var state = ReferralState(); state.switchOwner(a)
        for status in ["unknown", "applied", "already_attributed"] {
            let refresh = state.beginRequest(owner: a)!
            state.finish(reply(a, status, " \n"), token: refresh.token)
            XCTAssertNil(state.appliedCode)
            XCTAssertEqual(state.message, "Couldn’t confirm your referral status. Try refreshing again.")
        }
    }
    func testServerInvalidAndVerificationRepliesRemainAuthoritative() {
        var state = ReferralState(); state.switchOwner(a)
        let invalid = state.beginRequest(owner: a, code: "INVALID")!
        state.finish(reply(a, "invalid"), token: invalid.token)
        XCTAssertNil(state.appliedCode)
        XCTAssertEqual(state.message, "That code isn’t valid or has expired. Check it and try again.")
        let pending = state.beginRequest(owner: a, code: "PENDING")!
        state.finish(reply(a, "needs_verification"), token: pending.token)
        XCTAssertNil(state.appliedCode)
        XCTAssertEqual(state.message, "Confirm your email, then sign in to apply your code.")
        let applied = state.beginRequest(owner: a, code: "INPUT")!
        state.finish(reply(a, "already_attributed", "SERVER-ORIGINAL"), token: applied.token)
        XCTAssertEqual(state.appliedCode, "SERVER-ORIGINAL")
        let refresh = state.beginRequest(owner: a)!
        state.finish(reply(a, "none"), token: refresh.token)
        XCTAssertNil(state.appliedCode)
        XCTAssertEqual(state.message, "Add a referral code if someone referred you.")
    }
}

