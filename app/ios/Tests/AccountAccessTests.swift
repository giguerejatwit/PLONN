import XCTest
@testable import PLONNCore

@MainActor final class AccountAccessTests: XCTestCase {
    func testGuestAndAnonymousCannotCompleteProtectedActions() async {
        for feature in AccountFeature.allCases {
            let gate=AccountAccessCoordinator();var executions=0
            gate.require(feature,ownerID:nil,isAnonymous:nil) {executions += 1}
            XCTAssertEqual(gate.request?.feature,feature);XCTAssertEqual(executions,0)
            gate.complete(ownerID:nil,isAnonymous:false)
            gate.complete(ownerID:UUID(),isAnonymous:true)
            XCTAssertEqual(executions,0);XCTAssertNotNil(gate.request)
            gate.complete(ownerID:UUID(),isAnonymous:false)
            XCTAssertEqual(executions,1);XCTAssertNil(gate.request)
            gate.complete(ownerID:UUID(),isAnonymous:false)
            XCTAssertEqual(executions,1,"A session refresh must not replay the action.")
        }
    }
    func testCancelledSignInNeverReplaysSaveOnLaterLogin() async {
        let gate=AccountAccessCoordinator();var saves=0
        gate.require(.savedResearch,ownerID:nil,isAnonymous:nil) {saves += 1}
        gate.cancel();gate.present()
        gate.complete(ownerID:UUID(),isAnonymous:false)
        XCTAssertEqual(saves,0)
    }
    func testChangingIntentReplacesThePendingAction() async {
        let gate=AccountAccessCoordinator();var actions:[String]=[]
        gate.require(.savedResearch,ownerID:nil,isAnonymous:nil) {actions.append("old")}
        gate.require(.privateModels,ownerID:UUID(),isAnonymous:true) {actions.append("current")}
        gate.complete(ownerID:UUID(),isAnonymous:false)
        XCTAssertEqual(actions,["current"])
        gate.require(.savedResearch,ownerID:nil,isAnonymous:nil) {actions.append("cancelled")}
        gate.present(feature:.odds,creating:true)
        XCTAssertEqual(gate.request?.creating,true)
        gate.complete(ownerID:UUID(),isAnonymous:false)
        XCTAssertEqual(actions,["current"])
    }
    func testAuthenticatedActionsContinueWithoutPromptAndLogoutRequiresSignInAgain() async {
        let gate=AccountAccessCoordinator();var runs=0
        gate.require(.aiResearch,ownerID:UUID(),isAnonymous:false) {runs += 1}
        XCTAssertEqual(runs,1);XCTAssertNil(gate.request)
        gate.require(.aiResearch,ownerID:nil,isAnonymous:nil) {runs += 1}
        XCTAssertEqual(runs,1);XCTAssertEqual(gate.request?.feature,.aiResearch)
    }
}
