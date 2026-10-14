import XCTest
@testable import PLONNCore

final class LabSubmissionTests: XCTestCase {
    private func isolated() -> UserDefaults { UserDefaults(suiteName: "LabSubmissionTests." + UUID().uuidString)! }
    func testRetryRelaunchAndPendingJobRecovery() throws {
        let defaults=isolated(),owner=UUID(),payload=try LabSubmissionStore.canonical(["features":["season","recent"],"line":27.5])
        let store=LabSubmissionStore(defaults:defaults)
        let first=try store.prepare(owner:owner,payload:payload)
        XCTAssertEqual(try store.prepare(owner:owner,payload:payload).key,first.key)
        store.update(first,jobID:String(repeating:"a",count:32))
        let reopened=LabSubmissionStore(defaults:defaults)
        let resumed=try reopened.prepare(owner:owner,payload:payload)
        XCTAssertEqual(resumed.key,first.key)
        XCTAssertEqual(resumed.jobID,String(repeating:"a",count:32))
        XCTAssertThrowsError(try reopened.prepare(owner:owner,payload:payload,retrain:true))
        reopened.update(resumed,terminal:true)
        XCTAssertEqual(try reopened.prepare(owner:owner,payload:payload).key,first.key)
        let retrain=try reopened.prepare(owner:owner,payload:payload,retrain:true)
        XCTAssertNotEqual(retrain.key,first.key)
        XCTAssertNil(retrain.jobID)
    }
    func testOwnerAndSetupIsolationAndCanonicalFeatureOrder() throws {
        let store=LabSubmissionStore(defaults:isolated()),owner=UUID()
        let payload=try LabSubmissionStore.canonical(["features":["recent","season"],"line":1])
        XCTAssertEqual(payload,try LabSubmissionStore.canonical(["line":1,"features":["season","recent"]]))
        let first=try store.prepare(owner:owner,payload:payload)
        XCTAssertNotEqual(first.key,try store.prepare(owner:UUID(),payload:payload).key)
        XCTAssertNotEqual(first.key,try store.prepare(owner:owner,payload:LabSubmissionStore.canonical(["line":2])).key)
    }
    func testCapacityNeverEvictsOutstandingSubmission() throws {
        let store=LabSubmissionStore(defaults:isolated(),capacity:1),owner=UUID()
        let first=try store.prepare(owner:owner,payload:Data("first".utf8))
        XCTAssertThrowsError(try store.prepare(owner:owner,payload:Data("second".utf8)))
        XCTAssertEqual(store.record(owner:owner,payload:first.payload)?.key,first.key)
        store.update(first,terminal:true)
        XCTAssertNoThrow(try store.prepare(owner:owner,payload:Data("second".utf8)))
    }
    func testOutstandingRetrainTakesPriorityUntilItsTerminalStatusIsObserved() throws {
        let defaults=isolated(),owner=UUID(),payload=Data("trained setup".utf8)
        let store=LabSubmissionStore(defaults:defaults)
        let first=try store.prepare(owner:owner,payload:payload)
        store.update(first,jobID:String(repeating:"a",count:32),terminal:true)
        XCTAssertFalse(store.hasOutstanding(owner:owner,payload:payload))
        let retrain=try store.prepare(owner:owner,payload:payload,retrain:true)
        store.update(retrain,jobID:String(repeating:"b",count:32))
        let reopened=LabSubmissionStore(defaults:defaults)
        XCTAssertTrue(reopened.hasOutstanding(owner:owner,payload:payload))
        XCTAssertEqual(reopened.record(owner:owner,payload:payload)?.jobID,String(repeating:"b",count:32))
        XCTAssertThrowsError(try reopened.prepare(owner:owner,payload:payload,retrain:true))
        reopened.update(retrain,terminal:true) // Observed failed/completed/expired job.
        XCTAssertFalse(reopened.hasOutstanding(owner:owner,payload:payload))
        XCTAssertNoThrow(try reopened.prepare(owner:owner,payload:payload,retrain:true))
    }
}
