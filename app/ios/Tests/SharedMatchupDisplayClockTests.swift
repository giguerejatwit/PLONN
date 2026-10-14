import XCTest
@testable import PLONNCore

final class SharedMatchupDisplayClockTests: XCTestCase {
    struct Fixture: Decodable {let cases:[Row]}
    struct Row: Decodable {let rawUTF8:String;let validatedAt:String}
    func records() throws -> [(SharedMatchupReportResponse,Date)] {
        let url=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("contracts/shared-insights-presentation-v1.json")
        return try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url)).cases.map { row in
            let raw=Data(row.rawUTF8.utf8), now=try SharedMatchupReportValidator.date(row.validatedAt)
            let event=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw).event
            return (try SharedMatchupReportValidator.validate(raw,event:event,now:now),now)
        }
    }
    func testAll91ValuesPreserveFactsAndSuppressAtExactStart() throws {
        let rows=try records();XCTAssertEqual(rows.count,91)
        for (value,now) in rows {
            XCTAssertEqual(try SharedMatchupDisplayClock.update(value,now:now),value)
            let start=try SharedMatchupReportValidator.date(value.event.startISO)
            let before=try SharedMatchupDisplayClock.update(value,now:start.addingTimeInterval(-0.001))
            XCTAssertEqual(before.report,value.report);XCTAssertEqual(before.snapshot,value.snapshot)
            XCTAssertEqual(before.generatedAt,value.generatedAt);XCTAssertEqual(before.sourceFingerprint,value.sourceFingerprint)
            let at=try SharedMatchupDisplayClock.update(value,now:start)
            XCTAssertEqual(at.status,.missing);XCTAssertEqual(at.reason,"event_started")
            XCTAssertNil(at.report);XCTAssertNil(at.snapshot);XCTAssertNil(at.generatedAt)
        }
    }
    func testGenerationAndSourceAgeBoundariesAndClockRollback() throws {
        let (value,now)=try XCTUnwrap(records().first { value,now in
            (try? SharedMatchupReportValidator.date(value.event.startISO)) ?? now > now.addingTimeInterval(86401)
        })
        let generated=try SharedMatchupReportValidator.date(XCTUnwrap(value.generatedAt))
        let checked=try SharedMatchupReportValidator.date(XCTUnwrap(value.sourceCheckedAt))
        let boundary=min(generated,checked).addingTimeInterval(86400)
        XCTAssertEqual(try SharedMatchupDisplayClock.update(value,now:boundary.addingTimeInterval(-0.001)).status,.ready)
        let stale=try SharedMatchupDisplayClock.update(value,now:boundary)
        XCTAssertEqual(stale.status,.stale)
        XCTAssertEqual(stale.reason,generated<=checked ? "generation_expired":"source_expired")
        XCTAssertEqual(stale.report,value.report)
        XCTAssertEqual(try SharedMatchupDisplayClock.update(stale,now:now).status,.stale)
        XCTAssertThrowsError(try SharedMatchupDisplayClock.update(value,now:generated.addingTimeInterval(-1)))
    }
    func testWakeAtBoundaryOrWithinOneMinute() throws {
        let (value,now)=try XCTUnwrap(records().first)
        XCTAssertLessThanOrEqual(try SharedMatchupDisplayClock.nextCheck(value,now:now).timeIntervalSince(now),60)
        let start=try SharedMatchupReportValidator.date(value.event.startISO)
        XCTAssertEqual(try SharedMatchupDisplayClock.nextCheck(value,now:start.addingTimeInterval(-0.5)),start)
    }
}
