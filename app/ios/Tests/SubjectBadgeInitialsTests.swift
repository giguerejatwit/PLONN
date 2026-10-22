import XCTest
@testable import PLONNCore

final class SubjectBadgeInitialsTests: XCTestCase {
    func testNamesRetainReadableInitialsWithoutAnImageSource() {
        XCTAssertEqual(SubjectBadgeInitials.make("Rhamondre Stevenson"), "RS")
        XCTAssertEqual(SubjectBadgeInitials.make("Los Angeles Kings"), "LAK")
        XCTAssertEqual(SubjectBadgeInitials.make("  Émile  Dupont \n"), "ÉD")
        XCTAssertEqual(SubjectBadgeInitials.make("D'Andre Swift"), "DS")
        XCTAssertEqual(SubjectBadgeInitials.make("... --"), "")
        XCTAssertEqual(SubjectBadgeInitials.make(""), "")
    }
}
