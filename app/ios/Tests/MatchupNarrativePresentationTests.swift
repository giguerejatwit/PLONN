import XCTest
@testable import PLONNCore

final class MatchupNarrativePresentationTests: XCTestCase {
    typealias J = RichContextJSON
    private let start = "2026-09-29T23:00:00.000Z"
    private let end = "2026-10-13T20:38:00.097Z"

    private func sample(_ start: String, _ end: String, id: String = "fact:1") -> J {
        .object(["factID": .string(id), "windowStart": .string(start), "windowEnd": .string(end)])
    }
    private func claim(_ text: String, samples: [J]) -> MatchupNarrativeV4.Claim {
        .init(.object(["text": .string(text), "samples": .list(samples), "factIDs": .list([.string("fact:1")])]))
    }
    private func display(_ start: String, _ end: String) -> String {
        MatchupNarrativePresentation.text(for: claim("(\(start) to \(end))", samples: [sample(start, end)]))
    }
    private var ios: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }
    private func json(_ path: URL) throws -> J {
        var parser = RichContextJSONParser(try Data(contentsOf: path))
        return try parser.parse()
    }

    func testPublishedFourSportReportsKeepRawClaimsAndEvidenceExact() throws {
        let archive = try json(ios.appendingPathComponent("PLONN/MatchupNarrativeReports-v4.json"))
        let versions = try XCTUnwrap(MatchupNarrativeArchive.trustedVersions)
        var sports = Set<String>()
        for entry in archive["reports"].array {
            let raw = try XCTUnwrap(Data(base64Encoded: entry.string))
            var parser = RichContextJSONParser(raw)
            let envelope = try parser.parse()
            let now = try SharedMatchupReportValidator.date(envelope["generatedAt"].string)
            let result = try MatchupNarrativeV4.validate(raw, now: now, versions: versions)
            let originalClaims = result.claims
            sports.insert(result.event.sport)
            var formatted = 0
            for claim in result.claims {
                let shown = MatchupNarrativePresentation.text(for: claim)
                var restored = shown
                for sample in claim.samples.array {
                    let span = "(\(sample["windowStart"].string) to \(sample["windowEnd"].string))"
                    guard claim.text.contains(span) else { continue }
                    let readable = display(sample["windowStart"].string, sample["windowEnd"].string)
                    XCTAssertNotEqual(readable, span)
                    XCTAssertTrue(shown.contains(readable))
                    restored = restored.replacingOccurrences(of: readable, with: span)
                }
                XCTAssertEqual(restored, claim.text, "Only exact sample spans may change")
                if shown != claim.text { formatted += 1 }
                if result.event.sport == "NHL", claim.id == "home:shotsPerGame" {
                    XCTAssertEqual(shown, claim.text.replacingOccurrences(of: "(\(start) to \(end))", with: "(Sep 29–Oct 13, 2026 UTC)"))
                    XCTAssertTrue(shown.contains("27.6 shots on goal per game across 5 games in the 2026–2027 regular sample"))
                    XCTAssertTrue(shown.contains("do not establish an advantage or forecast"))
                }
            }
            XCTAssertGreaterThan(formatted, 0, result.event.sport)
            XCTAssertEqual(result.claims, originalClaims)
            XCTAssertEqual(result.rawData, raw)
            XCTAssertEqual(try MatchupNarrativeV4.validate(raw, now: now, versions: versions).report, result.report)
        }
        XCTAssertEqual(sports, ["NFL", "NBA", "MLB", "NHL"])
    }

    func testSyntheticFourSportCorpusKeepsCanonicalEnvelopesExact() throws {
        let fixture = try json(ios.deletingLastPathComponent().appendingPathComponent("contracts/matchup-narrative-v4.json"))
        guard case .object(let pins) = fixture["versions"] else { return XCTFail("Missing pins") }
        var sports = Set<String>()
        for row in fixture["cases"].array {
            let raw = Data(row["envelope"].canonical.utf8)
            let now = Date(timeIntervalSince1970: try XCTUnwrap(Double(row["now"].number ?? "")))
            let result = try MatchupNarrativeV4.validate(raw, now: now, versions: pins.mapValues(\.string))
            sports.insert(result.event.sport)
            for claim in result.claims {
                XCTAssertEqual(MatchupNarrativePresentation.text(for: claim), claim.text.replacingOccurrences(
                    of: "(2026-09-01T00:00:00.000Z to 2026-10-13T00:00:00.000Z)", with: "(Sep 1–Oct 13, 2026 UTC)"))
            }
            XCTAssertEqual(result.rawData, raw)
        }
        XCTAssertEqual(sports, ["NFL", "NBA", "MLB", "NHL"])
    }

    func testSameMonthSameDayLeapDayAndCrossYear() {
        XCTAssertEqual(display("2026-10-01T00:00:00Z", "2026-10-13T23:59:59Z"), "(Oct 1–13, 2026 UTC)")
        XCTAssertEqual(display("2026-10-13T01:02:03.004Z", "2026-10-13T20:38:00.097Z"), "(Oct 13, 2026 UTC)")
        XCTAssertEqual(display("2024-02-29T00:00:00Z", "2024-03-01T00:00:00Z"), "(Feb 29–Mar 1, 2024 UTC)")
        XCTAssertEqual(display("2025-12-31T23:30:00Z", "2026-01-01T01:00:00Z"), "(Dec 31, 2025–Jan 1, 2026 UTC)")
    }

    func testExplicitUTCPreventsCalendarDayDrift() {
        XCTAssertEqual(display("2026-09-30T00:30:00+02:00", "2026-10-14T01:00:00+05:00"), "(Sep 29–Oct 13, 2026 UTC)")
        XCTAssertEqual(display("2026-11-01T01:30:00-04:00", "2026-11-01T01:30:00-05:00"), "(Nov 1, 2026 UTC)")
    }

    func testOnlyCompleteSampleSpansChange() {
        let text = "27.6 in 5 games, season 2026–2027 (\(start) to \(end)); reported \(end). Other (\(start) to 2026-10-14T00:00:00Z)."
        let shown = MatchupNarrativePresentation.text(for: claim(text, samples: [sample(start, end)]))
        XCTAssertEqual(shown, "27.6 in 5 games, season 2026–2027 (Sep 29–Oct 13, 2026 UTC); reported \(end). Other (\(start) to 2026-10-14T00:00:00Z).")
    }

    func testMissingUnboundInvalidAndReversedWindowsStayUnchanged() {
        let text = "(\(start) to \(end))"
        XCTAssertEqual(MatchupNarrativePresentation.text(for: claim(text, samples: [])), text)
        XCTAssertEqual(MatchupNarrativePresentation.text(for: claim(text, samples: [sample(start, end, id: "unbound")])), text)
        XCTAssertEqual(display("invalid", end), "(invalid to \(end))")
        XCTAssertEqual(display(start, ""), "(\(start) to )")
        XCTAssertEqual(display(end, start), "(\(end) to \(start))")
    }

    func testMultipleAndDuplicateSampleWindows() {
        let second = "2026-10-14T12:39:09.610Z"
        let text = "First (\(start) to \(end)); second (\(start) to \(second))."
        let shown = MatchupNarrativePresentation.text(for: claim(text, samples: [sample(start, end), sample(start, end), sample(start, second)]))
        XCTAssertEqual(shown, "First (Sep 29–Oct 13, 2026 UTC); second (Sep 29–Oct 14, 2026 UTC).")
    }
}
