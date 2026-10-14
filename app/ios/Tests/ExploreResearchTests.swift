import XCTest
@testable import PLONNCore

final class ExploreResearchTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)
    private func market() -> Market {
        let time = ISO8601DateFormatter().string(from: now.addingTimeInterval(-30))
        return Market(id: "quote", event: "game", sport: "NBA", kind: "prop", name: "Fixture Player", opponent: "Fixture Opponent", label: "Over 20.5", aliases: "", p: 0.6, price: -110, market: "player_points", book: "Fixture book", quotes: [], updated: time, mode: "live", logs: nil, line: 20.5, eventLabel: nil, modelValidated: true, modelUpdated: time, modelVersion: "fixture-v1", eventState: "in", commence: ISO8601DateFormatter().string(from: now.addingTimeInterval(3600)), side: "over")
    }
    func testPriceProbabilityAndSignedEdgeAreSeparate() {
        var quote = market()
        let result = ExploreResearch.assess(quote, now: now)
        XCTAssertEqual(result.impliedProbability!, 110.0 / 210, accuracy: 0.00001)
        XCTAssertEqual(result.estimatedEdge!, 0.6 - 110.0 / 210, accuracy: 0.00001)
        quote.p = 0.4
        let negative = ExploreResearch.assess(quote, now: now)
        XCTAssertLessThan(negative.estimatedEdge!, 0)
        XCTAssertTrue(negative.rationale.contains("below"))
    }
    func testStaleFutureAndMissingModelEvidenceDoNotProduceEdge() {
        let baseline = market()
        var variants: [Market] = []
        var quote = baseline; quote.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-121)); variants.append(quote)
        quote = baseline; quote.modelUpdated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-121)); variants.append(quote)
        quote = baseline; quote.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(1)); variants.append(quote)
        quote = baseline; quote.modelValidated = false; variants.append(quote)
        quote = baseline; quote.modelVersion = " "; variants.append(quote)
        quote = baseline; quote.eventState = "post"; variants.append(quote)
        quote = baseline; quote.mode = "demo"; variants.append(quote)
        quote = baseline; quote.p = .nan; variants.append(quote)
        quote = baseline; quote.event = ""; variants.append(quote)
        for candidate in variants {
            let result = ExploreResearch.assess(candidate, now: now)
            XCTAssertNil(result.estimatedEdge)
            XCTAssertNil(result.modelProbability)
        }
    }
    func testUsesBestExactBookAndExcludesMismatchedOutcomes() {
        let selected = market()
        var best = selected; best.id = "better"; best.book = "Better book"; best.price = 120
        var wrong = selected; wrong.id = "wrong"; wrong.book = "Wrong outcome"; wrong.side = "under"; wrong.price = 500
        let result = ExploreResearch.assess(selected, markets: [selected,best,wrong], now: now)
        XCTAssertEqual(result.comparison.bestBooks, ["Better book"])
        XCTAssertEqual(result.impliedProbability!, 100.0/220, accuracy: 0.00001)
        XCTAssertEqual(result.estimatedEdge!, 0.6-100.0/220, accuracy: 0.00001)
    }
    func testSettlementAndIdentityConflictsFailClosed() {
        var selected = market(); selected.line = 20
        XCTAssertNil(ExploreResearch.assess(selected, now: now).estimatedEdge)
        XCTAssertTrue(ExploreResearch.assess(selected, now: now).uncertainty.contains("push"))
        selected = market()
        var conflict = selected; conflict.line = 21.5
        XCTAssertNil(ExploreResearch.assess(selected, markets:[selected,conflict], now:now).estimatedEdge)
        for sport in ["NFL","NHL"] {
            var moneyline = selected; moneyline.sport = sport; moneyline.kind = "team"; moneyline.market = "h2h"; moneyline.line = nil; moneyline.side = nil
            XCTAssertNil(ExploreResearch.assess(moneyline, now:now).estimatedEdge)
        }
    }

    func testInvalidQuotesNeverCreatePriceEvidence() {
        for price in [0, 99, Double.infinity, Double.nan] {
            var quote = market(); quote.price = price
            let result = ExploreResearch.assess(quote, now: now)
            XCTAssertNil(result.impliedProbability)
            XCTAssertNil(result.estimatedEdge)
        }
        var quote = market(); quote.book = " "
        XCTAssertNil(ExploreResearch.assess(quote, now: now).impliedProbability)
    }
    func testPriceOnlyResearchClearlyStatesMissingEdgeAndFreshness() {
        var quote = market(); quote.modelValidated = nil; quote.updated = nil
        let result = ExploreResearch.assess(quote, now: now)
        XCTAssertNil(result.comparison.bestQuote)
        XCTAssertTrue(result.uncertainty.contains("unavailable"))
        XCTAssertTrue(result.rationale.contains("does not establish an edge"))
    }

    private func slate(count: Int) -> [Market] {
        (0..<count).map { index in
            var row = market()
            row.id = "row-\(index)"; row.event = "event-\(index / 4)"
            row.book = "Book \(index % 4)"; row.price = Double(100 + index % 4 * 10)
            return row
        }
    }

    func testPreparationCapsAndDeduplicatesBeforeComparisons() async throws {
        let rows = slate(count: 2_000)
        let cards = try await ExploreResearch.prepareCards(candidates: [rows[0],rows[0]] + rows,
            markets: rows, now: now)
        XCTAssertEqual(cards.count, 12)
        XCTAssertEqual(Set(cards.map(\.market.id)).count, 12)
        XCTAssertEqual(cards.map(\.market.id), Array(rows.prefix(12)).map(\.id))
        for card in cards {
            XCTAssertEqual(card.assessment.comparison.bestBooks, ["Book 3"])
            XCTAssertEqual(card.assessment.comparison.quotes.count, 4)
        }
        let capped = try await ExploreResearch.prepareCards(candidates: rows, markets: rows, now: now, limit: 10_000)
        XCTAssertEqual(capped.count, 24)
        let empty = try await ExploreResearch.prepareCards(candidates: rows, markets: rows, now: now, limit: 0)
        XCTAssertTrue(empty.isEmpty)
    }

    func testEventIndexRetainsCrossEventQuoteIdentityConflicts() async throws {
        let selected = market()
        var better = selected; better.id = "better"; better.book = "Better book"; better.price = 150
        var conflict = better; conflict.event = "different-event"
        let rows = slate(count: 300) + [selected,better,conflict]
        let cards = try await ExploreResearch.prepareCards(candidates: [selected], markets: rows, now: now)
        let full = ExploreResearch.assess(selected, markets: rows, now: now)
        XCTAssertEqual(cards.first?.assessment.comparison.bestQuote, full.comparison.bestQuote)
        XCTAssertEqual(cards.first?.assessment.comparison.bestBooks, [selected.book])
        XCTAssertEqual(cards.first?.assessment.estimatedEdge, full.estimatedEdge)
        var selectedConflict = selected; selectedConflict.sport = "NHL"
        let conflicted = try await ExploreResearch.prepareCards(candidates: [selected],
            markets: rows + [selectedConflict], now: now)
        XCTAssertNil(conflicted.first?.assessment.estimatedEdge)
    }

    func testEventIndexRetainsHiddenClosedObservationAndStartBoundary() async throws {
        let selected = market()
        var closed = selected; closed.id = "closed-under"; closed.side = "under"; closed.eventState = "final"
        let rows = slate(count: 500) + [selected,closed]
        let cards = try await ExploreResearch.prepareCards(candidates: [selected], markets: rows, now: now)
        XCTAssertNil(cards.first?.assessment.comparison.bestQuote)
        XCTAssertTrue(cards.first?.assessment.comparison.unavailableReason?.contains("closed") == true)
        closed.commence = ISO8601DateFormatter().string(from: now.addingTimeInterval(7200))
        let separateStart = try await ExploreResearch.prepareCards(candidates: [selected],
            markets: [selected,closed], now: now)
        XCTAssertEqual(separateStart.first?.assessment.comparison.bestQuote?.book, selected.book)
    }

    func testCancelledPreparationCannotPublishCards() async throws {
        let rows = slate(count: 20_000)
        let pending = Task { try await ExploreResearch.prepareCards(candidates: rows, markets: rows, now: now) }
        pending.cancel()
        do { _ = try await pending.value; XCTFail("Cancelled preparation returned publishable cards") }
        catch is CancellationError { }
    }

    func testBoundedPreparationPerformanceComparedWithRenderingPath() async throws {
        let rows = slate(count: 128)
        let before = Date()
        let old = rows.map { ExploreResearch.assess($0, markets: rows, now: now) }
        let oldSeconds = Date().timeIntervalSince(before)
        let after = Date()
        let cards = try await ExploreResearch.prepareCards(candidates: rows, markets: rows, now: now)
        let newSeconds = Date().timeIntervalSince(after)
        XCTAssertEqual(old.count, 128); XCTAssertEqual(cards.count, 12)
        for (index, card) in cards.enumerated() {
            XCTAssertEqual(card.assessment.comparison.bestQuote, old[index].comparison.bestQuote)
            XCTAssertEqual(card.assessment.estimatedEdge, old[index].estimatedEdge)
        }
        print("EXPLORE_BENCHMARK rows=128 old_comparisons=128 old_seconds=\(oldSeconds) prepared_comparisons=12 prepared_seconds=\(newSeconds)")
    }
}
