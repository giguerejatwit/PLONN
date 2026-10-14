import Foundation

struct ExploreResearchAssessment: Sendable {
    let impliedProbability: Double?
    let modelProbability: Double?
    /// Model probability minus the best usable price probability, in probability units.
    let estimatedEdge: Double?
    let freshness: String
    let rationale: String
    let uncertainty: String
    let comparison: AskPLONNPriceComparison
}

struct ExploreResearchCard: Sendable {
    let market: Market
    let assessment: ExploreResearchAssessment
}

enum ExploreResearch {
    /// Limit comparisons before doing expensive evidence work. Rendering uses
    /// immutable cards and never assesses the whole slate on MainActor.
    static func prepareCards(candidates: [Market], markets: [Market], now: Date,
                             limit: Int = 12) async throws -> [ExploreResearchCard] {
        try Task.checkCancellation()
        let worker = Task.detached(priority: .userInitiated) {
            try prepare(candidates: candidates, markets: markets, now: now, limit: limit)
        }
        return try await withTaskCancellationHandler {
            let cards = try await worker.value
            try Task.checkCancellation()
            return cards
        } onCancel: { worker.cancel() }
    }

    private struct EventKey: Hashable {
        let sport: String
        let event: String
        init(_ market: Market) { sport = market.sport; event = market.event }
    }

    private static func prepare(candidates: [Market], markets: [Market], now: Date,
                                limit: Int) throws -> [ExploreResearchCard] {
        let count = max(0, min(limit, 24))
        guard count > 0 else { return [] }
        var selected: [Market] = [], seen = Set<String>()
        for (index, market) in candidates.enumerated() {
            if index.isMultiple(of: 128) { try Task.checkCancellation() }
            guard market.mode == "live", market.kind != "outright", !market.id.isEmpty,
                  BuilderLogic.decimalOdds(market.price) != nil,
                  seen.insert(market.id).inserted else { continue }
            selected.append(market)
            if selected.count == count { break }
        }
        guard !selected.isEmpty else { return [] }

        // Index raw observations once, retaining all copies of event row IDs.
        // Cross-event ID reuse and hidden closed-game observations must still
        // reach the strict comparator before it filters outcomes or timestamps.
        var rowsByID: [String: [Market]] = [:]
        var eventIDs: [EventKey: Set<String>] = [:]
        for (index, market) in markets.enumerated() {
            if index.isMultiple(of: 128) { try Task.checkCancellation() }
            rowsByID[market.id, default: []].append(market)
            eventIDs[EventKey(market), default: []].insert(market.id)
        }
        var cards: [ExploreResearchCard] = []
        for market in selected {
            try Task.checkCancellation()
            var ids = eventIDs[EventKey(market)] ?? []
            ids.insert(market.id)
            var relevant: [Market] = []
            for id in ids.sorted() {
                try Task.checkCancellation()
                relevant.append(contentsOf: rowsByID[id] ?? [])
            }
            cards.append(ExploreResearchCard(market: market,
                assessment: assess(market, markets: relevant, now: now)))
        }
        try Task.checkCancellation()
        return cards
    }

    static func assess(_ market: Market, markets: [Market]? = nil, now: Date = Date()) -> ExploreResearchAssessment {
        let comparison = AskPLONNResearch.priceComparison(market: market, markets: markets ?? [market], now: now)
        let best = comparison.bestQuote
        let edge = comparison.edgePercentagePoints.map { $0 / 100 }
        let rationale: String
        if let edge {
            rationale = edge > 0 ? "The validated estimate is above the probability implied by the best available price." : edge < 0 ? "The validated estimate is below the probability implied by the best available price." : "The validated estimate matches the probability implied by the best available price."
        } else {
            rationale = "A sportsbook price shows the break-even probability. Recent form adds context; it does not establish an edge."
        }
        let freshness = best == nil ? comparison.unavailableReason ?? "No current verified price comparison is available." : "Compared timestamped prices within 2 minutes · \(comparison.quotes.count) usable \(comparison.quotes.count == 1 ? "book" : "books")"
        let reason = comparison.unavailableReason ?? comparison.edgeUnavailableReason
        return ExploreResearchAssessment(impliedProbability: best?.impliedProbability,
            modelProbability: edge == nil ? nil : comparison.modelProbability, estimatedEdge: edge,
            freshness: freshness, rationale: rationale,
            uncertainty: edge == nil ? "Estimated edge unavailable: " + reason : "Estimated edge is uncertain. Sportsbook margin is included; returns are not guaranteed.", comparison: comparison)
    }
}
