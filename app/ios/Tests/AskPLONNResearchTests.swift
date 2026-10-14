import XCTest
@testable import PLONNCore

private actor AskResearchReadLog {
    private var successful: [String] = []
    func record(_ url: URL) { successful.append(url.absoluteString) }
    func urls() -> [String] { successful }
}

final class AskPLONNResearchTests: XCTestCase {
    private let now = sportsDate("2026-10-08T15:00:00Z")!
    private func market(book: String = "Book A", price: Double = -110, age: Double = 30) -> Market {
        Market(id: book, event: "event", sport: "NBA", kind: "prop", name: "Fixture Player", opponent: "Wrong team",
            label: "Over 20.5", aliases: "", p: 0.99, price: price, market: "player_points", book: book, quotes: [],
            updated: ISO8601DateFormatter().string(from: now.addingTimeInterval(-age)), mode: "live", line: 20.5,
            eventLabel: "Away at Home", commence: "2026-10-09T20:00:00Z", side: "Over")
    }
    private func game(id: String = "game", value: Double = 20, player: String = "99") -> PlayerGame {
        PlayerGame(id: id, date: now.addingTimeInterval(-3 * 86400), opponent: "Verified Opponent", value: value, workload: "30 MIN",
            sport: "NBA", playerID: player, playerTeamID: "1", opponentID: "2", marketKey: "player_points", season: 2027)
    }
    private func history(games: [PlayerGame], context: OpponentHistoryContext? = nil, name: String? = "Fixture Player", player: String? = "99") -> PlayerHistoryResult {
        PlayerHistoryResult(games: games, opponentGames: games, context: context, source: "ESPN", warnings: [], availability: "No previous appearances.", playerName: name, playerID: player)
    }
    func testHistoryEvidencePreservesOnlyOfficialRecordedSourceURLs() {
        let log = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/99/gamelog?season=2027"
        let roster = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/1/roster"
        var observed = history(games: [game()])
        observed.sourceURLs = [log, roster, log, "http://site.api.espn.com/not-secure", "https://site.api.espn.com.evil.test/fake", "https://user:password@statsapi.mlb.com/api/v1/people/99/stats", "https://example.com/invented"]
        let recent = AskPLONNResearch.recentForm(market: market(), history: observed, now: now)
        XCTAssertEqual(recent.sourceURLs, [log, roster])
        let selected = market()
        observed = history(games: [game()], context: OpponentHistoryContext(sport: "NBA", playerID: "99", playerName: selected.name,
            marketKey: selected.market, opponentID: "2", opponentName: "Verified Opponent", targetStart: sportsDate(selected.commence!)!,
            cutoff: now, targetEventID: selected.event, source: "ESPN", requestedSeasons: [2027], loadedSeasons: [2027]))
        observed.sourceURLs = [log, roster]
        XCTAssertEqual(AskPLONNResearch.opponentHistory(market: selected, history: observed, now: now).sourceURLs, [log, roster])
        XCTAssertTrue(AskPLONNResearch.comparePrices(market: selected, markets: [selected, market(book: "Book B")], now: now).sourceURLs.isEmpty)
        XCTAssertTrue(AskPLONNResearch.modelSetup(recipe: recipe(), now: now).sourceURLs.isEmpty)
    }
    func testConcurrentHistoryLoadsKeepActualSuccessfulURLProvenanceSeparate() async throws {
        let reads = AskResearchReadLog()
        let directory = Data(#"{"sports":[]}"#.utf8)
        let log = Data(#"{"filters":[{"name":"season","value":"2027"}],"names":["points","minutes"],"events":{"101":{"id":"101","gameDate":"2026-10-05T20:00:00Z","leagueAbbreviation":"NBA","homeTeamId":"1","awayTeamId":"2","homeTeamScore":"20","awayTeamScore":"10","gameResult":"W","team":{"id":"1","isAllStar":false},"opponent":{"id":"2","displayName":"Verified Opponent"}}},"seasonTypes":[{"displayName":"2026-27 Regular Season","categories":[{"events":[{"eventId":"101","stats":["12","30"]}]}]}]}"#.utf8)
        let service = PlayerHistoryService(transport: { url in
            if url.path.hasSuffix("/teams") { await reads.record(url); return directory }
            if url.path.hasSuffix("/search") {
                let name = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "query" })?.value ?? ""
                let playerID = name == "Other Player" ? "88" : "99"
                let data = try JSONSerialization.data(withJSONObject: ["count": 1, "pageCount": 1, "items": [["id": playerID, "displayName": name, "type": "player", "sport": "basketball", "league": "nba"]]])
                await reads.record(url); return data
            }
            if url.path.hasSuffix("/gamelog"), URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.contains(where: { $0.name == "season" && $0.value == "2027" }) == true {
                await reads.record(url); return log
            }
            throw URLError(.badServerResponse)
        })
        var first = market(); first.eventLabel = nil
        var second = first; second.id = "second"; second.name = "Other Player"
        let firstMarket = first, secondMarket = second
        async let firstResult = service.load(market: firstMarket, seasonCount: 2, now: now)
        async let secondResult = service.load(market: secondMarket, seasonCount: 2, now: now)
        let (a, b) = await (firstResult, secondResult)
        XCTAssertEqual(a.games.count, 1); XCTAssertEqual(b.games.count, 1)
        XCTAssertEqual(a.sourceURLs.count, 3); XCTAssertEqual(b.sourceURLs.count, 3)
        XCTAssertTrue(a.sourceURLs.first?.contains("/athletes/99/gamelog") == true)
        XCTAssertTrue(b.sourceURLs.first?.contains("/athletes/88/gamelog") == true)
        XCTAssertTrue(a.sourceURLs.contains { $0.contains("/athletes/99/gamelog") })
        XCTAssertFalse(a.sourceURLs.contains { $0.contains("/athletes/88/gamelog") })
        XCTAssertTrue(b.sourceURLs.contains { $0.contains("/athletes/88/gamelog") })
        XCTAssertFalse(b.sourceURLs.contains { $0.contains("/athletes/99/gamelog") })
        XCTAssertFalse((a.sourceURLs + b.sourceURLs).contains { $0.contains("season=2026") })
        let actualReads = Set(await reads.urls())
        XCTAssertTrue(Set(a.sourceURLs + b.sourceURLs).isSubset(of: actualReads))
        XCTAssertTrue(a.sourceURLs.allSatisfy { URL(string: $0).map(PlayerHistoryService.allowedURL) == true })
    }

    func testRecentFormRequiresPlayerBindingAndPreservesObservedZero() {
        let verified = AskPLONNResearch.recentForm(market: market(), history: history(games: [game(value: 0)]), now: now)
        XCTAssertEqual(verified.availability, .available); XCTAssertEqual(verified.sampleSize, 1)
        XCTAssertTrue(verified.text.contains("averaged 0.0")); XCTAssertTrue(verified.text.contains("Small sample"))
        XCTAssertEqual(AskPLONNResearch.recentForm(market: market(), history: history(games: [game()], name: "Other Player"), now: now).availability, .unavailable)
        XCTAssertEqual(AskPLONNResearch.recentForm(market: market(), history: history(games: [game()], player: nil), now: now).availability, .unavailable)
        XCTAssertEqual(AskPLONNResearch.recentForm(market: market(), history: history(games: [game(), game(id: "other", player: "88")]), now: now).availability, .unavailable)
    }
    func testRecentFormWithoutThresholdReportsOnlyObservedResults() {
        var selected = market()
        selected.line = nil
        let result = AskPLONNResearch.recentForm(market: selected, history: history(games: [game(value: 12)]), now: now)
        XCTAssertEqual(result.availability, .available)
        XCTAssertTrue(result.text.contains("averaged 12.0"))
        XCTAssertTrue(result.text.contains("Observed player results, not a forecast."))
        for claim in ["threshold", "above", "below", "equal", "probability"] {
            XCTAssertFalse(result.text.lowercased().contains(claim))
        }
    }
    func testRecentConflictsAreQuarantinedAndFutureAndWrongStatsExcluded() {
        var future = game(id: "future"); future = PlayerGame(id: future.id, date: now.addingTimeInterval(1), opponent: future.opponent, value: future.value, workload: future.workload, sport: future.sport, playerID: future.playerID, playerTeamID: future.playerTeamID, opponentID: future.opponentID, marketKey: future.marketKey, season: future.season)
        var wrongStat = game(id: "wrong"); wrongStat.marketKey = "player_assists"
        let result = AskPLONNResearch.recentForm(market: market(), history: history(games: [game(id: "conflict", value: 20), game(id: "conflict", value: 21), game(id: "good", value: 10), future, wrongStat]), now: now)
        XCTAssertEqual(result.sampleSize, 1); XCTAssertTrue(result.text.contains("averaged 10.0"))
    }
    func testOpponentUsesVerifiedContextRatherThanMarketOpponent() {
        let selected = market()
        let context = OpponentHistoryContext(sport: "NBA", playerID: "99", playerName: selected.name, marketKey: selected.market,
            opponentID: "2", opponentName: "Verified Opponent", targetStart: sportsDate(selected.commence!)!, cutoff: now,
            targetEventID: selected.event, source: "ESPN", requestedSeasons: [2027], loadedSeasons: [2027])
        let result = AskPLONNResearch.opponentHistory(market: selected, history: history(games: [game(value: 25)], context: context), now: now)
        XCTAssertEqual(result.availability, .available); XCTAssertEqual(result.sampleSize, 1)
        XCTAssertTrue(result.text.contains("Verified Opponent")); XCTAssertFalse(result.text.contains("Wrong team"))
        XCTAssertEqual(AskPLONNResearch.opponentHistory(market: selected, history: history(games: [game()]), now: now).availability, .unavailable)
        var other = selected; other.name = "Other Player"
        XCTAssertEqual(AskPLONNResearch.opponentHistory(market: other, history: history(games: [game()], context: context), now: now).availability, .unavailable)
    }
    func testComparePricesMatchesExactOutcomeAndNeverUsesModelProbability() {
        let selected = market(), better = market(book: "Book B", price: 120)
        var wrongLine = market(book: "Wrong line", price: 500); wrongLine.line = 21.5
        var wrongSide = market(book: "Wrong side", price: 500); wrongSide.side = "Under"
        var wrongEvent = market(book: "Wrong event", price: 500); wrongEvent.event = "other"
        var wrongStart = market(book: "Wrong start", price: 500); wrongStart.commence = "2026-10-09T21:00:00Z"
        var wrongSport = market(book: "Wrong sport", price: 500); wrongSport.sport = "NFL"
        var demo = market(book: "Demo", price: 500); demo.mode = "demo"
        let result = AskPLONNResearch.comparePrices(market: selected, markets: [selected, better, wrongLine, wrongSide, wrongEvent, wrongStart, wrongSport, demo], now: now)
        XCTAssertEqual(result.availability, .available); XCTAssertEqual(result.sampleSize, 2)
        XCTAssertTrue(result.text.contains("Highest recorded payout for this exact selection: Book B"))
        XCTAssertFalse(result.text.contains("Wrong line")); XCTAssertFalse(result.text.contains("99%")); XCTAssertFalse(result.text.contains("Demo"))
    }
    func testNumericPropsAndTotalsRequireThresholdAndCoherentSidesButMoneylineDoesNot() {
        var missingLine = market(); missingLine.line = nil
        var missingSecond = market(book: "Book B"); missingSecond.line = nil
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: missingLine, markets: [missingLine, missingSecond], now: now).availability, .unavailable)
        var arbitrarySide = market(); arbitrarySide.side = "Win"
        var arbitrarySecond = market(book: "Book B"); arbitrarySecond.side = "Win"
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: arbitrarySide, markets: [arbitrarySide, arbitrarySecond], now: now).availability, .unavailable)
        var total = market(); total.kind = "team"; total.market = "totals"; total.line = nil
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: total, markets: [total], now: now).availability, .unavailable)
        var spread = market(); spread.kind = "team"; spread.market = "spreads"; spread.line = -3.5
        spread.name = "Home"; spread.side = "Home"
        var otherSpread = spread; otherSpread.id = "other"; otherSpread.book = "Book B"; otherSpread.price = 120
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: spread, markets: [spread, otherSpread], now: now).availability, .available)
        var moneyline = spread; moneyline.market = "h2h"; moneyline.line = nil
        var otherMoneyline = moneyline; otherMoneyline.id = "other"; otherMoneyline.book = "Book B"; otherMoneyline.price = 120
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: moneyline, markets: [moneyline, otherMoneyline], now: now).availability, .available)
        moneyline.line = 0
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: moneyline, markets: [moneyline], now: now).availability, .unavailable)
    }

    func testStaleFutureMissingAndConflictingPricesCannotWinComparison() {
        let selected = market(), stale = market(book: "Stale", price: 500, age: 121), future = market(book: "Future", price: 600, age: -1)
        var missing = market(book: "Missing", price: 700); missing.updated = nil
        let result = AskPLONNResearch.comparePrices(market: selected, markets: [selected, stale, future, missing], now: now)
        XCTAssertEqual(result.sampleSize, 1); XCTAssertEqual(result.availability, .available)
        XCTAssertTrue(result.text.contains("Stale +500: stale")); XCTAssertTrue(result.text.contains("Highest recorded payout"))
        XCTAssertTrue(result.text.contains("limited comparison coverage"))
        let staleOnly = AskPLONNResearch.comparePrices(market: selected, markets: [stale], now: now)
        XCTAssertEqual(staleOnly.availability, .stale)
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: selected, markets: [future, missing], now: now).availability, .unavailable)
        var conflict = selected; conflict.price = 150
        let quarantined = AskPLONNResearch.comparePrices(market: selected, markets: [selected, conflict, market(book: "Book B", price: 120)], now: now)
        XCTAssertEqual(quarantined.sampleSize, 1); XCTAssertFalse(quarantined.text.contains("Book A -110"))
    }
    func testFreshPrimaryCannotMakeUnknownAgeBundledQuoteCurrent() {
        var selected = market(); selected.quotes = [Quote(book: "Book B", price: 500)]
        let bundledOnly = AskPLONNResearch.comparePrices(market: selected, markets: [selected], now: now)
        XCTAssertEqual(bundledOnly.sampleSize, 1); XCTAssertEqual(bundledOnly.availability, .available)
        XCTAssertTrue(bundledOnly.text.contains("Book B: bundled quote timestamp unavailable"))
        XCTAssertFalse(bundledOnly.text.contains("Book B +500"))
        XCTAssertTrue(bundledOnly.text.contains("Highest recorded payout"))
        let standalone = market(book: "Book B", price: 120)
        let verified = AskPLONNResearch.comparePrices(market: selected, markets: [selected, standalone], now: now)
        XCTAssertEqual(verified.sampleSize, 2); XCTAssertEqual(verified.availability, .available)
        XCTAssertTrue(verified.text.contains("Book B +120"))
        XCTAssertTrue(verified.text.contains("Highest recorded payout for this exact selection: Book B"))
        XCTAssertFalse(verified.text.contains("Book B +500"))
        selected.updated = nil
        XCTAssertEqual(AskPLONNResearch.comparePrices(market: selected, markets: [selected], now: now).sampleSize, 0)
    }
    private func validatedMarket(probability: Double = 0.6, modelAge: Double = 30, state: String? = "pre") -> Market {
        var selected = market()
        selected.p = probability; selected.modelValidated = true; selected.modelVersion = "verified-v1"
        selected.modelUpdated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-modelAge))
        selected.eventState = state
        return selected
    }
    func testStructuredBestPriceComputesRawImpliedProbabilityAndSignedModelDifference() throws {
        let selected = validatedMarket(), better = market(book: "Book B", price: 120)
        let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected, better], now: now)
        let best = try XCTUnwrap(result.bestQuote)
        XCTAssertEqual(best.book, "Book B"); XCTAssertEqual(best.price, 120)
        XCTAssertEqual(best.decimal, 2.2, accuracy: 0.000001)
        XCTAssertEqual(best.impliedProbability, 1 / 2.2, accuracy: 0.000001)
        XCTAssertEqual(try XCTUnwrap(result.edgePercentagePoints), 14.5454545, accuracy: 0.000001)
        XCTAssertEqual(result.modelProbability, 0.6)
        XCTAssertEqual(result.modelVersion, "verified-v1")
        XCTAssertEqual(result.comparedAt, now); XCTAssertEqual(result.selectionID, selected.id)
        XCTAssertEqual(result.expiresAt, now.addingTimeInterval(90))
        XCTAssertEqual(result.availability, .available); XCTAssertNil(result.unavailableReason)
        XCTAssertTrue(result.edgeUnavailableReason.isEmpty)
        let paragraph = AskPLONNResearch.comparePrices(market: selected, markets: [selected, better], now: now)
        XCTAssertTrue(paragraph.text.contains("Book B, +120"))
        XCTAssertTrue(paragraph.text.contains("45.45%"))
        XCTAssertTrue(paragraph.text.contains("+14.55 percentage points"))
        XCTAssertTrue(paragraph.text.contains("not a de-vig estimate"))
        let request = AskPLONNRequest(question: "Compare prices", intent: .comparePrices, evidence: [paragraph])
        XCTAssertNoThrow(try AskPLONNCore.validateProviderAnswer(AskPLONNAnswer(text: paragraph.text, evidence: [paragraph]), for: request))
        let decoded = try JSONDecoder().decode(AskPLONNPriceComparison.self, from: JSONEncoder().encode(result))
        XCTAssertEqual(decoded.quotes, result.quotes); XCTAssertEqual(decoded.edgePercentagePoints, result.edgePercentagePoints)
        let negative = AskPLONNResearch.priceComparison(market: validatedMarket(probability: 0.4), markets: [selected, better], now: now)
        XCTAssertEqual(try XCTUnwrap(negative.edgePercentagePoints), -5.45454545, accuracy: 0.000001)
        XCTAssertTrue(AskPLONNResearch.comparePrices(market: validatedMarket(probability: 0.4), markets: [selected, better], now: now).text.contains("-5.45 percentage points"))
    }
    func testFractionalAmericanPricesRemainConsistentWithComputedDecimalOdds() throws {
        for price in [105.5, -105.5] {
            var selected = validatedMarket(); selected.price = price
            let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
            let quote = try XCTUnwrap(result.bestQuote)
            XCTAssertEqual(quote.price, price)
            XCTAssertEqual(quote.decimal, price > 0 ? 2.055 : 1 + 100 / 105.5, accuracy: 0.000001)
            let evidence = AskPLONNResearch.comparePrices(market: selected, markets: [selected], now: now)
            XCTAssertTrue(evidence.text.contains(price > 0 ? "+105.5" : "-105.5"))
            XCTAssertFalse(evidence.text.contains("+106")); XCTAssertFalse(evidence.text.contains("-106"))
        }
    }
    func testPriceTiesAreStableAndSingleBookCoverageIsExplicit() {
        let selected = market(), b = market(book: "Book B", price: 120), c = market(book: "Book C", price: 120)
        let result = AskPLONNResearch.priceComparison(market: selected, markets: [c, selected, b], now: now)
        let reordered = AskPLONNResearch.priceComparison(market: selected, markets: [b, c, selected], now: now)
        XCTAssertEqual(result.bestBooks, ["Book B", "Book C"])
        XCTAssertEqual(result.quotes, reordered.quotes)
        XCTAssertEqual(result.bestQuote, reordered.bestQuote)
        let single = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
        XCTAssertEqual(single.bestBooks, ["Book A"])
        XCTAssertEqual(single.availability, .available)
        XCTAssertTrue(AskPLONNResearch.comparePrices(market: selected, markets: [selected], now: now).text.contains("limited comparison coverage"))
    }
    func testEdgeNeverBorrowsAnotherRowModelAndRequiresSelectedModelProvenance() {
        let selected = market(), other = validatedMarket()
        let unvalidated = AskPLONNResearch.priceComparison(market: selected, markets: [other, market(book: "Book B", price: 120)], now: now)
        XCTAssertNil(unvalidated.edgePercentagePoints); XCTAssertNil(unvalidated.modelProbability)
        XCTAssertFalse(AskPLONNResearch.comparePrices(market: selected, markets: [other], now: now).text.contains("99%"))
        var invalid: [Market] = []
        for probability in [Double.nan, .infinity, -0.1, 0, 1, 1.1] { invalid.append(validatedMarket(probability: probability)) }
        var missing = validatedMarket(); missing.p = nil; invalid.append(missing)
        missing = validatedMarket(); missing.modelValidated = nil; invalid.append(missing)
        missing = validatedMarket(); missing.modelValidated = false; invalid.append(missing)
        missing = validatedMarket(); missing.modelVersion = " \n"; invalid.append(missing)
        missing = validatedMarket(); missing.modelVersion = nil; invalid.append(missing)
        missing = validatedMarket(); missing.modelUpdated = nil; invalid.append(missing)
        missing = validatedMarket(); missing.modelUpdated = "invalid"; invalid.append(missing)
        invalid.append(validatedMarket(modelAge: 120.001)); invalid.append(validatedMarket(modelAge: -1))
        for model in invalid {
            let result = AskPLONNResearch.priceComparison(market: model, markets: [model, market(book: "Book B", price: 120)], now: now)
            XCTAssertNotNil(result.bestQuote)
            XCTAssertNil(result.edgePercentagePoints)
            XCTAssertFalse(result.edgeUnavailableReason.isEmpty)
        }
    }
    func testQuoteAndModelFreshnessBoundariesAndExpiration() {
        var selected = validatedMarket(modelAge: 120)
        selected.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-120))
        let boundary = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
        XCTAssertNotNil(boundary.bestQuote); XCTAssertNotNil(boundary.edgePercentagePoints)
        XCTAssertEqual(boundary.expiresAt, now)
        let expired = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now.addingTimeInterval(0.001))
        XCTAssertNil(expired.bestQuote); XCTAssertNil(expired.edgePercentagePoints)
        selected = validatedMarket(modelAge: 110)
        selected.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-10))
        XCTAssertEqual(AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now).expiresAt, now.addingTimeInterval(10))
    }
    func testIndependentlyStampedBundledPricesRankWithoutBorrowingParentTime() {
        var selected = market()
        selected.quotes = [Quote(book: "Stamped", price: 120, updated: ISO8601DateFormatter().string(from: now.addingTimeInterval(-60))),
                           Quote(book: "Legacy", price: 500),
                           Quote(book: "Stale bundled", price: 600, updated: ISO8601DateFormatter().string(from: now.addingTimeInterval(-121))),
                           Quote(book: "Future bundled", price: 700, updated: ISO8601DateFormatter().string(from: now.addingTimeInterval(1))),
                           Quote(book: "Broken bundled", price: 800, updated: "invalid")]
        let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
        XCTAssertEqual(result.bestBooks, ["Stamped"])
        XCTAssertEqual(result.quotes.count, 2)
        XCTAssertEqual(result.expiresAt, now.addingTimeInterval(60))
        XCTAssertTrue(result.excludedReasons.contains { $0.contains("Legacy: bundled quote timestamp unavailable") })
        selected.updated = nil
        let independent = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
        XCTAssertEqual(independent.bestBooks, ["Stamped"]); XCTAssertEqual(independent.quotes.count, 1)
        selected.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-150))
        XCTAssertEqual(AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now).bestBooks, ["Stamped"])
    }
    func testDuplicateBooksAndCrossSelectionIdentitiesAreQuarantined() {
        var selected = validatedMarket()
        selected.quotes = [Quote(book: "Book B", price: 500, updated: selected.updated)]
        let b = market(book: "Book B", price: 120)
        let conflict = AskPLONNResearch.priceComparison(market: selected, markets: [selected, b], now: now)
        XCTAssertEqual(conflict.bestBooks, ["Book A"])
        XCTAssertTrue(conflict.excludedReasons.contains { $0.contains("Book B: conflicting snapshots") })
        selected.quotes = []
        var reused = selected; reused.event = "other event"
        let identity = AskPLONNResearch.priceComparison(market: selected, markets: [selected, reused, b], now: now)
        XCTAssertEqual(identity.bestBooks, ["Book B"])
        XCTAssertNil(identity.edgePercentagePoints)
        XCTAssertTrue(identity.edgeUnavailableReason.contains("conflicting identity"))
        var changed = b; changed.id = "another snapshot"; changed.price = 130
        XCTAssertEqual(AskPLONNResearch.priceComparison(market: selected, markets: [selected, b, changed], now: now).bestBooks, ["Book A"])
    }
    func testEdgeRequiresKnownOpenEventAndWithholdsUnsupportedSettlement() {
        for state in ["post", "final", "completed", "closed", "canceled", "postponed", "suspended"] {
            let selected = validatedMarket(state: state)
            let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected, market(book: "Book B", price: 120)], now: now)
            XCTAssertNil(result.bestQuote); XCTAssertNil(result.edgePercentagePoints)
        }
        var selected = validatedMarket(state: nil); selected.commence = ISO8601DateFormatter().string(from: now.addingTimeInterval(-60))
        let unknown = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
        XCTAssertNotNil(unknown.bestQuote); XCTAssertNil(unknown.edgePercentagePoints)
        for state in ["pre", "in", "scheduled"] {
            selected = validatedMarket(state: state)
            XCTAssertNotNil(AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now).edgePercentagePoints)
        }
        for line in [20.0, 20.25, 20.75] {
            selected = validatedMarket(); selected.line = line
            let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
            XCTAssertNotNil(result.bestQuote); XCTAssertNil(result.edgePercentagePoints)
            XCTAssertFalse(result.edgeUnavailableReason.isEmpty)
        }
        selected = validatedMarket(); selected.kind = "team"; selected.sport = "NFL"; selected.market = "h2h"
        selected.name = "Home"; selected.side = "Home"; selected.line = nil
        let tie = AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now)
        XCTAssertNotNil(tie.bestQuote); XCTAssertNil(tie.edgePercentagePoints)
        XCTAssertTrue(tie.edgeUnavailableReason.contains("tie"))
        selected.market = "h2h_3_way"
        XCTAssertNotNil(AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now).edgePercentagePoints)
        selected = validatedMarket(); selected.market = "unknown_stat"
        XCTAssertNil(AskPLONNResearch.priceComparison(market: selected, markets: [selected], now: now).edgePercentagePoints)
    }
    func testPriceEvidenceIsBoundedAndPreservesBestPriceAndEdgeFacts() {
        var selected = validatedMarket(); selected.modelVersion = String(repeating: "v", count: 200)
        let rows = (0..<20).map { index in market(book: String(format: "Book %02d ", index) + String(repeating: "b", count: 85), price: 120) }
        let paragraph = AskPLONNResearch.comparePrices(market: selected, markets: rows, now: now)
        XCTAssertLessThanOrEqual(paragraph.text.count, 2000)
        XCTAssertTrue(paragraph.text.contains("+120"))
        XCTAssertTrue(paragraph.text.contains("+14.55 percentage points"))
        XCTAssertNoThrow(try AskPLONNCore.boundedRequest(AskPLONNRequest(question: "Compare prices", intent: .comparePrices, evidence: [paragraph])))
        let extreme = (0..<8).map { index in market(book: "Finite Book \(index)", price: Double.greatestFiniteMagnitude) }
        let bounded = AskPLONNResearch.comparePrices(market: selected, markets: extreme, now: now)
        XCTAssertLessThanOrEqual(bounded.text.count, 2000)
        XCTAssertTrue(bounded.text.contains("+60.00 percentage points"))
    }
    func testClosedExactEventObservationCannotBeHiddenByRowOrBookQuarantine() {
        let selected = validatedMarket()
        for sameID in [false, true] {
            for state in ["final", "closed", "postponed"] {
                var closed = selected; closed.id = sameID ? selected.id : "other-status-row"; closed.eventState = state
                let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected, closed, market(book: "Book B", price: 120)], now: now)
                XCTAssertNil(result.bestQuote); XCTAssertNil(result.edgePercentagePoints)
                XCTAssertEqual(result.availability, .unavailable)
                XCTAssertTrue(result.unavailableReason?.contains("closed or interrupted") == true)
            }
        }
    }
    func testClosedGameStatusBlocksAllOutcomesWithoutCrossEventOrScopeLeakage() {
        let selected = validatedMarket()
        var under = selected; under.id = "under-status"; under.side = "Under"; under.eventState = "final"
        var otherStat = under; otherStat.id = "stat-status"; otherStat.market = "player_assists"; otherStat.line = 5.5
        var otherPlayer = under; otherPlayer.id = "player-status"; otherPlayer.name = "Another Player"
        var team = under; team.id = "team-status"; team.kind = "team"; team.name = "Home"; team.side = "Home"; team.market = "h2h"; team.line = nil
        for status in [under, otherStat, otherPlayer, team] {
            let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected, status], now: now)
            XCTAssertNil(result.bestQuote); XCTAssertNil(result.edgePercentagePoints)
            XCTAssertEqual(result.availability, .unavailable)
            XCTAssertTrue(result.unavailableReason?.contains("closed or interrupted event") == true)
        }
        var otherEvent = under; otherEvent.event = "another-event"
        var otherStart = under; otherStart.commence = "2026-10-10T20:00:00Z"
        var otherSport = under; otherSport.sport = "NFL"
        var partial = under; partial.label = "First quarter Under 20.5"
        for status in [otherEvent, otherStart, otherSport, partial] {
            let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected, status], now: now)
            XCTAssertEqual(result.bestBooks, ["Book A"])
            XCTAssertNotNil(result.edgePercentagePoints)
        }
    }
    func testFreshBookCannotSurviveContradictoryStaleFutureInvalidOrMissingPrimaryObservation() {
        let selected = market()
        let fresh = market(book: "Book B", price: 120)
        var contradictory: [Market] = [market(book: "Book B", price: 500, age: 121), market(book: "Book B", price: 600, age: -1), market(book: "Book B", price: 0)]
        var missing = fresh; missing.updated = nil; contradictory.append(missing)
        for (index, row) in contradictory.enumerated() {
            var other = row; other.id = "contradictory-\(index)"
            let result = AskPLONNResearch.priceComparison(market: selected, markets: [selected, fresh, other], now: now)
            XCTAssertEqual(result.bestBooks, ["Book A"])
            XCTAssertTrue(result.excludedReasons.contains { $0.contains("Book B: conflicting snapshots") })
        }
        var bundled = selected
        bundled.quotes = [Quote(book: "Book B", price: 500, updated: ISO8601DateFormatter().string(from: now.addingTimeInterval(-121)))]
        XCTAssertEqual(AskPLONNResearch.priceComparison(market: bundled, markets: [bundled, fresh], now: now).bestBooks, ["Book A"])
    }
    func testPartialGameMarketsCannotMixWithFullGamePricesOrEdge() {
        let selected = validatedMarket()
        for label in ["First quarter Over 20.5", "2nd period Over 20.5", "First half Over 20.5", "First 5 innings Over 20.5"] {
            var partial = selected; partial.label = label
            let result = AskPLONNResearch.priceComparison(market: partial, markets: [partial, market(book: "Book B", price: 120)], now: now)
            XCTAssertNil(result.bestQuote); XCTAssertNil(result.edgePercentagePoints)
            XCTAssertTrue(result.unavailableReason?.contains("settlement scope") == true)
            partial.id = "partial-row"; partial.book = "Partial Book"; partial.price = 500
            let full = AskPLONNResearch.priceComparison(market: selected, markets: [selected, partial], now: now)
            XCTAssertEqual(full.bestBooks, ["Book A"])
        }
        for key in ["player_points_q1", "player_points_1st_half", "totals_h1"] {
            var partial = selected; partial.market = key
            let result = AskPLONNResearch.priceComparison(market: partial, markets: [partial], now: now)
            XCTAssertNil(result.bestQuote); XCTAssertNil(result.edgePercentagePoints)
            XCTAssertTrue(result.unavailableReason?.contains("settlement scope") == true)
        }
    }
    private func recipe() -> LabRecipe {
        LabRecipe(id: UUID(), name: "Configured recipe", player: "Fixture Player", quarterback: "", opponent: "Opponent", features: ["recent", "workload"], created: now.addingTimeInterval(-3600), line: 20.5, side: "Over", sport: "NBA", stat: "Points", targetKind: "Player", advanced: false, training: nil)
    }
    func testRecipeDescribesConfigurationAndNeverImpliesTrainingOrProbability() {
        let result = AskPLONNResearch.modelSetup(recipe: recipe(), now: now)
        XCTAssertEqual(result.availability, .available); XCTAssertNil(result.sampleSize)
        XCTAssertTrue(result.text.contains("configuration only")); XCTAssertTrue(result.text.contains("does not prove training"))
        XCTAssertFalse(result.text.contains("99%")); XCTAssertFalse(result.text.contains("200 epochs"))
    }
    func testSparseLegacyRecipeIsUnavailableWithoutInventingDefaults() {
        var legacy = LabRecipe(id: UUID(), name: "Legacy setup", player: "Fixture Player", quarterback: "", opponent: "Opponent",
            features: ["recent", "workload"], created: now.addingTimeInterval(-3600), line: 20.5, side: "Over",
            sport: nil, stat: nil, targetKind: nil, advanced: nil, training: nil)
        legacy.modelType = nil
        let evidence = AskPLONNResearch.modelSetup(recipe: legacy, now: now)
        XCTAssertEqual(evidence.availability, .unavailable)
        XCTAssertNil(evidence.sampleSize)
        XCTAssertTrue(evidence.text.contains("missing or invalid configuration"))
        for invented in ["NBA", "NFL", "NHL", "MLB", "Rec Yds", "Points", "Logistic regression", "50%"] {
            XCTAssertFalse(evidence.text.contains(invented), "Sparse metadata must not invent \(invented)")
        }
        XCTAssertFalse(evidence.text.contains("result reports"))
    }

    private func snapshot(_ recipe: LabRecipe, owner: UUID, probability: Double = 0.6, line: Double? = 20.5, brier: Double? = 0.2, mode: String = "fixed_threshold_retrospective") -> AskPLONNModelResultSnapshot {
        AskPLONNModelResultSnapshot(id: UUID(), ownerID: owner, recipeID: recipe.id, recipeFingerprint: recipe.configurationFingerprint!, sport: "NBA", player: "Fixture Player", stat: "Points", targetKind: "Player", side: "Over", threshold: line, historicalEventID: recipe.historicalEventID, probability: probability, validationSample: 20, brier: brier, mode: mode, generatedAt: now.addingTimeInterval(-10))
    }
    func testModelSnapshotRequiresOwnerRecipeThresholdAndValidNumbers() {
        let recipe = recipe(), owner = UUID()
        let result = AskPLONNResearch.modelResult(snapshot: snapshot(recipe, owner: owner), ownerID: owner, recipe: recipe, now: now)
        XCTAssertEqual(result.availability, .available); XCTAssertTrue(result.text.contains("60.0%"))
        XCTAssertTrue(result.text.contains("Retrospective")); XCTAssertTrue(result.text.contains("does not establish calibration"))
        XCTAssertEqual(AskPLONNResearch.modelResult(snapshot: snapshot(recipe, owner: owner), ownerID: UUID(), recipe: recipe, now: now).availability, .unavailable)
        for bad in [snapshot(recipe, owner: owner, probability: .nan), snapshot(recipe, owner: owner, probability: 1.1), snapshot(recipe, owner: owner, line: 21.5), snapshot(recipe, owner: owner, brier: .infinity), snapshot(recipe, owner: owner, mode: "live_prediction")] {
            XCTAssertEqual(AskPLONNResearch.modelResult(snapshot: bad, ownerID: owner, recipe: recipe, now: now).availability, .unavailable)
        }
        let changedRecipe = LabRecipe(id: recipe.id, name: recipe.name, player: recipe.player, quarterback: recipe.quarterback, opponent: recipe.opponent, features: ["season"], created: recipe.created, line: recipe.line, side: recipe.side, sport: recipe.sport, stat: recipe.stat, targetKind: recipe.targetKind, advanced: recipe.advanced, training: recipe.training)
        XCTAssertEqual(AskPLONNResearch.modelResult(snapshot: snapshot(recipe, owner: owner), ownerID: owner, recipe: changedRecipe, now: now).availability, .unavailable)
    }
    func testUnvalidatedSnapshotCannotClaimCalibration() {
        let recipe = recipe(), owner = UUID()
        let result = AskPLONNResearch.modelResult(snapshot: snapshot(recipe, owner: owner, brier: nil, mode: "unvalidated"), ownerID: owner, recipe: recipe, now: now)
        XCTAssertEqual(result.availability, .available)
        XCTAssertTrue(result.text.contains("Unvalidated model output")); XCTAssertTrue(result.text.contains("calibration is unverified"))
    }
}
