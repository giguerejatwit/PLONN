import XCTest
@testable import PLONNCore

final class SimulationForecastTests: XCTestCase {
    let now = Date(timeIntervalSince1970: 1_790_000_000)
    func history(_ sport: InterestSport, _ id: String, scored: Int, allowed: Int, count: Int = 10) -> PlayerSimulation.History {
        let team = InterestTeam(sport: sport, providerID: id, name: id == "1" ? "Away Team" : "Home Team", abbreviation: id)
        let boxes = (1...count).map { i in PlayerSimulation.Box(game: .init(id: "\(id)-\(i)", date: now.addingTimeInterval(-Double(i) * 86400), season: 2026, scored: scored, allowed: allowed, home: false), lines: []) }
        return .init(team: team, roster: [], boxes: boxes, partial: false)
    }
    func testOpponentDefenseAndSmallSampleShrinkageAcrossSports() {
        for sport in InterestSport.allCases {
            let prior = Int(SimulationForecast.neutral(sport).rounded()), gap = sport == .nba ? 20 : sport == .nfl ? 10 : 2
            let away = history(sport, "1", scored: prior, allowed: prior)
            let goodDefense = history(sport, "2", scored: prior, allowed: prior - gap)
            let poorDefense = history(sport, "2", scored: prior, allowed: prior + gap)
            let good = SimulationForecast.scoringTargets([away, goodDefense], asOf: now)
            let poor = SimulationForecast.scoringTargets([away, poorDefense], asOf: now)
            XCTAssertGreaterThan(poor.scores[0], good.scores[0], sport.rawValue)
            let tiny = history(sport, "2", scored: prior, allowed: prior + gap, count: 1)
            let one = SimulationForecast.scoringTargets([away, tiny], asOf: now)
            XCTAssertLessThan(one.scores[0], poor.scores[0])
            XCTAssertGreaterThan(SimulationForecast.weight(date: now.addingTimeInterval(-86400), asOf: now, sport: sport), SimulationForecast.weight(date: now.addingTimeInterval(-180 * 86400), asOf: now, sport: sport))
        }
    }
    func testBetaRatesDoNotTreatTinyPerfectSamplesAsCertainty() {
        var random = SportPlaySimulation.Random(state: 54)
        let tiny = (0..<10000).map { _ in random.beta(made: 2, attempts: 2, prior: 0.5, strength: 20) }
        let large = (0..<10000).map { _ in random.beta(made: 1000, attempts: 1000, prior: 0.5, strength: 20) }
        let sameLarge = (0..<10000).map { _ in random.beta(made: 500, attempts: 1000, prior: 0.5, strength: 20) }
        func mean(_ x: [Double]) -> Double { x.reduce(0, +) / Double(x.count) }
        func variance(_ x: [Double]) -> Double { let m = mean(x); return x.reduce(0) { $0 + pow($1-m,2) } / Double(x.count) }
        XCTAssertEqual(mean(tiny), 12.0 / 22, accuracy: 0.01)
        XCTAssertGreaterThan(mean(large), 0.98)
        XCTAssertGreaterThan(variance(tiny), variance(sameLarge) * 10)
        XCTAssertTrue((tiny + large).allSatisfy { $0 > 0 && $0 < 1 })
    }
    func event(_ sport: InterestSport = .nba) -> PlayerMatchupEvent {
        .init(game: .init(league: sport.rawValue, eventID: "401999999", home: "Home Team", away: "Away Team", startsAt: ISO8601DateFormatter().string(from: now.addingTimeInterval(3600))), homeID: "2", awayID: "1", seasonType: 2)
    }
    func quote(_ home: Bool, sport: InterestSport = .nba, book: String = "Book") -> Market {
        Market(id: home ? "home" : "away", event: "odds-event", sport: sport.rawValue, kind: "team", name: home ? "Home Team" : "Away Team", opponent: home ? "Away Team" : "Home Team", label: "Moneyline", aliases: "", p: nil, price: home ? -700 : 500, market: "h2h", book: book, quotes: [], updated: ISO8601DateFormatter().string(from: now), mode: "live", eventLabel: "Away Team at Home Team", eventState: "pre", commence: event(sport).game.startsAt, side: nil)
    }
    func testSameBookMarketPairNormalizesMarginAcrossAllSports() throws {
        for sport in InterestSport.allCases {
            let input = try XCTUnwrap(SimulationForecast.market(event: event(sport), markets: [quote(false,sport:sport),quote(true,sport:sport)], now: now))
            XCTAssertEqual(try XCTUnwrap(input.homeProbability), (7.0/8) / (7.0/8+1.0/6), accuracy: 1e-9)
            XCTAssertEqual(input.teamIDs,["1","2"])
        }
    }
    func testAlternateBookQuotesCanPairWithoutMixingBestPrices() throws {
        var a = quote(false, book: "Book A"), h = quote(true, book: "Book B")
        a.quotes = [.init(book: "Shared", price: 490, updated: a.updated)]
        h.quotes = [.init(book: "Shared", price: -675, updated: h.updated)]
        let matched = try XCTUnwrap(SimulationForecast.market(event: event(), markets: [a,h], now: now))
        XCTAssertEqual(matched.book, "Shared")
        XCTAssertEqual(try XCTUnwrap(matched.homeProbability), (675.0/775) / (675.0/775+100.0/590), accuracy: 1e-9)
        h.quotes[0].updated = nil
        XCTAssertNil(SimulationForecast.market(event: event(), markets: [a,h], now: now))
    }
    func testCapturedAppSnapshotMatchesVerifiedGame() throws {
        guard let path = ProcessInfo.processInfo.environment["PLONN_RECORDED_SIMULATION_FIXTURE"] else { throw XCTSkip("Requires an actual app snapshot capture") }
        let root = URL(fileURLWithPath: path)
        let quotes = try JSONDecoder().decode([Market].self, from: Data(contentsOf: root.appendingPathComponent("quotes.json")))
        let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("event.json"))) as! [String:String]
        let event = PlayerMatchupEvent(game: .init(league: metadata["sport"]!, eventID: metadata["id"]!, home: metadata["home"]!, away: metadata["away"]!, startsAt: metadata["start"]!), homeID: metadata["home_id"]!, awayID: metadata["away_id"]!, seasonType: 2)
        let now = try XCTUnwrap(sportsDate(metadata["checked"]!))
        XCTAssertTrue(quotes.contains { $0.eventState == nil })
        let market = try XCTUnwrap(SimulationForecast.market(event: event, markets: quotes, now: now))
        XCTAssertEqual(market.teamIDs, [metadata["away_id"]!,metadata["home_id"]!])
        XCTAssertGreaterThanOrEqual(market.quoteIDs.count, 2)
        XCTAssertNotNil(market.homeProbability)
    }
    func testLegacyMissingStatusUsesVerifiedFutureScheduleOnly() throws {
        var a = quote(false), h = quote(true)
        a.eventState = nil; h.eventState = nil
        XCTAssertNotNil(SimulationForecast.market(event: event(), markets: [a,h], now: now))
        XCTAssertNil(SimulationForecast.market(event: event(), markets: [a,h], now: now.addingTimeInterval(3600)))
        h.eventState = "unknown"
        XCTAssertNil(SimulationForecast.market(event: event(), markets: [a,h], now: now))
    }
    func testBadMarketInputsCannotConditionForecast() {
        let a = quote(false), h = quote(true)
        XCTAssertNil(SimulationForecast.market(event: event(), markets: [h], now: now))
        XCTAssertNil(SimulationForecast.market(event: event(), markets: [a,quote(true,book:"Different book")], now: now))
        for variation in 0..<7 {
            var bad = h
            switch variation {
            case 0: bad.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-86401))
            case 1: bad.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(1))
            case 2: bad.eventState = "in"
            case 3: bad.eventLabel = "Away Team at Wrong Opponent"
            case 4: bad.commence = ISO8601DateFormatter().string(from: now.addingTimeInterval(86400))
            case 5: bad.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-301))
            default: bad.price = .nan
            }
            XCTAssertNil(SimulationForecast.market(event: event(), markets: [a,bad], now: now), "variant \(variation)")
        }
    }
    func testMarketChangesDistributionWithoutChangingAnyPath() throws {
        for sport in InterestSport.allCases {
            let prior = Int(SimulationForecast.neutral(sport).rounded())
            let histories = [history(sport,"1",scored:prior,allowed:prior),history(sport,"2",scored:prior,allowed:prior)]
            let scores = (0..<5000).map { i in MatchupSimulation.Score(away: max(0,prior + i % 11 - 5),home:max(0,prior + (i / 11) % 11 - 5)) }
            let m = try XCTUnwrap(SimulationForecast.market(event:event(sport),markets:[quote(false,sport:sport),quote(true,sport:sport)],now:now))
            let selected = SimulationForecast.select(scores:scores,histories:histories,now:now,seed:43,market:m,live:nil)
            XCTAssertEqual(selected.indices.count,scores.count)
            XCTAssertTrue(selected.indices.allSatisfy { scores.indices.contains($0) })
            XCTAssertGreaterThan(selected.indices.filter { scores[$0].home > scores[$0].away }.count, 3000, sport.rawValue)
            XCTAssertGreaterThanOrEqual(selected.evidence.effectivePaths, 749.99)
            let again = SimulationForecast.select(scores:scores,histories:histories,now:now,seed:43,market:m,live:nil)
            XCTAssertEqual(again.indices,selected.indices)
        }
    }
    func testPuckAndRunLinePricesAreNotMistakenForExpectedMargins() throws {
        for sport in [InterestSport.nhl, .mlb] {
            let prior = SimulationForecast.neutral(sport)
            var a = quote(false,sport:sport), h = quote(true,sport:sport)
            a.market = "spreads"; h.market = "spreads"; a.label = "Spread"; h.label = "Spread"
            a.line = 1.5; h.line = -1.5; a.price = -180; h.price = 150
            let input = try XCTUnwrap(SimulationForecast.market(event:event(sport),markets:[a,h],now:now))
            let target = SimulationForecast.marketScores(input,baseline:[prior,prior],marginSD:2.5,totalSD:3)
            XCTAssertLessThan(target[1]-target[0],1.5)
            XCTAssertGreaterThan(target[1]-target[0],0)
            // With a moneyline, that market supplies low-scoring sport direction;
            // merely posting a negative puck/run line cannot override it.
            var mlA = quote(false,sport:sport), mlH = quote(true,sport:sport)
            mlA.id = "ml-away"; mlH.id = "ml-home"; mlA.price = -150; mlH.price = 130
            let withMoney = try XCTUnwrap(SimulationForecast.market(event:event(sport),markets:[a,h,mlA,mlH],now:now))
            let priced = SimulationForecast.marketScores(withMoney,baseline:[prior,prior],marginSD:2.5,totalSD:3)
            XCTAssertGreaterThan(priced[0],priced[1])
        }
    }
    func testUnsupportedTailCorrectionIsLimited() {
        let h = [history(.nba,"1",scored:170,allowed:170),history(.nba,"2",scored:170,allowed:170)]
        let s = (0..<1000).map { MatchupSimulation.Score(away:90+$0%20,home:90+($0/20)%20) }
        let chosen = SimulationForecast.select(scores:s,histories:h,now:now,seed:1,market:nil,live:nil)
        XCTAssertTrue(chosen.evidence.limited)
        XCTAssertGreaterThanOrEqual(chosen.evidence.effectivePaths,149.99)
    }
}
