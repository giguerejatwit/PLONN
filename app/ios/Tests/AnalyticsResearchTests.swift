import XCTest
@testable import PLONNCore

final class AnalyticsResearchTests: XCTestCase {
 func testTrendFitsCompletedDisplayedGamesAndLeavesMissingGaps() {
  let date = Date(timeIntervalSince1970: 1000)
  let rows = [2.0, 4.0, 6.0, 8.0].enumerated().map { AnalyticsObservation(id: "g\($0.offset)", date: date.addingTimeInterval(Double($0.offset)), opponent: "Other", value: $0.element) }
  let fit = AnalyticsResearch.trendLine(rows)
  XCTAssertEqual(fit.map(\.index), [0, 3])
  XCTAssertEqual(fit[0].value, 2, accuracy: 0.000001)
  XCTAssertEqual(fit[1].value, 8, accuracy: 0.000001)
  var missing = rows; missing[1].value = nil
  let gapFit = AnalyticsResearch.trendLine(missing)
  XCTAssertEqual(gapFit.map(\.index), fit.map(\.index))
  for (actual, expected) in zip(gapFit, fit) { XCTAssertEqual(actual.value, expected.value, accuracy: 0.000001) }
  XCTAssertTrue(AnalyticsResearch.trendLine(Array(rows.prefix(2))).isEmpty)
  let flat = rows.map { row in var row = row; row.value = 0; return row }
  XCTAssertEqual(AnalyticsResearch.trendLine(flat).map(\.value), [0, 0])
  let descending = rows.map { row in var row = row; row.value = -(row.value!); return row }
  XCTAssertEqual(AnalyticsResearch.trendLine(descending).map(\.value), [-2, -8])
  var invalid = rows; invalid[1].value = .nan; invalid[2].value = .infinity
  XCTAssertTrue(AnalyticsResearch.trendLine(invalid).isEmpty)
 }

 private let now = ISO8601DateFormatter().date(from: "2026-10-19T12:00:00Z")!
 private func candidate(sport: String = "NBA", player: Bool = false) -> AnalyticsFeedCandidate {
  let key = player ? "player_points" : "spreads"
  let event = AnalyticsEvent(sport: sport, providerID: "123", home: "Fixture Home", away: "Fixture Away", start: now.addingTimeInterval(3600), homeID: "1", awayID: "2")
  let market = Market(id: "candidate", event: "", sport: sport, kind: player ? "prop" : "team", name: player ? "Fixture Player" : "Fixture Home", opponent: "Fixture Away", label: "", aliases: "", p: nil, price: 0, market: key, book: "", quotes: [], updated: nil, mode: "analytics")
  let path = ["NBA": "basketball/nba", "NFL": "football/nfl", "MLB": "baseball/mlb", "NHL": "hockey/nhl"][sport]!
  let proof = AnalyticsFeedVerifiedSubject(sport: sport, subject: player ? .player : .team, name: market.name, providerID: player ? "77" : "1", teamName: player ? "Fixture Home" : nil, teamID: player ? "1" : nil, marketKeys: [key], event: event, sourceURL: "https://site.api.espn.com/apis/site/v2/sports/" + path + (player ? "/teams/1/roster" : "/scoreboard"), verifiedAt: now)
  return AnalyticsFeedCandidate(id: market.id, subject: player ? .player : .team, market: market, quotes: [], verifiedSubject: proof)
 }
 private func rows(_ count: Int = 25) -> [AnalyticsObservation] {
  (0..<count).map { i -> AnalyticsObservation in
   let date = now.addingTimeInterval(-Double(i + 1) * 86400)
   let opponent = i.isMultiple(of: 2) ? "Fixture Away" : "Other Opponent"
   return AnalyticsObservation(id: "game-\(i)", date: date, opponent: opponent, value: Double(i), scored: Double(i), allowed: 3, season: i < 12 ? 2026 : 2025)
  }
 }
 private func history(_ rows: [AnalyticsObservation], candidate: AnalyticsFeedCandidate) -> AnalyticsHistoryResult {
  let value = AnalyticsHistoryEngine.statistics(market: candidate.market, observations: rows, now: now)
  return AnalyticsHistoryResult(summary: nil, rows: [], observations: rows, source: "ESPN", warnings: [], availability: "Recorded", statistics: value.statistics, statsRows: value.statsRows, verifiedPlayerID: candidate.subject == .player ? "77" : nil)
 }
 private func replacing(_ original: AnalyticsFeedCandidate, quotes: [Market]? = nil, market: Market? = nil, withoutProof: Bool = false) -> AnalyticsFeedCandidate {
  AnalyticsFeedCandidate(id: original.id, subject: original.subject, market: market ?? original.market, quotes: quotes ?? original.quotes, verifiedSubject: withoutProof ? nil : original.verifiedSubject)
 }
 private func sample(_ candidate: AnalyticsFeedCandidate, _ rows: [AnalyticsObservation], window: AnalyticsResearchWindow = .last10, quote: Market? = nil) -> AnalyticsResearchSample {
  AnalyticsResearch.sample(candidate: candidate, history: history(rows, candidate: candidate), window: window, quote: quote, now: now)
 }
 private func quote(_ candidate: AnalyticsFeedCandidate, line: Double = -2, side: String = "home", id: String = "quote") -> Market {
  var quote = candidate.market
  quote.id = id; quote.event = "odds-event"; quote.mode = "live"; quote.line = line; quote.side = side; quote.price = -110
  quote.book = "Fixture Book"; quote.updated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-30))
  quote.commence = ISO8601DateFormatter().string(from: candidate.verifiedSubject!.event!.start)
  quote.eventLabel = "Fixture Away at Fixture Home"
  return quote
 }
 func testWindowsAcrossAllSportsRetainMissingAndDoNotBackfill() {
  for sport in ["NFL", "NBA", "MLB", "NHL"] {
   let candidate = candidate(sport: sport)
   var input = rows(); input[0].status = .missing
   for (window, count) in [(AnalyticsResearchWindow.last5, 5), (.last10, 10), (.last20, 20)] {
    let result = sample(candidate, input, window: window)
    XCTAssertEqual(result.statistics.statistics?.n, count, sport)
    XCTAssertEqual(result.statistics.statistics?.valid, count - 1, sport)
    XCTAssertEqual(result.statistics.statistics?.missing, 1, sport)
    XCTAssertNil(result.statistics.statsRows.first?.value)
   }
   XCTAssertEqual(AnalyticsHistoryEngine.statistics(market: candidate.market, observations: input, limit: 20, now: now).statistics?.n, 10)
  }
 }
 func testSeasonUsesProviderSeasonAndDeduplicatesBeforeFiltering() {
  let candidate = candidate()
  var input = rows()
  var conflict = input[0]; conflict.season = 2025; input.append(conflict)
  let result = sample(candidate, input, window: .season)
  XCTAssertEqual(result.statistics.statistics?.n, 11)
  XCTAssertEqual(result.label, "2026 season · Available games")
  XCTAssertFalse(result.statistics.statsRows.contains { $0.id == "game-0" })
  input = rows().map { row in var copy = row; copy.season = nil; return copy }
  let unavailable = sample(candidate, input, window: .season)
  XCTAssertNil(unavailable.statistics.statistics)
  XCTAssertTrue(unavailable.note?.contains("Season information") == true)
 }
 func testOpponentFilterRequiresVerifiedOpponentAndDoesNotFuzzyMatch() {
  var candidate = candidate()
  var input = rows(10)
  input[0] = AnalyticsObservation(id: input[0].id, date: input[0].date, opponent: "Fixture", scored: 0, allowed: 3)
  XCTAssertEqual(sample(candidate, input, window: .opponent).statistics.statistics?.n, 4)
  candidate = replacing(candidate, withoutProof: true)
  XCTAssertNil(sample(candidate, input, window: .opponent).statistics.statistics)
 }
 func testChangedPlayerIdentityCannotShowAnotherPlayersHistory() {
  let candidate = candidate(player: true)
  var history = history(rows(), candidate: candidate); history.verifiedPlayerID = "wrong"
  let result = AnalyticsResearch.sample(candidate: candidate, history: history, window: .last20, quote: nil, now: now)
  XCTAssertTrue(result.statistics.statsRows.isEmpty)
 }
 func testFutureTargetAndConflictingObservationsCannotLeakIntoChart() {
  let candidate = candidate()
  var input = rows(4)
  input.append(AnalyticsObservation(id: "future", date: now, opponent: "Away", scored: 999, allowed: 0))
  input.append(AnalyticsObservation(id: "123", date: now.addingTimeInterval(-100), opponent: "Away", scored: 999, allowed: 0))
  input.append(input[0])
  XCTAssertEqual(sample(candidate, input).statistics.statistics?.n, 4)
 }
 func testSpreadChartUsesOppositeSignedThresholdAndKeepsPushOutOfFrequency() throws {
  var candidate = candidate()
  let quote = quote(candidate)
  candidate = replacing(candidate, quotes: [quote])
  let result = sample(candidate, rows(10), quote: quote)
  let comparison = try XCTUnwrap(result.comparison?.summary)
  XCTAssertEqual(AnalyticsResearch.chartLine(quote), 2)
  XCTAssertEqual(comparison.hits, 4)
  XCTAssertEqual(comparison.misses, 5)
  XCTAssertEqual(comparison.pushes, 1)
  XCTAssertEqual(try XCTUnwrap(comparison.hitFrequency), 4.0 / 9, accuracy: 0.00001)
  var changed = quote; changed.line = 100
  XCTAssertNil(sample(candidate, rows(), quote: changed).comparison, "An unverified edited contract must be rejected")
 }
 func testExactQuoteSwitchChangesUnderOverComparisonWithoutChangingStats() throws {
  var candidate = candidate(player: true)
  let over = quote(candidate, line: 3, side: "over", id: "over")
  let under = quote(candidate, line: 3, side: "under", id: "under")
  candidate = replacing(candidate, quotes: [over, under])
  let a = sample(candidate, rows(10), quote: over), b = sample(candidate, rows(10), quote: under)
  XCTAssertEqual(a.statistics.statistics?.mean, b.statistics.statistics?.mean)
  XCTAssertEqual(try XCTUnwrap(a.comparison?.summary).hits, 6)
  XCTAssertEqual(try XCTUnwrap(b.comparison?.summary).hits, 3)
  XCTAssertEqual(AnalyticsResearch.chartLine(under), 3)
 }
 func testNoQuoteHasStatsWithoutProbabilityOrComparison() {
  let result = sample(candidate(), rows())
  XCTAssertEqual(result.statistics.statistics?.valid, 10)
  XCTAssertNil(result.comparison)
 }
 func testLargeAndUntrustedInputsDoNotFabricateSample() {
  XCTAssertTrue(sample(candidate(), rows(2001)).statistics.statsRows.isEmpty)
  var input = rows(5); input[0].scored = .infinity; input[1].status = .void
  let result = sample(candidate(), input)
  XCTAssertEqual(result.statistics.statistics?.valid, 3)
  XCTAssertEqual(result.statistics.statistics?.missing, 1)
  XCTAssertEqual(result.statistics.statistics?.voids, 1)
 }
 func testIdentityGroupsStatsWithoutMergingPlayersOrSports() {
  let player = candidate(player: true)
  var market = player.market; market.market = "player_rebounds"
  let anotherStat = replacing(player, market: market)
  XCTAssertEqual(AnalyticsResearch.subjectID(player), AnalyticsResearch.subjectID(anotherStat))
  XCTAssertNotEqual(AnalyticsResearch.subjectID(player), AnalyticsResearch.subjectID(candidate()))
  XCTAssertNotEqual(AnalyticsResearch.subjectID(candidate()), AnalyticsResearch.subjectID(candidate(sport: "NHL")))
 }
}
