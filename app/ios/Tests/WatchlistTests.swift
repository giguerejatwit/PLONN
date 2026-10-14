import XCTest
@testable import PLONNCore
final class WatchlistTests: XCTestCase {
 func market() -> Market {
  Market(id:"judge",event:"nyy-bos",sport:"MLB",kind:"prop",name:"Aaron Judge",opponent:"Boston",label:"Over 1.5 total bases",aliases:"NYY",p:nil,price:105,market:"total bases",book:"Book",quotes:[Quote(book:"Book",price:105)],updated:"2026-09-25",mode:"live")
 }
 func testRestoreRemoveAndClearWithIsolatedPreferences() throws {
  let name = "PLONNTests." + UUID().uuidString
  let defaults = try XCTUnwrap(UserDefaults(suiteName:name))
  defer { defaults.removePersistentDomain(forName:name) }
  let storage = WatchlistStorage<Market>(defaults:defaults)
  XCTAssertTrue(try storage.load().isEmpty)
  try storage.save([market()])
  let restored = WatchlistStorage<Market>(defaults:try XCTUnwrap(UserDefaults(suiteName:name)))
  XCTAssertEqual(try restored.load(), [market()])
  XCTAssertNil(try restored.load().first?.p)
  try restored.save([])
  XCTAssertTrue(try storage.load().isEmpty)
  try storage.save([market()]); storage.clear()
  XCTAssertTrue(try restored.load().isEmpty)
 }
 func testCorruptionAndFailedSavePreserveStoredData() throws {
  let name = "PLONNTests." + UUID().uuidString
  let defaults = try XCTUnwrap(UserDefaults(suiteName:name))
  defer { defaults.removePersistentDomain(forName:name) }
  let storage = WatchlistStorage<Market>(defaults:defaults)
  try storage.save([market()])
  var invalid = market(); invalid.price = .nan
  XCTAssertThrowsError(try storage.save([invalid]))
  XCTAssertEqual(try storage.load(), [market()])
  defaults.set(Data("invalid".utf8), forKey:"watchlist")
  XCTAssertThrowsError(try storage.load())
  storage.clear()
  XCTAssertTrue(try storage.load().isEmpty)
 }
 func testLegacyQuotesDecodeWithoutInventingPerBookTimestamp() throws {
  let encoded = try JSONEncoder().encode(market())
  let restored = try JSONDecoder().decode(Market.self,from:encoded)
  XCTAssertNil(restored.quotes.first?.updated)
  XCTAssertNotNil(restored.updated,"The primary quote timestamp cannot supply the bundled quote's provenance.")
 }
 func testDistinctBookTimestampsSurviveSaveAndReopen() throws {
  let name = "PLONNTests." + UUID().uuidString
  let defaults = try XCTUnwrap(UserDefaults(suiteName:name))
  defer { defaults.removePersistentDomain(forName:name) }
  var value = market()
  value.quotes = [Quote(book:"Book A",price:105,updated:"2026-10-10T22:00:00Z"),Quote(book:"Book B",price:110,updated:"2026-10-10T21:59:30Z"),Quote(book:"Legacy",price:120)]
  try WatchlistStorage<Market>(defaults:defaults).save([value])
  let reopened = WatchlistStorage<Market>(defaults:try XCTUnwrap(UserDefaults(suiteName:name)))
  XCTAssertEqual(try reopened.load(),[value])
  XCTAssertNil(try reopened.load().first?.quotes.last?.updated)
 }
}
