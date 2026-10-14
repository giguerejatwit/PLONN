import Foundation

struct Quote: Codable, Hashable, Sendable {
    var book: String
    var price: Double
    /// The book/market's observed timestamp; a parent snapshot cannot supply it.
    var updated: String? = nil
}
struct Market: Codable, Identifiable, Hashable, Sendable {
    var id: String; var event: String; var sport: String; var kind: String
    var name: String; var opponent: String; var label: String; var aliases: String
    var p: Double?; var price: Double; var market: String; var book: String
    var quotes: [Quote]; var updated: String?; var mode: String
    var logs: [Double]?; var line: Double?; var eventLabel: String?
    var modelValidated: Bool?; var modelUpdated: String?; var modelVersion: String?
    var eventState: String?
    var commence: String?
    var side: String?
}

struct WatchlistStorage<Item: Codable> {
    let defaults: UserDefaults
    private let key = "watchlist"

    func load() throws -> [Item] {
        guard let data = defaults.data(forKey: key) else { return [] }
        return try JSONDecoder().decode([Item].self, from: data)
    }

    func save(_ items: [Item]) throws {
        // Encode before replacing the previous value so a failed save preserves it.
        let data = try JSONEncoder().encode(items)
        defaults.set(data, forKey: key)
    }

    func clear() { defaults.removeObject(forKey: key) }
}

// Value ranking requires explicit live state and model provenance; legacy/sample rows cannot qualify.
func hotValue(_ market: Market, now: Date = Date()) -> Double? {
    guard market.mode == "live", market.eventState == "in", market.modelValidated == true,
          let version = market.modelVersion, !version.isEmpty,
          let p = market.p, p.isFinite, p > 0, p < 1,
          market.price.isFinite, abs(market.price) >= 100,
          let quoteDate = market.updated, let modelDate = market.modelUpdated else { return nil }
    func timestamp(_ text: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: text) { return date }
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: text)
    }
    guard let quote = timestamp(quoteDate), let model = timestamp(modelDate),
          (0...120).contains(now.timeIntervalSince(quote)),
          (0...120).contains(now.timeIntervalSince(model)) else { return nil }
    let decimal = market.price > 0 ? 1 + market.price / 100 : 1 + 100 / abs(market.price)
    let value = p * decimal - 1
    return value > 0 ? value : nil
}

func sportsDate(_ text: String) -> Date? {
 let formatter=ISO8601DateFormatter()
 if let value=formatter.date(from:text){return value}
 formatter.formatOptions=[.withInternetDateTime,.withFractionalSeconds]
 if let value=formatter.date(from:text){return value}
 let minuteFormat=DateFormatter();minuteFormat.locale=Locale(identifier:"en_US_POSIX");minuteFormat.dateFormat="yyyy-MM-dd'T'HH:mmX"
 return minuteFormat.date(from:text)
}
