import Foundation

struct BuilderPriceEstimate: Equatable {
    let decimal: Double
    let american: Double
    let priceImpliedProbability: Double
}
struct BuilderReturnEstimate: Equatable {
    let totalReturn: Double
    let profit: Double
}
struct BuilderAssessment {
    let estimate: BuilderPriceEstimate?
    let blockingReason: String?
    let mixedBooks: Bool
    let warnings: [String]
}
enum BuilderQuoteAge: Equatable {
    case sample, recent, stale, missing, future
    var label: String {
        switch self {
        case .sample: return "Illustrative sample"
        case .recent: return "Recorded within 2 min"
        case .stale: return "Snapshot over 2 min old"
        case .missing: return "Quote time unavailable"
        case .future: return "Quote clock mismatch"
        }
    }
}
enum BuilderLogic {
    static let limit = 8
    static func stakeAmount(_ stakeText: String) -> Double? {
        let text = stakeText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.range(of: #"^(?:[0-9]+(?:\.[0-9]{0,2})?|\.[0-9]{1,2})$"#, options: .regularExpression) != nil,
              let stake = Double(text), stake.isFinite, stake > 0 else { return nil }
        return stake
    }
    static func returnEstimate(stakeText: String, estimate: BuilderPriceEstimate?) -> BuilderReturnEstimate? {
        guard let stake = stakeAmount(stakeText),
              let estimate, estimate.decimal.isFinite, estimate.decimal > 1 else { return nil }
        let total = stake * estimate.decimal, profit = total - stake
        guard total.isFinite, profit.isFinite else { return nil }
        return BuilderReturnEstimate(totalReturn: total, profit: profit)
    }
    static func normalized(_ value: String) -> String {
        value.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }
    static func decimalOdds(_ american: Double) -> Double? {
        guard american.isFinite, abs(american) >= 100 else { return nil }
        let decimal = american > 0 ? 1 + american / 100 : 1 + 100 / abs(american)
        return decimal.isFinite && decimal > 1 ? decimal : nil
    }
    static func sameEvent(_ a: Market, _ b: Market) -> Bool {
        if !normalized(a.event).isEmpty, normalized(a.event) == normalized(b.event) { return true }
        guard let labelA = a.eventLabel, let labelB = b.eventLabel, !normalized(labelA).isEmpty,
              normalized(labelA) == normalized(labelB), a.sport == b.sport,
              let dateA = a.commence.flatMap(sportsDate), let dateB = b.commence.flatMap(sportsDate) else { return false }
        return abs(dateA.timeIntervalSince(dateB)) < 1800
    }
    private static func marketKey(_ m: Market) -> String {
        normalized(m.market).replacingOccurrences(of: "alternate_", with: "").replacingOccurrences(of: "_alternate", with: "")
    }
    private static func side(_ m: Market) -> String {
        if let side = m.side, !normalized(side).isEmpty { return normalized(side) }
        let first = normalized(m.label).split(separator: " ").first.map(String.init) ?? ""
        return ["over", "under"].contains(first) ? first : normalized(m.name)
    }
    static func rejection(for candidate: Market, existing: [Market]) -> String? {
        guard !normalized(candidate.id).isEmpty else { return "Selection has no stable source identifier." }
        guard ["live", "demo"].contains(candidate.mode) else { return "Selection source is unsupported." }
        if let mode = existing.first?.mode, mode != candidate.mode { return "Keep sample research and sportsbook selections in separate drafts." }
        for leg in existing {
            if leg.id == candidate.id { return "This selection is already in the draft." }
            guard sameEvent(leg, candidate), marketKey(leg) == marketKey(candidate) else { continue }
            let sameSubject = normalized(leg.name) == normalized(candidate.name)
            if sameSubject, side(leg) == side(candidate), leg.line == candidate.line {
                return "This outcome is already selected, possibly from another book." }
            let key = marketKey(leg)
            if ["h2h", "h2h_3_way", "outrights"].contains(key), !sameSubject {
                return "These selections name different winners of the same market." }
            if sameSubject, let a = leg.line, let b = candidate.line, a.isFinite, b.isFinite {
                let sides = Set([side(leg), side(candidate)])
                if sides == Set(["over", "under"]) {
                    let over = side(leg) == "over" ? a : b
                    let under = side(leg) == "under" ? a : b
                    if over >= under { return "These over/under selections cannot both win." }
                } else if side(leg) == side(candidate), a != b {
                    return "Choose one line for this outcome before adding another." }
            }
            if key == "spreads", !sameSubject, let a = leg.line, let b = candidate.line, a + b <= 0 {
                return "These opposing spreads cannot both win." }
        }
        if existing.count >= limit { return "A research draft supports up to eight legs." }
        return nil
    }
    static func sanitized(_ candidates: [Market]) -> (legs: [Market], notices: [String]) {
        var legs: [Market] = [], notices: [String] = []
        for leg in candidates {
            if let reason = rejection(for: leg, existing: legs) { notices.append(leg.name + ": " + reason) }
            else { legs.append(leg) }
        }
        return (legs, notices)
    }
    static func quoteAge(_ market: Market, now: Date = Date()) -> BuilderQuoteAge {
        if market.mode == "demo" { return .sample }
        guard let time = market.updated.flatMap(sportsDate) else { return .missing }
        let age = now.timeIntervalSince(time)
        if age < -30 { return .future }
        return age > 120 ? .stale : .recent
    }
    // Refresh only the exact saved outcome at its original book. A moved line requires user selection.
    static func refreshedQuote(for saved: Market, from markets: [Market], now: Date = Date()) -> Market? {
        guard saved.mode == "live", !normalized(saved.event).isEmpty, !normalized(saved.book).isEmpty else { return nil }
        let matches = markets.filter { candidate in
            candidate.mode == "live" && candidate.sport == saved.sport && candidate.event == saved.event &&
            candidate.kind == saved.kind && normalized(candidate.market) == normalized(saved.market) &&
            normalized(candidate.name) == normalized(saved.name) && side(candidate) == side(saved) &&
            candidate.line == saved.line && normalized(candidate.book) == normalized(saved.book) &&
            candidate.eventState != "post" && decimalOdds(candidate.price) != nil &&
            quoteAge(candidate, now: now) == .recent &&
            (saved.updated.flatMap(sportsDate).map { old in
                candidate.updated.flatMap(sportsDate).map { $0 >= old } ?? false
            } ?? true)
        }
        guard let newest = matches.compactMap({ $0.updated.flatMap(sportsDate) }).max() else { return nil }
        let latest = matches.filter { $0.updated.flatMap(sportsDate) == newest }
        // Conflicting equally recent prices are ambiguous, so retain the recorded snapshot.
        guard let quote = latest.first, latest.allSatisfy({ $0.price == quote.price }) else { return nil }
        return quote
    }

    static func assess(_ legs: [Market], now: Date = Date()) -> BuilderAssessment {
        let books = Set(legs.map { normalized($0.book) }.filter { !$0.isEmpty })
        var warnings: [String] = []
        if books.count > 1 { warnings.append("Mixed sportsbooks · This is a cross-book price illustration, not an available parlay.") }
        if legs.contains(where: { normalized($0.book).isEmpty }) { warnings.append("A sportsbook source is missing. Verify every leg at one book.") }
        if legs.contains(where: { quoteAge($0, now: now) == .stale }) { warnings.append("Some snapshots are over two minutes old. Confirm the current lines and prices.") }
        if legs.contains(where: { [.missing, .future].contains(quoteAge($0, now: now)) }) { warnings.append("Quote freshness cannot be established for every leg.") }
        if legs.contains(where: { $0.mode == "demo" }) { warnings.append("Sample prices are illustrative and cannot be placed at a sportsbook.") }
        func blocked(_ text: String) -> BuilderAssessment { BuilderAssessment(estimate:nil,blockingReason:text,mixedBooks:books.count > 1,warnings:warnings) }
        guard legs.count >= 2 else { return blocked("Add at least two selections to compare combined prices.") }
        let clean = sanitized(legs)
        guard clean.legs.count == legs.count else { return blocked(clean.notices.first ?? "Resolve conflicting selections.") }
        guard legs.allSatisfy({ !normalized($0.event).isEmpty }) else { return blocked("Event identity is missing. Correlation cannot be checked; a real sportsbook quote is required.") }
        for i in legs.indices {
            for j in legs.indices where j > i {
                if sameEvent(legs[i], legs[j]) { return blocked("Same-game legs may be correlated. Combined odds and implied probability require a real sportsbook quote; none is connected.") }
            }
        }
        guard !legs.contains(where: { $0.eventState == "post" }) else { return blocked("A selected event is final. Remove it before comparing a new parlay.") }
        var combined = 1.0
        for leg in legs {
            guard let decimal = decimalOdds(leg.price) else { return blocked("One or more prices are invalid or unavailable.") }
            combined *= decimal
            guard combined.isFinite else { return blocked("Combined price exceeds the supported numeric range.") }
        }
        let american = combined >= 2 ? (combined - 1) * 100 : -100 / (combined - 1)
        guard american.isFinite else { return blocked("Combined American odds are unavailable.") }
        return BuilderAssessment(estimate:BuilderPriceEstimate(decimal:combined,american:american,priceImpliedProbability:1/combined),blockingReason:nil,mixedBooks:books.count > 1,warnings:warnings)
    }
}

struct BuilderDraftStorage {
    let defaults: UserDefaults
    static let key = "builderResearchDraft.v1"
    private struct Draft: Codable { let version: Int; let legs: [Market] }
    enum DraftError: Error { case unsupportedVersion, oversized }
    func load() throws -> [Market] {
        guard let data = defaults.data(forKey: Self.key) else { return [] }
        guard data.count <= 1_000_000 else { throw DraftError.oversized }
        let draft = try JSONDecoder().decode(Draft.self, from: data)
        guard draft.version == 1 else { throw DraftError.unsupportedVersion }
        return draft.legs
    }
    func save(_ legs: [Market]) throws {
        let data = try JSONEncoder().encode(Draft(version:1,legs:legs))
        guard data.count <= 1_000_000 else { throw DraftError.oversized }
        defaults.set(data, forKey: Self.key)
    }
    func clear() { defaults.removeObject(forKey: Self.key) }
}
