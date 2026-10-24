import Foundation

/// Versioned pregame priors and whole-trajectory conditioning. No final scores
/// are edited: selecting a path selects its complete score, player and play ledger.
enum SimulationForecast {
    static let version = "opponent-market-v2"
    struct MarketInput: Sendable {
        let sport: InterestSport
        let teamIDs: [String]
        let eventID: String
        let start: Date
        let checked: Date
        let recorded: Date
        let book: String
        let homeProbability: Double?
        let homeMargin: Double?
        let total: Double?
        let quoteIDs: [String]
        var homeCoverProbability: Double? = nil
        var overProbability: Double? = nil
    }
    struct Evidence: Sendable {
        let version: String
        let targets: [Double]
        let effectiveGames: [Double]
        let market: MarketInput?
        let effectivePaths: Double
        let limited: Bool
    }
    struct Selection { let indices: [Int]; let evidence: Evidence }
    static func neutral(_ sport: InterestSport) -> Double {
        switch sport { case .nba: return 114; case .nfl: return 22; case .nhl: return 3; case .mlb: return 4.4 }
    }
    static func weight(date: Date, asOf: Date, sport: InterestSport) -> Double {
        exp(-log(2) * max(0, asOf.timeIntervalSince(date)) / (86400 * (sport == .nfl ? 180 : 90)))
    }
    static func scoringTargets(_ histories: [PlayerSimulation.History], asOf: Date) -> (scores: [Double], games: [Double]) {
        let prior = neutral(histories[0].team.sport)
        let weighted = histories.map { history -> (Double, Double, Double) in
            var scored = prior * 4, allowed = prior * 4, n = 0.0
            for box in history.boxes where box.game.date < asOf {
                let w = weight(date: box.game.date, asOf: asOf, sport: history.team.sport)
                scored += w * Double(box.game.scored); allowed += w * Double(box.game.allowed); n += w
            }
            return (scored / (n + 4), allowed / (n + 4), n)
        }
        return ([(weighted[0].0 + weighted[1].1) / 2, (weighted[1].0 + weighted[0].1) / 2], weighted.map { $0.2 })
    }
    /// Each market must have both sides from the same bookmaker and observation
    /// window. Cross-book best prices, alternate lines and one-sided prices cannot
    /// be assembled into a synthetic market. Existing exact-event matcher is reused.
    static func market(event: PlayerMatchupEvent?, markets: [Market], now: Date) -> MarketInput? {
        guard let event, let start = event.game.start, start > now,
              let sport = InterestSport(rawValue: event.game.league) else { return nil }
        let proof = AnalyticsEvent(sport: sport.rawValue, providerID: event.game.eventID, home: event.game.home, away: event.game.away, start: start, homeID: event.homeID, awayID: event.awayID, seasonType: event.seasonType)
        // Snapshot rows can carry the best price from different books. Expand
        // timestamped same-contract alternatives before pairing; never borrow
        // the parent timestamp for an alternative book.
        let bookQuotes = markets.flatMap { row -> [Market] in
            var copies = [row]
            for quote in row.quotes where quote.updated != nil && quote.book != row.book {
                var copy = row
                copy.id = row.id + "/book/" + quote.book
                copy.book = quote.book; copy.price = quote.price; copy.updated = quote.updated; copy.quotes = []
                copies.append(copy)
            }
            return copies
        }
        var accepted: [String: Market] = [:]
        for name in [event.game.away, event.game.home] {
            for key in ["h2h", "spreads", "totals"] {
                let target = AnalyticsStatsTarget(sport: sport.rawValue, kind: "team", name: name, marketKey: key, teamName: name)
                for q in AnalyticsOddsMatcher.matchCached(target: target, event: proof, markets: bookQuotes, now: now, recordedSnapshot: true).markets {
                    // Legacy snapshots omit eventState. The independently verified
                    // future schedule above supplies state; explicit live/final
                    // or unknown states remain rejected.
                    guard ["", "pre", "scheduled"].contains((q.eventState ?? "").lowercased()), q.mode == "live", !q.book.isEmpty,
                          let date = q.updated.flatMap(sportsDate), date < start,
                          (0...86400).contains(now.timeIntervalSince(date)), q.price.isFinite, abs(q.price) >= 100 else { continue }
                    accepted[q.id] = q
                }
            }
        }
        var candidates: [MarketInput] = []
        for (book, quotes) in Dictionary(grouping: accepted.values, by: \.book) {
            let sorted = quotes.sorted { ($0.updated ?? "", $0.id) > ($1.updated ?? "", $1.id) }
            func pair(_ key: String) -> (Market, Market)? {
                for a in sorted where a.market == key {
                    for h in sorted where h.market == key && h.event == a.event && h.id != a.id {
                        guard let ad = a.updated.flatMap(sportsDate), let hd = h.updated.flatMap(sportsDate), abs(ad.timeIntervalSince(hd)) <= 300 else { continue }
                        if key == "totals" {
                            if a.side?.lowercased() == "over", h.side?.lowercased() == "under", let line = a.line, line > 0, line.isFinite, h.line == line { return (a, h) }
                        } else if RecordedTeamIdentity.matches(a.name, event.game.away), RecordedTeamIdentity.matches(h.name, event.game.home) {
                            if key == "h2h" { return (a, h) }
                            if let al = a.line, let hl = h.line, al.isFinite, hl.isFinite, abs(al + hl) < 0.001 { return (a, h) }
                        }
                    }
                }
                return nil
            }
            let money = pair("h2h"), spread = pair("spreads"), total = pair("totals")
            let used = [money, spread, total].compactMap { $0 }.flatMap { [$0.0, $0.1] }
            guard !used.isEmpty, Set(used.map(\.event)).count == 1, let oldest = used.compactMap({ $0.updated.flatMap(sportsDate) }).min(),
                  let latest = used.compactMap({ $0.updated.flatMap(sportsDate) }).max(), latest.timeIntervalSince(oldest) <= 300 else { continue }
            func normalizedSecond(_ pair: (Market, Market)?) -> Double? {
                guard let pair, let ad = BuilderLogic.decimalOdds(pair.0.price), let hd = BuilderLogic.decimalOdds(pair.1.price) else { return nil }
                let ap = 1 / ad, hp = 1 / hd
                guard (0.95...1.25).contains(ap + hp) else { return nil }
                return hp / (hp + ap)
            }
            let probability = normalizedSecond(money), cover = normalizedSecond(spread), over = normalizedSecond(total).map { 1 - $0 }
            guard probability != nil || cover != nil || over != nil else { continue }
            candidates.append(.init(sport: sport, teamIDs: [event.awayID, event.homeID], eventID: event.game.eventID, start: start, checked: now, recorded: oldest, book: book, homeProbability: probability, homeMargin: cover == nil ? nil : spread.flatMap { $0.1.line }.map { -$0 }, total: over == nil ? nil : total.flatMap { $0.0.line }, quoteIDs: used.map(\.id), homeCoverProbability: cover, overProbability: over))
        }
        return candidates.sorted { a, b in
            if a.quoteIDs.count != b.quoteIDs.count { return a.quoteIDs.count > b.quoteIDs.count }
            if a.recorded != b.recorded { return a.recorded > b.recorded }
            return a.book < b.book
        }.first
    }
    static func select(scores: [MatchupSimulation.Score], histories: [PlayerSimulation.History], now: Date, seed: UInt64, market: MarketInput?, live: SimulationLiveSnapshot?) -> Selection {
        let n = scores.count, sport = histories[0].team.sport, base = scoringTargets(histories, asOf: live?.start ?? now)
        let source = market.flatMap { m -> MarketInput? in
            guard m.sport == sport, m.teamIDs == histories.map({ $0.team.providerID }), m.checked <= now,
                  m.recorded <= m.checked, m.checked < m.start,
                  (0...86400).contains(m.checked.timeIntervalSince(m.recorded)) else { return nil }
            if let live { guard m.eventID == live.eventID, m.start == live.start else { return nil } }
            else { guard m.start > now, now.timeIntervalSince(m.recorded) <= 86400 else { return nil } }
            return m
        }
        var targets = base.scores
        if let source {
            func deviation(_ values: [Double]) -> Double {
                let mean = values.reduce(0, +) / Double(n)
                return sqrt(values.reduce(0) { $0 + pow($1 - mean, 2) } / Double(n))
            }
            // Recover a full-game scale from remaining-game variance when a
            // saved pregame prior is reused during play; observed points stay fixed.
            let remaining = live.map { snapshot -> Double in
                if sport == .mlb { return max(0.05, Double(10 - snapshot.period) / 9) }
                return max(0.05, 1 - Double(snapshot.elapsed) / Double(snapshot.regulationSeconds))
            } ?? 1
            let marginSD = deviation(scores.map { Double($0.margin) }) / sqrt(remaining)
            let totalSD = deviation(scores.map { Double($0.total) }) / sqrt(remaining)
            let marketScores = marketScores(source, baseline: targets, marginSD: marginSD, totalSD: totalSD)
            let marketTargets = [max(0.1, marketScores[0]), max(0.1, marketScores[1])]
            targets = zip(targets, marketTargets).map { 0.2 * $0 + 0.8 * $1 }
        }
        if let live {
            for side in 0...1 {
                let fraction: Double
                if sport == .mlb {
                    let completed = (live.period - 1) * 3 + (side < (live.battingSide ?? 0) ? 3 : side == live.battingSide ? live.outs ?? 0 : 0)
                    fraction = max(0, Double(27 - completed) / 27)
                } else { fraction = max(0, 1 - Double(live.elapsed) / (sport == .nba ? 2880 : 3600)) }
                targets[side] = Double(live.scores[side]) + targets[side] * fraction
            }
        }
        let points = scores.map { [Double($0.away), Double($0.home)] }
        func moments(_ theta: [Double]) -> (weights: [Double], mean: [Double], covariance: [Double], ess: Double) {
            let logits = points.map { theta[0] * $0[0] + theta[1] * $0[1] }, peak = logits.max() ?? 0
            let raw = logits.map { exp($0 - peak) }, sum = raw.reduce(0, +), w = raw.map { $0 / sum }
            var mean = [0.0, 0.0], cov = [0.0, 0.0, 0.0]
            for i in 0..<n { mean[0] += w[i] * points[i][0]; mean[1] += w[i] * points[i][1] }
            for i in 0..<n { let a = points[i][0] - mean[0], h = points[i][1] - mean[1]; cov[0] += w[i] * a * a; cov[1] += w[i] * a * h; cov[2] += w[i] * h * h }
            return (w, mean, cov, 1 / w.reduce(0) { $0 + $1 * $1 })
        }
        var theta = [0.0, 0.0]
        for _ in 0..<20 {
            let m = moments(theta), c = m.covariance, a = targets[0] - m.mean[0], h = targets[1] - m.mean[1]
            if max(abs(a), abs(h)) < 0.01 { break }
            let determinant = (c[0] + 0.01) * (c[2] + 0.01) - c[1] * c[1]
            guard determinant > 1e-9 else { break }
            let da = ((c[2] + 0.01) * a - c[1] * h) / determinant, dh = ((c[0] + 0.01) * h - c[1] * a) / determinant
            let step = min(1, 0.35 / max(0.001, max(abs(da), abs(dh))))
            theta[0] = min(3, max(-3, theta[0] + step * da)); theta[1] = min(3, max(-3, theta[1] + step * dh))
        }
        var fitted = moments(theta), limited = false
        // Do not represent a handful of outliers as 5,000 independent paths.
        // Back off unsupported corrections and report the limitation.
        if fitted.ess < Double(n) * 0.15 {
            limited = true
            var low = 0.0, high = 1.0
            for _ in 0..<30 { let mid = (low + high) / 2; if moments(theta.map { $0 * mid }).ess >= Double(n) * 0.15 { low = mid } else { high = mid } }
            fitted = moments(theta.map { $0 * low })
        }
        limited = limited || zip(fitted.mean, targets).contains { abs($0 - $1) > max(0.25, neutral(sport) * 0.02) }
        var rng = SportPlaySimulation.Random(state: seed ^ 0x76f103ae89d0c41b)
        let offset = rng.uniform() / Double(n)
        var cumulative = fitted.weights[0], cursor = 0, indices: [Int] = []
        for i in 0..<n {
            let point = offset + Double(i) / Double(n)
            while cursor + 1 < n && point > cumulative { cursor += 1; cumulative += fitted.weights[cursor] }
            indices.append(cursor)
        }
        return Selection(indices: indices, evidence: .init(version: version, targets: targets, effectiveGames: base.games, market: source, effectivePaths: fitted.ess, limited: limited))
    }
    static func marketScores(_ market: MarketInput, baseline: [Double], marginSD: Double, totalSD: Double) -> [Double] {
        // A priced spread is a threshold, not the expected margin. This matters
        // especially for fixed ±1.5 puck/run lines. A normal approximation is a
        // soft prior only; it does not claim exact settlement for discrete scores.
        let pricedSpread = market.homeMargin.map { $0 + marginSD * quantile(market.homeCoverProbability ?? 0.5) }
        let moneylineMargin = market.homeProbability.map { marginSD * quantile($0) }
        let margin: Double
        if market.sport == .nhl || market.sport == .mlb {
            margin = moneylineMargin ?? pricedSpread ?? baseline[1] - baseline[0]
        } else { margin = pricedSpread ?? moneylineMargin ?? baseline[1] - baseline[0] }
        let total = market.total.map { $0 + totalSD * quantile(market.overProbability ?? 0.5) } ?? baseline.reduce(0, +)
        return [(total - margin) / 2, (total + margin) / 2]
    }
    static func quantile(_ probability: Double) -> Double {
        let p = min(0.995, max(0.005, probability))
        var low = -3.0, high = 3.0
        for _ in 0..<40 { let mid = (low + high) / 2; if 0.5 * (1 + erf(mid / sqrt(2))) < p { low = mid } else { high = mid } }
        return (low + high) / 2
    }
}
