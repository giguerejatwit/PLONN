import Foundation

/// Supplied only after the caller has verified an existing account-owned result.
/// A market probability or saved setup cannot create this evidence.
struct AskPLONNModelResultSnapshot: Sendable {
    let id: UUID
    let ownerID: UUID
    let recipeID: UUID
    let recipeFingerprint: Data
    let sport: String
    let player: String
    let stat: String
    let targetKind: String
    let side: String?
    let threshold: Double?
    let historicalEventID: String?
    let probability: Double
    let validationSample: Int?
    let brier: Double?
    let mode: String
    let generatedAt: Date
}

/// A book's independently timestamped quote for one exact selection.
struct AskPLONNPriceQuote: Codable, Sendable, Equatable {
    let book: String
    let price: Double
    let decimal: Double
    let impliedProbability: Double
    let updated: Date
}
struct AskPLONNPriceComparison: Codable, Sendable {
    let selectionID: String
    let comparedAt: Date
    var bestQuote: AskPLONNPriceQuote? = nil
    var bestBooks: [String] = []
    var quotes: [AskPLONNPriceQuote] = []
    var modelProbability: Double? = nil
    var modelVersion: String? = nil
    var modelUpdated: Date? = nil
    var edgePercentagePoints: Double? = nil
    var edgeUnavailableReason = "A matching validated model probability is unavailable."
    var unavailableReason: String? = nil
    var excludedReasons: [String] = []
    var expiresAt: Date? = nil
    var availability: AskPLONNAvailability = .unavailable
}

/// Local, deterministic evidence. These factories never refresh odds, train or run a model.
enum AskPLONNResearch {
    static func recentForm(market: Market, history: PlayerHistoryResult, now: Date = Date()) -> AskPLONNEvidence {
        let id = "recent|\(market.id)"
        func missing(_ reason: String) -> AskPLONNEvidence {
            unavailable(id: id, kind: .recentForm, title: "Recent player form", reason: reason, source: history.source, sourceURLs: historyURLs(history))
        }
        guard market.kind == "prop", !normalized(market.name).isEmpty,
              OpponentPropHistoryEngine.supportedMarkets[market.sport]?.contains(market.market) == true,
              !normalized(history.source).isEmpty, now.timeIntervalSince1970.isFinite else {
            return missing("Verified player history is unavailable for this selection.")
        }
        if let context = history.context {
            guard context.sport == market.sport, normalized(context.playerName) == normalized(market.name),
                  context.marketKey == market.market else { return missing("Player history does not match the selected player and stat.") }
        }
        guard let resolvedName = history.playerName, normalized(resolvedName) == normalized(market.name),
              let expectedPlayer = history.playerID, !normalized(expectedPlayer).isEmpty else {
            return missing("The returned history is not bound to the selected player’s verified identity.")
        }
        if let context = history.context, context.playerID != expectedPlayer { return missing("The history identity conflicts with the opponent context.") }
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = TimeZone(identifier: "America/New_York")!
        let cutoff = calendar.startOfDay(for: min(market.commence.flatMap(sportsDate) ?? now, now))
        let grouped = Dictionary(grouping: history.games, by: \.id)
        let coherent = grouped.values.compactMap { copies -> PlayerGame? in
            guard let first = copies.first, !first.id.isEmpty, copies.allSatisfy({ sameGame($0, first) }) else { return nil }
            return first
        }
        let playerIDs = Set(coherent.compactMap(\.playerID))
        guard playerIDs.count == 1, let playerID = playerIDs.first, !normalized(playerID).isEmpty,
              expectedPlayer == playerID else { return missing("Player identity is missing or ambiguous in the returned history.") }
        let signed = market.sport == "NFL" && ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_rush_reception_yds"].contains(market.market)
        let games = Array(coherent.filter {
            $0.sport == market.sport && $0.marketKey == market.market && $0.playerID == playerID &&
            $0.id != market.event && $0.date.timeIntervalSince1970.isFinite && $0.date < cutoff &&
            $0.value.isFinite && $0.value.rounded() == $0.value && ($0.value >= 0 || signed) &&
            !normalized($0.workload).isEmpty && $0.season != nil &&
            !normalized($0.playerTeamID ?? "").isEmpty && !normalized($0.opponentID ?? "").isEmpty && $0.playerTeamID != $0.opponentID
        }.sorted { $0.date == $1.date ? $0.id < $1.id : $0.date > $1.date }.prefix(10))
        guard !games.isEmpty else { return missing(history.availability.isEmpty ? "No verified earlier appearances are available." : history.availability) }
        let average = games.reduce(0.0) { $0 + $1.value / Double(games.count) }
        guard average.isFinite else { return missing("The available results could not be summarized safely.") }
        let stat = OpponentPropHistoryEngine.statLabel(market.market)
        var text = "\(market.name) averaged \(number(average, decimals: 1)) \(stat) across \(games.count) recent verified appearances."
        let comparedThreshold = market.line.map { $0.isFinite && ($0 >= 0 || signed) } ?? false
        if let line = market.line, comparedThreshold {
            text += " Compared with the selected \(number(line)) threshold: \(games.filter { $0.value > line }.count) above, \(games.filter { $0.value < line }.count) below, \(games.filter { $0.value == line }.count) equal."
        }
        if games.count < 5 { text += " Small sample." }
        text += " Earlier calendar dates only. "
        text += comparedThreshold ? "Past results compared with the selected threshold, not historical bet outcomes or a forecast." : "Observed player results, not a forecast."
        if !history.warnings.isEmpty { text += " " + history.warnings.joined(separator: " ") }
        return AskPLONNEvidence(id: id, kind: .recentForm, title: "Recent player form", text: text,
            source: history.source, capturedAt: now, sampleSize: games.count, availability: .available, sourceURLs: historyURLs(history))
    }

    static func opponentHistory(market: Market, history: PlayerHistoryResult, now: Date = Date()) -> AskPLONNEvidence {
        let id = "opponent|\(market.id)"
        guard now.timeIntervalSince1970.isFinite, let context = history.context, context.sport == market.sport, context.marketKey == market.market,
              normalized(context.playerName) == normalized(market.name),
              let start = market.commence.flatMap(sportsDate), start == context.targetStart,
              context.targetEventID == nil || context.targetEventID == market.event else {
            return unavailable(id: id, kind: .opponentHistory, title: "History against this opponent",
                reason: "The selected matchup’s opponent or player identity is unverified. Recent form cannot establish opponent-specific history.", source: history.source, sourceURLs: historyURLs(history))
        }
        let summary = OpponentPropHistoryEngine.summarize(games: history.opponentGames, context: context, line: market.line, displayCount: 10)
        var text = summary.text + " " + summary.evidence
        if !history.warnings.isEmpty { text += " " + history.warnings.joined(separator: " ") }
        return AskPLONNEvidence(id: id, kind: .opponentHistory, title: "History against \(context.opponentName)",
            text: text, source: context.source, capturedAt: now, sampleSize: summary.n,
            availability: summary.unavailableReason == nil ? .available : .unavailable, sourceURLs: historyURLs(history))
    }

    static let priceMaximumAge: TimeInterval = 120
    static let modelMaximumAge: TimeInterval = 120

    static func priceComparison(market: Market, markets: [Market], now: Date = Date()) -> AskPLONNPriceComparison {
        var result = AskPLONNPriceComparison(selectionID: market.id, comparedAt: now)
        guard !partialGameSelection(market) else {
            result.unavailableReason = "Quarter, period, half and partial-inning prices require a verified settlement scope."
            return result
        }
        guard market.mode == "live", validIdentity(market), now.timeIntervalSince1970.isFinite else {
            result.unavailableReason = "An exact event, start time, player, stat, side and line are required."
            return result
        }
        guard !closedEvent(market.eventState) else {
            result.unavailableReason = "The selected event is closed or interrupted; prices and edge are unavailable."
            return result
        }
        // Event state describes the game, regardless of player/stat/side/line.
        // Check raw live game observations before row or price quarantine.
        guard !markets.contains(where: { $0.mode == "live" && !partialGameSelection($0) && sameEvent($0, market) && closedEvent($0.eventState) }) else {
            result.unavailableReason = "Conflicting closed or interrupted event observations; prices and edge are unavailable."
            return result
        }
        // Quarantine identity conflicts before filtering, including duplicates
        // that try to reuse one row ID across two different events/outcomes.
        var identityConflict = false
        let rows = Dictionary(grouping: markets, by: \.id).sorted { $0.key < $1.key }.compactMap { _, copies -> Market? in
            guard let first = copies.first, !normalized(first.id).isEmpty else { return nil }
            guard copies.allSatisfy({ sameQuote($0, first) }) else {
                if copies.contains(where: { $0.id == market.id }) { identityConflict = true }
                result.excludedReasons.append("Conflicting row identity or snapshots excluded.")
                return nil
            }
            return first
        }.filter { sameSelection($0, market) && $0.mode == "live" && validIdentity($0) }
        var candidates: [AskPLONNPriceQuote] = []
        var observations: [String: [(price: Double, updated: Date?, state: String)]] = [:]
        var hasStale = false
        func append(book: String, price: Double, stamp: String?, bundled: Bool, state: String?) {
            guard !normalized(book).isEmpty, book.count <= 100 else {
                result.excludedReasons.append("Missing or invalid book identity; price excluded."); return
            }
            let parsedStamp = stamp.flatMap(sportsDate)
            // Untimestamped legacy bundles establish no quote observation;
            // independently identified primary rows retain their conflicts.
            if !bundled || parsedStamp != nil {
                observations[normalized(book), default: []].append((price, parsedStamp, normalized(state ?? "")))
            }
            guard let decimal = BuilderLogic.decimalOdds(price) else {
                result.excludedReasons.append("\(book): invalid price excluded."); return
            }
            guard let updated = parsedStamp, updated.timeIntervalSince1970.isFinite else {
                result.excludedReasons.append("\(book): \(bundled ? "bundled quote" : "quote") timestamp unavailable; excluded from price ranking."); return
            }
            let age = now.timeIntervalSince(updated)
            guard age >= 0 else {
                result.excludedReasons.append("\(book): future quote timestamp; excluded from price ranking."); return
            }
            guard age <= priceMaximumAge else {
                hasStale = true
                result.excludedReasons.append("\(book) \(american(price)): stale snapshot, excluded from price ranking."); return
            }
            candidates.append(AskPLONNPriceQuote(book: book, price: price, decimal: decimal, impliedProbability: 1 / decimal, updated: updated))
        }
        for row in rows {
            guard !closedEvent(row.eventState) else {
                result.excludedReasons.append("\(row.book): event is closed or interrupted; price excluded."); continue
            }
            append(book: row.book, price: row.price, stamp: row.updated, bundled: false, state: row.eventState)
            for quote in row.quotes {
                append(book: quote.book, price: quote.price, stamp: quote.updated, bundled: true, state: row.eventState)
            }
        }
        for (book, copies) in Dictionary(grouping: candidates, by: { normalized($0.book) }).sorted(by: { $0.key < $1.key }) {
            guard let first = copies.sorted(by: { $0.book < $1.book }).first else { continue }
            let observed = observations[book] ?? []
            guard copies.allSatisfy({ $0.price == first.price && $0.updated == first.updated }),
                  observed.allSatisfy({ $0.price == first.price && $0.updated == first.updated && $0.state == observed.first?.state }) else {
                result.excludedReasons.append("\(first.book): conflicting snapshots excluded."); continue
            }
            result.quotes.append(first)
        }
        result.quotes.sort {
            if $0.decimal != $1.decimal { return $0.decimal > $1.decimal }
            return normalized($0.book) == normalized($1.book) ? $0.book < $1.book : normalized($0.book) < normalized($1.book)
        }
        result.excludedReasons = Array(Set(result.excludedReasons)).sorted()
        guard let best = result.quotes.first else {
            result.availability = hasStale ? .stale : .unavailable
            result.unavailableReason = "No current timestamped prices are available for this exact selection."
            return result
        }
        result.availability = .available
        result.bestQuote = best
        result.bestBooks = result.quotes.filter { $0.decimal == best.decimal }.map(\.book)
        result.expiresAt = result.quotes.map { $0.updated.addingTimeInterval(priceMaximumAge) }.min()
        guard !identityConflict else {
            result.edgeUnavailableReason = "The selected market has conflicting identity or quote snapshots."
            return result
        }
        guard market.modelValidated == true, let probability = market.p, probability.isFinite, probability > 0, probability < 1 else {
            result.edgeUnavailableReason = "A matching validated model probability strictly between 0% and 100% is unavailable."
            return result
        }
        guard let version = market.modelVersion, !normalized(version).isEmpty, version.count <= 200 else {
            result.edgeUnavailableReason = "The selected model version is unavailable."
            return result
        }
        guard let modelUpdated = market.modelUpdated.flatMap(sportsDate), modelUpdated.timeIntervalSince1970.isFinite else {
            result.edgeUnavailableReason = "The selected model timestamp is unavailable."
            return result
        }
        guard (0...modelMaximumAge).contains(now.timeIntervalSince(modelUpdated)) else {
            result.edgeUnavailableReason = "The selected model is stale or has a future timestamp."
            return result
        }
        let state = normalized(market.eventState ?? "")
        let futureScheduled = ["", "scheduled"].contains(state) && market.commence.flatMap(sportsDate).map { $0 > now } == true
        guard ["pre", "in"].contains(state) || futureScheduled else {
            result.edgeUnavailableReason = "The selected event is not known to be open or future scheduled."
            return result
        }
        result.modelProbability = probability
        result.modelVersion = version
        result.modelUpdated = modelUpdated
        result.expiresAt = min(result.expiresAt!, modelUpdated.addingTimeInterval(modelMaximumAge))
        if market.kind == "prop", OpponentPropHistoryEngine.supportedMarkets[market.sport]?.contains(market.market) != true {
            result.edgeUnavailableReason = "The selected player-stat settlement rules are unsupported."
            return result
        }
        if let line = market.line {
            guard line.rounded() != line else {
                result.edgeUnavailableReason = "Integer thresholds can push; edge requires a verified push probability."
                return result
            }
            guard (line * 2).isFinite, (line * 2).rounded() == line * 2 else {
                result.edgeUnavailableReason = "This threshold has unsupported split or fractional settlement rules."
                return result
            }
        }
        if market.kind == "team", ["h2h", "moneyline"].contains(market.market), ["NFL", "NHL"].contains(market.sport) {
            result.edgeUnavailableReason = "Two-way moneyline tie or regulation settlement requires verified tie and overtime rules."
            return result
        }
        result.edgePercentagePoints = 100 * (probability - best.impliedProbability)
        result.edgeUnavailableReason = ""
        return result
    }

    static func comparePrices(market: Market, markets: [Market], now: Date = Date()) -> AskPLONNEvidence {
        let comparison = priceComparison(market: market, markets: markets, now: now)
        var text: String
        if let best = comparison.bestQuote {
            let books = comparison.bestBooks.prefix(4).joined(separator: ", ")
            text = "Highest recorded payout for this exact selection: \(books), \(american(best.price)) (decimal \(number(best.decimal, decimals: 3))). Raw price-implied probability: \(number(best.impliedProbability * 100, decimals: 2))%."
            if comparison.quotes.count == 1 { text += " Only one timestamped book was fresh when this comparison was captured; best among available books, with limited comparison coverage." }
            else { text += " \(comparison.quotes.count) timestamped books were fresh when this comparison was captured." }
            text += " " + comparison.quotes.prefix(4).map { "\($0.book) \(american($0.price))." }.joined(separator: " ")
            if let edge = comparison.edgePercentagePoints, let probability = comparison.modelProbability {
                let signed = String(format: "%+.2f", locale: Locale(identifier: "en_US_POSIX"), edge)
                text += " Validated selected-model probability: \(number(probability * 100, decimals: 2))%. Model minus raw price-implied probability: \(signed) percentage points. Model version: \(comparison.modelVersion!)."
            } else { text += " Edge unavailable: " + comparison.edgeUnavailableReason }
        } else { text = comparison.unavailableReason ?? "Price comparison unavailable." }
        let disclosureBudget = max(0, 1700 - text.count)
        var used = 0
        for reason in comparison.excludedReasons {
            guard used + reason.count + 1 <= disclosureBudget else { break }
            text += " " + reason; used += reason.count + 1
        }
        text += " Quotes need their own per-book timestamps. Prices can change. Raw implied probability includes bookmaker margin; this is not a de-vig estimate, a betting advantage guarantee or a prediction."
        // Book/version display fields are bounded above; cap the optional list
        // so the exact evidence paragraph always respects the provider contract.
        if text.count > 2000 { text = String(text.prefix(1790)) + " Additional book details omitted. Prices can change; no advantage is guaranteed." }
        return AskPLONNEvidence(id: "prices|\(market.id)", kind: .comparePrices, title: "Compare book prices", text: text, source: "Recorded sportsbook snapshots",
            capturedAt: comparison.quotes.map(\.updated).min(), sampleSize: comparison.quotes.count, availability: comparison.availability)
    }

    private static func closedEvent(_ state: String?) -> Bool {
        ["post", "final", "completed", "closed", "canceled", "cancelled", "postponed", "suspended", "abandoned"].contains(normalized(state ?? ""))
    }
    private static func partialGameSelection(_ market: Market) -> Bool {
        let scope = (market.market.replacingOccurrences(of: "_", with: " ") + " " + market.label).lowercased()
        return scope.range(of: #"\b(quarter|period|half|q[1-4]|p[1-3]|h[12])\b|\b(1st|2nd|3rd|4th|first|second|third|fourth)\s+innings?\b|\b(first|1st|last)\s+(five|5|seven|7)\s+innings?\b"#, options: .regularExpression) != nil
    }

    static func modelSetup(recipe: LabRecipe, now: Date = Date()) -> AskPLONNEvidence {
        let id = "setup|\(recipe.id.uuidString)"
        guard validRecipe(recipe), recipe.created <= now else {
            return unavailable(id: id, kind: .explainModel, title: "Saved model configuration", reason: "The saved recipe has missing or invalid configuration fields.", source: "Saved model recipe")
        }
        let model = SavedModelType(rawValue: recipe.resolvedModelType)!.title
        let threshold = recipe.line.map { "\(recipe.side ?? "Selected") \(number($0))" } ?? "No numeric threshold"
        var text = "\(recipe.name): \(model), \(recipe.sport!) \(recipe.targetKind!) target \(recipe.player), \(recipe.stat!), \(threshold). Configured features: \(recipe.features.joined(separator: ", "))."
        if !recipe.opponent.isEmpty { text += " Configured opponent: \(recipe.opponent)." }
        if let settings = recipe.training {
            text += " Saved training settings: \(settings.epochs) epochs, learning rate \(number(settings.learningRate)), regularization \(number(settings.regularization)), validation share \(number(settings.validationShare))."
        }
        if let event = recipe.historicalEventID, !event.isEmpty { text += " Historical event configuration: \(event)." }
        text += " This is configuration only. A saved recipe does not prove training, validation, calibration or a model probability. Its threshold is a research setting, not a current sportsbook quote."
        return AskPLONNEvidence(id: id, kind: .explainModel, title: "Saved model configuration", text: text,
            source: "Saved model recipe", capturedAt: recipe.created, sampleSize: nil, availability: .available)
    }

    static func modelResult(snapshot: AskPLONNModelResultSnapshot, ownerID: UUID, recipe: LabRecipe, now: Date = Date()) -> AskPLONNEvidence {
        let id = "model-result|\(snapshot.id.uuidString)", source = "Verified existing account model result"
        guard now.timeIntervalSince1970.isFinite, validRecipe(recipe), snapshot.ownerID == ownerID, snapshot.recipeID == recipe.id,
              snapshot.recipeFingerprint == recipe.configurationFingerprint,
              snapshot.sport == recipe.sport, normalized(snapshot.player) == normalized(recipe.player), snapshot.stat == recipe.stat,
              snapshot.targetKind == recipe.targetKind, snapshot.side?.lowercased() == recipe.side?.lowercased(), snapshot.threshold == recipe.line,
              snapshot.historicalEventID == recipe.historicalEventID,
              snapshot.probability.isFinite, (0...1).contains(snapshot.probability),
              snapshot.generatedAt.timeIntervalSince1970.isFinite, snapshot.generatedAt >= recipe.created, snapshot.generatedAt <= now,
              ["fixed_threshold_retrospective", "unvalidated"].contains(snapshot.mode),
              snapshot.validationSample.map({ $0 > 0 }) != false,
              snapshot.brier.map({ $0.isFinite && (0...2).contains($0) }) != false else {
            return unavailable(id: id, kind: .explainModel, title: "Existing model result", reason: "The existing result’s owner, recipe, threshold, identity or validation fields could not be verified.", source: source)
        }
        var text = "The existing \(SavedModelType(rawValue: recipe.resolvedModelType)!.title) result reports \(number(snapshot.probability * 100, decimals: 1))%. \(LabServiceContract.probabilityLabel(recipe))."
        if snapshot.mode == "fixed_threshold_retrospective" { text += " Retrospective fixed-threshold evaluation; this is not a live prediction or historical sportsbook return." }
        else { text += " Unvalidated model output; calibration and predictive performance are unverified." }
        if let sample = snapshot.validationSample, let brier = snapshot.brier {
            text += " Reported validation sample: \(sample); multiclass Brier score: \(number(brier, decimals: 3)). This alone does not establish calibration, profitability or a betting edge."
        } else { text += " Validation sample or Brier score is unavailable; calibration is unverified." }
        text += " No new training or inference was performed."
        return AskPLONNEvidence(id: id, kind: .explainModel, title: "Existing model result", text: text, source: source,
            capturedAt: snapshot.generatedAt, sampleSize: snapshot.validationSample, availability: .available)
    }

    private static func validRecipe(_ recipe: LabRecipe) -> Bool {
        guard !normalized(recipe.name).isEmpty, recipe.name.count <= 200, !normalized(recipe.player).isEmpty, recipe.player.count <= 200, recipe.opponent.count <= 200,
              let sport = recipe.sport, ["NBA", "NFL", "NHL", "MLB"].contains(sport),
              let target = recipe.targetKind, ["Player", "Team"].contains(target), let stat = recipe.stat,
              LabPropCatalog.options(league: sport, target: target).contains(stat), stat != "Custom stat",
              SavedModelType(rawValue: recipe.resolvedModelType) != nil,
              LabServiceContract.validLine(recipe.line, target: target, stat: stat),
              !recipe.features.isEmpty, recipe.features.count <= 5, Set(recipe.features).count == recipe.features.count,
              recipe.features.allSatisfy({ !normalized($0).isEmpty && $0.count <= 100 }),
              recipe.created.timeIntervalSince1970.isFinite, recipe.configurationFingerprint != nil else { return false }
        if target == "Player" || LabServiceContract.gameMarket(target: target, stat: stat) != "moneyline" {
            guard ["over", "under"].contains(recipe.side?.lowercased() ?? "") else { return false }
        }
        if let settings = recipe.training {
            guard settings.epochs > 0, settings.learningRate.isFinite, settings.learningRate > 0,
                  settings.regularization.isFinite, settings.regularization >= 0,
                  settings.validationShare.isFinite, settings.validationShare > 0, settings.validationShare < 1 else { return false }
        }
        return true
    }
    private static func validIdentity(_ market: Market) -> Bool {
        guard !partialGameSelection(market), !normalized(market.id).isEmpty, market.id.count <= 180, !normalized(market.event).isEmpty,
              market.commence.flatMap(sportsDate) != nil, !normalized(market.name).isEmpty, market.name.count <= 200,
              ["NBA", "NFL", "NHL", "MLB"].contains(market.sport),
              let side = market.side, !normalized(side).isEmpty else { return false }
        let selection = normalized(side), key = market.market
        if market.kind == "prop" {
            let signed = market.sport == "NFL" && ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_rush_reception_yds"].contains(key)
            guard !key.isEmpty, ["over", "under"].contains(selection), let line = market.line, line.isFinite else { return false }
            return line >= 0 || signed
        }
        guard market.kind == "team" else { return false }
        if ["total", "totals", "alternate_totals", "team_total", "team_totals", "alternate_team_totals"].contains(key) {
            guard ["over", "under"].contains(selection), let line = market.line else { return false }
            return line.isFinite && line >= 0
        }
        if ["spread", "spreads", "alternate_spreads"].contains(key) {
            guard coherentTeamSide(market), let line = market.line else { return false }
            return line.isFinite
        }
        if ["h2h", "h2h_3_way", "moneyline"].contains(key) { return market.line == nil && coherentTeamSide(market) }
        return false
    }
    private static func coherentTeamSide(_ market: Market) -> Bool {
        let side = normalized(market.side ?? ""), name = normalized(market.name)
        guard !["over", "under"].contains(side) else { return false }
        if side == name { return true }
        guard let label = market.eventLabel else { return false }
        let pieces = label.contains(" at ") ? label.components(separatedBy: " at ") : label.components(separatedBy: " @ ")
        guard pieces.count == 2 else { return false }
        return (side == "away" && name == normalized(pieces[0])) || (side == "home" && name == normalized(pieces[1]))
    }
    private static func sameSelection(_ a: Market, _ b: Market) -> Bool {
        // Reject unrelated props before parsing their clocks or settlement labels.
        a.kind == b.kind && a.market == b.market && a.line == b.line &&
        normalized(a.name) == normalized(b.name) && normalized(a.side ?? "") == normalized(b.side ?? "") &&
        partialGameSelection(a) == partialGameSelection(b) && sameEvent(a, b)
    }
    private static func sameEvent(_ a: Market, _ b: Market) -> Bool {
        a.sport == b.sport && a.event == b.event &&
        (a.commence == b.commence || a.commence.flatMap(sportsDate) == b.commence.flatMap(sportsDate))
    }
    private static func sameQuote(_ a: Market, _ b: Market) -> Bool {
        sameSelection(a, b) && normalized(a.book) == normalized(b.book) && a.price == b.price &&
        a.updated == b.updated && a.mode == b.mode && a.eventState == b.eventState && a.quotes == b.quotes
    }
    private static func sameGame(_ a: PlayerGame, _ b: PlayerGame) -> Bool {
        a.date == b.date && a.value == b.value && a.workload == b.workload && a.sport == b.sport && a.playerID == b.playerID &&
        a.playerTeamID == b.playerTeamID && a.opponentID == b.opponentID && a.marketKey == b.marketKey && a.season == b.season
    }
    private static func unavailable(id: String, kind: AskPLONNIntent, title: String, reason: String, source: String, sourceURLs: [String] = []) -> AskPLONNEvidence {
        AskPLONNEvidence(id: id, kind: kind, title: title, text: reason, source: source.isEmpty ? "Local research checks" : source,
            capturedAt: nil, sampleSize: nil, availability: .unavailable, sourceURLs: sourceURLs)
    }
    private static func historyURLs(_ history: PlayerHistoryResult) -> [String] {
        var seen = Set<String>()
        return Array(history.sourceURLs.filter { path in
            guard AskPLONNCore.validSourceURL(path), let url = URL(string: path), PlayerHistoryService.allowedURL(url) else { return false }
            return seen.insert(path).inserted
        }.prefix(12))
    }
    private static func normalized(_ value: String) -> String { BuilderLogic.normalized(value) }
    private static func number(_ value: Double, decimals: Int = 0) -> String {
        if abs(value) >= 1_000_000_000 {
            return String(format: "%g", locale: Locale(identifier: "en_US_POSIX"), value)
        }
        return decimals == 0 ? String(format: "%g", locale: Locale(identifier: "en_US_POSIX"), value)
            : String(format: "%.*f", locale: Locale(identifier: "en_US_POSIX"), decimals, value)
    }
    private static func american(_ value: Double) -> String {
        String(format: value.rounded() == value && abs(value) <= 1_000_000 ? "%+.0f" : "%+g", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}
