import Foundation

/// A local projection of verified evidence and already loaded prices. No fetching or inference.
/// Statistical directions are deliberately separate from calibrated model/price edges.
enum MatchupOutlookSelection {
    struct Fact: Hashable, Sendable {
        let id, team, role, metric, unit, period, start, end: String
        let value: Double
        let sample: Int
        var player: String? = nil
        var cohort: String? = nil
        var expiresAt: Date? = nil
    }

    struct Lead: Equatable, Sendable {
        let title, selection, explanation, shortReason, qualification: String
        let marketID: String?
        let factIDs: [String]
        let edge: Double?
        let quoteDate: Date?
        let strength: Double
        let sample: Int
    }

    struct Direction: Identifiable, Equatable, Sendable {
        let id: String
        let lead: Lead
        /// A qualitative relationship, not a measured correlation coefficient.
        let correlationNote: String?
    }

    private struct Candidate {
        let lead: Lead
        let market: Market?
        var trendKey: String = ""
        var trendTeam: String = ""
    }

    static let noDirection = "No clear supported edge or direction in the available evidence."
    static let trendQualification = "Historical direction, not a calibrated probability or a measured betting edge."
    private static let scoreMetric = ["NFL": "pointsPerGame", "NBA": "pointsPerGame", "MLB": "runsPerGame", "NHL": "goalsPerGame"]
    // Exact units only: net passing yards cannot support a gross passing-yards prop;
    // team batting strikeouts cannot establish a starting pitcher's strikeout total.
    private static let propMetric = [
        "player_pass_yds": "passingYardsPerGame", "player_pass_attempts": "passingAttemptsPerGame",
        "player_pass_completions": "completionsPerGame", "player_rush_yds": "rushingYardsPerGame",
        "player_rush_attempts": "rushingAttemptsPerGame", "player_points": "pointsPerGame",
        "player_rebounds": "reboundsPerGame", "player_assists": "assistsPerGame",
        "player_threes": "threePointersMadePerGame", "batter_hits": "hitsPerGame",
        "batter_home_runs": "homeRunsPerGame", "player_shots_on_goal": "shotsPerGame",
        "player_goals": "goalsPerGame"]

    static func facts(_ response: SharedMatchupReportResponse, now: Date) -> [Fact] {
        guard let aged = try? SharedMatchupDisplayClock.update(response, now: now), aged.status == .ready else { return [] }
        return (aged.snapshot?.facts ?? []).map {
            Fact(id: $0.id, team: $0.team, role: $0.role, metric: $0.metric, unit: $0.unit,
                 period: $0.period, start: $0.windowStart, end: $0.windowEnd, value: $0.value, sample: $0.sample)
        }
    }

    static func facts(_ result: MatchupNarrativeV4.Result, now: Date) -> [Fact] {
        guard let projected = try? RichSportsContext.project(Data(result.context.canonical.utf8), now: now) else { return [] }
        return projected["facts"].array.compactMap { f in
            guard f["sample"]["unit"].string == "games", let count = f["sample"]["count"].integer,
                  let raw = f["value"].number, let value = Double(raw) else { return nil }
            return Fact(id: f["id"].string, team: f["team"].string, role: f["role"].string,
                metric: f["metric"].string, unit: f["unit"].string,
                period: [f["season"].canonical, f["phase"].string, f["period"].string].joined(separator: "|"),
                start: f["windowStart"].string, end: f["windowEnd"].string, value: value, sample: count,
                player: f["subject"]["kind"].string == "player" ? f["subject"]["name"].string : nil)
        }
    }

    /// Current provider-verified season profiles, independent of archived AI prose.
    /// Every observation retains its own fetch time. Same-season samples may have
    /// different game counts; league references require the complete league.
    static func currentFacts(event: SharedMatchupEvent, profiles: [InsightTeamProfile], now: Date) -> [Fact] {
        guard let start = sportsDate(event.startISO), start > now,
              let expected = ["NFL":32,"NBA":30,"MLB":30,"NHL":32][event.sport] else { return [] }
        let grouped = Dictionary(grouping: profiles, by: \.team)
        let usable = grouped.values.compactMap { rows -> InsightTeamProfile? in
            guard rows.count == 1, let row = rows.first, row.sample > 0,
                  let time = row.through, time <= now, now.timeIntervalSince(time) < 900,
                  let period = row.periodKey, !row.source.isEmpty else { return nil }
            let parts = period.split(separator:"|")
            guard parts.count == 3, parts[0] == event.sport,
                  ["regular","preseason"].contains(String(parts[2])),
                  let year = Int(parts[1]), (2000...2099).contains(year) else { return nil }
            return row
        }
        let targets = usable.filter { [event.homeTeam,event.awayTeam].contains($0.team) }.sorted { $0.team < $1.team }
        guard targets.count == 2, targets[0].periodKey == targets[1].periodKey,
              let period = targets[0].periodKey else { return [] }
        let year = Int(period.split(separator:"|")[1])!
        let seasonStart = ["NBA","NHL"].contains(event.sport) ? "\(year-1)-07-01T00:00:00Z" : "\(year)-01-01T00:00:00Z"
        guard let beginning = sportsDate(seasonStart), targets.allSatisfy({ $0.through! >= beginning }) else { return [] }
        let mapping: [(String,String,String)]
        switch event.sport {
        case "NFL": mapping = [("totalPointsPerGame","pointsPerGame","points per game"),("passingYardsPerGame","passingYardsPerGame","passing yards per game"),("rushingYardsPerGame","rushingYardsPerGame","rushing yards per game"),("passingAttemptsPerGame","passingAttemptsPerGame","attempts per game"),("rushingAttemptsPerGame","rushingAttemptsPerGame","attempts per game")]
        case "NBA": mapping = [("avgPoints","pointsPerGame","points per game"),("avgAssists","assistsPerGame","assists per game"),("avgRebounds","reboundsPerGame","rebounds per game"),("avgThreePointFieldGoalsMade","threePointersMadePerGame","made threes per game")]
        case "MLB": mapping = [("runsPerGame","runsPerGame","runs per game"),("hitsPerGame","hitsPerGame","hits per game"),("strikeoutsPerGame","strikeoutsPerGame","strikeouts per game")]
        default: mapping = [("avgGoals","goalsPerGame","goals per game"),("avgShots","shotsPerGame","shots per game")]
        }
        func fact(team:String,role:String,metric:String,unit:String,value:Double,sample:Int,asOf:Date) -> Fact {
            Fact(id:"current|"+period+"|"+team+"|"+role+"|"+metric,team:team,role:role,metric:metric,unit:unit,
                 period:period,start:seasonStart,end:ISO8601DateFormatter().string(from:asOf),value:value,sample:sample,
                 cohort:period,expiresAt:asOf.addingTimeInterval(900))
        }
        var result:[Fact]=[]
        for (key,metric,unit) in mapping {
            for row in targets {
                for (role,values) in [("offense",row.offense),("defense",row.allowed)] {
                    if let value=values[key],value.isFinite,value>=0 {
                        result.append(fact(team:row.team,role:role,metric:metric,unit:unit,value:value,sample:row.sample,asOf:row.through!))
                    }
                }
            }
            let league=usable.filter { $0.periodKey == period && $0.through! >= beginning }.sorted { $0.team < $1.team }
            if league.count == expected, league.allSatisfy({ row in row.offense[key].map { $0.isFinite && $0>=0 } ?? false }) {
                let count=league.reduce(0) { $0+$1.sample }
                let mean=league.reduce(0.0) { $0+$1.offense[key]!*Double($1.sample) } / Double(count)
                result.append(fact(team:"league",role:"reference",metric:metric,unit:unit,value:mean,sample:count,asOf:league.map { $0.through! }.min()!))
            }
        }
        return result
    }

    static func select(event: SharedMatchupEvent, facts: [Fact], markets: [Market], now: Date) -> Lead? {
        directions(event: event, facts: facts, markets: markets, now: now).first?.lead
    }

    static func directions(event: SharedMatchupEvent, facts: [Fact], markets: [Market], now: Date) -> [Direction] {
        guard let start = sportsDate(event.startISO), start > now,
              let minimum = MatchupNarrativeV4.minimum[event.sport] else { return [] }
        let pairRows = markets.filter { samePair($0, event: event) }
        // Parse each distinct event clock once, not once per prop per candidate.
        let clocks = Dictionary(uniqueKeysWithValues: Set(pairRows.compactMap(\.commence)).compactMap { stamp in
            sportsDate(stamp).map { (stamp, $0) }
        })
        let identities = Dictionary(grouping: pairRows, by: \.event).flatMap { id, rows in
            Set(rows.compactMap { $0.commence.flatMap { clocks[$0] } }).map { (id, $0) }
        }
        let eventRows = pairRows.filter { row in
            guard let quoted = row.commence.flatMap({ clocks[$0] }) else { return false }
            return RecordedStartPolicy.accepts(eventID: row.event, quoteStart: quoted, start: start,
                                               sport: event.sport, candidates: identities, now: now)
        }
        // A closed observation overrides an otherwise plausible pregame row.
        guard !eventRows.contains(where: { ["in", "live", "post", "final", "completed", "cancelled", "canceled", "postponed", "suspended"].contains(key($0.eventState ?? "")) }) else { return [] }
        let groups = Dictionary(grouping: eventRows, by: \.id)
        let rows = groups.values.compactMap { copies -> Market? in
            guard let first = copies.first, !first.id.isEmpty, copies.allSatisfy({ $0 == first }) else { return nil }
            return first
        }.sorted { $0.id < $1.id }
        let usable = facts.filter { fact in
            guard fact.expiresAt.map({ now < $0 }) ?? true, fact.sample >= minimum, fact.value.isFinite, fact.value >= 0, !fact.id.isEmpty,
                  let first = sportsDate(fact.start), let last = sportsDate(fact.end) else { return false }
            return first <= last && last <= now && last < start
        }
        func get(_ team: String, _ role: String, _ metric: String) -> Fact? {
            let found = usable.filter { $0.team == team && $0.role == role && $0.metric == metric && $0.player == nil }
            return found.count == 1 ? found[0] : nil
        }
        func coherent(_ fs: [Fact]) -> Bool {
            guard let first = fs.first else { return false }
            if let cohort = first.cohort {
                return fs.allSatisfy { $0.cohort == cohort && $0.unit == first.unit && $0.period == first.period && $0.start == first.start && $0.expiresAt != nil }
            }
            return fs.allSatisfy { $0.cohort == nil && $0.unit == first.unit && $0.period == first.period && $0.start == first.start && $0.end == first.end }
        }
        var candidates: [Candidate] = []
        for market in rows {
            // A team rate cannot support a player prop by itself. Skip hundreds
            // of irrelevant prop prices before the stricter quote comparison.
            if market.kind == "prop", market.modelValidated != true,
               !usable.contains(where: { $0.player.map(key) == key(market.name) }) { continue }
            // Contradictory probabilities for the same model snapshot cannot supply an edge.
            let modelConflict = eventRows.contains { other in
                other.modelValidated == true && market.modelValidated == true &&
                other.market == market.market && key(other.name) == key(market.name) &&
                key(other.side ?? "") == key(market.side ?? "") && other.line == market.line &&
                other.modelVersion == market.modelVersion && other.modelUpdated == market.modelUpdated && other.p != market.p
            }
            if modelConflict { continue }
            // Reuse the exact-selection, book conflict, settlement and model freshness checks.
            let comparison = AskPLONNResearch.priceComparison(market: market, markets: eventRows, now: now)
            if let edge = comparison.edgePercentagePoints, edge > 0, let quote = comparison.bestQuote {
                candidates.append(Candidate(lead: Lead(title: "Strongest model edge", selection: label(market) + " · " + american(quote.price) + " · " + comparison.bestBooks.joined(separator: ", "),
                    explanation: "Validated model probability \(number((comparison.modelProbability ?? 0) * 100))% versus \(number(quote.impliedProbability * 100))% implied by the price: +\(number(edge)) percentage points.",
                    shortReason: "Model \(number((comparison.modelProbability ?? 0) * 100))% vs price-implied \(number(quote.impliedProbability * 100))%: +\(number(edge)) percentage points.",
                    qualification: "Largest positive model edge among the available matching markets. Prices can change; implied probability includes bookmaker margin.",
                    marketID: market.id, factIDs: [], edge: edge, quoteDate: quote.updated, strength: edge, sample: 0), market: market))
                continue
            }
            // Do not let a known nonpositive model edge masquerade as value via a trend.
            if comparison.edgePercentagePoints != nil { continue }
            guard let quote = recordedQuote(market, rows: eventRows, now: now), let metric = scoreMetric[event.sport] else { continue }
            var supporting: [Fact] = [], margins: [Double] = [], explanation = "", shortReason = ""
            let side = key(market.side ?? "")
            if market.kind == "team", ["h2h", "moneyline", "spreads", "spread", "totals", "total", "team_totals", "team_total"].contains(market.market) {
                guard let ho = get(event.homeTeam, "offense", metric), let hd = get(event.homeTeam, "defense", metric),
                      let ao = get(event.awayTeam, "offense", metric), let ad = get(event.awayTeam, "defense", metric), coherent([ho, hd, ao, ad]) else { continue }
                supporting = [ho, hd, ao, ad]
                if ["totals", "total"].contains(market.market), let line = market.line, line > 0 {
                    let offense = ho.value + ao.value, defense = hd.value + ad.value
                    let sign = side == "over" ? 1.0 : -1.0
                    margins = [sign * (offense - line) / line, sign * (defense - line) / line]
                    explanation = "The teams combine for \(number(offense)) \(ho.unit) scored and \(number(defense)) allowed; both are \(side == "over" ? "above" : "below") the \(number(line)) total."
                    shortReason = "Combined scoring (\(number(offense))) and allowance (\(number(defense))) are both \(side == "over" ? "above" : "below") \(number(line))."
                } else if [event.homeTeam, event.awayTeam].map(key).contains(key(market.name)) {
                    let home = key(market.name) == key(event.homeTeam)
                    let own = home ? ho : ao, opposing = home ? ad : hd
                    let other = home ? ao : ho, defense = home ? hd : ad
                    if ["team_totals", "team_total"].contains(market.market), let line = market.line, line > 0 {
                        let sign = side == "over" ? 1.0 : -1.0
                        margins = [sign * (own.value - line) / line, sign * (opposing.value - line) / line]
                        supporting = [own, opposing]
                        explanation = "\(own.team) scores \(number(own.value)) \(own.unit); \(opposing.team) allows \(number(opposing.value)). Both rates are \(side == "over" ? "above" : "below") \(number(line))."
                        shortReason = "Scores \(number(own.value)) per game; opponent allows \(number(opposing.value)). Both are \(side == "over" ? "above" : "below") \(number(line))."
                    } else if ["h2h", "moneyline", "spreads", "spread"].contains(market.market) {
                        let handicap = market.line ?? 0
                        let scale = max(1, (own.value + other.value + defense.value + opposing.value) / 4)
                        let attackGap = own.value - other.value, defenseGap = opposing.value - defense.value
                        margins = [(attackGap + handicap) / scale, (defenseGap + handicap) / scale]
                        explanation = "\(own.team) scores \(number(own.value)) and allows \(number(defense.value)) \(own.unit); \(other.team) scores \(number(other.value)) and allows \(number(opposing.value)). Both the scoring and defensive comparisons support \(market.name)\(market.line == nil ? "" : " at this spread") if those patterns continue."
                        shortReason = "Scores \(number(own.value)) vs \(number(other.value)); allows \(number(defense.value)) vs \(number(opposing.value)) per game. Both comparisons support this side."
                    }
                }
            } else if market.kind == "prop", let metric = propMetric[market.market], let line = market.line, line > 0 {
                let players = usable.filter { $0.role == "player" && $0.metric == metric && $0.player.map(key) == key(market.name) && [event.homeTeam, event.awayTeam].contains($0.team) }
                guard players.count == 1, let player = players.first,
                      let offense = get(player.team, "offense", metric),
                      let defense = get(player.team == event.homeTeam ? event.awayTeam : event.homeTeam, "defense", metric),
                      let reference = get("league", "reference", metric), reference.value > 0,
                      coherent([player, offense, defense, reference]) else { continue }
                let sign = side == "over" ? 1.0 : -1.0
                margins = [sign * (player.value - line) / line, sign * (offense.value - reference.value) / reference.value, sign * (defense.value - reference.value) / reference.value]
                supporting = [player, offense, defense, reference]
                explanation = "\(market.name) averages \(number(player.value)) \(player.unit), \(side == "over" ? "above" : "below") \(number(line)). \(player.team)'s \(number(offense.value)) and \(defense.team)'s \(number(defense.value)) allowed are also \(side == "over" ? "above" : "below") the \(number(reference.value)) same-period team benchmark."
                shortReason = "Player average \(number(player.value)) vs \(number(line)); team offense and opposing defense support the same direction."
            }
            // Require agreement on every premise and a material five-percent gap.
            // This ranks descriptive separation, never converts it to probability.
            guard let separation = margins.min(), separation >= 0.05, !supporting.isEmpty else { continue }
            let sample = supporting.map(\.sample).min() ?? 0
            candidates.append(Candidate(lead: Lead(title: "Strongest supported direction", selection: label(market) + " · " + american(quote.price) + " · " + quote.book,
                explanation: explanation + " Based on at least \(sample) games per sample.",
                shortReason: shortReason + " \(sample)+ games per sample.",
                qualification: trendQualification + (supporting.contains { $0.period.contains("preseason") } ? " Preseason lineups and minutes can change." : ""),
                marketID: market.id, factIDs: supporting.map(\.id), edge: nil, quoteDate: quote.updated,
                strength: separation * min(1, Double(sample) / Double(minimum * 2)), sample: sample), market: market))
        }
        if !candidates.isEmpty { return ranked(candidates) }

        // Search every supported metric in the validated snapshot, including facts the
        // older scoring-only report did not choose. Mixed/limited samples never lead.
        var trends: [Candidate] = []
        for team in [event.homeTeam, event.awayTeam] {
            for (metric, names) in MatchupNarrativeV4.pairs {
                guard let offense = get(team, "offense", metric),
                      let defense = get(team == event.homeTeam ? event.awayTeam : event.homeTeam, "defense", metric),
                      let reference = get("league", "reference", metric), reference.value > 0,
                      coherent([offense, defense, reference]) else { continue }
                let a = offense.value / reference.value - 1, b = defense.value / reference.value - 1
                guard (a >= 0.1 && b >= 0.1) || (a <= -0.1 && b <= -0.1) else { continue }
                let sample = min(offense.sample, defense.sample), up = a > 0
                trends.append(Candidate(lead: Lead(title: "Strongest supported direction", selection: "\(team): \(up ? "more" : "fewer") \(names[4])",
                    explanation: "\(team) \(names[2]) \(number(offense.value)) \(offense.unit); \(defense.team) \(names[3]) \(number(defense.value)). Both are at least 10% \(up ? "above" : "below") the \(number(reference.value)) same-period benchmark across at least \(sample) games. No matching supported line and odds are available.",
                    shortReason: "Team rate \(number(offense.value)) and opposing rate \(number(defense.value)) are both \(up ? "above" : "below") the \(number(reference.value)) benchmark. \(sample)+ games; matching odds unavailable.",
                    qualification: trendQualification + (offense.period.contains("preseason") ? " Preseason lineups and minutes can change." : ""),
                    marketID: nil, factIDs: [offense.id, defense.id, reference.id], edge: nil, quoteDate: nil,
                    strength: min(abs(a), abs(b)) * min(1, Double(sample) / Double(minimum * 2)), sample: sample), market: nil, trendKey: key(team) + "|" + metric, trendTeam: key(team)))
            }
        }
        return ranked(trends)
    }

    private static func ranked(_ candidates: [Candidate]) -> [Direction] {
        var chosen: [Candidate] = [], result: [Direction] = [], seen = Set<String>()
        for candidate in candidates.sorted(by: { stronger($0.lead, $1.lead) }) {
            let id = angleID(candidate)
            // Books, alternate thresholds and opposite sides of the same market are
            // one angle. Keep the strongest qualifying selection, never fill slots
            // with repeated expressions of it or mutually opposing directions.
            guard seen.insert(id).inserted else { continue }
            let related = chosen.enumerated().compactMap { index, prior -> String? in
                guard let reason = relationship(candidate, prior) else { return nil }
                let target = index == 0 ? "top direction" : "direction \(index + 1)"
                return "Potentially correlated with \(target) · \(reason)."
            }.first
            result.append(Direction(id: id, lead: candidate.lead, correlationNote: related))
            chosen.append(candidate)
            if result.count == 3 { break }
        }
        return result
    }

    private static func family(_ market: Market) -> String {
        switch market.market {
        case "h2h", "moneyline", "h2h_3_way": return "moneyline"
        case "spread", "spreads", "alternate_spreads": return "spread"
        case "total", "totals", "alternate_totals": return "total"
        case "team_total", "team_totals", "alternate_team_totals": return "team_total"
        default: return market.market
        }
    }

    private static func angleID(_ candidate: Candidate) -> String {
        guard let market = candidate.market else { return "trend|" + candidate.trendKey }
        let group = family(market)
        let subject = market.kind == "prop" || group == "team_total" ? key(market.name) : "game"
        return market.kind + "|" + group + "|" + subject
    }

    private static func relationship(_ a: Candidate, _ b: Candidate) -> String? {
        if !a.trendTeam.isEmpty, a.trendTeam == b.trendTeam { return "same team context" }
        if let am = a.market, let bm = b.market {
            if am.kind == "prop", bm.kind == "prop", key(am.name) == key(bm.name) { return "same player" }
            let scoringProps: Set<String> = ["player_points", "player_points_rebounds_assists", "player_goals", "batter_home_runs", "batter_rbis", "batter_runs_scored", "player_pass_tds", "player_anytime_td", "player_rush_tds", "player_reception_tds"]
            func scoring(_ market: Market) -> Bool {
                market.kind == "team" && ["moneyline", "spread", "total", "team_total"].contains(family(market)) || scoringProps.contains(market.market)
            }
            if scoring(am), scoring(bm) { return "shared scoring exposure" }
            let passing: Set<String> = ["player_pass_yds", "player_pass_attempts", "player_pass_completions", "player_reception_yds", "player_receptions"]
            if am.sport == "NFL", passing.contains(am.market), passing.contains(bm.market) { return "linked passing and receiving workload" }
        }
        if !Set(a.lead.factIDs).isDisjoint(with: b.lead.factIDs) { return "shared statistical evidence" }
        return nil
    }

    private static func stronger(_ a: Lead, _ b: Lead) -> Bool {
        if (a.edge != nil) != (b.edge != nil) { return a.edge != nil }
        if a.strength != b.strength { return a.strength > b.strength }
        if a.sample != b.sample { return a.sample > b.sample }
        if a.quoteDate != b.quoteDate { return (a.quoteDate ?? .distantPast) > (b.quoteDate ?? .distantPast) }
        return (a.marketID ?? a.selection) < (b.marketID ?? b.selection)
    }

    static func matches(_ m: Market, event: SharedMatchupEvent, candidates: [Market], now: Date) -> Bool {
        guard samePair(m, event: event), let quoted = m.commence.flatMap(sportsDate),
              let start = sportsDate(event.startISO) else { return false }
        let identities = candidates.filter { samePair($0, event: event) }.compactMap { row in
            row.commence.flatMap(sportsDate).map { (row.event, $0) }
        }
        return RecordedStartPolicy.accepts(eventID: m.event, quoteStart: quoted, start: start, sport: event.sport, candidates: identities, now: now)
    }
    private static func samePair(_ m: Market, event: SharedMatchupEvent) -> Bool {
        guard m.sport == event.sport, m.mode == "live", !m.event.isEmpty else { return false }
        let labels = [event.awayTeam + " at " + event.homeTeam, event.awayTeam + " @ " + event.homeTeam].map(key)
        // Provider IDs differ; the complete team pair is required in addition to start policy.
        return m.eventLabel.map { labels.contains(key($0)) } == true
    }

    private static func recordedQuote(_ market: Market, rows: [Market], now: Date) -> AskPLONNPriceQuote? {
        guard let observed = market.updated.flatMap(sportsDate), observed <= now,
              now.timeIntervalSince(observed) <= 86400 else { return nil }
        // At the observation instant, the existing validator still enforces selection,
        // timestamp and conflict rules. This is labelled as a snapshot, never live edge.
        return AskPLONNResearch.priceComparison(market: market, markets: rows, now: observed).bestQuote
    }

    static func quoteCaption(_ date: Date) -> String {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "MMM d, h:mm a z"
        return "Odds snapshot · " + f.string(from: date)
    }
    private static func label(_ m: Market) -> String {
        if ["h2h", "moneyline", "h2h_3_way"].contains(m.market) { return m.name + " moneyline" }
        let line = m.line.map(number) ?? ""
        if ["spreads", "spread", "alternate_spreads"].contains(m.market) {
            let term = m.sport == "MLB" ? "run line" : m.sport == "NHL" ? "puck line" : "spread"
            return m.name + " " + ((m.line ?? 0) > 0 ? "+" : "") + line + " " + term
        }
        let side = key(m.side ?? "").capitalized
        if m.kind == "prop" { return m.name + " · " + side + " " + line + " " + OpponentPropHistoryEngine.statLabel(m.market) }
        let team = ["team_total", "team_totals", "alternate_team_totals"].contains(m.market) ? m.name + " · " : ""
        return team + side + " " + line + " " + (["MLB": "runs", "NHL": "goals"][m.sport] ?? "points")
    }
    private static func key(_ text: String) -> String { AnalyticsOddsMatcher.normalized(text) }
    private static func number(_ n: Double) -> String { SharedMatchupPresentationCopy.number(n) }
    private static func american(_ n: Double) -> String { (n > 0 ? "+" : "") + number(n) }
}
