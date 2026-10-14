import Foundation

/// Display-only projection of already validated claims. Stored text and evidence stay exact.
enum MatchupNarrativePresentation {
    static func text(for claim: MatchupNarrativeV4.Claim) -> String {
        var text = claim.text
        for sample in claim.samples.array {
            let start = sample["windowStart"].string
            let end = sample["windowEnd"].string
            let span = "(\(start) to \(end))"
            guard claim.factIDs.contains(sample["factID"].string),
                  text.contains(span),
                  let first = try? SharedMatchupReportValidator.date(start),
                  let last = try? SharedMatchupReportValidator.date(end),
                  first <= last else { continue }
            // Replace the complete sample span only, never arbitrary timestamps or numbers.
            text = text.replacingOccurrences(of: span, with: "(\(window(first, last)))")
        }
        return text
    }

    private static func window(_ first: Date, _ last: Date) -> String {
        // UTC keeps source calendar dates stable across device time zones. The original
        // instants, including seconds and fractions, remain in claim.samples and rawData.
        let utc = TimeZone(secondsFromGMT: 0)!
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utc
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = calendar
        formatter.timeZone = utc
        func date(_ value: Date, _ pattern: String) -> String {
            formatter.dateFormat = pattern
            return formatter.string(from: value)
        }
        if calendar.isDate(first, inSameDayAs: last) {
            return date(first, "MMM d, yyyy") + " UTC"
        }
        if calendar.component(.year, from: first) == calendar.component(.year, from: last) {
            let endPattern = calendar.component(.month, from: first) == calendar.component(.month, from: last)
                ? "d, yyyy" : "MMM d, yyyy"
            return date(first, "MMM d") + "–" + date(last, endPattern) + " UTC"
        }
        return date(first, "MMM d, yyyy") + "–" + date(last, "MMM d, yyyy") + " UTC"
    }
}

/// Reader-facing copy only. Canonical reports, facts, source references and dates are untouched.
enum MatchupOutlookPresentation {
    struct Section: Identifiable {
        let id, title, takeaway, uncertainty: String
        let evidenceLines: [String]
        let comparison: String?
    }

    struct Starter: Identifiable, Equatable {
        let id, team, name, status: String
    }

    static func starters(_ designations: RichContextJSON, event: SharedMatchupEvent) -> [Starter] {
        let kind = ["MLB": "starting_pitcher", "NHL": "starting_goalie", "NBA": "starting_lineup", "NFL": "starting_lineup"][event.sport]
        return designations.array.compactMap { row -> Starter? in
            let state = row["state"].string, name = row["subject"]["name"].string
            guard row["kind"].string == kind, row["eventID"].string == event.eventID,
                  [event.homeTeam, event.awayTeam].contains(row["team"].string),
                  ["probable", "confirmed"].contains(state), !name.isEmpty else { return nil }
            return Starter(id: row["id"].string, team: row["team"].string, name: name,
                status: state == "confirmed" ? "Confirmed" : "Probable")
        }
    }

    static func notes(for result: MatchupNarrativeV4.Result) -> [String] {
        // Unavailable topics stay in missingContext; only a qualification of displayed facts is useful here.
        let used = Set(result.claims.flatMap(\.factIDs))
        return result.context["facts"].array.contains(where: { used.contains($0["id"].string) && $0["phase"].string == "preseason" })
            ? ["These are preseason results; lineups and minutes can change."] : []
    }

    /// Keep cited role premises visible even when outside the general six-row injury view.
    static func injuryObservations(_ result: MatchupNarrativeV4.Result) -> [RichContextJSON] {
        var seen = Set<String>()
        return (result.roleStatusEvidence.array + result.injuryContext["observations"].array).filter {
            seen.insert($0["claimID"].string).inserted
        }
    }

    static func roleLimitations(_ result: MatchupNarrativeV4.Result) -> [String] {
        guard result.injuryRoleContext != .null else { return [] }
        if result.injuryContext["state"].string == "stale" {
            return ["Current player availability could not be verified; earlier reports are out of date."]
        }
        guard !result.injuryRoleGaps.array.isEmpty else { return [] }
        return ["Some player updates cannot support a current role or replacement assessment; their matchup impact is unknown."]
    }

    static func supportingInsights(_ insights: [MatchupInsight]) -> [MatchupInsight] {
        // The existing ordering prioritizes a useful comparison for each team.
        Array(insights.filter(\.hasPairedEvidence).prefix(2))
    }

    static func injuryText(_ row: RichContextJSON, context: RichContextJSON) -> String {
        let team = context["teams"].array.first { $0["id"] == row["teamID"] }?["name"].string ?? ""
        let identity = (team.isEmpty ? "" : team + " · ") + row["playerName"].string + ": " + row["reportedStatus"].string
        let scope = " · " + row["scope"].string + " report"
        return identity + scope + (row["eventStatusConfirmed"] == .bool(true) ? " for this game" : "; game status unconfirmed")
    }

    static func text(for claim: MatchupNarrativeV4.Claim, sport: String) -> String {
        var text = claim.text
        for sample in claim.samples.array where claim.factIDs.contains(sample["factID"].string) {
            // These exact phrases come from the validated v4 renderer, not arbitrary prose.
            let count = sample["sample"]["count"].number ?? "0"
            let unit = sample["sample"]["unit"].string
            let season = MatchupNarrativeV4.season(sample, sport)
            let phase = sample["phase"].string
            for span in [
                " across " + MatchupNarrativeV4.sample(sample, sport),
                " across \(count) \(unit) in that same window",
                " across \(count) \(unit) in that window",
                " across \(count) games in the same window",
                " in the same \(season) \(phase) window"
            ] { text = text.replacingOccurrences(of: span, with: "") }
        }
        if claim.kind == "limited_sample", let minimum = MatchupNarrativeV4.minimum[sport] {
            for pair in MatchupNarrativeV4.pairs {
                let outcome = pair.1[4]
                text = text.replacingOccurrences(
                    of: "This does not meet the \(minimum)-game rule for a directional \(outcome) expectation; the recorded comparison is descriptive only.",
                    with: "There is too little history to suggest a direction for \(outcome); these rates describe past games only.")
            }
        }
        return text.replacingOccurrences(of: "The available samples do not establish a comparative tempo direction.",
            with: "The available history does not show a consistent tempo direction.")
    }

    static func uncertainty(_ value: String) -> String {
        value == "Conditional sample-based interpretation, not a calibrated prediction. Availability and game conditions can change the pattern."
            ? "These trends are not a prediction. Availability and game conditions can change the outcome." : value
    }

    static func sections(_ response: SharedMatchupReportResponse) -> [Section] {
        SharedMatchupPresentationCopy.sections(response).map { original in
            let facts = (response.snapshot?.facts ?? []).filter { original.factIDs.contains($0.id) }
            var lines = original.evidenceLines
            var comparison = original.comparisonText
            var takeaway = original.takeaway
            // Only the known display projection has removable metadata. Unknown copy stays intact.
            if original.mapped {
                for team in [response.event.awayTeam, response.event.homeTeam] {
                    let counts = Set(facts.filter { $0.team == team && $0.role != "reference" }.map(\.sample))
                        .sorted().map(String.init).joined(separator: "/")
                    lines = lines.map { $0.replacingOccurrences(of: " (\(counts) recorded games).", with: ".") }
                }
                if let reference = facts.first(where: { $0.role == "reference" }) {
                    comparison = comparison?
                        .replacingOccurrences(of: "\(response.event.sport) sample average: ", with: "Recorded \(response.event.sport) average: ")
                        .replacingOccurrences(of: " (\(reference.sample) team-game observations).", with: ".")
                        .replacingOccurrences(of: "; \(reference.sample) team-game observations.", with: ".")
                }
                if original.id == "game:limited-sample" {
                    takeaway = "There is too little history to draw a scoring expectation from these results."
                } else {
                    takeaway = takeaway.replacingOccurrences(of: "this sample", with: "these results")
                }
            }
            return Section(id: original.id, title: original.title, takeaway: takeaway,
                uncertainty: original.uncertainty, evidenceLines: lines, comparison: comparison)
        }
    }

    static func opponentCaution(count: Int, incomplete: Bool) -> String {
        let limitation = count > 0 && count < 5 ? "There is little history against this opponent. " : ""
        let coverage = incomplete ? "Some past seasons could not be loaded. " : ""
        return limitation + coverage + "Past results compared with this line are not settled bet results or a prediction."
    }
}
