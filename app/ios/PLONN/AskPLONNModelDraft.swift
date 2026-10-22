import Foundation

/// Untrusted AI configuration only. Saving and testing use the existing model
/// builder's identity, capability and historical-data checks after user review.
struct AskPLONNModelDraft: Codable, Equatable, Sendable {
    var league: String? = nil
    var target: String? = nil
    var subject: String? = nil
    var stat: String? = nil
    var line: Double? = nil
    var side: String? = nil
    var opponent: String? = nil
    var scoringPreset: String? = nil
    var firstBasketMethod: String? = nil

    static let wireKeys: Set<String> = ["league", "target", "subject", "stat", "line", "side", "opponent", "scoringPreset", "firstBasketMethod"]
    enum CodingKeys: String, CodingKey { case league, target, subject, stat, line, side, opponent, scoringPreset, firstBasketMethod }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(league, forKey: .league); try c.encode(target, forKey: .target)
        try c.encode(subject, forKey: .subject); try c.encode(stat, forKey: .stat)
        try c.encode(line, forKey: .line); try c.encode(side, forKey: .side)
        try c.encode(opponent, forKey: .opponent); try c.encode(scoringPreset, forKey: .scoringPreset)
        try c.encode(firstBasketMethod, forKey: .firstBasketMethod)
    }
    var isValid: Bool {
        if let league, !["NFL", "NBA", "MLB", "NHL"].contains(league) { return false }
        if let target, !["Player", "Team"].contains(target) { return false }
        if let side, !["Over", "Under"].contains(side) { return false }
        for value in [subject, opponent].compactMap({ $0 }) {
            guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, value.unicodeScalars.count <= 100,
                  value.unicodeScalars.allSatisfy({ $0.value >= 32 }),
                  value.rangeOfCharacter(from: CharacterSet(charactersIn: "<>{}[]\\/:")) == nil else { return false }
        }
        if let stat {
            guard let league, let target, stat != "Custom stat",
                  LabPropCatalog.options(league: league, target: target).contains(stat) else { return false }
        }
        if let line {
            guard stat != nil, LabServiceContract.validLine(line, target: target, stat: stat) else { return false }
        }
        if let scoringPreset {
            guard LabPropCatalog.isFantasy(stat ?? ""), ["draftkings_classic", "fanduel_classic"].contains(scoringPreset) else { return false }
        }
        if let firstBasketMethod {
            guard stat == "Method of first basket", ["two_pointer", "three_pointer", "free_throw"].contains(firstBasketMethod) else { return false }
        }
        return true
    }
    var effectiveLine: Double? {
        if target == "Team", stat == "Moneyline" { return 0 }
        if target == "Player", LabPropCatalog.isYesNo(stat ?? "") { return 0.5 }
        return line
    }
    var effectiveSide: String? {
        if target == "Team", ["Moneyline", "Spread"].contains(stat ?? "") { return "Over" }
        return side
    }
    var missingFields: [String] {
        var missing: [String] = []
        if league == nil { missing.append("sport (NFL, NBA, MLB or NHL)") }
        if target == nil { missing.append("player or team") }
        if subject == nil { missing.append("player or team name") }
        if stat == nil { missing.append("stat or game outcome") }
        if let stat {
            if effectiveLine == nil { missing.append("line to test") }
            if effectiveSide == nil { missing.append(LabPropCatalog.isYesNo(stat) ? "yes or no" : "over or under") }
            if LabPropCatalog.isFantasy(stat), scoringPreset == nil { missing.append("DraftKings or FanDuel scoring") }
            if stat == "Method of first basket", firstBasketMethod == nil { missing.append("two-pointer, three-pointer or free throw") }
        }
        return missing
    }
    var canReview: Bool { isValid && missingFields.isEmpty }
    var guidance: String {
        if missingFields.isEmpty {
            return "Your draft is ready to review. Choose a completed game in the builder to test it. Nothing has been saved or tested yet."
        }
        return "Let’s finish your draft. Tell me the " + missingFields.joined(separator: ", ") + "."
    }
    var title: String { [subject, stat.map(LabPropCatalog.displayName)].compactMap({ $0 }).joined(separator: " · ") }
    var thresholdDescription: String? {
        guard let stat else { return nil }
        if target == "Team", stat == "Moneyline" { return "To win" }
        if target == "Player", LabPropCatalog.isYesNo(stat) { return effectiveSide.map { $0 == "Under" ? "No" : "Yes" } }
        guard let line = effectiveLine else { return nil }
        if target == "Team", stat == "Spread" { return "Cover " + String(format: "%+g", line) }
        return (effectiveSide ?? "Choose direction") + " " + String(format: "%g", line)
    }
    static func isCreationRequest(_ question: String) -> Bool {
        let text = question.lowercased()
        // Explicit creation commands only; ordinary questions about models stay in chat.
        return text.range(of: #"\b(build|create|make|set up|setup|draft)\b.{0,90}\bmodel\b"#, options: .regularExpression) != nil
    }
}
