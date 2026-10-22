import Foundation

enum AskPLONNIntent: String, CaseIterable, Codable, Sendable {
    case recentForm, opponentHistory, comparePrices, explainModel, generalChat
}
enum AskPLONNResearchPurpose: String, Codable, Sendable { case matchupInsights, modelCreation }
enum AskPLONNAvailability: String, Codable, Sendable { case available, stale, unavailable }
enum AskPLONNAnswerMode: String, Codable, Sendable { case localResearch, provider, generalExplanation }
enum AskPLONNGeneralResponseKind: String, CaseIterable, Codable, Sendable { case explanation, needs_research, unsupported }
struct AskPLONNEvidence: Identifiable, Codable, Sendable, Equatable {
    let id: String
    let kind: AskPLONNIntent
    let title: String
    let text: String
    let source: String
    var capturedAt: Date? = nil
    var sampleSize: Int? = nil
    var availability: AskPLONNAvailability = .available
    var sourceURLs: [String] = []
}
struct AskPLONNAnswer: Codable, Sendable {
    let text: String
    let evidence: [AskPLONNEvidence]
    var mode: AskPLONNAnswerMode = .localResearch
    var providerLabel: String? = nil
    var requiresResearchContext = false
    var matchupSummary: String? = nil
    var modelDraft: AskPLONNModelDraft? = nil
}
extension AskPLONNAnswer {
    private enum CodingKeys: String, CodingKey { case text, evidence, mode, providerLabel, requiresResearchContext, matchupSummary, modelDraft }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        text = try values.decode(String.self, forKey: .text)
        evidence = try values.decode([AskPLONNEvidence].self, forKey: .evidence)
        mode = try values.decodeIfPresent(AskPLONNAnswerMode.self, forKey: .mode) ?? .localResearch
        providerLabel = try values.decodeIfPresent(String.self, forKey: .providerLabel)
        requiresResearchContext = try values.decodeIfPresent(Bool.self, forKey: .requiresResearchContext) ?? false
        matchupSummary = try values.decodeIfPresent(String.self, forKey: .matchupSummary)
        modelDraft = try values.decodeIfPresent(AskPLONNModelDraft.self, forKey: .modelDraft)
    }
}
/// Research retains questions/topics only; current evidence supplies its facts.
/// General education may also retain its prior responses as conversational memory.
struct AskPLONNQuestionContext: Codable, Sendable, Equatable {
    let question: String
    let intent: AskPLONNIntent
    /// General education memory only; never included with research evidence.
    var generalResponse: String? = nil
}
struct AskPLONNRequest: Codable, Sendable {
    let question: String
    let intent: AskPLONNIntent
    let evidence: [AskPLONNEvidence]
    var ownerID: String? = nil
    var priorTurns: [AskPLONNQuestionContext]? = nil
    var researchPurpose: AskPLONNResearchPurpose? = nil
    var matchupTeams: [String]? = nil
    var modelDraft: AskPLONNModelDraft? = nil
}
protocol AskPLONNProvider: Sendable {
    var displayName: String { get }
    func answer(_ request: AskPLONNRequest) async throws -> AskPLONNAnswer
}
extension AskPLONNProvider { var displayName: String { "Connected model" } }
enum AskPLONNError: Error, Equatable { case questionTooLong, invalidEvidence, invalidHistory, unsupportedAnswer, unsupportedQuestion, providerUnavailable, superseded }
struct UnconfiguredAskPLONNProvider: AskPLONNProvider {
    func answer(_ request: AskPLONNRequest) async throws -> AskPLONNAnswer { throw AskPLONNError.providerUnavailable }
}

enum AskPLONNCore {
    static let maximumQuestionLength = 1000
    static let maximumEvidenceCount = 20
    static let maximumPriorTurnCount = 6
    static let maximumPriorQuestionCharacters = 3000
    static let maximumConversationTurnCount = 20
    static let maximumGeneralAnswerLength = 2000
    static let maximumMatchupSummaryLength = 700

    private static func matches(_ text: String, _ pattern: String) -> Bool {
        text.range(of: pattern, options: .regularExpression) != nil
    }

    private static func isGeneralConcept(_ question: String) -> Bool {
        let text = question.lowercased()
        return matches(text, #"\b(what (is|are|does)|how (does|do|can)|explain|define|help me understand)\b"#) &&
            matches(text, #"\b(implied probability|moneylines?|overfitting|overfit|calibration|brier|variance|sample size|regression|correlation|expected value|betting edge|edge|vig|overround|decimal odds|american odds|probabilities|probability|models?|averages?|analytics|random forest|standard deviation|neural network|expected goals|poisson distribution|possessions?|sacks?|rebounds?|assists?)\b"#)
    }

    static func requiresResearchForQuestion(_ question: String) -> Bool {
        let text = question.lowercased()
        if matches(text, #"\b(best|highest|better)\b.{0,30}\b(sportsbooks?|books?|prices?|odds|payouts?)\b|\b(my|this|our|recorded|live)\s+((betting|model)\s+)?edge\b|\bedge\s+(here|for|on|in this)\b|\b(show|calculate|compare|give)\b.{0,40}\bedge\b|\bwhat(?:[’']s| is)\s+the\s+((betting|model)\s+)?edge\b"#) { return true }
        let statistics = #"stats?|statistics|results?|scores?|totals?|averages?|points?|hits?|home runs?|rebounds?|assists?|possessions?|sacks?|goals?|runs?|yards?|touchdowns?|interceptions?|steals?|blocks?|saves?|receptions?|catches|strikeouts?|shots?|batting average|shooting percentage|passer rating|wins?|losses"#
        // Recorded-stat questions stay data-dependent regardless of how a name
        // is spelled or capitalized. Concept definitions have no observed cue.
        if matches(text, #"\bhow (many|much)\b.{0,60}\b("# + statistics + #")\b.{0,100}\b(did|does|has|have|had|was|were)\b"#) ||
            matches(text, #"\b(?:his|her|their|[a-z]+[’']s)\s+(?:[a-z]+\s+){0,2}(?:"# + statistics + #")\b"#) ||
            matches(text, #"\b(show|give|list|compare|summarize)\b.{0,80}\b("# + statistics + #")\b"#) ||
            matches(text, #"\bwhat (are|were|was|is)\s+(?:the\s+)?("# + statistics + #")\s+(for|of)\b"#) { return true }
        if matches(text, #"\b(my|this|our|saved)\s+(player|team|opponent|matchup|recipe|configuration|setup|result)\b|\b(my|this|our|saved)\s+model\s+(setup|configuration|result|probability|features|settings)\b|\b(current|latest|today|tonight|tomorrow|yesterday|recent|last\s+\d+|news|injur(?:y|ies))\b|\b(recorded|live)\s+(odds|prices?|stats?|results?)\b|\b(who (won|wins)|what (were|was) .{0,40}(score|result)|how has .{0,60}(played|performed|scored)|against this opponent)\b"#) { return true }
        // A named person is context, not permission to invent their statistics.
        // Exclude common title-cased educational terms from this conservative cue.
        let educationalTerms = ["implied probability", "expected value", "betting edge", "model edge", "sample size", "american odds", "decimal odds", "brier score", "machine learning", "linear regression", "logistic regression", "batting average", "random forest", "standard deviation", "neural network", "expected goals", "poisson distribution", "what is", "what are", "how can", "how does"]
        let namedText = question.replacingOccurrences(of: #"^(?:what (?:is|are|does)|how (?:does|do|can)|explain|define|help me understand)\s+"#, with: "", options: [.regularExpression, .caseInsensitive])
        let expression = try! NSRegularExpression(pattern: #"\b[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\b"#)
        return expression.matches(in: namedText, range: NSRange(namedText.startIndex..., in: namedText)).contains { match in
            guard let range = Range(match.range, in: namedText) else { return false }
            return !educationalTerms.contains(String(namedText[range]).lowercased())
        }
    }

    static func chatIntentForQuestion(_ question: String, fallback: AskPLONNIntent, hasResearchContext: Bool) -> AskPLONNIntent {
        guard hasResearchContext else { return .generalChat }
        let needsResearch = requiresResearchForQuestion(question)
        if isGeneralConcept(question) && !needsResearch { return .generalChat }
        let researchFallback: AskPLONNIntent = needsResearch && fallback == .generalChat ? .recentForm : fallback
        return intentForQuestion(question, fallback: researchFallback)
    }

    static func isUnsupportedGeneralQuestion(_ question: String) -> Bool {
        matches(question.lowercased(), #"\b(guaranteed?|sure thing|winning bet|betting (pick|recommendation)|pick me|should i (bet|wager)|which .{0,20}(bet|wager)|predict .{0,40}(win|score|hit)|will .{0,40}(win|score|hit)|(project|predict|estimate) .{0,80}(team averages?|player.{0,20}(score|points|result)))\b"#)
    }

    static func generalResearchGuidance() -> AskPLONNAnswer {
        AskPLONNAnswer(text: "I don’t have verified data for that here. You can optionally attach research context for a player or saved model, or we can discuss the concept generally.", evidence: [], mode: .generalExplanation, requiresResearchContext: true)
    }

    static func generalUnsupportedGuidance() -> AskPLONNAnswer {
        AskPLONNAnswer(text: "I can explain odds and uncertainty, but I can’t recommend bets, guarantee outcomes, or project a player's result from team averages. What concept would you like to explore?", evidence: [], mode: .generalExplanation)
    }

    static func generalFallback(_ request: AskPLONNRequest) -> AskPLONNAnswer {
        let disclosure = "AI chat is unavailable. "
        if isUnsupportedGeneralQuestion(request.question) {
            return AskPLONNAnswer(text: disclosure + "I can explain how odds and uncertainty work, but I can’t recommend bets, guarantee outcomes, or predict a player's result from team averages. What concept would you like to explore?", evidence: [], mode: .generalExplanation)
        }
        if requiresResearchForQuestion(request.question) {
            let guidance = generalResearchGuidance()
            return AskPLONNAnswer(text: disclosure + guidance.text, evidence: [], mode: .generalExplanation, requiresResearchContext: true)
        }
        let text = request.question.lowercased()
        let topic = matches(text, #"\b(that|more|second point)\b"#) ? (request.priorTurns?.last?.question.lowercased() ?? text) : text
        let explanation: String
        if topic.contains("implied probability") {
            explanation = "Implied probability converts a quoted price into a break-even percentage. For a hypothetical decimal price of 2.00, 1 ÷ 2.00 = 50%. This is an illustrative example, not current odds or a forecast. Bookmaker margin can make the probabilities across all outcomes add up to more than 100%."
        } else if topic.contains("moneyline") {
            explanation = "A moneyline expresses a price for a team or player to win outright. In a hypothetical example, +150 means a 100-unit stake would return 150 units of profit; −150 means staking 150 units for 100 units of profit. These examples are illustrative, not current prices or recommendations."
        } else if topic.contains("overfit") {
            explanation = "A model overfits when it learns noise or quirks in its training sample instead of patterns that generalize. It may look strong on training data and weaker on unseen examples. Separate validation data, simpler models, regularization, and checks for data leakage help assess this."
        } else if topic.contains("calibration") {
            explanation = "Calibration describes whether predicted probabilities match observed frequencies across many cases. A model’s probability is not proof of calibration; you need outcomes from a suitable held-out sample to evaluate it."
        } else if matches(topic, #"\b(edge|betting edge)\b"#) {
            explanation = "A model edge compares a model’s probability with the break-even probability implied by a price. In a hypothetical no-push example, a 55% model probability and a decimal price of 2.00 imply an edge of 55% − 50% = +5 percentage points. The price includes bookmaker margin; model validation and matching settlement rules matter. This is an illustration, not a current edge or a guarantee."
        } else {
            explanation = "We can talk about sports and analytics without choosing a player or model. I can explain odds, probability, model validation, or how to interpret a sample. What would you like to understand?"
        }
        return AskPLONNAnswer(text: disclosure + explanation, evidence: [], mode: .generalExplanation)
    }

    /// Resolves an explicitly named topic only. The selected player, stat and
    /// matchup remain the caller's context; this never invents a new context.
    static func intentForQuestion(_ question: String, fallback: AskPLONNIntent) -> AskPLONNIntent {
        let text = question.lowercased()
        func matches(_ pattern: String) -> Bool { text.range(of: pattern, options: .regularExpression) != nil }
        guard question.count <= maximumQuestionLength,
              !matches(#"\b(predict|forecast|guaranteed|guarantee|lock|sure thing)\b|\b(will|going to)\b.{0,60}\b(win|score|hit|cover|beat)\b|\b(should i|recommend|pick me)\b.{0,40}\b(bet|wager|pick)\b"#) else { return fallback }
        var topics: [AskPLONNIntent] = []
        let hasEdge = matches(#"\bedge\b"#)
        if hasEdge || matches(#"\b(odds|prices?|sportsbooks?|payouts?)\b|\b(compare|best|better|highest)\b.{0,30}\bbooks?\b"#) { topics.append(.comparePrices) }
        if !hasEdge && matches(#"\b(models?|configuration|recipes?|calibration|features|training)\b|\bsaved (setup|result)\b"#) { topics.append(.explainModel) }
        if matches(#"\b(opponents?|versus|vs)\b|\bagainst\b(?!\s+(the |this |a )?(line|threshold))"#) { topics.append(.opponentHistory) }
        if topics.count == 1 { return topics[0] }
        if topics.isEmpty, matches(#"\b(recent|recently|form|history|averaged|average)\b|\blast\s+\d+\s+(games?|appearances?)\b|\b(recorded|earlier) games?\b"#) { return .recentForm }
        return fallback
    }

    static func boundedRequest(_ request: AskPLONNRequest) throws -> AskPLONNRequest {
        guard request.question.count <= maximumQuestionLength else { throw AskPLONNError.questionTooLong }
        if request.researchPurpose == .modelCreation {
            guard request.intent == .generalChat, request.evidence.isEmpty, request.matchupTeams == nil,
                  request.modelDraft?.isValid != false else { throw AskPLONNError.invalidEvidence }
        } else if request.modelDraft != nil { throw AskPLONNError.invalidEvidence }
        let priorTurns = request.priorTurns ?? []
        if request.researchPurpose == .matchupInsights {
            guard request.intent == .recentForm, priorTurns.isEmpty,
                  let teams = request.matchupTeams, teams.count == 2,
                  teams.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.unicodeScalars.count <= 100 && $0.unicodeScalars.allSatisfy({ $0.value >= 32 }) }),
                  Set(teams.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }).count == 2,
                  (1...4).contains(request.evidence.count),
                  request.evidence.allSatisfy({ $0.kind == .recentForm && $0.availability == .available &&
                      !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !$0.source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
            else { throw AskPLONNError.invalidEvidence }
        } else if request.matchupTeams != nil { throw AskPLONNError.invalidEvidence }
        guard priorTurns.count <= maximumPriorTurnCount,
              priorTurns.allSatisfy({ !$0.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.question.count <= maximumQuestionLength &&
                  ($0.generalResponse == nil || (request.intent == .generalChat && $0.intent == .generalChat && validGeneralText($0.generalResponse!))) &&
                  (request.intent != .generalChat || $0.intent == .generalChat) }),
              priorTurns.reduce(0, { $0 + $1.question.count + ($1.generalResponse?.count ?? 0) }) <= maximumPriorQuestionCharacters else { throw AskPLONNError.invalidHistory }
        guard request.intent != .generalChat || request.evidence.isEmpty else { throw AskPLONNError.invalidEvidence }
        guard request.evidence.count <= maximumEvidenceCount,
              Set(request.evidence.map(\.id)).count == request.evidence.count,
              request.evidence.allSatisfy({
                  !$0.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.id.count <= 200 &&
                  $0.title.count <= 200 && $0.text.count <= 2000 && $0.source.count <= 500 &&
                  $0.sourceURLs.count <= 12 && Set($0.sourceURLs).count == $0.sourceURLs.count &&
                  $0.sourceURLs.allSatisfy(validSourceURL) &&
                  ($0.sampleSize == nil || $0.sampleSize! >= 0) &&
                  ($0.capturedAt == nil || $0.capturedAt!.timeIntervalSince1970.isFinite)
              }) else { throw AskPLONNError.invalidEvidence }
        return request
    }

    static func validSourceURL(_ value: String) -> Bool {
        guard value.count <= 2000, let url = URLComponents(string: value),
              url.scheme == "https", url.user == nil, url.password == nil, url.port == nil, url.fragment == nil,
              let host = url.host?.lowercased(),
              ["statsapi.mlb.com", "site.api.espn.com", "site.web.api.espn.com", "sports.core.api.espn.com"].contains(host) else { return false }
        return !url.path.isEmpty
    }

    static func localAnswer(_ request: AskPLONNRequest) -> AskPLONNAnswer {
        guard (try? boundedRequest(request)) != nil else {
            if request.intent == .generalChat {
                return AskPLONNAnswer(text: "The request exceeds the supported limits or contains invalid context.", evidence: [], mode: .generalExplanation)
            }
            return AskPLONNAnswer(text: "Verified data summary\nThe request exceeds the supported limits or contains invalid evidence.", evidence: [])
        }
        if request.intent == .generalChat { return generalFallback(request) }
        let selected = request.evidence.filter { $0.kind == request.intent }
        let available = selected.filter { $0.availability == .available && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !$0.source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        var paragraphs = ["Verified data summary"]
        paragraphs += available.map(\.text)
        if available.isEmpty { paragraphs.append("Verified data for this topic is unavailable.") }
        if selected.contains(where: { $0.availability == .stale }) { paragraphs.append("Some evidence is stale and has been excluded from this summary.") }
        if selected.contains(where: { $0.availability == .unavailable }) { paragraphs.append("Some requested evidence is unavailable.") }
        return AskPLONNAnswer(text: paragraphs.joined(separator: "\n\n"), evidence: selected)
    }

    /// Fact paragraphs always match the supplied observations exactly. A requested
    /// matchup summary is separate, bounded model commentary without new numbers.
    /// General education is separately labeled and never carries evidence.
    static func validateProviderAnswer(_ answer: AskPLONNAnswer, for request: AskPLONNRequest) throws -> AskPLONNAnswer {
        _ = try boundedRequest(request)
        if request.researchPurpose == .modelCreation {
            guard let draft = answer.modelDraft, draft.isValid, answer.text == draft.guidance,
                  answer.mode == .generalExplanation, answer.evidence.isEmpty, answer.matchupSummary == nil,
                  !answer.requiresResearchContext, answer.providerLabel?.count ?? 0 <= 200 else { throw AskPLONNError.unsupportedAnswer }
            return answer
        }
        guard answer.modelDraft == nil else { throw AskPLONNError.unsupportedAnswer }
        if request.researchPurpose == nil, answer.matchupSummary != nil { throw AskPLONNError.unsupportedAnswer }
        if request.intent == .generalChat {
            guard answer.mode == .generalExplanation, answer.evidence.isEmpty,
                  validGeneralText(answer.text), answer.providerLabel?.count ?? 0 <= 200 else { throw AskPLONNError.unsupportedAnswer }
            if isUnsupportedGeneralQuestion(request.question) {
                return AskPLONNAnswer(text: generalUnsupportedGuidance().text, evidence: [], mode: .generalExplanation, providerLabel: answer.providerLabel)
            }
            if answer.requiresResearchContext || requiresResearchForQuestion(request.question) {
                return AskPLONNAnswer(text: generalResearchGuidance().text, evidence: [], mode: .generalExplanation, providerLabel: answer.providerLabel, requiresResearchContext: true)
            }
            return answer
        }
        guard answer.mode != .generalExplanation, !answer.requiresResearchContext else { throw AskPLONNError.unsupportedAnswer }
        guard !answer.evidence.isEmpty, answer.evidence.count <= maximumEvidenceCount,
              Set(answer.evidence.map(\.id)).count == answer.evidence.count,
              answer.evidence.allSatisfy({ item in
                  item.kind == request.intent && item.availability == .available &&
                  !item.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
                  !item.source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && request.evidence.contains(item)
              }), answer.providerLabel?.count ?? 0 <= 200 else { throw AskPLONNError.unsupportedAnswer }
        let facts = answer.evidence.map(\.text).joined(separator: "\n\n")
        guard answer.text == facts || answer.text == "Verified data summary\n\n" + facts else { throw AskPLONNError.unsupportedAnswer }
        if request.researchPurpose == .matchupInsights {
            guard answer.mode == .provider, answer.evidence.count <= 3,
                  let summary = answer.matchupSummary, let teams = request.matchupTeams,
                  validMatchupSummary(summary, evidence: request.evidence, teams: teams)
            else { throw AskPLONNError.unsupportedAnswer }
        }
        return AskPLONNAnswer(text: "Verified data summary\n\n" + facts, evidence: answer.evidence, mode: .provider,
                             providerLabel: answer.providerLabel, matchupSummary: answer.matchupSummary)
    }

    /// Exact rates and sample counts are shown in the observation cards. Model
    /// commentary cannot add quantities, outside names, prices or forecasts.
    static func validMatchupSummary(_ text: String, evidence: [AskPLONNEvidence], teams: [String]) -> Bool {
        let compatible = text.precomposedStringWithCompatibilityMapping
        // Quantities are permitted only inside a complete, unchanged evidence
        // sentence with its named team. Never authorize a bag of numeric values.
        let sentenceEnd = try! NSRegularExpression(pattern: #"[.!?](?=\s|$)"#)
        func sentences(_ value: String) -> [String] {
            let source = value as NSString
            var start = 0
            var result: [String] = []
            for end in sentenceEnd.matches(in: value, range: NSRange(location: 0, length: source.length)) {
                result.append(source.substring(with: NSRange(location: start, length: end.range.location - start)).trimmingCharacters(in: .whitespacesAndNewlines))
                start = NSMaxRange(end.range)
            }
            result.append(source.substring(from: start).trimmingCharacters(in: .whitespacesAndNewlines))
            return result.map { sentence in
                var value = sentence
                while let last = value.last, ".!?".contains(last) { value.removeLast() }
                return value
            }.filter { !$0.isEmpty }
        }
        func qualifies(_ clause: String) -> Bool {
            guard clause.contains(where: \.isNumber) else { return false }
            if teams.contains(where: { matches(clause, #"(?<!\w)"# + NSRegularExpression.escapedPattern(for: $0) + #"(?!\w)"#) }) { return true }
            // A complete season-provenance sentence may contain its single year.
            let years = try! NSRegularExpression(pattern: #"(?<!\w)[12][0-9]{3}(?!\w)"#)
            let found = years.matches(in: clause, range: NSRange(clause.startIndex..., in: clause))
            guard found.count == 1 else { return false }
            let remainder = years.stringByReplacingMatches(in: clause, range: NSRange(clause.startIndex..., in: clause), withTemplate: "year")
            return !remainder.contains(where: \.isNumber) && ["season", "team", "statistics"].allSatisfy { matches(clause.lowercased(), "\\b" + $0 + "\\b") }
        }
        let clauses = Set(evidence.flatMap { sentences($0.text) }.filter(qualifies))
        let residual = sentences(text).map { sentence -> String in
            if clauses.contains(sentence) { return "observation" }
            let joined = sentence.components(separatedBy: " whereas ")
            if joined.count > 1 && joined.allSatisfy({ clauses.contains($0) }) { return "observation" }
            return sentence
        }.joined(separator: ". ")
        // Official names such as Philadelphia 76ers are identity, not a newly
        // generated quantity. Only the exact supplied names get this exception.
        let names = teams.sorted { $0.count > $1.count }.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
        let namesExpression = try! NSRegularExpression(pattern: #"(?<!\w)(?:"# + names + #")(?!\w)"#, options: .caseInsensitive)
        let quantities = namesExpression.stringByReplacingMatches(in: residual, range: NSRange(residual.startIndex..., in: residual), withTemplate: "team")
        guard text.unicodeScalars.count <= maximumMatchupSummaryLength, text.utf8.count <= 2800,
              validGeneralText(compatible), !quantities.contains(where: \.isNumber),
              text.unicodeScalars.allSatisfy({ $0.value >= 32 && !CharacterSet.controlCharacters.contains($0) &&
                  !CharacterSet.illegalCharacters.contains($0) && !CharacterSet.symbols.contains($0) }),
              !matches(compatible, #"[%‰‱]|\[[^\]]*\]"#) else { return false }
        let limitationNormalized = compatible.lowercased().replacingOccurrences(of: #"\bnot (?:an? |individual )?(?:player |individual player )?(?:forecast|projection|prediction)s?\b"#, with: "", options: .regularExpression)
        // Descriptive sample limitations are not future-outcome claims. Keep
        // modals forbidden everywhere except this bounded comparison phrase.
        let normalized = limitationNormalized.replacingOccurrences(
            of: #"\b(?:can|could|would) (?:only )?support (?:a )?(?:cautious |limited |descriptive )?comparison\b"#,
            with: "support comparison", options: .regularExpression)
        guard !matches(normalized, #"\b(shall|favor(?:ed|ite|ites|able)|favour(?:ed|ite|ites|able)|calibrat\w*|gambl\w*|moneyline|parlay\w*|spread\w*|win|wins|winning|will|would|could|can|expect(?:ed)?|likely|forecast\w*|predict\w*|project\w*|guarantee\w*|lock|sure thing|bet\w*|wager\w*|pick\w*|stake\w*|payout\w*|odds|price\w*|sportsbook\w*|edge|probabilit\w*|percent\w*|player\w*|skater\w*|goalie\w*|goaltender\w*|roster\w*|lineup\w*|injur\w*|power play\w*|penalty kill\w*|xg|corsi|fenwick|pdo|special teams|save percentage\w*|shooting percentage\w*|fatigue|weather|travel|rest|zero|one|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b"#) else { return false }
        // Entity/content validation is case-independent below. Capitalization at
        // a sentence boundary must not impose a second, conflicting vocabulary.
        let supportedWords = evidence.map { $0.title.lowercased() + " " + $0.text.lowercased() }.joined(separator: " ")
        for metric in ["goals?", "shots?", "assists?", "rebounds?", "rushing", "passing", "slugging", "batting", "threes"] {
            let pattern = "\\b" + metric + "\\b"
            if matches(normalized, pattern), !matches(supportedWords, pattern) { return false }
        }
        // Names and metrics remain content even when lowercased. Every content
        // word must come from the observations themselves; source labels and
        // URLs cannot authorize prose. This is a conservative lexical boundary,
        // not proof that arbitrary model prose is semantically grounded.
        guard !matches(normalized, #"\b(superior|inferior|best|worst|better|worse|stronger|weaker|dominant|dominat\w*|rank\w*|elite|advantage\w*|outperform\w*|outcome\w*|caus\w*|ensure\w*|prove\w*)\b"#) else { return false }
        let neutral = "a an and at against both because but by context conversely despite different each for from however historical history in is are was were it its meanwhile meets neither of on only opposing opponents overall patterns per recent regarding rather rates sample samples scoring season short similar small supplied taken than that the their these they this those to versus volume what whereas while with without yet have has had lower higher fewer more low high average averages averaged number size few completed games game based cautious offensive defensive offense defense s team teams"
        func lexicalWords(_ value: String) -> [String] {
            value.lowercased().split(whereSeparator: { !$0.isLetter }).map(String.init)
        }
        // A bounded morphology contract, not sentence-specific exceptions. These
        // families express comparison, provenance and sample limitations only.
        // They never authorize a new metric, entity, quantity or forecast.
        let families = [
            "compare compares compared comparing comparison comparisons comparable",
            "contrast contrasts contrasted contrasting",
            "describe describes described describing descriptive description descriptions",
            "observe observes observed observing observation observations",
            "record records recorded recording",
            "limit limits limited limiting limitation limitations",
            "produce produces produced producing production",
            "allow allows allowed allowing allowance allowances",
            "support supports supported supporting"
        ]
        var canonical: [String: String] = [:]
        let metricFamilies = ["goal goals", "shot shots", "assist assists", "rebound rebounds", "yard yards", "point points", "turnover turnovers"]
        for family in families + metricFamilies {
            let forms = family.split(separator: " ").map(String.init)
            for form in forms { canonical[form] = forms[0] }
        }
        func lexicalKey(_ word: String) -> String {
            if let family = canonical[word] { return family }
            return word
        }
        let neutralWords = Set((lexicalWords(neutral) + families.flatMap(lexicalWords)).map(lexicalKey))
        func withoutTeamNames(_ value: String) -> String {
            namesExpression.stringByReplacingMatches(in: value, range: NSRange(value.startIndex..., in: value), withTemplate: "team")
        }
        // A supplied name is one identity, not a bag of reusable words. Removing
        // complete names from both sides rejects invented recombinations and
        // partial names while still allowing exact names with ordinary casing.
        let factWords = Set(lexicalWords(withoutTeamNames(supportedWords)).map(lexicalKey))
        let residualNormalized = residual.precomposedStringWithCompatibilityMapping.lowercased()
            .replacingOccurrences(of: #"\bnot (?:an? |individual )?(?:player |individual player )?(?:forecast|projection|prediction)s?\b"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\b(?:can|could|would) (?:only )?support (?:a )?(?:cautious |limited |descriptive )?comparison\b"#, with: "support comparison", options: .regularExpression)
        guard lexicalWords(withoutTeamNames(residualNormalized)).allSatisfy({ word in
            let key = lexicalKey(word)
            return neutralWords.contains(key) || factWords.contains(key)
        }) else { return false }
        return true
    }

    static func validGeneralText(_ text: String) -> Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && text.count <= maximumGeneralAnswerLength && text.utf8.count <= 8192 &&
            !matches(text.lowercased(), #"\b[a-z][a-z0-9+.-]*://|\bwww\.|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b|\[[^\]]+\]\([^)]*\)|\[\d+\]|\baccording to\b|\b(sources?|citations?|references?)\s*:|verified data summary"#)
    }
}

struct AskPLONNTurn: Sendable {
    let question: String
    let intent: AskPLONNIntent
    let answer: AskPLONNAnswer
}

actor AskPLONNConversation {
    private var ownerID: String?
    private var contextID: String?
    private var generation = UUID()
    private var turns: [AskPLONNTurn] = []

    func reset(ownerID: String?, contextID: String? = nil) {
        self.ownerID = ownerID; self.contextID = contextID; generation = UUID(); turns = []
    }
    func history() -> [AskPLONNTurn] { turns }

    func ask(question: String, intent: AskPLONNIntent, evidence: [AskPLONNEvidence], ownerID: String?, contextID: String? = nil, provider: (any AskPLONNProvider)? = nil) async throws -> AskPLONNAnswer {
        try Task.checkCancellation()
        if self.ownerID != ownerID || self.contextID != contextID { reset(ownerID: ownerID, contextID: contextID) }
        generation = UUID()
        let token = generation
        var priorTurns: [AskPLONNQuestionContext] = []
        var priorCharacters = 0
        let eligibleTurns = intent == .generalChat ? turns.filter { $0.intent == .generalChat && $0.answer.mode == .generalExplanation } : turns
        for turn in eligibleTurns.reversed().prefix(AskPLONNCore.maximumPriorTurnCount) {
            let generalResponse = intent == .generalChat ? turn.answer.text : nil
            let characters = turn.question.count + (generalResponse?.count ?? 0)
            guard priorCharacters + characters <= AskPLONNCore.maximumPriorQuestionCharacters else { break }
            // An empty local question has no conversational topic to retain.
            guard !turn.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            priorTurns.insert(AskPLONNQuestionContext(question: turn.question, intent: turn.intent, generalResponse: generalResponse), at: 0)
            priorCharacters += characters
        }
        let request = try AskPLONNCore.boundedRequest(AskPLONNRequest(question: question, intent: intent, evidence: evidence, ownerID: ownerID, priorTurns: priorTurns.isEmpty ? nil : priorTurns))
        let answer: AskPLONNAnswer
        let verified = request.evidence.filter { $0.kind == intent && $0.availability == .available && !$0.text.isEmpty && !$0.source.isEmpty }
        if let provider, intent == .generalChat || !verified.isEmpty {
            do {
                answer = try AskPLONNCore.validateProviderAnswer(await provider.answer(request), for: request)
            } catch is CancellationError { throw CancellationError() }
            catch AskPLONNError.unsupportedQuestion {
                if intent == .generalChat { answer = AskPLONNCore.generalFallback(request) }
                else {
                    let local = AskPLONNCore.localAnswer(request)
                    answer = AskPLONNAnswer(text: "This question cannot be answered from the supplied evidence. Showing verified data for the selected topic.\n\n" + local.text, evidence: local.evidence)
                }
            }
            catch {
                try Task.checkCancellation()
                guard token == generation, self.ownerID == ownerID, self.contextID == contextID else { throw AskPLONNError.superseded }
                AskPLONNFailureDiagnostic.recordMatchupFailure(error)
                if intent == .generalChat { throw error }
                else {
                    let local = AskPLONNCore.localAnswer(request)
                    answer = AskPLONNAnswer(text: local.text + "\n\nThe provider response was unavailable or could not be verified. Showing verified local data.", evidence: local.evidence)
                }
            }
        } else { answer = AskPLONNCore.localAnswer(request) }
        try Task.checkCancellation()
        guard token == generation, self.ownerID == ownerID, self.contextID == contextID else { throw AskPLONNError.superseded }
        turns.append(AskPLONNTurn(question: question, intent: intent, answer: answer))
        if turns.count > AskPLONNCore.maximumConversationTurnCount { turns.removeFirst(turns.count - AskPLONNCore.maximumConversationTurnCount) }
        return answer
    }
}
