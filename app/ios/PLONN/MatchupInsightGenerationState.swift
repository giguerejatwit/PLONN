import Foundation

/// Eligibility changes never authorize a request. Only an explicit begin creates a ticket.
struct MatchupInsightGenerationContext: Equatable, Sendable {
    let snapshot: GemmaMatchupInsightSnapshot
    let ready: Bool
    let providerAvailable: Bool
    let usesLocalProvider: Bool
    let active: Bool

    var canGenerate: Bool {
        ready && providerAvailable && active && !snapshot.insights.isEmpty &&
            (usesLocalProvider || snapshot.ownerID.map { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } == true)
    }
}

struct MatchupInsightGenerationState {
    struct Ticket: Equatable, Sendable {
        let id: UUID
        let context: MatchupInsightGenerationContext
    }

    private(set) var pending: Ticket?

    mutating func begin(in context: MatchupInsightGenerationContext) -> Ticket? {
        guard pending == nil, context.canGenerate else { return nil }
        let ticket = Ticket(id: UUID(), context: context)
        pending = ticket
        return ticket
    }

    func accepts(_ ticket: Ticket, in context: MatchupInsightGenerationContext) -> Bool {
        pending == ticket && ticket.context == context && context.canGenerate
    }

    @discardableResult
    mutating func finish(_ ticket: Ticket, in context: MatchupInsightGenerationContext) -> Bool {
        guard accepts(ticket, in: context) else { return false }
        pending = nil
        return true
    }

    mutating func invalidate() { pending = nil }
}
