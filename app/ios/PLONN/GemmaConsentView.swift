import SwiftUI

/// One account-specific permission covers questions and requested AI research.
struct GemmaConsentView: View {
    @EnvironmentObject private var account: AccountStore
    let ownerID: UUID?
    let onAllowed: () -> Void
    let onDeclined: () -> Void

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20) {
                Text("Enable Gemma4").font(.title2.bold())
                Text("Gemma4 is a temporary beta service. Your questions, recent conversation, and research you ask AI to explain will be sent to PLONN’s AI service. Player and model attachments are optional. Avoid including sensitive personal information.")
                    .accessibilityIdentifier("gemma-consent-disclosure")
                Text("Up to 5 requests per account each UTC day, with 20 shared across the beta. Accepted requests count even if generation fails. Availability can change.")
                    .font(.subheadline).foregroundStyle(.secondary)
                Link("PLONN privacy policy", destination: PLONNServicePolicy.privacyURL)
                Link("PLONN support", destination: PLONNServicePolicy.supportURL)
                Button("Enable Gemma4") {
                    guard let ownerID, account.allowChat(for: ownerID) else { return }
                    onAllowed()
                }.buttonStyle(.borderedProminent)
                    .accessibilityIdentifier("gemma-consent-allow")
                    .disabled(ownerID == nil || ownerID != account.user?.id || account.user?.isAnonymous != false || !CloudConfiguration.servicePolicy.chatAvailable)
                Button("Use research without AI", action: onDeclined).buttonStyle(.bordered)
                    .accessibilityIdentifier("gemma-consent-decline")
                Spacer()
            }.padding(24).navigationTitle("AI research").navigationBarTitleDisplayMode(.inline)
        }
    }
}
