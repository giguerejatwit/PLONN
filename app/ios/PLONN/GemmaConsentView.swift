import SwiftUI

/// One account-specific permission covers questions and requested AI research.
struct GemmaConsentView: View {
    @EnvironmentObject private var account: AccountStore
    let ownerID: UUID?
    let onAllowed: () -> Void
    let onDeclined: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView { VStack(alignment: .leading, spacing: 20) {
                Text("Enable AI research").font(.title2.bold())
                Text(AccountAIConsentPolicy.disclosure)
                    .accessibilityIdentifier("gemma-consent-disclosure")
                Text("Up to 5 requests per account each UTC day, with 20 shared across the beta. Accepted requests count even if generation fails. Availability can change.")
                    .font(.subheadline).foregroundStyle(.secondary)
                Link("PLONN privacy policy", destination: PLONNServicePolicy.privacyURL)
                Link("Ollama privacy policy", destination: URL(string: "https://ollama.com/privacy")!)
                Link("PLONN support", destination: PLONNServicePolicy.supportURL)
                Button("Enable AI research") {
                    guard let ownerID, account.allowChat(for: ownerID) else { return }
                    onAllowed()
                }.plonnGlassButton(prominent:true)
                    .accessibilityIdentifier("gemma-consent-allow")
                    .disabled(ownerID == nil || ownerID != account.user?.id || account.user?.isAnonymous != false || !CloudConfiguration.servicePolicy.chatAvailable)
                Button("Use research without AI", action: onDeclined).plonnGlassButton()
                    .accessibilityIdentifier("gemma-consent-decline")
                Spacer()
            }.padding(24) }.navigationTitle("AI research").navigationBarTitleDisplayMode(.inline)
        }
    }
}
