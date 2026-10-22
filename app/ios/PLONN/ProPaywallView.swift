import SwiftUI
import StoreKit

struct ProSubscriptionSection: View {
    @EnvironmentObject private var purchases: PurchaseStore
    var body: some View {
        Section("Plan") {
            Label(purchases.hasAccess ? "PLONN Pro" : (PurchaseConfiguration.salesEnabled && purchases.plan == nil ? "Plan unavailable" : "PLONN Free"), systemImage: purchases.hasAccess ? "sparkles" : "person.crop.circle")
                .accessibilityIdentifier("pro-account-plan-state")
            Text(purchases.hasAccess ? "AI research and model testing included." : "Basic team and player stats, plus sportsbook odds.")
                .font(.caption).foregroundStyle(.secondary)
            NavigationLink(purchases.hasAccess ? "Manage Pro" : "Explore Pro") { ProPaywallView() }
                .accessibilityIdentifier("pro-account-plans")
        }
    }
}

struct ProPaywallView: View {
    @EnvironmentObject private var purchases: PurchaseStore
    @EnvironmentObject private var account: AccountStore
    @State private var manage = false
    let initialProductID: String?
    init(initialProductID: String? = nil) { self.initialProductID = initialProductID }
    private func period(_ product: Product) -> String {
        guard let period = product.subscription?.subscriptionPeriod else { return "" }
        switch period.unit {
        case .month: return period.value == 1 ? "month" : "\(period.value) months"
        case .year: return period.value == 1 ? "year" : "\(period.value) years"
        case .week: return period.value == 1 ? "week" : "\(period.value) weeks"
        case .day: return period.value == 1 ? "day" : "\(period.value) days"
        @unknown default: return "billing period"
        }
    }
    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 14) {
                    Image(systemName: "sparkles").font(.title2).foregroundStyle(PLONNPalette.highlight)
                    Text("AI research & models")
                        .font(.system(.title, design: .monospaced).bold())
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("pro-paywall-title")
                }.padding(.vertical, 6)
            }
            if purchases.hasAccess {
                Section("Subscription") {
                    Label("Pro is active", systemImage: "checkmark.seal.fill").foregroundStyle(.green)
                    if purchases.plan?.environment == "Sandbox" {
                        Text("Sandbox subscription").font(.caption).foregroundStyle(.secondary)
                    }
                    if let date = purchases.plan?.expiration { Text("Current access through \(date.formatted(date: .abbreviated, time: .omitted)).") }
                    allowanceDetails
                    Button("Manage subscription") { manage = true }
                }
            } else {
                Section(purchases.products.isEmpty ? "Planned US pricing" : "Plans") {
                    if purchases.products.isEmpty {
                        HStack { Text("Monthly"); Spacer(); Text("$15 / month").fontWeight(.semibold) }
                        HStack { Text("Yearly"); Spacer(); Text("$150 / year").fontWeight(.semibold) }
                        Text("Yearly is $12.50 per month, billed as one $150 payment. Save $30 compared with 12 monthly payments.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        VStack(spacing:12) { ForEach(purchases.products) { product in
                            Button {
                                Task { await purchases.purchase(product, for: account.user?.id) }
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(product.id.hasSuffix("yearly") ? "Yearly" : "Monthly").fontWeight(.semibold).foregroundStyle(.primary)
                                        Text("\(product.displayPrice) / \(period(product))").foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                }.padding(12).frame(minHeight:56)
                                 .plonnGlassSurface(selected:initialProductID == product.id,radius:16)
                            }.buttonStyle(.plain)
                             .disabled(purchases.busy || !purchases.canPurchase || !account.hasSignedInAccount)
                             .accessibilityLabel("Subscribe \(product.id.hasSuffix("yearly") ? "yearly" : "monthly"), \(product.displayPrice) per \(period(product))")
                             .accessibilityHint("Opens Apple’s purchase confirmation.")
                             .accessibilityIdentifier("pro-plan-" + product.id)
                        } }.plonnGlassGroup().listRowBackground(Color.clear)
                    }
                    allowanceDetails
                    if purchases.canPurchase {
                        if purchases.isSandboxCheckout {
                            Label("Test purchase · no charge", systemImage: "testtube.2")
                                .font(.caption).foregroundStyle(.secondary)
                                .accessibilityIdentifier("pro-sandbox-checkout")
                        }
                    } else if !account.hasSignedInAccount {
                        Button("Sign in to subscribe") { account.access.present() }
                            .plonnGlassButton(prominent:true)
                    } else {
                        Label("Subscriptions aren’t available yet", systemImage: "clock")
                            .accessibilityIdentifier("pro-paywall-unavailable")
                        Text("You won’t be charged. Final local pricing and included usage will appear here before checkout.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            Section("Pro features") {
                benefit("AI research", detail: "Ask questions, compare evidence, and understand the factors behind a matchup.", icon: "bubble.left.and.text.bubble.right")
                benefit("Model testing", detail: "Test supported player props and team markets against historical results.", icon: "chart.xyaxis.line")
                Text("Sports data coverage varies. AI can make mistakes; past results don’t guarantee future outcomes.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Free features") {
                Label("Basic team and player stats", systemImage: "chart.bar")
                Label("Sportsbook odds with a free account", systemImage: "list.bullet.rectangle")
            }
            Section {
                if purchases.busy { HStack { ProgressView(); Text("Checking with Apple…").font(.caption) } }
                if !purchases.message.isEmpty { Text(purchases.message).font(.caption).accessibilityIdentifier("pro-paywall-purchase-message") }
                Button("Restore purchases") { Task { await purchases.restore(for: account.user?.id) } }.disabled(purchases.busy || !account.hasSignedInAccount)
                Button("Manage Apple subscriptions") { manage = true }
                Button("Refresh plans") { Task { await purchases.loadProducts() } }.disabled(purchases.busy)
                Link("Get help", destination: PLONNServicePolicy.supportURL)
            }
            Section {
                Text("Payment is charged to your Apple Account when you confirm. Subscriptions renew automatically unless cancelled at least 24 hours before the current period ends. Manage or cancel in your Apple Account settings. Deleting PLONN or your PLONN account does not cancel a subscription.")
                    .font(.caption).foregroundStyle(.secondary)
                Link("Privacy policy", destination: PurchaseConfiguration.privacyURL)
                Link("Terms of use", destination: PurchaseConfiguration.termsURL)
            }
        }
        .plonnGlassButton()
        .scrollContentBackground(.hidden).background(PLONNBrandBackdrop())
        .fontDesign(.monospaced).navigationTitle("PLONN Pro").navigationBarTitleDisplayMode(.inline)
        .task { await purchases.loadProducts() }
        .manageSubscriptionsSheet(isPresented: $manage)
    }
    private func benefit(_ title: String, detail: String, icon: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon).foregroundStyle(.green).frame(width: 24)
            VStack(alignment: .leading, spacing: 6) { Text(title).fontWeight(.semibold); Text(detail).font(.subheadline).foregroundStyle(.secondary) }
        }.padding(.vertical, 8)
    }
    @ViewBuilder private var allowanceDetails: some View {
        if let allowance = purchases.plan?.usageAllowance, allowance.isSupported {
            Text("\(allowance.aiRequestsPerDay) AI requests · \(allowance.modelRunsPerDay) new model runs per day")
                .font(.subheadline).accessibilityIdentifier("pro-usage-allowance")
            Text("Resets at midnight UTC. Unused requests don’t roll over. Started requests count even if they fail; viewing saved results doesn’t use your allowance.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
