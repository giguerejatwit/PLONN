import SwiftUI
import StoreKit

/// Reusable in Settings and Account; server plan status never derives from StoreKit.
struct ProSubscriptionSection: View {
    @EnvironmentObject private var account: AccountStore

    var body: some View {
        Section("Subscription") {
            Text(account.user == nil ? "Sign in to check your plan when subscriptions become available." : "Plan status unavailable")
                .accessibilityIdentifier("pro-account-plan-state")
            Text("AI token allowances are still being finalized.")
                .accessibilityIdentifier("pro-account-allowance-state")
            NavigationLink("Pro plans") { ProPaywallView() }
                .accessibilityIdentifier("pro-account-plans")
            Text("Pro subscriptions are unavailable in this beta.")
                .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
        }
    }
}

/// App Store information is separate from trusted server Pro access.
struct ProPaywallView: View {
    @EnvironmentObject private var purchases: PurchaseStore
    @EnvironmentObject private var account: AccountStore
    @State private var manage = false

    private var configured: Bool { !PurchaseConfiguration.productIDs.isEmpty }
    private var plans: [Product] {
        purchases.products.filter { PurchaseConfiguration.productIDs.contains($0.id) && $0.type == .autoRenewable }
    }

    private func billing(_ product: Product) -> String {
        guard let period = product.subscription?.subscriptionPeriod else { return "" }
        let unit: String
        switch period.unit {
        case .day: unit = "day"
        case .week: unit = "week"
        case .month: unit = "month"
        case .year: unit = "year"
        @unknown default: unit = "period"
        }
        return "Every \(period.value) \(unit)\(period.value == 1 ? "" : "s") · Auto-renewing"
    }

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 10) {
                    Text("PLONN Pro")
                        .font(.system(.largeTitle, design: .monospaced).weight(.bold))
                        .accessibilityIdentifier("pro-paywall-title")
                    Text("Pro subscriptions are unavailable in this beta.")
                        .font(.system(.headline, design: .monospaced))
                        .accessibilityIdentifier("pro-paywall-unavailable")
                    Text("Plan details are being prepared. Purchases are unavailable in this beta.")
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("pro-paywall-read-only")
                }.padding(.vertical, 8)
            }
            Section("Pro access") {
                Text("Plan status cannot be confirmed in this beta.")
                    .accessibilityIdentifier("pro-paywall-server-state")
                Text("App Store verification on this device does not grant server Pro access or an AI allowance.")
                    .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                if configured {
                    Text(purchases.hasAccess ? "Device verification: active App Store entitlement for this account." : "Device verification: no active App Store entitlement for this account.")
                        .accessibilityIdentifier("pro-paywall-device-state")
                }
            }
            Section("AI allowance") {
                Text("Free and Pro token allowances are still being finalized.")
                    .accessibilityIdentifier("pro-paywall-allowance-state")
                Text("Beta safety caps are separate from tokens: 5 attempts per account and 20 globally per UTC day. Pro does not increase these caps in this beta.")
                    .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            }
            Section("Subscription plans") {
                if !configured {
                    Text("Subscription plans are not available yet.")
                        .accessibilityIdentifier("pro-paywall-plans-unavailable")
                } else if plans.isEmpty && !purchases.busy {
                    Text("Configured subscription plans are not available from the App Store.")
                        .accessibilityIdentifier("pro-paywall-plans-unavailable")
                }
                ForEach(plans) { product in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(product.displayName).font(.system(.headline, design: .monospaced))
                        Text(product.description)
                        Text(product.displayPrice).fontWeight(.semibold)
                            .accessibilityIdentifier("pro-paywall-product-price")
                        Text(billing(product)).foregroundStyle(.secondary)
                    }.padding(.vertical, 6)
                }
                if purchases.busy { PLONNLoader() }
                if !purchases.message.isEmpty {
                    Text(purchases.message).font(.system(.caption, design: .monospaced))
                        .accessibilityIdentifier("pro-paywall-purchase-message")
                }
            }
            if configured {
                Section("App Store") {
                    Button("Restore purchases") { Task { await purchases.restore(for: account.user?.id) } }
                        .disabled(purchases.busy || account.user == nil)
                    Button("Manage App Store subscriptions") { manage = true }.disabled(purchases.busy)
                    Text("Restoring checks App Store purchases on this device. Server Pro access remains unavailable. Deleting your PLONN account does not cancel an App Store subscription.")
                        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                }
            }
            Section("Terms") {
                if let url = PurchaseConfiguration.privacyURL { Link("Privacy policy", destination: url) }
                if let url = PurchaseConfiguration.termsURL { Link("Terms of use", destination: url) }
                if PurchaseConfiguration.privacyURL == nil || PurchaseConfiguration.termsURL == nil {
                    Text("Public privacy and subscription terms must be configured before subscriptions become available.")
                        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                }
            }
        }
        .fontDesign(.monospaced)
        .navigationTitle("Pro")
        .navigationBarTitleDisplayMode(.inline)
        .task { if configured { await purchases.loadProducts() } }
        .manageSubscriptionsSheet(isPresented: $manage)
    }
}
