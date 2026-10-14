import SwiftUI
import StoreKit

enum PurchaseConfiguration {
    // Fail closed until trusted server verification and token accounting are implemented.
    // This readiness cannot be enabled by an Info.plist flag or device entitlement.
    static let trustedProCheckoutReady = false
    static let productIDs = Set((Bundle.main.object(forInfoDictionaryKey: "PLONNStoreKitProductIDs") as? [String] ?? [])
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })
    static let salesEnabled = Bundle.main.object(forInfoDictionaryKey: "PLONNPurchasesEnabled") as? Bool == true
    static func url(_ key: String) -> URL? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String,
              let url = URL(string: value), url.scheme == "https", url.host != nil else { return nil }
        return url
    }
    static var privacyURL: URL? { url("PLONNPrivacyPolicyURL") }
    static var termsURL: URL? { url("PLONNTermsURL") }
}

@MainActor final class PurchaseStore: ObservableObject {
    @Published private(set) var products: [Product] = []
    @Published private(set) var entitlements: [VerifiedProductEntitlement] = []
    @Published private(set) var busy = false
    @Published private(set) var message = ""
    private(set) var owner: UUID?
    var hasAccess: Bool { entitlements.contains { $0.grantsAccess(to: owner, configuredProducts: PurchaseConfiguration.productIDs) } }
    var canPurchase: Bool {
        PurchaseConfiguration.trustedProCheckoutReady && owner != nil && PurchaseConfiguration.salesEnabled && PurchaseConfiguration.privacyURL != nil && PurchaseConfiguration.termsURL != nil
    }
    func bind(_ account: UUID?) async {
        owner = account; entitlements = []; message = ""
        await refreshEntitlements()
    }
    func loadProducts() async {
        guard !busy else { return }
        guard !PurchaseConfiguration.productIDs.isEmpty else { message = "Purchases are unavailable. No App Store products are configured."; return }
        busy = true
        defer { busy = false }
        do {
            products = try await Product.products(for: PurchaseConfiguration.productIDs)
                .filter { $0.type == .autoRenewable || $0.type == .nonConsumable }.sorted { $0.id < $1.id }
            message = products.isEmpty ? "Configured products are not available from the App Store." : ""
        } catch { products = []; message = "Could not load App Store products. Try again." }
    }
    func refreshEntitlements() async {
        let account = owner
        var current: [VerifiedProductEntitlement] = []
        for await result in StoreKit.Transaction.currentEntitlements {
            guard case .verified(let transaction) = result,
                  PurchaseConfiguration.productIDs.contains(transaction.productID),
                  transaction.productType == .autoRenewable || transaction.productType == .nonConsumable else { continue }
            current.append(VerifiedProductEntitlement(productID: transaction.productID, accountToken: transaction.appAccountToken,
                expiration: transaction.expirationDate, revoked: transaction.revocationDate != nil || transaction.isUpgraded))
        }
        guard owner == account else { return }
        entitlements = current
    }
    func observeTransactions() async {
        for await result in StoreKit.Transaction.updates {
            guard !Task.isCancelled else { return }
            guard case .verified(let transaction) = result,
                  PurchaseConfiguration.productIDs.contains(transaction.productID) else { continue }
            await refreshEntitlements()
            await transaction.finish()
        }
    }
    func purchase(_ product: Product, for account: UUID?) async {
        guard !busy, canPurchase, let account, owner == account, products.contains(where: { $0.id == product.id }) else { return }
        busy = true; message = ""
        defer { busy = false }
        do {
            switch try await product.purchase(options: [.appAccountToken(account)]) {
            case .success(let result):
                guard case .verified(let transaction) = result else { message = "The purchase could not be verified. Access was not granted."; return }
                await refreshEntitlements()
                await transaction.finish()
                guard owner == account else { return }
                message = hasAccess ? "Purchase verified for this account on this device." : "Purchase returned without an active entitlement for this account. Contact support."
            case .pending: if owner == account { message = "Purchase pending approval. Access has not been granted." }
            case .userCancelled: if owner == account { message = "Purchase cancelled." }
            @unknown default: if owner == account { message = "Purchase status unavailable. Try restoring purchases." }
            }
        } catch { if owner == account { message = "Purchase could not complete. No access was granted by the app." } }
    }
    func restore(for account: UUID?) async {
        guard !busy, let account, owner == account else { return }
        busy = true
        defer { busy = false }
        do {
            try await AppStore.sync()
            await refreshEntitlements()
            guard owner == account else { return }
            message = hasAccess ? "Purchases restored for this account." : "No active purchase is linked to this PLONN account. Use the account used for purchase."
        } catch { if owner == account { message = "Could not restore purchases. Try again." } }
    }
}

struct SubscriptionView: View {
    @EnvironmentObject private var purchases: PurchaseStore
    @EnvironmentObject private var account: AccountStore
    @State private var manage = false
    private func billing(_ product: Product) -> String {
        guard let period = product.subscription?.subscriptionPeriod else { return "One-time purchase" }
        let unit: String
        switch period.unit { case .day: unit = "day"; case .week: unit = "week"; case .month: unit = "month"; case .year: unit = "year"; @unknown default: unit = "period" }
        return "Every \(period.value) \(unit)\(period.value == 1 ? "" : "s") · Auto-renewing"
    }
    var body: some View {
        Form {
            Section("Access") {
                Text(purchases.hasAccess ? "Verified App Store access on this device" : "No active purchase for this account")
                Text("Purchases do not make unsupported datasets or model types available. Cloud access requires separate server verification.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                if account.user == nil { NavigationLink("Sign in") { AccountView() } }
            }
            Section("App Store products") {
                if !PurchaseConfiguration.salesEnabled { Text("Purchases are not enabled for this release.").foregroundStyle(.secondary) }
                ForEach(purchases.products) { product in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(product.displayName).font(.system(.headline, design: .monospaced))
                        Text(product.description).font(.system(.subheadline, design: .monospaced))
                        Text(product.displayPrice + " · " + billing(product)).font(.system(.subheadline, design: .monospaced).weight(.semibold))
                        Button("Purchase") { Task { await purchases.purchase(product, for: account.user?.id) } }.disabled(purchases.busy || !purchases.canPurchase)
                    }.padding(.vertical, 6)
                }
                if purchases.busy { PLONNLoader() }
                if !purchases.message.isEmpty { Text(purchases.message).font(.system(.caption, design: .monospaced)) }
                Button("Reload products") { Task { await purchases.loadProducts() } }.disabled(purchases.busy)
            }
            Section {
                Button("Restore purchases") { Task { await purchases.restore(for: account.user?.id) } }.disabled(purchases.busy || account.user == nil || PurchaseConfiguration.productIDs.isEmpty)
                Button("Manage App Store subscriptions") { manage = true }
                Text("Deleting your PLONN account does not cancel App Store subscriptions. Manage them in your Apple account before deletion.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            }
            Section("Terms") {
                if let url = PurchaseConfiguration.privacyURL { Link("Privacy policy", destination: url) }
                if let url = PurchaseConfiguration.termsURL { Link("Terms of use", destination: url) }
                if PurchaseConfiguration.privacyURL == nil || PurchaseConfiguration.termsURL == nil { Text("Public privacy and terms URLs must be configured before purchases are enabled.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
            }
        }.navigationTitle("Purchases & access")
         .task { await purchases.loadProducts() }
         .manageSubscriptionsSheet(isPresented: $manage)
    }
}
