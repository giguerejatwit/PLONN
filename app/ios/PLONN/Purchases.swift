import SwiftUI
import StoreKit

enum PurchaseConfiguration {
    static let productIDs = Set((Bundle.main.object(forInfoDictionaryKey: "PLONNStoreKitProductIDs") as? [String] ?? [])
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })
    // Enable only after Apple products, server verification, capacity and usage
    // limits have passed the release checklist. No local flag grants server access.
    static let salesEnabled = Bundle.main.object(forInfoDictionaryKey: "PLONNPurchasesEnabled") as? Bool == true
    static let privacyURL = PLONNServicePolicy.privacyURL
    static let termsURL = URL(string: "https://www.apple.com/legal/internet-services/itunes/dev/stdeula/")!
}

@MainActor final class PurchaseStore: ObservableObject {
    struct PurchaseRequest: Identifiable { let id = UUID(); let productID: String }
    @Published private(set) var products: [Product] = []
    @Published private(set) var plan: VerifiedPlanStatus?
    @Published private(set) var busy = false
    @Published private(set) var message = ""
    @Published private(set) var appleEnvironment: String?
    @Published var purchaseRequest: PurchaseRequest?
    private(set) var owner: UUID?
    private weak var accountStore: AccountStore?
    var hasAccess: Bool { plan?.grantsProAccess() == true }
    var mayUseProFeatures: Bool {
        hasAccess || plan?.betaAccess == true || (!PurchaseConfiguration.salesEnabled && plan?.checkoutMode != "sandbox")
    }
    var canPurchase: Bool {
        #if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
        if LocalStoreKitLab.active {
            return owner != nil && plan?.checkoutReady == true && plan?.checkoutMode == "local" && appleEnvironment == "Xcode"
        }
        #endif
        return SubscriptionCheckoutPolicy.permits(plan: plan, owner: owner,
            salesEnabled: PurchaseConfiguration.salesEnabled, appleEnvironment: appleEnvironment)
    }
    var isSandboxCheckout: Bool { canPurchase && plan?.checkoutMode == "sandbox" }

    func bind(_ owner: UUID?, accountStore: AccountStore) async {
        self.owner = owner; self.accountStore = accountStore; plan = nil; message = ""
        await refreshEntitlements()
    }
    func loadProducts() async {
        guard !busy else { return }
        busy = true; message = ""
        defer { busy = false }
        do {
            products = try await Product.products(for: PurchaseConfiguration.productIDs)
                .filter { $0.type == .autoRenewable }.sorted { $0.price < $1.price }
            await verifyAppleEnvironment()
            if products.isEmpty { message = "Plans could not be loaded from Apple. Please try again." }
        } catch { products = []; message = "Could not load plans. Please try again." }
        await refreshEntitlements()
    }
    private func verifyAppleEnvironment() async {
        appleEnvironment = nil
        #if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
        if LocalStoreKitLab.active {
            guard let result = try? await AppTransaction.shared,
                  case .verified(let transaction) = result, transaction.environment == .xcode,
                  transaction.bundleID == Bundle.main.bundleIdentifier else { return }
            appleEnvironment = transaction.environment.rawValue
            return
        }
        #endif
        guard let result = try? await AppTransaction.shared,
              case .verified(let transaction) = result else { return }
        // A server tester designation is not proof that Apple won't charge.
        // Only Apple's verified app transaction identifies the sandbox here.
        appleEnvironment = SubscriptionCheckoutPolicy.appleEnvironment(bundleID: transaction.bundleID,
            appID: transaction.appID, environment: transaction.environment.rawValue)
    }
    func observePurchaseIntents() async {
        for await intent in PurchaseIntent.intents {
            guard !Task.isCancelled else { return }
            guard PurchaseConfiguration.productIDs.contains(intent.product.id) else { continue }
            purchaseRequest = PurchaseRequest(productID: intent.product.id)
        }
    }
    private func sync(signedTransaction: String? = nil, owner: UUID) async throws -> VerifiedPlanStatus {
        guard let accountStore, self.owner == owner, accountStore.user?.id == owner else { throw CancellationError() }
        #if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
        if LocalStoreKitLab.active {
            let status = try await LocalStoreKitLab.shared.verify(signedTransaction: signedTransaction, owner: owner)
            guard self.owner == owner, accountStore.user?.id == owner else { throw CancellationError() }
            plan = status
            return status
        }
        #endif
        let session = try await accountStore.client.auth.session
        guard session.user.id == owner, !session.user.isAnonymous, !session.isExpired else { throw CancellationError() }
        struct Body: Encodable { let signedTransaction: String? }
        let status: VerifiedPlanStatus = try await accountStore.client.functions.invoke("subscription",
            options: .init(headers: ["Authorization": "Bearer " + session.accessToken], body: Body(signedTransaction: signedTransaction), timeoutInterval: 30))
        guard self.owner == owner, accountStore.user?.id == owner else { throw CancellationError() }
        plan = status
        return status
    }
    func refreshEntitlements() async {
        guard let owner else { plan = nil; return }
        do {
            for await result in StoreKit.Transaction.currentEntitlements {
                guard case .verified(let transaction) = result,
                      PurchaseConfiguration.productIDs.contains(transaction.productID), transaction.appAccountToken == owner else { continue }
                _ = try await sync(signedTransaction: result.jwsRepresentation, owner: owner)
                await transaction.finish()
            }
            _ = try await sync(owner: owner)
        } catch {
            guard self.owner == owner else { return }
            plan = nil
            if PurchaseConfiguration.salesEnabled { message = "We couldn’t confirm your plan. Check your connection and try again." }
        }
    }
    func observeTransactions() async {
        for await result in StoreKit.Transaction.updates {
            guard !Task.isCancelled else { return }
            guard case .verified(let transaction) = result, let owner,
                  transaction.appAccountToken == owner, PurchaseConfiguration.productIDs.contains(transaction.productID) else { continue }
            do {
                _ = try await sync(signedTransaction: result.jwsRepresentation, owner: owner)
                await transaction.finish()
            } catch { if self.owner == owner { message = "Your purchase is awaiting confirmation. Restore purchases to try again." } }
        }
    }
    func purchase(_ product: Product, for account: UUID?) async {
        guard !busy, canPurchase, let account, owner == account, products.contains(where: { $0.id == product.id }) else { return }
        busy = true; message = ""
        defer { busy = false }
        do {
            await verifyAppleEnvironment()
            guard canPurchase, owner == account else {
                message = "Checkout could not be confirmed. Refresh plans and try again."; return
            }
            switch try await product.purchase(options: [.appAccountToken(account)]) {
            case .success(let result):
                guard case .verified(let transaction) = result, transaction.appAccountToken == account else {
                    message = "The purchase could not be confirmed for this PLONN account. Please contact support."; return
                }
                let status = try await sync(signedTransaction: result.jwsRepresentation, owner: account)
                await transaction.finish()
                guard owner == account else { return }
                message = status.hasPro ? "You’re on Pro. AI research and model testing are ready." : "Apple has no active Pro subscription for this account. Try Restore purchases."
            case .pending: if owner == account { message = "Waiting for Apple’s approval. Your access will update when it’s approved." }
            case .userCancelled: if owner == account { message = "" }
            @unknown default: if owner == account { message = "Check your subscription with Restore purchases." }
            }
        } catch {
            #if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
            LocalStoreKitLab.shared.record(error: error)
            #endif
            if owner == account { message = "We couldn’t finish confirming the purchase. If Apple charged you, use Restore purchases; you won’t need to buy it again." }
        }
    }
    func restore(for account: UUID?) async {
        guard !busy, let account, owner == account else { return }
        busy = true; message = ""
        defer { busy = false }
        do {
            try await AppStore.sync()
            await refreshEntitlements()
            guard owner == account else { return }
            if plan == nil { message = "We couldn’t confirm your purchases. Please try again." }
            else { message = hasAccess ? "Pro restored for this account." : "No active Pro subscription is linked to this account. Sign in with the PLONN account used to purchase." }
        } catch StoreKitError.userCancelled {
            if owner == account { message = "" }
        } catch let error as SKError where error.code == .paymentCancelled {
            if owner == account { message = "" }
        } catch { if owner == account { message = "Could not restore purchases. Please try again." } }
    }
}

struct SubscriptionView: View {
    var body: some View { ProPaywallView() }
}
