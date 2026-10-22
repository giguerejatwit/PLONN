#if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
import SwiftUI
import StoreKit
import StoreKitTest

/// Built only by the StoreKitLab configuration, under a separate simulator bundle.
/// Local Apple transactions are never submitted to Supabase or used for cloud access.
@MainActor final class LocalStoreKitLab: ObservableObject {
    static var active: Bool { Bundle.main.bundleIdentifier == "com.oakas.plonn.storekitlab" }
    static let shared = LocalStoreKitLab()
    static let owner = UUID(uuidString: "00000000-0000-4000-8000-000000000001")!
    static let otherOwner = UUID(uuidString: "00000000-0000-4000-8000-000000000002")!
    @Published var status = "Starting local StoreKit…"
    @Published var ready = false
    @Published var running = false
    var session: SKTestSession?
    var rejectVerification = false
    private(set) var results: [[String: String]] = []
    private var diagnostics: [[String: String]] = []
    func record(error: Error) {
        diagnostics.append(["purchaseError": String(describing: error), "localized": error.localizedDescription])
    }

    struct LabFailure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }
    func require(_ value: @autoclosure () -> Bool, _ message: String) throws {
        if !value() { throw LabFailure(message: message) }
    }
    func start(account: AccountStore, purchases: PurchaseStore) async throws {
        try require(Self.active, "Local test bundle is required")
        if session == nil {
            let newSession = try SKTestSession(configurationFileNamed: "PLONNLocal")
            newSession.resetToDefaultState()
            newSession.disableDialogs = ProcessInfo.processInfo.environment["PLONN_STOREKIT_RUN_SUITE"] == "1"
            newSession.timeRate = .realTime
            session = newSession
        }
        account.prepareLocalStoreKitUser(Self.owner)
        await purchases.bind(Self.owner, accountStore: account)
        await purchases.loadProducts()
        ready = true
        status = "Local StoreKit · no charge · server simulated"
    }
    func reset(account: AccountStore, purchases: PurchaseStore) async throws {
        guard let session else { throw LabFailure(message: "No local session") }
        rejectVerification = false
        // resetToDefaultState clears injected errors on these runtimes. Setting a
        // simulated error to nil can instead create an empty, failing error override.
        session.resetToDefaultState()
        session.disableDialogs = true
        session.timeRate = .realTime
        session.clearTransactions()
        account.prepareLocalStoreKitUser(Self.owner)
        await purchases.bind(Self.owner, accountStore: account)
        await purchases.loadProducts()
    }
    private func firstProduct(_ purchases: PurchaseStore) throws -> Product {
        guard let product = purchases.products.first else { throw LabFailure(message: "Local catalog unavailable") }
        return product
    }
    func verify(signedTransaction: String?, owner: UUID) async throws -> VerifiedPlanStatus {
        try require(Self.active && session != nil, "Local StoreKit session required")
        if rejectVerification { throw LabFailure(message: "Simulated verification outage") }
        var activeTransaction: StoreKit.Transaction?
        var transactionMatched = signedTransaction == nil
        var claim: [String: Any]?
        if let signedTransaction {
            let segments = signedTransaction.split(separator: ".")
            try require(segments.count == 3, "Malformed local transaction")
            var encoded = String(segments[1]).replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
            encoded += String(repeating: "=", count: (4 - encoded.count % 4) % 4)
            if let data = Data(base64Encoded: encoded) { claim = try JSONSerialization.jsonObject(with: data) as? [String: Any] }
        }
        for await result in StoreKit.Transaction.currentEntitlements {
            switch result {
            case .verified(let item): diagnostics.append(["id": String(item.id), "product": item.productID, "environment": item.environment.rawValue, "ownerMatches": String(item.appAccountToken == owner), "expires": item.expirationDate?.description ?? "nil", "revoked": String(item.revocationDate != nil), "jwsMatches": String(result.jwsRepresentation == signedTransaction)])
            case .unverified(_, let error): diagnostics.append(["error": error.localizedDescription])
            }

            guard case .verified(let transaction) = result,
                  transaction.environment == .xcode,
                  transaction.appAccountToken == owner,
                  PurchaseConfiguration.productIDs.contains(transaction.productID),
                  transaction.revocationDate == nil,
                  let expiration = transaction.expirationDate, expiration > Date() else { continue }
            activeTransaction = transaction
        }
        // Expired/refunded transactions are valid evidence too; their status is free.
        // The local verifier checks Apple-verified history, independently of current access.
        if signedTransaction != nil {
            for await result in StoreKit.Transaction.all {
                guard case .verified(let transaction) = result, transaction.environment == .xcode,
                      transaction.appAccountToken == owner, PurchaseConfiguration.productIDs.contains(transaction.productID) else { continue }
            // StoreKit can re-sign the same transaction. Compare its identity to a
            // verified local entitlement; byte equality of two JWS strings is invalid.
            if claim?["transactionId"] as? String == String(transaction.id),
               claim?["productId"] as? String == transaction.productID,
               (claim?["appAccountToken"] as? String)?.lowercased() == owner.uuidString.lowercased(),
               claim?["environment"] as? String == "Xcode",
               claim?["bundleId"] as? String == Bundle.main.bundleIdentifier { transactionMatched = true }
            }
        }
        diagnostics.append(["signatureMatched": String(transactionMatched), "hasActive": String(activeTransaction != nil)])
        try require(transactionMatched, "Local transaction does not match this account")
        return VerifiedPlanStatus(hasPro: activeTransaction != nil, betaAccess: false, checkoutReady: true,
            productID: activeTransaction?.productID,
            expiresAt: activeTransaction?.expirationDate.map { ISO8601DateFormatter().string(from: $0) },
            checkoutMode: "local", environment: "Xcode",
            usageAllowance: VerifiedUsageAllowance(version: "plonn_pro_daily_v1", aiRequestsPerDay: 20,
                modelRunsPerDay: 5, resetTimeZone: "UTC", rollover: false, counting: "started_attempts"))
    }
    private func check(_ name: String, _ test: () async throws -> Void) async {
        status = "Testing: \(name)"
        do {
            try await test()
            results.append(["test": name, "result": "passed"])
        } catch { results.append(["test": name, "result": "failed", "error": error.localizedDescription]) }
    }
    private func observeChange(_ purchases: PurchaseStore, access: Bool, operation: () async throws -> Void) async throws {
        try await observeUpdate(purchases, until: { purchases.hasAccess == access },
                                failure: "Transaction update did not change access to \(access) within 10 seconds",
                                operation: operation)
    }
    private func observeUpdate(_ purchases: PurchaseStore, until changed: () -> Bool,
                               failure: String, operation: () async throws -> Void) async throws {
        let listener = Task { await purchases.observeTransactions() }
        do {
            await Task.yield()
            try await operation()
            for _ in 0..<100 {
                if changed() { listener.cancel(); await listener.value; return }
                try await Task.sleep(for: .milliseconds(100))
            }
            throw LabFailure(message: failure)
        } catch { listener.cancel(); await listener.value; throw error }
    }
    func runSuite(account: AccountStore, purchases: PurchaseStore) async {
        guard !running, let session else { return }
        running = true
        results = []
        diagnostics = []
        defer { running = false }
        await check("catalog, prices, periods and verified Xcode environment") {
            try await reset(account: account, purchases: purchases)
            try require(purchases.products.count == 2, "Expected two products")
            try require(purchases.appleEnvironment == "Xcode" && purchases.canPurchase, "Local checkout not ready")
            for (suffix, price, unit) in [("monthly", Decimal(15), Product.SubscriptionPeriod.Unit.month), ("yearly", Decimal(150), .year)] {
                let product = purchases.products.first { $0.id.hasSuffix(suffix) }
                try require(product?.price == price && product?.subscription?.subscriptionPeriod.unit == unit, "Wrong \(suffix) price or period")
            }
        }
        for suffix in ["monthly", "yearly"] {
            await check("\(suffix) purchase, unlock, new store and restore") {
                try await reset(account: account, purchases: purchases)
                guard let product = purchases.products.first(where: { $0.id.hasSuffix(suffix) }) else { throw LabFailure(message: "Missing product") }
                try require(!purchases.hasAccess, "Must begin without subscription")
                await purchases.purchase(product, for: Self.owner)
                try require(purchases.hasAccess && purchases.plan?.productID == product.id, "Purchase did not unlock: \(purchases.message)")
                try require(purchases.message.contains("You’re on Pro"), "Missing purchase confirmation")
                let restored = PurchaseStore()
                await restored.bind(Self.owner, accountStore: account)
                await restored.restore(for: Self.owner)
                try require(restored.hasAccess && restored.message == "Pro restored for this account.", "Restore failed: \(restored.message)")
            }
            await check("\(suffix) renewal extends access and restores the renewed term") {
                try await reset(account: account, purchases: purchases)
                guard let product = purchases.products.first(where: { $0.id.hasSuffix(suffix) }) else { throw LabFailure(message: "Missing product") }
                await purchases.purchase(product, for: Self.owner)
                guard purchases.hasAccess, let before = purchases.plan?.expiresAt,
                      let expiration = ISO8601DateFormatter().date(from: before) else { throw LabFailure(message: "Setup purchase has no expiry") }
                let previousIDs = Set(session.allTransactions().map(\.identifier))
                try await observeUpdate(purchases, until: {
                    guard let value = purchases.plan?.expiresAt, let renewed = ISO8601DateFormatter().date(from: value) else { return false }
                    return purchases.hasAccess && purchases.plan?.productID == product.id && renewed > expiration
                }, failure: "Renewal did not extend the verified term through transaction updates") {
                    // Forced renewal starts a new term now, not at the old expiry.
                    // Keep the two purchases in distinct seconds for the ISO-date assertion.
                    try await Task.sleep(for: .milliseconds(1100))
                    try session.forceRenewalOfSubscription(productIdentifier: product.id)
                }
                try require(!Set(session.allTransactions().map(\.identifier)).subtracting(previousIDs).isEmpty,
                            "Apple did not create a renewal transaction")
                let renewedExpiration = purchases.plan?.expiresAt
                let restored = PurchaseStore()
                await restored.bind(Self.owner, accountStore: account)
                await restored.restore(for: Self.owner)
                try require(restored.hasAccess && restored.plan?.expiresAt == renewedExpiration && restored.plan?.productID == product.id,
                            "Restore did not retain the renewed term")
            }
        }
        await check("turning off renewal preserves paid access until expiry") {
            try await reset(account: account, purchases: purchases)
            let product = try firstProduct(purchases)
            await purchases.purchase(product, for: Self.owner)
            guard purchases.hasAccess, let transaction = session.allTransactions().last else { throw LabFailure(message: "Setup purchase failed") }
            let expiration = purchases.plan?.expiresAt
            try session.disableAutoRenewForTransaction(identifier: transaction.identifier)
            await purchases.refreshEntitlements()
            try require(purchases.hasAccess && purchases.plan?.expiresAt == expiration, "Cancelling renewal revoked the paid term")
            try await observeChange(purchases, access: false) {
                try session.expireSubscription(productIdentifier: product.id)
            }
            await purchases.restore(for: Self.owner)
            try require(!purchases.hasAccess, "Restore reactivated an expired subscription")
        }
        await check("cancelled purchase does not unlock or show an error") {
            try await reset(account: account, purchases: purchases)
            try await session.setSimulatedError(.generic(.userCancelled), forAPI: .purchase)
            await purchases.purchase(try firstProduct(purchases), for: Self.owner)
            try require(!purchases.hasAccess && purchases.message.isEmpty && !purchases.busy, "Cancellation shown as failure: \(purchases.message)")
        }
        await check("verification failure stays free; restore recovers without repurchase") {
            try await reset(account: account, purchases: purchases)
            rejectVerification = true
            await purchases.purchase(try firstProduct(purchases), for: Self.owner)
            try require(!purchases.hasAccess && purchases.message.contains("couldn’t finish confirming"), "Unverified purchase unlocked")
            let count = session.allTransactions().count
            try require(count > 0, "Apple purchase was not recorded")
            rejectVerification = false
            await purchases.restore(for: Self.owner)
            try require(purchases.hasAccess && purchases.message == "Pro restored for this account.", "Retry did not restore")
            try require(session.allTransactions().count == count, "Restore created another purchase")
        }
        await check("another PLONN account cannot claim the purchase") {
            try await reset(account: account, purchases: purchases)
            await purchases.purchase(try firstProduct(purchases), for: Self.owner)
            try require(purchases.hasAccess, "Setup purchase failed")
            account.prepareLocalStoreKitUser(Self.otherOwner)
            await purchases.bind(Self.otherOwner, accountStore: account)
            await purchases.restore(for: Self.otherOwner)
            try require(!purchases.hasAccess && purchases.message.contains("No active Pro"), "Wrong account received access")
            account.prepareLocalStoreKitUser(Self.owner)
            await purchases.bind(Self.owner, accountStore: account)
            await purchases.restore(for: Self.owner)
            try require(purchases.hasAccess, "Original owner lost access")
        }
        await check("expired subscription loses access") {
            try await reset(account: account, purchases: purchases)
            let product = try firstProduct(purchases)
            await purchases.purchase(product, for: Self.owner)
            try require(purchases.hasAccess, "Setup purchase failed")
            try await observeChange(purchases, access: false) {
                try session.expireSubscription(productIdentifier: product.id)
            }
        }
        await check("refunded subscription loses access") {
            try await reset(account: account, purchases: purchases)
            await purchases.purchase(try firstProduct(purchases), for: Self.owner)
            try require(purchases.hasAccess, "Setup purchase failed")
            guard let transaction = session.allTransactions().last else { throw LabFailure(message: "Missing transaction") }
            try await observeChange(purchases, access: false) {
                try session.refundTransaction(identifier: transaction.identifier)
            }
        }
        await check("pending approval remains locked; approval updates access") {
            try await reset(account: account, purchases: purchases)
            session.askToBuyEnabled = true
            await purchases.purchase(try firstProduct(purchases), for: Self.owner)
            try require(!purchases.hasAccess && purchases.message.contains("Waiting for Apple"), "Pending purchase did not remain locked")
            guard let transaction = session.allTransactions().last else { throw LabFailure(message: "Missing pending transaction") }
            try await observeChange(purchases, access: true) {
                try session.approveAskToBuyTransaction(identifier: transaction.identifier)
            }
        }
        await check("restore cancellation is silent; sync failure is recoverable") {
            try await reset(account: account, purchases: purchases)
            await purchases.purchase(try firstProduct(purchases), for: Self.owner)
            try require(purchases.hasAccess, "Setup purchase failed")
            try await session.setSimulatedError(.generic(.userCancelled), forAPI: .appStoreSync)
            await purchases.restore(for: Self.owner)
            try require(purchases.message.isEmpty && !purchases.busy, "Cancelled restore shows error")
            try await session.setSimulatedError(.generic(.notAvailableInStorefront), forAPI: .appStoreSync)
            await purchases.restore(for: Self.owner)
            try require(purchases.message.contains("Could not restore") && !purchases.busy, "Restore error missing")
            session.resetToDefaultState(); session.disableDialogs = true
            await purchases.restore(for: Self.owner)
            try require(purchases.hasAccess && purchases.message == "Pro restored for this account.", "Restore retry failed")
        }
        await check("catalog failure shows error and refresh recovers") {
            try await reset(account: account, purchases: purchases)
            try await session.setSimulatedError(.generic(.notAvailableInStorefront), forAPI: .loadProducts)
            await purchases.loadProducts()
            try require(purchases.products.isEmpty && !purchases.message.isEmpty, "Catalog failure missing")
            session.resetToDefaultState(); session.disableDialogs = true
            await purchases.loadProducts()
            try require(purchases.products.count == 2 && purchases.canPurchase && purchases.message.isEmpty, "Catalog refresh failed or retained a stale error")
        }
        let passed = results.filter { $0["result"] == "passed" }.count
        let receipt: [String: Any] = ["date": ISO8601DateFormatter().string(from: Date()), "bundle": Bundle.main.bundleIdentifier!,
            "appleEnvironment": purchases.appleEnvironment ?? "unknown", "serverVerification": "simulated locally; no Supabase calls", "liveSandboxValidated": false,
            "passed": passed, "total": results.count, "results": results, "diagnostics": diagnostics]
        let url = URL.documentsDirectory.appendingPathComponent("storekit-lab-results.json")
        try? JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
        try? await reset(account: account, purchases: purchases)
        session.disableDialogs = false
        status = "Local tests: \(passed)/\(results.count) passed · no charge"
    }
}

struct LocalStoreKitLabView: View {
    @EnvironmentObject private var account: AccountStore
    @EnvironmentObject private var purchases: PurchaseStore
    @StateObject private var lab = LocalStoreKitLab.shared
    @State private var booted = false
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                VStack(spacing: 5) {
                    Text(lab.status).font(.caption.bold()).accessibilityIdentifier("storekit-lab-status")
                    Text("Apple local StoreKit · simulated server verification").font(.caption2).foregroundStyle(.secondary)
                    HStack {
                        Button("Reset demo") { Task { try? await lab.reset(account: account, purchases: purchases); lab.session?.disableDialogs = false } }
                        Button("Run tests") { Task { await lab.runSuite(account: account, purchases: purchases) } }
                    }.font(.caption).disabled(!lab.ready || lab.running || purchases.busy)
                }.padding(10).frame(maxWidth: .infinity).background(.pink.opacity(0.12))
                if lab.ready && !lab.running { ProPaywallView().task { await purchases.observeTransactions() } }
                else { ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity) }
            }.task {
                guard !booted else { return }; booted = true
                do {
                    try await lab.start(account: account, purchases: purchases)
                    if ProcessInfo.processInfo.environment["PLONN_STOREKIT_RUN_SUITE"] == "1" {
                        await lab.runSuite(account: account, purchases: purchases)
                    }
                } catch { lab.status = "Local test setup failed: \(error.localizedDescription)" }
            }
        }
    }
}
#endif
