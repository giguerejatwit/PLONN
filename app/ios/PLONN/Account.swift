import SwiftUI
import Supabase
import AuthenticationServices

/// Wraps the SDK's storage with fixed diagnostic categories. The callback guard
/// checks verifier presence without retaining or exposing credential bytes.
private final class DiagnosedAuthStorage: AuthLocalStorage, @unchecked Sendable {
    private let base = KeychainLocalStorage(service: "com.plonn.auth")
    private let lock = NSLock()
    private var status = "verifier_unobserved"
    private var operations: [String] = []
    private func storageFailure(_ error: Error) -> String {
        // Only fixed OS error categories; never include a credential or raw error.
        let description = error.localizedDescription
        if description.contains("-34018") { return "missing_entitlement" }
        if description.contains("errSecItemNotFound") { return "not_found" }
        if description.contains("errSecInteractionNotAllowed") { return "locked" }
        return "keychain_error"
    }
    func verifierPresent(key: String) -> Bool {
        (try? retrieve(key: key))?.isEmpty == false
    }
    var category: String { lock.lock(); defer { lock.unlock() }; return operations.isEmpty ? status : operations.joined(separator: ",") }
    private func record(_ category: String, key: String) {
        guard key.hasSuffix("-code-verifier") else { return }
        lock.lock(); status = category; operations.append(category); if operations.count > 8 { operations.removeFirst() }; lock.unlock()
    }
    func store(key: String, value: Data) throws {
        do { try base.store(key: key, value: value); record(value.isEmpty ? "verifier_write_empty" : "verifier_write_ok", key: key) }
        catch { record("verifier_write_failed_" + storageFailure(error), key: key); throw error }
    }
    func retrieve(key: String) throws -> Data? {
        do {
            let value = try base.retrieve(key: key)
            record(value?.isEmpty == false ? "verifier_present" : "verifier_missing", key: key)
            return value
        } catch { record("verifier_read_failed_" + storageFailure(error), key: key); throw error }
    }
    func remove(key: String) throws {
        do { try base.remove(key: key); record("verifier_removed", key: key) }
        catch { record("verifier_remove_failed", key: key); throw error }
    }
}

struct CloudWatch: Codable, Identifiable {
    let id: UUID
    let user_id: UUID
    let market_id: String
    let snapshot: Market
}
struct CloudWatchInsert: Encodable {
    let user_id: UUID
    let market_id: String
    let snapshot: Market
}

@MainActor final class AccountStore: ObservableObject {
    let access = AccountAccessCoordinator()
    @Published private(set) var restoringSession = true
    var hasSignedInAccount: Bool { AccountAccessPolicy.permits(ownerID:user?.id,isAnonymous:user?.isAnonymous) }

    private static let authStorage = DiagnosedAuthStorage()
    private var completedEmailCallbacks: [(fingerprint: String, owner: UUID)] = []
    private var verifierStorageKey: String {
        "sb-" + (CloudConfiguration.url.host?.split(separator: ".").first.map(String.init) ?? "") + "-auth-token-code-verifier"
    }
    private let authCallback = AuthCallbackPolicy(bundleIdentifier: Bundle.main.bundleIdentifier)
    private static var offlineReportFixture: Bool {
#if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
        if LocalStoreKitLab.active { return true }
#endif
#if DEBUG && targetEnvironment(simulator)
        return SharedReportOfflineFixture.active
#else
        return false
#endif
    }
    private static var clientStorage: any AuthLocalStorage { offlineReportFixture ? SocialAuthMemoryStorage() : authStorage }
    let client = SupabaseClient(
        supabaseURL: CloudConfiguration.url,
        supabaseKey: CloudConfiguration.publishableKey,
        options: .init(auth: .init(storage: AccountStore.clientStorage,
                                  redirectToURL: AuthCallbackPolicy(bundleIdentifier: Bundle.main.bundleIdentifier)?.redirectURL,
                                  flowType: .pkce,
                                  autoRefreshToken: !AccountStore.offlineReportFixture,
                                  emitLocalSessionAsInitialSession: true),
                       global: .init(logger: nil))
    )
    @Published private(set) var user: User? {
        didSet {
            restoreChatConsent()
            interestsState.restore(owner: user?.id, isAnonymous: user?.isAnonymous, metadata: user?.userMetadata[UserInterests.metadataKey]?.stringValue)
            #if DEBUG && targetEnvironment(simulator)
            if (ProcessInfo.processInfo.environment["PLONN_EXECUTE_FROZEN_ODDS"] == "plonn-20261013-through-sunday" || ProcessInfo.processInfo.environment["PLONN_VERIFY_NBA_CACHE"] == "1"), user?.id.uuidString.lowercased() == "fd31452b-35c7-4cf4-bab7-5bed17072001" {
                Task { await executeAuthorizedFrozenOdds() }
            }
            #endif

            #if DEBUG && targetEnvironment(simulator)
            if ProcessInfo.processInfo.environment["PLONN_OPERATOR_IDENTITY_ONLY"] == "1", let user, !user.isAnonymous {
                try? Data(user.id.uuidString.lowercased().utf8).write(to:URL(fileURLWithPath:"/tmp/plonn-frozen-operator-user-id.txt"),options:.atomic)
            }
            #endif

            guard oldValue?.id != user?.id else { return }
            chatConsentSyncMessage = nil
            interestsSaveMessage = nil
            cancelSocialAuthentication()
            socialAttempt.switchOwner(user?.id)
            referral.switchOwner(user?.id)
            if let next = user,
               let operation = referral.beginAutomatic(owner: next.id, pendingCode: next.userMetadata["pending_referral_code"]?.stringValue) {
                Task { await performReferral(operation) }
            }
        }
    }
    @Published private(set) var interestsState = InterestsAccountState()
    #if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
    func prepareLocalStoreKitUser(_ id: UUID) {
        precondition(LocalStoreKitLab.active)
        user = User(id: id, appMetadata: [:], userMetadata: [:], aud: "authenticated",
                    email: "storekit-lab@example.invalid", createdAt: Date(), updatedAt: Date())
        restoringSession = false
    }
    #endif
    @Published private(set) var interestsSaving = false
    @Published private(set) var interestsSaveMessage: String?
    var shouldOfferInterests: Bool {
        hasSignedInAccount && interestsState.shouldOnboard(
            pending: user?.userMetadata[UserInterests.onboardingKey]?.stringValue,
            createdAt: user?.createdAt)
    }
    func deferInterestsOnboarding() { interestsState.deferOnboarding() }
    func clearInterestsMessage() { interestsSaveMessage = nil }
    func saveInterests(_ value: UserInterests, owner: UUID) async -> Bool {
        guard !busy, !chatConsentUpdating, hasSignedInAccount, user?.id == owner,
              let operation = interestsState.operation(), operation.owner == owner else { return false }
        busy = true; interestsSaving = true; interestsSaveMessage = nil
        defer { busy = false; interestsSaving = false }
        do {
            let encoded = try value.encoded()
            let session = try await client.auth.session
            guard interestsState.accepts(operation), user?.id == owner, session.user.id == owner,
                  !session.isExpired, !session.user.isAnonymous else { throw InterestsError.wrongAccount }
            // Auth merges these keys, preserving display name, consent and referral metadata.
            let updated = try await client.auth.update(user: UserAttributes(data: [
                UserInterests.metadataKey: .string(encoded),
                UserInterests.onboardingKey: .string(value.onboardingCompleted ? "completed_v1" : UserInterests.onboardingPending)
            ]))
            guard interestsState.accepts(operation), user?.id == owner, updated.id == owner else { return false }
            guard UserInterests.decode(updated.userMetadata[UserInterests.metadataKey]?.stringValue) == value else {
                interestsSaveMessage = "Your interests weren’t confirmed as saved. Please try again."; return false
            }
            user = updated
            return true
        } catch {
            guard interestsState.accepts(operation), user?.id == owner else { return false }
            interestsSaveMessage = "Couldn’t confirm your interests were saved. Check your connection and try again."
            return false
        }
    }
    @Published private(set) var chatConsentOwner: UUID?
    @Published private(set) var chatConsentSyncMessage: String?
    @Published private(set) var chatConsentUpdating = false
    private var chatConsentSyncTask: Task<Void, Never>?
    private func restoreChatConsent() {
        guard let user else { chatConsentOwner = nil; return }
        let localChoice = UserDefaults.standard.object(forKey: AccountAIConsentPolicy.preferenceKey(owner: user.id)) as? Bool
        chatConsentOwner = AccountAIConsentPolicy.permits(isAnonymous: user.isAnonymous,
            metadataVersion: user.userMetadata[AccountAIConsentPolicy.metadataKey]?.stringValue,
            localChoice: localChoice) ? user.id : nil
    }
    var hasChatConsent: Bool {
        user?.isAnonymous == false && CloudConfiguration.servicePolicy.mayUseChat(ownerID: user?.id, consentOwnerID: chatConsentOwner)
    }
    func allowChat(for owner: UUID) -> Bool {
        guard CloudConfiguration.servicePolicy.chatAvailable, user?.id == owner, user?.isAnonymous == false else { return false }
        guard !hasChatConsent else { return true }
        UserDefaults.standard.set(true, forKey: AccountAIConsentPolicy.preferenceKey(owner: owner))
        chatConsentOwner = owner
        persistChatConsent(true, owner: owner)
        return true
    }
    func revokeChat(for owner: UUID) {
        guard user?.id == owner else { return }
        UserDefaults.standard.set(false, forKey: AccountAIConsentPolicy.preferenceKey(owner: owner))
        chatConsentOwner = nil
        persistChatConsent(false, owner: owner)
    }
    private func persistChatConsent(_ enabled: Bool, owner: UUID) {
        // Serialize explicit changes so a slow enable cannot overwrite a later revoke.
        let previous = chatConsentSyncTask
        chatConsentSyncMessage = "Saving your AI choice to your account…"
        chatConsentSyncTask = Task { [weak self] in
            await previous?.value
            guard let self, self.user?.id == owner, self.user?.isAnonymous == false,
                  UserDefaults.standard.object(forKey: AccountAIConsentPolicy.preferenceKey(owner: owner)) as? Bool == enabled else { return }
            guard AccountAIConsentPolicy.mayBeginConsentSync(accountBusy: self.busy,
                socialPending: self.socialAttempt.pending != nil, consentUpdating: self.chatConsentUpdating) else {
                self.chatConsentSyncMessage = "Your AI choice is saved on this device. Finish the current account action before changing this choice again to sync it."
                return
            }
            self.chatConsentUpdating = true
            defer { self.chatConsentUpdating = false }
            do {
                let session = try await self.client.auth.session
                guard self.user?.id == owner, session.user.id == owner, !session.isExpired,
                      !session.user.isAnonymous,
                      UserDefaults.standard.object(forKey: AccountAIConsentPolicy.preferenceKey(owner: owner)) as? Bool == enabled else { return }
                let updated = try await self.client.auth.update(user: UserAttributes(data: [
                    AccountAIConsentPolicy.metadataKey: .string(enabled ? AccountAIConsentPolicy.version : "declined_" + AccountAIConsentPolicy.version)
                ]))
                guard self.user?.id == owner, updated.id == owner,
                      UserDefaults.standard.object(forKey: AccountAIConsentPolicy.preferenceKey(owner: owner)) as? Bool == enabled else { return }
                guard AccountAIConsentPolicy.synchronizedChoice(requested: enabled,
                    pending: UserDefaults.standard.object(forKey: AccountAIConsentPolicy.preferenceKey(owner: owner)) as? Bool,
                    metadataVersion: updated.userMetadata[AccountAIConsentPolicy.metadataKey]?.stringValue) else {
                    self.chatConsentSyncMessage = "Your AI choice is saved on this device, but account sync was not confirmed."
                    return
                }
                UserDefaults.standard.removeObject(forKey: AccountAIConsentPolicy.preferenceKey(owner: owner))
                self.user = updated
                self.chatConsentSyncMessage = "AI choice saved to your account."
            } catch {
                guard self.user?.id == owner,
                      UserDefaults.standard.object(forKey: AccountAIConsentPolicy.preferenceKey(owner: owner)) as? Bool == enabled else { return }
                self.chatConsentSyncMessage = "Your AI choice is saved on this device, but could not sync to your account. Other devices may still use the previous choice."
            }
        }
    }
    #if DEBUG && targetEnvironment(simulator)
    private var frozenOperatorStarted = false
    private func executeAuthorizedFrozenOdds() async {
        guard !frozenOperatorStarted else {return};frozenOperatorStarted=true
        let run="plonn-20261013-through-sunday"
        let cacheOnly=ProcessInfo.processInfo.environment["PLONN_VERIFY_NBA_CACHE"] == "1"
        let marker="PLONN.frozenOperator.executed."+run+(cacheOnly ? ".cacheRead" : "")
        guard !UserDefaults.standard.bool(forKey:marker) else {return}
        // Mark before any request: a crash or ambiguous result never retries automatically.
        UserDefaults.standard.set(true,forKey:marker)
        let output=URL(fileURLWithPath:cacheOnly ? "/tmp/plonn-nba-cache-read.json":"/tmp/plonn-frozen-odds-results.json")
        var results=[[String:Any]]()
        func save() {if let bytes=try? JSONSerialization.data(withJSONObject:results,options:[.sortedKeys]) {try? bytes.write(to:output,options:.atomic)}}
        for sport in (cacheOnly ? ["NBA"] : ["NFL","NHL","MLB","NBA"]) {
            do {
                let session=try await client.auth.session
                guard session.user.id.uuidString.lowercased()=="fd31452b-35c7-4cf4-bab7-5bed17072001", user?.id==session.user.id,!session.user.isAnonymous,!session.isExpired else {
                    results.append(["sport":sport,"state":"stopped_auth_mismatch"]);save();return
                }
                var body=["sport":sport,"frozenRun":run]
                if sport=="NBA" {body["phase"]="preseason"}
                results.append(["sport":sport,"state":"request_started","at":ISO8601DateFormatter().string(from:Date())]);save()
                struct CacheRead:Encodable {let sport="NBA",phase="preseason",cacheOnly=true}
                let response:(Data,Int)
                if cacheOnly {
                    response=try await client.functions.invoke("odds",options:.init(headers:["Authorization":"Bearer "+session.accessToken],body:CacheRead(),timeoutInterval:60)) {data,response in (data,response.statusCode)}
                } else {
                    response=try await client.functions.invoke("odds",options:.init(headers:["Authorization":"Bearer "+session.accessToken],body:body,timeoutInterval:60)) {data,response in (data,response.statusCode)}
                }
                guard response.0.count<=8_000_000,let object=try JSONSerialization.jsonObject(with:response.0) as? [String:Any] else {
                    results.append(["sport":sport,"state":"stopped_invalid_response"]);save();return
                }
                var safe:[String:Any]=["sport":sport,"httpStatus":response.1,"at":ISO8601DateFormatter().string(from:Date())]
                for key in ["run","eventCount","marketCount","events","providerHeaders","markets","cacheOnly","fetchedAt","quoteUpdatedAt","servedAt","stale","message"] {if let value=object[key] {safe[key]=value}}
                results.append(safe);save()
                if cacheOnly {
                    let markets=object["markets"] as? [[String:Any]]
                    let valid=response.1==200 && object["cacheOnly"] as? Bool == true && markets?.isEmpty == false && markets!.allSatisfy {$0["sport"] as? String == "NBA"}
                    results.append(["state":valid ? "cache_read_verified":"stopped_cache_read_invalid"]);save();return
                }
                let headers=object["providerHeaders"] as? [String:Any]
                let value=headers?["x-requests-last"]
                let last=(value as? String).flatMap(Int.init) ?? (value as? Int)
                guard response.1==200,(object["run"] as? [String:Any])?["state"] as? String == "succeeded",let last,last>=0,last<=3 else {
                    results.append(["sport":sport,"state":"stopped_unconfirmed_success_or_cost"]);save();return
                }
            } catch {
                results.append(["sport":sport,"state":"stopped_request_error_no_retry","at":ISO8601DateFormatter().string(from:Date())]);save();return
            }
        }
        results.append(["state":"four_requests_succeeded"]);save()
    }
    #endif
    func sharedMatchupReportClient() -> SharedMatchupReportClient? {
        #if DEBUG && targetEnvironment(simulator)
        let previewEnvironment=ProcessInfo.processInfo.environment
        if previewEnvironment["PLONN_MAIN_RECORDED_REPORT_BUNDLE"] != nil || previewEnvironment["PLONN_MAIN_RECORDED_REPORT"] != nil {
            let records:[Data]
            do {
                if let path=previewEnvironment["PLONN_MAIN_RECORDED_REPORT_BUNDLE"] {
                    let handle=try FileHandle(forReadingFrom:URL(fileURLWithPath:path))
                    defer {try? handle.close()}
                    let bytes=try handle.read(upToCount:SharedMatchupLocalCache.maximumBundleBytes+1) ?? Data()
                    records=try SharedMatchupLocalCache.records(bundle:bytes,now:Date())
                } else {
                    guard let bytes=Data(base64Encoded:previewEnvironment["PLONN_MAIN_RECORDED_REPORT"] ?? ""),bytes.count <= SharedMatchupReportValidator.maximumBytes else {return nil}
                    records=[bytes]
                }
            } catch {return nil}
            return SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{"local-recorded-public-report"},transport:{ request in
                let query=URLComponents(url:request.url!,resolvingAgainstBaseURL:false)?.queryItems ?? []
                let values=Dictionary(uniqueKeysWithValues:query.map {($0.name,$0.value ?? "")})
                let event=SharedMatchupEvent(sport:values["sport"] ?? "",eventID:values["eventID"] ?? "",homeTeam:values["homeTeam"] ?? "",awayTeam:values["awayTeam"] ?? "",startISO:values["startISO"] ?? "")
                let body=try SharedMatchupLocalCache.response(records:records,event:event,now:Date())
                return (body,HTTPURLResponse(url:request.url!,statusCode:200,httpVersion:nil,headerFields:nil)!)
            })
        }
        #endif
        if SharedMatchupFrozenArchive.bundled {
            guard user?.isAnonymous == false,let owner=user?.id else {return nil}
            return SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{ @MainActor [weak self] in
                guard let self,self.user?.id==owner else {throw CancellationError()}
                let session=try await self.client.auth.session
                guard self.user?.id==owner,session.user.id==owner,!session.isExpired,!session.user.isAnonymous else {throw SharedMatchupReportError.authenticationRequired}
                return session.accessToken
            },localReader:{event,now in
                try SharedMatchupFrozenArchive.read(SharedMatchupFrozenArchive.records.get(),event:event,now:now)
            })
        }
        guard CloudConfiguration.sharedReportsEnabled, user?.isAnonymous == false, let owner=user?.id else {return nil}
        return SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{ @MainActor [weak self] in
            guard let self,self.user?.id==owner else {throw CancellationError()}
            let session=try await self.client.auth.session
            guard self.user?.id==owner,session.user.id==owner,!session.isExpired,!session.user.isAnonymous else {throw SharedMatchupReportError.authenticationRequired}
            return session.accessToken
        })
    }
    func matchupNarrativeClient() -> MatchupNarrativeClient? {
        guard MatchupNarrativeArchive.bundled || MatchupNarrativeV5Archive.bundled,
              let legacy = sharedMatchupReportClient(), user?.isAnonymous == false else { return nil }
        return MatchupNarrativeClient(tokenProvider: legacy.tokenProvider, reader: { event, now in
            if let current = try MatchupNarrativeV5Archive.read(event: event, now: now) { return current }
            return try MatchupNarrativeArchive.read(event: event, now: now)
        })
    }
    func hostedChatProvider() -> (any AskPLONNProvider)? {
        guard hasChatConsent, let owner = user?.id else { return nil }
        return HostedAskPLONNProvider(tokenProvider: { @MainActor [weak self] in
            guard let self, self.user?.id == owner, self.hasChatConsent else { throw CancellationError() }
            let session = try await self.client.auth.session
            guard self.user?.id == owner, self.hasChatConsent, session.user.id == owner, !session.user.isAnonymous, !session.isExpired else { throw CancellationError() }
            return session.accessToken
        })
    }
    @Published private(set) var referral = ReferralState()

    func syncReferral(owner: UUID, code: String? = nil) async {
        guard user?.id == owner, let operation = referral.beginRequest(owner: owner, code: code) else { return }
        await performReferral(operation)
    }

    private func performReferral(_ operation: ReferralRequest) async {
        guard user?.id == operation.owner, referral.isCurrent(operation) else { return }
        struct Parameters: Encodable { let p_owner: UUID; let p_code: String? }
        do {
            let session = try await client.auth.session
            guard user?.id == operation.owner, referral.isCurrent(operation),
                  session.user.id == operation.owner, !session.isExpired else { referral.fail(token: operation.token); return }
            let reply: ReferralReply = try await client.rpc("apply_referral_code", params: Parameters(p_owner: operation.owner, p_code: operation.code)).execute().value
            guard user?.id == operation.owner, referral.isCurrent(operation) else { return }
            referral.finish(reply, token: operation.token)
        } catch { referral.fail(token: operation.token) }
    }
    @Published private(set) var watches: [CloudWatch] = []
    @Published var message = ""
    @Published var busy = false
    @Published private(set) var authFailureCategory: String?
    @Published private(set) var authFlowDiagnostic: String?
    private let recoveryIntentKey = "plonn.passwordRecoveryIntent.v1"
    private var recoveryIntent: PasswordRecoveryIntent? {
        get { UserDefaults.standard.data(forKey: recoveryIntentKey).flatMap { try? JSONDecoder().decode(PasswordRecoveryIntent.self, from: $0) } }
        set {
            if let newValue, let data = try? JSONEncoder().encode(newValue) { UserDefaults.standard.set(data, forKey: recoveryIntentKey) }
            else { UserDefaults.standard.removeObject(forKey: recoveryIntentKey) }
        }
    }
    private func recoveryFingerprint(_ email: String) -> String {
        SocialAuthNonce.hash(email.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }
    @Published var passwordRecovery = false
    @Published private(set) var socialReadiness = SocialProviderReadiness()
    @Published private(set) var socialCommitting = false
    private var socialAttempt = SocialAuthAttemptState()
    private var socialTask: Task<Void, Never>?
    private var socialTaskID: UUID?
    private var socialCommitOwner: UUID?
    private var deferredSocialAuthEvent: (AuthChangeEvent, Session?)?
    private var appleFlow: NativeAppleAuthorization?
    private var googleFlow: SystemGoogleAuthorization?
    private var handledSocialCallbacks: [String] = []
    private var socialReadinessLoading = false
    private var socialPolicy: SocialAuthPolicy {
        SocialAuthPolicy(bundleIdentifier: Bundle.main.bundleIdentifier, info: Bundle.main.infoDictionary ?? [:], readiness: socialReadiness)
    }

    func socialAuthAvailable(_ provider: SocialAuthProvider, action: SocialAuthAction) -> Bool {
        guard action != .linkIdentity || user?.isAnonymous == false else { return false }
        return socialPolicy.available(provider, action: action)
    }

    /// No settings call occurs in builds whose provider flags are off.
    func refreshSocialReadiness() async {
        guard socialPolicy.needsReadinessCheck, !socialReadinessLoading else { return }
        socialReadinessLoading = true
        defer { socialReadinessLoading = false }
        var request = URLRequest(url: CloudConfiguration.url.appendingPathComponent("auth/v1/settings"))
        request.timeoutInterval = 5
        request.setValue(CloudConfiguration.publishableKey, forHTTPHeaderField: "apikey")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil; configuration.urlCache = nil
        let session = URLSession(configuration: configuration, delegate: SocialSettingsRedirectGuard(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        do {
            let (bytes, response) = try await session.bytes(for: request)
            guard !Task.isCancelled, let response = response as? HTTPURLResponse, response.statusCode == 200,
                  response.url == request.url else { socialReadiness = .init(); return }
            var data = Data()
            for try await byte in bytes {
                guard data.count < 16_384 else { socialReadiness = .init(); return }
                data.append(byte)
            }
            socialReadiness = SocialProviderReadiness.decode(data)
        } catch { socialReadiness = .init() }
    }

    func beginSocialAuthentication(_ provider: SocialAuthProvider, action: SocialAuthAction) {
        guard !busy, !chatConsentUpdating, socialAuthAvailable(provider, action: action) else { return }
        do {
            let nonce = provider == .apple ? try SocialAuthNonce.generate() : nil
            guard let attempt = socialAttempt.begin(provider: provider, action: action, policy: socialPolicy, nonce: nonce) else { return }
            busy = true; socialTaskID = attempt.id
            socialTask = Task { [weak self] in await self?.performSocialAuthentication(attempt) }
        } catch { message = "Sign-in couldn’t start. Your existing account is unchanged." }
    }

    func cancelSocialAuthentication() {
        guard let attempt = socialAttempt.pending, socialAttempt.cancel(attempt) else { return }
        socialTask?.cancel(); appleFlow?.cancel(); googleFlow?.cancel()
        socialTask = nil; socialTaskID = nil; socialCommitOwner = nil; busy = false
    }

    private func checkSocialAttempt(_ attempt: SocialAuthAttempt) throws {
        try Task.checkCancellation()
        guard socialAttempt.isCurrent(attempt, currentOwner: user?.id) else { throw SocialAuthError.invalidAttempt }
    }

    private func performSocialAuthentication(_ attempt: SocialAuthAttempt) async {
        let memory = SocialAuthMemoryStorage()
        defer {
            memory.clear()
            if socialTaskID == attempt.id {
                socialAttempt.endAfterPersistence(attempt)
                socialTask = nil; socialTaskID = nil; socialCommitOwner = nil
                appleFlow = nil; googleFlow = nil; busy = false
                socialCommitting = false
                drainDeferredSocialAuthEvent()
            }
        }
        do {
            try checkSocialAttempt(attempt)
            guard let callback = socialPolicy.callback, let window = NativeAuthPresentation.keyWindow else { throw SocialAuthError.noPresentationAnchor }
            let auth = AuthClient(configuration: .init(url: CloudConfiguration.url.appendingPathComponent("auth/v1"),
                headers: ["apikey": CloudConfiguration.publishableKey], flowType: .pkce, redirectToURL: callback.redirectURL,
                storageKey: "plonn-social-" + attempt.id.uuidString, localStorage: memory, logger: nil, autoRefreshToken: false))
            if attempt.action == .linkIdentity {
                let current = try await client.auth.session
                try checkSocialAttempt(attempt)
                guard current.user.id == attempt.owner, !current.user.isAnonymous, !current.isExpired else { throw SocialAuthError.invalidAttempt }
                _ = try await auth.setSession(accessToken: current.accessToken, refreshToken: current.refreshToken)
                try checkSocialAttempt(attempt)
            }
            let result: Session
            if attempt.provider == .apple {
                let flow = NativeAppleAuthorization(window: window); appleFlow = flow
                let credential = try await flow.authorize(nonceHash: SocialAuthNonce.hash(attempt.nonce!), state: attempt.id.uuidString)
                try checkSocialAttempt(attempt)
                guard let data = credential.identityToken, let token = String(data: data, encoding: .utf8),
                      socialAttempt.acceptsAppleCredential(token: token, state: credential.state, attempt: attempt, currentOwner: user?.id) else { throw SocialAuthError.invalidCredential }
                let credentials = OpenIDConnectCredentials(provider: .apple, idToken: token, nonce: attempt.nonce)
                if attempt.action == .linkIdentity { result = try await auth.linkIdentityWithIdToken(credentials: credentials) }
                else { result = try await auth.signInWithIdToken(credentials: credentials) }
            } else {
                let flow = SystemGoogleAuthorization(window: window); googleFlow = flow
                if attempt.action == .linkIdentity {
                    let response = try await auth.getLinkIdentityURL(provider: .google, scopes: "openid email profile", redirectTo: callback.redirectURL, queryParams: [("prompt", "select_account")])
                    try checkSocialAttempt(attempt)
                    let url = try await googleCallback(response.url, flow: flow, attempt: attempt, callback: callback)
                    result = try await auth.session(from: url)
                } else {
                    result = try await auth.signInWithOAuth(provider: .google, redirectTo: callback.redirectURL, scopes: "openid email profile", queryParams: [("prompt", "select_account")], launchFlow: { [weak self] url in
                        guard let self else { throw CancellationError() }
                        return try await self.googleCallback(url, flow: flow, attempt: attempt, callback: callback)
                    })
                }
            }
            try checkSocialAttempt(attempt)
            guard !result.isExpired, !result.user.isAnonymous,
                  attempt.action != .linkIdentity || result.user.id == attempt.owner else { throw SocialAuthError.invalidCredential }
            // Only this guarded, verified result reaches the existing Keychain.
            // Provider failures/cancellations stayed in attempt-local memory.
            guard socialAttempt.beginCommit(attempt, currentOwner: user?.id, sessionOwner: result.user.id,
                                            isAnonymous: result.user.isAnonymous, isExpired: result.isExpired) else { throw SocialAuthError.invalidAttempt }
            socialCommitting = true
            socialCommitOwner = result.user.id
            let adopted = try await client.auth.setSession(accessToken: result.accessToken, refreshToken: result.refreshToken)
            try checkSocialAttempt(attempt)
            guard client.auth.currentSession?.accessToken == adopted.accessToken,
                  client.auth.currentSession?.user.id == adopted.user.id else { throw SocialAuthError.invalidCredential }
            guard socialAttempt.finish(attempt, currentOwner: user?.id, sessionOwner: adopted.user.id,
                                       isAnonymous: adopted.user.isAnonymous, isExpired: adopted.isExpired) else { throw SocialAuthError.invalidAttempt }
            user = adopted.user
            if attempt.action == .signIn { watches = []; passwordRecovery = false }
            message = attempt.action == .linkIdentity ? "Sign-in provider linked to your existing account." : "Signed in. Your device research is unchanged."
        } catch {
            guard socialAttempt.isCurrent(attempt, currentOwner: user?.id) else { return }
            let cancelled = error is CancellationError || (error as? ASAuthorizationError)?.code == .canceled ||
                (error as? ASWebAuthenticationSessionError)?.code == .canceledLogin
            message = cancelled ? "Sign-in cancelled. Your existing account is unchanged." : "Couldn’t complete provider sign-in. Your existing account is unchanged. Try again or use email."
        }
    }

    private func drainDeferredSocialAuthEvent() {
        guard let (event, session) = deferredSocialAuthEvent else { return }
        deferredSocialAuthEvent = nil
        // Ignore superseded SDK events; never restore a captured old session.
        guard session?.accessToken == client.auth.currentSession?.accessToken else { return }
        applyAuthEvent(event, session: session)
    }

    private func applyAuthEvent(_ event: AuthChangeEvent, session: Session?) {
        restoringSession = false
        let next = session?.isExpired == false ? session?.user : nil
        if next?.id != user?.id { watches = []; passwordRecovery = false }
        user = next
        if event == .passwordRecovery, next != nil { passwordRecovery = true }
        if next == nil { passwordRecovery = false }
    }

    private func googleCallback(_ url: URL, flow: SystemGoogleAuthorization, attempt: SocialAuthAttempt, callback: AuthCallbackPolicy) async throws -> URL {
        try checkSocialAttempt(attempt)
        guard SocialAuthPolicy.allowedGoogleAuthorizationURL(url, authURL: CloudConfiguration.url) else { throw SocialAuthError.invalidCallback }
        let returned = try await flow.authorize(url: url, callbackScheme: callback.scheme)
        try checkSocialAttempt(attempt)
        guard socialAttempt.acceptsOAuthCallback(returned, attempt: attempt, currentOwner: user?.id, callback: callback) else { throw SocialAuthError.invalidCallback }
        handledSocialCallbacks.append(SocialAuthNonce.hash(returned.absoluteString))
        if handledSocialCallbacks.count > 8 { handledSocialCallbacks.removeFirst(handledSocialCallbacks.count - 8) }
        return returned
    }

    func observe() async {
        for await (event, session) in client.auth.authStateChanges {
            if Task.isCancelled { return }
            // The SDK emits the stored session before refreshing it. Never expose
            // an expired session as a signed-in account while that refresh runs.
            if socialAttempt.isCommitting {
                deferredSocialAuthEvent = (event, session)
                continue
            }
            if event == .initialSession, session?.isExpired == true {
                // Let the SDK refresh its persisted session before showing a guest welcome.
                let restored = try? await client.auth.session
                applyAuthEvent(event, session: restored)
            } else {applyAuthEvent(event, session: session)}
        }
    }
    func perform(_ operation: () async throws -> Void) async {
        guard !busy, !chatConsentUpdating else { return }
        busy = true
        authFailureCategory = nil
        message = ""
        defer { busy = false }
        do { try await operation() }
        catch {
            let category: String
            if let error = error as? EmailAuthLinkError { category = error.category }
            else if let error = error as? AuthError {
                switch error {
                case .api(_, let code, _, let response):
                    category = AuthFailurePresentation.category(serverCode: code.rawValue, httpStatus: response.statusCode)
                case .pkceGrantCodeExchange(_, _, let code):
                    category = AuthFailurePresentation.category(serverCode: code, httpStatus: nil)
                case .sessionMissing: category = "session_missing"
                default: category = "auth_request"
                }
            } else if error is URLError { category = "network" }
            else if error is DecodingError { category = "response_format" }
            else { category = "auth_request" }
            authFailureCategory = category
            message = AuthFailurePresentation.message(forCategory: category)
        }
    }
    func signIn(email: String, password: String) async {
        await perform {
            let session = try await client.auth.signIn(email: email.trimmingCharacters(in: .whitespacesAndNewlines), password: password)
            user = session.user
            watches = []; passwordRecovery=false; recoveryIntent=nil
            message = "Signed in. Refresh cloud watchlist to load your saved research."
        }
    }
    func requestSignInLink(email: String) async {
        let value = email.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.contains("@") else { message = "Enter your account email."; return }
        guard let authCallback else { message = "Account links are unavailable in this app build."; return }
        await perform {
            // A new sign-in flow supersedes recovery intent. The SDK creates and
            // stores its own PKCE verifier on this device.
            recoveryIntent = nil; passwordRecovery = false
            defer { authFlowDiagnostic = authCallback.scheme + " · " + Self.authStorage.category }
            try await client.auth.signInWithOTP(email: value, redirectTo: authCallback.redirectURL, shouldCreateUser: false)
            guard Self.authStorage.verifierPresent(key: verifierStorageKey) else { throw EmailAuthLinkError.storageUnavailable }
            message = "Sign-in email requested. Open the new link on this device."
        }
    }
    func signUp(email: String, password: String, displayName: String, referralCode: String = "", allowAI: Bool = false) async {
        guard let authCallback else { message = "Account links are unavailable in this app build."; return }
        await perform {
            let code = ReferralState.normalized(referralCode)
            var metadata: [String: AnyJSON] = ["display_name": .string(displayName.trimmingCharacters(in: .whitespacesAndNewlines)), UserInterests.onboardingKey: .string(UserInterests.onboardingPending)]
            if allowAI { metadata[AccountAIConsentPolicy.metadataKey] = .string(AccountAIConsentPolicy.version) }
            if !code.isEmpty { metadata["pending_referral_code"] = .string(code) }
            let response = try await client.auth.signUp(email: email.trimmingCharacters(in: .whitespacesAndNewlines), password: password, data: metadata, redirectTo: authCallback.redirectURL)
            if let session = response.session {
                user = session.user
                message = "Account created."
            } else {
                message = "Check your email to confirm your account, then sign in here."
                if !code.isEmpty { message += " Your code will be checked after confirmation. It hasn’t been applied yet." }
            }
        }
    }
    func handleLoginURL(_ url: URL) async {
        guard authCallback?.accepts(url) == true, socialAttempt.pending == nil,
              !handledSocialCallbacks.contains(SocialAuthNonce.hash(url.absoluteString)) else { return }
        let fingerprint = SocialAuthNonce.hash(url.absoluteString)
        // A successful one-time link can be delivered twice by the OS/email browser.
        // Do not exchange it again or replace the already-verified session with an error.
        if completedEmailCallbacks.contains(where: { $0.fingerprint == fingerprint && $0.owner == user?.id }) { return }
        await perform {
            authFlowDiagnostic = (authCallback?.scheme ?? "unknown_app") + " · " + AuthCallbackPresence.category(for: url)
            defer { authFlowDiagnostic = (authFlowDiagnostic ?? "callback") + " · " + Self.authStorage.category }
            let failure = EmailAuthCallbackValidation.failure(for: url,
                verifierPresent: Self.authStorage.verifierPresent(key: verifierStorageKey))
            if failure == "sign_in_context" { throw EmailAuthLinkError.missingRequest }
            if failure != nil { throw EmailAuthLinkError.incomplete }
            let session = try await client.auth.session(from: url)
            completedEmailCallbacks.append((fingerprint, session.user.id))
            if completedEmailCallbacks.count > 16 { completedEmailCallbacks.removeFirst() }
            // The pinned SDK emits only signedIn for PKCE. A previously requested
            // recovery may open its form only after SDK verification, for the
            // same email fingerprint and a bounded request window.
            let recovery = !session.isExpired && !session.user.isAnonymous &&
                recoveryIntent?.permitsRecovery(verifiedEmailFingerprint: session.user.email.map(recoveryFingerprint), now: Date()) == true
            user = session.user
            watches = []
            if recovery { passwordRecovery = true }
            message = recovery ? "Recovery link verified. Enter your new password." : "Authentication link verified. You are signed in."
        }
    }
    func signOut() async {
        await perform {
            // This button signs out this installation, not the user's other devices.
            try await client.auth.signOut(scope: .local)
            user = nil; watches = []; passwordRecovery=false; recoveryIntent=nil
            message = "Signed out. Device-only watchlist is unchanged."
        }
    }
    func requestPasswordReset(email:String) async {
        let value=email.trimmingCharacters(in:.whitespacesAndNewlines)
        guard value.contains("@") else {message="Enter your account email.";return}
        guard let authCallback else {message="Account links are unavailable in this app build.";return}
        await perform {
            // Store intent only; the SDK retains the verifier in its existing Keychain.
            // Keep intent after an ambiguous send failure because delivery can succeed.
            recoveryIntent = PasswordRecoveryIntent(emailFingerprint: recoveryFingerprint(value), requestedAt: Date())
            try await client.auth.resetPasswordForEmail(value,redirectTo:authCallback.redirectURL)
            message="If this address has an account, a password reset email will arrive. Open the link on this device."
        }
    }
    func updateRecoveredPassword(_ password:String) async {
        guard passwordRecovery,user != nil,password.count>=8 else {message="Open a valid recovery email and use at least 8 characters.";return}
        await perform {
            _ = try await client.auth.update(user:UserAttributes(password:password))
            passwordRecovery=false; recoveryIntent=nil
            message="Password updated. You are signed in."
        }
    }
    func refresh() async {
        guard hasSignedInAccount else {access.present(feature:.savedResearch);return}
        await perform {
            let session = try await client.auth.session
            let owner = session.user.id
            let rows: [CloudWatch] = try await client.from("watchlist_items").select().eq("user_id", value: owner).order("created_at").execute().value
            guard user?.id == owner else { return }
            watches = rows
            message = "Cloud watchlist refreshed."
        }
    }
    func upload(_ markets: [Market]) async {
        guard hasSignedInAccount else {access.present(feature:.savedResearch);return}
        guard !markets.isEmpty else { message = "Save a market on this device first."; return }
        await perform {
            let session = try await client.auth.session
            let rows = markets.map { CloudWatchInsert(user_id: session.user.id, market_id: $0.id, snapshot: $0) }
            try await client.from("watchlist_items").upsert(rows, onConflict: "user_id,market_id").execute()
            message = "Device selections copied to your account. Refresh to view them."
        }
    }
    func remove(_ row: CloudWatch) async {
        guard hasSignedInAccount else {access.present(feature:.savedResearch);return}
        await perform {
            let owner = try await client.auth.session.user.id
            guard owner == row.user_id else { return }
            try await client.from("watchlist_items").delete().eq("id", value: row.id).eq("user_id", value: owner).execute()
            watches.removeAll { $0.id == row.id }
            message = "Removed from cloud watchlist."
        }
    }
    func deleteAccount() async {
        guard !busy, !chatConsentUpdating else { return }
        guard let deletingOwner = user?.id else { return }
        busy = true; message = "Deleting your account…"
        defer { busy = false }
        do {
            struct DeletionReply: Decodable { let status: String; let message: String }
            let result: DeletionReply = try await client.functions.invoke("delete-account", options: .init(timeoutInterval: 70))
            guard result.status == "deleted" else { message = result.message; return }
            LabSubmissionStore().deleteAccountRecords(owner: deletingOwner)
            UserDefaults.standard.removeObject(forKey: AccountAIConsentPolicy.preferenceKey(owner: deletingOwner))
            chatConsentOwner = nil
            try? await client.auth.signOut(scope: .local)
            user = nil; watches = []; passwordRecovery = false
            message = "Account deleted. Device-only watchlist is unchanged."
        } catch {
            if case let FunctionsError.httpError(_, data) = error,
               let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let detail = body["message"] as? String {
                message = detail
            } else { message = "Deletion could not be confirmed. Please tap Delete account again to retry safely." }
        }
    }

}

private final class SocialSettingsRedirectGuard: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
}

private final class SocialAuthMemoryStorage: AuthLocalStorage, @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String: Data] = [:]
    func store(key: String, value: Data) throws { lock.lock(); defer { lock.unlock() }; values[key] = value }
    func retrieve(key: String) throws -> Data? { lock.lock(); defer { lock.unlock() }; return values[key] }
    func remove(key: String) throws { lock.lock(); defer { lock.unlock() }; values.removeValue(forKey: key) }
    func clear() { lock.lock(); defer { lock.unlock() }; values.removeAll() }
}

@MainActor private enum NativeAuthPresentation {
    static var keyWindow: UIWindow? {
        UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .filter { $0.activationState == .foregroundActive }.flatMap(\.windows).first(where: \.isKeyWindow)
    }
}

@MainActor private final class NativeAppleAuthorization: NSObject, ASAuthorizationControllerDelegate, ASAuthorizationControllerPresentationContextProviding {
    private let window: UIWindow
    private var controller: ASAuthorizationController?
    private var continuation: CheckedContinuation<ASAuthorizationAppleIDCredential, Error>?
    init(window: UIWindow) { self.window = window }
    func authorize(nonceHash: String, state: String) async throws -> ASAuthorizationAppleIDCredential {
        try Task.checkCancellation()
        return try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                let request = ASAuthorizationAppleIDProvider().createRequest()
                request.requestedScopes = [.email]; request.nonce = nonceHash; request.state = state
                let controller = ASAuthorizationController(authorizationRequests: [request])
                self.controller = controller; controller.delegate = self; controller.presentationContextProvider = self
                controller.performRequests()
            }
        }, onCancel: { Task { @MainActor [weak self] in self?.cancel() } })
    }
    func cancel() { controller?.cancel(); complete(.failure(CancellationError())) }
    func presentationAnchor(for controller: ASAuthorizationController) -> ASPresentationAnchor { window }
    func authorizationController(controller: ASAuthorizationController, didCompleteWithAuthorization authorization: ASAuthorization) {
        guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential else { complete(.failure(SocialAuthError.invalidCredential)); return }
        complete(.success(credential))
    }
    func authorizationController(controller: ASAuthorizationController, didCompleteWithError error: Error) { complete(.failure(error)) }
    private func complete(_ result: Result<ASAuthorizationAppleIDCredential, Error>) {
        guard let continuation else { return }
        self.continuation = nil; controller = nil
        continuation.resume(with: result)
    }
}

@MainActor private final class SystemGoogleAuthorization: NSObject, ASWebAuthenticationPresentationContextProviding {
    private let window: UIWindow
    private var session: ASWebAuthenticationSession?
    private var continuation: CheckedContinuation<URL, Error>?
    init(window: UIWindow) { self.window = window }
    func authorize(url: URL, callbackScheme: String) async throws -> URL {
        try Task.checkCancellation()
        return try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                let session = ASWebAuthenticationSession(url: url, callbackURLScheme: callbackScheme) { [weak self] url, error in
                    Task { @MainActor in
                        if let error { self?.complete(.failure(error)) }
                        else if let url { self?.complete(.success(url)) }
                        else { self?.complete(.failure(SocialAuthError.invalidCallback)) }
                    }
                }
                self.session = session; session.presentationContextProvider = self
                session.prefersEphemeralWebBrowserSession = true
                if !session.start() { complete(.failure(SocialAuthError.noPresentationAnchor)) }
            }
        }, onCancel: { Task { @MainActor [weak self] in self?.cancel() } })
    }
    func cancel() { session?.cancel(); complete(.failure(CancellationError())) }
    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor { window }
    private func complete(_ result: Result<URL, Error>) {
        guard let continuation else { return }
        self.continuation = nil; session = nil
        continuation.resume(with: result)
    }
}

struct AccountView: View {
    @EnvironmentObject var account: AccountStore
    @EnvironmentObject var store: Store
    @State private var email = ""
    @State private var password = ""
    @State private var confirmDelete = false
    @State private var manageSubscriptions = false
    @State private var creating = false
    init(initiallyCreating: Bool = false) { _creating = State(initialValue:initiallyCreating) }
    @State private var displayName = ""
    @State private var referralCode = ""
    @State private var confirmation = ""
    @State private var allowAI = false
    private var validEmail:Bool {let value=email.trimmingCharacters(in:.whitespacesAndNewlines);return value.contains("@") && value.split(separator:"@").count == 2 && value.split(separator:"@").last?.contains(".") == true}
    private var valid:Bool {validEmail && !password.isEmpty && (!creating || (!displayName.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && password.count >= 8 && password == confirmation))}
    var body: some View {
        Form {
            if account.hasSignedInAccount && (!account.message.isEmpty || account.busy || account.socialCommitting) {
                Section {
                    if !account.message.isEmpty {
                        accountStatusMessage
                    }
                    if account.busy { PLONNLoader() }
                    if account.socialCommitting { Text("Finishing secure sign-in. Account actions will be available when it completes.").font(.caption) }
                }
            }
            if let user = account.user, account.hasSignedInAccount {
                Section {
                    Text(user.email ?? "Signed in")
                    Button("Sign out") { Task { await account.signOut() } }
                }
                Section {
                    NavigationLink {
                        InterestsView().id(user.id)
                    } label: {
                        Label("Interests & alerts", systemImage: "heart.text.square")
                    }
                }
                if !user.isAnonymous {
                    Section("AI research") {
                        Text(AccountAIConsentPolicy.disclosure).font(.caption)
                        Toggle("Allow requested AI research", isOn: Binding(
                            get: { account.hasChatConsent },
                            set: { enabled in
                                if enabled { _ = account.allowChat(for: user.id) }
                                else { account.revokeChat(for: user.id) }
                            }))
                            .disabled(!CloudConfiguration.servicePolicy.chatAvailable)
                            .accessibilityIdentifier("account-ai-consent")
                        if let status = account.chatConsentSyncMessage {
                            Text(status).font(.caption).foregroundStyle(.secondary)
                                .accessibilityIdentifier("account-ai-consent-sync")
                        }
                        Link("PLONN privacy policy", destination: PLONNServicePolicy.privacyURL)
                    }
                }
                ReferralSection()
                if account.socialAuthAvailable(.apple, action: .linkIdentity) || account.socialAuthAvailable(.google, action: .linkIdentity) {
                    Section("Sign-in methods") {
                        Text("Link a provider to this signed-in account. Your email sign-in and saved research stay with the same account.").font(.caption)
                        socialButtons(action: .linkIdentity)
                    }
                }
                Section {
                    NavigationLink("My Models") {MyModelsView()}

                }
                Section("Cloud watchlist") {
                    Text("Copy device selections to your account, then refresh. Removing cloud selections does not remove device copies.").font(.system(.caption, design: .monospaced))
                    Button("Copy device watchlist to cloud") { Task { await account.upload(store.saved) } }
                    Button("Refresh cloud watchlist") { Task { await account.refresh() } }
                    ForEach(account.watches) { row in
                        VStack(alignment: .leading) {
                            Text(row.snapshot.name).font(.system(.headline, design: .monospaced))
                            Text(row.snapshot.label)
                            Text("Saved snapshot · " + (row.snapshot.mode == "demo" ? "Sample" : "Sportsbook")).font(.system(.caption, design: .monospaced))
                            Button("Remove cloud selection", role: .destructive) { Task { await account.remove(row) } }
                        }
                    }
                }
                Section("Delete account") {
                    Text("Permanently removes your account and its cloud profile, watchlist, saved parlays, private model versions, hosted results and account-linked usage records. App Store subscriptions are not cancelled by account deletion. Device-only selections remain until cleared in Settings.").font(.system(.caption, design: .monospaced))
                    Button("Manage Apple subscriptions") { manageSubscriptions = true }
                    Button("Delete account", role: .destructive) { confirmDelete = true }
                }
            } else {
                Section {
                    VStack(alignment:.leading,spacing:10) {
                        Brand()
                        Text(creating ? "Create your account":"Welcome back").font(.system(.title2, design: .monospaced).bold())
                        if account.message.isEmpty {
                            Text(creating ? "Save your research and make PLONN yours.":"Sign in to access your saved research.").font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary)
                        } else {
                            accountStatusMessage
                        }
                        if account.busy { PLONNLoader() }
                        if account.socialCommitting { Text("Finishing secure sign-in…").font(.caption).foregroundStyle(.secondary) }
                    }.padding(.vertical,10)
                    Picker("Account action",selection:$creating) {Text("Sign in").tag(false);Text("Sign up").tag(true)}.pickerStyle(.segmented)
                }
                if account.socialAuthAvailable(.apple, action: .signIn) || account.socialAuthAvailable(.google, action: .signIn) {
                    Section {
                        socialButtons(action: .signIn)
                        Text("Provider sign-in can create an account. To add a provider to an existing account, sign in to that account first, then choose Link.").font(.caption).foregroundStyle(.secondary)
                        Text("If you have a referral code, apply it in Account after provider sign-in.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Section(creating ? "Your details":"Sign in") {
                    if creating {TextField("Display name",text:$displayName).textContentType(.nickname).autocorrectionDisabled()}
                    TextField("Email", text: $email).textContentType(.emailAddress).keyboardType(.emailAddress).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Password", text: $password).textContentType(creating ? .newPassword:.password)
                    if creating {
                        SecureField("Confirm password",text:$confirmation).textContentType(.newPassword)
                        TextField("Referral code (optional)", text: $referralCode)
                            .textInputAutocapitalization(.characters).autocorrectionDisabled()
                            .accessibilityIdentifier("signup-referral-code")
                        Text("We’ll check your code after you confirm your email.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                        Text("At least 8 characters").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                        if !confirmation.isEmpty && password != confirmation {Text("Passwords don’t match").font(.system(.caption, design: .monospaced)).foregroundStyle(.red)}
                        Text(AccountAIConsentPolicy.disclosure).font(.caption)
                        Toggle("Allow requested AI research (optional)", isOn: $allowAI)
                            .accessibilityIdentifier("signup-ai-consent")
                        Link("PLONN privacy policy", destination: PLONNServicePolicy.privacyURL)
                    }
                    Button {Task {
                        if creating {await account.signUp(email:email,password:password,displayName:displayName,referralCode:referralCode,allowAI:allowAI)} else {await account.signIn(email:email,password:password)}
                        password="";confirmation=""
                    }} label: {Text(creating ? "Create account":"Sign in").font(.system(.headline, design: .monospaced)).frame(maxWidth:.infinity).padding(.vertical,8)}
                    .plonnGlassButton(prominent:true).disabled(!valid)
                    if !creating {
                        Button("Email sign-in link") {Task {await account.requestSignInLink(email:email)}}
                            .disabled(!validEmail).accessibilityIdentifier("account-email-signin-link")
                        Button("Forgot password?") {Task {await account.requestPasswordReset(email:email)}}.disabled(!validEmail)
                    }
                    if creating {Text("We’ll email you a confirmation link.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
                }

            }
            ProSubscriptionSection()
            Section("Data use") {
                Text("Account email, private model configurations, explicitly copied selections, and previously published community posts are stored in PLONN's Supabase project. Community display names, captions, selections, and reposts are visible to signed-in members. Login sessions are stored in the iOS Keychain. We do not collect payments in this beta.").font(.system(.caption, design: .monospaced))
            }
        }
        .plonnGlassButton()
        .disabled(account.busy || account.chatConsentUpdating)
        .task { await account.refreshSocialReadiness() }
        .navigationTitle("Account")
        .onChange(of:creating) {_,_ in password="";confirmation="";referralCode=""}
        .onChange(of:account.user?.id) {_,_ in referralCode="";password="";confirmation=""}
        .manageSubscriptionsSheet(isPresented: $manageSubscriptions)
        .confirmationDialog("Permanently delete your PLONN account?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete permanently", role: .destructive) { Task { await account.deleteAccount() } }
        }
    }
    private var accountStatusMessage: some View {
        HStack(alignment: .top, spacing: 10) {
            if account.authFailureCategory != nil {
                Image(systemName: "exclamationmark.triangle.fill")
                    .accessibilityHidden(true)
            }
            Text(account.message)
                .accessibilityLabel(account.authFailureCategory == nil ? account.message : "Warning. " + account.message)
                .accessibilityIdentifier("cloud-status")
                #if DEBUG && targetEnvironment(simulator)
                .accessibilityValue(ProcessInfo.processInfo.environment["PLONN_AUTH_DIAGNOSTICS"] == "1" ? (account.authFlowDiagnostic ?? "") : "")
                #endif
        }
        .font(.system(.subheadline, design: .monospaced))
        .foregroundStyle(account.authFailureCategory == nil ? Color.secondary : Color.red)
    }
    @ViewBuilder private func socialButtons(action: SocialAuthAction) -> some View {
        if account.socialAuthAvailable(.apple, action: action) {
            Button { account.beginSocialAuthentication(.apple, action: action) } label: {
                Label(action == .linkIdentity ? "Link Apple" : "Continue with Apple", systemImage: "apple.logo").frame(maxWidth: .infinity).padding(.vertical, 8)
            }.plonnGlassButton().accessibilityIdentifier(action == .linkIdentity ? "link-apple-account" : "signin-apple-account")
        }
        if account.socialAuthAvailable(.google, action: action) {
            Button { account.beginSocialAuthentication(.google, action: action) } label: {
                Text(action == .linkIdentity ? "Link Google" : "Continue with Google").frame(maxWidth: .infinity).padding(.vertical, 8)
            }.plonnGlassButton().accessibilityIdentifier(action == .linkIdentity ? "link-google-account" : "signin-google-account")
        }
    }
}

struct ReferralSection: View {
    @EnvironmentObject private var account: AccountStore
    @State private var code = ""
    var body: some View {
        Section("Referral code") {
            Text(account.referral.message)
                .font(.system(.callout, design: .monospaced))
                .accessibilityIdentifier("referral-status")
            if account.referral.appliedCode == nil {
                TextField("Referral code", text: $code)
                    .textInputAutocapitalization(.characters).autocorrectionDisabled()
                    .accessibilityIdentifier("account-referral-code")
                Button(account.referral.busy ? "Checking…" : "Apply code") {
                    guard let owner = account.user?.id else { return }
                    Task { await account.syncReferral(owner: owner, code: code) }
                }
                .font(.system(.headline, design: .monospaced)).plonnGlassButton()
                .disabled(account.referral.busy || ReferralState.normalized(code).isEmpty)
                .accessibilityIdentifier("apply-referral-code")
                Button("Refresh code status") {
                    guard let owner = account.user?.id else { return }
                    Task { await account.syncReferral(owner: owner) }
                }.disabled(account.referral.busy)
            }
        }
        .onChange(of: account.user?.id) { _, _ in code = "" }
    }
}

// MARK: - Community
private struct CommunityMarketSnapshot: Decodable {
    let value: Market?
    init(from decoder:Decoder) throws {value=try? Market(from:decoder)}
}
struct CommunityPost: Decodable, Identifiable {
    let id: UUID
    let user_id: UUID
    let author_name: String
    let caption: String
    let legs: [Market]
    let created_at: String
    enum CodingKeys:String,CodingKey {case id,user_id,author_name,caption,legs,created_at}
    init(from decoder:Decoder) throws {
        let c=try decoder.container(keyedBy:CodingKeys.self)
        id=try c.decode(UUID.self,forKey:.id);user_id=try c.decode(UUID.self,forKey:.user_id)
        author_name=try c.decode(String.self,forKey:.author_name);caption=try c.decode(String.self,forKey:.caption)
        legs=try c.decode([CommunityMarketSnapshot].self,forKey:.legs).compactMap(\.value)
        created_at=try c.decode(String.self,forKey:.created_at)
    }
}
private struct CommunityPostInsert: Encodable {
    let user_id: UUID
    let author_name: String
    let caption: String
    let legs: [Market]
}
private struct CommunityRepost: Codable { let user_id: UUID; let post_id: UUID; var author_name: String = "Member" }
private struct CommunityReport: Encodable { let user_id: UUID; let post_id: UUID; let reason: String }
private struct CommunityBlock: Encodable { let user_id: UUID; let blocked_id: UUID }

@MainActor final class CommunityStore: ObservableObject {
    @Published var posts: [CommunityPost] = []
    @Published var reposted: Set<UUID> = []
    @Published var sharedBy: [UUID:String] = [:]
    @Published var busy = false
    @Published var message: String?
    func refresh(_ account: AccountStore) async {
        guard let owner=account.user?.id else { posts=[]; reposted=[]; sharedBy=[:]; return }
        await perform {
            let rows: [CommunityPost] = try await account.client.from("community_posts").select().order("created_at",ascending:false).limit(100).execute().value
            let shares: [CommunityRepost] = try await account.client.from("community_reposts").select().order("created_at",ascending:false).limit(200).execute().value
            guard account.user?.id == owner else { return }
            posts=rows; reposted=Set(shares.filter{$0.user_id == owner}.map(\.post_id)); sharedBy=[:]
            for share in shares where sharedBy[share.post_id] == nil {sharedBy[share.post_id]=share.author_name}
        }
    }
    func perform(_ work: () async throws -> Void) async {
        guard !busy else {return}; busy=true; message=nil
        defer {busy=false}
        do {try await work()} catch {message="Community couldn’t connect. Try again shortly. Your draft stays on this device."}
    }
    func publish(_ account:AccountStore, name:String, caption:String, legs:[Market]) async -> Bool {
        guard let owner=account.user?.id, (2...8).contains(legs.count), legs.allSatisfy({$0.mode == "live"}) else {return false}
        var success=false
        await perform {
            let row=CommunityPostInsert(user_id:owner,author_name:name,caption:caption,legs:legs)
            try await account.client.from("community_posts").insert(row).execute()
            success=true
        }
        if success {await refresh(account)}
        return success
    }
    func repost(_ post:CommunityPost, account:AccountStore) async {
        guard let owner=account.user?.id else{return}
        await perform {
            if reposted.contains(post.id) {
                try await account.client.from("community_reposts").delete().eq("user_id",value:owner).eq("post_id",value:post.id).execute()
                reposted.remove(post.id)
            } else {
                try await account.client.from("community_reposts").insert(CommunityRepost(user_id:owner,post_id:post.id,author_name:String((UserDefaults.standard.string(forKey:"communityDisplayName").flatMap{$0.isEmpty ? nil:$0} ?? "Member").prefix(40)))).execute()
                reposted.insert(post.id)
            }
        }
    }
    func report(_ post:CommunityPost, reason:String, account:AccountStore) async {
        guard let owner=account.user?.id else{return}
        await perform {
            try await account.client.from("community_reports").insert(CommunityReport(user_id:owner,post_id:post.id,reason:reason)).execute()
            message="Report submitted for review."
        }
    }
    func block(_ post:CommunityPost, account:AccountStore) async {
        guard let owner=account.user?.id else{return}
        await perform {
            try await account.client.from("community_blocks").insert(CommunityBlock(user_id:owner,blocked_id:post.user_id)).execute()
            posts.removeAll{$0.user_id == post.user_id}
        }
    }
    func remove(_ post:CommunityPost, account:AccountStore) async {
        await perform {
            try await account.client.from("community_posts").delete().eq("id",value:post.id).execute()
            posts.removeAll{$0.id == post.id}
        }
    }
}

struct CommunityView: View {
    @EnvironmentObject private var account:AccountStore
    @StateObject private var community=CommunityStore()
    @State private var filter="Feed"
    @State private var compose=false
    @State private var report:CommunityPost?
    @State private var profile:CommunityPost?
    private var visible:[CommunityPost] {
        community.posts.filter {filter == "My posts" ? $0.user_id == account.user?.id : filter == "Reposts" ? community.reposted.contains($0.id) : true}
    }
    var body:some View {AccountRequiredView(feature:.savedResearch){protectedBody}}
    private var protectedBody:some View {
        NavigationStack {
            ScrollView {
                VStack(alignment:.leading,spacing:20) {
                    HStack {Brand();Spacer();NavigationLink {SettingsView()} label:{Image(systemName:"person.crop.circle").font(.system(.title2, design: .monospaced))}}
                    HStack(alignment:.top) {
                        VStack(alignment:.leading,spacing:5){Text("Community").font(.system(.largeTitle, design: .monospaced).bold());Text("Share the thinking behind your picks.").font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary)}
                        Spacer()
                        Button {compose=true} label:{Image(systemName:"square.and.pencil").font(.system(.title2, design: .monospaced)).padding(12).background(navySurface,in:RoundedRectangle(cornerRadius:12))}.accessibilityLabel("Create a parlay post")
                    }
                    Picker("Community feed",selection:$filter){ForEach(["Feed","My posts","Reposts"],id:\.self){Text($0)}}.pickerStyle(.segmented)
                    if let message=community.message {Text(message).font(.system(.callout, design: .monospaced)).foregroundStyle(.secondary)}
                    if community.busy && community.posts.isEmpty {PLONNLoader().frame(maxWidth:.infinity).padding(40)}
                    else if account.user == nil {
                        ContentUnavailableView {Label("Your picks. Your community.",systemImage:"person.2")} description:{Text("Sign in to share parlays and explore other members’ research.")} actions:{NavigationLink{AccountView()}label:{Text("Sign in")}.plonnGlassButton(prominent:true);Button("Create a draft"){compose=true}}
                    } else if visible.isEmpty {
                        ContentUnavailableView("No posts yet",systemImage:"bubble.left.and.bubble.right",description:Text("Share a parlay from your Builder to start the conversation."))
                    }
                    ForEach(visible){post in
                        if let member=community.sharedBy[post.id] {Label(member+" reposted",systemImage:"arrow.2.squarepath").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
                        CommunityPostCard(post:post,reposted:community.reposted.contains(post.id),onProfile:{profile=post},onRepost:{Task{await community.repost(post,account:account)}}) {
                            if post.user_id == account.user?.id {Button("Delete post",role:.destructive){Task{await community.remove(post,account:account)}}}
                            else {Button("Report post"){report=post};Button("Block member",role:.destructive){Task{await community.block(post,account:account)}}}
                        }
                    }
                    NavigationLink("Community guidelines"){CommunityGuidelines()}.font(.system(.footnote, design: .monospaced)).foregroundStyle(.secondary)
                }.padding()
            }.background(navy).toolbar(.hidden,for:.navigationBar)
            .plonnRefreshable {await community.refresh(account)}
            .task(id:account.user?.id){await community.refresh(account)}
            .sheet(isPresented:$compose){CommunityComposer(community:community)}
            .sheet(item:$profile){person in NavigationStack{ScrollView{VStack(alignment:.leading,spacing:16){Image(systemName:"person.crop.circle.fill").font(.system(size:52,design:.monospaced));Text(person.author_name).font(.system(.title, design: .monospaced).bold());Text("Member research · Results are not verified").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary);ForEach(community.posts.filter{$0.user_id == person.user_id}){post in CommunitySlip(legs:post.legs);Text(post.caption)}}.padding()}.navigationTitle("Member")}}
            .confirmationDialog("Report this post",isPresented:Binding(get:{report != nil},set:{if !$0 {report=nil}}),titleVisibility:.visible){ForEach(["Spam","Abuse","Misleading content","Other"],id:\.self){reason in Button(reason){if let post=report {Task{await community.report(post,reason:reason,account:account)}}}}}
        }
    }
}

struct CommunitySlip:View {
    let legs:[Market]
    private var combined:String {
        guard legs.count>1,Set(legs.map(\.event)).count == legs.count,legs.allSatisfy({$0.price.isFinite && abs($0.price)>=100}) else{return "Quote needed"}
        let decimal=legs.reduce(1.0){result,leg in let p=leg.price;return result*(p>0 ? 1+p/100:1+100/abs(p))}
        return odds(decimal>=2 ? (decimal-1)*100 : -100/(decimal-1))
    }
    var body:some View {
        VStack(alignment:.leading,spacing:14){
            ForEach(legs){leg in HStack(spacing:10){if leg.kind == "prop" {PlayerPortrait(market:leg)};VStack(alignment:.leading,spacing:4){Text(leg.name).font(.system(.subheadline, design: .monospaced).bold());Text(leg.label).font(.system(.subheadline, design: .monospaced));Text(leg.sport+" · "+leg.book).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)};Spacer();Text(odds(leg.price)).font(.system(.subheadline, design: .monospaced).weight(.semibold)).monospacedDigit()}}
            Divider()
            HStack{Text(legs.count<2 ? "Selections unavailable":"\(legs.count)-leg parlay").font(.system(.subheadline, design: .monospaced).bold());Spacer();Text(combined).font(.system(.title3, design: .monospaced).bold()).monospacedDigit()}
            Text("Estimated combined odds · Not a sportsbook quote").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
        }.padding().background(navySurface,in:RoundedRectangle(cornerRadius:16))
    }
}
struct CommunityPostCard<Actions:View>:View {
    let post:CommunityPost
    let reposted:Bool
    let onProfile:()->Void
    let onRepost:()->Void
    @ViewBuilder let actions:()->Actions
    var body:some View {
        VStack(alignment:.leading,spacing:12){
            HStack{Button(action:onProfile){HStack{Image(systemName:"person.crop.circle.fill").font(.system(.title, design: .monospaced));VStack(alignment:.leading){Text(post.author_name).font(.system(.subheadline, design: .monospaced).bold());Text(String(post.created_at.prefix(10))).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}}.foregroundStyle(.primary)};Spacer();Menu{actions()} label:{Image(systemName:"ellipsis").padding(10)}.accessibilityLabel("Post options")}
            if !post.caption.isEmpty {Text(post.caption).font(.system(.subheadline, design: .monospaced))}
            CommunitySlip(legs:post.legs)
            HStack{Button(action:onRepost){Label(reposted ? "Reposted":"Repost",systemImage:"arrow.2.squarepath")};Spacer();Text("Member-shared · Unverified").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)}.font(.system(.subheadline, design: .monospaced))
            Divider()
        }
    }
}
struct CommunityComposer:View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var account:AccountStore
    @EnvironmentObject private var store:Store
    @ObservedObject var community:CommunityStore
    @AppStorage("communityDraftCaption") private var caption=""
    @AppStorage("communityDisplayName") private var name=""
    @AppStorage("communityDraftLegs") private var draftData=Data()
    @State private var legs:[Market]=[]
    @State private var agreed=false
    @State private var saved=false
    private var valid:Bool {account.user != nil && (2...8).contains(legs.count) && legs.allSatisfy{$0.mode == "live"} && !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && name.count<=40 && caption.count<=500 && agreed}
    var body:some View {AccountRequiredView(feature:.savedResearch){protectedBody}}
    private var protectedBody:some View {
        NavigationStack{ScrollView{VStack(alignment:.leading,spacing:18){
            Text("Share your parlay").font(.system(.title, design: .monospaced).bold())
            TextField("Display name",text:$name).textFieldStyle(.roundedBorder)
            TextField("What’s your thinking?",text:$caption,axis:.vertical).lineLimit(3...6).textFieldStyle(.roundedBorder)
            Text("\(caption.count)/500").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            if legs.isEmpty {ContentUnavailableView("Add selections in Builder",systemImage:"square.stack",description:Text("Choose two to eight selections, then return here to share them."))}
            else {CommunitySlip(legs:legs)}
            Button("Use current Builder selections"){legs=store.legs;saveDraft()}.disabled(store.legs.isEmpty)
            if legs.contains(where:{$0.mode != "live"}) {Text("Sample selections can be saved as a draft, but cannot be published.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
            Toggle("I agree to the community guidelines",isOn:$agreed)
            NavigationLink("Read guidelines"){CommunityGuidelines()}.font(.system(.footnote, design: .monospaced))
            if account.user == nil {NavigationLink("Sign in to publish"){AccountView()}}
            if let message=community.message {Text(message).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
            Button{Task{saveDraft();if await community.publish(account,name:name.trimmingCharacters(in:.whitespacesAndNewlines),caption:caption,legs:legs){saved=true;caption="";draftData=Data();dismiss()}}}label:{HStack{Spacer();if community.busy {PLONNLoader()} else {Text("Publish parlay").bold()};Spacer()}.padding(8)}.plonnGlassButton(prominent:true).disabled(!valid || community.busy)
            Text("Posted selections are a snapshot of your research, not a verified betting slip or result.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
        }.padding()}.background(navy).navigationTitle("New post").navigationBarTitleDisplayMode(.inline).toolbar{ToolbarItem(placement:.cancellationAction){Button("Save draft"){saveDraft();dismiss()}}}
        .onAppear{legs=(try? JSONDecoder().decode([Market].self,from:draftData)) ?? store.legs}
        .onDisappear{if !saved {saveDraft()}}
        }
    }
    private func saveDraft(){guard account.hasSignedInAccount else{return};if let data=try? JSONEncoder().encode(legs){draftData=data}}
}
struct CommunityGuidelines:View {
    var body:some View {List{Section("Share useful research"){Text("Explain your selections. Respect other members. Never promise guaranteed wins or misrepresent a result.")};Section("Keep it safe"){Text("No harassment, spam, promotions, private information, or requests for money. Use the post menu to report content or block a member.")};Section("Your content"){Text("Only share content you own. You can delete your posts. Deleting your account also removes your community posts, reposts, and reports.")}}.navigationTitle("Community guidelines")}
}


struct PasswordRecoveryView:View {
    @EnvironmentObject private var account:AccountStore
    @Environment(\.dismiss) private var dismiss
    @State private var password=""
    @State private var confirmation=""
    var body:some View {
        NavigationStack {
            Form {
                Section("Set a new password") {
                    SecureField("New password",text:$password).textContentType(.newPassword)
                    SecureField("Confirm new password",text:$confirmation).textContentType(.newPassword)
                    Text("At least 8 characters").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    Button("Update password") {Task {await account.updateRecoveredPassword(password);password="";confirmation=""}}
                        .disabled(account.busy || password.count<8 || password != confirmation || !account.passwordRecovery)
                }
                Text(account.message).font(.system(.caption, design: .monospaced))
            }.plonnGlassButton().navigationTitle("Reset password")
             .toolbar {ToolbarItem(placement:.cancellationAction){Button("Cancel"){account.passwordRecovery=false;dismiss()}}}
        }
    }
}
