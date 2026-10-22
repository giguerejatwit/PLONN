import SwiftUI

/// The same host handles first launch, feature prompts and recovery links.
struct PreviewAccessView: View {
    @EnvironmentObject private var account: AccountStore
    @EnvironmentObject private var access: AccountAccessCoordinator
    @State private var onboardingOwner: UUID?
    private var canEnterApp: Bool {
        #if DEBUG && targetEnvironment(simulator)
        // Existing offline Analytics fixtures exercise guest feature guards only.
        // Normal launches and every Release build require a signed-in account.
        if ProcessInfo.processInfo.environment["PLONN_ANALYTICS_MARKETS_FIXTURE"] != nil {return true}
        #endif
        return account.hasSignedInAccount
    }
    private enum Sheet: Identifiable {
        case signIn(AccountSignInRequest), recovery, interests(UUID)
        var id: String { switch self {case .signIn(let request): return request.id.uuidString; case .recovery: return "recovery"; case .interests(let owner): return "interests-" + owner.uuidString} }
    }
    private var sheet: Binding<Sheet?> {
        Binding(get: {
            if account.passwordRecovery { return .recovery }
            if let request = access.request { return .signIn(request) }
            return onboardingOwner.map { .interests($0) }
        }, set: { value in
            if value == nil {
                if onboardingOwner != nil { account.deferInterestsOnboarding(); onboardingOwner = nil }
                account.passwordRecovery = false; access.cancel()
            }
        })
    }
    var body: some View {
        Group {
            if account.restoringSession {
                VStack(spacing:20) {Brand();PLONNLoader("Restoring your session…")}
                    .frame(maxWidth:.infinity,maxHeight:.infinity).background(navy)
            } else if canEnterApp {
                LaunchExperience()
            } else {
                welcome
            }
        }
        .sheet(item: sheet) { item in
            switch item {
            case .recovery: PasswordRecoveryView()
            case .interests(let owner):
                NavigationStack { InterestsView(onboarding: true).id(owner) }
            case .signIn(let request):
                NavigationStack {
                    AccountView(initiallyCreating:request.creating)
                        .safeAreaInset(edge:.top) {
                            if let feature=request.feature {
                                Text("Sign in to continue with \(feature.title).")
                                    .font(.subheadline).frame(maxWidth:.infinity,alignment:.leading)
                                    .padding().background(navySurface)
                            }
                        }
                        .toolbar {
                            ToolbarItem(placement:.cancellationAction) {
                                Button("Back") {access.cancel()}.accessibilityIdentifier("account-access-cancel")
                            }
                        }
                }
            }
        }
        .task(id: mayOfferInterests) {
            guard mayOfferInterests else { return }
            // Let the sign-in/recovery sheet finish dismissing before optional onboarding.
            try? await Task.sleep(for: .milliseconds(450))
            guard !Task.isCancelled, mayOfferInterests else { return }
            onboardingOwner = account.user?.id
        }
        .onChange(of: account.user?.id) { _, owner in
            if owner != onboardingOwner { onboardingOwner = nil }
        }
        .onChange(of:account.hasSignedInAccount) {_,signedIn in
            if signedIn {completeSignIn()}
        }
        .onChange(of:account.passwordRecovery) {_,recovering in if !recovering {completeSignIn()}}
    }
    private var mayOfferInterests: Bool {
        account.shouldOfferInterests && !account.restoringSession && !account.passwordRecovery && access.request == nil
    }
    private func completeSignIn() {
        guard !account.passwordRecovery else {return}
        access.complete(ownerID:account.user?.id,isAnonymous:account.user?.isAnonymous)
    }
    private var welcome: some View {
        ScrollView {
            VStack(alignment:.leading,spacing:24) {
                Brand().padding(.top,40)
                Text("Welcome to PLONN").font(.largeTitle.bold()).accessibilityIdentifier("preview-welcome-heading")
                VStack(spacing:12) {
                    Button {access.present(creating:true)} label: {
                        Text("Create account").frame(maxWidth:.infinity).padding(.vertical,10)
                    }.plonnGlassButton(prominent:true).accessibilityIdentifier("preview-create-account")
                    Button {access.present()} label: {
                        Text("Sign in").frame(maxWidth:.infinity).padding(.vertical,10)
                    }.plonnGlassButton().accessibilityIdentifier("preview-sign-in")
                }.plonnGlassGroup()
                Link("Privacy policy",destination:PLONNServicePolicy.privacyURL).font(.footnote)
            }.padding(24).frame(maxWidth:560)
        }.frame(maxWidth:.infinity,maxHeight:.infinity).background(LeagueBackdrop(league:"All"))
    }
}

struct AccountFeaturePrompt: View {
    @EnvironmentObject private var account: AccountStore
    let feature: AccountFeature
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            Label("Sign in for \(feature.title)",systemImage:"lock")
                .font(.headline).accessibilityIdentifier("account-required-"+feature.rawValue)
            Text(feature.detail).font(.subheadline).foregroundStyle(.secondary)
            Button("Sign in") {account.access.present(feature:feature)}
                .plonnGlassButton(prominent:true).frame(minHeight:44)
                .accessibilityIdentifier("sign-in-for-"+feature.rawValue)
        }.frame(maxWidth:.infinity,alignment:.leading).padding(20)
            .background(navySurface,in:RoundedRectangle(cornerRadius:18))
    }
}

/// Conditional construction prevents protected screens from starting tasks in preview.
struct AccountRequiredView<Content: View>: View {
    @EnvironmentObject private var account: AccountStore
    let feature: AccountFeature
    let content: () -> Content
    init(feature: AccountFeature, @ViewBuilder content: @escaping () -> Content) {
        self.feature=feature;self.content=content
    }
    var body: some View {
        if account.hasSignedInAccount {content()}
        else {ScrollView {AccountFeaturePrompt(feature:feature).padding()}.background(navy)}
    }
}
