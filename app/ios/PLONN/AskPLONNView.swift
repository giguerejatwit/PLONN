import SwiftUI

/// Research uses explicit, read-only requests. Opening the screen never refreshes
/// prices, starts training, or changes the existing selection draft.
struct AskPLONNView: View {
    @EnvironmentObject private var store: Store
    @EnvironmentObject private var account: AccountStore
    @State private var question = ""
    @FocusState private var questionFocused: Bool
    @State private var intent: AskPLONNIntent = .generalChat
    @State private var selectedContext = ""
    @State private var initialMarket: Market?
    @State private var selectedRecipe: LabRecipe?
    @State private var modelResult: AskPLONNModelResultSnapshot?
    @State private var draft: [Market] = []
    @State private var fixtureMarkets: [Market] = []
    @State private var recipes: [LabRecipe] = []
    @State private var contextExpanded = false
    @State private var researchingPlayer = true
    @State private var playerName = ""
    @FocusState private var playerNameFocused: Bool
    @State private var researchSport = "NBA"
    @State private var researchStat = "player_points"
    @State private var messages: [Message] = []
    @State private var working = false
    @State private var task: Task<Void, Never>?
    @State private var generation = UUID()
    @State private var didLoad = false
    @State private var sheet: ResearchSheet?
    @State private var conversation = AskPLONNConversation()
    @State private var automaticContext: AskPLONNContextTarget?
    @State private var automaticChoice: AskPLONNContextTarget?
    @State private var declinedChatOwner: UUID?
    @State private var consentRequestedOwner: UUID?
    private let initialProvider: (any AskPLONNProvider)?
    private let localPrototype: Bool

    init(initialMarket: Market? = nil, initialRecipe: LabRecipe? = nil,
         initialModelResult: AskPLONNModelResultSnapshot? = nil,
         provider: (any AskPLONNProvider)? = nil) {
        _initialMarket = State(initialValue: initialMarket)
        _selectedRecipe = State(initialValue: initialRecipe)
        _modelResult = State(initialValue: initialModelResult)
        _selectedContext = State(initialValue: initialRecipe.map { "recipe:" + $0.id.uuidString } ?? initialMarket.map { "market:" + $0.id } ?? "")
        _intent = State(initialValue: initialRecipe != nil || initialModelResult != nil ? .explainModel : (initialMarket != nil ? .recentForm : .generalChat))
        _researchingPlayer = State(initialValue: initialMarket == nil && initialRecipe == nil)
        #if DEBUG && targetEnvironment(simulator)
        let useLocal = provider == nil && ProcessInfo.processInfo.environment["PLONN_ASK_LOCAL_MODEL_DISABLED"] != "1"
        self.initialProvider = provider ?? (useLocal ? LocalAskPLONNProvider() : nil)
        self.localPrototype = useLocal
        #else
        self.initialProvider = provider
        self.localPrototype = false
        #endif
    }
    private struct Message: Identifiable {
        let id = UUID()
        let question: String
        var answer: AskPLONNAnswer?
        var failure: String?
        var guidance: String?
        var contextLabel: String = ""
        var priceComparison: AskPLONNPriceComparison?
        var contextChoices: [AskPLONNContextTarget] = []
        var researchLimitations: [AskPLONNEvidence] = []
    }
    private enum ResearchSheet: String, Identifiable {
        case calculator, models, context, chatConsent, account
        var id: String { rawValue }
    }
    private var hostedChatEligible: Bool {
        initialProvider == nil && CloudConfiguration.servicePolicy.chatAvailable && account.user?.isAnonymous == false
    }
    private var provider: (any AskPLONNProvider)? {
        if let initialProvider { return initialProvider }
        return account.hostedChatProvider()
    }
    private var markets: [Market] {
        var seen = Set<String>()
        return comparisonMarkets
            .filter { seen.insert($0.id).inserted }
    }
    private var comparisonMarkets: [Market] {
        (initialMarket.map { [$0] } ?? []) + fixtureMarkets + store.saved + store.live + store.legs + draft
    }
    private var availableRecipes: [LabRecipe] {
        var seen = Set<UUID>()
        return ((selectedRecipe.map { [$0] } ?? []) + recipes).filter { seen.insert($0.id).inserted }
    }
    private var market: Market? { markets.first { selectedContext == "market:" + $0.id } }
    private var recipe: LabRecipe? { availableRecipes.first { selectedContext == "recipe:" + $0.id.uuidString } }
    private var canSend: Bool {
        !working && !effectiveQuestion.isEmpty && question.count <= AskPLONNCore.maximumQuestionLength
    }
    private var effectiveQuestion: String {
        let typed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        if typed.isEmpty, publicPlayerContext != nil { return suggestedQuestion(.recentForm) }
        return typed.isEmpty && !researchingPlayer && (market != nil || recipe != nil) ? suggestedQuestion(intent) : typed
    }
    private var publicPlayerContext: Market? {
        let name = playerName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard researchingPlayer, !name.isEmpty, name.count <= 100, researchStats.contains(researchStat) else { return nil }
        return Market(id: "public-research:" + researchSport + ":" + researchStat + ":" + name.lowercased(), event: "", sport: researchSport, kind: "prop",
            name: name, opponent: "", label: OpponentPropHistoryEngine.statLabel(researchStat), aliases: "",
            p: nil, price: 0, market: researchStat, book: "No sportsbook quote", quotes: [], updated: nil,
            mode: "research", logs: nil, line: nil, eventLabel: nil, modelValidated: nil, modelUpdated: nil,
            modelVersion: nil, eventState: nil, commence: nil, side: nil)
    }
    private var activeContextID: String {
        researchingPlayer ? (publicPlayerContext?.id ?? "") : selectedContext
    }
    private var activeContextLabel: String {
        if researchingPlayer, let player = publicPlayerContext { return player.name + " · " + player.label }
        return recipe?.name ?? market.map { $0.name + " · " + $0.label } ?? "Attach research (optional)"
    }
    var body: some View {
        ScrollViewReader { proxy in
            VStack(spacing: 0) {
                conversationBar
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 28) {
                        if messages.isEmpty { introduction }
                        ForEach(messages) { message in messageView(message) }
                        Color.clear.frame(height: 1).id("conversation-end")
                    }.padding(.horizontal, 20).padding(.vertical, 24)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: messages.count) { _, _ in withAnimation { proxy.scrollTo("conversation-end", anchor: .bottom) } }
                .onChange(of: working) { _, pending in if !pending { withAnimation { proxy.scrollTo("conversation-end", anchor: .bottom) } } }
                .accessibilityIdentifier("ask-plonn-screen")
            }
            .background(Color(uiColor: .systemBackground))
        }
            .safeAreaInset(edge: .bottom, spacing: 0) { composer }
            .fontDesign(.default)
            .navigationTitle("LLM").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button(action: newChat) { Image(systemName: "square.and.pencil") }
                        .accessibilityLabel("New chat").accessibilityIdentifier("ask-plonn-new-chat")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {ProPaywallView()} label:{Text("Pro")}.accessibilityLabel("Pro plans")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { sheet = .calculator } label: { Image(systemName: "plus.forwardslash.minus") }
                        .accessibilityLabel("Selection calculator")
                }
            }
            .sheet(item: $sheet) { item in
                if item == .calculator { BuilderView().environmentObject(store) }
                else if item == .chatConsent { chatConsentView }
                else if item == .account {
                    NavigationStack {
                        AccountView().environmentObject(account)
                            .toolbar {
                                ToolbarItem(placement: .confirmationAction) {
                                    Button("Done") { sheet = nil }
                                        .accessibilityIdentifier("ask-plonn-account-done")
                                }
                            }
                    }
                }
                else if item == .context {
                    NavigationStack {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 20) {
                                if researchingPlayer { playerResearch }
                                else {
                                    Text(activeContextLabel).font(.system(.headline, design: .default))
                                    Button("Research a player") {
                                        selectedContext = ""; intent = .recentForm; researchingPlayer = true
                                    }.accessibilityIdentifier("ask-plonn-new-player")
                                }
                                contextPicker
                                if !activeContextID.isEmpty {
                                    Button("Remove research attachment") {
                                        cancelPending(); selectedContext = ""; playerName = ""
                                        researchingPlayer = true; intent = .generalChat
                                        playerNameFocused = false; sheet = nil
                                    }.accessibilityIdentifier("ask-plonn-remove-context")
                                }
                            }.padding(20)
                        }
                        .navigationTitle("Chat context").navigationBarTitleDisplayMode(.inline)
                        .toolbar {
                            ToolbarItem(placement: .confirmationAction) {
                                Button("Done") { playerNameFocused = false; sheet = nil }
                                    .accessibilityIdentifier("ask-plonn-context-done")
                            }
                        }
                    }.presentationDetents([.large])
                }
                else {
                    NavigationStack { MyModelsView(onLoad: { version in
                        guard version.user_id == account.user?.id else { return }
                        cancelPending()
                        selectedRecipe = version.configuration; modelResult = nil
                        selectedContext = "recipe:" + version.configuration.id.uuidString
                        researchingPlayer = false; intent = .explainModel; sheet = nil
                    }).environmentObject(account) }
                }
            }
        .task {
            guard !didLoad else { return }; didLoad = true
            draft = (try? BuilderDraftStorage(defaults: .standard).load()) ?? []
            #if DEBUG
            if let fixture = ProcessInfo.processInfo.environment["PLONN_ASK_MARKETS_FIXTURE"]?.data(using: .utf8) {
                // Temporary read-only UI fixtures never enter saved preferences.
                fixtureMarkets = (try? JSONDecoder().decode([Market].self, from: fixture)) ?? []
            }
            let recipeJSON = ProcessInfo.processInfo.environment["PLONN_ASK_RECIPES_FIXTURE"]
                ?? UserDefaults.standard.string(forKey: "labFeatureRecipes")
            #else
            let recipeJSON = UserDefaults.standard.string(forKey: "labFeatureRecipes")
            #endif
            if let data = recipeJSON?.data(using: .utf8) {
                recipes = (try? JSONDecoder().decode([LabRecipe].self, from: data)) ?? []
            }
        }
        .onChange(of: selectedContext) { _, value in
            cancelPending()
            if !value.isEmpty { researchingPlayer = false; intent = value.hasPrefix("recipe:") ? .explainModel : .recentForm }
        }
        .onChange(of: activeContextID) { _, _ in
            cancelPending(); conversation = AskPLONNConversation(); automaticContext = nil; automaticChoice = nil
        }
        .onChange(of: account.user?.id) { _, owner in
            declinedChatOwner = nil; consentRequestedOwner = nil
            if sheet == .chatConsent || (sheet == .account && account.user?.isAnonymous == false) { sheet = nil }
            cancelPending(); messages = []; question = ""; selectedContext = ""
            initialMarket = nil; selectedRecipe = nil; modelResult = nil; recipes = []; draft = []; fixtureMarkets = []
            playerName = ""; playerNameFocused = false; questionFocused = false; researchSport = "NBA"; researchStat = "player_points"; researchingPlayer = true
            conversation = AskPLONNConversation(); automaticContext = nil; automaticChoice = nil
        }
        .onChange(of: account.user?.isAnonymous) { _, anonymous in
            if sheet == .account, anonymous == false { sheet = nil }
        }
        .onDisappear { cancelPending(reason: "Research stopped when you left LLM. Ask again to continue.") }
    }
    private var introduction: some View {
        VStack(alignment: .leading, spacing: 18) {
            Image(systemName: "bubble.left.and.bubble.right").font(.system(size: 30)).padding(.top, 24)
            Text("What would you like to explore?").font(.system(.title2, design: .default).bold()).accessibilityIdentifier("ask-plonn-welcome")
            Text("Ask about a game, team, or player in NFL, NBA, NHL, or MLB. I’ll look for matching verified research and ask when the matchup is unclear. Player and saved model attachments are also available.")
                .font(.system(.body, design: .default)).foregroundStyle(.secondary)
            Button("How do odds work?") { question = "How do odds work?"; questionFocused = true }
                .buttonStyle(.bordered).font(.system(.subheadline, design: .default))
                .accessibilityIdentifier("ask-plonn-start-chat")
            HStack(spacing: 10) {
                Button("Attach player") { researchingPlayer = true; selectedContext = ""; sheet = .context }
                    .buttonStyle(.bordered).accessibilityIdentifier("ask-plonn-start-player")
                Button("Attach model") { researchingPlayer = false; contextExpanded = true; sheet = .context }
                    .buttonStyle(.bordered).accessibilityIdentifier("ask-plonn-start-model")
            }.font(.system(.subheadline, design: .default))
        }
    }
    private var conversationBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button { questionFocused = false; sheet = .context } label: {
                HStack(spacing: 8) {
                    Image(systemName: activeContextID.isEmpty ? "plus.circle" : "person.crop.circle")
                    Text(activeContextLabel).lineLimit(1)
                    Spacer()
                    Image(systemName: "chevron.down").font(.system(.caption, design: .default))
                }.font(.system(.subheadline, design: .default)).foregroundStyle(.primary)
            }.accessibilityIdentifier("ask-plonn-chat-context")
            Text(providerStatus)
                .font(.system(.caption2, design: .default)).foregroundStyle(.secondary)
                .accessibilityIdentifier("ask-plonn-provider-status")
            if hostedChatEligible, !account.hasChatConsent {
                Button("Enable AI chat") { requestChatConsent() }.font(.caption)
            } else if initialProvider == nil, CloudConfiguration.servicePolicy.chatAvailable, account.user?.isAnonymous != false {
                Button("Sign in for Gemma4") { questionFocused = false; playerNameFocused = false; sheet = .account }
                    .font(.caption).accessibilityIdentifier("ask-plonn-sign-in")
            }
        }.padding(.horizontal, 20).padding(.vertical, 10)
    }
    private var chatConsentView: some View {
        GemmaConsentView(ownerID: consentRequestedOwner, onAllowed: {
            guard hostedChatEligible, account.user?.id == consentRequestedOwner else { sheet = nil; return }
            declinedChatOwner = nil; sheet = nil
            questionFocused = true
        }, onDeclined: {
            guard let owner = account.user?.id, owner == consentRequestedOwner else { sheet = nil; return }
            declinedChatOwner = owner; sheet = nil
            questionFocused = true
        }).environmentObject(account)
    }
    private func requestChatConsent() {
        guard hostedChatEligible else { return }
        consentRequestedOwner = account.user?.id; questionFocused = false; sheet = .chatConsent
    }
    private var summaryLabel: String {
        #if DEBUG
        return "Verified local summary"
        #else
        return "Research summary"
        #endif
    }
    private var providerStatus: String {
        #if DEBUG
        if localPrototype { return "Local prototype · " + LocalAskPLONNProvider.modelLabel }
        #endif
        if let initialProvider { return initialProvider.displayName + " · Connection checked when you send" }
        return HostedAskPLONNConnectionState.resolve(serviceAvailable: CloudConfiguration.servicePolicy.chatAvailable,
                                                    signedIn: account.user?.isAnonymous == false, consented: account.hasChatConsent).statusText
    }
    private var contextPicker: some View {
        Panel {
            VStack(alignment: .leading, spacing: 10) {
                DisclosureGroup(isExpanded: $contextExpanded) {
                Picker("Research context", selection: $selectedContext) {
                    Text("No saved attachment").tag("")
                    ForEach(markets) { item in Text(item.name + " · " + item.label).tag("market:" + item.id) }
                    ForEach(availableRecipes) { item in Text(item.name + " · Model setup").tag("recipe:" + item.id.uuidString) }
                }.accessibilityIdentifier("ask-plonn-context")
                if let market {
                    Text(market.sport + " · " + (market.mode == "research" ? market.label : (market.eventLabel ?? "vs " + market.opponent)))
                        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    Text(market.mode == "research" ? "Public player history · No sportsbook quote" :
                         (market.mode == "live" ? "Recorded quote · " + BuilderLogic.quoteAge(market).label : "Illustrative market · Prices are sample data"))
                        .font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                }
                if recipe != nil {
                    Text("Saved configuration · Training results are shown only when a verified result is supplied.")
                        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                }
                Button("Choose from My Models") { sheet = .models }.font(.system(.caption, design: .monospaced))
                } label: {
                    Text(!researchingPlayer ? (recipe?.name ?? market?.name ?? "Saved players & models") : "Saved players & models")
                        .font(.system(.subheadline, design: .monospaced))
                        .accessibilityIdentifier("ask-plonn-saved-contexts")
                }
            }
        }
    }
    private var researchStats: [String] {
        (OpponentPropHistoryEngine.supportedMarkets[researchSport] ?? []).sorted {
            OpponentPropHistoryEngine.statLabel($0) < OpponentPropHistoryEngine.statLabel($1)
        }
    }
    private var playerResearch: some View {
        Panel {
            VStack(alignment: .leading, spacing: 12) {
                Text("PLAYER").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                TextField("Player name", text: $playerName).textFieldStyle(.roundedBorder)
                    .focused($playerNameFocused).submitLabel(.done).onSubmit { playerNameFocused = false }
                    .font(.system(.subheadline, design: .monospaced)).accessibilityIdentifier("ask-plonn-player-name")
                Picker("Sport", selection: $researchSport) {
                    ForEach(["NBA", "NFL", "NHL", "MLB"], id: \.self) { sport in Text(sport).tag(sport) }
                }.pickerStyle(.segmented).accessibilityIdentifier("ask-plonn-player-sport")
                Picker("Statistic", selection: $researchStat) {
                    ForEach(researchStats, id: \.self) { stat in Text(OpponentPropHistoryEngine.statLabel(stat).capitalized).tag(stat) }
                }.accessibilityIdentifier("ask-plonn-player-stat")
                Text("Choose a player, then ask below. Leave the question blank to check recent form.")
                    .font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
            }
        }
        .onChange(of: researchSport) { _, _ in
            cancelPending()
            researchStat = researchStats.contains("player_points") ? "player_points" : (researchStats.first ?? "")
        }
        .onChange(of: researchStat) { _, _ in cancelPending() }
        .onChange(of: playerName) { _, _ in cancelPending() }
    }
    private var suggestions: some View {
        let topics: [AskPLONNIntent] = activeContextID.isEmpty ? [.generalChat] :
            (!researchingPlayer && recipe != nil ? [.explainModel] : [.recentForm, .comparePrices, .opponentHistory])
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                if !messages.isEmpty {
                    Button("Tell me more") { question = "What does that mean?"; questionFocused = true }
                        .buttonStyle(.bordered).accessibilityIdentifier("ask-plonn-follow-up")
                }
                ForEach(topics, id: \.rawValue) { topic in
                    Button {
                        intent = topic; question = suggestedQuestion(topic)
                    } label: {
                        Text(title(topic)).font(.system(.caption, design: .default)).padding(.vertical, 2)
                    }.buttonStyle(.bordered).accessibilityIdentifier(topic == .comparePrices ? "Compare prices" : title(topic))
                }
            }
        }.accessibilityIdentifier("ask-plonn-suggestions")
    }
    private var composer: some View {
            VStack(alignment: .leading, spacing: 8) {
                if (!activeContextID.isEmpty || !messages.isEmpty) && !working { suggestions }
                HStack(alignment: .bottom, spacing: 10) {
                TextField(messages.isEmpty ? "Message LLM…" : "Ask a follow-up…", text: $question, axis: .vertical)
                    .focused($questionFocused)
                    .lineLimit(1...4).font(.system(.body, design: .default))
                    .padding(12).background(navySurface,in:RoundedRectangle(cornerRadius:12))
                    .submitLabel(.send).onSubmit { submit() }.disabled(working)
                    .accessibilityIdentifier("ask-plonn-question")
                if working {
                    Button("Stop") { cancelPending(reason: "Research stopped. Send a new question to continue.") }
                        .font(.system(.subheadline, design: .default).bold()).padding(12)
                        .accessibilityIdentifier("ask-plonn-stop")
                } else {
                Button(action: submit) {
                    Image(systemName: "arrow.up").font(.system(.headline, design: .default)).frame(width:44,height:44)
                }.plonnGlassButton(prominent: true).disabled(!canSend).accessibilityIdentifier("ask-plonn-send")
                    .accessibilityLabel("Send question")
                }
                }
                if question.count > AskPLONNCore.maximumQuestionLength { Text("Keep your question under 1,000 characters.").font(.system(.caption, design: .default)).foregroundStyle(.orange) }
                Text("General discussion · Attached research uses recorded sources")
                    .font(.system(.caption2, design: .default)).foregroundStyle(.secondary)
            }
            .padding(.horizontal,20).padding(.vertical,12).background(.regularMaterial)
    }
    private func messageView(_ message: Message) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Spacer(minLength: 44)
                Text(message.question).font(.system(.body, design: .default))
                    .padding(14).background(Color.accentColor.opacity(0.14), in: RoundedRectangle(cornerRadius: 20))
                    .accessibilityIdentifier("ask-plonn-user-message")
            }
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "sparkle").font(.system(.subheadline, design: .default)).padding(.top, 3).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 12) {
            if let comparison = message.priceComparison {
                SportsbookPriceCard(comparison: comparison)
            }
            if let answer = message.answer {
                        if !message.contextLabel.isEmpty { Text(message.contextLabel).font(.system(.caption, design: .default)).foregroundStyle(.secondary) }
                        if message.priceComparison?.bestQuote != nil {
                            DisclosureGroup("Recorded comparison summary") {
                                if let captured = message.priceComparison?.comparedAt {
                                    Text("Recorded as of " + captured.formatted(date: .abbreviated, time: .standard) + ". This summary retains that snapshot.")
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                                answerText(answer).font(.system(.subheadline, design: .default))
                                    .textSelection(.enabled).accessibilityIdentifier("ask-plonn-answer")
                            }.font(.system(.caption, design: .default))
                        } else {
                        answerText(answer).font(.system(.body, design: .default))
                            .textSelection(.enabled).accessibilityIdentifier("ask-plonn-answer")
                        }
                        Text(answer.mode == .generalExplanation ? (answer.providerLabel ?? "General chat · AI unavailable") :
                            (answer.mode == .provider ? (answer.providerLabel ?? "Evidence-backed explanation") : summaryLabel))
                            .accessibilityIdentifier("ask-plonn-answer-provider")
                            .font(.system(.caption2, design: .default)).foregroundStyle(.secondary)
                        ForEach(message.researchLimitations + answer.evidence.filter { $0.availability != .available && message.priceComparison == nil && !message.researchLimitations.contains($0) }) { evidence in
                            Text(evidence.text).font(.system(.subheadline, design: .default)).foregroundStyle(.secondary)
                        }
                        if answer.requiresResearchContext {
                            Button(activeContextID.isEmpty ? "Attach research (optional)" : "Change research attachment") {
                                question = message.question; sheet = .context
                            }.buttonStyle(.bordered).accessibilityIdentifier("ask-plonn-choose-context")
                        }
            } else if let guidance = message.guidance {
                Text(guidance).font(.system(.body, design: .default)).accessibilityIdentifier("ask-plonn-answer")
                ForEach(message.contextChoices) { choice in
                    Button(choice.label) {
                        automaticChoice = choice; question = message.question; submit()
                    }.buttonStyle(.bordered).disabled(working)
                        .accessibilityIdentifier("ask-plonn-context-choice")
                }
            } else if let failure = message.failure {
                Text(failure).font(.system(.body, design: .default)).foregroundStyle(.secondary).accessibilityIdentifier("ask-plonn-answer")
                Button("Ask again") { question = message.question; questionFocused = true }
                    .font(.system(.subheadline, design: .default)).disabled(working).accessibilityIdentifier("ask-plonn-retry")
            } else {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Looking into that…").font(.system(.subheadline, design: .default)).foregroundStyle(.secondary)
                        .accessibilityIdentifier("ask-plonn-typing")
                }
            }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
    private func displayText(_ answer: AskPLONNAnswer) -> String {
        // Presentation only: the stored/provider answer keeps its exact verified
        // fact contract. Attribution remains visible below the message.
        answer.text.replacingOccurrences(of: "Verified data summary\n\n", with: "")
    }
    private func answerText(_ answer: AskPLONNAnswer) -> Text {
        let value = displayText(answer)
        if answer.mode == .generalExplanation,
           let formatted = try? AttributedString(markdown: value, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            return Text(formatted)
        }
        return Text(value)
    }
    private func newChat() {
        cancelPending(); messages = []; question = ""; conversation = AskPLONNConversation(); automaticContext = nil; automaticChoice = nil
        intent = activeContextID.isEmpty ? .generalChat : (researchingPlayer || recipe == nil ? .recentForm : .explainModel)
        questionFocused = false
    }
    private func title(_ intent: AskPLONNIntent) -> String {
        switch intent {
        case .generalChat: return "Explain odds"
        case .recentForm: return "Recent form"
        case .opponentHistory: return "Against opponent"
        case .comparePrices: return "Best price & edge"
        case .explainModel: return "Explain model"
        }
    }
    private func suggestedQuestion(_ intent: AskPLONNIntent) -> String {
        switch intent {
        case .generalChat: return "How do odds work?"
        case .recentForm: return "What do the recent games show?"
        case .opponentHistory: return "How has this player performed against this opponent?"
        case .comparePrices: return "What is the best recorded sportsbook price and model edge for this selection?"
        case .explainModel: return "What does this model setup or verified result tell me?"
        }
    }
    private func cancelPending(reason: String = "Research stopped because the context changed. Send a new question to continue.") {
        generation = UUID(); task?.cancel(); task = nil; working = false
        for index in messages.indices where messages[index].answer == nil && messages[index].failure == nil && messages[index].guidance == nil {
            messages[index].failure = reason
        }
    }
    @MainActor private func submit() {
        guard canSend else { return }
        if hostedChatEligible, !account.hasChatConsent, declinedChatOwner != account.user?.id {
            requestChatConsent(); return
        }
        let text = effectiveQuestion
        let chosenAutomatic = automaticChoice; automaticChoice = nil
        let discoverContext = activeContextID.isEmpty && (chosenAutomatic != nil || AskPLONNContextResolver.shouldDiscover(text, previous: automaticContext, markets: markets))
        let previousAutomatic = automaticContext
        let topic = AskPLONNCore.chatIntentForQuestion(text, fallback: intent, hasResearchContext: !activeContextID.isEmpty)
        let chosenMarket = researchingPlayer ? publicPlayerContext : market
        let chosenRecipe = researchingPlayer ? nil : recipe, chosenResult = modelResult
        let owner = account.user?.id.uuidString, token = UUID()
        let contextID = topic == .generalChat ? "general-chat" : activeContextID
        intent = topic
        generation = token; questionFocused = false; playerNameFocused = false; question = ""
        let contextLabel = topic == .generalChat ? "General chat" :
            (chosenMarket.map { $0.name + " · " + $0.sport + " · " + $0.label } ?? chosenRecipe?.name ?? "")
        let greeting = text.lowercased().trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
        let isGreeting = ["hi", "hello", "hey", "hi plonn", "hello plonn", "help", "what can you do"].contains(greeting)
        let guidance: String? = isGreeting ? "Hi! We can talk sports, odds, and analytics. Just ask a question. Player and model attachments are optional when you want specific research." : nil
        let message = Message(question: text, guidance: guidance, contextLabel: contextLabel); messages.append(message)
        if messages.count > 20 { messages.removeFirst(messages.count - 20) }
        guard guidance == nil else { return }
        working = true
        let currentConversation = conversation
        let reportClient = discoverContext ? account.sharedMatchupReportClient() : nil
        let narrativeClient = discoverContext ? account.matchupNarrativeClient() : nil
        let availableMarkets = markets
        task = Task { @MainActor in
            #if DEBUG && targetEnvironment(simulator)
            if let milliseconds = ProcessInfo.processInfo.environment["PLONN_ASK_RESEARCH_DELAY_MS"].flatMap(Int.init),
               (1...5000).contains(milliseconds) {
                do { try await Task.sleep(for: .milliseconds(milliseconds)) }
                catch { return }
            }
            #endif
            guard !Task.isCancelled, generation == token, account.user?.id.uuidString == owner else { return }
            var evidence: [AskPLONNEvidence] = []
            var resolvedTopic = topic
            var resolvedContextID = contextID
            if discoverContext {
                do {
                    let readReport: AskPLONNContextService.ReportReader? = reportClient.map { client in
                        { @Sendable event in try await client.fetch(event: event) }
                    }
                    let readNarrative: AskPLONNContextService.NarrativeReader? = narrativeClient.map { client in
                        { @Sendable event in try await client.fetch(event: event) }
                    }
                    let research = try await AskPLONNContextService().discover(question: text, markets: availableMarkets,
                        previous: previousAutomatic, chosen: chosenAutomatic, report: readReport, narrative: readNarrative)
                    guard !Task.isCancelled, generation == token, account.user?.id.uuidString == owner else { return }
                    if let guidance = research.guidance {
                        automaticContext = nil
                        if let index = messages.firstIndex(where: { $0.id == message.id }) {
                            messages[index].guidance = guidance; messages[index].contextChoices = research.choices
                            messages[index].contextLabel = "Find research"
                        }
                        working = false; task = nil; return
                    }
                    automaticContext = research.target
                    evidence = research.evidence; resolvedTopic = .recentForm
                    resolvedContextID = research.target?.id ?? "automatic-research"
                    if let index = messages.firstIndex(where: { $0.id == message.id }) {
                        messages[index].contextLabel = research.target?.label ?? "Verified research"
                        // Provider selection may omit unavailable evidence; limitations remain visible.
                        messages[index].researchLimitations = evidence.filter { $0.availability != .available }
                    }
                } catch {
                    guard !Task.isCancelled, generation == token, account.user?.id.uuidString == owner else { return }
                    if let index = messages.firstIndex(where: { $0.id == message.id }) { messages[index].failure = "I couldn’t verify matching research. Try again with the league, teams or player, and date." }
                    working = false; task = nil; return
                }
            } else { switch topic {
            case .generalChat: break
            case .recentForm, .opponentHistory:
                if let market = chosenMarket, topic == .opponentHistory, market.mode == "research" {
                    evidence = [AskPLONNEvidence(id: "public-context-no-opponent", kind: topic, title: "Opponent context unavailable", text: "Public player research has no verified matchup. Choose a market with a known opponent for an opponent comparison.", source: "PLONN research context", availability: .unavailable)]
                } else if let market = chosenMarket {
                    let history = await PlayerHistoryService.shared.load(market: market, includeOpponent: topic == .opponentHistory)
                    guard !Task.isCancelled, generation == token, account.user?.id.uuidString == owner else { return }
                    evidence = [topic == .recentForm ? AskPLONNResearch.recentForm(market: market, history: history) : AskPLONNResearch.opponentHistory(market: market, history: history)]
                }
            case .comparePrices:
                if let market = chosenMarket {
                    let capturedAt = Date(), candidates = comparisonMarkets
                    let comparison = AskPLONNResearch.priceComparison(market: market, markets: candidates, now: capturedAt)
                    if let index = messages.firstIndex(where: { $0.id == message.id }) { messages[index].priceComparison = comparison }
                    evidence = [AskPLONNResearch.comparePrices(market: market, markets: candidates, now: capturedAt)]
                }
            case .explainModel:
                if let recipe = chosenRecipe { evidence = [AskPLONNResearch.modelSetup(recipe: recipe)] }
                if let snapshot = chosenResult, let recipe = chosenRecipe, let ownerID = account.user?.id {
                    evidence.append(AskPLONNResearch.modelResult(snapshot: snapshot, ownerID: ownerID, recipe: recipe))
                }
            }
            }
            if evidence.isEmpty && resolvedTopic != .generalChat {
                evidence = [AskPLONNEvidence(id: "missing-context", kind: topic, title: "Research context needed", text: "Choose a matching player or model before asking about this topic.", source: "PLONN research context", availability: .unavailable)]
            }
            do {
                let answer = try await currentConversation.ask(question: text, intent: resolvedTopic, evidence: evidence, ownerID: owner, contextID: resolvedContextID, provider: provider)
                guard !Task.isCancelled, generation == token, account.user?.id.uuidString == owner else { return }
                if let index = messages.firstIndex(where: { $0.id == message.id }) { messages[index].answer = answer }
            } catch {
                guard !Task.isCancelled, generation == token, account.user?.id.uuidString == owner else { return }
                if let index = messages.firstIndex(where: { $0.id == message.id }) {
                    messages[index].failure = "The explanation is unavailable. Try your question again."
                }
            }
            if generation == token { working = false; task = nil }
        }
    }
}

/// A recorded comparison remains useful without LLM inference. Its expiry is
/// checked on screen; opening this card never refreshes odds or trains a model.
struct SportsbookPriceCard: View {
    let comparison: AskPLONNPriceComparison

    var body: some View {
        TimelineView(.periodic(from: Date(), by: 5)) { context in
            content(now: context.date)
        }
    }

    private func content(now: Date) -> some View {
        let expired = comparison.expiresAt.map { now > $0 } ?? false
        let best = expired ? nil : comparison.bestQuote
        return VStack(alignment: .leading, spacing: 14) {
            Text("Best price & model edge").font(.system(.headline, design: .default)).accessibilityIdentifier("sportsbook-price-card")
            if let best {
                HStack(alignment: .top, spacing: 18) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Best recorded price").font(.caption).foregroundStyle(.secondary)
                        Text(american(best.price)).font(.system(size: 26, weight: .semibold)).monospacedDigit()
                            .accessibilityIdentifier("sportsbook-best-price")
                        Text(comparison.bestBooks.joined(separator: " · ")).font(.subheadline)
                            .accessibilityIdentifier("sportsbook-best-book")
                    }.frame(maxWidth: .infinity, alignment: .leading)
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Model edge").font(.caption).foregroundStyle(.secondary)
                        if let edge = comparison.edgePercentagePoints {
                            Text(String(format: "%+.2f pp", locale: Locale(identifier: "en_US_POSIX"), edge))
                                .font(.system(size: 26, weight: .semibold)).monospacedDigit()
                                .accessibilityIdentifier("sportsbook-model-edge")
                        } else {
                            Text("Unavailable").font(.headline).accessibilityIdentifier("sportsbook-model-edge")
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
                Text("Price implied probability: " + percent(best.impliedProbability))
                    .font(.subheadline).accessibilityIdentifier("sportsbook-implied-probability")
                if let probability = comparison.modelProbability {
                    Text("Validated model probability: " + percent(probability))
                        .font(.subheadline).accessibilityIdentifier("sportsbook-model-probability")
                }
                if comparison.edgePercentagePoints == nil {
                    Text(comparison.edgeUnavailableReason).font(.caption).foregroundStyle(.secondary)
                        .accessibilityIdentifier("sportsbook-edge-unavailable")
                }
                Text(comparison.quotes.count == 1 ? "1 timestamped book · Broader comparison unavailable" :
                     "Best among \(comparison.quotes.count) timestamped books for this exact selection")
                    .font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("sportsbook-coverage")
                Text("Quote recorded " + best.updated.formatted(date: .abbreviated, time: .standard))
                    .font(.caption2).foregroundStyle(.secondary)
                if let version = comparison.modelVersion, let updated = comparison.modelUpdated {
                    Text("Model " + version + " · " + updated.formatted(date: .abbreviated, time: .standard))
                        .font(.caption2).foregroundStyle(.secondary)
                }
                DisclosureGroup("Recorded book prices") {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(comparison.quotes, id: \.book) { quote in
                            VStack(alignment: .leading, spacing: 3) {
                                HStack { Text(quote.book); Spacer(); Text(american(quote.price)).monospacedDigit() }
                                Text(quote.updated.formatted(date: .abbreviated, time: .standard))
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }.padding(.top, 8)
                }.font(.subheadline).accessibilityIdentifier("sportsbook-recorded-quotes")
            } else {
                Text(expired ? "Comparison expired. Prices and model edge need fresh timestamps." :
                     (comparison.unavailableReason ?? "No current timestamped price is available."))
                    .font(.subheadline).foregroundStyle(.secondary).accessibilityIdentifier("sportsbook-price-unavailable")
                Text("Model edge unavailable").font(.subheadline).accessibilityIdentifier("sportsbook-model-edge")
            }
            if !comparison.excludedReasons.isEmpty {
                Text("Some recorded prices could not be used because their freshness or selection match could not be verified.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Edge = model probability − price implied probability, in percentage points (pp). Price includes bookmaker margin. Quotes can change; a positive edge does not guarantee a return.")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(16).background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
    }
    private func american(_ price: Double) -> String {
        String(format: price.rounded() == price && abs(price) <= 1_000_000 ? "%+.0f" : "%+g", locale: Locale(identifier: "en_US_POSIX"), price)
    }
    private func percent(_ value: Double) -> String {
        String(format: "%.2f%%", locale: Locale(identifier: "en_US_POSIX"), value * 100)
    }
}
