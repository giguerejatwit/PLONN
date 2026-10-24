import SwiftUI

/// Browsing Analytics never starts a paid model run. The user chooses a verified
/// upcoming event, player and contract before explicitly building the outlook.
struct PlayerMatchupCard: View {
    let candidate: AnalyticsFeedCandidate
    let quote: Market?
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let event = candidate.verifiedSubject?.event, event.displayLifecycle(now: Date()) == .scheduled {
                Text(event.away + " at " + event.home).font(.subheadline.bold())
            }
            NavigationLink { PlayerMatchupBuilder(candidate: candidate, quote: quote) } label: {
                Label("Research matchup", systemImage: "arrow.up.right").frame(minHeight: 44)
            }.plonnGlassButton().accessibilityIdentifier("player-matchup-open")
        }.padding(18).background(navySurface, in: RoundedRectangle(cornerRadius: 18))
    }
}

struct PlayerMatchupBuilder: View {
    let candidate: AnalyticsFeedCandidate
    let quote: Market?
    @EnvironmentObject private var account: AccountStore
    @EnvironmentObject private var purchases: PurchaseStore
    @EnvironmentObject private var store: Store
    @State private var events: [PlayerMatchupEvent] = []
    @State private var eventID = ""
    @State private var players: [PlayerMatchupPlayer] = []
    @State private var playerID = ""
    @State private var stat = ""
    @State private var lineDraft = PlayerMatchupLineDraft()
    @State private var recordedMarkets: [Market] = []
    @State private var recordedScope = ""
    @State private var openingQuoteScope: String?
    @State private var oddsLoading = false
    @State private var side = "Over"
    @State private var scoring = "draftkings_classic"
    @State private var method = "two_pointer"
    @State private var loading = false
    @State private var rosterLoading = false
    @State private var message = ""
    @State private var rosterMessage = ""
    @State private var showPro = false
    @State private var showResult = false
    @State private var runSetup: PlayerMatchupSetup?
    @State private var runQuote: Market?
    @FocusState private var lineFocused: Bool
    private var event: PlayerMatchupEvent? { events.first { $0.id == eventID } }
    private var player: PlayerMatchupPlayer? { players.first { $0.id == playerID } }
    private var propOptions: [String: String] {
        if LabPropCatalog.isFantasy(stat) { return ["scoring_preset": scoring] }
        if stat == "Method of first basket" { return ["first_basket_method": method] }
        return [:]
    }
    private var line: String { lineDraft.text }
    private var selectionKey: String {
        [eventID, playerID, stat, scoring, method].joined(separator: "|")
    }
    private var accountScope: String {
        let scope = store.scope(for: account)
        return (account.hasSignedInAccount ? scope.owner?.uuidString ?? "guest" : "guest") + "|" + scope.backend
    }
    private var oddsScope: String { accountScope + "|" + eventID }
    private var matchingQuotes: [Market] {
        guard account.hasSignedInAccount, let event, !rosterLoading else { return [] }
        let retained = recordedScope == oddsScope ? recordedMarkets : []
        // Reuse opening-card quotes only when the shared cache has no copy of that ID.
        let opening = openingQuoteScope == accountScope ? (quote.map { [$0] } ?? []) + candidate.quotes : []
        let knownIDs = Set(retained.map(\.id))
        let markets = retained + opening.filter { !knownIDs.contains($0.id) }
        return PlayerMatchupLineMatching.quotes(event: event, playerID: playerID, players: players,
            stat: stat, side: side, markets: Array(Set(markets)))
    }
    private var selectedQuote: Market? {
        guard let value = Double(line) else { return nil }
        return PlayerMatchupLineMatching.preferred(matchingQuotes, book: quote?.book, line: value)
    }
    private var setup: PlayerMatchupSetup? {
        guard let event, let player, let value = Double(line), value.isFinite,
              (event.game.start ?? .distantPast) > Date() else { return nil }
        let request = PlayerMatchupSetup(league: candidate.market.sport, eventID: event.game.eventID,
            home: event.game.home, away: event.game.away, startsAt: event.game.startsAt,
            player: player.name, playerID: player.id, teamID: player.teamID,
            stat: stat, line: value, side: side, book: selectedQuote?.book ?? "Manual line", propOptions: propOptions)
        return (try? request.payload()) != nil ? request : nil
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if loading { PLONNLoader("Finding upcoming games…") }
                if !message.isEmpty { Text(message).font(.subheadline).foregroundStyle(.secondary) }
                if !events.isEmpty { selectors }
                if events.isEmpty && !loading { Button("Refresh games") { Task { await discover() } }.plonnGlassButton() }

            }.padding(20)
        }.scrollDismissesKeyboard(.interactively)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                buildControls.padding(.horizontal, 20).padding(.top, 12).padding(.bottom, 8)
                    .frame(maxWidth: .infinity).background(navy)
                    .overlay(alignment: .top) { Rectangle().fill(Color.white.opacity(0.08)).frame(height: 1) }
            }
            .background(navy).navigationTitle("Next matchup").navigationBarTitleDisplayMode(.inline).toolbar(.hidden, for: .tabBar)
            .toolbar { ToolbarItemGroup(placement: .keyboard) { Spacer(); Button("Done") { lineFocused = false } } }
            .task {
                if events.isEmpty {
                    stat = PlayerMatchupSetup.stat(sport: candidate.market.sport, marketKey: candidate.market.market) ?? LabPropCatalog.options(league: candidate.market.sport, target: "Player").first ?? ""
                    side = quote?.side?.lowercased() == "under" ? "Under" : "Over"
                    await discover()
                }
            }
            .task(id: eventID) { await loadRoster() }
            .task(id: oddsScope) { await loadRecordedLines() }
            .onChange(of: selectionKey, initial: true) { _, _ in resetLine() }
            .onChange(of: side) { _, _ in prefillLine() }
            .onChange(of: matchingQuotes) { _, _ in prefillLine() }
            .navigationDestination(isPresented: $showResult) { if let runSetup { PlayerMatchupView(setup: runSetup, quote: runQuote, autoStart: true) } }
            .sheet(isPresented: $showPro) { NavigationStack { ProPaywallView().toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showPro = false } } } } }
    }
    private var buildControls: some View {
        VStack(alignment: .leading, spacing: 8) {
                if !account.hasSignedInAccount { AccountFeaturePrompt(feature: .privateModels) }
                else if !purchases.mayUseProFeatures {
                    Button("Unlock matchup outlooks with Pro") { showPro = true }.plonnGlassButton(prominent: true).frame(minHeight: 44)
                } else {
                    Button {
                        guard let setup, (try? setup.payload()) != nil else { return }
                        lineFocused = false; runSetup = setup; runQuote = selectedQuote; showResult = true
                    } label: { Label("Build matchup outlook", systemImage: "sparkles").frame(maxWidth: .infinity).frame(minHeight: 44) }
                        .plonnGlassButton(prominent: true).disabled(setup == nil || rosterLoading)
                        .accessibilityIdentifier("player-matchup-build")
                    Text("1 model run per new outlook").font(.caption).foregroundStyle(.secondary)
                }
        }
    }
    private var selectors: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Upcoming game").font(.caption).foregroundStyle(.secondary)
            Menu {
                ForEach(events) { choice in
                    Button { eventID = choice.id } label: { Text(choice.game.title + " · " + (choice.game.start?.formatted(date: .abbreviated, time: .shortened) ?? "")) }
                }
            } label: {
                HStack(alignment: .center, spacing: 10) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(event?.game.title ?? "Choose a game").font(.subheadline.bold()).multilineTextAlignment(.leading)
                        if let start = event?.game.start { Text(start, format: .dateTime.month(.abbreviated).day().hour().minute()).font(.caption).foregroundStyle(.secondary) }
                    }
                    Spacer(minLength: 0); Image(systemName: "chevron.up.chevron.down").font(.caption)
                }.frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            }.plonnGlassButton().accessibilityIdentifier("player-matchup-event")
            if rosterLoading { PLONNLoader("Verifying players…") }
            if !rosterMessage.isEmpty {
                Text(rosterMessage).font(.caption).foregroundStyle(.orange)
                if players.isEmpty && !rosterLoading { Button("Retry roster") { Task { await loadRoster() } }.frame(minHeight: 44) }
            }
            Text("Player").font(.caption).foregroundStyle(.secondary)
            Menu {
                if let event {
                    ForEach([event.game.home, event.game.away], id: \.self) { team in
                        Section(team) {
                            ForEach(players.filter { $0.team == team }) { choice in
                                Button { playerID = choice.id } label: { Text(choice.name + " · " + choice.position) }
                            }
                        }
                    }
                }
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(player?.name ?? "Choose a player").font(.subheadline.bold())
                        if let player { Text(player.position + " · " + player.team).font(.caption).foregroundStyle(.secondary) }
                    }
                    Spacer(minLength: 0); Image(systemName: "chevron.up.chevron.down").font(.caption)
                }.frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            }.plonnGlassButton().disabled(rosterLoading || players.isEmpty).accessibilityIdentifier("player-matchup-player")
            Picker("Player prop", selection: $stat) {
                ForEach(LabPropCatalog.playerGroups(league: candidate.market.sport)) { group in
                    Section(group.title) { ForEach(group.props, id: \.self) { Text(LabPropCatalog.displayName($0)).tag($0) } }
                }
            }.pickerStyle(.menu).accessibilityIdentifier("player-matchup-stat")
            if LabPropCatalog.isFantasy(stat) {
                Picker("Scoring format", selection: $scoring) {
                    Text("DraftKings Classic").tag("draftkings_classic")
                    Text("FanDuel Classic").tag("fanduel_classic")
                }.pickerStyle(.menu)
            }
            if stat == "Method of first basket" {
                Picker("First basket method", selection: $method) {
                    Text("Two-pointer").tag("two_pointer"); Text("Three-pointer").tag("three_pointer"); Text("Free throw").tag("free_throw")
                }.pickerStyle(.menu)
            }
            Picker("Direction", selection: $side) {
                Text(LabPropCatalog.isYesNo(stat) ? "Yes" : "Over").tag("Over")
                Text(LabPropCatalog.isYesNo(stat) ? "No" : "Under").tag("Under")
            }.pickerStyle(.segmented)
            if !LabPropCatalog.isYesNo(stat) {
                HStack {
                    Text("Comparison line").font(.subheadline)
                    TextField("Enter line", text: Binding(get: { lineDraft.text }, set: { lineDraft.edit($0) })).focused($lineFocused).keyboardType(.numbersAndPunctuation).multilineTextAlignment(.trailing).frame(minHeight: 44).accessibilityIdentifier("player-matchup-line")
                }
            }
            if let quote = selectedQuote {
                Text(odds(quote.price) + " · " + quote.book).font(.subheadline.bold()).foregroundStyle(brandAccent)
                if let date = quote.updated.flatMap(sportsDate) { Text("Recorded " + date.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(.secondary) }
            } else {
                Text(oddsLoading ? "Checking recorded lines…" : "No recorded line available. You can enter one.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let note = LabPropCatalog.propNote(stat) { Text(note).font(.caption).foregroundStyle(.secondary) }
            Text("Every listed prop can be checked. An outlook may be limited when workload, opponent, injury or lineup data cannot be verified.").font(.caption).foregroundStyle(.secondary)
        }.padding(18).background(navySurface, in: RoundedRectangle(cornerRadius: 18))
    }
    @MainActor private func resetLine() {
        lineDraft.select(selectionKey, binary: LabPropCatalog.isYesNo(stat))
        prefillLine()
    }
    @MainActor private func prefillLine() {
        lineDraft.fill(PlayerMatchupLineMatching.preferred(matchingQuotes, book: quote?.book)?.line, for: selectionKey)
    }
    @MainActor private func loadRecordedLines() async {
        let expectedScope = oddsScope
        if openingQuoteScope == nil { openingQuoteScope = accountScope }
        recordedMarkets = []; recordedScope = expectedScope; oddsLoading = false
        guard account.hasSignedInAccount, let expected = event, let start = expected.game.start else { return }
        recordedMarkets = store.analyticsSnapshots(for: account)
        prefillLine()
        let phase = NBARecordedOddsRouting.phase(sport: expected.game.league, seasonTypes: [expected.seasonType])
        guard phase != .unknown else { return }
        oddsLoading = true
        defer { if oddsScope == expectedScope { oddsLoading = false } }
        do {
            // Reads only the existing shared snapshot. No live provider refresh,
            // scoped market expansion or additional paid request is permitted.
            let slate = try await store.recordedAnalyticsSnapshot(sport: expected.game.league, phase: phase, event: nil, account: account)
            try Task.checkCancellation()
            guard oddsScope == expectedScope, slate.cacheOnly,
                  let metadata = RecordedEventSelection.select(slate.events, sport: expected.game.league,
                    phase: phase, home: expected.game.home, away: expected.game.away, start: start, now: Date()) else { return }
            let rows: [Market]
            if slate.markets.contains(where: { $0.event == metadata.id && $0.kind == "prop" }) {
                rows = slate.markets
            } else {
                let snapshot = try await store.recordedAnalyticsSnapshot(sport: expected.game.league, phase: phase, event: metadata.id, account: account)
                try Task.checkCancellation()
                guard oddsScope == expectedScope, snapshot.cacheOnly else { return }
                rows = snapshot.markets
            }
            // Keep the full event rows so the matcher can reject conflicting IDs.
            recordedMarkets = rows
            prefillLine()
        } catch { /* Cached comparisons and manual entry remain available. */ }
    }
    @MainActor private func discover() async {
        loading = true; message = ""; defer { loading = false }
        do {
            let discovery = try await PlayerMatchupSource().discover(league: candidate.market.sport)
            try Task.checkCancellation()
            events = discovery.events
            let proof = candidate.verifiedSubject
            let matching = events.first { event in proof?.event.map { PlayerMatchupSource.matches($0, event: event, within: events) } ?? false }
                ?? events.first { event in proof?.teamName.map { [$0].contains(event.game.home) || [$0].contains(event.game.away) } ?? false }
            eventID = matching?.id ?? events.first?.id ?? ""
            if events.isEmpty { message = "No verified upcoming games in the next seven days. Refresh later or choose another sport in Analytics." }
            else if discovery.incomplete { message = "Some dates couldn’t refresh. Showing the scheduled games we could verify." }
        } catch is CancellationError { }
        catch { message = "Upcoming games couldn’t be verified. Please try again." }
    }
    @MainActor private func loadRoster() async {
        players = []; playerID = ""; lineDraft = PlayerMatchupLineDraft(); rosterMessage = ""
        guard let expected = event else { return }
        rosterLoading = true; defer { if expected.id == eventID { rosterLoading = false } }
        do {
            let roster = try await PlayerMatchupSource().roster(event: expected)
            try Task.checkCancellation(); guard expected.id == eventID else { return }
            players = roster
            if let selected = PlayerMatchupSource.selection(for: candidate, players: roster) { playerID = selected.id; resetLine() }
            else { rosterMessage = "We couldn’t confirm \(candidate.market.name) on this matchup’s current rosters. Choose a verified player or another game." }
        } catch is CancellationError { }
        catch { if expected.id == eventID { rosterMessage = "Current rosters couldn’t be verified. Choose another game or reopen this matchup." } }
    }
}

struct PlayerMatchupView: View {
    let setup: PlayerMatchupSetup
    var artifactID: String? = nil
    var quote: Market? = nil
    var autoStart = false
    @EnvironmentObject private var account: AccountStore
    @EnvironmentObject private var purchases: PurchaseStore
    @State private var forecast: PlayerMatchupForecast?
    @State private var owner: UUID?
    @State private var message = ""
    @State private var running = false
    @State private var attempted = false
    @State private var retryIsNewRun = false
    @State private var showPro = false
    @State private var didAutoStart = false
    @State private var quoteOwner: UUID?
    var body: some View {
        AccountRequiredView(feature: .privateModels) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text(setup.player).font(.title2.bold())
                    Text(setup.game.title).font(.subheadline)
                    if let start = setup.start { Text(start, format: .dateTime.weekday().month(.abbreviated).day().hour().minute()).font(.caption).foregroundStyle(.secondary) }
                    Text(LabPropCatalog.displayName(setup.stat) + " · " + (LabPropCatalog.isYesNo(setup.stat) ? (setup.side == "Over" ? "Yes" : "No") : setup.side + " " + setup.line.formatted())).font(.headline).foregroundStyle(brandAccent)
                    if let options = LabPropCatalog.optionsLabel(setup.propOptions) { Text(options).font(.caption).foregroundStyle(.secondary) }
                    if running { PLONNLoader(message); Text("You can leave this screen and reopen the same selection to resume.").font(.caption).foregroundStyle(.secondary) }
                    else if !message.isEmpty { Text(message).font(.subheadline).foregroundStyle(.orange) }
                    if let forecast, owner == account.user?.id { result(forecast) }
                    if !running {
                        if purchases.mayUseProFeatures {
                            if (setup.start ?? .distantPast) > Date() {
                                Button(forecast == nil ? (retryIsNewRun ? "Try again" : attempted ? "Resume outlook" : "Build matchup outlook") : "Update outlook") { Task { await run(retrain: forecast != nil || retryIsNewRun) } }.plonnGlassButton(prominent: true)
                                if forecast != nil { Text("Updating uses another model run.").font(.caption).foregroundStyle(.secondary) }
                            }
                        } else { Button("Unlock matchup outlooks with Pro") { showPro = true }.plonnGlassButton(prominent: true) }
                    }
                }.padding(20)
            }.background(navy).navigationTitle("Matchup outlook").navigationBarTitleDisplayMode(.inline)
        }
        .task(id: account.user?.id) {
            if owner != account.user?.id { forecast = nil; owner = nil; message = ""; attempted = false; retryIsNewRun = false }
            guard account.hasSignedInAccount else { return }
            if !didAutoStart && (autoStart || artifactID != nil) {
                didAutoStart = true; quoteOwner = account.user?.id
                await run()
            }
        }
        .sheet(isPresented: $showPro) { NavigationStack { ProPaywallView().toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showPro = false } } } } }
    }
    private func result(_ forecast: PlayerMatchupForecast) -> some View {
        TimelineView(.periodic(from: .now, by: 30)) { _ in
            PlayerMatchupResultCard(setup: setup, forecast: forecast, quote: quoteOwner == account.user?.id ? quote : nil)
        }
    }
    @MainActor private func request(_ path: String, owner: UUID, method: String = "GET", body: Data? = nil, key: String? = nil) async throws -> (Int, [String: Any]) {
        let session = try await account.client.auth.session
        guard account.user?.id == owner else { throw CancellationError() }
        var request = URLRequest(url: CloudConfiguration.labURL.appendingPathComponent(path)); request.httpMethod = method; request.timeoutInterval = 25; request.httpBody = body
        request.setValue("Bearer " + session.accessToken, forHTTPHeaderField: "Authorization")
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        if let key { request.setValue(key, forHTTPHeaderField: "Idempotency-Key") }
        let (data, response) = try await URLSession.shared.data(for: request)
        try Task.checkCancellation(); guard account.user?.id == owner else { throw CancellationError() }
        guard let code = (response as? HTTPURLResponse)?.statusCode, let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw URLError(.cannotParseResponse) }
        return (code, object)
    }
    @MainActor private func run(retrain: Bool = false) async {
        guard !running, account.hasSignedInAccount, let expectedOwner = account.user?.id else { return }
        guard purchases.mayUseProFeatures else { showPro = true; return }
        running = true; attempted = true; message = "Checking your outlook…"
        defer { running = false }
        do {
            let payload = try LabSubmissionStore.canonical(setup.payload()), submissions = LabSubmissionStore()
            if !retrain && !submissions.hasOutstanding(owner: expectedOwner, payload: payload) {
                let path = artifactID.map { "artifacts/" + $0 + "/infer" } ?? "artifacts/resolve"
                let (code, saved) = try await request(path, owner: expectedOwner, method: "POST", body: payload)
                if code == 200 { forecast = try setup.forecast((saved["result"] as? [String: Any]) ?? saved); owner = expectedOwner; message = ""; return }
                if code != 404 { message = PlayerMatchupSetup.failure(saved); return }
                if artifactID != nil { message = "This saved outlook is no longer available."; return }
            }
            guard (setup.start ?? .distantPast) > Date() else { message = "This game has started. Choose another upcoming matchup."; return }
            let record = try submissions.prepare(owner: expectedOwner, payload: payload, retrain: retrain && !submissions.hasOutstanding(owner: expectedOwner, payload: payload))
            var jobID = record.jobID
            if jobID == nil {
                let (code, accepted) = try await request("jobs", owner: expectedOwner, method: "POST", body: payload, key: record.key)
                guard code == 202, let id = accepted["job_id"] as? String else { message = PlayerMatchupSetup.failure(accepted); return }
                jobID = id; submissions.update(record, jobID: id)
            }
            guard let jobID else { throw URLError(.cannotParseResponse) }
            for attempt in 0..<100 {
                if attempt > 0 { try await Task.sleep(nanoseconds: 3_000_000_000) }
                let (code, job) = try await request("jobs/" + jobID, owner: expectedOwner)
                if code == 404 { submissions.update(record, terminal: true); retryIsNewRun = true; message = "This run expired. Try again to build a new outlook."; return }
                guard code == 200 else { message = PlayerMatchupSetup.failure(job); return }
                switch job["status"] as? String {
                case "completed":
                    submissions.update(record, terminal: true)
                    guard let body = job["result"] as? [String: Any] else { throw URLError(.cannotParseResponse) }
                    forecast = try setup.forecast(body); owner = expectedOwner; message = ""; retryIsNewRun = false; return
                case "failed": submissions.update(record, terminal: true); retryIsNewRun = true; message = PlayerMatchupSetup.failure((job["result"] as? [String: Any]) ?? job); return
                case "queued": message = "Waiting to build your outlook…"
                case "running": message = "Checking history, the opponent and model performance…"
                default: throw URLError(.cannotParseResponse)
                }
            }
            message = "Your outlook is still running. Return to this selection to check progress."
        } catch is CancellationError { }
        catch is LabServiceContract.ContractError { message = "This matchup result couldn’t be verified. Please try again later." }
        catch { if !Task.isCancelled { message = "We couldn’t connect. Try again to resume the same request." } }
    }
}

private struct PlayerMatchupResultCard: View {
    let setup: PlayerMatchupSetup
    let forecast: PlayerMatchupForecast
    let quote: Market?
    private var projection: PlayerMatchupForecast.Projection { forecast.inference.projection }
    private var recordedQuote: Market? {
        guard let quote, PlayerMatchupLineMatching.matches(quote, setup: setup) else { return nil }
        return quote
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            if projection.available {
                Text((setup.start ?? .distantPast) > Date() ? "Experimental estimate · Assumes appearance" : "Saved estimate · Assumes appearance")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if projection.available, LabPropCatalog.isYesNo(setup.stat), let probability = projection.selectedProbability {
                Text(probability.formatted(.percent.precision(.fractionLength(0)))).font(.system(size: 34, weight: .bold, design: .rounded)).monospacedDigit().foregroundStyle(brandAccent)
                Text("Estimated " + (setup.side == "Over" ? "Yes" : "No") + " chance").font(.subheadline)
            } else if projection.available, let mean = projection.mean, let low = projection.lower, let high = projection.upper {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Expected").font(.caption).foregroundStyle(.secondary)
                        Text(mean.formatted(.number.precision(.fractionLength(0...1)))).font(.system(size: 30, weight: .bold, design: .rounded)).monospacedDigit()
                        Text(LabPropCatalog.displayName(setup.stat)).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let probability = projection.selectedProbability {
                        VStack(alignment: .trailing, spacing: 4) {
                            Text(LabPropCatalog.isYesNo(setup.stat) ? (setup.side == "Over" ? "Yes chance" : "No chance") : setup.side + " " + setup.line.formatted()).font(.caption).foregroundStyle(.secondary)
                            Text(probability.formatted(.percent.precision(.fractionLength(0)))).font(.system(size: 30, weight: .bold, design: .rounded)).monospacedDigit().foregroundStyle(brandAccent)
                        }
                    }
                }
                PlayerMatchupRange(mean: mean, low: low, high: high, line: setup.line)
                Text(projection.intervalLevel.formatted(.percent.precision(.fractionLength(0))) + " model range · " + low.formatted(.number.precision(.fractionLength(0...1))) + "–" + high.formatted(.number.precision(.fractionLength(0...1))))
                    .font(.caption.weight(.medium))
                if let push = projection.probabilities?.push, push > 0.005 { Text("On the line: " + push.formatted(.percent.precision(.fractionLength(0)))).font(.caption2).foregroundStyle(.secondary) }
            } else {
                Label("Estimate unavailable", systemImage: "info.circle").font(.headline).foregroundStyle(.orange)
                Text(forecast.validation.reasons.first ?? "There isn’t enough evidence to estimate this prop reliably.").font(.caption).foregroundStyle(.secondary)
            }
            if let recordedQuote {
                Text(odds(recordedQuote.price) + " · " + recordedQuote.book).font(.subheadline.bold())
                if let date = recordedQuote.updated.flatMap(sportsDate) { Text("Recorded " + date.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(.secondary) }
            } else { Text("No recorded price for this selection").font(.caption).foregroundStyle(.secondary) }
            Divider()
            ForEach(forecast.inference.factors) { factor in
                VStack(alignment: .leading, spacing: 5) {
                    Label(factor.title, systemImage: factor.direction == "supports" ? "arrow.up.right" : factor.direction == "opposes" ? "arrow.down.right" : "minus.circle")
                        .font(.subheadline.bold()).foregroundStyle(factor.direction == "supports" ? Color.green : factor.direction == "opposes" ? Color.red : Color.secondary)
                        .accessibilityLabel(factor.title + (factor.direction == "supports" ? ", supports this selection" : factor.direction == "opposes" ? ", opposes this selection" : ", neutral context"))
                    Text(factor.detail).font(.caption).foregroundStyle(.secondary)
                }
            }
            DisclosureGroup("Injuries & lineups") { availability.padding(.top, 8) }
            DisclosureGroup("Details") { details.padding(.top, 8) }
            Label("Saved to My Models", systemImage: "checkmark.circle").font(.caption).foregroundStyle(brandAccent)
        }.padding(18).background(navySurface, in: RoundedRectangle(cornerRadius: 18))
    }
    private var details: some View {
        VStack(alignment: .leading, spacing: 10) {
            if projection.available {
                Text("Assumes the player appears. Playing time is not guaranteed, and outcomes can fall outside the range.").font(.caption).foregroundStyle(.secondary)
            }
            if let push = projection.probabilities?.push, push > 0.005 {
                Text("Results on the line may be settled as a push, depending on the sportsbook’s rules.").font(.caption).foregroundStyle(.secondary)
            }
            if projection.available, let adjustment = projection.opponentAdjustment, abs(adjustment) > 0.00001 {
                Text(projection.adjustmentKind == "opponent_defense" ? "Validated opponent-defense adjustment included." : "Validated head-to-head adjustment included.").font(.caption).foregroundStyle(.secondary)
            } else {
                Text("No opponent adjustment applied.").font(.caption).foregroundStyle(.secondary)
            }
            if let deviation = projection.standardDeviation {
                Text("Standard deviation: " + deviation.formatted(.number.precision(.fractionLength(0...1)))).font(.subheadline.bold())
                Text("Describes the model’s spread around its average. It is not a guaranteed ± range.").font(.caption).foregroundStyle(.secondary)
            }
            Text("Distribution: " + projection.distribution.replacingOccurrences(of: "_", with: " ")).font(.caption)
            Text("Training: \(forecast.validation.trainingCount) games · Tested: \(forecast.validation.testCount) newer games").font(.caption)
            if let actual = forecast.validation.brier, let baseline = forecast.validation.baselineBrier {
                Text(actual < baseline ? "Probability estimates beat the simple comparison model in this test." : "Probability estimates did not beat the simple comparison model in this test.").font(.caption).foregroundStyle(.secondary)
            }
            Text("Validation uses this comparison line on historical results, not archived sportsbook prices. It does not establish betting profit.").font(.caption).foregroundStyle(.secondary)
            if !forecast.inference.missingInputs.isEmpty {
                Text("Not modeled").font(.subheadline.bold())
                ForEach(forecast.inference.missingInputs, id: \.self) { Text("• " + $0).font(.caption).foregroundStyle(.secondary) }
            }
            if let generated = forecast.generatedAt { Text("Created " + generated.formatted(date: .abbreviated, time: .shortened)).font(.caption2).foregroundStyle(.secondary) }
        }
    }
    private var availability: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Context only · Not included in the estimate").font(.caption).foregroundStyle(.secondary)
            if let checked = forecast.availability.checked { Text("Checked " + checked.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(.secondary) }
            if let expires = forecast.availability.expires, expires < Date() { Text("This saved availability check has expired. Update the outlook before relying on it.").font(.caption).foregroundStyle(.orange) }
            ForEach(forecast.availability.injuries) { team in
                VStack(alignment: .leading, spacing: 6) {
                    Text(team.team).font(.subheadline.bold())
                    if team.players.isEmpty { Text("No fresh injury report verified.").font(.caption).foregroundStyle(.secondary) }
                    ForEach(team.players) { player in Text(player.name + " · Reported " + player.status).font(.caption) }
                    if let lineup = forecast.availability.lineups.first(where: { $0.team == team.team }) {
                        if let starter = lineup.starter, let name = starter.name { Text((starter.kind == "starting_goalie" ? "Goalie: " : "Starting pitcher: ") + name + (starter.state == "confirmed" ? " · Confirmed" : " · Probable")).font(.caption) }
                        if !lineup.players.isEmpty { Text("Reported starters: " + lineup.players.joined(separator: ", ")).font(.caption) }
                        if lineup.state != "confirmed" { Text("Full lineup not confirmed.").font(.caption).foregroundStyle(.secondary) }
                    }
                }
            }
            Text("Missing reports do not mean everyone is available.").font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct PlayerMatchupRange: View {
    let mean: Double, low: Double, high: Double, line: Double
    private var lower: Double { min(low, line, mean) }
    private var upper: Double { max(high, line, mean) }
    private func position(_ value: Double, width: CGFloat) -> CGFloat { CGFloat((value - lower) / max(upper - lower, 1)) * max(width - 12, 0) + 6 }
    var body: some View {
        VStack(spacing: 5) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.10)).frame(height: 5)
                    Capsule().fill(brandAccent.opacity(0.45)).frame(width: max(position(high, width: geometry.size.width) - position(low, width: geometry.size.width), 5), height: 7).offset(x: position(low, width: geometry.size.width))
                    Circle().fill(brandAccent).frame(width: 10, height: 10).offset(x: position(mean, width: geometry.size.width) - 5)
                    Rectangle().fill(Color.white.opacity(0.8)).frame(width: 2, height: 16).offset(x: position(line, width: geometry.size.width) - 1)
                }.frame(height: 20)
            }.frame(height: 20)
            HStack(spacing: 14) {
                HStack(spacing: 4) { Circle().fill(brandAccent).frame(width: 5, height: 5); Text("Estimate") }
                HStack(spacing: 4) { Rectangle().fill(Color.white.opacity(0.8)).frame(width: 2, height: 8); Text("Line") }
                Spacer()
            }.font(.system(size: 10)).foregroundStyle(.secondary)
        }.accessibilityElement(children: .ignore).accessibilityLabel("Model range \(low.formatted()) to \(high.formatted()), expected \(mean.formatted()), comparison line \(line.formatted())")
    }
}
