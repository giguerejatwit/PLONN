import SwiftUI
import Charts

struct SimulatorView: View {
    let active: Bool
    var body: some View {
        CurrentMatchupSimulatorView(active: active)
            .background(LeagueBackdrop(league: "All"))
            .navigationTitle("Simulator").navigationBarTitleDisplayMode(.inline)
    }
}

private struct CurrentMatchupSimulatorView: View {
    let active: Bool
    @EnvironmentObject private var account: AccountStore
    @EnvironmentObject private var purchases: PurchaseStore
    @EnvironmentObject private var store: Store
    @State private var selection = SimulationSelection()
    @State private var teams: [InterestTeam] = []
    @State private var games: [PlayerMatchupEvent] = []
    @State private var loadingTeams = false
    @State private var loadingGames = false
    @State private var catalogMessage = ""
    @State private var gamesMessage = ""
    @State private var result: PlayerSimulation?
    @State private var showResults = false
    @State private var recordedMarkets: [Market] = []
    @State private var running = false
    @State private var runStage = "Loading player history…"
    @State private var message = ""
    @State private var runTask: Task<Void, Never>?
    @State private var runID = UUID()
    @State private var catalogID = UUID()
    @State private var sheet: Sheet?
    private enum Sheet: String, Identifiable { case away, home, games, method, pro; var id: String { rawValue } }

    var body: some View {
        AccountRequiredView(feature: .privateModels) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Picker("Sport", selection: Binding(get: { selection.sport }, set: { selection.changeSport($0) })) {
                        ForEach(InterestSport.allCases) { Text($0.rawValue).tag($0) }
                    }.pickerStyle(.segmented).accessibilityIdentifier("simulator-sport")
                    HStack {
                        Button { sheet = .games } label: {
                            Label(loadingGames ? "Finding games…" : "Live & upcoming", systemImage: "calendar")
                        }.plonnGlassButton().disabled(loadingTeams || loadingGames || teams.isEmpty)
                            .accessibilityIdentifier("simulator-upcoming")
                        Spacer(minLength: 0)
                        Button { sheet = .method } label: { Image(systemName: "info.circle") }
                            .plonnGlassButton().accessibilityLabel("How the simulator works")
                    }
                    VStack(spacing: 10) {
                        teamButton(home: false)
                        HStack {
                            Text(selection.event == nil ? "Custom matchup" : "Scheduled matchup").font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Button { selection.swap() } label: { Image(systemName: "arrow.up.arrow.down") }
                                .plonnGlassButton().disabled(!selection.ready).accessibilityLabel("Swap home and away teams")
                                .accessibilityIdentifier("simulator-swap")
                        }
                        teamButton(home: true)
                        if let start = selection.event?.game.start {
                            Text(start, format: .dateTime.weekday(.abbreviated).month(.abbreviated).day().hour().minute())
                                .font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    if loadingTeams { PLONNLoader("Loading teams…") }
                    if !catalogMessage.isEmpty {
                        Text(catalogMessage).font(.subheadline).foregroundStyle(.secondary)
                        Button("Reload teams") { Task { await loadCatalog() } }.plonnGlassButton()
                    }
                    Button {
                        if purchases.mayUseProFeatures { simulate() } else { sheet = .pro }
                    } label: {
                        HStack {
                            if running { ProgressView().tint(.white) }
                            Label(running ? runStage : result == nil ? "Simulate matchup" : "Run again", systemImage: "play.fill")
                        }.frame(maxWidth: .infinity).padding(.vertical, 4)
                    }.plonnGlassButton(prominent: true).disabled(!selection.ready || running)
                        .accessibilityIdentifier("simulator-run")
                    if !message.isEmpty { Text(message).font(.subheadline).foregroundStyle(.orange).accessibilityIdentifier("simulator-error") }
                    if let result, result.away.team == selection.away, result.home.team == selection.home {
                        Button("Open simulation") { showResults = true }.plonnGlassButton().accessibilityIdentifier("simulator-open-results")
                    }
                    else if !running {
                        Text("Choose any two teams, or choose a live or upcoming game.")
                            .font(.subheadline).foregroundStyle(.secondary)
                    }
                }.padding(16)
            }.background(LeagueBackdrop(league: selection.sport.rawValue))
        }
        .navigationTitle("Simulator").navigationBarTitleDisplayMode(.inline)
        .navigationDestination(isPresented: $showResults) {
            if let result, result.away.team == selection.away, result.home.team == selection.home {
                PlayerSimulationView(result: result, event: selection.event, markets: recordedMarkets, active: active, rerun: { simulate() })
            }
        }
        .task(id: selection.sport) { await loadCatalog() }
        .onChange(of: selection.revision) { _, _ in clearRun() }
        .onChange(of: account.user?.id) { _, _ in clearRun() }
        .onChange(of: purchases.mayUseProFeatures) { _, allowed in if !allowed { clearRun() } }
        .onChange(of: active) { _, value in if !value && running { clearRun() } }
        .onDisappear { if running { clearRun() } }
        .sheet(item: $sheet) { item in
            switch item {
            case .away, .home:
                SimulationTeamPicker(teams: teams, excluding: item == .home ? selection.away?.id : selection.home?.id,
                                     title: item == .home ? "Home team" : "Away team") { team in
                    selection.choose(team, homeSide: item == .home); sheet = nil
                }
            case .games: gamesSheet
            case .method: methodSheet
            case .pro:
                NavigationStack { ProPaywallView().toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { sheet = nil } } } }
            }
        }
    }
    private func teamButton(home: Bool) -> some View {
        let team = home ? selection.home : selection.away
        return Button { sheet = home ? .home : .away } label: {
            HStack(spacing: 14) {
                Text(team?.abbreviation ?? "—").font(.headline.monospaced()).foregroundStyle(PLONNPalette.highlight)
                    .frame(width: 48, height: 44).background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
                VStack(alignment: .leading, spacing: 4) {
                    Text(home ? "Home" : "Away").font(.caption).foregroundStyle(.secondary)
                    Text(team?.name ?? "Choose team").font(.headline).multilineTextAlignment(.leading)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.up.chevron.down").font(.caption)
            }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 4)
        }.plonnGlassButton().disabled(teams.isEmpty || loadingTeams)
            .accessibilityIdentifier(home ? "simulator-home-team" : "simulator-away-team")
    }
    private var gamesSheet: some View {
        NavigationStack {
            List {
                if !gamesMessage.isEmpty { Text(gamesMessage).foregroundStyle(.secondary) }
                ForEach(games) { game in
                    Button {
                        selection.choose(game, teams: teams); sheet = nil
                    } label: {
                        VStack(alignment: .leading, spacing: 5) {
                            Text(game.game.title).font(.headline).foregroundStyle(.primary)
                            if game.isLive { Text("LIVE").font(.caption.bold()).foregroundStyle(.green) }
                            if let date = game.game.start { Text(date, format: .dateTime.weekday(.abbreviated).month().day().hour().minute()).font(.caption).foregroundStyle(.secondary) }
                        }.padding(.vertical, 6)
                    }.accessibilityIdentifier("simulator-game-" + game.game.eventID)
                }
                if games.isEmpty { Text("Choose any two teams on the Simulator to create a custom matchup.").font(.subheadline).foregroundStyle(.secondary) }
            }.navigationTitle(selection.sport.rawValue + " · Next 7 days").navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { sheet = nil } } }
        }.preferredColorScheme(.dark)
    }
    private var methodSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("5,000 simulated games").font(.title2.bold())
                    Text("Current rosters and injuries are checked before each run. Reported-out players are excluded; uncertain statuses remain visible. Recent box scores supply player rates for sport-specific plays.")
                    Text("Up to 10 completed regular-season and playoff games per team, including the previous season when needed. At least five games per team are required. Preseason and unfinished games are excluded.")
                    Text("These are experimental scenarios, not calibrated forecasts. Starters and rotations are estimated from recent appearances, not confirmed lineups. Equal regulation scores remain level.")
                    Text("Watch illustrative plays, inspect player stats and compare existing recorded odds. Availability is checked every minute while the results are visible and when you return.")
                    Text("Source: ESPN schedules, completed box scores and rosters").font(.caption).foregroundStyle(.secondary)
                }.padding(20)
            }.background(navy).navigationTitle("Simulation details").navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { sheet = nil } } }
        }.preferredColorScheme(.dark)
    }
    @MainActor private func clearRun() {
        runTask?.cancel(); runTask = nil; runID = UUID(); result = nil; showResults = false; recordedMarkets = []; running = false; message = ""
    }
    @MainActor private func loadCatalog() async {
        let token = UUID(); catalogID = token
        let sport = selection.sport
        teams = []; games = []; catalogMessage = ""; gamesMessage = ""; loadingTeams = true; loadingGames = true
        async let discovery = try? SimulationLiveSource().discover(league: sport.rawValue)
        do {
            let fetched = try await InterestsTeamDirectory().load(sport)
            guard !Task.isCancelled, catalogID == token, selection.sport == sport else { return }
            teams = fetched; loadingTeams = false
        } catch {
            guard !Task.isCancelled, catalogID == token else { return }
            loadingTeams = false; catalogMessage = "Teams couldn’t load. Check your connection and try again."
        }
        let discovered = await discovery
        guard !Task.isCancelled, catalogID == token, selection.sport == sport else { return }
        loadingGames = false
        games = (discovered?.events ?? []).filter { event in
            teams.contains { $0.providerID == event.homeID } && teams.contains { $0.providerID == event.awayID }
        }
        if discovered == nil { gamesMessage = "The schedule couldn’t load. Custom matchups are still available." }
        else if games.isEmpty { gamesMessage = "No live or upcoming games found in the next seven days." }
        else if discovered?.incomplete == true { gamesMessage = "Some dates couldn’t refresh. Showing verified games." }
        // Populate once from the schedule, never replace a user's custom selection.
        if selection.away == nil && selection.home == nil, let first = games.first { selection.choose(first, teams: teams) }
    }
    @MainActor private func simulate() {
        guard account.hasSignedInAccount, purchases.mayUseProFeatures, selection.ready,
              let away = selection.away, let home = selection.home else { return }
        clearRun()
        let token = runID, revision = selection.revision, owner = account.user?.id, now = Date()
        let event = selection.event
        let cutoff = min(now, event?.game.start ?? now)
        running = true; runStage = "Loading player history…"
        runTask = Task {
            do {
                async let a = MatchupSimulationSource().history(away, cutoff: cutoff)
                async let h = MatchupSimulationSource().history(home, cutoff: cutoff)
                let histories = try await (a, h)
                try Task.checkCancellation()
                async let awayPlayers = PlayerSimulationSource().history(histories.0, cutoff: cutoff)
                async let homePlayers = PlayerSimulationSource().history(histories.1, cutoff: cutoff)
                async let quotes = loadRecordedMarkets(event)
                let playerHistories = try await (awayPlayers, homePlayers)
                try Task.checkCancellation()
                runStage = "Checking rosters & injuries…"
                async let awayAvailability = SimulationAvailabilitySource().load(away)
                async let homeAvailability = SimulationAvailabilitySource().load(home)
                let availability = try await [awayAvailability, homeAvailability]
                try Task.checkCancellation()
                var live: SimulationLiveSnapshot?
                if let event, let start = event.game.start, start <= Date() {
                    runStage = "Reading live plays…"
                    // Open the live screen even when its first read is delayed.
                    // It hides the pregame forecast and retries automatically.
                    live = try? await SimulationLiveSource().fetch(event)
                    try Task.checkCancellation()
                }
                let snapshot = live?.forecastIssue == nil ? live : nil
                runStage = snapshot == nil ? "Simulating 5,000 games…" : "Simulating the remaining game…"
                let recorded = await quotes
                let market = SimulationForecast.market(event: event, markets: recorded, now: Date())
                #if DEBUG
                // Explicit local QA capture contains public market data only.
                if ProcessInfo.processInfo.arguments.contains("-simulation-audit"), let data = try? JSONEncoder().encode(recorded) {
                    try? data.write(to: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("simulation-recorded-quotes.json"), options: .atomic)
                }
                NSLog("PLONNSimForecast market=%@ contracts=%ld", market?.book ?? "none", market?.quoteIDs.count ?? 0)
                #endif
                let worker = Task.detached(priority: .userInitiated) {
                    try PlayerSimulation.currentRun(away: playerHistories.0, home: playerHistories.1, availability: availability, now: Date(), seed: UInt64.random(in: 0...UInt64.max), live: snapshot, market: market)
                }
                let generated = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
                guard !Task.isCancelled, token == runID, revision == selection.revision, owner == account.user?.id else { return }
                result = generated; recordedMarkets = recorded; running = false; showResults = true
            } catch {
                guard !Task.isCancelled, token == runID, revision == selection.revision, owner == account.user?.id else { return }
                running = false
                if case MatchupSimulation.Failure.history = error { message = "Not enough recent completed games for this matchup. Both teams need at least five verified player box scores." }
                else if error is SportPlaySimulation.Failure { message = "The current available roster doesn’t have enough verified player history for this simulation. No past-roster replacements were used." }
                else { message = "Player box scores or rosters couldn’t be verified. Try again or choose another matchup." }
            }
        }
    }

    @MainActor private func loadRecordedMarkets(_ event: PlayerMatchupEvent?) async -> [Market] {
        guard let event, let start = event.game.start, start > Date() else { return [] }
        let retained = store.analyticsSnapshots(for: account)
        var fallback = retained
        let phase = NBARecordedOddsRouting.phase(sport: event.game.league, seasonTypes: [event.seasonType])
        #if DEBUG
        NSLog("PLONNSimOdds phase=%@ retained=%ld", String(describing: phase), retained.count)
        #endif
        guard phase != .unknown else { return retained }
        do {
            let slate = try await store.recordedAnalyticsSnapshot(sport: event.game.league, phase: phase, event: nil, account: account)
            try Task.checkCancellation()
            if slate.cacheOnly { fallback = slate.markets }
            #if DEBUG
            NSLog("PLONNSimOdds slate=%ld metadata=%ld", slate.markets.count, slate.events.count)
            #endif
            guard slate.cacheOnly, let metadata = RecordedEventSelection.select(slate.events, sport: event.game.league, phase: phase,
                home: event.game.home, away: event.game.away, start: start, now: Date()) else { return fallback }
            let detail = try await store.recordedAnalyticsSnapshot(sport: event.game.league, phase: phase, event: metadata.id, account: account)
            try Task.checkCancellation()
            guard detail.cacheOnly else { return slate.markets }
            #if DEBUG
            NSLog("PLONNSimOdds detail=%ld", detail.markets.count)
            #endif
            // Prefer the event snapshot, retaining slate-only contracts without
            // replacing a newer quote with a same-ID local cache copy.
            let ids = Set(detail.markets.map(\.id))
            return detail.markets + slate.markets.filter { !ids.contains($0.id) }
        } catch {
            #if DEBUG
            NSLog("PLONNSimOdds read failed: %@", String(describing: type(of: error)))
            #endif
            return fallback
        }
    }

}

private struct SimulationTeamPicker: View {
    let teams: [InterestTeam]
    let excluding: String?
    let title: String
    let choose: (InterestTeam) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    private var filtered: [InterestTeam] {
        teams.filter { $0.id != excluding && (query.isEmpty || ($0.name + " " + $0.abbreviation).localizedCaseInsensitiveContains(query)) }
    }
    var body: some View {
        NavigationStack {
            List(filtered) { team in
                Button { choose(team) } label: {
                    HStack {
                        Text(team.abbreviation).font(.subheadline.monospaced().bold()).foregroundStyle(PLONNPalette.highlight).frame(width: 50, alignment: .leading)
                        Text(team.name).foregroundStyle(.primary)
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 2)
                }.accessibilityIdentifier("simulator-team-" + team.id)
            }.searchable(text: $query, prompt: "Team or abbreviation")
                .overlay { if filtered.isEmpty { ContentUnavailableView.search(text: query) } }
                .navigationTitle(title).navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }.preferredColorScheme(.dark)
    }
}

#if DEBUG && targetEnvironment(simulator)
/// Explicitly synthetic, local-only layout/playback fixture. Never ships on devices.
struct SimulationLiveDemo: View {
    static var active: Bool { ProcessInfo.processInfo.environment["PLONN_LIVE_SIMULATION_DEMO"] == "1" }
    @State private var result: PlayerSimulation?
    @State private var event: PlayerMatchupEvent?
    @State private var error = ""
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Text("SYNTHETIC CHECKPOINT · UI TEST").font(.caption.bold()).foregroundStyle(.orange).padding(8)
                if let result, let event { PlayerSimulationView(result: result, event: event, markets: [], active: false, rerun: {}) }
                else if !error.isEmpty { Text(error) }
                else { ProgressView("Preparing local checkpoint…") }
            }.background(PLONNBrandBackdrop())
        }.task {
            do {
                let now = Date(), start = now.addingTimeInterval(-3600)
                var histories: [PlayerSimulation.History] = [], availability: [SimulationAvailability] = []
                for side in 0...1 {
                    let id = String(side + 1), team = InterestTeam(sport: .nba, providerID: String(side + 1), name: side == 0 ? "Test Away" : "Test Home", abbreviation: side == 0 ? "AWY" : "HME")
                    let roster = (1...5).map { PlayerMatchupPlayer(id: id + String($0), name: "Player \(id)\($0)", teamID: id, team: team.name, position: "G") }
                    let values: [String:Double] = ["minutes":48,"points":23,"fieldGoalsAttempted":20,"fieldGoalsMade":10,"threePointFieldGoalsAttempted":0,"threePointFieldGoalsMade":0,"freeThrowsAttempted":4,"freeThrowsMade":3,"turnovers":3,"offensiveRebounds":3,"rebounds":10,"assists":5,"steals":1,"blocks":1]
                    let boxes = (1...6).map { day in PlayerSimulation.Box(game: .init(id: "99\(side)\(day)", date: start.addingTimeInterval(-Double(day)*86400), season: 2027, scored:115, allowed:115, home:side == 1), lines: roster.map { .init(playerID:$0.id,name:$0.name,group:"players",values:values) }) }
                    histories.append(.init(team:team,roster:roster,boxes:boxes,partial:false))
                    availability.append(.init(roster:roster,observations:[],checked:now,injuryFeedUpdated:now,injuriesVerified:true))
                }
                let teams = try (0...1).map { try SportPlaySimulation.team(histories[$0], availability:availability[$0]) }
                let full = try SportPlaySimulation.run(away:teams[0],home:teams[1],seed:71)
                let prefix = full.through(full.plays.firstIndex(where: { $0.period >= 4 }) ?? full.plays.count/2)
                let live = SimulationLiveSnapshot(eventID:"999000001",sport:.nba,teamIDs:["1","2"],start:start,checked:now,phase:.live,status:"4th quarter · 12:00 · synthetic checkpoint",scores:[prefix.awayScore,prefix.homeScore],period:4,remaining:720,possession:0,down:nil,distance:nil,yard:nil,battingSide:nil,outs:nil,bases:[nil,nil,nil],batter:nil,pitcher:nil,battingOrders:[[],[]],lines:[prefix.away,prefix.home],events:[.init(id:"test-play",text:"Synthetic play recorded. Observed scores stay fixed while the remaining quarter is simulated.",period:3,clock:"0:00",awayScore:prefix.awayScore,homeScore:prefix.homeScore)],fingerprint:"synthetic-ui-checkpoint")
                let a = histories[0], h = histories[1], snapshots = availability
                result = try await Task.detached { try PlayerSimulation.currentRun(away:a,home:h,availability:snapshots,now:Date(),seed:54,live:live) }.value
                event = .init(game:.init(league:"NBA",eventID:live.eventID,home:h.team.name,away:a.team.name,startsAt:ISO8601DateFormatter().string(from:start)),homeID:"2",awayID:"1",seasonType:2,isLive:true)
            } catch { self.error = String(describing:error) }
        }
    }
}
#endif
