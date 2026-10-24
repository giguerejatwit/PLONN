import SwiftUI
import Charts

struct PlayerSimulationView: View {
    private let initialResult: PlayerSimulation
    @State private var updatedResult: PlayerSimulation?
    private var result: PlayerSimulation { updatedResult ?? initialResult }
    init(result: PlayerSimulation, event: PlayerMatchupEvent?, markets: [Market], active: Bool, rerun: @escaping () -> Void) {
        initialResult = result; self.event = event; self.markets = markets; self.active = active; self.rerun = rerun
    }
    let event: PlayerMatchupEvent?
    let markets: [Market]
    let active: Bool
    let rerun: () -> Void
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.scenePhase) private var scenePhase
    @EnvironmentObject private var builderStore: Store
    @State private var builderNotices: [String: String] = [:]
    @State private var representative: MatchupSimulation.RepresentativeScenario?
    @State private var trial = 0
    @State private var homeSide = false
    @State private var tab = "Players"
    @State private var groupID = ""
    @State private var average = false
    @State private var playing = false
    @State private var detail: PlayerSimulation.Line?
    @State private var comparisons: [SimulationOdds.Comparison] = []
    @State private var matchingOdds = true
    @State private var book = "All books"
    @State private var replay: SportPlaySimulation.Game?
    @State private var frame: SportPlaySimulation.Game?
    @State private var cursor = 0
    @State private var progress = 0.0
    @State private var speed = 2.0
    @State private var watch: SimulationAvailabilityWatch?
    @State private var checking = false
    @State private var playbackError = ""
    @State private var liveSnapshot: SimulationLiveSnapshot?
    @State private var liveMessage = ""
    @State private var updatingLive = false
    @State private var lastLiveProgress = Date()
    private var observed: SimulationLiveSnapshot? { liveSnapshot ?? result.current?.live }
    private var hasStarted: Bool { event?.game.start.map { $0 <= Date() } == true }
    private var usable: Bool {
        if observed?.phase == .final { return true }
        return watch?.usable(at: Date()) == true && (!hasStarted || (liveMessage.isEmpty && observed?.isStale(at: Date()) == false && result.current?.live?.fingerprint == observed?.fingerprint))
    }
    private var currentPlay: SportPlaySimulation.Play? {
        guard let replay, !replay.plays.isEmpty else { return nil }
        return replay.plays[min(cursor, replay.plays.count - 1)]
    }
    private var score: MatchupSimulation.Score { .init(away: frame?.awayScore ?? 0, home: frame?.homeScore ?? 0) }
    private var history: PlayerSimulation.History { homeSide ? result.home : result.away }
    private var groups: [PlayerSimulation.Group] {
        let modeled = Set((replay?.away ?? []).map(\.group) + (replay?.home ?? []).map(\.group))
        let available = PlayerSimulation.groups(result.away.team.sport).filter { modeled.contains($0.id) }
        return available.isEmpty ? [PlayerSimulation.groups(result.away.team.sport)[0]] : available
    }
    private var group: PlayerSimulation.Group { groups.first { $0.id == groupID } ?? groups[0] }
    private var box: PlayerSimulation.Box {
        .init(game: .init(id: "replay", date: result.scores.generated, season: 0, scored: homeSide ? score.home : score.away, allowed: homeSide ? score.away : score.home, home: homeSide), lines: homeSide ? frame?.home ?? [] : frame?.away ?? [])
    }
    private var lines: [PlayerSimulation.Line] {
        let templates = result.current?.templates[homeSide ? 1 : 0] ?? []
        return templates.filter { $0.group == group.id }.sorted { $0.name < $1.name }
    }
    var body: some View {
        ScrollViewReader { scroll in
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                liveStatus
                availabilityStatus
                if !hasStarted || usable { broadcast }
                else { Text("Forecast paused until the live state is verified.").font(.caption).foregroundStyle(.secondary) }
                if !playbackError.isEmpty { Text(playbackError).font(.caption).foregroundStyle(.orange) }
                Text("Illustrative movement · Experimental simulation").font(.caption2).foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(["Players", "Odds", "Plays", "Outcomes"], id: \.self) { item in
                            Button(item) {
                                tab = item
                                withAnimation(reduceMotion ? nil : .easeOut(duration: 0.22)) {
                                    scroll.scrollTo("simulation-result-controls", anchor: .top)
                                }
                            }.plonnGlassButton(prominent: tab == item)
                                .accessibilityAddTraits(tab == item ? .isSelected : [])
                        }
                    }.padding(.vertical, 3)
                }.id("simulation-result-controls").accessibilityIdentifier("simulation-results-tabs")
                if !hasStarted || usable || tab == "Plays" || tab == "Odds" {
                switch tab {
                case "Players": players
                case "Odds": oddsPanel
                case "Plays": playLog
                default:
                    if observed?.phase == .final { Text("Game complete · the observed score is final.").font(.subheadline) }
                    else { distribution }
                }
                }
                DisclosureGroup("Data & method") { method.font(.caption).foregroundStyle(.secondary).padding(.top, 8) }.tint(brandAccent)
            }.padding(16)
        }.background(PLONNBrandBackdrop())
        }
            .navigationTitle("\(result.away.team.abbreviation) × \(result.home.team.abbreviation)").navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden()
            .toolbar { ToolbarItem(placement: .topBarLeading) {
                Button { playing = false; dismiss() } label: { Label("Teams", systemImage: "chevron.left") }
                    .accessibilityIdentifier("simulation-change-teams")
            } }
            .onAppear {
                if let current = result.current, watch == nil { let values = current.teams.map(\.availability); watch = .init(original: values, latest: values) }
                groupID = groups[0].id; chooseRepresentative(); loadReplay()
            }
            .onChange(of: trial) { _, _ in loadReplay() }
            .onChange(of: cursor) { _, _ in frame = replay?.through(cursor) }
            .onDisappear { playing = false }
            .onChange(of: scenePhase) { _, phase in if phase != .active { playing = false } }
            .onChange(of: tab) { _, value in if value != "Players" { playing = false } }
            .task(id: playing) {
                guard playing else { return }
                let clock = ContinuousClock()
                var previousTick = clock.now
                while !Task.isCancelled && playing {
                    do { try await Task.sleep(for: .milliseconds(16)); try Task.checkCancellation() } catch { return }
                    let tick = clock.now, elapsed = previousTick.duration(to: tick).components
                    let delta = min(0.1, Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18)
                    previousTick = tick
                    guard usable, let replay, cursor < replay.plays.count else { playing = false; return }
                    progress += delta * speed / 3.2
                    if progress >= 1 { progress -= 1; cursor += 1; if cursor >= replay.plays.count { progress = 1; playing = false } }
                }
            }
            .task(id: active && scenePhase == .active) {
                guard active && scenePhase == .active else { playing = false; return }
                repeat {
                    matchingOdds = true
                    if observed?.phase != .final { await refreshAvailability() }
                    if hasStarted { await refreshLive() }
                    if usable && !hasStarted {
                        let result = result, event = event, markets = markets
                        let updated = await Task.detached(priority: .utility) { SimulationOdds.compare(result, event: event, markets: markets, now: Date()) }.value
                        guard !Task.isCancelled else { return }
                        comparisons = usable ? updated : []
                        #if DEBUG
                        NSLog("PLONNSimOdds received=%ld comparisons=%ld usable=%@", markets.count, comparisons.count, String(usable))
                        #endif
                    } else { comparisons = [] }
                    matchingOdds = false
                    if observed?.phase == .final { return }
                    do { try await Task.sleep(for: .seconds(30)) } catch { return }
                } while !Task.isCancelled
            }
            .sheet(item: $detail) { line in playerDetail(line) }
    }
    private var replayFinished: Bool { replay.map { cursor >= $0.plays.count } ?? false }
    private var periodLabel: String {
        guard let play = currentPlay else { return "" }
        switch result.away.team.sport {
        case .mlb: return "Inning \(play.period)"
        case .nhl: return "P\(play.period)"
        case .nba, .nfl: return play.period <= 4 ? "Q\(play.period)" : "OT\(play.period - 4)"
        }
    }
    private var currentAction: String {
        guard !replayFinished, let play = currentPlay else { return replayFinished ? "Replay complete" : "Preparing plays…" }
        let player = result.current?.teams[play.home ? 1 : 0].profiles.first { $0.id == play.actorID }?.player.name ?? "Player"
        switch result.away.team.sport {
        case .mlb: return player + " at bat"
        case .nhl: return player + " with the puck"
        case .nba, .nfl: return player + " with the ball"
        }
    }
    private var broadcast: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text(observed?.phase == .final ? "Observed result" : hasStarted ? "Remaining game · representative scenario" : "Representative scenario").font(.caption).foregroundStyle(.secondary)
                Spacer(minLength: 6)
                if observed?.phase != .final, let scenario = representative {
                    Text(scenario.tiedForTop ? "Even outcome split" :
                         (scenario.outcome == .level ? "Level" : (scenario.outcome == .away ? result.away.team.abbreviation : result.home.team.abbreviation) + " ahead") +
                         " · " + scenario.share.formatted(.percent.precision(.fractionLength(1))))
                        .font(.caption.weight(.semibold)).foregroundStyle(PLONNPalette.highlight)
                }
            }
            if observed?.phase != .final, let forecast = result.current?.forecast {
                HStack {
                    Text(forecast.market.map { "Market-informed · " + $0.book } ?? "History + opponent · no matched odds")
                    Spacer(minLength: 0)
                    if forecast.limited { Image(systemName: "exclamationmark.circle").accessibilityLabel("Limited simulation support") }
                }.font(.caption2).foregroundStyle(.secondary)
            }
            HStack {
                teamScore(result.away.team, score: score.away, color: PLONNPalette.ink)
                VStack(spacing: 3) {
                    Text(observed?.phase == .final ? "OBSERVED" : "SIMULATED").font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                    Text(replayFinished ? "Final" : currentPlay?.clock ?? "Ready").font(.title3.monospacedDigit().bold())
                    if !replayFinished { Text(periodLabel).font(.caption2).foregroundStyle(.secondary) }
                }.frame(maxWidth: .infinity)
                teamScore(result.home.team, score: score.home, color: PLONNPalette.fuchsia)
            }
            SimulationPlaySurface(sport: result.away.team.sport, teams: result.current?.teams ?? [], play: currentPlay, previous: replay.flatMap { cursor > 0 && cursor <= $0.plays.count ? $0.plays[max(0, min(cursor - 1, $0.plays.count - 1))] : nil }, progress: replayFinished ? 1 : progress, reduceMotion: reduceMotion)
                .aspectRatio(1.78, contentMode: .fit).clipShape(RoundedRectangle(cornerRadius: 16))
                .accessibilityLabel("Illustrative simulated play. " + currentAction)
            HStack(alignment: .top) {
                Text(currentAction).font(.subheadline).frame(maxWidth: .infinity, alignment: .leading)
                Text("\(cursor)/\(replay?.plays.count ?? 0)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            VStack(spacing: 6) {
                Slider(value: Binding(get: { Double(cursor) }, set: { playing = false; progress = 0; cursor = Int($0) }), in: 0...Double(max(1, replay?.plays.count ?? 1)), step: 1)
                    .tint(brandAccent).accessibilityLabel("Replay progress").accessibilityValue("\(cursor) completed plays")
                HStack(spacing: 8) {
                    Button { playing = false; cursor = 0; progress = 0 } label: {
                        Image(systemName: "backward.end.fill").frame(minWidth: 20, minHeight: 28)
                    }.plonnGlassButton().accessibilityLabel("Restart replay")
                    Button { if replayFinished { cursor = 0; progress = 0 }; playing.toggle() } label: {
                        Label(playing ? "Pause" : "Play", systemImage: playing ? "pause.fill" : "play.fill").frame(minWidth: 58, minHeight: 28)
                    }.plonnGlassButton(prominent: true).disabled(!usable || replay == nil).accessibilityIdentifier("simulation-play")
                    Button { playing = false; progress = 0; cursor = min(cursor + 1, replay?.plays.count ?? 0) } label: {
                        Image(systemName: "forward.end.fill").frame(minWidth: 20, minHeight: 28)
                    }.plonnGlassButton().disabled(replayFinished || replay == nil).accessibilityLabel("Next play").accessibilityIdentifier("simulation-next")
                    Spacer(minLength: 0)
                    Menu {
                        ForEach([1.0, 2, 4, 8], id: \.self) { rate in Button("\(Int(rate))×") { speed = rate } }
                    } label: { Text("\(Int(speed))×").monospacedDigit().frame(minWidth: 20, minHeight: 28) }
                        .plonnGlassButton().accessibilityLabel("Playback speed")
                }.plonnGlassGroup()
            }.disabled(!usable)
        }.padding(12).background(navySurface, in: RoundedRectangle(cornerRadius: 22))
    }
    private var playLog: some View {
        LazyVStack(alignment: .leading, spacing: 12) {
            if let live = observed {
                Text("Observed plays").font(.caption.bold()).foregroundStyle(.green)
                ForEach(Array(live.events.suffix(30)).reversed()) { play in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(play.awayScore)–\(play.homeScore) · \(play.clock)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        Text(play.text).font(.subheadline)
                    }
                    Divider()
                }
                if observed?.phase != .final { Text("Simulated remainder").font(.caption.bold()).foregroundStyle(brandAccent) }
            }
            if cursor == 0 && observed?.phase != .final { Text("Play or step forward to see completed plays.").font(.subheadline).foregroundStyle(.secondary) }
            ForEach(Array((usable ? replay?.plays ?? [] : []).prefix(cursor)).reversed()) { play in
                HStack(alignment: .top, spacing: 10) {
                    Text((result.away.team.sport == .mlb ? "Inning " : result.away.team.sport == .nhl ? "P" : "Q") + "\(play.period)\n" + play.clock)
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                    Text(play.title).font(.subheadline).frame(maxWidth: .infinity, alignment: .leading)
                    if play.points > 0 { Text("+\(play.points)").font(.caption.monospacedDigit()).foregroundStyle(PLONNPalette.highlight) }
                }
                Divider()
            }
        }.padding(14).frame(maxWidth: .infinity, alignment: .leading).background(navySurface, in: RoundedRectangle(cornerRadius: 20))
    }
    private func teamScore(_ team: InterestTeam, score: Int, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(team.abbreviation).font(.subheadline.bold()).foregroundStyle(color)
            Text("\(score)").font(.system(size: 30, weight: .semibold, design: .rounded)).monospacedDigit()
        }.frame(maxWidth: .infinity).accessibilityElement(children: .combine)
    }
    private var players: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Picker("Team", selection: $homeSide) {
                    Text(result.away.team.abbreviation).tag(false); Text(result.home.team.abbreviation).tag(true)
                }.pickerStyle(.segmented).accessibilityIdentifier("simulation-player-team")
                Menu {
                    Button("This run") { average = false }
                    Button("Average") { average = true }
                } label: { Label(average ? "Average" : "This run", systemImage: "chevron.down").font(.caption) }
                    .plonnGlassButton().accessibilityIdentifier("simulation-stat-mode")
            }
            if groups.count > 1 {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(groups) { value in
                            Button(value.title) { groupID = value.id }.font(.caption).plonnGlassButton(prominent: group.id == value.id)
                        }
                    }
                }
            }
            ScrollView(.horizontal, showsIndicators: true) {
                VStack(spacing: 0) {
                    HStack(spacing: 0) {
                        Text("PLAYER").frame(width: 154, alignment: .leading)
                        ForEach(group.stats) { stat in Text(stat.label).frame(width: 55).accessibilityLabel(stat.title) }
                    }.font(.caption2).foregroundStyle(.secondary).padding(.vertical, 10)
                    ForEach(lines) { line in
                        Button { playing = false; detail = line } label: {
                            HStack(spacing: 0) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(line.name).font(.subheadline).lineLimit(1)
                                    if let status = watch?.latest[homeSide ? 1 : 0].observations.first(where: { $0.playerID == line.playerID }) { Text(status.status).font(.caption2).foregroundStyle(.orange) }
                                }.frame(width: 154, alignment: .leading)
                                ForEach(group.stats) { stat in
                                    let value = average ? result.range(homeSide: homeSide, lineID: line.id, stat: stat.id)?.average : box.lines.first { $0.id == line.id }?.values[stat.id]
                                    Text(value.map { stat.display($0, average: average) } ?? "—").font(.subheadline.monospacedDigit())
                                        .foregroundStyle(stat.id == group.stats[0].id ? brandAccent : .primary).frame(width: 55)
                                }
                            }.frame(minHeight: 43).contentShape(Rectangle())
                        }.buttonStyle(.plain).accessibilityIdentifier("simulation-player-" + line.id)
                        Divider().opacity(0.3)
                    }
                }.padding(.horizontal, 12)
            }.background(navySurface.opacity(0.5), in: RoundedRectangle(cornerRadius: 16))
            if lines.isEmpty { Text("No simulated \(group.title.lowercased()) stats for this roster.").font(.subheadline).foregroundStyle(.secondary) }
            Text(average ? "Simulated averages. Tap a player for ranges or why a stat is unavailable." : "Stats update as plays finish. — means this stat isn’t modeled in this run.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
    private var oddsPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            if hasStarted {
                ContentUnavailableView("Live prices unavailable", systemImage: "clock", description: Text("The simulation updates from live plays. Recorded pregame odds are not used as live prices."))
            } else if event == nil {
                ContentUnavailableView("Choose a scheduled game", systemImage: "calendar", description: Text("Custom matchups have no sportsbook market. Choose Live & upcoming to compare recorded pregame odds."))
            } else if matchingOdds && comparisons.isEmpty {
                ProgressView("Matching recorded prices…").font(.subheadline).tint(brandAccent).padding(.vertical, 20)
            } else if comparisons.isEmpty {
                ContentUnavailableView("No matching recorded odds", systemImage: "chart.line.uptrend.xyaxis", description: Text("Prices must match this game, its start time and the selected player or team. Simulations are still available."))
            } else {
                HStack {
                    Picker("Sportsbook", selection: $book) {
                        Text("All books").tag("All books")
                        ForEach(Array(Set(comparisons.map { $0.quote.book })).sorted(), id: \.self) { Text($0).tag($0) }
                    }.tint(brandAccent)
                    Spacer()
                    Label("Hit rate", systemImage: "arrow.down").font(.caption).foregroundStyle(.secondary)
                        .accessibilityLabel("Sorted by scenario hit rate, highest first")
                }
                Text(result.current?.forecast?.market == nil ? "Scenario hit rates are experimental. Implied probability includes the bookmaker’s margin." : "Odds helped shape these scenarios; the difference is not an independent betting edge.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(SimulationOdds.ranked(comparisons.filter { book == "All books" || $0.quote.book == book })) { value in
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(value.quote.name).font(.headline)
                                Text(value.quote.label).font(.subheadline).foregroundStyle(brandAccent)
                            }
                            Spacer()
                            Text(odds(value.quote.price)).font(.headline.monospacedDigit())
                        }
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(value.hitRate.map { String(format: "%.1f%%", $0 * 100) } ?? value.missingEstimate?.title ?? "No estimate")
                                    .font(value.hitRate == nil ? .subheadline.bold() : .title3.bold())
                                Text("Scenario hit rate").font(.caption2).foregroundStyle(.secondary)
                            }
                            Spacer()
                            VStack(alignment: .trailing, spacing: 4) {
                                Text(value.implied.map { String(format: "%.1f%%", $0 * 100) } ?? "—").font(.title3)
                                Text("Odds implied").font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                        if let missing = value.missingEstimate {
                            Text(missing.detail).font(.caption2).foregroundStyle(.secondary)
                        } else if let appearances = value.playerAppearances {
                            Text("\(appearances) recorded games · assumed role")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        if value.counted > 0 && (value.push > 0 || value.unavailable > 0) {
                            Text("\(value.push) pushes · \(value.unavailable) unscored scenarios").font(.caption2).foregroundStyle(.secondary)
                        }
                        HStack {
                            Text(value.quote.book)
                            Spacer()
                            if let date = value.quote.updated.flatMap(sportsDate) { Text("Recorded " + date.formatted(.dateTime.month().day().hour().minute())) }
                        }.font(.caption2).foregroundStyle(.secondary)
                        builderButton(value.quote)
                    }.padding(14).background(navySurface.opacity(0.7), in: RoundedRectangle(cornerRadius: 18))
                }
            }
        }.accessibilityIdentifier("simulation-odds")
    }
    private func builderButton(_ quote: Market) -> some View {
        let added = builderStore.legs.contains { $0.id == quote.id }
        return VStack(alignment: .leading, spacing: 6) {
            Button {
                if added {
                    builderStore.legs.removeAll { $0.id == quote.id }
                    builderNotices[quote.id] = nil
                } else if let reason = SimulationOdds.builderRejection(for: quote, event: event, existing: builderStore.legs, now: Date()) {
                    builderNotices[quote.id] = reason
                } else {
                    builderStore.add(quote)
                    builderNotices[quote.id] = nil
                }
            } label: {
                Label(added ? "Added to Builder" : "Add to Builder", systemImage: added ? "checkmark" : "plus")
                    .font(.subheadline).frame(maxWidth: .infinity, minHeight: 32)
            }.plonnGlassButton(prominent: added)
                .accessibilityIdentifier("simulation-add-to-builder-" + quote.id)
                .accessibilityHint(added ? "Remove this pick from Builder" : "Add \(quote.name), \(quote.label), \(odds(quote.price)) at \(quote.book) to Builder")
            if let notice = builderNotices[quote.id] { Text(notice).font(.caption).foregroundStyle(.secondary) }
        }
    }
    private var distribution: some View {
        return VStack(alignment: .leading, spacing: 16) {
            HStack {
                summary(result.away.team, result.scores.awaySummary, result.scores.awayAhead)
                Spacer()
                summary(result.home.team, result.scores.homeSummary, result.scores.homeAhead)
            }
            SimulationOutcomeChart(margins: result.scores.scores.map(\.margin), away: result.away.team.abbreviation, home: result.home.team.abbreviation, unit: result.scores.unit)
            HStack {
                Text("Combined · \(result.scores.totalSummary.average, specifier: "%.1f") \(result.scores.unit)")
                Spacer()
                if result.scores.level > 0 { Text("Level · \(result.scores.level * 100, specifier: "%.1f")%") }
            }.font(.caption).foregroundStyle(.secondary)
            Text("Scenario shares, not calibrated win probabilities.").font(.caption).foregroundStyle(.secondary)
        }.padding(16).background(navySurface.opacity(0.7), in: RoundedRectangle(cornerRadius: 18))
    }
    private func summary(_ team: InterestTeam, _ value: MatchupSimulation.Summary, _ ahead: Double) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(team.abbreviation).font(.headline).foregroundStyle(brandAccent)
            Text("\(value.average, specifier: "%.1f")").font(.title.bold())
            Text("80% range \(value.low)–\(value.high)").font(.caption)
            Text("Ahead in \(ahead * 100, specifier: "%.1f")%").font(.caption).foregroundStyle(.secondary)
        }
    }
    private var method: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Current roster members with verified recent appearances supply player rates. Reported-out and inactive players are excluded. Questionable players remain in the assumed lineup; starters and rotations are inferred, not confirmed.")
            Text("Sport-specific rules generate plays and update a shared score and player-stat ledger. These experimental frequencies are not calibrated forecasts. No overtime is assigned. Basketball simplifies rotations and rebounds; football omits sacks, penalties and returns; baseball omits errors, steals and sacrifices; hockey omits power plays and penalties. Unmodeled stats show a dash.")
            if let forecast = result.current?.forecast {
                Text("Recent scoring and opponent points allowed shape the score distribution. Older games receive less weight; small samples shrink toward broad sport priors. Player scoring rates include uncertainty. Opponent strength here is not adjusted for schedule difficulty.")
                Text("Complete plays and box scores are reweighted together; no score is edited. Effective path support: \(Int(forecast.effectivePaths.rounded())) of \(result.trials.count)." + (forecast.limited ? " Available paths could not fully support the requested adjustment." : ""))
                if let market = forecast.market { Text("Market prior: " + market.book + " · " + market.recorded.formatted(.dateTime.month().day().hour().minute()) + ". Same-book paired prices, with margin removed from moneylines. Full-game odds are a soft prior for regulation; tied paths remain tied. This is not an independent edge.") }
            }
            Text("The displayed replay represents the most frequent simulated outcome; its exact score and play sequence have no assigned probability.")
            Text("The movement is illustrative, not tracking data. Availability is checked before a run, every 30 seconds while visible and on return. Changes pause the run. Missing injury coverage is flagged; no report does not prove a player is healthy.")
            Text("\(result.away.team.abbreviation): \(result.away.boxes.count) recent games · \(result.home.team.abbreviation): \(result.home.boxes.count). 5,000 simulations do not create 5,000 independent observations.")
            if result.away.partial || result.home.partial { Text("Some historical games could not be verified and were excluded.").foregroundStyle(.orange) }
            if hasStarted { Text("Observed plays, scores and player stats are fixed; only the remaining regulation game is simulated. The feed is checked every 30 seconds while this screen is visible. Corrections replace the previous snapshot. Baseball updates between plate appearances; current pitch counts, football timeouts and hockey power-play states are not modeled. Overtime and extra innings pause projections.") }
            Text("Sources: ESPN live play-by-play, rosters, injury reports and completed box scores. Odds use existing recorded quotes and their timestamps.")
        }
    }
    private var availabilityStatus: some View {
        VStack(alignment: .leading, spacing: 8) {
            if observed?.phase != .final && (watch?.changed == true || watch?.failed == true || (watch?.usable(at: Date()) != true && !checking)) {
                Text(watch?.changed == true ? "Roster or injury update · run again" : "Availability needs a fresh check").font(.subheadline).foregroundStyle(.orange)
                Button("Refresh simulation") { playing = false; rerun() }.plonnGlassButton(prominent: true)
            }
            DisclosureGroup(checking ? "Checking rosters & injuries…" : watch?.latest.contains(where: { !$0.injuriesVerified }) == true ? "Injury coverage incomplete" : "Rosters & injuries") {
                ForEach(Array((watch?.latest ?? []).enumerated()), id: \.offset) { side, value in
                    VStack(alignment: .leading, spacing: 6) {
                        Text((side == 0 ? result.away.team.abbreviation : result.home.team.abbreviation) + " · checked " + value.checked.formatted(.dateTime.hour().minute())).font(.caption.bold())
                        if !value.injuriesVerified { Text("Injury coverage incomplete").foregroundStyle(.orange) }
                        ForEach(value.observations) { item in
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(item.name)
                                    if let date = item.updated { Text("Reported " + date.formatted(.dateTime.month(.abbreviated).day())).font(.caption2).foregroundStyle(.secondary) }
                                }
                                Spacer()
                                Text(item.status + (item.excluded ? " · excluded" : "")).foregroundStyle(item.excluded ? .red : .orange)
                            }
                        }
                        if let team = result.current?.teams[side], !team.missingHistory.isEmpty {
                            Text("No recent team history: " + team.missingHistory.map(\.name).joined(separator: ", ")).foregroundStyle(.secondary)
                        }
                    }.font(.caption).padding(.top, 8)
                }
            }.font(.caption).tint(brandAccent)
        }
    }
    private var liveStatus: some View {
        Group {
            if hasStarted {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label(observed?.phase == .final ? "Final · observed" : "Live game", systemImage: "dot.radiowaves.left.and.right").font(.caption.bold()).foregroundStyle(.green)
                        Spacer()
                        if updatingLive { ProgressView().controlSize(.small) }
                        else { Button { Task { await refreshAvailability(); await refreshLive() } } label: { Image(systemName: "arrow.clockwise") }.plonnGlassButton().accessibilityLabel("Refresh live game") }
                    }
                    if let live = observed {
                        Text("\(result.away.team.abbreviation) \(live.scores[0])  –  \(live.scores[1]) \(result.home.team.abbreviation)").font(.title3.bold()).monospacedDigit()
                        Text(live.status).font(.caption).foregroundStyle(.secondary)
                        if let play = live.events.last { Text(play.text).font(.caption).lineLimit(3) }
                        Text("Checked " + live.checked.formatted(.dateTime.hour().minute().second())).font(.caption2).foregroundStyle(.secondary)
                    }
                    if !liveMessage.isEmpty { Text(liveMessage).font(.caption).foregroundStyle(.orange) }
                }.padding(14).frame(maxWidth: .infinity, alignment: .leading).background(navySurface, in: RoundedRectangle(cornerRadius: 18)).accessibilityIdentifier("simulation-live-status")
            }
        }
    }
    @MainActor private func refreshLive() async {
        guard let event, !updatingLive else { return }
        updatingLive = true
        defer { updatingLive = false }
        do {
            let snapshot = try await SimulationLiveSource().fetch(event)
            try Task.checkCancellation()
            if let previous = observed, !snapshot.supersedes(previous) {
                liveMessage = "The feed is catching up. Forecast paused."; playing = false; return
            }
            if observed?.fingerprint != snapshot.fingerprint { lastLiveProgress = Date() }
            liveSnapshot = snapshot
            if snapshot.phase == .live && Date().timeIntervalSince(lastLiveProgress) > 300 {
                liveMessage = "No new game state for five minutes. Forecast paused until play updates."; playing = false; return
            }
            if let issue = snapshot.forecastIssue { liveMessage = issue; playing = false; comparisons = []; return }
            guard let availability = watch?.latest, watch?.failed != true else { liveMessage = "Waiting for a roster refresh."; playing = false; return }
            if snapshot.fingerprint == result.current?.live?.fingerprint && watch?.changed != true {
                liveMessage = ""; return
            }
            playing = false; comparisons = []
            let prior = result
            let worker = Task.detached(priority: .userInitiated) {
                try PlayerSimulation.currentRun(away: prior.away, home: prior.home, availability: availability, now: Date(), seed: prior.scores.seed, live: snapshot, market: prior.current?.forecast?.market)
            }
            let next = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
            try Task.checkCancellation()
            updatedResult = next; watch = .init(original: availability, latest: availability); liveMessage = ""
            chooseRepresentative(); loadReplay()
        } catch {
            if !Task.isCancelled { liveMessage = "Live state couldn’t be verified. Retrying automatically."; playing = false; comparisons = [] }
        }
    }
    @MainActor private func refreshAvailability() async {
        checking = true
        defer { checking = false }
        if watch == nil, let current = result.current { let original = current.teams.map(\.availability); watch = .init(original: original, latest: original) }
        do {
            async let away = SimulationAvailabilitySource().load(result.away.team)
            async let home = SimulationAvailabilitySource().load(result.home.team)
            let values = try await [away, home]
            try Task.checkCancellation()
            watch?.update(values)
            if !usable { playing = false; comparisons = [] }
        } catch {
            if !Task.isCancelled { watch?.failed = true; playing = false; comparisons = [] }
        }
    }
    private func loadReplay() {
        playing = false; progress = 0; cursor = 0
        do { replay = try result.replay(trial); frame = replay?.through(0); playbackError = "" }
        catch { replay = nil; frame = nil; playbackError = "This play sequence couldn’t be generated. Run the simulation again." }
    }
    private func playerDetail(_ line: PlayerSimulation.Line) -> some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(history.team.name).font(.subheadline).foregroundStyle(.secondary)
                    ForEach(group.stats) { stat in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(stat.title).font(.headline)
                            if let range = result.range(homeSide: homeSide, lineID: line.id, stat: stat.id) {
                                HStack {
                                    Text(stat.display(range.average, average: true)).font(.title2.bold()).foregroundStyle(brandAccent)
                                    Spacer()
                                    Text("\(stat.display(range.low))–\(stat.display(range.high))").font(.title3.monospacedDigit())
                                }
                                Text("Average · middle 80% · \(range.appearances) recorded appearances").font(.caption).foregroundStyle(.secondary)
                            } else { Text(result.rangeUnavailableReason(homeSide: homeSide, lineID: line.id, stat: stat.id)).font(.caption).foregroundStyle(.secondary) }
                        }.padding(14).background(navySurface, in: RoundedRectangle(cornerRadius: 16))
                    }
                    Text("Simulated with an assumed role from recent appearances. Starting lineup is unconfirmed.").font(.caption).foregroundStyle(.secondary)
                }.padding(16)
            }.background(PLONNBrandBackdrop()).navigationTitle(line.name).navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { detail = nil } } }
        }.preferredColorScheme(.dark)
    }
    private func chooseRepresentative() {
        representative = MatchupSimulation.representative(in: result.scores.scores)
        trial = representative?.index ?? 0
    }
}

/// Shared by the current-game simulator and bundled NBA replay.
struct SimulationOutcomeChart: View {
    let margins: [Int]
    let away: String
    let home: String
    let unit: String
    var body: some View {
        let histogram = MatchupSimulation.MarginHistogram(margins)
        let typical = abs(histogram.median).formatted(.number.precision(.fractionLength(0...1)))
        let label = histogram.median == 0 ? "Typically level" : "Typical · \(histogram.median < 0 ? away : home) by \(typical)"
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(label).font(.subheadline.weight(.medium))
                Spacer()
                Text("\(margins.count.formatted()) runs").font(.caption2).foregroundStyle(.secondary)
            }
            VStack(spacing: 4) {
                plot(histogram)
                HStack {
                    Text(String(histogram.extent))
                    Spacer()
                    Text("Level")
                    Spacer()
                    Text(String(histogram.extent))
                }.font(.caption2).foregroundStyle(.secondary)
            }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Simulated score difference. \(label) \(unit). \(margins.count) runs. Taller bars mean more simulations.")
            HStack {
                Text("← \(away) ahead")
                Spacer()
                Text("\(home) ahead →")
            }.font(.caption2).foregroundStyle(.secondary)
        }.accessibilityIdentifier("simulation-outcomes-chart")
    }
    private func bar(_ bin: MatchupSimulation.MarginHistogram.Bin, width: Int) -> some ChartContent {
        let halfWidth = Double(width) * 0.43
        let lower = Double(bin.center) - halfWidth
        let upper = Double(bin.center) + halfWidth
        return RectangleMark(xStart: .value("From", lower), xEnd: .value("To", upper), yStart: .value("Base", 0), yEnd: .value("Runs", bin.count))
            .foregroundStyle(LinearGradient(colors: [PLONNPalette.ink.opacity(0.85), PLONNPalette.indigo.opacity(0.32)], startPoint: .top, endPoint: .bottom))
            .cornerRadius(3)
    }
    private func plot(_ histogram: MatchupSimulation.MarginHistogram) -> some View {
        Chart {
            ForEach(histogram.bins) { bin in bar(bin, width: histogram.width) }
            RuleMark(x: .value("Level", 0))
                .foregroundStyle(.white.opacity(0.18)).lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 4]))
            RuleMark(x: .value("Typical", histogram.median))
                .foregroundStyle(PLONNPalette.highlight.opacity(0.9)).lineStyle(StrokeStyle(lineWidth: 1.5))
        }
        .chartXScale(domain: Double(-histogram.extent)...Double(histogram.extent))
        .chartXAxis(.hidden)
        .chartYAxis(.hidden).frame(height: 138)
    }

}

/// Illustrative movement from the same play ledger that updates the box score.
private struct SimulationPlaySurface: View {
    let sport: InterestSport
    let teams: [SportPlaySimulation.Team]
    let play: SportPlaySimulation.Play?
    let previous: SportPlaySimulation.Play?
    let progress: Double
    let reduceMotion: Bool
    var body: some View {
        Canvas { context, size in
            let w = size.width, h = size.height
            func point(_ x: Double, _ y: Double) -> CGPoint { .init(x: x * w, y: y * h) }
            let rect = CGRect(x: 2, y: 2, width: w - 4, height: h - 4)
            let boundary = Path(roundedRect: rect, cornerRadius: sport == .nhl ? 35 : 8)
            context.fill(boundary, with: .linearGradient(Gradient(colors: [PLONNPalette.indigo.opacity(0.17), PLONNPalette.fuchsia.opacity(0.08)]), startPoint: .zero, endPoint: point(1, 1)))
            var markings = boundary
            if sport == .mlb {
                markings.move(to: point(0.5, 0.86)); markings.addLine(to: point(0.72, 0.52)); markings.addLine(to: point(0.5, 0.18)); markings.addLine(to: point(0.28, 0.52)); markings.closeSubpath()
                markings.move(to: point(0.5, 0.86)); markings.addLine(to: point(0.1, 0.12))
                markings.move(to: point(0.5, 0.86)); markings.addLine(to: point(0.9, 0.12))
            } else {
                markings.move(to: point(0.5, 0.02)); markings.addLine(to: point(0.5, 0.98))
                if sport == .nfl {
                    for step in 1..<10 { let x = Double(step) / 10; markings.move(to: point(x, 0.03)); markings.addLine(to: point(x, 0.97)) }
                } else {
                    markings.addEllipse(in: CGRect(x: w / 2 - 17, y: h / 2 - 17, width: 34, height: 34))
                    for x in [0.035, 0.865] { markings.addRect(CGRect(x: w * x, y: h * 0.28, width: w * 0.1, height: h * 0.44)) }
                }
            }
            if sport == .nba {
                for x in [0.074, 0.926] {
                    var arc = Path()
                    arc.addArc(center: point(x, 0.5), radius: w * 0.237,
                               startAngle: .degrees(x < 0.5 ? -90 : 90), endAngle: .degrees(x < 0.5 ? 90 : 270), clockwise: false)
                    markings.addPath(arc)
                    context.stroke(Path(ellipseIn: CGRect(x: w * x - 3, y: h * 0.5 - 3, width: 6, height: 6)), with: .color(.orange), lineWidth: 1.5)
                }
                for x in [0.22, 0.78] {
                    markings.addEllipse(in: CGRect(x: w * x - w * 0.064, y: h * 0.5 - w * 0.064, width: w * 0.128, height: w * 0.128))
                }
            } else if sport == .nhl {
                for x in [0.33, 0.67] { markings.move(to: point(x, 0.02)); markings.addLine(to: point(x, 0.98)) }
                for x in [0.22, 0.78] {
                    for y in [0.27, 0.73] { markings.addEllipse(in: CGRect(x: w * x - 14, y: h * y - 14, width: 28, height: 28)) }
                }
            } else if sport == .nfl {
                for step in 1..<10 {
                    let label = String(min(step, 10 - step) * 10)
                    context.draw(Text(label).font(.system(size: 9)).foregroundStyle(.white.opacity(0.4)), at: point(Double(step) / 10, 0.12))
                    context.draw(Text(label).font(.system(size: 9)).foregroundStyle(.white.opacity(0.4)), at: point(Double(step) / 10, 0.88))
                }
            } else if sport == .mlb {
                for base in [point(0.5, 0.86), point(0.72, 0.52), point(0.5, 0.18), point(0.28, 0.52)] {
                    context.fill(Path(CGRect(x: base.x - 3, y: base.y - 3, width: 6, height: 6)), with: .color(.white.opacity(0.8)))
                }
            }
            context.draw(Text("PLONN").font(.system(size: 17, weight: .heavy, design: .rounded)).foregroundStyle(.white.opacity(0.09)), at: point(0.5, 0.5))
            context.stroke(markings, with: .color(.white.opacity(0.25)), lineWidth: 1)
            let scene = SportReplayMotion.scene(sport: sport, teams: teams, play: play, previous: previous, progress: progress, reduceMotion: reduceMotion)
            for marker in scene.markers {
                let center = point(marker.point.x, marker.point.y)
                let color = marker.team == 0 ? PLONNPalette.ink : PLONNPalette.fuchsia
                let radius: CGFloat = marker.playerID == nil ? 5 : max(7, w * 0.021)
                let circle = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
                context.fill(circle, with: .color(color.opacity(marker.playerID == nil ? 0.32 : 0.9)))
                context.stroke(circle, with: .color(.white.opacity(marker.emphasized ? 0.95 : 0.5)), lineWidth: marker.emphasized ? 1.5 : 0.6)
                if let id = marker.playerID, let player = teams[marker.team].profiles.first(where: { $0.id == id })?.player {
                    let initials = player.name.split(separator: " ").prefix(2).compactMap { $0.first }.map(String.init).joined()
                    context.draw(Text(initials).font(.system(size: 7, weight: .bold)).foregroundStyle(.white), at: center)
                    if marker.emphasized {
                        let name = player.name.split(separator: " ").last.map(String.init) ?? player.name
                        let label = CGPoint(x: min(w - 28, max(28, center.x)), y: min(h - 7, center.y + 16))
                        context.draw(Text(name).font(.system(size: 9, weight: .semibold)).foregroundStyle(.white), at: label)
                    }
                }
            }
            if let ball = scene.ball {
                let shadow = point(ball.x, ball.y), center = point(ball.x, ball.y - scene.ballHeight)
                let radius: CGFloat = sport == .nhl ? 3 : 4
                context.fill(Path(ellipseIn: CGRect(x: shadow.x - radius, y: shadow.y - radius / 2, width: radius * 2, height: radius)), with: .color(.black.opacity(0.35)))
                let path = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
                context.fill(path, with: .color(sport == .nba ? .orange : .white))
                context.stroke(path, with: .color(.black.opacity(0.5)), lineWidth: 0.7)
            }
        }.clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
