import SwiftUI
import Charts

/// The offline fixture is deliberately separate from current roster/injury simulations.
struct NBAReplayView: View {
    let active: Bool
    @State private var simulation: BasketballSimulation?
    @State private var failed = false
    @State private var loadAttempt = 0
    var body: some View {
        Group {
            if let simulation, let scenario = simulation.representativeScenario { NBAReplayContent(simulation: simulation, scenario: scenario, active: active) }
            else if failed {
                ContentUnavailableView {
                    Label("Replay unavailable", systemImage: "basketball")
                } description: {
                    Text("The bundled example could not be loaded.")
                } actions: {
                    Button("Try again") { failed = false; loadAttempt += 1 }.plonnGlassButton()
                }
            } else { PLONNLoader("Loading NBA replay…") }
        }
        .task(id: loadAttempt) {
            guard simulation == nil else { return }
            do {
                let loaded = try await Task.detached(priority: .userInitiated) { try BasketballSimulation.bundled() }.value
                guard loaded.representativeScenario != nil else { throw BasketballSimulation.LoadError.invalidReplay }
                simulation = loaded
            }
            catch { failed = true }
        }
    }
}

private struct NBAReplayContent: View {
    let simulation: BasketballSimulation
    let scenario: MatchupSimulation.RepresentativeScenario
    let active: Bool
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var position = 0.0
    @State private var playing = false
    @State private var speed = 4.0
    @State private var selectedTeam = 0
    @State private var profileID = ""
    @State private var section = "Players"
    @State private var showingMethod = false
    @State private var selectedPlayer: BasketballSimulation.Player?
    private var replay: BasketballSimulation.Replay { simulation.replays[scenario.index] }
    private var frame: BasketballSimulation.Frame { replay.frame(at: position) }
    private var canPlay: Bool { playing && active && !showingMethod && selectedPlayer == nil && scenePhase == .active }
    private var h2hEnabled: Bool { (Bundle.main.object(forInfoDictionaryKey: "PLONNSimulatorH2HDetailsEnabled") as? Bool) == true }
    private var sections: [String] { ["Players", "Shots", "Plays", "Outcomes"] + (h2hEnabled && simulation.meta.headToHead != nil ? ["H2H"] : []) }
    var body: some View {
        ScrollViewReader { scroll in
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                VStack(spacing: 12) {
                    scenarioSummary
                    scoreboard
                    SimulationCourt(simulation: simulation, replay: replay, frame: frame, reduceMotion: reduceMotion)
                        .aspectRatio(1.78, contentMode: .fit).clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(alignment: .top) {
                            if replay.events[frame.index].zone == "unlocated_two" {
                                Text("Unlocated 2 · No shot position")
                                    .font(.caption2).padding(6).background(.black.opacity(0.8), in: Capsule()).padding(6)
                            }
                        }
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("Illustrative court. " + simulation.description(replay.events[frame.index], completed: frame.finished))
                    HStack(alignment: .top) {
                        Text(frame.finished ? "Replay complete" : simulation.description(replay.events[frame.index], completed: false))
                            .font(.subheadline).frame(maxWidth: .infinity, alignment: .leading)
                        Text("\(frame.completed)/\(replay.events.count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }.accessibilityIdentifier("simulation-current-play")
                    controls
                }.padding(12).background(navySurface, in: RoundedRectangle(cornerRadius: 22))
                Text("Illustrative movement · Historical participants")
                    .font(.caption2).foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(sections, id: \.self) { item in
                            Button(item) {
                                section = item
                                withAnimation(reduceMotion ? nil : .easeOut(duration: 0.22)) {
                                    scroll.scrollTo("simulation-result-controls", anchor: .top)
                                }
                            }.plonnGlassButton(prominent: section == item)
                                .accessibilityAddTraits(section == item ? .isSelected : [])
                                .accessibilityIdentifier("simulation-section-\(item.lowercased())")
                        }
                    }.padding(.vertical, 3)
                }.id("simulation-result-controls")
                Group {
                    switch section {
                    case "Shots": profiles
                    case "Plays": playLog
                    case "Outcomes": outcomes
                    case "H2H": headToHead
                    default: boxScore
                    }
                }.padding(14).background(navySurface, in: RoundedRectangle(cornerRadius: 20))
            }.padding(16)
        }
        }
        .sheet(isPresented: $showingMethod) { methodSheet }
        .sheet(item: $selectedPlayer) { player in playerSheet(player) }
        .onChange(of: selectedTeam) { _, _ in profileID = "" }
        .onChange(of: active) { _, value in if !value { playing = false } }
        .onChange(of: scenePhase) { _, value in if value != .active { playing = false } }
        .onChange(of: showingMethod) { _, value in if value { playing = false } }
        .onChange(of: selectedPlayer?.id) { _, value in if value != nil { playing = false } }
        .onDisappear { playing = false }
        .task(id: canPlay) {
            guard canPlay else { return }
            var last = Date()
            while !Task.isCancelled, canPlay {
                do { try await Task.sleep(for: .milliseconds(50)) } catch { return }
                guard !Task.isCancelled else { return }
                let current = Date(), delta = min(0.15, max(0, current.timeIntervalSince(last)))
                last = current
                position = min(Double(replay.events.count), position + delta * speed / 3.2)
                if position >= Double(replay.events.count) { playing = false; return }
            }
        }
    }
    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("\(simulation.data.teams[0].abbrev) at \(simulation.data.teams[1].abbrev)").font(.title2.bold())
                Text("\(simulation.meta.openingNight.targetDateET) · Offline example").font(.caption).foregroundStyle(.secondary)
                Text(simulation.meta.modelVersion).font(.caption.weight(.medium)).foregroundStyle(PLONNPalette.highlight)
            }
            Spacer(minLength: 4)
            Button { showingMethod = true } label: { Image(systemName: "info.circle").frame(width: 26, height: 28) }
                .plonnGlassButton().accessibilityLabel("Simulation details and limitations")
                .accessibilityIdentifier("simulation-details")
        }
    }
    private var scenarioSummary: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(scenario.tiedForTop ? "Even outcome split" : "Representative scenario").font(.caption).foregroundStyle(.secondary)
            Spacer(minLength: 6)
            if !scenario.tiedForTop {
                Text((scenario.outcome == .level ? "Level" : simulation.data.teams[scenario.outcome == .away ? 0 : 1].abbrev + " wins") + " " + scenario.share.formatted(.percent.precision(.fractionLength(1))))
                    .font(.caption.weight(.semibold)).foregroundStyle(PLONNPalette.highlight)
            }
        }.accessibilityIdentifier("simulation-selected-scenario")
    }
    private var scoreboard: some View {
        HStack {
            teamScore(0)
            VStack(spacing: 3) {
                Text("SIMULATED").font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                Text(frame.clock).font(.title3.monospacedDigit().bold()).accessibilityIdentifier("simulation-clock")
                if !frame.finished { Text(frame.periodLabel).font(.caption2).foregroundStyle(.secondary) }
            }.frame(maxWidth: .infinity)
            teamScore(1)
        }.accessibilityElement(children: .combine)
    }
    private func teamScore(_ index: Int) -> some View {
        VStack(spacing: 3) {
            Text(simulation.data.teams[index].abbrev).font(.caption.bold()).foregroundStyle(teamColor(index))
            Text("\(frame.score[index])").font(.system(size: 30, weight: .semibold, design: .rounded)).monospacedDigit()
        }.frame(maxWidth: .infinity).accessibilityLabel("\(simulation.data.teams[index].name), \(frame.score[index]) simulated points")
    }
    private var controls: some View {
        VStack(spacing: 6) {
            Slider(value: $position, in: 0...Double(replay.events.count), onEditingChanged: { if $0 { playing = false } })
                .accessibilityLabel("Replay progress").accessibilityValue("\(frame.completed) of \(replay.events.count) plays")
                .accessibilityIdentifier("simulation-timeline")
            HStack(spacing: 8) {
                Button { playing = false; position = 0 } label: { Image(systemName: "backward.end.fill").frame(minWidth: 20, minHeight: 28) }
                    .plonnGlassButton().accessibilityLabel("Restart simulation")
                Button {
                    if frame.finished { position = 0 }; playing.toggle()
                } label: { Label(playing ? "Pause" : "Play", systemImage: playing ? "pause.fill" : "play.fill").frame(minWidth: 58, minHeight: 28) }
                    .plonnGlassButton(prominent: true).accessibilityIdentifier("simulation-play")
                Button { playing = false; position = min(Double(replay.events.count), floor(position) + 1) } label: {
                    Image(systemName: "forward.end.fill").frame(minWidth: 20, minHeight: 28)
                }.plonnGlassButton().disabled(frame.finished).accessibilityLabel("Next play").accessibilityIdentifier("simulation-next")
                Spacer(minLength: 0)
                Menu {
                    Picker("Playback speed", selection: $speed) {
                        ForEach([1.0, 2, 4, 8], id: \.self) { Text("\(Int($0))×").tag($0) }
                    }
                } label: { Text("\(Int(speed))×").monospacedDigit().frame(minWidth: 20, minHeight: 28) }
                    .plonnGlassButton().accessibilityLabel("Playback speed, \(Int(speed)) times")
            }.plonnGlassGroup()
        }
    }
    private var teamPicker: some View {
        Picker("Team", selection: $selectedTeam) {
            ForEach(0..<2, id: \.self) { Text(simulation.data.teams[$0].abbrev).tag($0) }
        }.pickerStyle(.segmented)
    }
    private var boxScore: some View {
        let counts = replay.counts(completed: frame.completed)
        return VStack(spacing: 10) {
            teamPicker
            HStack(spacing: 4) {
                Text("Player").frame(maxWidth: .infinity, alignment: .leading)
                ForEach(["PTS", "REB", "AST", "MIN"], id: \.self) { stat($0) }
            }.font(.caption2).foregroundStyle(.secondary)
            ForEach(simulation.data.teams[selectedTeam].players) { player in
                let row = counts[player.id, default: .init()]
                Button { selectedPlayer = player } label: {
                    HStack(spacing: 4) {
                        Text(player.shortName).font(.subheadline).frame(maxWidth: .infinity, alignment: .leading)
                            .lineLimit(2)
                        stat("\(row.points)").foregroundStyle(PLONNPalette.highlight)
                        stat("\(row.rebounds)"); stat("\(row.assists)"); stat(String(format: "%.1f", row.minutes))
                    }.font(.caption.monospacedDigit()).frame(minHeight: 40).contentShape(Rectangle())
                }.buttonStyle(.plain)
                    .accessibilityLabel("\(player.name), \(row.points) points, \(row.rebounds) rebounds, \(row.assists) assists, \(row.minutes.formatted(.number.precision(.fractionLength(1)))) minutes. Full stats")
            }
        }
    }
    private func stat(_ value: String) -> some View { Text(value).frame(width: 34, alignment: .trailing) }
    private func playerSheet(_ player: BasketballSimulation.Player) -> some View {
        let row = replay.counts(completed: frame.completed)[player.id, default: .init()]
        return NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Simulated · \(frame.periodLabel) \(frame.clock)").font(.caption).foregroundStyle(.secondary)
                    HStack { metric("PTS", "\(row.points)"); metric("REB", "\(row.rebounds)"); metric("AST", "\(row.assists)") }
                    HStack { metric("FG", "\(row.made)/\(row.attempts)"); metric("3PT", "\(row.threesMade)/\(row.threesAttempted)"); metric("FT", "\(row.freeThrowsMade)/\(row.freeThrowsAttempted)") }
                    HStack { metric("MIN", String(format: "%.1f", row.minutes)); metric("TOV", "\(row.turnovers)") }
                    Divider()
                    shotProfile(player)
                }.padding(20)
            }.navigationTitle(player.name).navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { selectedPlayer = nil } } }
        }.presentationDetents([.medium, .large]).presentationDragIndicator(.visible)
    }
    private var profiles: some View {
        let players = simulation.data.teams[selectedTeam].players
        let player = players.first(where: { $0.id == profileID }) ?? players[0]
        return VStack(alignment: .leading, spacing: 14) {
            teamPicker
            Picker("Player shot profile", selection: Binding(get: { player.id }, set: { profileID = $0 })) {
                ForEach(players) { Text($0.name).tag($0.id) }
            }.pickerStyle(.menu).tint(PLONNPalette.highlight)
            shotProfile(player)
        }
    }
    private func shotProfile(_ player: BasketballSimulation.Player) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { metric("Projected MIN", String(format: "%.1f", player.projectedMinutes)); metric("Earlier games", "\(player.historicalGames)") }
            HStack { Text("Shot share"); Spacer(); Text("Base make %") }.font(.caption).foregroundStyle(.secondary)
            ForEach(0..<4, id: \.self) { zone in
                VStack(spacing: 5) {
                    HStack {
                        Text(BasketballSimulation.zoneLabels[zone])
                        Spacer()
                        Text(player.zones[zone], format: .percent.precision(.fractionLength(0)))
                        Text(player.accuracy[zone], format: .percent.precision(.fractionLength(0))).frame(width: 65, alignment: .trailing).foregroundStyle(.secondary)
                    }.font(.subheadline.monospacedDigit())
                    GeometryReader { proxy in
                        Capsule().fill(.white.opacity(0.07))
                        Capsule().fill(PLONNPalette.highlight.opacity(zone == 3 ? 0.4 : 0.85)).frame(width: proxy.size.width * player.zones[zone])
                    }.frame(height: 5)
                }
            }
            Text("Smoothed shot shares and base make probabilities, before opponent adjustments. Unlocated twos have no measured position.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
    private var playLog: some View {
        LazyVStack(alignment: .leading, spacing: 12) {
            if frame.completed == 0 { Text("Play or step forward to see completed plays.").font(.subheadline).foregroundStyle(.secondary) }
            ForEach(Array(replay.events.prefix(frame.completed).enumerated()).reversed(), id: \.offset) { _, event in
                HStack(alignment: .top, spacing: 10) {
                    Text(event.periodLabel + "\n" + String(format: "%d:%02d", event.clock / 60, event.clock % 60))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                    Text(simulation.description(event)).font(.subheadline).frame(maxWidth: .infinity, alignment: .leading)
                    Text("\(event.score[0])–\(event.score[1])").font(.caption.monospacedDigit()).foregroundStyle(PLONNPalette.highlight)
                }
                Divider()
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private var outcomes: some View {
        let away = simulation.data.teams[0].abbrev, home = simulation.data.teams[1].abbrev
        return VStack(alignment: .leading, spacing: 16) {
            Text("\(simulation.result.runs.formatted()) simulated games").font(.headline)
            HStack {
                metric("\(away) wins", simulation.result.win_probability.formatted(.percent.precision(.fractionLength(1))))
                metric("\(home) wins", (1 - simulation.result.win_probability).formatted(.percent.precision(.fractionLength(1))))
            }
            SimulationOutcomeChart(margins: simulation.result.score_pairs.map { $0[1] - $0[0] }, away: away, home: home, unit: "points")
            HStack {
                metric("\(away) avg", String(format: "%.1f", simulation.result.mean_score[0]))
                metric("\(home) avg", String(format: "%.1f", simulation.result.mean_score[1]))
            }
            metric("Middle 80% of simulated totals", "\(Int(simulation.result.total_p10))–\(Int(simulation.result.total_p90)) pts")
            Text("This range describes the simulation runs. It is not a validated prediction interval.").font(.caption).foregroundStyle(.secondary)
        }
    }
    @ViewBuilder private var headToHead: some View {
        if let h2h = simulation.meta.headToHead {
            VStack(alignment: .leading, spacing: 14) {
                Text("Earlier meetings").font(.headline)
                HStack { metric("Games used", "\(h2h.usable_meeting_count)"); metric("Age-weighted sample", String(format: "%.1f", h2h.effective_meeting_count)) }
                HStack {
                    ForEach(simulation.data.teams) { team in
                        metric("\(team.abbrev) adjustment", String(format: "%+.2f pts", h2h.adjustment_by_team[team.id] ?? 0))
                    }
                }
                Text("Already included in these simulations. A small team-level adjustment, capped at \(h2h.maximum_absolute_points_per_team.formatted()) points per team—not individual defender analysis.")
                    .font(.caption).foregroundStyle(.secondary)
                DisclosureGroup("Meeting history") {
                    ForEach(h2h.meetings.sorted { $0.date_et > $1.date_et }) { meeting in
                        HStack {
                            Text(meeting.date_et).foregroundStyle(.secondary)
                            Spacer()
                            Text("\(abbrev(meeting.away_id)) \(meeting.actual_away) · \(abbrev(meeting.home_id)) \(meeting.actual_home)")
                        }.font(.caption.monospacedDigit()).padding(.vertical, 5)
                    }
                }.font(.subheadline)
                Text("Older games carry less weight (\(Int(h2h.half_life_days))-day half-life), then shrink toward no adjustment with \(Int(h2h.shrinkage_games)) prior games. Research has not demonstrated an accuracy improvement.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
    private func abbrev(_ id: String) -> String { simulation.data.teams.first { $0.id == id }?.abbrev ?? id }
    private func teamColor(_ index: Int) -> Color { simulation.data.teams[index].replayColor }
    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(label).font(.caption).foregroundStyle(.secondary); Text(value).font(.title3.bold()).monospacedDigit() }
            .frame(maxWidth: .infinity, alignment: .leading)
    }
    private var methodSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(simulation.meta.modelVersion).font(.title2.bold()).foregroundStyle(PLONNPalette.highlight)
                    Text("Bundled historical example · \(simulation.meta.openingNight.targetDateET)").font(.subheadline)
                    Text("Observed final: \(simulation.data.teams[0].abbrev) \(simulation.meta.actualScore[0]) · \(simulation.data.teams[1].abbrev) \(simulation.meta.actualScore[1])")
                        .font(.headline).accessibilityIdentifier("simulation-observed-final")
                    Text("The observed result is separate from all simulated scores.").font(.caption).foregroundStyle(.secondary)
                    Text("Uses observed participants and starters. Player observations end \(simulation.meta.openingNight.lastPlayerObservationET); no current-season games or recent residual calibration were available.")
                    DisclosureGroup("Data limitations") {
                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(simulation.meta.limitations, id: \.self) { Text($0) }
                            Text("Free-throw events represent two attempts; their individual order is unavailable. Minutes accumulate at completed-play boundaries. Steals, blocks and fouls are not supplied.")
                        }.font(.subheadline).foregroundStyle(.secondary).padding(.top, 8)
                    }
                    DisclosureGroup("Method & validation") {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Frozen team baseline: \(simulation.meta.training.games) games from \(simulation.meta.training.season).")
                            Text(simulation.meta.backtest.label)
                            Text("Separate research on 335 games: total-score MAE 14.990 vs 14.967 previously; H2H changed MAE by −0.010. Neither demonstrated an improvement, and that configuration does not validate this opener.")
                            Text("No live simulation API or sportsbook prices are included in this example. The displayed replay represents the most frequent outcome among the saved runs. Its exact play sequence has no assigned probability.")
                        }.font(.subheadline).foregroundStyle(.secondary).padding(.top, 8)
                    }
                }.padding(20)
            }.navigationTitle("About this replay").navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showingMethod = false } } }
        }.presentationDetents([.medium, .large]).presentationDragIndicator(.visible)
    }
}

private extension BasketballSimulation.Team {
    var replayColor: Color {
        let hex = color.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        guard hex.count == 6, let value = UInt32(hex, radix: 16) else { return .purple }
        return Color(red: Double((value >> 16) & 255) / 255, green: Double((value >> 8) & 255) / 255, blue: Double(value & 255) / 255)
    }
}

private struct SimulationCourt:View {
 let simulation:BasketballSimulation
 let replay:BasketballSimulation.Replay
 let frame:BasketballSimulation.Frame
 let reduceMotion:Bool
 private struct Spot {var x:Double;var y:Double}
 private func random(_ text:String)->Double {
  var hash:UInt32=2166136261
  for byte in text.utf8 {hash=(hash ^ UInt32(byte)) &* 16777619}
  return Double(hash)/Double(UInt32.max)
 }
 private func hoop(_ event:BasketballSimulation.Event)->Spot {Spot(x:(event.team == 0) == (event.period<=2) ? 92.6:7.4,y:28)}
 private func spots(_ event:BasketballSimulation.Event,index:Int)->[String:Spot] {
  let basket=hoop(event),sign=basket.x>50 ? 1.0:-1.0
  let formation:[Spot]=[.init(x:22,y:-14.5),.init(x:28,y:12.5),.init(x:16,y:-8),.init(x:9,y:11.5),.init(x:31,y:0)]
  var result:[String:Spot]=[:]
  for (k,id) in event.lineups[event.team].enumerated() {
   let base=formation[k]
   result[id]=Spot(x:basket.x-sign*(base.x+(random("\(index)x"+id)-0.5)*3.4),y:28+base.y+(random("\(index)y"+id)-0.5)*2.4)
  }
  let distance=event.kind == "turnover" ? 25:event.kind == "free throws" ? 14.5:event.zone == "three" ? 26:event.zone == "mid" ? 14.5:4.2
  if event.zone != "unlocated_two" { result[event.player]=Spot(x:basket.x-sign*distance,y:28+(event.kind == "free throws" ? 0:(random("\(index)shot")-0.5)*(event.zone == "three" ? 29.6:8))) }
  for (k,id) in event.lineups[1-event.team].enumerated() {
   let mark=result[event.lineups[event.team][k]]!
   result[id]=Spot(x:min(93.5,max(6.5,mark.x+sign*3.2)),y:mark.y+(k%2 == 0 ? 1.8:-1.8))
  }
  return result
 }
 var body:some View {
  Canvas {context,size in
   func point(_ p:Spot)->CGPoint {CGPoint(x:p.x/100*size.width,y:p.y/56*size.height)}
   func rect(_ x:Double,_ y:Double,_ w:Double,_ h:Double)->CGRect {CGRect(x:x/100*size.width,y:y/56*size.height,width:w/100*size.width,height:h/56*size.height)}
   context.fill(Path(CGRect(origin:.zero,size:size)),with:.linearGradient(Gradient(colors:[Color(red:0.10,green:0.09,blue:0.16),Color(red:0.08,green:0.10,blue:0.15)]),startPoint:.zero,endPoint:CGPoint(x:size.width,y:size.height)))
   var lines=Path()
   lines.addRoundedRect(in:rect(3,3,94,50),cornerSize:CGSize(width:4,height:4))
   lines.move(to:point(.init(x:50,y:3)));lines.addLine(to:point(.init(x:50,y:53)))
   lines.addEllipse(in:rect(43.6,21.6,12.8,12.8))
   for x in [3.0,78.0] {lines.addRect(rect(x,19.5,19,17))}
   for x in [15.6,71.6] {lines.addEllipse(in:rect(x,21.6,12.8,12.8))}
   for x in [7.4,92.6] {
    var arc=Path()
    arc.addArc(center:point(.init(x:x,y:28)),radius:size.width*0.237,startAngle:.degrees(x<50 ? -90:90),endAngle:.degrees(x<50 ? 90:270),clockwise:false)
    lines.addPath(arc)
   }
   context.stroke(lines,with:.color(.white.opacity(0.25)),lineWidth:1)
   context.draw(Text("PLONN").font(.system(size:17,weight:.heavy,design:.rounded)).foregroundStyle(.white.opacity(0.09)),at:point(.init(x:50,y:28)))
   for x in [7.4,92.6] {context.stroke(Path(ellipseIn:rect(x-0.8,27.2,1.6,1.6)),with:.color(.orange),lineWidth:1.5)}
   let players=simulation.players
   let event=replay.events[frame.index]
   let targets=spots(event,index:frame.index)
   let starts=frame.index>0 ? spots(replay.events[frame.index-1],index:frame.index-1):[:]
   let phase=frame.phase
   let t=reduceMotion ? 1:min(1,phase/0.55)
   let eased=t*t*(3-2*t)
   var locations:[String:Spot]=[:]
   for team in 0...1 {
    for id in event.lineups[team] {
     if id == event.player && event.zone == "unlocated_two" { continue }
     let end=targets[id]!,start=starts[id] ?? Spot(x:50+(team == 0 ? -3:3),y:20+random(id)*14)
     let pos=Spot(x:start.x+(end.x-start.x)*eased,y:start.y+(end.y-start.y)*eased)
     locations[id]=pos
     let center=point(pos),radius=max(7,size.width*0.018)
     let circle=Path(ellipseIn:CGRect(x:center.x-radius,y:center.y-radius,width:radius*2,height:radius*2))
     context.fill(circle,with:.color(simulation.data.teams[team].replayColor.opacity(0.85)))
     context.stroke(circle,with:.color(.white.opacity(0.8)),lineWidth:1)
     context.draw(Text(players[id]?.jersey ?? "").font(.system(size:max(7,size.width*0.022),weight:.bold)).foregroundStyle(.white),at:center)
    }
   }
   // Unknown two-point locations have no player position or ball trajectory.
   guard let shooter=locations[event.player] else { return }
   let basket=hoop(event)
   var ball=Spot(x:shooter.x+1.3,y:shooter.y-1),height=0.0
   if event.hasIllustratedShotLocation && !reduceMotion && phase>=0.55 {
    let flight=min(1,(phase-0.55)/0.31)
    ball=Spot(x:shooter.x+(basket.x-shooter.x)*flight,y:shooter.y+(basket.y-shooter.y)*flight)
    height=reduceMotion ? 0:sin(flight*Double.pi)*8.6
    if phase>=0.86,let rebound=event.rebound,let receiver=locations[rebound] {
     let landing=(phase-0.86)/0.14
     ball=Spot(x:basket.x+(receiver.x-basket.x)*landing,y:basket.y+(receiver.y-basket.y)*landing)
    }
   } else if !reduceMotion, let passer=event.assist.flatMap({locations[$0]}) {
    let pass=min(1,max(0,(phase-0.25)/0.23))
    ball=Spot(x:passer.x+(shooter.x-passer.x)*pass,y:passer.y+(shooter.y-passer.y)*pass)
   }
   // Turnovers have no stealing defender ID; do not invent one.
   if reduceMotion { ball=shooter; height=0 }
   let center=point(.init(x:ball.x,y:ball.y-height)),radius=max(3,size.width*0.009)
   context.fill(Path(ellipseIn:CGRect(x:center.x-radius,y:center.y-radius,width:radius*2,height:radius*2)),with:.color(.orange))
  }
 }
}
