import SwiftUI

private enum LabModel: String, CaseIterable, Identifiable {
    case empirical = "Historical frequency"
    case poisson = "Poisson count model"
    case normal = "Normal distribution"
    var id: String { rawValue }
    var detail: String {
        switch self {
        case .empirical: return "How often your supplied observations exceeded the line. Includes a 95% Wilson interval."
        case .poisson: return "Count-event baseline fitted to the sample mean. Assumes independent events and equal mean/variance."
        case .normal: return "Fits a continuous normal distribution to your sample mean and standard deviation."
        }
    }
}
private struct LabResult: Identifiable {
    var id: String { model }
    let model: String
    let probability: Double
    let push: Double
    let detail: String
}

struct StatisticalLabView: View {
    @State private var sport = "NBA"
    @State private var subject = ""
    @State private var stat = "Points"
    @State private var observations = ""
    @State private var line = ""
    @State private var selected: Set<LabModel> = [.empirical, .normal]
    @State private var results: [LabResult] = []
    @State private var error = ""
    @State private var runSummary = ""
    @State private var isExample = false
    private var inputSignature: String { sport + subject + stat + observations + line + selected.map(\.rawValue).sorted().joined() }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HStack { Brand(); Spacer(); Image(systemName: "sparkles").foregroundStyle(charcoalAccent) }
                Text("Compare models").font(.system(.title2, design: .monospaced).bold())
                Panel {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Inputs").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                        Picker("Sport", selection: $sport) { Text("NBA").tag("NBA"); Text("MLB").tag("MLB") }.pickerStyle(.segmented)
                        TextField("Player or team name", text: $subject).textFieldStyle(.roundedBorder)
                        TextField("Stat, e.g. points or strikeouts", text: $stat).textFieldStyle(.roundedBorder)
                        TextField("Over line, e.g. 27.5", text: $line).keyboardType(.decimalPad).textFieldStyle(.roundedBorder).accessibilityIdentifier("lab-line")
                        Text("Past observations").font(.system(.subheadline, design: .monospaced).bold())
                        TextEditor(text: $observations).frame(height: 90).scrollContentBackground(.hidden).padding(8).background(navy).clipShape(RoundedRectangle(cornerRadius: 10)).accessibilityIdentifier("lab-observations")
                        Text("Paste 5–500 nonnegative values separated by commas, spaces or newlines. Use the same stat and units throughout.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                        Button("Load illustrative example") { subject = "Example player"; stat = sport == "NBA" ? "Points" : "Strikeouts"; observations = sport == "NBA" ? "24, 31, 28, 20, 35, 26, 33, 22, 30, 29" : "6, 4, 8, 5, 7, 3, 6, 9, 4, 7"; line = sport == "NBA" ? "27.5" : "5.5"; isExample = true }
                    }
                }
                VStack(alignment: .leading, spacing: 12) {
                    Text("Methods").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    ForEach(LabModel.allCases) { model in
                        Button {
                            if selected.contains(model) { selected.remove(model) } else { selected.insert(model) }
                        } label: {
                            HStack(alignment: .top, spacing: 12) {
                                Image(systemName: selected.contains(model) ? "checkmark.circle.fill" : "circle").foregroundStyle(selected.contains(model) ? Color.white : .secondary)
                                VStack(alignment: .leading, spacing: 5) { Text(model.rawValue).font(.system(.headline, design: .monospaced)).foregroundStyle(.primary); Text(model.detail).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
                                Spacer()
                            }.padding(16).plonnGlassSurface(selected:selected.contains(model),radius:14)
                        }.buttonStyle(.plain).accessibilityLabel(model.rawValue).accessibilityValue(selected.contains(model) ? "Selected" : "Not selected")
                    }
                }
                Button(action: run) { Label("Run comparison", systemImage: "sparkles").font(.system(.headline, design: .monospaced)).frame(maxWidth: .infinity).padding(.vertical, 8) }.plonnGlassButton(prominent:true)
                if !error.isEmpty { Text(error).foregroundStyle(.orange).accessibilityIdentifier("lab-error") }
                if !results.isEmpty {
                    Text("Comparison").font(.system(.title2, design: .monospaced).bold())
                    Text(runSummary).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    ForEach(results) { result in
                        Panel {
                            VStack(alignment: .leading, spacing: 12) {
                                Text(result.model).font(.system(.headline, design: .monospaced))
                                HStack { Metric(title: "Chance over", value: probability(result.probability)); Metric(title: "Fair odds*", value: odds(fair(result.probability / max(0.000001, 1 - result.push)))) }
                                if result.push > 0 { Text("Push probability: " + probability(result.push)).font(.system(.caption, design: .monospaced)) }
                                Text(result.detail).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                            }
                        }
                    }
                    Text("*Fair odds condition on a decisive result when pushes are possible. These are unvalidated statistical baselines fitted to your inputs, not trained PLONN AI predictions. Small samples, opponents, injuries and changes in playing time can materially affect results.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                }
                Panel { VStack(alignment: .leading, spacing: 10) { Label("PLONN trained models", systemImage: "cpu").font(.system(.headline, design: .monospaced)); Text("NBA totals · MLB strikeouts"); Text("These models aren’t available yet.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) } }
            }.padding(20)
        }.background(PLONNBrandBackdrop()).navigationTitle("Models").navigationBarTitleDisplayMode(.inline)
        .onChange(of: inputSignature) { _, _ in results = []; error = "" }
    }
    private func run() {
        results = []; error = ""
        let tokens = observations.components(separatedBy: CharacterSet(charactersIn: ",; \n\t")).filter { !$0.isEmpty }
        let values = tokens.compactMap(Double.init)
        guard values.count == tokens.count, (5...500).contains(values.count), values.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 10000 }) else { error = "Enter 5–500 valid values between 0 and 10,000."; return }
        guard let threshold = Double(line), threshold.isFinite, threshold >= 0, threshold <= 10000 else { error = "Enter a valid line between 0 and 10,000."; return }
        guard !selected.isEmpty else { error = "Select at least one model."; return }
        let n = Double(values.count), mean = values.reduce(0, +) / Double(values.count)
        let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / (n - 1)
        for model in LabModel.allCases where selected.contains(model) {
            switch model {
            case .empirical:
                let p = Double(values.filter { $0 > threshold }.count) / n
                let push = Double(values.filter { $0 == threshold }.count) / n
                let z = 1.96, denom = 1 + 1.96 * 1.96 / n
                let center = (p + z*z/(2*n))/denom
                let half = z * sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
                results.append(LabResult(model: model.rawValue, probability: p, push: push, detail: "95% Wilson interval: \(probability(max(0,center-half)))–\(probability(min(1,center+half))). Describes sampling uncertainty under independent observations."))
            case .normal:
                guard variance > 0 else { error += "Normal model requires variation in the observations. "; continue }
                guard let fitted = PropDistribution(values:values,line:threshold)?.normal else { continue }
                results.append(LabResult(model:model.rawValue,probability:fitted.over,push:fitted.push,detail:String(format:"Fitted mean %.2f · Standard deviation %.2f. Continuity correction for count stats. Calibration unverified.",mean,sqrt(variance))))
            case .poisson:
                guard values.allSatisfy({ $0.rounded() == $0 }) else { error += "Poisson requires integer count observations. "; continue }
                guard mean <= 500 else { error += "Poisson preview supports means up to 500. "; continue }
                guard let fitted = PropDistribution(values:values,line:threshold)?.poisson else { continue }
                results.append(LabResult(model:model.rawValue,probability:fitted.over,push:fitted.push,detail:String(format:"Fitted rate %.2f. Count baseline; independence and equal mean/variance assumptions may not hold. Calibration unverified.",mean)))
            }
        }
        runSummary = "\(subject.isEmpty ? "Untitled analysis" : subject) · \(sport) · \(stat) over \(line) · \(values.count) observations · \(isExample ? "Illustrative example inputs" : "User-supplied inputs")"
    }
}

private struct LabFeature: Identifiable {
    let id:String
    let title:String
    let detail:String
    let group:String
}

struct AILabView: View {
    @EnvironmentObject private var account:AccountStore
    @EnvironmentObject private var purchases:PurchaseStore
    @StateObject private var players=SportsDirectory()
    @StateObject private var schedule=SportsDirectory()
    @StateObject private var marketStore=Store()
    @StateObject private var modelLibrary=ModelLibraryStore()
    @State private var matchedPlayer=""
    @FocusState private var nameFocused:Bool
    @State private var lookupMessage=""
    @State private var resolving=false
    @State private var marketSource=""
    @State private var playerTeam=""
    private var suggestions:[SportsItem] {
        guard nameFocused,player.trimmingCharacters(in:.whitespacesAndNewlines).count>=2 else{return []}
        return Array(players.items.filter {LabNameSearch.matches(query:player,name:$0.name,detail:$0.subtitle)}.sorted {LabNameSearch.rank(query:player,name:$0.name)<LabNameSearch.rank(query:player,name:$1.name)}.prefix(8))
    }
    @State private var selectedQuote:Market?
    @State private var sourceContext=""
    @State private var marketLookupContext=""
    private var matchupContext:String {
        [account.user?.id.uuidString ?? "guest",league,targetKind,player,opponent,historicalEventID].joined(separator:"|")
    }
    private var quoteContext:String {
        [matchupContext,statName,preferredBook,targetLine,side].joined(separator:"|")
    }
    private var offeredProps:[Market] {
        guard marketLookupContext==matchupContext,marketStore.oddsLoadSucceeded,!marketStore.oddsStale else{return []}
        return LabQuoteSelection.offers(marketStore.live,league:league,player:matchedPlayer,book:preferredBook)
    }
    private func lineSourceLabel(now:Date)->String {
        if let quote=selectedQuote,sourceContext==quoteContext {
            let time=quote.updated.flatMap(sportsDate)?.formatted(date:.abbreviated,time:.shortened) ?? "time unavailable"
            let status=LabQuoteSelection.canUsePregameQuote(quote,now:now) ? "Recorded pregame snapshot; refresh before use":"Not a current pregame quote"
            return "\(quote.book) · \(odds(quote.price)) · \(BuilderLogic.quoteAge(quote,now:now).label) · \(time) · \(status)"
        }
        if sourceContext==quoteContext,!marketSource.isEmpty {return marketSource}
        return "Research threshold · Not a verified current sportsbook quote"
    }
    private var lineSource:some View {
        TimelineView(.periodic(from:.now,by:30)) { context in
            Text(lineSourceLabel(now:context.date)).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
        }
    }
    @State private var preferredBook="Research"
    @State private var scoringPreset="draftkings_classic"
    @State private var firstBasketMethod="two_pointer"
    @AppStorage("labFeatureRecipes") private var savedRecipes="[]"
    @State private var showAdvancedAccess=false
    private let basicFeatureIDs=LabPropCatalog.basicFeatureIDs
    @State private var advanced=false
    @State private var training=LabTrainingSettings()
    @State private var modelType="logistic_regression"
    @State private var capabilities:LabCapabilities?
    @State private var capabilityMessage="Checking model testing…"
    @State private var checkingCapabilities=true
    @State private var creating=false
    @State private var creationMessage=""
    @State private var hasRestoredRecipe=false
    @State private var customizedInputs=false
    @State private var loadedVersion:AccountModelVersion?
    @State private var didRestoreInitialVersion=false
    @State private var resultContext=""
    private let initialVersion:AccountModelVersion?
    private let initialDraft:AskPLONNModelDraft?
    init(initialVersion:AccountModelVersion? = nil, initialDraft:AskPLONNModelDraft? = nil) {
        self.initialVersion=initialVersion;self.initialDraft=initialDraft
    }
    private var supportedSetup:LabCapabilities.Support? {capabilities?.match(league:league,stat:statName,target:targetKind)}
    private var retainedSupport:LabCapabilities.Support? {capabilities?.matchRetained(league:league,stat:statName,target:targetKind)}
    private var gameMarket:String? {LabServiceContract.gameMarket(target:targetKind,stat:statName)}
    private var isGame:Bool {gameMarket != nil}
    private var isMoneyline:Bool {gameMarket == "moneyline"}
    private var isYesNoProp:Bool {targetKind == "Player" && LabPropCatalog.isYesNo(statName)}
    private var effectiveLine:Double? {isMoneyline ? 0:isYesNoProp ? 0.5:Double(targetLine)}
    private var requiresHistoricalEvent:Bool { !hasRestoredRecipe || isGame || (supportedSetup ?? retainedSupport)?.historical_event_required == true || league == "NFL" }
    private var trainingCoverageMessage:String {
        guard capabilities != nil else{return "Training coverage is unverified. This setup can be saved, but the service has not confirmed support."}
        guard let support=supportedSetup else{return "Training is unavailable for this target. The catalog offers a configuration only."}
        guard support.supports(modelType) else{return "This model type is unavailable for the selected target."}
        guard !selected.isEmpty,selected.isSubset(of:Set(support.features)) else{return "The selected features are not supported together. Adjust them before requesting a data check and run."}
        if isGame {return "Experimental full-game research using completed ESPN games. Scores include overtime and extra innings. NHL includes the shootout award; NFL moneyline keeps ties separate. No archived odds or betting-return claim."}
        return "History and matchup eligibility are checked when you run. Results require enough verified data for this setup."
    }
    private var canRun:Bool {
        guard creationAvailability == .ready,let support=supportedSetup else{return false}
        return support.supports(modelType) && !selected.isEmpty && selected.isSubset(of:Set(support.features))
    }
    private var canReuse:Bool {
        guard let support=retainedSupport else{return false}
        return support.supports(modelType) && !selected.isEmpty && selected.isSubset(of:Set(support.features))
    }
    private var selectedPropOptions:[String:String]? {
        if LabPropCatalog.isFantasy(statName) {return ["scoring_preset":scoringPreset]}
        if statName == "Method of first basket" {return ["first_basket_method":firstBasketMethod]}
        return nil
    }
    private var currentRecipe:LabRecipe {
        LabRecipe(id:UUID(),name:name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty ? player+" · "+statName:name.trimmingCharacters(in:.whitespacesAndNewlines),player:player,quarterback:quarterback,opponent:opponent,features:selected.sorted(),created:Date(),line:effectiveLine,side:gameMarket == "spread" || isMoneyline ? "Over":side,sport:league,stat:statName,targetKind:targetKind,advanced:advanced,training:training,modelType:modelType,book:preferredBook,historicalEventID:historicalEventID.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty ? nil:historicalEventID,propOptions:selectedPropOptions)
    }
    private var versionContext:String {
        if let version=loadedVersion,version.user_id==account.user?.id,
           currentRecipe.configurationFingerprint == version.configuration.configurationFingerprint {
            return "\(version.name) v\(version.version)"
        }
        return "Unsaved setup"
    }
    @MainActor private func loadCapabilities() async {
        checkingCapabilities=true;capabilities=nil
        defer {checkingCapabilities=false;applyRecommendations()}
        #if DEBUG
        // Simulator UI tests can exercise capability gating without hosted jobs or accounts.
        if let fixture=ProcessInfo.processInfo.environment["PLONN_LAB_CAPABILITIES_FIXTURE"],
           let data=fixture.data(using:.utf8),let local=try? JSONDecoder().decode(LabCapabilities.self,from:data) {
            capabilities=local;capabilityMessage="Local UI test capabilities";return
        }
        #endif
        do {
            var request=URLRequest(url:CloudConfiguration.labURL.appendingPathComponent("capabilities"));request.timeoutInterval=15
            let(data,response)=try await URLSession.shared.data(for:request)
            guard (response as? HTTPURLResponse)?.statusCode==200 else{throw URLError(.badServerResponse)}
            capabilities=try JSONDecoder().decode(LabCapabilities.self,from:data)
            capabilityMessage="Ready to check your model."
        } catch {capabilityMessage="Models are temporarily unavailable. Try again."}
    }
    private var modelTypeGrid:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Model type").font(.system(.headline, design: .monospaced))
            LazyVGrid(columns:[GridItem(.flexible()),GridItem(.flexible())],spacing:8) {
                ForEach(SavedModelType.allCases) { option in
                    let available=supportedSetup?.supports(option.rawValue) == true
                    Button {customizedInputs=true;modelType=option.rawValue} label: {
                        VStack(alignment:.leading,spacing:5) {
                            HStack {Image(systemName:modelType == option.rawValue ? "checkmark.circle.fill":"circle").foregroundStyle(modelType == option.rawValue ? Color.white:Color.secondary);Spacer()}
                            Text(option.title).font(.system(.subheadline, design: .monospaced).weight(.semibold))
                        }.frame(maxWidth:.infinity,alignment:.leading).padding(12)
                         .plonnGlassSurface(selected:modelType == option.rawValue)
                         .overlay(RoundedRectangle(cornerRadius:12).strokeBorder(modelType == option.rawValue ? Color.white.opacity(0.65):Color.clear,lineWidth:1))
                         .contentShape(Rectangle())
                    }.buttonStyle(.plain).disabled(!available)
                     .accessibilityAddTraits(modelType == option.rawValue ? .isSelected:[])
                }
            }.plonnGlassGroup()
            if let support=supportedSetup,!selected.isSubset(of:Set(support.features)) {
                Text("This target supports: \(support.features.joined(separator:", ")). Adjust the selected features before running.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            }
        }
    }
    @State private var targetKind="Team"
    @State private var featureSearch=""
    @State private var step=0
    @State private var player=""
    @State private var quarterback=""
    @State private var historicalEventID=""
    @State private var historicalGame:LabHistoricalGame?
    @State private var historicalTeam:LabHistoricalGame.Team?
    @State private var showGamePicker=false
    @State private var showPropPicker=false
    @State private var showPro=false
    private var historicalSource:LabHistoricalGames.Source { .source(league:league,fullGame:isGame,propSource:supportedSetup?.historical_event_source) }
    private var usesHistoricalMatchup:Bool {requiresHistoricalEvent || !historicalEventID.isEmpty}
    private var opponentLabel:String {historicalGame.flatMap {game in historicalTeam.flatMap {game.opponent(for:$0)?.name}} ?? opponent}
    @State private var opponent=""
    @State private var name=""
    @State private var league="NFL"
    @State private var targetStat="Spread"
    @State private var customStat=""
    @State private var targetLine=""
    @State private var side="Over"
    @State private var selected:Set<String>=["recent","season","homeAway"]
    @State private var restoringRecipe=false
    @State private var saved=false
    @State private var running=false
    @State private var runMessage=""
    @State private var runProbability:Double?
    @State private var askModelContext:(recipe:LabRecipe,result:AskPLONNModelResultSnapshot)?
    @State private var validationCount:Int?
    @State private var validationSummary=""
    @State private var brierScore:Double?
    @State private var pushProbability:Double?
    @State private var tieProbability:Double?
    @State private var baselineScore:Double?
    private var propOptions:[String] {LabPropCatalog.options(league:league,target:targetKind)}
    private var statName:String {targetStat == "Custom stat" ? customStat:targetStat}
    private var features:[LabFeature] {
        if isGame {return [
            .init(id:"recent",title:"Recent scoring",detail:"Selected team's scoring and scores allowed over its last five eligible games.",group:"Team"),
            .init(id:"season",title:"Season baseline",detail:"Selected team's earlier same-season scoring and scores allowed.",group:"Team"),
            .init(id:"homeAway",title:"Home / away",detail:"Home, away or neutral venue for the selected team.",group:"Team"),
            .init(id:"rest",title:"Eligible-game gap",detail:"Days since the last date-guard-eligible game, not actual schedule rest.",group:"Team")
        ]}
        var rows:[LabFeature]=[
            .init(id:"recent",title:"Recent \(statName)",detail:"Pregame history of the selected stat.",group:"Player"),
            .init(id:"season",title:"Season baseline",detail:"Season-to-date production before the matchup.",group:"Player"),
            .init(id:"workload",title:league == "NBA" ? "Minutes & usage":league == "NHL" ? "Ice time & role":league == "MLB" ? "Playing time & opportunities":"Snaps & opportunities",detail:"Expected opportunity and recent workload.",group:"Player"),
            .init(id:"opponent",title:"Opponent matchup splits",detail:"Opponent performance relevant to \(statName).",group:"Matchup"),
            .init(id:"homeAway",title:"Home / away",detail:"Whether the player is at home or away.",group:"Player"),
            .init(id:"rest",title:"Days of rest",detail:"Days since the player’s previous appearance.",group:"Player"),
            .init(id:"availability",title:"Lineup & availability",detail:"Known teammate availability before the game.",group:"Matchup")
        ]
        let columns:[(String,String)]
        switch league {
        case "NFL": columns=[("rush_yards_per_game","Rush Yds/Game"),("rush_attempts_per_game","Rush Att/Game"),("yards_per_carry","Yds/Carry"),("pass_yards_per_game","Pass Yds/Game"),("pass_attempts_per_game","Pass Att/Game"),("completion_pct","Completion %"),("receiving_yards_per_game","Rec Yds/Game"),("receptions_per_game","Receptions/Game"),("targets_per_game","Targets/Game"),("sacks_per_game","Sacks/Game"),("interceptions_per_game","INT/Game")]
        case "NBA": columns=[("points_per_game","Points/Game"),("minutes_per_game","Minutes/Game"),("rebounds_per_game","Rebounds/Game"),("assists_per_game","Assists/Game"),("field_goal_attempts_per_game","FGA/Game"),("three_point_attempts_per_game","3PA/Game"),("three_point_pct","3P %"),("free_throw_attempts_per_game","FTA/Game"),("turnovers_per_game","Turnovers/Game"),("steals_per_game","Steals/Game"),("blocks_per_game","Blocks/Game")]
        case "MLB": columns=[("hits_per_game","Hits/Game"),("at_bats_per_game","AB/Game"),("batting_average","Batting AVG"),("on_base_pct","OBP"),("slugging_pct","SLG"),("ops","OPS"),("rbi_per_game","RBI/Game"),("stolen_bases_per_game","SB/Game"),("era","ERA"),("whip","WHIP"),("strikeouts_per_nine","K/9"),("walks_per_nine","BB/9")]
        default: columns=[("goals_per_game","Goals/Game"),("assists_per_game","Assists/Game"),("shots_on_goal_per_game","Shots/Game"),("shooting_pct","Shooting %"),("blocked_shots_per_game","Blocks/Game"),("time_on_ice_per_game","TOI/Game"),("save_pct","Save %"),("goals_against_average","GAA"),("power_play_pct","Power Play %"),("penalty_kill_pct","Penalty Kill %")]
        }
        rows.removeAll { !basicFeatureIDs.contains($0.id) }
        for scope in [targetKind, "Opponent team"] {
            for (key,label) in columns {
                rows.append(.init(id:"column.\(league.lowercased()).\(scope == "Player" ? "player":scope == "Team" ? "team":"opponent").\(key)",title:label,detail:"\(scope) · Pregame season to date",group:scope))
            }
        }
        if league == "NFL" {
            let advancedColumns:[(String,String,String)] = [
                ("rush_epa_per_play","Rush EPA/Play","Expected points added per rushing play."),
                ("rush_success_rate","Rush Success %","Share of rushing plays with positive EPA."),
                ("yards_after_contact_per_attempt","Rush YAC/Att","Rushing yards after first contact per attempt."),
                ("rush_yards_over_expected","Rush Yds Above Expected","Actual rushing yards minus a tracking model’s expected yards."),
                ("pass_epa_per_dropback","Pass EPA/Dropback","Expected points added per passing dropback."),
                ("completion_pct_over_expected","CPOE","Completion percentage minus model-expected completion percentage."),
                ("yards_per_route_run","Yds/Route Run","Receiving yards divided by routes run."),
                ("pressure_rate","Pressure %","Share of opponent dropbacks with a recorded pressure; definition varies by provider.")
            ]
            for (key,title,definition) in advancedColumns {
                let scope = key == "pressure_rate" ? "Opponent team":targetKind
                rows.append(.init(id:"advanced.nfl.\(scope).\(key)",title:title,detail:definition+" Data source pending.",group:scope))
            }
        }
        return rows
    }
    private var recipes:[LabRecipe] { (try? JSONDecoder().decode([LabRecipe].self,from:Data(savedRecipes.utf8))) ?? [] }
    private var trainingSignature:String {"\(advanced)|\(training.epochs)|\(training.learningRate)|\(training.regularization)|\(training.validationShare)"}
    private var setupSignature:String { [account.user?.id.uuidString ?? "guest",preferredBook,historicalEventID,modelType,targetKind,trainingSignature,league,targetStat,customStat,player,quarterback,opponent,name,targetLine,side,scoringPreset,firstBasketMethod,selected.sorted().joined()].joined(separator:"|") }
    private var validTarget:Bool { (!requiresHistoricalEvent || !historicalEventID.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty) && LabPropCatalog.configurationLimitation(statName)==nil && LabServiceContract.validLine(effectiveLine,target:targetKind,stat:statName) && !player.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && !statName.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && !opponent.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty }
    var body:some View {AccountRequiredView(feature:.privateModels){protectedBody}
        .sheet(isPresented:$showPro){NavigationStack{ProPaywallView().toolbar{ToolbarItem(placement:.confirmationAction){Button("Done"){showPro=false}}}}}}
    private var protectedBody:some View {
        Group {
            if initialVersion != nil || initialDraft != nil { propBody }
            else {
                ScrollView {
                    VStack(spacing:16) {
                        Text("Models").font(.system(.title2,design:.monospaced).bold()).frame(maxWidth:.infinity,alignment:.leading)
                        NavigationLink { UpcomingGamesView() } label: {
                            modelEntry("Upcoming game", icon:"bolt.fill", prominent:true)
                        }.accessibilityIdentifier("lab-one-click")
                        NavigationLink { AskPLONNView(startModelBuilder:true) } label: {
                            modelEntry("Create with AI", icon:"sparkles")
                        }.accessibilityIdentifier("lab-create-with-ai")
                        NavigationLink { propBody } label: {
                            modelEntry("Custom model", icon:"plus")
                        }
                        NavigationLink { MyModelsView() } label: {
                            modelEntry("My Models", icon:"square.stack.3d.up")
                        }
                        if LaunchSports.includes("Soccer") {
                        NavigationLink { NationalTeamModelView() } label: {
                            HStack(spacing:12) {
                                Image(systemName:"soccerball").foregroundStyle(brandAccent)
                                Text("Soccer Elo").font(.system(.subheadline, design: .monospaced).weight(.semibold))
                                Spacer()
                                Image(systemName:"chevron.right").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                            }.padding(.horizontal,8).padding(.vertical,16)
                        }
                        }
                    }.buttonStyle(.plain).padding(20)
                }.background(ModelLandscape()).navigationTitle("PLONN").navigationBarTitleDisplayMode(.inline).toolbar {ToolbarItem(placement:.principal) {Brand()}}
            }
        }
    }
    private func modelEntry(_ title:String, icon:String, prominent:Bool=false)->some View {
        HStack(spacing:16) {
            Image(systemName:icon).font(.system(.title2, design: .monospaced)).foregroundStyle(brandAccent).frame(width:32)
            VStack(alignment:.leading,spacing:8) {
                Text(title).font(.system(.headline, design: .monospaced))
            }
            Spacer(minLength:0)
            Image(systemName:"chevron.right").font(.system(.caption, design: .monospaced).weight(.semibold)).foregroundStyle(.secondary)
        }.padding(22).frame(maxWidth:.infinity,alignment:.leading)
            .background(navySurface,in:RoundedRectangle(cornerRadius:18))
            .overlay(RoundedRectangle(cornerRadius:18).strokeBorder(brandAccent.opacity(prominent ? 0.4:0),lineWidth:1))
    }
    private var creationAvailability:LabCreationPolicy.Availability {
        let status=LabCreationPolicy.availability(capabilities,loading:checkingCapabilities,support:supportedSetup)
        if status == .ready,let support=supportedSetup,
           (!support.supports(modelType) || selected.isEmpty || !selected.isSubset(of:Set(support.features))) {return .unsupported}
        return status
    }
    private var questionOptions:[String] {LabCreationPolicy.questions(league:league,target:targetKind,capabilities:capabilities)}
    private var savedMatchesCurrent:Bool {
        guard let version=loadedVersion,version.user_id==account.user?.id else{return false}
        return version.configuration.configurationFingerprint == currentRecipe.configurationFingerprint
    }
    private var primaryTitle:String {
        if creating {return "Saving your model…"}
        if running {return "Testing your model…"}
        if !savedMatchesCurrent {return loadedVersion == nil ? (canRun ? "Create model":"Save setup"):"Save changes"}
        return canRun || canReuse ? (runProbability == nil ? "Test model":"Refresh result"):"Saved to My Models"
    }
    private var availabilityCard:some View {
        VStack(alignment:.leading,spacing:6) {
            Label(creationAvailability.title,systemImage:creationAvailability == .ready ? "checkmark.circle":"info.circle")
                .font(.subheadline.weight(.semibold))
            if creationAvailability != .ready {Text(creationAvailability.detail).font(.caption).foregroundStyle(.secondary)}
            if creationAvailability != .ready && !checkingCapabilities {
                Button("Check again") {Task {await loadCapabilities()}}.font(.caption)
            }
        }.frame(maxWidth:.infinity,alignment:.leading).padding(14)
            .background(navySurface,in:RoundedRectangle(cornerRadius:14)).accessibilityIdentifier("lab-availability")
    }
    private var propBody:some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment:.leading,spacing:20) {
                    HStack(spacing:8) {
                        ForEach(Array(["Question","Game","Review"].enumerated()),id:\.offset) { index,title in
                            Button {if index<step {step=index}} label: {
                                HStack(spacing:5) {
                                    Text("\(index+1)").font(.caption.weight(.semibold)).frame(width:24,height:24)
                                        .background(step==index ? Color.primary:Color.secondary.opacity(0.15),in:Circle())
                                        .foregroundStyle(step==index ? navy:Color.primary)
                                    Text(title).font(.subheadline.weight(.semibold))
                                }.frame(maxWidth:.infinity,minHeight:44).plonnGlassSurface(selected:step==index,radius:12)
                            }.buttonStyle(.plain).accessibilityAddTraits(step==index ? .isSelected:[])
                        }
                    }.plonnGlassGroup().id("lab-step-top")
                    if step==0 {availabilityCard;questionCard}
                    if step==1 {targetCard}
                    if step==2 {reviewCard}
                }.padding(20).disabled(creating || running)
            }
            .safeAreaInset(edge:.bottom) {
                VStack(spacing:7) {
                    if step<2 {
                        Button {withAnimation {step+=1}} label: {
                            Text(step==0 ? "Choose game":"Review model").font(.headline).frame(maxWidth:.infinity).padding(.vertical,9)
                        }.plonnGlassButton(prominent:true).disabled(step==0 ? !questionOptions.contains(statName):!validTarget)
                    } else {
                        Button {Task {await createModel()}} label: {
                            HStack {if creating || running {ProgressView()};Text(primaryTitle).font(.headline)}
                                .frame(maxWidth:.infinity).padding(.vertical,9)
                        }.plonnGlassButton(prominent:true)
                            .disabled(creating || running || !validTarget || selected.isEmpty || (savedMatchesCurrent && !canRun && !canReuse))
                            .accessibilityIdentifier("lab-create-model")
                        Text(savedMatchesCurrent ? "Your model is saved privately.":canRun ? "Saves privately and starts a historical test.":"Saves privately. Testing can be started later.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }.padding(.horizontal,20).padding(.vertical,12).background(.ultraThinMaterial)
            }
            .onChange(of:step) {_,_ in withAnimation {proxy.scrollTo("lab-step-top",anchor:.top)}}
            .onChange(of:running) {wasRunning,isRunning in
                if wasRunning && !isRunning && step==2 {withAnimation {proxy.scrollTo("lab-result",anchor:.top)}}
            }
            .onChange(of:creating) {wasCreating,isCreating in
                if wasCreating && !isCreating && !running && !creationMessage.isEmpty && step==2 {
                    withAnimation {proxy.scrollTo("lab-result",anchor:.top)}
                }
            }
        }.background(ModelAurora()).background(LeagueBackdrop(league:league))
        .navigationTitle(initialVersion == nil ? "Create a model":"Your model").navigationBarTitleDisplayMode(.inline)
        .task {
            if !didRestoreInitialVersion {
                didRestoreInitialVersion=true
                if let version=initialVersion,version.user_id==account.user?.id {loadVersion(version)}
                else if let draft=initialDraft {applyDraft(draft)}
            }
            await loadCapabilities()
            if initialVersion != nil {await runLocal(resolveOnly:true)}
        }
        .sheet(isPresented:$showGamePicker) {
            LabGamePicker(league:league,source:historicalSource,initialDate:historicalGame?.date) {game,team in selectHistoricalGame(game,team:team)}
        }
        .sheet(isPresented:$showPropPicker) {
            LabPlayerPropPicker(league:league,selection:targetStat,capabilities:capabilities) {targetStat=$0}
        }
        .onChange(of:historicalSource) {_,_ in guard !restoringRecipe else{return};clearHistoricalGame()}
        .task(id:league+targetKind) {if targetKind == "Player" {await players.load(sport:league,view:"Players",date:Date())}}
        .onChange(of:targetKind) {_,_ in guard !restoringRecipe else{return};resetQuestion()}
        .onChange(of:league) {_,_ in guard !restoringRecipe else{return};resetQuestion()}
        .onChange(of:player) {_,value in
            if !restoringRecipe && value != matchedPlayer {
                if historicalGame == nil {opponent=""}
                targetLine="";marketSource="";matchedPlayer="";marketStore.live=[];lookupMessage=""
            }
        }
        .onChange(of:targetStat) {_,_ in
            guard !restoringRecipe else{return}
            customizedInputs=false;applyRecommendations(force:true);targetLine="";side="Over"
        }
        .onChange(of:account.user?.id) {_,_ in
            loadedVersion=nil;clearHistoricalGame();player="";matchedPlayer="";opponent="";name="";targetLine="";quarterback=""
            marketStore.live=[];step=0;creationMessage=""
        }
        .onChange(of:setupSignature) {_,_ in
            saved=false;runProbability=nil;askModelContext=nil;runMessage="";resultContext="";validationCount=nil;validationSummary="";brierScore=nil;pushProbability=nil;tieProbability=nil;baselineScore=nil;creationMessage=""
        }
    }
    private var questionCard:some View {
        VStack(alignment:.leading,spacing:18) {
            Text("Market").font(.title2.bold())
            Picker("League",selection:$league) {ForEach(LabPropCatalog.leagues,id:\.self) {Text($0)}}.pickerStyle(.segmented)
            Picker("Analyze",selection:$targetKind) {Text("A team").tag("Team");Text("A player").tag("Player")}.pickerStyle(.segmented)
            if targetKind == "Player" {
                Button {showPropPicker=true} label: {
                    HStack(spacing:12) {
                        VStack(alignment:.leading,spacing:6) {
                            Text("Player prop").font(.caption).foregroundStyle(.secondary)
                            Text(LabPropCatalog.displayName(statName)).font(.headline)
                            Text(LabPropCatalog.testingStatus(league:league,stat:statName,capabilities:capabilities)).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName:"chevron.down")
                    }.padding(16).frame(maxWidth:.infinity,alignment:.leading).plonnGlassSurface(radius:16)
                }.buttonStyle(.plain).accessibilityIdentifier("lab-player-prop-picker")
                Text("Props marked ‘Save setup only’ can be saved privately, but model testing isn’t connected for them.").font(.caption).foregroundStyle(.secondary)
            } else {
            if questionOptions.isEmpty {
                Text("No connected questions for this sport yet. Choose another sport or check again.").foregroundStyle(.secondary)
            }
            ForEach(questionOptions,id:\.self) { option in
                Button {targetStat=option} label: {
                    HStack(spacing:12) {
                        Image(systemName:targetStat==option ? "checkmark.circle.fill":"circle").foregroundStyle(targetStat==option ? brandAccent:Color.secondary)
                        VStack(alignment:.leading,spacing:5) {
                            Text(LabCreationPolicy.question(option,target:targetKind)).font(.headline)
                            if targetKind == "Team" {Text(option == "Moneyline" ? "Win or lose":option == "Spread" ? "Against a spread you choose":"Combined score against a line you choose").font(.caption).foregroundStyle(.secondary)}
                            if league == "NFL" && option == "Moneyline" {Text("Needs enough historical wins, losses and ties.").font(.caption).foregroundStyle(.secondary)}
                        }
                        Spacer(minLength:0)
                    }.padding(16).frame(maxWidth:.infinity,alignment:.leading).plonnGlassSurface(selected:targetStat==option,radius:16)
                }.buttonStyle(.plain).accessibilityAddTraits(targetStat==option ? .isSelected:[])
            }
            }
            DisclosureGroup("More options") {
                NavigationLink("Advanced statistical tools") {StatisticalLabView()}.padding(.vertical,8)
                if !recipes.isEmpty {
                    Text("Previously saved on this device").font(.caption).foregroundStyle(.secondary)
                    ForEach(recipes) {recipe in Button(recipe.name) {restore(recipe)}.disabled(!LaunchSports.includes(recipe.sport ?? "NFL"))}
                }
            }.font(.subheadline)
        }
    }
    private var targetCard:some View {
        VStack(alignment:.leading,spacing:18) {
            Text("Matchup").font(.title2.bold())
            Text(LabCreationPolicy.question(statName,target:targetKind)).font(.subheadline).foregroundStyle(.secondary)
            Panel {VStack(alignment:.leading,spacing:10) {
                Label("Completed game",systemImage:"calendar").font(.headline)
                if let game=historicalGame {
                    Text(game.title).font(.headline)
                    Text(LabHistoricalGames.displayDate(game.date)).font(.subheadline).foregroundStyle(.secondary)
                    if let team=historicalTeam {Text((isGame ? "Your team: ":"Player’s team: ")+team.name).font(.subheadline)}
                } else if !historicalEventID.isEmpty {
                    Text("\(player) vs \(opponentLabel)").font(.headline)
                    Text("Saved matchup").font(.caption).foregroundStyle(.secondary)
                }
                Button {showGamePicker=true} label: {Label(historicalEventID.isEmpty ? "Choose completed game":"Change game or team",systemImage:"calendar")}
                    .accessibilityIdentifier("lab-choose-game")
                Text(targetKind == "Player" ? "Choose the team your player was on in this game.":"Choose the team you want to analyze. We’ll fill in its opponent.").font(.caption).foregroundStyle(.secondary)
            }}
            if !isGame {
                VStack(alignment:.leading,spacing:0) {
                    Text("Player").font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.bottom,7)
                    HStack {
                        Image(systemName:"magnifyingglass").foregroundStyle(.secondary)
                        TextField("Search player name or team",text:$player).focused($nameFocused)
                            .autocorrectionDisabled().submitLabel(.done).onSubmit {nameFocused=false}
                            .accessibilityIdentifier("lab-player-search")
                        if !player.isEmpty {Button {player="";nameFocused=true} label:{Image(systemName:"xmark.circle.fill")}.accessibilityLabel("Clear player search")}
                    }.padding(12).background(navySurface,in:RoundedRectangle(cornerRadius:10))
                    if nameFocused,player.trimmingCharacters(in:.whitespacesAndNewlines).count>=2 {
                        VStack(alignment:.leading,spacing:0) {
                            ForEach(suggestions) {item in
                                Button {matchedPlayer=item.name;player=item.name;nameFocused=false} label: {
                                    HStack {
                                        VStack(alignment:.leading,spacing:3) {Text(item.name).font(.subheadline.weight(.semibold));Text(item.subtitle).font(.caption).foregroundStyle(.secondary)}
                                        Spacer();Image(systemName:"arrow.up.left").font(.caption).foregroundStyle(.secondary)
                                    }.padding(12).contentShape(Rectangle())
                                }.plonnGlassButton().accessibilityIdentifier("lab-player-suggestion-"+item.id)
                                if item.id != suggestions.last?.id {Divider().padding(.horizontal,12)}
                            }
                            if players.loading {ProgressView("Loading players…").font(.caption).padding(12)}
                            else if suggestions.isEmpty {Text("No matching names in the current directory. You can enter a former player’s full name for historical testing.").font(.caption).foregroundStyle(.secondary).padding(12)}
                        }.background(navySurface,in:RoundedRectangle(cornerRadius:10))
                         .overlay(RoundedRectangle(cornerRadius:10).stroke(Color.secondary.opacity(0.25)))
                         .padding(.top,4).accessibilityIdentifier("lab-player-dropdown")
                    }
                }
            }
            if LabPropCatalog.isFantasy(statName) {
                Text("Fantasy scoring").font(.headline)
                Picker("Fantasy scoring",selection:$scoringPreset) {
                    Text("DraftKings").tag("draftkings_classic")
                    Text("FanDuel").tag("fanduel_classic")
                }.pickerStyle(.segmented)
                Text("Classic fantasy scoring, without captain multipliers. Your selected rules are saved with the model.").font(.caption).foregroundStyle(.secondary)
            }
            if statName == "Method of first basket" {
                Picker("First score method",selection:$firstBasketMethod) {
                    Text("Two-pointer").tag("two_pointer")
                    Text("Three-pointer").tag("three_pointer")
                    Text("Free throw").tag("free_throw")
                }
            }
            if league == "MLB",historicalSource == .espn,["Pitcher strikeouts","Pitcher outs","Earned runs","Hits allowed","Walks allowed","Pitcher win"].contains(statName) {Text("History includes starting and relief pitching appearances.").font(.caption).foregroundStyle(.secondary)}
            if let note=LabPropCatalog.propNote(statName) {Text(note).font(.caption).foregroundStyle(.secondary)}
            if isYesNoProp {
                Text(LabPropCatalog.displayName(statName)).font(.headline)
                Picker("Outcome",selection:$side) {Text("Yes").tag("Over");Text("No").tag("Under")}.pickerStyle(.segmented)
                Text("Choose whether this outcome happens in the selected game.").font(.caption).foregroundStyle(.secondary)
            } else if !isMoneyline {
                if gameMarket != "spread" {
                    Picker("Direction",selection:$side) {Text("Over").tag("Over");Text("Under").tag("Under")}.pickerStyle(.segmented)
                }
                entry(gameMarket == "spread" ? "Spread for \(player.isEmpty ? "your team":player)":gameMarket == "total" ? "Combined-score line":"\(LabPropCatalog.displayName(statName)) line",placeholder:gameMarket == "spread" ? "e.g. -2.5":"Enter the line to test",text:$targetLine)
                    .keyboardType(gameMarket == "spread" ? .numbersAndPunctuation:.decimalPad)
                Text(gameMarket == "spread" ? "A positive spread gives your team points; a negative spread gives points away.":"Choose the line you want to test against past results.").font(.caption).foregroundStyle(.secondary)
            }
            if !opponentLabel.isEmpty {LabeledContent("Opponent",value:opponentLabel).font(.subheadline)}
            Text("Historical testing only. This won’t place a bet or predict an upcoming game.").font(.caption).foregroundStyle(.secondary)
        }
    }
    private var featureCards:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Model inputs").font(.headline)
            ForEach(features.filter{selected.contains($0.id)}) {feature in
                HStack(alignment:.top,spacing:10) {
                    Image(systemName:"checkmark.circle.fill").foregroundStyle(brandAccent)
                    VStack(alignment:.leading,spacing:3) {Text(feature.title).font(.subheadline.weight(.semibold));Text(feature.detail).font(.caption).foregroundStyle(.secondary)}
                }
            }
            DisclosureGroup("Advanced") {
                VStack(alignment:.leading,spacing:18) {
                    modelTypeGrid
                    Text("Inputs (up to 5)").font(.headline)
                    featureGrid(features.filter{supportedSetup?.features.contains($0.id) ?? basicFeatureIDs.contains($0.id)},locked:false)
                    Button("Use recommended settings") {customizedInputs=false;applyRecommendations(force:true)}
                    SportsbookSelector(selection:$preferredBook,options:["Research","DraftKings","FanDuel","Hard Rock Bet","BetMGM","Caesars","Fanatics","BetRivers"])
                    Text("Research uses your chosen line without claiming a sportsbook quote.").font(.caption).foregroundStyle(.secondary)
                }.padding(.top,12)
            }.font(.subheadline).accessibilityIdentifier("lab-advanced")
        }
    }
    private func featureGrid(_ entries:[LabFeature],locked:Bool)->some View {
        LazyVGrid(columns:Array(repeating:GridItem(.flexible(),spacing:10),count:2),spacing:6) {
            ForEach(entries) {feature in
                let available = supportedSetup.map { $0.features.contains(feature.id) } ?? !locked
                let access = purchases.hasAccess && purchases.owner == account.user?.id
                let blocked = !available || (locked && !access)
                Button {
                    customizedInputs=true
                    if selected.contains(feature.id) {selected.remove(feature.id)}
                    else if !blocked && (selected.count<5 || access) {selected.insert(feature.id)}
                    else {showAdvancedAccess=true}
                } label: {
                    VStack(alignment:.leading,spacing:5) {
                        HStack {Image(systemName:blocked ? "lock.fill":selected.contains(feature.id) ? "checkmark.circle.fill":"circle").foregroundStyle(!blocked && selected.contains(feature.id) ? Color.white:Color.secondary);Spacer()}.font(.system(.caption, design: .monospaced))
                        Text(feature.title).font(.system(.caption, design: .monospaced).weight(.semibold)).multilineTextAlignment(.leading).fixedSize(horizontal:false,vertical:true)
                    }.frame(maxWidth:.infinity,minHeight:76,alignment:.topLeading).padding(7)
                        .foregroundStyle(Color.primary)
                        .plonnGlassSurface(selected:!blocked && selected.contains(feature.id),radius:14)
                        .overlay(RoundedRectangle(cornerRadius:14).strokeBorder(!blocked && selected.contains(feature.id) ? Color.white.opacity(0.65):Color.clear,lineWidth:1))
                        .contentShape(Rectangle())
                }.buttonStyle(.plain).help(feature.detail).accessibilityHint(feature.detail).accessibilityValue(blocked ? "Unavailable":selected.contains(feature.id) ? "Selected":"Not selected")
            }
        }.plonnGlassGroup().sheet(isPresented:$showAdvancedAccess) {
            NavigationStack {
                VStack(alignment:.leading,spacing:16) {
                    Text("Unavailable inputs").font(.system(.title2, design: .monospaced).weight(.semibold))
                    Text("These stats aren’t available yet. Upgrading won’t unlock missing data or unsupported models.")
                    NavigationLink("Subscription") {SubscriptionView()}
                    Button("Continue with available features") {showAdvancedAccess=false}
                }.padding(24)
            }.presentationDetents([.medium,.large])
        }
    }
    private var reviewCard:some View {
        VStack(alignment:.leading,spacing:18) {
            Text("Review model").font(.title2.bold())
            Panel {VStack(alignment:.leading,spacing:12) {
                Label("Test on past games",systemImage:"clock.arrow.circlepath").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(player).font(.title3.bold())
                Text(isYesNoProp ? "\(LabPropCatalog.displayName(statName)) · \(side == "Over" ? "Yes":"No") · \(league)":isMoneyline ? "Team to win · \(league)":gameMarket == "spread" ? "Spread \(targetLine) · \(league)":"\(side) \(targetLine) \(LabPropCatalog.displayName(statName)) · \(league)").font(.subheadline)
                if let label=LabPropCatalog.optionsLabel(selectedPropOptions) {Text(label).font(.subheadline).foregroundStyle(.secondary)}
                LabeledContent("Opponent",value:opponentLabel).font(.subheadline)
                if let game=historicalGame {Text(LabHistoricalGames.displayDate(game.date)).font(.caption).foregroundStyle(.secondary)}
                Divider()
                entry("Model name",placeholder:currentRecipe.name,text:$name)
                Text("Private · Saved to your account").font(.caption).foregroundStyle(.secondary)
            }}
            Panel {featureCards}
            availabilityCard
            Color.clear.frame(height:1).id("lab-result")
            if !creationMessage.isEmpty {Label(creationMessage,systemImage:savedMatchesCurrent ? "checkmark.circle.fill":"exclamationmark.circle").font(.subheadline).accessibilityIdentifier("lab-save-status")}
            if savedMatchesCurrent {
                NavigationLink("Open My Models") {MyModelsView()}.font(.subheadline)
            }
            if running {ProgressView(runMessage.isEmpty ? "Testing your model…":runMessage)}
            else if !runMessage.isEmpty {Text(runMessage).font(.subheadline).accessibilityIdentifier("lab-run-status")}
            if let value=runProbability {
                Panel {VStack(alignment:.leading,spacing:10) {
                    if let brierScore,let baselineScore {
                        Label(brierScore < baselineScore ? "Beat the baseline in this test":"Did not beat the baseline",systemImage:"chart.bar.xaxis")
                            .font(.headline).accessibilityIdentifier("lab-validation-quality")
                        Text(brierScore < baselineScore ? "More accurate than a simple historical-rate estimate on the held-out games. This is a limited historical test.":"This model wasn’t more accurate than a simple historical-rate estimate on the held-out games. Treat its estimate cautiously.")
                            .font(.caption).foregroundStyle(.secondary)
                        Divider()
                    }
                    Text(LabServiceContract.probabilityLabel(currentRecipe)).font(.headline)
                    AnimatedProbability(value:value).font(.largeTitle)
                    Text("Experimental historical estimate").font(.subheadline)
                    if let tieProbability {Text("Final tie: "+probability(tieProbability)).font(.caption)}
                    if let pushProbability,pushProbability>0 {Text("Push: "+probability(pushProbability)).font(.caption)}
                    Text("This result uses earlier games and does not establish betting returns.").font(.caption).foregroundStyle(.secondary)
                    if let count=validationCount {Text("Checked against \(count) held-out games").font(.subheadline)}
                    DisclosureGroup("Test details") {
                        VStack(alignment:.leading,spacing:8) {
                            Text(visibleValidationSummary).font(.caption)
                            if let brierScore {Metric(title:"Brier score",value:String(format:"%.3f",brierScore))}
                            if let baselineScore {Metric(title:"Baseline Brier",value:String(format:"%.3f",baselineScore))}
                        }
                    }.font(.caption)
                    if let context=askModelContext,context.result.ownerID==account.user?.id {
                        NavigationLink("Explain this result") {AskPLONNView(initialRecipe:context.recipe,initialModelResult:context.result)}
                    }
                }}
            }
            if savedMatchesCurrent && canRun {
                DisclosureGroup("Run options") {
                    Button("Retrain with new data") {Task {await runLocal(retrain:true)}}.disabled(running || creating)
                    Text("Starts a new test using currently available historical data.").font(.caption).foregroundStyle(.secondary)
                }.font(.subheadline)
            }
        }
    }
    private func resetQuestion() {
        clearHistoricalGame();player="";matchedPlayer="";opponent="";targetLine="";marketStore.live=[];lookupMessage="";step=0
        customizedInputs=false
        targetStat=questionOptions.first(where:{capabilities?.match(league:league,stat:$0,target:targetKind) != nil}) ?? questionOptions.first ?? (targetKind == "Team" ? "Moneyline":propOptions.first ?? "Custom stat")
        applyRecommendations(force:true)
    }
    private func applyRecommendations(force:Bool=false) {
        guard force || (!hasRestoredRecipe && !customizedInputs) else{return}
        selected=LabCreationPolicy.recommendedFeatures(supportedSetup,fullGame:isGame)
        modelType=LabCreationPolicy.recommendedModel(supportedSetup)
    }
    @MainActor private func createModel() async {
        guard !creating,!running,validTarget,!selected.isEmpty else{return}
        guard account.hasSignedInAccount,let owner=account.user?.id else {account.access.present(feature:.privateModels);return}
        creating=true;creationMessage=""
        defer {creating=false}
        let signature=setupSignature
        if !savedMatchesCurrent {
            let recipe=currentRecipe
            let modelID=loadedVersion?.user_id == owner ? loadedVersion?.model_id:nil
            guard let version=await modelLibrary.save(recipe,modelID:modelID,account:account) else {
                if account.user?.id==owner {creationMessage=modelLibrary.message.isEmpty ? "Couldn’t save. Please try again.":modelLibrary.message}
                return
            }
            guard account.user?.id==owner,signature==setupSignature else{return}
            loadedVersion=version
        }
        creationMessage="Saved to My Models"
        creating=false
        if canRun || canReuse {await runLocal()}
    }
    private var visibleValidationSummary: String {
        #if DEBUG
        return validationSummary.isEmpty ? "Awaiting model and historical pregame data. No backtest has run." : validationSummary
        #else
        guard let count = validationCount, count > 0, brierScore != nil else {
            return "Historical validation is unavailable for this result."
        }
        return "Checked against held-out historical games. This does not establish betting returns or readiness for live bets."
        #endif
    }
    private func clearHistoricalGame() {
        historicalEventID="";historicalGame=nil;historicalTeam=nil
        selectedQuote=nil;sourceContext="";marketSource=""
    }
    private func selectHistoricalGame(_ game:LabHistoricalGame,team:LabHistoricalGame.Team) {
        guard let other=game.opponent(for:team) else{return}
        historicalGame=game;historicalTeam=team;historicalEventID = league == "MLB" && !isGame && historicalSource == .espn ? "espn:"+game.id : game.id
        if isGame {matchedPlayer=team.name;player=team.name}
        opponent=other.requestName
        if initialDraft == nil || selectedQuote != nil {targetLine="";marketSource=""}
        selectedQuote=nil;sourceContext="";marketLookupContext="";marketStore.live=[];lookupMessage=""
    }
    private func loadVersion(_ version:AccountModelVersion) {
        guard version.user_id==account.user?.id, LaunchSports.includes(version.configuration.sport ?? "NFL") else{return}
        if currentRecipe.configurationFingerprint == version.configuration.configurationFingerprint {
            let previous=versionContext
            loadedVersion=version
            if resultContext.hasPrefix(previous) {resultContext=versionContext + resultContext.dropFirst(previous.count)}
        } else {restore(version.configuration);loadedVersion=version}
    }
    private func applyDraft(_ draft:AskPLONNModelDraft) {
        guard draft.canReview,let sport=draft.league,let target=draft.target,let stat=draft.stat,let subject=draft.subject else{return}
        restoringRecipe=true;hasRestoredRecipe=false;customizedInputs=false;loadedVersion=nil
        league=sport;targetKind=target;targetStat=stat;customStat=""
        player=subject;matchedPlayer=subject;opponent=draft.opponent ?? "";name=draft.title
        targetLine=draft.effectiveLine.map{String(format:"%g",$0)} ?? "";side=draft.effectiveSide ?? "Over"
        scoringPreset=draft.scoringPreset ?? "draftkings_classic";firstBasketMethod=draft.firstBasketMethod ?? "two_pointer"
        marketSource="AI draft · Research threshold, not a sportsbook quote";selectedQuote=nil
        historicalEventID="";historicalGame=nil;historicalTeam=nil;step=1
        Task {@MainActor in await Task.yield();restoringRecipe=false}
    }
    private func restore(_ recipe:LabRecipe) {
        guard LaunchSports.includes(recipe.sport ?? "NFL") else{return}
        loadedVersion=nil
        restoringRecipe=true;hasRestoredRecipe=true
        if let book=recipe.book {preferredBook=book}
        scoringPreset=recipe.propOptions?["scoring_preset"] ?? "draftkings_classic";firstBasketMethod=recipe.propOptions?["first_basket_method"] ?? "two_pointer"
        advanced=recipe.advanced ?? false;training=recipe.training ?? LabTrainingSettings();modelType=recipe.resolvedModelType
        targetKind=recipe.targetKind ?? "Player"
        league=recipe.sport ?? "NFL"
        let restoredStat=recipe.stat ?? "Rec Yds"
        if propOptions.contains(restoredStat) {targetStat=restoredStat} else {targetStat="Custom stat";customStat=restoredStat}
        historicalGame=nil;historicalTeam=nil
        historicalEventID=recipe.historicalEventID ?? ""
        matchedPlayer=recipe.player;player=recipe.player;quarterback=recipe.quarterback;opponent=recipe.opponent;name=recipe.name
        selected=Set(recipe.features);targetLine=recipe.line.map{String($0)} ?? "";side=recipe.side ?? "Over"
        marketSource="Saved recipe line · Research threshold, not a current sportsbook quote";selectedQuote=nil;sourceContext=quoteContext;marketStore.live=[];step=2;saved=false
        Task {@MainActor in await Task.yield();restoringRecipe=false;if capabilities != nil {await runLocal(resolveOnly:true)}}
    }
    @MainActor private func runLocal(retrain:Bool = false, resolveOnly:Bool = false) async {
        guard account.hasSignedInAccount else {account.access.present(feature:.privateModels);return}
        guard purchases.mayUseProFeatures else {showPro=true;return}
        guard !running else{return}
        guard validTarget,canRun || (!retrain && canReuse),let support=supportedSetup ?? retainedSupport else{runMessage="This model, target, or feature set is not supported by the connected service.";return}
        running=true;runMessage="";runProbability=nil;askModelContext=nil;validationCount=nil;validationSummary="";brierScore=nil;pushProbability=nil;tieProbability=nil;baselineScore=nil
        defer{running=false}
        let signature=setupSignature
        let context=versionContext + " · " + (SavedModelType(rawValue:modelType)?.title ?? modelType) + " · " + selected.sorted().joined(separator:", ")
        do {
            guard let owner=account.user?.id,account.hasSignedInAccount else{throw URLError(.userAuthenticationRequired)}
            let recipe=currentRecipe
            let payload=try LabServiceContract.payload(recipe,support:support)
            let canonical=try LabSubmissionStore.canonical(payload)
            let submissions=LabSubmissionStore()
            let session = try await account.client.auth.session
            guard signature==setupSignature,account.user?.id==owner else{return}
            var retainedBody:[String:Any]?
            var retainedArtifact:String?
            let resumeOutstanding=submissions.hasOutstanding(owner:owner,payload:canonical)
            if !retrain && !resumeOutstanding && capabilities?.retained_artifacts == true {
                var resolve=URLRequest(url:CloudConfiguration.labURL.appendingPathComponent("artifacts/resolve"))
                resolve.httpMethod="POST";resolve.timeoutInterval=20
                resolve.setValue("Bearer " + session.accessToken,forHTTPHeaderField:"Authorization")
                resolve.setValue("application/json",forHTTPHeaderField:"Content-Type")
                resolve.httpBody=canonical
                let(data,response)=try await URLSession.shared.data(for:resolve)
                guard signature==setupSignature else{return}
                guard let resolved=try JSONSerialization.jsonObject(with:data) as? [String:Any] else{throw URLError(.cannotParseResponse)}
                if (response as? HTTPURLResponse)?.statusCode==200,
                   let artifact=resolved["artifact_id"] as? String,artifact.count==32,artifact.allSatisfy({$0.isHexDigit}) {
                    resolve.url=CloudConfiguration.labURL.appendingPathComponent("artifacts").appendingPathComponent(artifact).appendingPathComponent("infer")
                    let(inference,inferenceResponse)=try await URLSession.shared.data(for:resolve)
                    guard signature==setupSignature else{return}
                    guard (inferenceResponse as? HTTPURLResponse)?.statusCode==200,
                          let body=try JSONSerialization.jsonObject(with:inference) as? [String:Any] else{throw URLError(.badServerResponse)}
                    retainedBody=body;retainedArtifact=artifact
                } else {
                    let reason=resolved["reason"] as? String ?? resolved["error"] as? String ?? ""
                    guard (response as? HTTPURLResponse)?.statusCode==404,["artifact_not_found","route_not_found"].contains(reason) else{throw URLError(.badServerResponse)}
                }
            }
            if let artifact=retainedArtifact,let previous=submissions.record(owner:owner,payload:canonical),previous.jobID==artifact {submissions.update(previous,terminal:true)}
            if resolveOnly && retainedBody==nil && !resumeOutstanding && submissions.record(owner:owner,payload:canonical)?.jobID == nil {return}
            var body:[String:Any]
            if let retained=retainedBody {body=retained}
            else {
            if !retrain,let record=submissions.record(owner:owner,payload:canonical),let existing=record.jobID {
                guard let completed=try await waitForModelJob(existing,signature:signature,record:record) else{return}
                body=completed
            } else {
            guard canRun else {
                runMessage="Training is disabled. A retained model for this exact setup is required."
                return
            }
            guard capabilities?.idempotency == true else {
                runMessage="Model runs are temporarily unavailable. You can still save this setup."
                return
            }
            let record=try submissions.prepare(owner:owner,payload:canonical,retrain:retrain)
            var request=URLRequest(url:CloudConfiguration.labURL.appendingPathComponent("jobs"))
            request.setValue(record.key,forHTTPHeaderField:"Idempotency-Key")
            request.setValue("Bearer " + session.accessToken,forHTTPHeaderField:"Authorization")
            request.httpMethod="POST";request.timeoutInterval=20
            request.setValue("application/json",forHTTPHeaderField:"Content-Type")
            request.httpBody=canonical
            let(data,response)=try await URLSession.shared.data(for:request)
            guard let accepted=try JSONSerialization.jsonObject(with:data) as? [String:Any] else{throw URLError(.cannotParseResponse)}
            guard (response as? HTTPURLResponse)?.statusCode==202,
                  let jobID=accepted["job_id"] as? String,
                  jobID.count==32,jobID.allSatisfy({$0.isHexDigit}) else {
                if signature==setupSignature {runMessage=LabServiceContract.userFacingMessage(accepted)}
                return
            }
            submissions.update(record,jobID:jobID)
            guard signature==setupSignature else{return}
            guard let completed=try await waitForModelJob(jobID,signature:signature,record:record) else{return}
            body=completed
            }
            }
            guard signature==setupSignature else{return}
            runMessage=LabServiceContract.userFacingMessage(body)
            let result=try LabServiceContract.result(body,recipe:recipe)
            runProbability=result.probability;pushProbability=result.push;tieProbability=result.tie;baselineScore=result.baseline
            validationCount=result.count;brierScore=result.brier;validationSummary=result.validation
            if let probability=result.probability,let sport=recipe.sport,let stat=recipe.stat,let target=recipe.targetKind,
               let fingerprint=recipe.configurationFingerprint {
                askModelContext=(recipe,AskPLONNModelResultSnapshot(id:UUID(),ownerID:owner,
                    recipeID:recipe.id,recipeFingerprint:fingerprint,sport:sport,player:recipe.player,
                    stat:stat,targetKind:target,side:recipe.side,threshold:recipe.line,historicalEventID:recipe.historicalEventID,
                    probability:probability,validationSample:result.count,brier:result.brier,
                    mode:body["evaluation_mode"] as? String == "fixed_threshold_retrospective" ? "fixed_threshold_retrospective":"unvalidated",
                    generatedAt:Date()))
            }
            if result.probability != nil || result.count != nil {
                #if DEBUG
                resultContext=context + (retainedBody == nil ? "":" · Retained weights · Frozen historical feature snapshot") + ((body["run_id"] as? String).map {" · Run " + $0} ?? "")
                #else
                resultContext=context + (retainedBody == nil ? "":" · Saved model · Original historical data")
                #endif
            }
        } catch is LabSubmissionStore.StoreError {
            if signature==setupSignature {runMessage="Resume the outstanding run before retraining. Too many outstanding submissions can also prevent a new run."}
        } catch is LabServiceContract.ContractError {
            if signature==setupSignature {runMessage="This result couldn’t be verified. Check the setup and try again."}
        } catch {if signature==setupSignature {runMessage="The model service is unavailable. Check your connection and retry."}}
    }
    @MainActor private func waitForModelJob(_ id:String,signature:String,record:LabSubmissionStore.Record) async throws -> [String:Any]? {
        runMessage="Waiting for training…"
        for attempt in 0..<80 {
            try Task.checkCancellation()
            guard signature==setupSignature else{return nil}
            if attempt>0 {try await Task.sleep(nanoseconds:10_000_000_000)}
            guard signature==setupSignature else{return nil}
            let session=try await account.client.auth.session
            var request=URLRequest(url:CloudConfiguration.labURL.appendingPathComponent("jobs").appendingPathComponent(id))
            request.timeoutInterval=20
            request.setValue("Bearer " + session.accessToken,forHTTPHeaderField:"Authorization")
            let(data,response)=try await URLSession.shared.data(for:request)
            guard signature==setupSignature else{return nil}
            let code=(response as? HTTPURLResponse)?.statusCode
            if code==429 {runMessage="Waiting for an update…";continue}
            guard let job=try JSONSerialization.jsonObject(with:data) as? [String:Any] else{throw URLError(.cannotParseResponse)}
            if code==404,job["reason"] as? String == "job_not_found" {
                LabSubmissionStore().update(record,terminal:true)
                runMessage="This job’s progress has expired. Choose Retrain with new data to start again."
                return nil
            }
            guard code==200 else {
                runMessage="Couldn’t retrieve this model’s progress. Your job may still be running."
                return nil
            }
            switch job["status"] as? String {
            case "completed":
                LabSubmissionStore().update(record,terminal:true)
                guard let result=job["result"] as? [String:Any] else{throw URLError(.cannotParseResponse)}
                return result
            case "failed":
                LabSubmissionStore().update(record,terminal:true)
                if let result=job["result"] as? [String:Any] {return result}
                let reason=job["reason"] as? String ?? ""
                runMessage=reason=="training_interrupted" ? "Training was interrupted. Choose Retrain with new data to try again.":
                    reason=="training_time_limit" ? "Training took too long. Adjust the setup or choose Retrain with new data.":
                    reason=="mlb_provider_request_rejected" ? "The baseball data provider rejected this request. Try again later.":
                    "Training couldn’t finish with the available data. Choose Retrain with new data for a new attempt."
                return nil
            case "running":runMessage="Training your model…"
            case "queued":runMessage="Waiting for training…"
            default:throw URLError(.cannotParseResponse)
            }
        }
        runMessage="This model is taking longer than expected. The server may still be working."
        return nil
    }
    private func apply(_ market:Market) {
        guard !usesHistoricalMatchup, offeredProps.contains(market), LabQuoteSelection.canUsePregameQuote(market) else {
            lookupMessage="This snapshot cannot establish a current pregame quote. Refresh the matchup or enter a research threshold."
            return
        }
        if !market.book.isEmpty {preferredBook=market.book}
        let title=propTitle(market.market)
        if propOptions.contains(title) {targetStat=title} else {targetStat="Custom stat";customStat=title}
        targetLine=market.line.map{String($0)} ?? ""
        side=market.side?.capitalized ?? "Over"
        selectedQuote=market;sourceContext=quoteContext;marketSource=""
    }
    @MainActor private func resolve(_ item:SportsItem,preservingThreshold:Bool=false) async {
        guard !resolving,!usesHistoricalMatchup else{return}
        resolving=true;defer{resolving=false}
        matchedPlayer=item.name;player=item.name;playerTeam=targetKind == "Team" ? item.name:item.subtitle.components(separatedBy:" · ").first ?? ""
        opponent="";marketSource="";marketStore.live=[];quarterback=""
        if !preservingThreshold {targetLine=""}
        selectedQuote=nil;marketLookupContext=""
        let chosenLeague=league
        let lookupContext=quoteContext
        let owner=account.user?.id
        for offset in 0...7 {
            guard quoteContext==lookupContext,account.user?.id==owner,!Task.isCancelled else{return}
            let date=Calendar.current.date(byAdding:.day,value:offset,to:Date())!
            await schedule.load(sport:chosenLeague,view:"Games",date:date)
            guard quoteContext==lookupContext,account.user?.id==owner,!Task.isCancelled else{return}
            guard let game=schedule.items.first(where:{($0.home==playerTeam || $0.away==playerTeam) && $0.state == "pre" && (sportsDate($0.start).map{$0>Date()} ?? false)}) else{continue}
            opponent=game.home==playerTeam ? game.away:game.home
            lookupMessage="\(game.name) · \((sportsDate(game.start)?.formatted(date:.abbreviated,time:.shortened) ?? "Start time unavailable"))"
            
            if targetKind == "Team" {lookupMessage += " · Enter a team prop line from your book.";return}
            let requestContext=quoteContext
            await marketStore.load(chosenLeague,account:account,home:game.home,away:game.away,start:game.start)
            guard quoteContext==requestContext,account.user?.id==owner,!Task.isCancelled else{marketStore.live=[];return}
            guard marketStore.oddsLoadSucceeded,!marketStore.oddsStale else {
                lookupMessage += " · " + marketStore.status
                return
            }
            marketStore.live=marketStore.live.filter { market in
                guard market.sport==chosenLeague,let label=market.eventLabel,
                      label.localizedCaseInsensitiveContains(game.home),label.localizedCaseInsensitiveContains(game.away),
                      let start=market.commence.flatMap(sportsDate),let expected=sportsDate(game.start) else{return false}
                return abs(start.timeIntervalSince(expected))<1800
            }
            marketLookupContext=matchupContext
            let candidates=offeredProps.filter{propTitle($0.market)==statName && $0.side?.lowercased()==side.lowercased() && LabQuoteSelection.canUsePregameQuote($0)}
            if !preservingThreshold, preferredBook != "Ask each time", let market=candidates.sorted(by:{abs($0.price)<abs($1.price)}).first {apply(market)}
            else {lookupMessage += candidates.isEmpty ? " · No verified current quote for this prop and book. \(marketStore.status)":" · Snapshots refreshed. Choose a quoted line explicitly; your research threshold is unchanged."}
            return
        }
        lookupMessage="No upcoming matchup found in the next seven days. No line filled."
    }
    private func entry(_ title:String,placeholder:String,text:Binding<String>)->some View {
        VStack(alignment:.leading,spacing:7) {Text(title).font(.system(.caption, design: .monospaced).weight(.semibold)).foregroundStyle(.secondary);TextField(placeholder,text:text).textFieldStyle(.roundedBorder)}
    }
    private func save() {
        guard account.hasSignedInAccount else {account.access.present(feature:.privateModels);return}
        let recipe=currentRecipe
        if let data=try? JSONEncoder().encode([recipe]+recipes),let text=String(data:data,encoding:.utf8) {savedRecipes=text;saved=true}
    }
}

private struct NationalTeamModelView:View {
    @State private var home=""
    @State private var away=""
    @State private var cutoff=Date()
    @State private var neutral=false
    @State private var modelType="logistic_regression"
    @State private var available:Set<String>=[]
    @State private var running=false
    @State private var message=""
    @State private var result:NationalTeamSetup.Result?
    private var setup:NationalTeamSetup {NationalTeamSetup(home:home,away:away,cutoff:NationalTeamSetup.dateString(cutoff),neutral:neutral,modelType:modelType)}
    var body:some View {
        ScrollView {
            VStack(alignment:.leading,spacing:18) {
                Text("National-team Elo").font(.system(.title2, design: .monospaced).weight(.semibold))
                Text("Experimental retrospective").font(.system(.caption, design: .monospaced).weight(.semibold)).foregroundStyle(.secondary)
                TextField("Home national team",text:$home).textFieldStyle(.roundedBorder)
                TextField("Away national team",text:$away).textFieldStyle(.roundedBorder)
                DatePicker("Historical cutoff",selection:$cutoff,in:...Date(),displayedComponents:.date)
                Toggle("Neutral venue",isOn:$neutral)
                Text("Use team names from the configured national-team history. Cutoff uses UTC; only earlier results enter the model.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                LazyVGrid(columns:[GridItem(.flexible()),GridItem(.flexible())],spacing:8) {
                    ForEach(SavedModelType.allCases) {method in
                        Button {modelType=method.rawValue} label: {
                            Label(method.title,systemImage:modelType == method.rawValue ? "checkmark.circle.fill":"circle")
                                .font(.system(.subheadline, design: .monospaced)).frame(maxWidth:.infinity,minHeight:48,alignment:.leading).padding(8)
                                .plonnGlassSurface(selected:modelType == method.rawValue,radius:12)
                        }.buttonStyle(.plain).disabled(!available.contains(method.rawValue))
                    }
                }
                Button {Task {await train()}} label: {
                    HStack {if running {PLONNLoader()};Text(running ? "Training…":"Train model").font(.system(.headline, design: .monospaced))}.frame(maxWidth:.infinity,minHeight:44)
                }.plonnGlassButton(prominent:true).foregroundStyle(.primary)
                 .disabled(running || !available.contains(modelType) || (try? setup.payload()) == nil)
                if !message.isEmpty {Text(message).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
                if available.isEmpty {Button("Retry connection") {Task {await loadAvailability()}}}
                if let result {
                    HStack(spacing:8) {
                        outcome("Home win",result.home)
                        outcome("Draw",result.draw)
                        outcome("Away win",result.away)
                    }
                    VStack(alignment:.leading,spacing:10) {
                        Text("Historical holdout").font(.system(.headline, design: .monospaced))
                        Text("\(result.count) matches").font(.system(.subheadline, design: .monospaced)).monospacedDigit()
                        Text(String(format:"Brier %.3f",result.brier)).monospacedDigit()
                        Text(result.baseline.map{String(format:"Frequency baseline %.3f",$0)} ?? "Baseline unavailable").monospacedDigit()
                        Text("Lower Brier is better. Baseline uses training-set outcome frequencies.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    }.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(navySurface,in:RoundedRectangle(cornerRadius:14))
                }
                Text("National teams only. Source results may include extra time; these are not 90-minute moneyline or live-match probabilities. Historical tests do not establish betting returns.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            }.padding(20)
        }.background(navy).navigationTitle("National teams").navigationBarTitleDisplayMode(.inline)
         .task {await loadAvailability()}
         .onChange(of:setup) {_,_ in result=nil;message=""}
    }
    private func outcome(_ label:String,_ p:Double)->some View {
        VStack(spacing:8) {Text(label).font(.system(.caption, design: .monospaced));AnimatedProbability(value:p).font(.system(.title3,design:.monospaced,weight:.semibold)).monospacedDigit().minimumScaleFactor(0.7)}
            .frame(maxWidth:.infinity).padding(.vertical,16).background(navySurface,in:RoundedRectangle(cornerRadius:12))
    }
    @MainActor private func loadAvailability() async {
        available=[]
        #if DEBUG && targetEnvironment(simulator)
        do {
            var request=URLRequest(url:URL(string:"http://127.0.0.1:8779/capabilities")!);request.timeoutInterval=10
            let(data,response)=try await URLSession.shared.data(for:request)
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let body=try JSONSerialization.jsonObject(with:data) as? [String:Any],
                  body["enabled"] as? Bool != false,body["service_enabled"] as? Bool != false,
                  let entries=body["optional_team_models"] as? [[String:Any]],
                  let entry=entries.first(where:{$0["context"] as? String == "national" && $0["route"] as? String == "/team-outcome/train-run"}),
                  entry["dataset_configured"] as? Bool == true,
                  let methods=entry["model_types"] as? [String] else {message="National-team history is not connected.";return}
            available=Set(methods).intersection(Set(SavedModelType.allCases.map(\.rawValue)));message=""
        } catch {message="Training service unavailable. Retry the connection."}
        #else
        message="National-team training requires the local simulator service."
        #endif
    }
    @MainActor private func train() async {
        let requested=setup
        guard !running,available.contains(requested.modelType),let payload=try? requested.payload() else{return}
        running=true;result=nil;message=""
        defer {running=false}
        #if DEBUG && targetEnvironment(simulator)
        do {
            var request=URLRequest(url:URL(string:"http://127.0.0.1:8779/team-outcome/train-run")!)
            request.httpMethod="POST";request.timeoutInterval=120;request.setValue("application/json",forHTTPHeaderField:"Content-Type")
            request.httpBody=try JSONSerialization.data(withJSONObject:payload)
            let(data,response)=try await URLSession.shared.data(for:request)
            guard setup==requested else{return}
            guard let body=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {throw URLError(.cannotParseResponse)}
            guard (response as? HTTPURLResponse)?.statusCode == 200,body["status"] as? String == "ok" else {message=body["message"] as? String ?? "Historical training unavailable.";return}
            result=try requested.result(body)
        } catch {if setup==requested {message="No result shown. The service was unavailable or returned an unmatched or invalid result."}}
        #endif
    }
}


private struct LabPlayerPropPicker: View {
    let league:String
    let selection:String
    let capabilities:LabCapabilities?
    let choose:(String)->Void
    @Environment(\.dismiss) private var dismiss
    @State private var search=""
    private var groups:[LabPropCatalog.PropGroup] {LabPropCatalog.playerGroups(league:league,search:search)}
    var body:some View {
        NavigationStack {
            List {
                if groups.isEmpty {Text("No matching props. Try another name or category.").foregroundStyle(.secondary)}
                ForEach(groups) {group in
                    Section(group.title) {
                        ForEach(group.props,id:\.self) {prop in
                            Button {choose(prop);dismiss()} label: {
                                HStack(alignment:.top,spacing:12) {
                                    VStack(alignment:.leading,spacing:5) {
                                        Text(LabPropCatalog.displayName(prop)).foregroundStyle(.primary)
                                        Text(LabPropCatalog.testingStatus(league:league,stat:prop,capabilities:capabilities)).font(.caption).foregroundStyle(.secondary)
                                        if let reason=LabPropCatalog.configurationLimitation(prop) {Text(reason).font(.caption).foregroundStyle(.secondary)}
                                    }
                                    Spacer(minLength:0)
                                    if selection==prop {Image(systemName:"checkmark").foregroundStyle(brandAccent)}
                                }.padding(.vertical,4).contentShape(Rectangle())
                            }.plonnGlassButton(prominent:selection==prop).disabled(LabPropCatalog.configurationLimitation(prop) != nil)
                                .accessibilityIdentifier("lab-prop-"+league+"-"+prop)
                                .accessibilityAddTraits(selection==prop ? .isSelected:[])
                        }
                    }
                }
                Section {
                    Text("For alternate lines or milestones, choose the underlying stat and enter the line on the next step.").font(.caption).foregroundStyle(.secondary)
                }
            }
            .searchable(text:$search,prompt:"Search props or categories")
            .navigationTitle("\(league) player props").navigationBarTitleDisplayMode(.inline)
            .toolbar {ToolbarItem(placement:.cancellationAction) {Button("Cancel") {dismiss()}}}
        }.preferredColorScheme(.dark)
    }
}
