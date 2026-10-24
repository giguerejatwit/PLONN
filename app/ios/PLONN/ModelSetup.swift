import Foundation

struct LabTrainingSettings:Codable,Equatable {
    var epochs=200
    var learningRate=0.05
    var regularization=0.001
    var validationShare=0.2
}
struct LabRecipe: Codable, Identifiable {
    let id:UUID
    var name:String
    let player:String
    let quarterback:String
    let opponent:String
    let features:[String]
    let created:Date
    let line:Double?
    let side:String?
    let sport:String?
    let stat:String?
    let targetKind:String?
    let advanced:Bool?
    let training:LabTrainingSettings?
    var modelType:String? = "logistic_regression"
    var book:String? = nil
    var historicalEventID:String? = nil
    var propOptions:[String:String]? = nil
    var resolvedModelType:String { modelType ?? "logistic_regression" }
    enum CodingKeys: String, CodingKey { case id, name, player, quarterback, opponent, features, line, side, sport, stat, targetKind, advanced, training, modelType, book, historicalEventID, propOptions, created }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(player, forKey: .player)
        try c.encode(quarterback, forKey: .quarterback)
        try c.encode(opponent, forKey: .opponent)
        try c.encode(features, forKey: .features)
        try c.encodeIfPresent(line, forKey: .line)
        try c.encodeIfPresent(side, forKey: .side)
        try c.encodeIfPresent(sport, forKey: .sport)
        try c.encodeIfPresent(stat, forKey: .stat)
        try c.encodeIfPresent(targetKind, forKey: .targetKind)
        try c.encodeIfPresent(advanced, forKey: .advanced)
        try c.encodeIfPresent(training, forKey: .training)
        try c.encodeIfPresent(modelType, forKey: .modelType)
        try c.encodeIfPresent(book, forKey: .book)
        try c.encodeIfPresent(historicalEventID, forKey: .historicalEventID)
        try c.encodeIfPresent(propOptions, forKey: .propOptions)
        try c.encode(created.timeIntervalSinceReferenceDate, forKey: .created)
    }

}


enum SavedModelType: String, CaseIterable, Identifiable, Codable {
    case logistic = "logistic_regression", tree = "decision_tree", forest = "random_forest", neural = "neural_network"
    var id: String { rawValue }
    var title: String {
        switch self {
        case .logistic: return "Logistic regression"
        case .tree: return "Decision tree"
        case .forest: return "Random forest"
        case .neural: return "Neural network"
        }
    }
}
struct LabCapabilities: Decodable {
    struct Support: Decodable {
        let league: String
        let stat: String
        let features: [String]
        let model_types: [String]?
        let evaluation_mode: String?
        var historical_event_required: Bool? = nil
        var target_kind: String? = nil
        var market: String? = nil
        var historical_event_source: String? = nil
        func supports(_ model: String) -> Bool { (model_types ?? ["logistic_regression"]).contains(model) }
    }
    let supported: [Support]
    var supported_data_targets: [Support]? = nil
    var enabled: Bool? = nil
    var service_enabled: Bool? = nil
    var retained_artifacts: Bool? = nil
    var idempotency: Bool? = nil
    func matchRetained(league: String, stat: String, target: String) -> Support? {
        guard retained_artifacts == true else { return nil }
        var available = self
        available.enabled = true; available.service_enabled = true
        return available.match(league: league, stat: stat, target: target)
    }
    func match(league: String, stat: String, target: String) -> Support? {
        guard enabled != false, service_enabled != false, ["Player","Team"].contains(target) else { return nil }
        guard let canonical = LabPropCatalog.serviceStat(league:league,target:target,label:stat) else{return nil}
        return (supported_data_targets ?? supported).first { $0.league == league && $0.stat == canonical && ($0.target_kind ?? "player") == target.lowercased() }
    }
}

/// The guided flow and its submit action use the same availability rules.
enum LabCreationPolicy {
    enum Availability: Equatable {
        case checking, offline, paused, needsUpdate, unsupported, ready
        var title: String {
            switch self {
            case .checking: return "Checking model testing…"
            case .offline: return "Model testing couldn’t connect"
            case .paused: return "Model testing is paused"
            case .needsUpdate: return "Model testing is temporarily unavailable"
            case .unsupported: return "Historical testing isn’t connected for this setup"
            case .ready: return "Ready to test on past games"
            }
        }
        var detail: String {
            switch self {
            case .checking: return "We’re checking which models can run."
            case .ready: return "Your model will be saved privately, then tested using earlier game data. Enough history is required."
            case .unsupported: return "You can save this setup privately. No model test will run for this selection."
            default: return "You can create and save your model now. Run its historical test when the service is available."
            }
        }
    }
    static func availability(_ capabilities: LabCapabilities?, loading: Bool, support: LabCapabilities.Support?) -> Availability {
        if loading { return .checking }
        guard let capabilities else { return .offline }
        guard capabilities.enabled != false, capabilities.service_enabled != false else { return .paused }
        guard capabilities.idempotency == true else { return .needsUpdate }
        return support == nil ? .unsupported : .ready
    }
    static func recommendedFeatures(_ support: LabCapabilities.Support?, fullGame: Bool) -> Set<String> {
        let preferred: Set<String> = fullGame ? ["recent", "season", "homeAway"] : ["recent", "season", "workload"]
        guard let support else { return preferred }
        let connected = preferred.intersection(Set(support.features))
        return connected.isEmpty ? Set(support.features.prefix(3)) : connected
    }
    static func recommendedModel(_ support: LabCapabilities.Support?) -> String {
        SavedModelType.allCases.first { support?.supports($0.rawValue) ?? ($0 == .logistic) }?.rawValue ?? SavedModelType.logistic.rawValue
    }
    static func questions(league: String, target: String, capabilities: LabCapabilities?) -> [String] {
        let teamQuestions = league == "NFL" ? ["Spread", "Game total", "Moneyline"] : ["Moneyline", "Spread", "Game total"]
        let fallback = target == "Team" ? teamQuestions : LabPropCatalog.options(league: league, target: target).filter { $0 != "Custom stat" }
        guard target == "Team", let capabilities else { return fallback }
        let connected = (capabilities.supported_data_targets ?? capabilities.supported).filter {
            $0.league == league && ($0.target_kind ?? "player") == target.lowercased()
        }.map(\.stat)
        return fallback.filter(connected.contains)
    }
    static func question(_ stat: String, target: String) -> String {
        guard target == "Team" else { return LabPropCatalog.isYesNo(stat) ? LabPropCatalog.displayName(stat):"Over or under \(LabPropCatalog.displayName(stat).lowercased())?" }
        switch stat {
        case "Moneyline": return "Would this team win?"
        case "Spread": return "Would this team cover?"
        case "Game total": return "Would the total go over or under?"
        default: return stat
        }
    }
}
extension LabRecipe {
    // Version identity excludes device ID and creation time; all executable setup fields remain.
    var configurationFingerprint: Data? {
        guard let data = try? JSONEncoder().encode(self),
              var object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        object.removeValue(forKey: "id"); object.removeValue(forKey: "created")
        object["modelType"] = resolvedModelType
        return try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }
}


// A successful result must identify the exact requested adapter and feature set.
func labRunMatches(_ body:[String:Any],modelType:String,features:Set<String>,targetKind:String = "Player") -> Bool {
    guard let model=body["model"] as? [String:Any],model["model_type"] as? String == modelType,
          let inference=body["inference"] as? [String:Any],let returned=inference["features"] as? [String:Any] else{return false}
    let expected = targetKind == "Team" ? Set(features.flatMap { feature -> [String] in
        switch feature {
        case "recent": return ["recent_scored","recent_allowed"]
        case "season": return ["season_scored","season_allowed"]
        default: return [feature]
        }
    }) : features
    return Set(returned.keys)==expected
}


// Single source for native configuration choices, not proof of training coverage.
enum LabPropCatalog {
    #if SWIFT_PACKAGE
    // The Foundation-only package excludes the SwiftUI app entry point.
    static let leagues = ["NBA", "MLB", "NFL", "NHL"]
    #else
    static let leagues = LaunchSports.supported
    #endif
    static let targets = ["Player", "Team"]
    static let basicFeatureIDs:Set<String> = ["recent", "season", "workload", "opponent", "homeAway", "rest"]
    static func options(league:String,target:String) -> [String] {
        guard targets.contains(target) else{return []}

        if target == "Team" {
            switch league {
            case "NBA":return ["Game total","Spread","Moneyline","Team points","Team rebounds","Team assists","Team 3-pointers","Team turnovers","Custom stat"]
            case "MLB":return ["Game total","Spread","Moneyline","Team runs","Team hits","Team home runs","Team strikeouts","Custom stat"]
            case "NHL":return ["Game total","Spread","Moneyline","Team goals","Team shots on goal","Team saves","Custom stat"]
            case "NFL":return ["Game total","Spread","Moneyline","Team points","Team rush yards","Team pass yards","Team total yards","Team sacks","Team turnovers","Custom stat"]
            default:return []
            }
        }
        switch league {
        case "NBA":return ["Points","Rebounds","Assists","3-pointers","Steals","Blocks","Turnovers","Pts + Reb + Ast","Pts + Reb","Pts + Ast","Reb + Ast","Field goals made","Free throws made","Free throw attempts","Blocks + Steals","Double-double","Triple-double","First basket scorer","First team basket scorer","1st-quarter points","1st-quarter rebounds","1st-quarter assists","Fantasy points","Method of first basket","Custom stat"]
        case "MLB":return ["Hits","Total bases","Home runs","RBI","Runs","Stolen bases","Walks","Batter strikeouts","Pitcher strikeouts","Pitcher outs","Earned runs","Hits allowed","Walks allowed","Singles","Doubles","Triples","Hits + Runs + RBI","First home run scorer","Pitcher win","Batter fantasy points","Custom stat"]
        case "NHL":return ["Shots on goal","Goals","Assists","Points","Blocked shots","Saves","Goals allowed","Power play points","Faceoffs won","Time on ice","Anytime goal scorer","First goal scorer","Last goal scorer","Custom stat"]
        case "NFL":return ["Rec Yds","Receptions","Rush Yds","Pass Yds","Pass TDs","Interceptions thrown","Rush + Rec Yds","Rush attempts","Pass attempts","Completions","Touchdowns","Tackles","Sacks","Longest pass completion","1st-quarter passing yards","Receiving touchdowns","Longest reception","Rushing touchdowns","Longest rush","Pass + Rush Yds","Pass + Rush + Rec Yds","Rush + Rec TDs","Pass + Rush + Rec TDs","Anytime touchdown scorer","First touchdown scorer","Last touchdown scorer","Solo tackles","Assisted tackles","Tackles + Assists","Defensive interceptions","Field goals made","Kicking points","Extra points made","Custom stat"]
        default:return []
        }
    
    }
    struct PropGroup: Identifiable {
        let title:String
        let props:[String]
        var id:String {title}
    }
    static func playerGroups(league:String, search:String="") -> [PropGroup] {
        let groups:[PropGroup]
        switch league {
        case "NBA": groups=[
            .init(title:"Scoring",props:["Points","3-pointers","Field goals made","Free throws made","Free throw attempts"]),
            .init(title:"Rebounds & assists",props:["Rebounds","Assists"]),
            .init(title:"Defense & turnovers",props:["Steals","Blocks","Blocks + Steals","Turnovers"]),
            .init(title:"Combinations",props:["Pts + Reb + Ast","Pts + Reb","Pts + Ast","Reb + Ast"]),
            .init(title:"Milestones & scorers",props:["Double-double","Triple-double","First basket scorer","First team basket scorer"]),
            .init(title:"First quarter",props:["1st-quarter points","1st-quarter rebounds","1st-quarter assists"]),
            .init(title:"Special formats",props:["Fantasy points","Method of first basket"])
        ]
        case "NFL": groups=[
            .init(title:"Passing",props:["Pass Yds","Pass TDs","Pass attempts","Completions","Interceptions thrown","Longest pass completion","1st-quarter passing yards"]),
            .init(title:"Receiving",props:["Rec Yds","Receptions","Receiving touchdowns","Longest reception"]),
            .init(title:"Rushing",props:["Rush Yds","Rush attempts","Rushing touchdowns","Longest rush"]),
            .init(title:"Combinations",props:["Rush + Rec Yds","Pass + Rush Yds","Pass + Rush + Rec Yds","Rush + Rec TDs","Pass + Rush + Rec TDs"]),
            .init(title:"Touchdowns",props:["Touchdowns","Anytime touchdown scorer","First touchdown scorer","Last touchdown scorer"]),
            .init(title:"Defense",props:["Tackles","Solo tackles","Assisted tackles","Tackles + Assists","Sacks","Defensive interceptions"]),
            .init(title:"Kicking",props:["Field goals made","Kicking points","Extra points made"])
        ]
        case "MLB": groups=[
            .init(title:"Hitting",props:["Hits","Total bases","Home runs","RBI","Singles","Doubles","Triples","Walks","Batter strikeouts","Hits + Runs + RBI"]),
            .init(title:"Baserunning",props:["Runs","Stolen bases"]),
            .init(title:"Pitching",props:["Pitcher strikeouts","Pitcher outs","Earned runs","Hits allowed","Walks allowed"]),
            .init(title:"Milestones",props:["First home run scorer","Pitcher win"]),
            .init(title:"Special formats",props:["Batter fantasy points"])
        ]
        case "NHL": groups=[
            .init(title:"Skaters",props:["Shots on goal","Goals","Assists","Points","Power play points","Blocked shots","Faceoffs won","Time on ice"]),
            .init(title:"Goalies",props:["Saves","Goals allowed"]),
            .init(title:"Goal scorers",props:["Anytime goal scorer","First goal scorer","Last goal scorer"])
        ]
        default: groups=[]
        }
        let query=search.trimmingCharacters(in:.whitespacesAndNewlines)
        guard !query.isEmpty else{return groups}
        return groups.compactMap {group in
            let props=group.props.filter { displayName($0).localizedCaseInsensitiveContains(query) || $0.localizedCaseInsensitiveContains(query) || group.title.localizedCaseInsensitiveContains(query) }
            return props.isEmpty ? nil:PropGroup(title:group.title,props:props)
        }
    }
    static func displayName(_ stat:String) -> String {
        ["Rec Yds":"Receiving yards","Rush Yds":"Rushing yards","Pass Yds":"Passing yards","Pass TDs":"Passing touchdowns",
         "Pass attempts":"Passing attempts","Rush attempts":"Rushing attempts","Completions":"Passing completions",
         "Rush + Rec Yds":"Rushing + receiving yards","Pass + Rush Yds":"Passing + rushing yards",
         "Pass + Rush + Rec Yds":"Passing + rushing + receiving yards","Rush + Rec TDs":"Rushing + receiving touchdowns",
         "Pass + Rush + Rec TDs":"Passing + rushing + receiving touchdowns",
         "Pts + Reb + Ast":"Points + rebounds + assists","Pts + Reb":"Points + rebounds","Pts + Ast":"Points + assists","Reb + Ast":"Rebounds + assists",
         "RBI":"Runs batted in","Hits + Runs + RBI":"Hits + runs + RBIs","Tackles":"Total tackles" ][stat] ?? stat
    }
    static func isYesNo(_ stat:String) -> Bool {
        ["Double-double","Triple-double","First basket scorer","First team basket scorer","Method of first basket","Anytime touchdown scorer","First touchdown scorer","Last touchdown scorer","First home run scorer","Pitcher win","Anytime goal scorer","First goal scorer","Last goal scorer"].contains(stat)
    }
    static func configurationLimitation(_ stat:String) -> String? { nil }
    static func isFantasy(_ stat:String) -> Bool {["Fantasy points","Batter fantasy points"].contains(stat)}
    static func validOptions(stat:String,options:[String:String]) -> Bool {
        if isFantasy(stat) {return Set(options.keys)==["scoring_preset"] && ["draftkings_classic","fanduel_classic"].contains(options["scoring_preset"] ?? "")}
        if stat == "Method of first basket" {return Set(options.keys)==["first_basket_method"] && ["two_pointer","three_pointer","free_throw"].contains(options["first_basket_method"] ?? "")}
        return options.isEmpty
    }
    static func optionsLabel(_ options:[String:String]?) -> String? {
        if let preset=options?["scoring_preset"] {return preset == "draftkings_classic" ? "DraftKings Classic scoring":"FanDuel Classic scoring"}
        if let method=options?["first_basket_method"] {return ["two_pointer":"Two-pointer","three_pointer":"Three-pointer","free_throw":"Free throw"][method]}
        return nil
    }
    static func propNote(_ stat:String) -> String? {
        if ["First basket scorer","First team basket scorer","Method of first basket"].contains(stat) {return "First score includes free throws. Method tests whether this player scores first using the selected method."}
        if ["Anytime touchdown scorer","First touchdown scorer","Last touchdown scorer","Touchdowns"].contains(stat) {return "Counts touchdowns scored by this player, including returns. Passing touchdowns belong to the receiver."}
        if stat == "Time on ice" {return "Enter the line in minutes. For example, 20.5 means 20 minutes, 30 seconds."}
        if stat == "Sacks" {return "Half-sacks count. You can enter a quarter-point line, such as 0.75."}
        return nil
    }
    static func testingStatus(league:String,stat:String,capabilities:LabCapabilities?) -> String {
        if configurationLimitation(stat) != nil {return "Not configurable"}
        guard let capabilities else{return "Availability unverified"}
        if capabilities.enabled == false || capabilities.service_enabled == false {return "Testing paused"}
        return capabilities.idempotency == true && capabilities.match(league:league,stat:stat,target:"Player") != nil ? "Testing available":"Save setup only"
    }
    static func serviceStat(league:String,target:String,label:String) -> String? {
        guard label != "Custom stat" else{return nil}
        return label
    }
}

// UI and transport share these guards; capabilities are availability, not validation evidence.
enum LabServiceContract {
    /// Server diagnostics can contain paths, identifiers or provider details.
    /// Only recognized failure codes become customer-facing explanations.
    static func userFacingMessage(_ body: [String: Any]) -> String {
        let inference = body["inference"] as? [String: Any] ?? [:]
        if inference["status"] as? String == "ok" { return "Model result ready." }
        let reason = body["reason"] as? String
            ?? (body["training"] as? [String: Any])?["reason"] as? String
            ?? inference["reason"] as? String
            ?? (body["validation"] as? [String: Any])?["reason"] as? String
        switch reason {
        case "authentication_required", "authenticated_account_required", "invalid_token": return "Sign in again to run this model."
        case "book_required": return "Choose a sportsbook before running this setup."
        case "training_dataset_missing", "verified_target_missing", "insufficient_training_rows", "insufficient_holdout_rows":
            return "There isn’t enough verified historical data for this setup. Adjust the selection and try again."
        case "historical_player_target_missing": return "This player has no verified appearance in the selected game. Check the player and matchup."
        case "completed_regular_season_required": return "Choose a completed regular-season game for this test."
        case "incomplete_player_event_history": return "The play-by-play history is incomplete for this player and prop. No estimate was produced. Try another player or game."
        case "insufficient_history_minimum_40_feature_complete_starts", "insufficient_training_or_holdout_rows", "insufficient_prior_starts": return "There aren’t enough verified player appearances to train and test this setup. Choose a player with more history or a later game."
        case "insufficient_training_class_coverage": return "This player’s history doesn’t contain enough different outcomes for this line. Try another line or player."
        case "prop_options_required": return "Choose a scoring preset or scoring method before testing."
        case "stale_target": return "The selected matchup data has expired. Refresh it before running."
        case "game_not_completed_historical_target": return "This game is too recent for historical testing. Choose an earlier completed game."
        case "insufficient_game_training_class_coverage_tie": return "There aren’t enough historical ties to test this NFL win model. Try a spread or game total."
        case "insufficient_game_training_or_holdout_rows": return "There isn’t enough verified game history for this setup. Choose another matchup."
        case "adapter_not_supported", "unsupported_model", "unsupported_model_type", "unsupported_hosted_coverage", "unsupported_features", "advanced_access_required":
            return "This model or feature selection is unavailable. Adjust the setup and try again."
        case "training_interrupted": return "Training was interrupted. Choose Retrain with new data to try again."
        case "training_time_limit": return "Training took too long. Adjust the setup or retry later."
        case "training_worker_failed": return "The model service stopped before finishing. Try again with the same selection."
        case "rate_limit", "quota_exceeded", "owner_busy", "training_capacity_busy", "training_quota_exceeded":
            return "The model service is busy or your run limit has been reached. Try again later."
        case "hosted_training_disabled": return "Model runs are temporarily unavailable. You can still save this setup."
        default:
            if reason?.hasPrefix("insufficient_game_training_class_coverage_") == true {
                return "There aren’t enough varied historical results to test this line. Try another line or matchup."
            }
            return "The model service couldn’t complete this setup. Check your selection and try again later."
        }
    }
    static func gameMarket(target:String?,stat:String?) -> String? {
        guard target == "Team" else{return nil}
        return ["Game total":"total","Spread":"spread","Moneyline":"moneyline"][stat ?? ""]
    }
    static func validLine(_ line:Double?,target:String?,stat:String?) -> Bool {
        if target == "Player",LabPropCatalog.isYesNo(stat ?? "") {return line == 0.5}
        let market=gameMarket(target:target,stat:stat)
        if market == "moneyline" {return line == nil || line == 0}
        guard let line,line.isFinite else{return false}
        let precision = market == nil ? 100.0:2.0
        guard abs(line * precision - (line * precision).rounded()) < 0.000001 else{return false}
        return market == "spread" ? (-1000...1000).contains(line) : (0...(market == nil ? 10000.0:1000.0)).contains(line)
    }
    static func probabilityLabel(_ recipe:LabRecipe) -> String {
        if recipe.targetKind == "Player",LabPropCatalog.isYesNo(recipe.stat ?? "") {
            return "Historical estimate · \(LabPropCatalog.displayName(recipe.stat ?? "")): \(recipe.side == "Under" ? "No":"Yes")"
        }
        switch gameMarket(target:recipe.targetKind,stat:recipe.stat) {
        case "moneyline":return "Historical estimate · \(recipe.player) to win"
        case "spread":return "Historical estimate · \(recipe.player) to cover \(recipe.line.map { String(format:"%+.1f",$0) } ?? "")"
        case "total":return "Historical estimate · Combined score \((recipe.side ?? "Over").lowercased()) \(recipe.line.map(String.init(describing:)) ?? "")"
        default:return "\((recipe.historicalEventID ?? "").isEmpty ? "Probability of finishing":"Historical estimate of finishing") \((recipe.side ?? "Over").lowercased()) \(recipe.line.map(String.init(describing:)) ?? "") \(recipe.stat ?? "")"
        }
    }
    static func defaultFeatures(_ support: LabCapabilities.Support?) -> Set<String> {
        let preferred: Set<String> = ["recent", "season", "workload"]
        return support.map { preferred.intersection(Set($0.features)) } ?? preferred
    }
    static func payload(_ recipe: LabRecipe, support: LabCapabilities.Support) throws -> [String:Any] {
        let game=gameMarket(target:recipe.targetKind,stat:recipe.stat)
        guard support.supports(recipe.resolvedModelType),
              (support.target_kind ?? "player") == (recipe.targetKind ?? "Player").lowercased(),
              recipe.targetKind != "Team" || game != nil,
              recipe.sport == support.league, recipe.stat == support.stat,
              !recipe.features.isEmpty, recipe.features.count <= 5,
              Set(recipe.features).count == recipe.features.count,
              Set(recipe.features).isSubset(of:Set(support.features)),
              validLine(recipe.line,target:recipe.targetKind,stat:recipe.stat),
              LabPropCatalog.validOptions(stat:recipe.stat ?? "",options:recipe.propOptions ?? [:]),
              let side=recipe.side,["Over","Under"].contains(side),
              let book=recipe.book,!book.isEmpty,book != "Ask each time",
              !recipe.player.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
              !recipe.opponent.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {throw ContractError.invalidSetup}
        if game != nil || support.historical_event_required == true || recipe.sport == "NFL" {
            guard let event=recipe.historicalEventID,!event.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
                  event.count <= 80 else {throw ContractError.invalidSetup}
            if game != nil && (event.count > 15 || !event.allSatisfy({$0.isASCII && $0.isNumber})) {throw ContractError.invalidSetup}
        }
        let settings=recipe.training ?? LabTrainingSettings()
        var body:[String:Any] = ["model_type":recipe.resolvedModelType,"league":support.league,"stat":support.stat,
            "player":recipe.player,"opponent":recipe.opponent,"book":book,"line":recipe.line ?? 0,"side":side,"features":recipe.features.sorted(),
            "training":["epochs":settings.epochs,"learning_rate":settings.learningRate,"l2":settings.regularization,"validation_fraction":settings.validationShare]]
        if recipe.targetKind == "Team" {body["targetKind"]="Team"}
        if let options=recipe.propOptions,!options.isEmpty {body["prop_options"]=options}
        if let event=recipe.historicalEventID,!event.isEmpty {body["historical_event_id"]=event}
        if let mode=support.evaluation_mode {body["evaluation_mode"]=mode}
        return body
    }
    enum ContractError:Error {case invalidSetup, mismatchedResult, invalidProbabilities, invalidValidation}
    struct Result {
        let probability:Double?
        let push:Double?
        let tie:Double?
        let baseline:Double?
        let count:Int?
        let brier:Double?
        let validation:String
    }
    static func result(_ body:[String:Any],recipe:LabRecipe) throws -> Result {
        let inference=body["inference"] as? [String:Any] ?? [:]
        let validation=body["validation"] as? [String:Any] ?? [:]
        let predicts=inference["status"] as? String == "ok"
        let validates=validation["status"] as? String == "ok"
        if predicts || validates {
            guard labRunMatches(body,modelType:recipe.resolvedModelType,features:Set(recipe.features),targetKind:recipe.targetKind ?? "Player") else {throw ContractError.mismatchedResult}
            if let event=recipe.historicalEventID,!event.isEmpty {
                guard
                      let target=inference["target"] as? [String:Any],target["event_id"] as? String == event,
                      body["live_prediction"] as? Bool == false else {throw ContractError.mismatchedResult}
            }
            if let options=recipe.propOptions,!options.isEmpty,body["prop_options"] as? [String:String] != options {throw ContractError.mismatchedResult}
            if let prop=body["target_prop"] as? String,prop != recipe.stat {throw ContractError.mismatchedResult}
            if let threshold=inference["threshold"] as? Double,threshold != recipe.line {throw ContractError.mismatchedResult}
            if let book=body["requested_book"] as? String,book != recipe.book {throw ContractError.mismatchedResult}
        }
        let game=gameMarket(target:recipe.targetKind,stat:recipe.stat)
        var probability:Double?,push:Double?,tie:Double?,count:Int?,brier:Double?,baselineValue:Double?
        var comparison=""
        if predicts {
            if let game {
                guard body["target_kind"] as? String == "game",body["market"] as? String == game,
                      let target=inference["target"] as? [String:Any],target["league"] as? String == recipe.sport,
                      let selected=target["selected_team"] as? String,["home","away"].contains(selected) else{throw ContractError.mismatchedResult}
                let other=selected == "home" ? "away":"home"
                let selectedNames=[target[selected+"_team"],target[selected+"_team_name"],target[selected+"_team_id"]].compactMap{$0 as? String}.map{$0.lowercased()}
                let opponentNames=[target[other+"_team"],target[other+"_team_name"],target[other+"_team_id"]].compactMap{$0 as? String}.map{$0.lowercased()}
                guard selectedNames.contains(recipe.player.lowercased()),opponentNames.contains(recipe.opponent.lowercased()),
                      body["live_prediction"] as? Bool == false,
                      body["evaluation_mode"] as? String == "fixed_threshold_retrospective" else{throw ContractError.mismatchedResult}
                if game == "moneyline" {
                    guard inference["threshold"] is NSNull,body["line"] is NSNull else{throw ContractError.mismatchedResult}
                } else {
                    guard inference["threshold"] as? Double == recipe.line,body["line"] as? Double == recipe.line else{throw ContractError.mismatchedResult}
                }
            }
            let required:Set<String>
            let selection:String
            if game == "moneyline" {required=recipe.sport == "NFL" ? ["win","loss","tie"]:["win","loss"];selection="win"}
            else if game == "spread" {required=(recipe.line ?? 0).truncatingRemainder(dividingBy:1)==0 ? ["cover","not_cover","push"]:["cover","not_cover"];selection="cover"}
            else if game == "total" {required=(recipe.line ?? 0).truncatingRemainder(dividingBy:1)==0 ? ["over","under","push"]:["over","under"];selection=(recipe.side ?? "").lowercased()}
            else {required=["over","under"];selection=(recipe.side ?? "").lowercased()}
            guard let probabilities=inference["probabilities"] as? [String:Double],
                  required.isSubset(of:Set(probabilities.keys)),
                  Set(probabilities.keys).isSubset(of:game == nil ? ["over","under","push"]:required),
                  probabilities.values.allSatisfy({$0.isFinite && (0...1).contains($0)}),
                  abs(probabilities.values.reduce(0,+)-1)<0.00001 else {throw ContractError.invalidProbabilities}
            probability=probabilities[selection];push=probabilities["push"];tie=probabilities["tie"]
        }
        if validates {
            guard let n=validation["sample_size"] as? Int,n>0,
                  let metrics=validation["metrics"] as? [String:Any],let score=metrics["multiclass_brier"] as? Double,
                  score.isFinite,(0...2).contains(score) else {throw ContractError.invalidValidation}
            count=n;brier=score
            if let baseline=metrics["baseline"] as? [String:Any],
               let baselineScore=baseline["multiclass_brier"] as? Double,baselineScore.isFinite,(0...2).contains(baselineScore) {
                baselineValue=baselineScore
                comparison = score < baselineScore ? " Lower error than the simple baseline in this test; still experimental." :
                    " Did not beat the simple baseline in this test. Experimental only."
            } else {comparison=" Baseline comparison unavailable. Experimental only."}
        }
        if let provenance=body["provenance"] as? [String:Any],let coverage=provenance["event_history_coverage"] as? [String:Any],
           let verified=coverage["verified_games"] as? Int,let total=coverage["window_games"] as? Int,total>0,verified<=total {
            comparison += " Verified event details for \(verified) of \(total) prior games."
        }
        return Result(probability:probability,push:push,tie:tie,baseline:baselineValue,count:count,brier:brier,
            validation:validates ? ((validation["label"] as? String ?? "Held-out validation reported; profitability is not established.") + comparison) : "Validation unavailable: " + (validation["reason"] as? String ?? "not supplied"))
    }
}

struct AccountModelVersion: Codable, Identifiable {
    let id: UUID
    let model_id: UUID
    let user_id: UUID
    let version: Int
    let name: String
    let configuration: LabRecipe
    let created_at: String
}

struct NationalTeamSetup:Equatable {
    var home="",away="",cutoff="",neutral=false,modelType="logistic_regression"
    static let mode="retrospective_result_date_only"
    static func dateString(_ date:Date)->String {
        let formatter=DateFormatter();formatter.calendar=Calendar(identifier:.gregorian)
        formatter.locale=Locale(identifier:"en_US_POSIX");formatter.timeZone=TimeZone(secondsFromGMT:0)
        formatter.dateFormat="yyyy-MM-dd";return formatter.string(from:date)
    }
    func payload(now:Date=Date()) throws -> [String:Any] {
        let h=home.trimmingCharacters(in:.whitespacesAndNewlines),a=away.trimmingCharacters(in:.whitespacesAndNewlines)
        let formatter=DateFormatter();formatter.calendar=Calendar(identifier:.gregorian)
        formatter.locale=Locale(identifier:"en_US_POSIX");formatter.timeZone=TimeZone(secondsFromGMT:0);formatter.dateFormat="yyyy-MM-dd";formatter.isLenient=false
        guard !h.isEmpty,!a.isEmpty,h.count<=100,a.count<=100,h.caseInsensitiveCompare(a) != .orderedSame,
              let date=formatter.date(from:cutoff),Self.dateString(date)==cutoff,cutoff<=Self.dateString(now),
              SavedModelType(rawValue:modelType) != nil else {throw LabServiceContract.ContractError.invalidSetup}
        return ["sport":"Soccer","context":"national","target_kind":"team","evaluation_mode":Self.mode,
                "home":h,"away":a,"cutoff":cutoff,"neutral":neutral,"model_type":modelType]
    }
    struct Result {let home:Double,draw:Double,away:Double,count:Int,brier:Double,baseline:Double?}
    func result(_ body:[String:Any]) throws -> Result {
        guard body["status"] as? String == "ok",body["evaluation_mode"] as? String == Self.mode,
              let model=body["model"] as? [String:Any],model["model_type"] as? String == modelType,
              model["adapter"] as? String == "soccer_national_team_elo_v1",
              let inference=body["inference"] as? [String:Any],inference["status"] as? String == "ok",inference["live_prediction"] as? Bool == false,
              let target=inference["target"] as? [String:Any],target["home"] as? String == home.trimmingCharacters(in:.whitespacesAndNewlines),
              target["away"] as? String == away.trimmingCharacters(in:.whitespacesAndNewlines),target["cutoff"] as? String == cutoff,target["neutral"] as? Bool == neutral,
              let p=inference["probabilities"] as? [String:Double],Set(p.keys)==Set(["home_win","draw","away_win"]),
              p.values.allSatisfy({$0.isFinite && (0...1).contains($0)}),abs(p.values.reduce(0,+)-1)<1e-8,
              let validation=body["validation"] as? [String:Any],validation["status"] as? String == "ok",
              let count=validation["sample_size"] as? Int,count>0,let metrics=validation["metrics"] as? [String:Any],
              let brier=metrics["multiclass_brier"] as? Double,brier.isFinite,(0...2).contains(brier)
        else {throw LabServiceContract.ContractError.mismatchedResult}
        var baseline:Double?
        if let value=metrics["baseline"] as? [String:Any],value["status"] as? String == "ok" {
            guard let score=value["multiclass_brier"] as? Double,score.isFinite,(0...2).contains(score) else {throw LabServiceContract.ContractError.invalidValidation}
            baseline=score
        }
        return Result(home:p["home_win"]!,draw:p["draw"]!,away:p["away_win"]!,count:count,brier:brier,baseline:baseline)
    }
}

extension LabRecipe {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        player = try c.decode(String.self, forKey: .player)
        quarterback = try c.decode(String.self, forKey: .quarterback)
        opponent = try c.decode(String.self, forKey: .opponent)
        features = try c.decode([String].self, forKey: .features)
        line = try c.decodeIfPresent(Double.self, forKey: .line)
        side = try c.decodeIfPresent(String.self, forKey: .side)
        sport = try c.decodeIfPresent(String.self, forKey: .sport)
        stat = try c.decodeIfPresent(String.self, forKey: .stat)
        targetKind = try c.decodeIfPresent(String.self, forKey: .targetKind)
        advanced = try c.decodeIfPresent(Bool.self, forKey: .advanced)
        training = try c.decodeIfPresent(LabTrainingSettings.self, forKey: .training)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType)
        book = try c.decodeIfPresent(String.self, forKey: .book)
        historicalEventID = try c.decodeIfPresent(String.self, forKey: .historicalEventID)
        propOptions = try c.decodeIfPresent([String:String].self, forKey: .propOptions)
        created = Date(timeIntervalSinceReferenceDate: try c.decode(Double.self, forKey: .created))
    }
}

// A saved recipe stores a research threshold, never evidence of a current quote.
enum LabQuoteSelection {
    static func offers(_ markets: [Market], league: String, player: String, book: String) -> [Market] {
        markets.filter {
            $0.mode == "live" && $0.sport == league && $0.kind == "prop" &&
            BuilderLogic.normalized($0.name) == BuilderLogic.normalized(player) &&
            $0.line?.isFinite == true && $0.eventState != "post"
        }.compactMap { market in
            var selected = market
            if book != "Ask each time", BuilderLogic.normalized(market.book) != BuilderLogic.normalized(book) {
                // Legacy secondary quotes carry no timestamp. Never borrow the primary book's time.
                guard let quote = market.quotes.first(where: { BuilderLogic.normalized($0.book) == BuilderLogic.normalized(book) }) else { return nil }
                selected.book = quote.book; selected.price = quote.price; selected.updated = nil
            }
            return BuilderLogic.decimalOdds(selected.price) == nil ? nil : selected
        }
    }

    static func canUsePregameQuote(_ market: Market, now: Date = Date()) -> Bool {
        guard market.mode == "live", !market.book.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !market.event.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              ["over", "under"].contains(market.side?.lowercased() ?? ""),
              market.line?.isFinite == true, BuilderLogic.decimalOdds(market.price) != nil,
              BuilderLogic.quoteAge(market, now: now) == .recent,
              market.eventState != "in", market.eventState != "post",
              let start = market.commence.flatMap(sportsDate), start > now else { return false }
        return true
    }
}

// Persist only request identity/progress, never authentication tokens or model weights.
// Outstanding requests cannot be evicted: losing a key could duplicate a timed-out POST.
final class LabSubmissionStore {
    struct Record: Codable, Equatable {
        let owner: UUID
        let payload: Data
        let key: String
        var jobID: String?
        var terminal = false
    }
    enum StoreError: Error { case capacity, pendingRetrain }
    private let defaults: UserDefaults
    private let storageKey = "labSubmissionRecords.v1"
    private let capacity: Int
    init(defaults: UserDefaults = .standard, capacity: Int = 100) {
        self.defaults = defaults; self.capacity = capacity
    }
    private var records: [Record] {
        get { defaults.data(forKey: storageKey).flatMap { try? JSONDecoder().decode([Record].self, from: $0) } ?? [] }
        set { if let data = try? JSONEncoder().encode(newValue) { defaults.set(data, forKey: storageKey) } }
    }
    static func canonical(_ payload: [String:Any]) throws -> Data {
        var normalized = payload
        if let features = payload["features"] as? [String] { normalized["features"] = features.sorted() }
        return try JSONSerialization.data(withJSONObject: normalized, options: [.sortedKeys])
    }
    func record(owner: UUID, payload: Data) -> Record? {
        records.first { $0.owner == owner && $0.payload == payload }
    }
    func deleteAccountRecords(owner: UUID) {
        records = records.filter { $0.owner != owner }
    }
    func forgetCompleted(owner: UUID, payload: Data) {
        records = records.filter { !($0.owner == owner && $0.payload == payload && $0.terminal) }
    }
    func hasOutstanding(owner: UUID, payload: Data) -> Bool {
        record(owner: owner, payload: payload)?.terminal == false
    }
    func prepare(owner: UUID, payload: Data, retrain: Bool = false) throws -> Record {
        var all = records
        if let index = all.firstIndex(where: { $0.owner == owner && $0.payload == payload }) {
            if !retrain { return all[index] }
            guard all[index].terminal else { throw StoreError.pendingRetrain }
            all.remove(at: index)
        }
        if all.count >= capacity {
            guard let index = all.firstIndex(where: { $0.terminal }) else { throw StoreError.capacity }
            all.remove(at: index)
        }
        let value = Record(owner: owner, payload: payload, key: UUID().uuidString)
        all.append(value); records = all; return value
    }
    func update(_ record: Record, jobID: String? = nil, terminal: Bool = false) {
        var all = records
        guard let index = all.firstIndex(where: { $0.key == record.key && $0.owner == record.owner }) else { return }
        if let jobID { all[index].jobID = jobID }
        all[index].terminal = terminal; records = all
    }
}

/// Shared name matching for model selectors. Every typed word must match a
/// player/team label; accents, case and extra spaces do not affect selection.
enum LabNameSearch {
    static func normalized(_ text:String)->String {
        text.folding(options:[.caseInsensitive,.diacriticInsensitive],locale:Locale(identifier:"en_US_POSIX"))
            .split(whereSeparator:{$0.isWhitespace}).joined(separator:" ")
    }
    static func matches(query:String,name:String,detail:String="")->Bool {
        let terms=normalized(query).split(separator:" ")
        let target=normalized(name+" "+detail)
        return !terms.isEmpty && terms.allSatisfy {target.contains($0)}
    }
    static func rank(query:String,name:String)->String {
        let q=normalized(query),n=normalized(name)
        return (n==q ? "0":n.hasPrefix(q) ? "1":"2")+n
    }
}
