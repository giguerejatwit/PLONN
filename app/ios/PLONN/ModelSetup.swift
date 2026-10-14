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
    var resolvedModelType:String { modelType ?? "logistic_regression" }
    enum CodingKeys: String, CodingKey { case id, name, player, quarterback, opponent, features, line, side, sport, stat, targetKind, advanced, training, modelType, book, historicalEventID, created }
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
        case "NBA":return ["Points","Rebounds","Assists","3-pointers","Steals","Blocks","Turnovers","Pts + Reb + Ast","Pts + Reb","Pts + Ast","Reb + Ast","Custom stat"]
        case "MLB":return ["Hits","Total bases","Home runs","RBI","Runs","Stolen bases","Walks","Batter strikeouts","Pitcher strikeouts","Pitcher outs","Earned runs","Hits allowed","Walks allowed","Custom stat"]
        case "NHL":return ["Shots on goal","Goals","Assists","Points","Blocked shots","Saves","Goals allowed","Custom stat"]
        case "NFL":return ["Rec Yds","Receptions","Rush Yds","Pass Yds","Pass TDs","Interceptions thrown","Rush + Rec Yds","Rush attempts","Pass attempts","Completions","Touchdowns","Tackles","Sacks","Custom stat"]
        default:return []
        }
    
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
        case "stale_target": return "The selected matchup data has expired. Refresh it before running."
        case "adapter_not_supported", "unsupported_model", "unsupported_model_type", "unsupported_hosted_coverage", "unsupported_features", "advanced_access_required":
            return "This model or feature selection is unavailable. Adjust the setup and try again."
        case "training_interrupted": return "Training was interrupted. Choose Retrain with new data to try again."
        case "training_time_limit": return "Training took too long. Adjust the setup or retry later."
        case "rate_limit", "quota_exceeded", "owner_busy", "training_capacity_busy", "training_quota_exceeded":
            return "The model service is busy or your run limit has been reached. Try again later."
        case "hosted_training_disabled": return "Model runs are temporarily unavailable. You can still save this setup."
        default: return "The model service couldn’t complete this setup. Check your selection and try again later."
        }
    }
    static func gameMarket(target:String?,stat:String?) -> String? {
        guard target == "Team" else{return nil}
        return ["Game total":"total","Spread":"spread","Moneyline":"moneyline"][stat ?? ""]
    }
    static func validLine(_ line:Double?,target:String?,stat:String?) -> Bool {
        let market=gameMarket(target:target,stat:stat)
        if market == "moneyline" {return line == nil || line == 0}
        guard let line,line.isFinite,line * 2 == (line * 2).rounded() else{return false}
        return market == "spread" ? (-1000...1000).contains(line) : (0...(market == nil ? 10000.0:1000.0)).contains(line)
    }
    static func probabilityLabel(_ recipe:LabRecipe) -> String {
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
