import Foundation

/// Independently recomputes v4 claims. Never executes model or network requests.
enum MatchupNarrativeV4 {
 typealias J = RichContextJSON
 static let pairs: [(String,[String])] = [
("passingAttemptsPerGame", ["passing","Passing workload","attempted","faced","passing attempts"]),
("netPassingYardsPerGame", ["passing","Net passing production","gained","allowed","net passing yards"]),
("completionsPerGame", ["passing","Completed passes","completed","allowed","completed passes"]),
("passingYardsPerGame", ["passing","Passing production","gained","allowed","passing yards"]),
("rushingAttemptsPerGame", ["rushing","Rushing workload","attempted","faced","rushing attempts"]),
("rushingYardsPerGame", ["rushing","Rushing production","gained","allowed","rushing yards"]),
("pointsPerGame", ["scoring","Scoring conditions","scored","allowed","points"]),
("offensiveEfficiency", ["efficiency","Scoring per possession","scored","allowed","points per 100 possessions"]),
("hitsPerGame", ["hits","Hitting against opposing pitching","recorded","allowed","hits"]),
("strikeoutsPerGame", ["strikeouts","Batters against opposing strikeouts","struck out at a rate of","recorded pitching strikeouts at a rate of","batting strikeouts"]),
("runsPerGame", ["run_scoring","Run-scoring conditions","scored","allowed","runs"]),
("goalsPerGame", ["scoring","Goal-scoring conditions","scored","allowed","goals"]),
("shotsPerGame", ["shots","Shots against opposing defense","recorded","allowed","shots on goal"]),
("assistsPerGame", ["ball_movement","Assisted baskets","recorded","allowed opponents","assists"]),
("reboundsPerGame", ["rebounds","Rebounding context","recorded","allowed opponents","rebounds"]),
("turnoversPerGame", ["turnovers","Ball security","committed","forced","turnovers"]),
("fieldGoalsMadePerGame", ["shooting","Field goals made","made","allowed opponents","made field goals"]),
("fieldGoalAttemptsPerGame", ["shooting","Shot attempts","attempted","allowed opponents","field-goal attempts"]),
("threePointersMadePerGame", ["shooting","Three-point shooting","made","allowed opponents","made three-pointers"]),
("threePointAttemptsPerGame", ["shooting","Three-point attempts","attempted","allowed opponents","three-point attempts"])
]
 static let priorities: [String:[String]] = ["NFL": ["passing", "rushing", "play_volume", "scoring"], "NBA": ["pace", "efficiency", "shooting", "ball_movement", "rebounds", "turnovers", "scoring"], "MLB": ["hits", "strikeouts", "run_scoring"], "NHL": ["shots", "special_teams", "goaltending", "scoring"]]
 static let minimum: [String:Int] = ["NFL": 3, "NBA": 5, "NHL": 5, "MLB": 10]
 struct Claim: Sendable, Equatable {
  let id, dimension, label, kind, text, uncertainty: String
  let factIDs, injuryIDs, sourceIDs, roleEvidenceIDs: [String]
  let samples: J
  init(_ j:J) { id=j["id"].string; dimension=j["dimension"].string;label=j["label"].string;kind=j["kind"].string;text=j["text"].string;uncertainty=j["uncertainty"].string;factIDs=j["factIDs"].array.map(\.string);injuryIDs=j["injuryIDs"].array.map(\.string);sourceIDs=j["sourceIDs"].array.map(\.string);samples=j["samples"];roleEvidenceIDs=j["roleEvidenceIDs"].array.map(\.string) }
 }
 struct Result: Sendable {
  let rawData:Data, event:SharedMatchupEvent, generatedAt:String, claims:[Claim], missingContext:[String], contextState:String
  let report: J, context: J
  var providers:[String] {Set((context["sources"].array+report["injuryContext"]["sources"].array+report["injuryRoleContext"]["sources"].array).map{$0["provider"].string}).filter{!$0.isEmpty}.sorted()}
  var designations:J {report["designations"]}
  var injuryContext:J {report["injuryContext"]}
  var injuryRoleContext:J {report["injuryRoleContext"]}
  var roleStatusEvidence:J {report["roleStatusEvidence"]}
  var injuryRoleGaps:J {report["injuryRoleGaps"]}
 }
 static func require(_ condition:Bool)throws {if !condition {throw SharedMatchupReportError.invalidResponse}}
 static func str(_ s:String)->J{.string(s)}
 static func list(_ s:[String])->J{.list(s.map(str))}
 static func decimal(_ j:J)->Decimal {Decimal(string:j.number ?? "0",locale:Locale(identifier:"en_US_POSIX")) ?? 0}
 static func number(_ j:J)->String {var n=decimal(j), rounded=Decimal();NSDecimalRound(&rounded,&n,2,.bankers);return NSDecimalNumber(decimal:rounded).stringValue}
 static func same(_ items:[J],unit:Bool=true)->Bool {
  guard !items.contains(.null) else{return false}
  let keys=["season","phase","period","windowStart","windowEnd"] + (unit ? ["unit"]:[])
  return keys.allSatisfy{key in items.allSatisfy{$0[key]==items[0][key]}}
 }
 static func enough(_ items:[J],_ sport:String)->Bool {items.allSatisfy{$0["sample"]["unit"].string=="games" && ($0["sample"]["count"].integer ?? 0)>=(minimum[sport] ?? Int.max)}}
 static func season(_ f:J,_ sport:String)->String {let n=f["season"].integer ?? 0;return ["NBA","NHL"].contains(sport) ? "\(n-1)–\(n)":"\(n)"}
 static func sample(_ f:J,_ sport:String)->String {"\(f["sample"]["count"].number ?? "0") \(f["sample"]["unit"].string) in the \(season(f,sport)) \(f["phase"].string) sample (\(f["windowStart"].string) to \(f["windowEnd"].string))"}
 static func band(_ f:J,_ r:J)->String {let v=decimal(f["value"]),b=decimal(r["value"]);return v>=b*Decimal(string:"1.10")! ? "high":v<=b*Decimal(string:"0.90")! ? "low":"middle"}
 static func claims(_ projected:J)->[J] {
  let event=projected["event"],sport=event["sport"].string,home=event["homeTeam"].string,away=event["awayTeam"].string
  let facts=projected["facts"].array.filter{["team","league"].contains($0["subject"]["kind"].string)}
  func get(_ team:String,_ role:String,_ metric:String)->J {let found=facts.filter{$0["team"].string==team && $0["role"].string==role && $0["metric"].string==metric};return found.count==1 ? found[0]:.null}
  var result:[J]=[]
  func add(_ id:String,_ dimension:String,_ label:String,_ kind:String,_ text:String,_ items:[J]) {
   let samples=items.map{f in J.object(["factID":f["id"],"sample":f["sample"],"season":f["season"],"phase":f["phase"],"windowStart":f["windowStart"],"windowEnd":f["windowEnd"]])}
   result.append(.object(["id":str(id),"dimension":str(dimension),"label":str(label),"kind":str(kind),"text":str(text),"factIDs":.list(items.map{$0["id"]}),"injuryIDs":.list([]),"sourceIDs":list(Set(items.flatMap{$0["sourceIDs"].array.map(\.string)}).sorted()),"samples":.list(samples),"uncertainty":str("Conditional sample-based interpretation, not a calibrated prediction. Availability and game conditions can change the pattern.")]))
  }
  for (side,team,opponent) in [("home",home,away),("away",away,home)] {
   for (metric,p) in pairs {
    let own=get(team,"offense",metric),opposing=get(opponent,"defense",metric)
    guard same([own,opposing]) else{continue}
    let dimension=p[0],label=p[1],outcome=p[4]
    var text="\(team) \(p[2]) \(number(own["value"])) \(own["unit"].string) across \(sample(own,sport)); \(opponent) \(p[3]) \(number(opposing["value"])) \(opposing["unit"].string) across \(opposing["sample"]["count"].number ?? "0") \(opposing["sample"]["unit"].string) in that same window. "
    var items=[own,opposing],kind="descriptive"
    if !enough(items,sport) {kind="limited_sample";text += "This does not meet the \(minimum[sport] ?? 0)-game rule for a directional \(outcome) expectation; the recorded comparison is descriptive only."}
    else {
     let ref=get("league","reference",metric)
     if same(items+[ref]) && enough([ref],sport) && decimal(ref["value"])>0 {
      items.append(ref);let a=band(own,ref),b=band(opposing,ref)
      text += "The recorded same-period comparison is \(number(ref["value"])) \(ref["unit"].string). "
      if a==b && ["high","low"].contains(a) {kind="conditional";text += "Both rates are at least ten percent \(a=="high" ? "above":"below") that comparison. If these patterns continue, \(team) could record \(a=="high" ? "more":"fewer") \(outcome) than that benchmark against \(opponent)."}
      else if b=="low" && decimal(own["value"])>=decimal(ref["value"]) {kind="conditional";text += "\(opponent)'s rate is at least ten percent below that comparison. If that pattern continues, it could constrain \(team)'s \(outcome); the two rates point in different directions."}
      else {text += "These paired rates do not establish a clear direction for \(outcome). Other supported matchup dimensions still need separate consideration."}
     } else {text += "Without a comparable same-period benchmark, these rates describe the \(outcome) matchup but do not establish an advantage or forecast."}
    }
    add("\(side):\(metric)",dimension,label,kind,text,items)
   }
   if sport=="NFL" {
    let own=get(team,"offense","passAttemptShare"),opposing=get(opponent,"defense","passAttemptShare")
    if same([own,opposing]) && enough([own,opposing],sport) {
     let high=decimal(own["value"])>=60 && decimal(opposing["value"])>=60,low=decimal(own["value"])<=40 && decimal(opposing["value"])<=40
     let balanced=decimal(own["value"])>40 && decimal(own["value"])<60 && decimal(opposing["value"])>40 && decimal(opposing["value"])<60
     if high || low || balanced {
      let direction=high ? "passing":"rushing"
      var text="Passing accounted for \(number(own["value"]))% of \(team)'s pass-and-rush attempts across \(sample(own,sport)), and \(number(opposing["value"]))% of attempts faced by \(opponent) across \(opposing["sample"]["count"].number ?? "0") games in the same window. "
      text += balanced ? "Both pass shares are between forty and sixty percent. If these mixes continue, \(team) could combine passing and rushing rather than lean heavily on one. " : "If this mix continues, \(team) could lean more on \(direction) than \(high ? "rushing":"passing"). "
      text += "This excludes sacks and does not establish tempo or yardage."
      add("\(side):attempt-mix",balanced ? "passing":direction,balanced ? "Balanced passing and rushing":"Passing and rushing mix","conditional",text,[own,opposing])
     }
    }
   }
   if sport=="NHL" {
    let pp=get(team,"offense","powerPlayPercentage"),pk=get(opponent,"defense","penaltyKillPercentage")
    if same([pp,pk],unit:false) {
     let text="\(team) converted \(number(pp["value"]))% of recorded power-play chances; \(opponent) killed \(number(pk["value"]))% of recorded penalties in the same \(season(pp,sport)) \(pp["phase"].string) window. Power-play conversion and penalty-kill success are different measures. Their interaction is worth watching if penalties occur; these figures alone do not predict penalty volume or special-teams goals."
     add("\(side):special-teams","special_teams","Power play against penalty kill","descriptive",text,[pp,pk])
    }
   }
  }
  for metric in ["pace","secondsPerOffensivePlay","offensivePlaysPerGame"] {
   let own=get(home,"offense",metric),other=get(away,"offense",metric)
   guard same([own,other]) else{continue}
   let dimension=sport=="NBA" ? "pace":"play_volume"
   var text="\(home) recorded \(number(own["value"])) \(own["unit"].string) across \(sample(own,sport)); \(away) recorded \(number(other["value"])) \(other["unit"].string) across \(other["sample"]["count"].number ?? "0") \(other["sample"]["unit"].string) in that window. "
   var kind="descriptive",items=[own,other];let ref=get("league","reference",metric)
   if metric=="offensivePlaysPerGame" {text += "Plays per game describe volume, not seconds per play or pace. Game length and possession also affect this count."}
   else if metric=="secondsPerOffensivePlay" {text += "This is possession-clock seconds divided by offensive plays. It does not measure time between snaps or establish a faster or slower tempo."}
   else if same(items+[ref]) && enough(items+[ref],sport) && decimal(ref["value"])>0 {
    items.append(ref);let a=band(own,ref),b=band(other,ref)
    text += "The recorded same-period comparison is \(number(ref["value"])) \(ref["unit"].string). "
    if a==b && ["high","low"].contains(a) {kind="conditional";let fast=metric=="pace" ? a=="high":a=="low";text += "If both recorded patterns continue, the matchup could be \(fast ? "faster":"slower") than that comparison. Tempo alone does not predict scoring efficiency or a game total."}
    else {text += "The recorded speeds do not give a consistent tempo direction."}
   } else {text += "The available samples do not establish a comparative tempo direction."}
   add("game:"+metric,dimension,sport=="NBA" ? "Recorded pace":metric=="offensivePlaysPerGame" ? "Offensive play count":"Possession time per play",kind,text,items)
  }
  return result
 }
 static func prepare(_ context:J,now:Date,injuryContext:J = .null)throws->J {
  let projected=try RichSportsContext.project(Data(context.canonical.utf8),now:now)
  try require(["fresh","partial"].contains(projected["state"].string) && !projected["facts"].array.isEmpty)
  let priority=priorities[projected["event"]["sport"].string] ?? []
  let ranks=["conditional":0,"limited_sample":1,"descriptive":2]
  let metricOrder=["attempt-mix":0,"netPassingYardsPerGame":1,"passingAttemptsPerGame":2,"rushingYardsPerGame":1,"rushingAttemptsPerGame":2,"fieldGoalAttemptsPerGame":1,"fieldGoalsMadePerGame":2]
  var all=claims(projected)
  try require(!all.isEmpty)
  all.sort {a,b in
   let ap=priority.firstIndex(of:a["dimension"].string) ?? 99,bp=priority.firstIndex(of:b["dimension"].string) ?? 99
   if ap != bp{return ap<bp};let ar=ranks[a["kind"].string] ?? 99,br=ranks[b["kind"].string] ?? 99
   if ar != br{return ar<br}
   let am=metricOrder[String(a["id"].string.split(separator:":").last ?? "")] ?? 5,bm=metricOrder[String(b["id"].string.split(separator:":").last ?? "")] ?? 5
   return am != bm ? am<bm:a["id"].string<b["id"].string
  }
  all=Array(priority.flatMap{d -> [J] in
   let options=all.filter{$0["dimension"].string==d}
   var chosen=["home","away"].compactMap{side in options.first{$0["id"].string.hasPrefix(side+":")}}
   chosen.append(contentsOf:options.filter{!chosen.contains($0)})
   return Array(chosen.prefix(3))
  }.prefix(24))
  let dimensions=priority.filter{d in all.contains{$0["dimension"].string==d}},nonScoring=dimensions.filter{!["scoring","run_scoring"].contains($0)}
  let required=Array((nonScoring.isEmpty ? dimensions:nonScoring).prefix(3))
  var missing=projected["coverage"].array.filter{$0["state"].string != "available"}.map{"\($0["team"].string): \($0["topic"].string) (\($0["reason"].string))."}
  if !projected["excluded"].array.isEmpty {missing.append("Excluded stale, conflicting, or started-event evidence is not used in this explanation.")}
  if projected["facts"].array.contains(where:{$0["phase"].string=="preseason"}) {missing.append("Preseason samples describe exhibition play; rotations and availability can differ in regular-season games.")}
  var injury=J.null
  if injuryContext != .null {
   injury=try NarrativeInjuryContext.model(injuryContext,now:now,maxObservations:6,maxBytes:6144)
   let normalizedInjuryEvent=injury["event"].setting("startISO",str(RichSportsContext.iso(Date(timeIntervalSince1970:try NarrativeInjuryContext.time(injury["event"]["startISO"])))))
   try require(normalizedInjuryEvent==projected["event"])
  } else {missing.append("Verified current injury context is not supplied; missing status does not mean available.")}
  return .object(["event":projected["event"],"contextFingerprint":projected["fingerprint"],"contextRevision":projected["revision"],"contextState":projected["state"],"claims":.list(all),"requiredDimensions":list(required),"missingContext":list(missing),"injuryContext":injury,"projection":projected])
 }
 static func validateSelection(_ selection:J,plan:J)throws {
  try selection.keys(["claimIDs"])
  let ids=selection["claimIDs"].array.map(\.string),all=plan["claims"].array
  try require((1...6).contains(ids.count) && Set(ids).count==ids.count && ids.allSatisfy{ id in all.contains{$0["id"].string==id}})
  let selected=ids.compactMap{id in all.first{$0["id"].string==id}},dimensions=selected.map{$0["dimension"].string}
  try require(Set(dimensions).count==dimensions.count && Set(plan["requiredDimensions"].array.map(\.string)).isSubset(of:Set(dimensions)))
  for d in plan["requiredDimensions"].array.map(\.string) where all.contains(where:{$0["dimension"].string==d && $0["kind"].string=="conditional"}) {
   try require(selected.contains{$0["dimension"].string==d && $0["kind"].string=="conditional"})
  }
 }
 static func render(_ selection:J,plan:J)throws->J {
  try validateSelection(selection,plan:plan)
  let selected=selection["claimIDs"].array.map{ id in plan["claims"].array.first{$0["id"]==id}! }
  return .object(["claims":.list(selected),"missingContext":plan["missingContext"],"contextState":plan["contextState"],"requiredDimensions":plan["requiredDimensions"],"injuryContext":plan["injuryContext"],"designations":plan["projection"]["designations"],"disclosure":str("Gemma selected these source-backed conditional explanations. Wording is produced by validated rules, not a calibrated forecast.")])
 }
 static func injuryIdentity(_ view:J)->J {
  if view == .null{return .null}
  return .object(["injuryFingerprint":view["injuryFingerprint"],"contextFingerprint":view["contextFingerprint"],"observations":.list(view["observations"].array.map{.list([$0["claimID"],$0["assertion"]])}),"exclusions":view["exclusions"],"policy":str("six-observations-6144-bytes-v1")])
 }
 static func validate(_ raw:Data,now:Date,versions:[String:String])throws->Result {
  try require(raw.count<=131072)
  var parser=RichContextJSONParser(raw);let wire=try parser.parse()
  try wire.keys(["contractVersion","versions","event","generatedAt","sourceFingerprint","context","injuryContext","selection","report"])
  try require(wire["contractVersion"].string=="matchup-narrative-v4" && wire["versions"] == .object(versions.mapValues(str)))
  let generated=try RichSportsContext.time(wire["generatedAt"]),start=try RichSportsContext.time(wire["event"]["startISO"])
  try require(generated<=now && generated<start)
  let plan=try prepare(wire["context"],now:generated,injuryContext:wire["injuryContext"])
  let report=try render(wire["selection"],plan:plan)
  let fingerprint=RichSportsContext.digest(.object(["sports":plan["contextFingerprint"],"revision":plan["contextRevision"],"injury":injuryIdentity(plan["injuryContext"])]))
  let expected=wire.setting("event",plan["event"]).setting("generatedAt",str(RichSportsContext.iso(generated))).setting("sourceFingerprint",str(fingerprint)).setting("report",report)
  try require(raw == Data(expected.canonical.utf8))
  let current=try prepare(wire["context"],now:now,injuryContext:wire["injuryContext"])
  try require(try render(wire["selection"],plan:current)==report)
  let event=try JSONDecoder().decode(SharedMatchupEvent.self,from:Data(wire["event"].canonical.utf8))
  return Result(rawData:raw,event:event,generatedAt:wire["generatedAt"].string,claims:report["claims"].array.map(Claim.init),missingContext:report["missingContext"].array.map(\.string),contextState:report["contextState"].string,report:report,context:wire["context"])
 }
}
