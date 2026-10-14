import Foundation
import CryptoKit

indirect enum RichContextJSON: Equatable, Sendable {
    case object([String:RichContextJSON]), list([RichContextJSON]), string(String), number(String), bool(Bool), null
    subscript(_ key: String) -> RichContextJSON { if case .object(let fields) = self { return fields[key] ?? .null }; return .null }
    var array: [RichContextJSON] { if case .list(let items) = self {return items}; return [] }
    var string: String { if case .string(let value) = self {return value}; return "" }
    var number: String? { if case .number(let value) = self {return value}; return nil }
    var integer: Int? { guard let number, number.range(of:#"^-?(0|[1-9][0-9]*)$"#,options:.regularExpression) != nil else {return nil}; return Int(number) }
    func keys(_ names: Set<String>) throws { guard case .object(let fields) = self, Set(fields.keys) == names else {throw SharedMatchupReportError.invalidResponse} }
    func setting(_ key: String, _ value: RichContextJSON) -> RichContextJSON { guard case .object(var fields) = self else {return self};fields[key]=value;return .object(fields) }
    func removing(_ keys: Set<String>) -> RichContextJSON { guard case .object(var fields) = self else {return self};for key in keys {fields.removeValue(forKey:key)};return .object(fields) }
    private func quote(_ string: String) -> String {
        let encoded = try! JSONSerialization.data(withJSONObject:string,options:[.fragmentsAllowed,.withoutEscapingSlashes])
        return String(data:encoded,encoding:.utf8)!
    }
    var canonical: String {
        switch self {
        case .object(let fields):return "{"+fields.keys.sorted().map {quote($0)+":"+fields[$0]!.canonical}.joined(separator:",")+"}"
        case .list(let values):return "["+values.map(\.canonical).joined(separator:",")+"]"
        case .string(let value):return quote(value)
        case .number(let value):return value
        case .bool(let value):return value ? "true":"false"
        case .null:return "null"
        }
    }
}
struct RichContextJSONParser {
    let bytes: [UInt8]; var index=0
    init(_ data:Data){bytes=Array(data)}
    mutating func parse() throws -> RichContextJSON {let result=try value(0);space();guard index==bytes.count else{throw SharedMatchupReportError.invalidResponse};return result}
    mutating func space(){while index<bytes.count && [9,10,13,32].contains(bytes[index]){index+=1}}
    mutating func take(_ byte:UInt8)->Bool{space();if index<bytes.count && bytes[index]==byte{index+=1;return true};return false}
    mutating func string() throws -> String {
        space();let start=index;guard index<bytes.count && bytes[index]==34 else{throw SharedMatchupReportError.invalidResponse};index+=1
        while index<bytes.count {
            if bytes[index]==92 {index+=2;continue}
            if bytes[index]==34 {index+=1;return try JSONDecoder().decode(String.self,from:Data(bytes[start..<index]))}
            index+=1
        }
        throw SharedMatchupReportError.invalidResponse
    }
    mutating func value(_ depth:Int)throws->RichContextJSON {
        guard depth<=16 else{throw SharedMatchupReportError.invalidResponse};space()
        guard index<bytes.count else{throw SharedMatchupReportError.invalidResponse}
        if bytes[index]==34{return .string(try string())}
        if take(123){var fields:[String:RichContextJSON]=[:];if take(125){return .object(fields)}
            repeat {let key=try string();guard fields[key]==nil,take(58) else{throw SharedMatchupReportError.invalidResponse};fields[key]=try value(depth+1);if take(125){return .object(fields)}}while take(44)
            throw SharedMatchupReportError.invalidResponse}
        if take(91){var items:[RichContextJSON]=[];if take(93){return .list(items)}
            repeat {items.append(try value(depth+1));if take(93){return .list(items)}}while take(44)
            throw SharedMatchupReportError.invalidResponse}
        for (literal,result) in [("null",RichContextJSON.null),("true",.bool(true)),("false",.bool(false))] {
            let token=Array(literal.utf8);if index+token.count<=bytes.count && Array(bytes[index..<index+token.count])==token{index+=token.count;return result}
        }
        let start=index;while index<bytes.count && ![9,10,13,32,44,93,125].contains(bytes[index]){index+=1}
        let token=String(decoding:bytes[start..<index],as:UTF8.self)
        guard token.range(of:#"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"#,options:.regularExpression) != nil else{throw SharedMatchupReportError.invalidResponse}
        return .number(token)
    }
}

enum RichSportsContext {
 static let maximumBytes = 32768
 static let metrics: [String:(String,Double,Set<String>)] = [
  "pointsPerGame": ("points per game", 250, ["NBA","NFL"]),
  "passingAttemptsPerGame": ("pass attempts per game", 100, ["NFL"]),
  "rushingAttemptsPerGame": ("rush attempts per game", 100, ["NFL"]),
  "passingYardsPerGame": ("gross passing yards per game", 1000, ["NFL"]),
  "netPassingYardsPerGame": ("net passing yards per game", 1000, ["NFL"]),
  "rushingYardsPerGame": ("rushing yards per game", 1000, ["NFL"]),
  "completionsPerGame": ("completions per game", 100, ["NFL"]),
  "completionPercentage": ("percent of pass attempts completed", 100, ["NFL"]),
  "passAttemptShare": ("percent of pass and rush attempts", 100, ["NFL"]),
  "offensivePlaysPerGame": ("offensive plays per game", 200, ["NFL"]),
  "secondsPerOffensivePlay": ("possession seconds per offensive play", 120, ["NFL"]),
  "sacksTakenPerGame": ("passing sacks taken per game", 30, ["NFL"]),
  "sacksMadePerGame": ("defensive sacks made per game", 30, ["NFL"]),
  "runsPerGame": ("runs per game", 40, ["MLB"]),
  "hitsPerGame": ("hits per game", 50, ["MLB"]),
  "strikeoutsPerGame": ("strikeouts per game", 50, ["MLB"]),
  "walksPerGame": ("walks per game", 40, ["MLB"]),
  "homeRunsPerGame": ("home runs per game", 20, ["MLB"]),
  "battingAverage": ("hits per at-bat", 1, ["MLB"]),
  "onBasePercentage": ("on-base fraction", 1, ["MLB"]),
  "sluggingPercentage": ("total bases per at-bat", 4, ["MLB"]),
  "pitchesLast3Days": ("pitches in the recorded three-day window", 500, ["MLB"]),
  "outsLast3Days": ("outs in the recorded three-day window", 100, ["MLB"]),
  "appearancesLast3Days": ("appearances in the recorded three-day window", 10, ["MLB"]),
  "goalsPerGame": ("goals per game", 20, ["NHL"]),
  "shotsPerGame": ("shots on goal per game", 100, ["NHL"]),
  "savesPerGame": ("saves per game", 100, ["NHL"]),
  "savePercentage": ("fraction of shots saved", 1, ["NHL"]),
  "powerPlayGoalsPerGame": ("power-play goals per game", 20, ["NHL"]),
  "shortHandedGoalsPerGame": ("shorthanded goals per game", 20, ["NHL"]),
  "powerPlayPercentage": ("percent of power plays converted", 100, ["NHL"]),
  "penaltyKillPercentage": ("percent of opponent power plays killed", 100, ["NHL"]),
  "fieldGoalsMadePerGame": ("field goals made per game", 100, ["NBA"]),
  "fieldGoalAttemptsPerGame": ("field goal attempts per game", 200, ["NBA"]),
  "threePointersMadePerGame": ("three-pointers made per game", 70, ["NBA"]),
  "threePointAttemptsPerGame": ("three-point attempts per game", 150, ["NBA"]),
  "freeThrowsMadePerGame": ("free throws made per game", 100, ["NBA"]),
  "freeThrowAttemptsPerGame": ("free throw attempts per game", 150, ["NBA"]),
  "fieldGoalPercentage": ("percent of field goal attempts made", 100, ["NBA"]),
  "threePointPercentage": ("percent of three-point attempts made", 100, ["NBA"]),
  "freeThrowPercentage": ("percent of free throw attempts made", 100, ["NBA"]),
  "reboundsPerGame": ("rebounds per game", 120, ["NBA"]),
  "assistsPerGame": ("assists per game", 100, ["NBA"]),
  "turnoversPerGame": ("turnovers per game", 100, ["NBA"]),
  "minutesPerGame": ("minutes per game", 100, ["NBA"]),
  "possessionsPerGame": ("possessions per game", 200, ["NBA"]),
  "pace": ("possessions per 48 minutes", 200, ["NBA"]),
  "offensiveEfficiency": ("points per 100 possessions", 250, ["NBA"]),
 ]
 static let topics: [String:[String:Set<String>]] = [
 "NFL": ["scoring": ["pointsPerGame"],"passing": ["netPassingYardsPerGame","passingAttemptsPerGame"],"rushing": ["rushingAttemptsPerGame","rushingYardsPerGame"],"pace": ["offensivePlaysPerGame","secondsPerOffensivePlay"],"playerContext": []],
 "NBA": ["scoring": ["pointsPerGame"],"shooting": ["fieldGoalAttemptsPerGame","fieldGoalsMadePerGame"],"playmaking": ["assistsPerGame","turnoversPerGame"],"pace": ["pace"],"playerContext": []],
 "MLB": ["batting": ["battingAverage","hitsPerGame","strikeoutsPerGame"],"pitching": ["hitsPerGame","strikeoutsPerGame"],"starter": [],"bullpenWorkload": ["outsLast3Days","pitchesLast3Days"],"bullpenAvailability": [],"playerContext": []],
 "NHL": ["scoring": ["goalsPerGame"],"shots": ["shotsPerGame"],"goaltending": ["savePercentage","savesPerGame"],"starter": [],"specialTeams": ["penaltyKillPercentage","powerPlayPercentage"],"playerContext": []],
 ]
 static func require(_ valid:Bool) throws {if !valid {throw SharedMatchupReportError.invalidResponse}}
 static func match(_ value:String,_ pattern:String)->Bool {value.range(of:"\\A(?:"+pattern+")\\z",options:.regularExpression) != nil}
 static func text(_ value:RichContextJSON,_ limit:Int)throws {let s=value.string;try require(!s.isEmpty && s.count<=limit && s==s.trimmingCharacters(in:.whitespacesAndNewlines) && s==s.precomposedStringWithCanonicalMapping && s.unicodeScalars.allSatisfy { !CharacterSet.controlCharacters.contains($0) && !CharacterSet.symbols.contains($0) })}
 static func time(_ value:RichContextJSON)throws->Date {try SharedMatchupReportValidator.date(value.string)}
 static func iso(_ date:Date)->String {let f=ISO8601DateFormatter();f.formatOptions=[.withInternetDateTime,.withFractionalSeconds];return f.string(from:date)}
 static func digest(_ value:RichContextJSON)->String {SHA256.hash(data:Data(value.canonical.utf8)).map {String(format:"%02x",$0)}.joined()}
 static func number(_ value:RichContextJSON)throws->Double {guard let s=value.number,let v=Double(s),v.isFinite,v>=0 else {throw SharedMatchupReportError.invalidResponse};return v}
 static func list(_ value:RichContextJSON)throws->[RichContextJSON] {guard case .list(let a)=value else{throw SharedMatchupReportError.invalidResponse};return a}
 static func sourceURL(_ value:RichContextJSON,sport:String)throws {
  let s=value.string;try require(!s.isEmpty && s.count<=1000 && !s.contains(where:{$0.isWhitespace}))
  guard let u=URLComponents(string:s),u.scheme=="https",let host=u.host,u.user==nil,u.password==nil,u.port==nil,u.fragment==nil else {throw SharedMatchupReportError.invalidResponse}
  try require(s.hasPrefix("https://"+host+"/") && u.percentEncodedPath==u.path)
  let paths=["NFL":"football/nfl","NBA":"basketball/nba","MLB":"baseball/mlb","NHL":"hockey/nhl"]
  if ["site.api.espn.com","site.web.api.espn.com"].contains(host) {try require(match(u.path,"/apis/site/v2/sports/"+paths[sport,default:""]+"/(?:scoreboard|summary|teams(?:/\\d+/(?:statistics|schedule))?)"))}
  else if sport=="MLB" && host=="statsapi.mlb.com" {try require(match(u.path,#"/api/v1/(?:teams(?:/stats|/\d+/stats)?|schedule|people/\d+/stats|game/\d+/boxscore)"#))}
  else if sport=="NHL" && host=="api.nhle.com" {try require(["/stats/rest/en/team/powerplay","/stats/rest/en/team/penaltykill","/stats/rest/en/goalie/summary"].contains(u.path))}
  else {throw SharedMatchupReportError.invalidResponse}
  let rules=["season":#"20\d{2}"#,"seasontype":"[123]","limit":#"(?:[1-9]\d{0,3}|-1)"#,"dates":#"20\d{6}"#,"event":#"\d{1,15}"#,"stats":"(?:season|gameLog)","group":"(?:hitting|pitching)","sportIds":"1","sportId":"1","gameType":"[RSPFAWDL]","date":#"20\d{2}-\d{2}-\d{2}"#,"startDate":#"20\d{2}-\d{2}-\d{2}"#,"endDate":#"20\d{2}-\d{2}-\d{2}"#,"teamId":#"\d{1,10}"#,"hydrate":"probablePitcher","cayenneExp":#"seasonId=20\d{6} and gameTypeId=[123]"#,"isAggregate":"(?:true|false)","isGame":"(?:true|false)","start":#"\d{1,4}"#]
  let items=u.queryItems ?? [];try require(items.count<=10 && Set(items.map(\.name)).count==items.count)
  for q in items {guard let v=q.value,let rule=rules[q.name] else{throw SharedMatchupReportError.invalidResponse};try require(match(v.replacingOccurrences(of:"+",with:" "),rule))}
 }
 static func subject(_ s:RichContextJSON,team:String,role:String)throws {
  try s.keys(["kind","id","name"]);try require(["team","player","league"].contains(s["kind"].string) && match(s["id"].string,#"(?:espn|mlb|nhl|nba|league):[A-Za-z0-9_-]{1,30}"#));try text(s["name"],100)
  try require((role=="player")==((s["kind"].string)=="player") && (role=="reference")==((s["kind"].string)=="league"));if s["kind"].string=="team" {try require(s["name"].string==team)}
 }
 static func validate(_ data:Data,now:Date)throws->RichContextJSON {
  try require(data.count<=maximumBytes && now.timeIntervalSince1970.isFinite);var p=RichContextJSONParser(data);var b=try p.parse()
  try b.keys(["contractVersion","event","sources","facts","designations","coverage"]);try require(b["contractVersion"].string=="sports-context-v1" && b.canonical.utf8.count<=maximumBytes)
  try b["event"].keys(["sport","eventID","homeTeam","awayTeam","startISO"])
  let event=try SharedMatchupReportValidator.normalizedEvent(JSONDecoder().decode(SharedMatchupEvent.self,from:Data(b["event"].canonical.utf8)))
  b=b.setting("event",b["event"].setting("startISO",.string(iso(try time(b["event"]["startISO"])))))
  let sport=event.sport,teams:Set<String>=[event.homeTeam,event.awayTeam];let eventStart=try time(b["event"]["startISO"])
  let ss=try list(b["sources"]);try require(ss.count<=24);var sources:[String:RichContextJSON]=[:]
  for s in ss {try s.keys(["id","provider","url","retrievedAt","checkedAt","sourceAsOf","contentSHA256"]);let id=s["id"].string;try require(match(id,"[A-Za-z0-9:._-]{1,100}") && sources[id]==nil);try sourceURL(s["url"],sport:sport);let host=URLComponents(string:s["url"].string)?.host;try require(s["provider"].string == (host=="statsapi.mlb.com" ? "MLB Stats API":host=="api.nhle.com" ? "NHL":"ESPN"));let r=try time(s["retrievedAt"]),c=try time(s["checkedAt"]);try require(r<=c && c<=now);if s["sourceAsOf"] != .null {try require(time(s["sourceAsOf"])<=r)};try require(match(s["contentSHA256"].string,"[a-f0-9]{64}"));sources[id]=s}
  func ids(_ value:RichContextJSON,empty:Bool=false)throws->[String] {let a=try list(value);let ids=a.map(\.string);try require(a.count >= (empty ? 0:1) && a.count<=4 && Set(ids).count==a.count && ids.allSatisfy{sources[$0] != nil});return ids}
  let facts=try list(b["facts"]);try require(facts.count<=48);var seen=Set<String>(),dimensions=Set<String>()
  for f in facts {
   try f.keys(["id","team","subject","role","metric","value","unit","sample","season","phase","period","windowStart","windowEnd","sourceIDs","derivation"])
   try require(match(f["id"].string,"[A-Za-z0-9:._-]{1,120}") && seen.insert(f["id"].string).inserted)
   let role=f["role"].string,team=f["team"].string,metric=f["metric"].string
   try require(["offense","defense","player","reference"].contains(role) && (role=="reference" ? team=="league":teams.contains(team)));try subject(f["subject"],team:team,role:role)
   if metric=="sacksMadePerGame" {try require(["defense","reference"].contains(role))};if metric=="sacksTakenPerGame" {try require(["offense","defense","reference"].contains(role))}
   guard let m=metrics[metric] else{throw SharedMatchupReportError.invalidResponse};let value=try number(f["value"]);try require(m.2.contains(sport) && f["unit"].string==m.0 && value<=m.1)
   try f["sample"].keys(["count","unit"]);try require((1...100000).contains(f["sample"]["count"].integer ?? 0) && ["games","attempts","shots","possessions","plate_appearances","batters_faced"].contains(f["sample"]["unit"].string))
   try require((2000...2099).contains(f["season"].integer ?? 0) && ["preseason","regular","postseason"].contains(f["phase"].string));try text(f["period"],100)
   let sourceIDs=try ids(f["sourceIDs"]),start=try time(f["windowStart"]),end=try time(f["windowEnd"]);try require(start<end && end<eventStart);for id in sourceIDs {try require(end<=time(sources[id]!["retrievedAt"]))}
   let d=f["derivation"];try d.keys(["kind","numerator","denominator"]);let numerator=try number(d["numerator"]);var expected=numerator
   if d["kind"].string=="reported" {try require(d["denominator"] == .null)} else {try require(["per_game","percent","per_100","ratio"].contains(d["kind"].string));let denominator=try number(d["denominator"]);try require(denominator>0);expected=numerator/denominator * (["percent","per_100"].contains(d["kind"].string) ? 100:1);if d["kind"].string=="per_game" {try require(Double(f["sample"]["count"].integer ?? 0)==denominator && f["sample"]["unit"].string=="games")}}
   try require(abs(value-expected)<=max(1,abs(expected))*1e-12)
   if metric.hasSuffix("Last3Days") {try require(f["subject"]["kind"].string=="player" && end.timeIntervalSince(start)<=259200)}
   let dimension=RichContextJSON.list(["team","subject","role","metric","season","phase","windowStart","windowEnd"].map{f[$0]}).canonical;try require(dimensions.insert(dimension).inserted)
  }
  let ds=try list(b["designations"]);try require(ds.count<=8);seen=[]
  for d in ds {try d.keys(["id","team","kind","state","subject","eventID","sourceIDs","reportedAt","basis"]);try require(match(d["id"].string,"[A-Za-z0-9:._-]{1,120}") && seen.insert(d["id"].string).inserted && teams.contains(d["team"].string) && d["eventID"].string==event.eventID)
   let kind=["MLB":"starting_pitcher","NHL":"starting_goalie","NFL":"starting_lineup","NBA":"starting_lineup"][sport]!,state=d["state"].string;try require(d["kind"].string==kind && ["confirmed","probable","unknown"].contains(state));let src=try ids(d["sourceIDs"],empty:state=="unknown");try require(d["basis"].string==["confirmed":"provider_confirmed","probable":"provider_probable","unknown":"not_reported"][state]);if state=="unknown" {try require(d["subject"] == .null && d["reportedAt"] == .null)} else {try subject(d["subject"],team:d["team"].string,role:"player");if d["reportedAt"] != .null {for id in src {try require(time(d["reportedAt"])<=time(sources[id]!["retrievedAt"]))}}}
  }
  let cs=try list(b["coverage"]);try require(cs.count==2*topics[sport]!.count);seen=[]
  for c in cs {try c.keys(["team","topic","state","reason","sourceIDs"]);try require(teams.contains(c["team"].string) && topics[sport]![c["topic"].string] != nil && seen.insert(c["team"].string+"|"+c["topic"].string).inserted);try require(["available","partial","missing"].contains(c["state"].string) && ["supported","partial_source","not_supplied","source_unavailable","unsupported_metric","no_reported_designation"].contains(c["reason"].string));let src=try ids(c["sourceIDs"],empty:true);try require(c["state"].string=="missing" || !src.isEmpty);if c["topic"].string=="bullpenAvailability" {try require(c["state"].string=="missing")}}
  let derived=try coverage(b,facts:facts,designations:ds)
  for c in cs {guard let expected=derived.first(where:{$0["team"]==c["team"] && $0["topic"]==c["topic"]}) else{throw SharedMatchupReportError.invalidResponse};try require(c["state"]==expected["state"] && c["sourceIDs"]==expected["sourceIDs"]);try require(c["reason"]==expected["reason"] || (c["state"].string=="missing" && ["source_unavailable","unsupported_metric"].contains(c["reason"].string)))}
  return b.setting("sources",.list(ss.sorted{$0["id"].string<$1["id"].string})).setting("facts",.list(facts.sorted{$0["id"].string<$1["id"].string})).setting("designations",.list(ds.sorted{$0["id"].string<$1["id"].string})).setting("coverage",.list(cs.sorted{($0["team"].string,$0["topic"].string)<($1["team"].string,$1["topic"].string)}))
 }
 static func fingerprint(_ validated:RichContextJSON)->String {digest(validated.setting("sources",.list(validated["sources"].array.map{$0.removing(["retrievedAt","checkedAt"])})))}
 static func coverage(_ b:RichContextJSON,facts:[RichContextJSON],designations:[RichContextJSON])throws->[RichContextJSON] {
  var rows:[RichContextJSON]=[];let sport=b["event"]["sport"].string
  for team in [b["event"]["awayTeam"].string,b["event"]["homeTeam"].string] {
   // Python topic insertion order is part of the projection revision.
   let order:[String:[String]]=["NFL":["scoring","passing","rushing","pace","playerContext"],"NBA":["scoring","shooting","playmaking","pace","playerContext"],"MLB":["batting","pitching","starter","bullpenWorkload","bullpenAvailability","playerContext"],"NHL":["scoring","shots","goaltending","starter","specialTeams","playerContext"]]
   for topic in order[sport]! {let needed=topics[sport]![topic]!
    var relevant=facts.filter{$0["team"].string==team && needed.contains($0["metric"].string)}
    if topic=="specialTeams" {relevant=facts.filter{$0["team"].string==team && needed.union(["powerPlayGoalsPerGame","shortHandedGoalsPerGame"]).contains($0["metric"].string)}}
    if topic=="playerContext" {relevant=facts.filter{$0["team"].string==team && $0["role"].string=="player"}}
    if sport=="MLB" && ["batting","pitching"].contains(topic) {relevant=relevant.filter{$0["role"].string==(topic=="batting" ? "offense":"defense")}}
    let ds=topic=="starter" ? designations.filter{$0["team"].string==team && $0["state"].string != "unknown"}:[]
    let found=Set(relevant.map{$0["metric"].string});var state=(!needed.isEmpty && needed.isSubset(of:found)) || !ds.isEmpty ? "available":(!relevant.isEmpty ? "partial":"missing")
    if topic=="playerContext" && !relevant.isEmpty {state="partial"}
    if (topic=="bullpenWorkload" || (sport=="NFL" && topic=="pace")) && !relevant.isEmpty {state="partial"}
    let ids=Set((relevant+ds).flatMap{$0["sourceIDs"].array.map(\.string)}).sorted();try require(ids.count<=4)
    rows.append(.object(["team":.string(team),"topic":.string(topic),"state":.string(state),"reason":.string(state=="available" ? "supported":state=="partial" ? "partial_source":topic=="starter" ? "no_reported_designation":"not_supplied"),"sourceIDs":.list(ids.map{.string($0)})]))
   }
  };return rows
 }
 static func project(_ data:Data,now:Date)throws->RichContextJSON {
  let b=try validate(data,now:now),started=try time(b["event"]["startISO"])<=now
  let sources=Dictionary(uniqueKeysWithValues:b["sources"].array.map{($0["id"].string,$0)})
  func stale(_ ids:RichContextJSON,_ ttl:Double)throws->Bool {try ids.array.contains {id in let s=sources[id.string]!;return now.timeIntervalSince(min(try time(s["retrievedAt"]),try time(s["checkedAt"])))>=ttl}}
  var facts:[RichContextJSON]=[],designations:[RichContextJSON]=[],excluded:[RichContextJSON]=[]
  for f in b["facts"].array {let workload=f["metric"].string.hasSuffix("Last3Days");var reason:String?=started ? "event_started":(try stale(f["sourceIDs"],workload ? 21600:86400)) ? "source_stale":nil
   if try reason==nil && workload && now.timeIntervalSince(time(f["windowEnd"]))>=21600 {reason="sample_stale"}
   if let reason {excluded.append(.object(["id":f["id"],"reason":.string(reason)]))} else {facts.append(f)}
  }
  var groups:[String:Set<String>]=[:]
  for d in b["designations"].array where d["state"].string != "unknown" {groups[d["team"].string+"|"+d["kind"].string,default:[]].insert(d["subject"]["id"].string)}
  for d in b["designations"].array {var reason:String?
   if started {reason="event_started"} else if groups[d["team"].string+"|"+d["kind"].string,default:[]].count>1 {reason="conflicting_designation"}
   else if try stale(d["sourceIDs"],21600) {reason="designation_stale"}
   else if try d["reportedAt"] != .null && now.timeIntervalSince(time(d["reportedAt"]))>=21600 {reason="designation_stale"}
   if let reason {excluded.append(.object(["id":d["id"],"reason":.string(reason)]))} else {designations.append(d)}
  }
  let cs=try coverage(b,facts:facts,designations:designations),anyEvidence = !facts.isEmpty || designations.contains{$0["state"].string != "unknown"}
  let state=anyEvidence ? (cs.allSatisfy{$0["state"].string=="available"} ? "fresh":"partial"):(!excluded.isEmpty && !started ? "stale":"missing")
  let fp=fingerprint(b)
  var result=RichContextJSON.object(["contractVersion":.string("sports-context-v1"),"event":b["event"],"fingerprint":.string(fp),"evaluatedAt":.string(iso(now)),"state":.string(state),"facts":.list(facts),"designations":.list(designations),"sources":b["sources"],"coverage":.list(cs),"excluded":.list(excluded)])
  let revision=digest(.object(["policy":.string("sports-context-freshness-v1"),"fingerprint":.string(fp),"facts":.list(facts.map{$0["id"]}),"designations":.list(designations.map{$0["id"]}),"coverage":.list(cs),"excluded":.list(excluded)]))
  result=result.setting("revision",.string(revision));try require(result.canonical.utf8.count<=maximumBytes);return result
 }
}
