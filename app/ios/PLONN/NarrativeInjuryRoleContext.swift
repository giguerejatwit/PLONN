import Foundation

/// Independent injury-role-context-v1 validation and deterministic projection. No I/O.
enum NarrativeInjuryRoleContext {
 typealias J = RichContextJSON
 typealias I = NarrativeInjuryContext
 static let contract = "injury-role-context-v1", policy = "injury-role-freshness-v1"
 static let roles:[String:Set<String>] = ["NFL":["quarterback","running_back","receiver","tight_end","offensive_line","defensive_line","linebacker","secondary"],"NBA":["guard","wing","big"],"MLB":["starting_pitcher","relief_pitcher","batter"],"NHL":["goalie","forward","defenseman"]]
 static let labels = ["quarterback":"quarterback","running_back":"rushing","receiver":"receiving","tight_end":"tight-end","offensive_line":"offensive-line","defensive_line":"defensive-front","linebacker":"linebacker","secondary":"secondary","guard":"guard rotation","wing":"wing rotation","big":"frontcourt rotation","starting_pitcher":"starting-pitching","relief_pitcher":"bullpen","batter":"batting-order","goalie":"goaltending","forward":"forward-line","defenseman":"defense-pair"]
 static let providers = ["site.api.espn.com":"ESPN","site.web.api.espn.com":"ESPN","statsapi.mlb.com":"MLB","www.mlb.com":"MLB","api.nhle.com":"NHL","www.nhl.com":"NHL","www.nfl.com":"NFL","www.seahawks.com":"Seattle Seahawks","www.denverbroncos.com":"Denver Broncos","official.nba.com":"NBA","ak-static.cms.nba.com":"NBA","www.nba.com":"NBA"]
 static let metrics:[String:(Set<String>,String,Double)] = ["passing_attempts":(["NFL"],"attempts",10000),"rushing_attempts":(["NFL"],"attempts",10000),"targets":(["NFL"],"targets",10000),"offensive_snaps":(["NFL"],"snaps",10000),"defensive_snaps":(["NFL"],"snaps",10000),"minutes":(["NBA"],"minutes",10000),"usage_rate":(["NBA"],"percent",100),"shot_attempts":(["NBA"],"attempts",10000),"pitches":(["MLB"],"pitches",10000),"plate_appearances":(["MLB"],"plate appearances",10000),"batters_faced":(["MLB"],"batters faced",10000),"time_on_ice":(["NHL"],"minutes",10000),"shots_faced":(["NHL"],"shots",10000),"starts":(["NFL","NBA","MLB","NHL"],"starts",200)]
 static func identity(_ value:J,_ sport:String,team:Bool=false)throws {try I.check(I.matches(value.string,"(?:espn:"+sport.lowercased()+":"+(team ? "team":"player")+":[0-9]{1,15}|"+sport.lowercased()+":[a-z0-9-]{1,50})"))}
 static func factID(_ f:J)->String {"role:"+I.hash(f.removing(["id"])).prefix(24)}
 static func sourceURL(_ value:J,_ sport:String)throws {
  try I.text(value,600);let raw=value.string
  guard let u=URLComponents(string:raw),u.scheme == "https",let host=u.host,host == host.lowercased(),u.user == nil,u.password == nil,u.port == nil,u.fragment == nil,!raw.contains(where:{$0.isWhitespace}) else {throw SharedMatchupReportError.invalidResponse}
  func query(_ rules:[String:String])throws->[URLQueryItem] {
   let q=u.queryItems ?? [];try I.check(Set(q.map(\.name)).count == q.count)
   if let rawQuery=u.percentEncodedQuery,!rawQuery.isEmpty {try I.check(rawQuery.split(separator:"&",omittingEmptySubsequences:false).allSatisfy{$0.contains("=") && !$0.isEmpty})}
   for pair in q {guard let rule=rules[pair.name],let value=pair.value else {throw SharedMatchupReportError.invalidResponse};try I.check(I.matches(value,rule))};return q
  }
  if ["site.api.espn.com","site.web.api.espn.com"].contains(host) {
   guard let path=["NFL":"football/nfl","NBA":"basketball/nba","MLB":"baseball/mlb","NHL":"hockey/nhl"][sport] else {throw SharedMatchupReportError.invalidResponse}
   try I.check(I.matches(u.percentEncodedPath,"/apis/(?:site/v2/sports/"+path+"/(?:summary|injuries|teams/[0-9]+/roster)|common/v3/sports/"+path+"/athletes/[0-9]+/(?:stats|overview))"))
   let q=try query(["event":"[0-9]{1,15}","season":"[0-9]{1,15}","seasontype":"[0-9]{1,15}"])
   try I.check(u.path.hasSuffix("/summary") ? q.count == 1 && q[0].name == "event" : !q.contains{$0.name == "event"})
  } else if host == "statsapi.mlb.com" && sport == "MLB" {try I.check(I.matches(u.percentEncodedPath,"/api/v1/game/[0-9]+/boxscore") && (u.query == nil || u.query == ""))}
  else if host == "api.nhle.com" && sport == "NHL" {try I.check(["/stats/rest/en/goalie/summary","/stats/rest/en/skater/summary"].contains(u.path));_ = try query(["cayenneExp":"seasonId=20[0-9]{6} and gameTypeId=[123]","isAggregate":"(?:true|false)","isGame":"(?:true|false)","start":"[0-9]{1,4}","limit":"(?:-1|[0-9]{1,4})"])}
  else {try I.sourceURL(value,sport)}
 }
 static func validate(_ p:J,now:Date)throws->J {
  let clock=now.timeIntervalSince1970;try I.check(clock.isFinite && p.canonical.utf8.count<=12288)
  try p.keys(["contractVersion","event","sources","facts","coverage"]);try I.check(p["contractVersion"] == .string(contract))
  let e=p["event"],sport=e["sport"].string;try e.keys(["sport","eventID","homeTeam","awayTeam","startISO"])
  try I.check(roles[sport] != nil && I.matches(e["eventID"].string,"espn:[0-9]{1,15}"));try I.text(e["homeTeam"],100);try I.text(e["awayTeam"],100);try I.check(e["homeTeam"].string.lowercased() != e["awayTeam"].string.lowercased());_ = try I.time(e["startISO"])
  let ss=try I.list(p["sources"]);try I.check(ss.count<=8);var sources:[String:J]=[:]
  for s in ss {
   try s.keys(["id","provider","url","issuedAt","retrievedAt","checkedAt","contentSHA256"]);try I.text(s["id"],80);try I.text(s["provider"],80);try sourceURL(s["url"],sport)
   try I.check(providers[URLComponents(string:s["url"].string)!.host!] == s["provider"].string && sources[s["id"].string] == nil && I.matches(s["contentSHA256"].string,"[a-f0-9]{64}"))
   try I.check(I.time(s["retrievedAt"])<=I.time(s["checkedAt"]) && I.time(s["checkedAt"])<=clock)
   if s["issuedAt"] != .null {try I.check(I.time(s["issuedAt"])<=I.time(s["retrievedAt"]))};sources[s["id"].string]=s
  }
  let facts=try I.list(p["facts"]);try I.check(facts.count<=12);var ids=Set<String>(),identities:[String:Set<String>]=[:]
  for f in facts {
   try f.keys(["id","teamID","teamName","playerID","playerName","eventID","eventStartISO","kind","role","detail","reportedAt","timeBasis","sourceIDs"])
   try identity(f["playerID"],sport);try identity(f["teamID"],sport,team:true);try I.text(f["playerName"],100);try I.text(f["teamName"],100)
   try I.check([e["homeTeam"],e["awayTeam"]].contains(f["teamName"]) && roles[sport]!.contains(f["role"].string));identities[f["teamID"].string,default:[]].insert(f["teamName"].string)
   try I.check(f["id"].string == factID(f) && ids.insert(f["id"].string).inserted && I.matches(f["eventID"].string,"espn:[0-9]{1,15}"));_ = try I.time(f["eventStartISO"])
   let refs=try I.list(f["sourceIDs"]);try I.check((1...4).contains(refs.count) && Set(refs.map(\.string)).count == refs.count && refs.allSatisfy{sources[$0.string] != nil})
   if f["reportedAt"] != .null {for ref in refs {try I.check(I.time(f["reportedAt"])<=I.time(sources[ref.string]!["retrievedAt"]))}}
   let basis=f["timeBasis"].string,d=f["detail"];try I.check(["source_report","historical_window","observed_position"].contains(basis))
   switch f["kind"].string {
   case "role":try d.keys(["designation"]);try I.check(["starter","backup","rotation","position","team_leader","league_leader"].contains(d["designation"].string));try I.check(basis == "source_report" || basis == "observed_position" && d["designation"] == .string("position"))
   case "usage":
    try d.keys(["metric","value","unit","season","phase","sample","windowStart","windowEnd"]);try I.check(basis == "historical_window")
    guard let metric=metrics[d["metric"].string],let number=d["value"].number,let value=Double(number),let season=d["season"].integer,let count=d["sample"]["count"].integer else {throw SharedMatchupReportError.invalidResponse}
    try I.check(metric.0.contains(sport) && d["unit"].string == metric.1 && value.isFinite && value>=0 && value<=metric.2 && (2000...2100).contains(season) && ["preseason","regular","postseason"].contains(d["phase"].string))
    try d["sample"].keys(["count","unit"]);try I.check((1...200).contains(count) && d["sample"]["unit"] == .string("games"));try I.check(I.time(d["windowStart"])<I.time(d["windowEnd"]))
    for ref in refs {try I.check(I.time(d["windowEnd"])<=I.time(sources[ref.string]!["retrievedAt"]))}
   case "replacement":try d.keys(["playerID","playerName","designation","basis"]);try I.check(basis == "source_report" && ["confirmed","probable"].contains(d["designation"].string) && d["basis"] == .string("direct_event_designation"));try identity(d["playerID"],sport);try I.text(d["playerName"],100);try I.check(d["playerID"] != f["playerID"] && d["playerName"] != f["playerName"])
   default:throw SharedMatchupReportError.invalidResponse
   }
   if basis == "source_report" && f["reportedAt"] != .null {try I.check(refs.contains{sources[$0.string]!["issuedAt"] == f["reportedAt"]})}
   if basis == "observed_position" {try I.check(f["reportedAt"] == .null)}
  }
  try I.check(identities.values.allSatisfy{$0.count == 1});let cs=try I.list(p["coverage"]);try I.check(cs.count == 2);var seen=Set<String>()
  for c in cs {try c.keys(["teamName","state","reason"]);try I.check([e["homeTeam"],e["awayTeam"]].contains(c["teamName"]) && seen.insert(c["teamName"].string).inserted && ["partial","missing"].contains(c["state"].string) && ["partial_evidence","not_supplied","identity_unresolved","no_current_role","source_unavailable"].contains(c["reason"].string));try I.check(c["state"] != .string("missing") || !facts.contains{$0["teamName"] == c["teamName"]})}
  return p.setting("sources",.list(ss.sorted{$0["id"].string<$1["id"].string})).setting("facts",.list(facts.sorted{$0["id"].string<$1["id"].string})).setting("coverage",.list(cs.sorted{$0["teamName"].string<$1["teamName"].string}))
 }
 static func fingerprint(_ p:J)throws->String {
  let checks=try I.list(p["sources"]).map{try I.time($0["checkedAt"])};let v=try validate(p,now:Date(timeIntervalSince1970:checks.max() ?? 0))
  return I.hash(.object(["policy":.string(policy),"packet":v.setting("sources",.list(v["sources"].array.map{$0.removing(["retrievedAt","checkedAt","contentSHA256"])}))]))
 }
 static func rank(_ f:J)->Int {f["kind"] == .string("role") ? ["starter":0,"backup":0,"rotation":0,"team_leader":1,"league_leader":1,"position":2][f["detail"]["designation"].string]! : f["kind"] == .string("usage") ? 3:4}
 static func ranked(_ a:J,_ b:J)->Bool {rank(a) == rank(b) ? a["id"].string<b["id"].string : rank(a)<rank(b)}
 static func project(_ packet:J,now:Date)throws->J {
  let p=try validate(packet,now:now),e=p["event"],clock=now.timeIntervalSince1970,sport=e["sport"].string
  let sources=Dictionary(uniqueKeysWithValues:p["sources"].array.map{($0["id"].string,$0)});var states:[String:String]=[:],identities:[String:Set<String>]=[:],names:[String:Set<String>]=[:],current:[String:Set<String>]=[:]
  func key(_ f:J,_ fields:[String])->String {J.list(fields.map{f[$0]}).canonical}
  for f in p["facts"].array {
   let ss=f["sourceIDs"].array.map{sources[$0.string]!};var state="fresh"
   if try ss.contains(where:{try clock-I.time($0["checkedAt"])>=21600}) {state="source_stale"}
   else if f["timeBasis"] == .string("source_report") {if f["reportedAt"] == .null {state="role_update_unknown"} else if try clock-I.time(f["reportedAt"]) >= (["NFL","MLB"].contains(sport) ? 86400:21600) {state="role_stale"}}
   else if f["timeBasis"] == .string("observed_position") {if try ss.contains(where:{try clock-I.time($0["retrievedAt"])>=21600}) {state="role_stale"}}
   else if try clock-I.time(f["detail"]["windowEnd"]) >= (sport == "NFL" ? 14:7)*86400 {state="usage_stale"}
   if state == "fresh" && (f["eventID"] != e["eventID"] || f["eventStartISO"] != e["startISO"]) {state="different_event"}
   if try state == "fresh" && clock >= I.time(e["startISO"]) {state="event_started"}
   states[f["id"].string]=state
   if state == "fresh" {identities[key(f,["teamID","playerName"]),default:[]].insert(f["playerID"].string);names[key(f,["teamID","playerID"]),default:[]].insert(f["playerName"].string)
    if f["kind"] != .string("usage") {current[key(f,["teamID","playerID","kind","role"]),default:[]].insert(f["detail"].canonical)}
   }
  }
  var eligible:[J]=[],excluded:[J]=[]
  for f in p["facts"].array {
   var state=states[f["id"].string]!;let details=current[key(f,["teamID","playerID","kind","role"])] ?? []
   if state == "fresh" && (identities[key(f,["teamID","playerName"])]!.count>1 || names[key(f,["teamID","playerID"])]!.count>1) {state="identity_unresolved"}
   if state == "fresh" && (f["kind"] == .string("replacement") && details.count>1 || f["kind"] == .string("role") && details.contains("{\"designation\":\"starter\"}") && details.contains("{\"designation\":\"backup\"}")) {state="conflicting_role"}
   if state == "fresh" {eligible.append(f.setting("freshness",.string("fresh")).setting("historicalOnly",.bool(f["kind"] == .string("usage"))))} else {excluded.append(.object(["id":f["id"],"teamName":f["teamName"],"reason":.string(state)]))}
  }
  let fp=try fingerprint(p),state = !eligible.isEmpty ? "partial":excluded.contains{$0["reason"].string.contains("stale")} ? "stale":"missing"
  let result:J = .object(["contractVersion":.string(contract),"policyVersion":.string(policy),"event":e,"fingerprint":.string(fp),"eligibilityFingerprint":.string(I.hash(.list([.string(policy),.string(fp),.list(eligible.map{$0["id"]}),.list(excluded)]))),"state":.string(state),"facts":.list(eligible),"sources":p["sources"],"coverage":p["coverage"],"excluded":.list(excluded)])
  try I.check(result.canonical.utf8.count<=16384);return result
 }
 static func model(_ packet:J,now:Date,maxFacts:Int=6,maxBytes:Int=4096)throws->J {
  try I.check((0...6).contains(maxFacts) && (1024...4096).contains(maxBytes));let p=try project(packet,now:now);var selected=Array(p["facts"].array.sorted(by:ranked).prefix(maxFacts))
  while true {
   let used=Set(selected.flatMap{$0["sourceIDs"].array.map(\.string)}),ids=Set(selected.map{$0["id"].string}),omitted=p["facts"].array.filter{!ids.contains($0["id"].string)}.map{$0["id"]}
   let r=p.setting("facts",.list(selected)).setting("sources",.list(p["sources"].array.filter{used.contains($0["id"].string)})).setting("omissions",.object(["count":.number(String(omitted.count)),"factIDs":.list(omitted)]))
   if r.canonical.utf8.count<=maxBytes {return r};try I.check(!selected.isEmpty);selected.removeLast()
  }
 }
 static func auditedClaims(injury packet:J,roles rolePacket:J,now:Date)throws->J {
  let injury=try I.project(packet,now:now),roles=try model(rolePacket,now:now),event=injury["event"]
  try I.check(event == roles["event"]);let teams=Dictionary(uniqueKeysWithValues:injury["teams"].array.map{($0["id"].string,$0["name"])});var grouped:[String:[(J,[J],[J])]]=[:],gaps:[J]=[]
  for r in injury["observations"].array where !["available","probable"].contains(r["statusKind"].string) {
   let matches=roles["facts"].array.filter{$0["teamID"] == r["teamID"] && $0["teamName"] == teams[r["teamID"].string] && $0["playerID"] == r["playerID"] && $0["playerName"] == r["playerName"]}
   let substantive=matches.filter{$0["kind"] != .string("replacement")}
   if substantive.isEmpty {gaps.append(.object(["injuryID":r["claimID"],"reason":.string("no_eligible_exact_role")]))}
   else {grouped[r["teamID"].string,default:[]].append((r,substantive,matches.filter{$0["kind"] == .string("replacement")}))}
  }
  for e in injury["excluded"].array {gaps.append(.object(["injuryID":e["id"],"reason":e["reason"]]))}
  var claims:[J]=[]
  func confirmed(_ r:J)->Bool {r["eventStatusConfirmed"] == .bool(true) && r["statusKind"] == .string("out")}
  func direct(_ facts:[J])->Bool {facts.contains{$0["kind"] == .string("role") && $0["detail"]["designation"] != .string("position")}}
  for teamID in grouped.keys.sorted() {
   let item=grouped[teamID]!.sorted {a,b in
    if confirmed(a.0) != confirmed(b.0) {return confirmed(a.0)}
    if direct(a.1) != direct(b.1) {return direct(a.1)}
    return a.0["claimID"].string<b.0["claimID"].string
   }[0]
   let r=item.0,f=item.1.sorted(by:ranked)[0],role=labels[f["role"].string]!,name=r["playerName"].string,isConfirmed=confirmed(r)
   var words=isConfirmed ? "\(name) is confirmed Out for this game. ":"\(name) is reported \(r["reportedStatus"].string). ",uncertainty:String
   if f["kind"] == .string("usage") {
    let metric=f["detail"]["metric"].string.replacingOccurrences(of:"_",with:" ")
    words += "Historical \(metric) evidence suggests possible relevance to \(role) depth if unavailable. "
    uncertainty="Historical usage does not establish a current assignment or the size of any matchup effect."
   } else if f["detail"]["designation"] == .string("backup") {
    words += "The source identifies a backup \(role) role; an absence would affect depth, not establish a change to the starter. "
    uncertainty="No replacement promotion or quantified matchup effect is established."
   } else if f["detail"]["designation"] == .string("position") {
    words += "The listed position makes this a \(role) depth concern if unavailable. "
    uncertainty="A listed position alone does not establish a starting assignment or usage."
   } else {
    let descriptor=["starter":"starting","rotation":"rotation","team_leader":"team-leading","league_leader":"league-leading"][f["detail"]["designation"].string]!
    words += "The source identifies a \(descriptor) \(role) role. "
    words += isConfirmed ? "The team must cover that role without this player. ":"If unavailable, the team would need another option in that role. "
    uncertainty="The size of any scoring, pace or win-probability effect is not established."
   }
   var used=[f];let replacements=item.2.filter{$0["role"] == f["role"]}.sorted{$0["id"].string<$1["id"].string}
   if let replacement=replacements.first {used.append(replacement);let d=replacement["detail"];words += "\(d["playerName"].string) is the source-designated \(d["designation"].string) replacement for this role. "}
   if !isConfirmed {uncertainty="This is not a confirmed game absence. "+uncertainty}
   let side=f["teamName"] == event["homeTeam"] ? "home":"away"
   let claimID=side+":availability:"+I.hash(.list([r["claimID"],.list(used.map{$0["id"]})])).prefix(16)
   let refs=Set([r["sourceID"].string]+used.flatMap{$0["sourceIDs"].array.map(\.string)}).sorted().map{J.string($0)}
   let samples=used.filter{$0["kind"] == .string("usage")}.map{$0["detail"].setting("roleEvidenceID",$0["id"])}
   claims.append(.object(["id":.string(claimID),"dimension":.string(side+"_availability"),"label":.string("Availability and role"),"kind":.string("conditional"),"text":.string(words.trimmingCharacters(in:.whitespacesAndNewlines)),"factIDs":.list([]),"injuryIDs":.list([r["claimID"]]),"roleEvidenceIDs":.list(used.map{$0["id"]}),"sourceIDs":.list(refs),"samples":.list(samples),"uncertainty":.string(uncertainty)]))
  }
  return .object(["claims":.list(claims),"gaps":.list(gaps),"roleContext":roles])
 }
}
