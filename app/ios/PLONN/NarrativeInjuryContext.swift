import Foundation
import CryptoKit

/// Pure, independent injury-input-v2 validation. No fetching or inference.
enum NarrativeInjuryContext {
 typealias J = RichContextJSON
 static let policy = "injury-model-freshness-v1"
 static func check(_ ok: Bool) throws { if !ok { throw SharedMatchupReportError.invalidResponse } }
 static func matches(_ s: String, _ pattern: String) -> Bool { s.range(of: "\\A(?:" + pattern + ")\\z", options: .regularExpression) != nil }
 static func text(_ j: J, _ max: Int) throws { guard case .string(let s) = j else { throw SharedMatchupReportError.invalidResponse }; try check(!s.isEmpty && s.unicodeScalars.count <= max && s == s.trimmingCharacters(in: .whitespacesAndNewlines) && s == s.precomposedStringWithCanonicalMapping && !s.unicodeScalars.contains { [.control,.format,.surrogate,.privateUse,.unassigned].contains($0.properties.generalCategory) }) }
 static func time(_ j: J) throws -> Double {
  try text(j,40)
  try check(matches(j.string, #"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]+)?)?(?:Z|[+-][0-9]{2}:[0-9]{2})"#))
  let f = ISO8601DateFormatter(); f.formatOptions = [.withInternetDateTime,.withFractionalSeconds]
  let g = ISO8601DateFormatter(); g.formatOptions = [.withInternetDateTime]
  let minute = DateFormatter(); minute.locale = Locale(identifier:"en_US_POSIX"); minute.dateFormat = "yyyy-MM-dd'T'HH:mmXXXXX"; minute.isLenient = false
  guard let d = f.date(from:j.string) ?? g.date(from:j.string) ?? minute.date(from:j.string) else { throw SharedMatchupReportError.invalidResponse }; return d.timeIntervalSince1970
 }
 static func list(_ j:J) throws -> [J] { guard case .list(let a)=j else {throw SharedMatchupReportError.invalidResponse};return a }
 static func hash(_ j:J)->String { SHA256.hash(data:Data(j.canonical.utf8)).map {String(format:"%02x",$0)}.joined() }
 static func id(_ r:J)->String { "injury-observation:" + hash(.list([r["teamID"],r["playerID"],r["sourceID"],r["scope"],r["eventID"],r["eventStartISO"]])).prefix(24) }
 static let games:[String:Set<String>] = ["NFL":["Out","Questionable","Doubtful"],"NBA":["Out","Questionable","Doubtful","Probable","Available"],"MLB":["Out","Day-to-Day","Day to day"],"NHL":["Out","Day-to-Day","Day to day","day to day","week to week"]]
 static let rosters:[String:Set<String>] = ["NFL":["Injured Reserve","IR","PUP","physically unable to perform list"],"NBA":[],"MLB":["Injured 7-Day","Injured 10-Day","Injured 15-Day","Injured 60-Day"],"NHL":["Injured Reserve","injured reserve","IR","Long-Term Injured Reserve","long-term injured reserve","LTIR"]]
 static let practice:Set<String> = ["Did not participate","Limited participation","Full participation","Did Not Participate In Practice","Limited Participation in Practice","Full Participation in Practice"]
 static func official(_ sport:String,_ status:String,_ scope:String)throws->String {
  if scope == "practice" {try check(sport == "NFL" && practice.contains(status));return "practice"}
  if scope == "game" {try check(games[sport]!.contains(status)); return ["Out":"out","Questionable":"questionable","Doubtful":"doubtful","Probable":"probable","Available":"available","week to week":"week_to_week"][status] ?? "day_to_day"}
  try check(scope == "roster" && rosters[sport]!.contains(status))
  if sport == "MLB" {return "il_" + status.filter(\.isNumber)}
  if ["PUP","physically unable to perform list"].contains(status){return "pup"}
  return status.lowercased().contains("long") || status == "LTIR" ? "ltir":"ir"
 }
 static func sourceURL(_ value:J,_ sport:String)throws {
  try text(value,600); let s=value.string
  guard let u=URLComponents(string:s),u.scheme == "https",let host=u.host,host == host.lowercased(),u.user == nil,u.password == nil,u.port == nil,u.fragment == nil,!s.contains(where:{$0.isWhitespace}) else {throw SharedMatchupReportError.invalidResponse}
  let rules:[String:[String:String]] = [
   "NFL":["www.nfl.com":#"/injuries/(?:\d{4}/(?:reg|post|pre)\d+/)?"#,"www.seahawks.com":#"/news/[a-z0-9-]+"#,"www.denverbroncos.com":#"/news/(?:injuryreport/)?[a-z0-9-]+"#],
   "NBA":["official.nba.com":#"/nba-injury-report-\d{4}-\d{2}-season/"#,"ak-static.cms.nba.com":#"/referee/injury/Injury-Report_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}(?:AM|PM)\.pdf"#,"www.nba.com":#"/[a-z]+/news/[a-z0-9-]+"#],
   "MLB":["statsapi.mlb.com":#"/api/v1/teams/\d+/roster"#,"www.mlb.com":#"/news/[a-z0-9-]+"#],"NHL":["www.nhl.com":#"/(?:[a-z]+/)?news/[a-z0-9-]+"#]]
  guard let pattern=rules[sport]?[host] else {throw SharedMatchupReportError.invalidResponse}; try check(matches(u.percentEncodedPath,pattern))
  if host == "statsapi.mlb.com" {try check(u.queryItems == [URLQueryItem(name:"rosterType",value:"fullRoster"),URLQueryItem(name:"hydrate",value:"person(currentTeam)")])} else {try check(u.query == nil || u.query == "")}
 }
 static func validate(_ p:J,now:Date)throws->J {
  let clock=now.timeIntervalSince1970;try check(clock.isFinite && p.canonical.utf8.count <= 24576)
  try p.keys(["contractVersion","event","teams","sources","observations","coverage"]);try check(p["contractVersion"] == .string("injury-input-v2"))
  let e=p["event"],sport=e["sport"].string
  try e.keys(["sport","eventID","homeTeam","awayTeam","startISO"]);try check(games[sport] != nil && matches(e["eventID"].string,#"espn:[0-9]{1,15}"#));try text(e["homeTeam"],100);try text(e["awayTeam"],100);try check(e["homeTeam"].string.lowercased() != e["awayTeam"].string.lowercased());_ = try time(e["startISO"])
  let ts=try list(p["teams"]);try check(ts.count == 2); var teams:[String:String]=[:]
  for t in ts {try t.keys(["id","name"]);try text(t["id"],80);try text(t["name"],100);try check(teams[t["id"].string] == nil);teams[t["id"].string]=t["name"].string}
  try check(Set(teams.values) == Set([e["homeTeam"].string,e["awayTeam"].string]))
  let ss=try list(p["sources"]);try check(ss.count<=8);var sources:[String:J]=[:]
  for s in ss {
   try s.keys(["id","provider","url","sourceIssuedAt","feedUpdatedAt","retrievedAt","checkedAt","contentSHA256"]);try text(s["id"],80);try text(s["provider"],80)
   if s["provider"] == .string("ESPN") { let path=["NFL":"football/nfl","NBA":"basketball/nba","MLB":"baseball/mlb","NHL":"hockey/nhl"][sport]!;try check(s["url"] == .string("https://site.api.espn.com/apis/site/v2/sports/"+path+"/injuries") && s["sourceIssuedAt"] == .null) } else {try sourceURL(s["url"],sport);try check(s["feedUpdatedAt"] == .null)}
   try check(sources[s["id"].string] == nil);let retrieved=try time(s["retrievedAt"]),checked=try time(s["checkedAt"]);try check(retrieved<=checked && checked<=clock)
   for k in ["sourceIssuedAt","feedUpdatedAt"] where s[k] != .null {try check(time(s[k])<=retrieved)}
   try check(matches(s["contentSHA256"].string,"[a-f0-9]{64}"));sources[s["id"].string]=s
  }
  let rows=try list(p["observations"]);try check(rows.count<=48);var seen=Set<String>()
  for r in rows {
   try r.keys(["id","teamID","playerID","identityKind","playerName","position","reportedStatus","statusKind","scope","eventID","eventStartISO","updatedAt","timeBasis","sourceID","estimated"])
   guard teams[r["teamID"].string] != nil,let s=sources[r["sourceID"].string] else {throw SharedMatchupReportError.invalidResponse}
   for k in ["id","playerID","playerName"] {try text(r[k],100)};if r["position"] != .null {try text(r["position"],40)}
   try check(["provider_id","source_name"].contains(r["identityKind"].string) && r["id"].string == id(r) && seen.insert(r["id"].string).inserted)
   guard case .bool(let estimated)=r["estimated"] else {throw SharedMatchupReportError.invalidResponse};try check(!estimated || r["scope"] == .string("practice"))
   if s["provider"] == .string("ESPN") {
    var statuses:[String:[String]] = ["Out":["out","general"]]
    if sport == "NFL" {statuses.merge(["Questionable":["questionable","general"],"Doubtful":["doubtful","general"],"Injured Reserve":["ir","roster"],"Suspension":["suspension","roster"],"Reserve-Ret":["reserve_retired","roster"]]){a,_ in a}}
    else {statuses["Day-To-Day"]=["day_to_day","general"]}
    if sport == "NHL" {statuses["Injured Reserve"]=["ir","roster"];statuses["Suspension"]=["suspension","roster"]}
    if sport == "MLB" {for (a,b) in ["7-Day IL":"il_7","10-Day-IL":"il_10","15-Day-IL":"il_15","60-Day-IL":"il_60","suspension":"suspension","paternity":"paternity"] {statuses[a]=[b,"roster"]}}
    try check(statuses[r["reportedStatus"].string] == [r["statusKind"].string,r["scope"].string] && r["identityKind"] == .string("provider_id"))
    try check(matches(r["playerID"].string,"espn:"+sport.lowercased()+":player:[0-9]{1,15}") && matches(r["teamID"].string,"espn:"+sport.lowercased()+":team:[0-9]{1,15}"))
    try check(r["eventID"] == .null && r["eventStartISO"] == .null && r["timeBasis"] == .string("player_update"))
   } else {
    try check(r["statusKind"].string == official(sport,r["reportedStatus"].string,r["scope"].string))
    try check(["source_report","current_roster"].contains(r["timeBasis"].string))
    if r["timeBasis"] == .string("current_roster") {try check(sport == "MLB" && s["url"].string.hasPrefix("https://statsapi.mlb.com/") && r["scope"] == .string("roster") && r["updatedAt"] == .null)} else {try check(r["updatedAt"] == s["sourceIssuedAt"])}
    if r["identityKind"] == .string("provider_id") {try check(matches(r["playerID"].string,sport.lowercased()+":[a-z0-9-]+"))} else {try check(r["playerID"].string == "source-name:"+hash(.list([.string(sport),r["teamID"],r["playerName"]])).prefix(24))}
   }
   if r["updatedAt"] != .null {try check(time(r["updatedAt"])<=time(s["retrievedAt"]))}
   if r["eventID"] == .null {try check(r["eventStartISO"] == .null)} else {try check(matches(r["eventID"].string,"espn:[0-9]{1,15}"));_ = try time(r["eventStartISO"])}
  }
  let cs=try list(p["coverage"]);try check(cs.count==2);var covered=Set<String>()
  for c in cs {
   try c.keys(["teamID","state","reason","sourceIDs","excludedRows"]);let team=c["teamID"].string;try check(teams[team] != nil && covered.insert(team).inserted)
   try check(["partial","missing"].contains(c["state"].string) && ["partial_source","team_not_reported","source_unavailable","no_supported_observations","unsupported_rows"].contains(c["reason"].string))
   guard let n=c["excludedRows"].integer else {throw SharedMatchupReportError.invalidResponse};try check((0...10000).contains(n))
   let ids=try list(c["sourceIDs"]);try check(ids.allSatisfy{if case .string = $0{return true};return false});let keys=Set(ids.map(\.string));try check(keys.count==ids.count && keys.isSubset(of:Set(sources.keys)))
   let teamRows=rows.filter{$0["teamID"].string==team};try check(Set(teamRows.map{$0["sourceID"].string}).isSubset(of:keys));try check(c["state"] == .string("missing") ? teamRows.isEmpty : !ids.isEmpty)
  }
  var result=p
  for key in ["teams","sources","observations"] {result=result.setting(key,.list(p[key].array.sorted{$0["id"].string<$1["id"].string}))}
  result=result.setting("coverage",.list(cs.sorted{$0["teamID"].string<$1["teamID"].string}.map{$0.setting("sourceIDs",.list($0["sourceIDs"].array.sorted{$0.string<$1.string}))}))
  return result
 }
 static func fingerprint(_ p:J)->String {
  let stripped=p.setting("sources",.list(p["sources"].array.map{$0.removing(["retrievedAt","checkedAt","feedUpdatedAt","contentSHA256"])}))
  return hash(.object(["policy":.string(policy),"evidence":stripped]))
 }
 static func project(_ packet:J,now:Date)throws->J {
  let p=try validate(packet,now:now),e=p["event"],sport=e["sport"].string,clock=now.timeIntervalSince1970
  let sources=Dictionary(uniqueKeysWithValues:p["sources"].array.map{($0["id"].string,$0)})
  var states:[String:String]=[:],identities:[String:Set<String>]=[:],names:[String:Set<String>]=[:],conflicts:[String:Set<String>]=[:]
  func key(_ a:J,_ b:J)->String {J.list([a,b]).canonical}
  for r in p["observations"].array {
   let s=sources[r["sourceID"].string]!
   var state="fresh"
   if clock - (try time(s["checkedAt"])) >= 21600 {state="source_stale"}
   else if r["timeBasis"] == .string("current_roster") {if clock - (try time(s["retrievedAt"])) >= 21600 {state="observation_stale"}}
   else if r["updatedAt"] == .null {state="update_unknown"}
   else if clock - (try time(r["updatedAt"])) >= (["NFL","MLB"].contains(sport) ? 86400:21600) {state="observation_stale"}
   if try state == "fresh" && clock >= time(e["startISO"]) {state="event_started"}
   if state == "fresh" && r["eventID"] != .null && (r["eventID"] != e["eventID"] || r["eventStartISO"] != e["startISO"]) {state="different_event"}
   states[r["id"].string]=state
   if state == "fresh" {identities[key(r["teamID"],r["playerName"]),default:[]].insert(r["playerID"].string);names[key(r["teamID"],r["playerID"]),default:[]].insert(r["playerName"].string)
    if r["scope"] != .string("practice") {conflicts[key(r["teamID"],r["playerID"]),default:[]].insert(r["statusKind"].string)}
   }
  }
  var eligible:[J]=[],excluded:[J]=[]
  for r in p["observations"].array {
   var state=states[r["id"].string]!
   if state == "fresh" && (identities[key(r["teamID"],r["playerName"])]?.count ?? 0)>1 {state="identity_unresolved"}
   if state == "fresh" && (names[key(r["teamID"],r["playerID"])]?.count ?? 0)>1 {state="identity_unresolved"}
   if state == "fresh" && (conflicts[key(r["teamID"],r["playerID"])]?.count ?? 0)>1 {state="conflicting_status"}
   if state != "fresh" {excluded.append(.object(["id":r["id"],"teamID":r["teamID"],"sourceID":r["sourceID"],"reason":.string(state)]));continue}
   let confirmed=r["scope"] == .string("game") && r["eventID"] == e["eventID"] && r["eventStartISO"] == e["startISO"]
   eligible.append(r.setting("assertion",.string((confirmed ? "confirmed_":"reported_")+r["statusKind"].string)).setting("eventStatusConfirmed",.bool(confirmed)).setting("freshness",.string("fresh")).setting("claimID",.string("injury-claim:"+hash(.list([r,sources[r["sourceID"].string]!["url"],.string(policy)])).prefix(24))))
  }
  let coverage=p["coverage"].array.map { c -> J in
   let rows=eligible.filter{$0["teamID"]==c["teamID"]}, ex=excluded.filter{$0["teamID"]==c["teamID"]}
   let state = !rows.isEmpty ? "partial" : ex.contains{["source_stale","observation_stale"].contains($0["reason"].string)} ? "stale":"missing"
   return c.setting("state",.string(state)).setting("eligibleCount",.number(String(rows.count))).setting("excludedObservationCount",.number(String(ex.count)))
  }
  let state = !eligible.isEmpty ? "partial" : coverage.contains{$0["state"] == .string("stale")} ? "stale":"missing", fp=fingerprint(p)
  let contextFP=hash(.list([.string(policy),.string(fp),.list(eligible.map{.list([$0["claimID"],$0["assertion"]])}),.list(excluded),.list(coverage)]))
  let result:J = .object(["contractVersion":.string("injury-model-context-v1"),"policyVersion":.string(policy),"event":e,"teams":p["teams"],"injuryFingerprint":.string(fp),"contextFingerprint":.string(contextFP),"state":.string(state),"observations":.list(eligible),"excluded":.list(excluded),"coverage":.list(coverage),"sources":p["sources"]])
  try check(result.canonical.utf8.count<=24576);return result
 }
 static func model(_ packet:J,now:Date,maxObservations:Int=6,maxBytes:Int=6144)throws->J {
  try check((0...24).contains(maxObservations) && (2048...12288).contains(maxBytes))
  let p=try project(packet,now:now)
  let ordered=p["observations"].array.sorted {a,b in
   if a["eventStatusConfirmed"] != b["eventStatusConfirmed"] {return a["eventStatusConfirmed"] == .bool(true)}
   if a["teamID"] != b["teamID"] {return a["teamID"].string<b["teamID"].string};return a["id"].string<b["id"].string
  }
  var selected=Array(ordered.prefix(maxObservations))
  while true {
   let exclusions=p["coverage"].array.map { c -> J in
    let reasons=Set(p["excluded"].array.filter{$0["teamID"]==c["teamID"]}.map{$0["reason"].string}).sorted()
    let count=ordered.filter{$0["teamID"]==c["teamID"]}.count-selected.filter{$0["teamID"]==c["teamID"]}.count
    return .object(["teamID":c["teamID"],"reasons":.list(reasons.map{.string($0)}),"excludedCount":c["excludedObservationCount"],"promptOmittedCount":.number(String(count))])
   }
   let result=p.removing(["observations","sources","excluded"]).setting("observations",.list(selected)).setting("sources",p["sources"]).setting("exclusions",.list(exclusions)).setting("interpretation",.string("General reports are not exact-game designations. Practice is not game status. Missing or excluded data never implies availability. No severity, role contribution, replacement starter, scoring adjustment or guaranteed matchup impact is supplied."))
   if result.canonical.utf8.count<=maxBytes {return result};try check(!selected.isEmpty);selected.removeLast()
  }
 }
}
