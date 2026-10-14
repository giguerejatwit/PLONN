import Foundation

/// Additive v5 validation. Rebuilds every selected sports and availability claim locally.
/// It does not execute inference or fetch evidence, and leaves the v4 reader unchanged.
enum MatchupNarrativeV5 {
 typealias J = RichContextJSON
 typealias V4 = MatchupNarrativeV4
 static func roleIdentity(_ view:J)->J {
  if view == .null { return .null }
  return .object(["fingerprint":view["fingerprint"],"eligibilityFingerprint":view["eligibilityFingerprint"],"factIDs":.list(view["facts"].array.map{$0["id"]}),"omissions":view["omissions"],"policy":.string("six-role-facts-4096-bytes-v1")])
 }
 static func prepare(_ context:J,now:Date,injuryContext:J = .null,injuryRoleContext:J = .null)throws->J {
  var plan=try V4.prepare(context,now:now,injuryContext:injuryContext)
  plan=plan.setting("claims",.list(plan["claims"].array.map{$0.setting("roleEvidenceIDs",.list([]))}))
   .setting("injuryRoleContext",.null).setting("roleStatusEvidence",.list([])).setting("injuryRoleGaps",.list([]))
  var missing=plan["missingContext"].array
  if injuryRoleContext == .null {
   missing.append(.string("Verified player role and replacement context is not supplied."))
   return plan.setting("missingContext",.list(missing))
  }
  try V4.require(injuryContext != .null)
  let result=try NarrativeInjuryRoleContext.auditedClaims(injury:injuryContext,roles:injuryRoleContext,now:now)
  let roleEvent=result["roleContext"]["event"]
  let normalized=roleEvent.setting("startISO",.string(RichSportsContext.iso(try RichSportsContext.time(roleEvent["startISO"]))))
  try V4.require(normalized==plan["event"])
  let claims=result["claims"].array
  try V4.require(claims.count<=2 && Set(claims.map{$0["dimension"].string}).count==claims.count)
  let wanted=Set(claims.flatMap{$0["injuryIDs"].array.map(\.string)})
  let injuries=try NarrativeInjuryContext.project(injuryContext,now:now)
  let evidence=injuries["observations"].array.filter{wanted.contains($0["claimID"].string)}.sorted{$0["claimID"].string<$1["claimID"].string}
  try V4.require(Set(evidence.map{$0["claimID"].string})==wanted)
  let required=plan["requiredDimensions"].array+claims.map{$0["dimension"]}
  try V4.require(required.count<=5)
  for side in ["home","away"] where !claims.contains(where:{$0["dimension"].string==side+"_availability"}) {
   missing.append(.string(plan["event"][side+"Team"].string+": No eligible exact status-and-role claim; missing evidence does not imply availability."))
  }
  return plan.setting("roleStatusEvidence",.list(evidence))
   .setting("injuryRoleContext",result["roleContext"]).setting("injuryRoleGaps",result["gaps"])
   .setting("claims",.list(plan["claims"].array+claims)).setting("requiredDimensions",.list(required))
   .setting("missingContext",.list(missing))
 }
 static func render(_ selection:J,plan:J)throws->J {
  var report=try V4.render(selection,plan:plan)
  for key in ["injuryRoleContext","roleStatusEvidence","injuryRoleGaps"] {report=report.setting(key,plan[key])}
  return report
 }
 static func validate(_ raw:Data,now:Date,versions:[String:String])throws->V4.Result {
  try V4.require(raw.count<=131072)
  var parser=RichContextJSONParser(raw);let wire=try parser.parse()
  try wire.keys(["contractVersion","versions","event","generatedAt","sourceFingerprint","context","injuryContext","injuryRoleContext","selection","report"])
  try V4.require(wire["contractVersion"].string=="matchup-narrative-v5" && wire["versions"] == .object(versions.mapValues(J.string)))
  let generated=try RichSportsContext.time(wire["generatedAt"]),start=try RichSportsContext.time(wire["event"]["startISO"])
  try V4.require(generated<=now && generated<start)
  let plan=try prepare(wire["context"],now:generated,injuryContext:wire["injuryContext"],injuryRoleContext:wire["injuryRoleContext"])
  let report=try render(wire["selection"],plan:plan)
  let fingerprint=RichSportsContext.digest(.object(["sports":plan["contextFingerprint"],"revision":plan["contextRevision"],"injury":V4.injuryIdentity(plan["injuryContext"]),"role":roleIdentity(plan["injuryRoleContext"]),"roleStatusIDs":.list(plan["roleStatusEvidence"].array.map{$0["claimID"]})]))
  let expected=wire.setting("event",plan["event"]).setting("generatedAt",.string(RichSportsContext.iso(generated))).setting("sourceFingerprint",.string(fingerprint)).setting("report",report)
  try V4.require(raw==Data(expected.canonical.utf8))
  let current=try prepare(wire["context"],now:now,injuryContext:wire["injuryContext"],injuryRoleContext:wire["injuryRoleContext"])
  try V4.require(try render(wire["selection"],plan:current)==report)
  let event=try JSONDecoder().decode(SharedMatchupEvent.self,from:Data(wire["event"].canonical.utf8))
  return V4.Result(rawData:raw,event:event,generatedAt:wire["generatedAt"].string,claims:report["claims"].array.map(V4.Claim.init),missingContext:report["missingContext"].array.map(\.string),contextState:report["contextState"].string,report:report,context:wire["context"])
 }
}
