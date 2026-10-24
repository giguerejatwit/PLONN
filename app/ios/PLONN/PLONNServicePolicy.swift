import Foundation

/// Text identity stays available while artwork loads or when an image is missing.
enum SubjectBadgeInitials {
    static func make(_ name: String) -> String {
        name.split(whereSeparator: { $0.isWhitespace })
            .compactMap { $0.first(where: { $0.isLetter || $0.isNumber }) }
            .prefix(3).map { String($0).uppercased() }.joined()
    }
}

/// Routine recorded-price provenance is a timestamp, not a feed failure.
/// Only exact known informational messages are suppressed; errors remain visible.
struct OddsIssuePresentation: Equatable {
 let asOf: String?
 let status: String?
 let warning: String?
 init(_ message: String) {
  let parts=message.components(separatedBy:" · ")
  let wrapped=message.hasPrefix("Cached odds · ") && parts.count >= 3
  asOf=wrapped ? parts[1]:nil
  let detail=wrapped ? parts.dropFirst(2).joined(separator:" · "):message
  switch detail {
  case "Recorded sportsbook odds · Refresh unavailable · Prices may have changed. See each quote's as-of time.",
       "Daily recorded sportsbook odds · Prices may have changed. See each quote's as-of time.",
       "Cached odds. Provider refreshes paused until midnight Eastern.",
       "Provider refreshes paused until midnight Eastern.":
   status=wrapped ? nil:"Recorded prices";warning=nil
  case "No cached markets for this matchup.":
   status="Prices unavailable for this matchup.";warning=nil
  case "No recorded odds available for this selection.",
       "No cached odds available for this selection. Provider refreshes paused until midnight Eastern.":
   status="Prices unavailable for this selection.";warning=nil
  default:
   status=nil;warning=detail
  }
 }
}
import CoreFoundation

/// Publication and service activation are separate operator gates. Both default off.
struct PLONNServicePolicy: Sendable {
    let chatEnabled: Bool
    let publicPagesApproved: Bool
    static let privacyURL = URL(string: "https://oakas.ai/plonn/privacy")!
    static let supportURL = URL(string: "https://oakas.ai/plonn/support")!

    init(chatEnabled: Bool = false, publicPagesApproved: Bool = false) {
        self.chatEnabled = chatEnabled; self.publicPagesApproved = publicPagesApproved
    }
    init(info: [String: Any]) {
        func flag(_ key: String) -> Bool {
            guard let value = info[key] as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID() else { return false }
            return value.boolValue
        }
        self.init(chatEnabled: flag("PLONNChatEnabled"), publicPagesApproved: flag("PLONNPublicPagesApproved"))
    }
    var chatAvailable: Bool { chatEnabled && publicPagesApproved }
    func mayUseChat(ownerID: UUID?, consentOwnerID: UUID?) -> Bool {
        chatAvailable && ownerID != nil && ownerID == consentOwnerID
    }
}


/// NBA season metadata is evidence, not a date or pregame-status heuristic.
enum NBARecordedOddsPhase: String, Hashable, Sendable { case preseason, regular, unknown }
enum NBARecordedOddsRouting {
 static func seasonType(event: [String:Any]) -> Int? {
  guard let season=event["season"] as? [String:Any],let value=season["type"] as? NSNumber,
        String(cString:value.objCType) != "c", value.doubleValue.rounded() == value.doubleValue,
        [1,2,3].contains(value.intValue) else{return nil}
  return value.intValue
 }
 static func phase(sport:String, seasonTypes:[Int?]) -> NBARecordedOddsPhase? {
  guard sport == "NBA" else{return nil}
  guard !seasonTypes.isEmpty,seasonTypes.allSatisfy({$0 != nil}) else{return .unknown}
  let types=Set(seasonTypes.compactMap {$0})
  if types == [1] {return .preseason}
  if types.isSubset(of:[2,3]) {return .regular}
  return .unknown
 }
 static func canRead(_ phase:NBARecordedOddsPhase?, cacheOnly:Bool, preseasonHandlerEnabled:Bool)->Bool {
  guard let phase else{return true}
  guard cacheOnly else{return false}
  return phase == .regular || (phase == .preseason && preseasonHandlerEnabled)
 }
}

struct OddsRequestBody: Encodable { let sport:String; let event:String?; let cacheOnly:Bool; var phase:String? = nil; var recordedArchive:Bool? = nil }

/// ESPN uses Montreal where the recorded provider uses Montréal. Preserve the
/// entire team name and word order; normalize only accents, case and whitespace.
enum RecordedTeamIdentity {
 static func key(_ value:String)->String {
  value.folding(options:[.diacriticInsensitive,.caseInsensitive],locale:Locale(identifier:"en_US_POSIX"))
   .split(whereSeparator:{$0.isWhitespace}).joined(separator:" ")
 }
 static func matches(_ lhs:String,_ rhs:String)->Bool {
  let left=key(lhs),right=key(rhs)
  return !left.isEmpty && left == right
 }
}

/// Provider tip-off / puck-drop estimates may differ from the schedule by ten minutes.
/// Resolve only one exact-pair event; never select between ambiguous provider events.
enum RecordedStartPolicy {
 static func accepts(eventID:String,quoteStart:Date,start:Date,sport:String,candidates:[(String,Date)],now:Date,includeStarted:Bool=false)->Bool {
  let exact = abs(start.timeIntervalSince(quoteStart)) <= 1
  guard sport == "NHL" || (sport == "NBA" && !exact) else{return exact}
  guard includeStarted || start > now else{return false}
  var utc=Calendar(identifier:.gregorian);utc.timeZone=TimeZone(secondsFromGMT:0)!
  func eligible(_ date:Date)->Bool {(includeStarted || date > now) && utc.isDate(start,inSameDayAs:date) && abs(start.timeIntervalSince(date)) <= 600}
  let ids=Set(candidates.filter{eligible($0.1)}.map{$0.0})
  return eligible(quoteStart) && ids.count == 1 && ids.contains(eventID)
 }
}


struct RecordedEventMetadata: Codable, Sendable {
 let id:String
 let code:String
 let homeTeam:String
 let awayTeam:String
 let commence:String
}
enum RecordedEventSelection {
 static func providerCode(sport:String,phase:NBARecordedOddsPhase?)->String? {
  switch sport {
  case "NFL": return phase == nil ? "americanfootball_nfl":nil
  case "NHL": return phase == nil ? "icehockey_nhl":nil
  case "MLB": return phase == nil ? "baseball_mlb":nil
  case "NBA":
   if phase == .preseason{return "basketball_nba_preseason"}
   if phase == .regular{return "basketball_nba"}
   return nil
  default:return nil
  }
 }
 static func select(_ events:[RecordedEventMetadata],sport:String,phase:NBARecordedOddsPhase?,home:String,away:String,start:Date,now:Date,includeStarted:Bool=false)->RecordedEventMetadata? {
  guard (includeStarted || start > now),let code=providerCode(sport:sport,phase:phase),events.count<=512 else{return nil}
  let formatter=ISO8601DateFormatter()
  func date(_ text:String)->Date? {
   formatter.formatOptions=[.withInternetDateTime,.withFractionalSeconds]
   if let d=formatter.date(from:text){return d}
   formatter.formatOptions=[.withInternetDateTime];return formatter.date(from:text)
  }
  let pair=events.filter{$0.code == code && RecordedTeamIdentity.matches($0.homeTeam,home) && RecordedTeamIdentity.matches($0.awayTeam,away)}
  var unique:[String:RecordedEventMetadata]=[:]
  for row in pair {
   guard !row.id.isEmpty,row.id.utf8.count<=128,row.id.utf8.allSatisfy({(33...126).contains($0)}),let d=date(row.commence),(includeStarted || d>now) else{return nil}
   if let previous=unique[row.id],previous.commence != row.commence || previous.homeTeam != row.homeTeam || previous.awayTeam != row.awayTeam{return nil}
   unique[row.id]=row
  }
  let candidates=unique.values.compactMap{row -> (String,Date)? in guard let d=date(row.commence) else{return nil};return(row.id,d)}
  let matches=unique.values.filter{row in guard let d=date(row.commence) else{return false};return RecordedStartPolicy.accepts(eventID:row.id,quoteStart:d,start:start,sport:sport,candidates:candidates,now:now,includeStarted:includeStarted)}
  guard matches.count == 1,let chosen=matches.first else{return nil}
  // A reused provider ID with a different identity is not safe to request.
  guard events.filter({$0.id == chosen.id}).allSatisfy({$0.code == chosen.code && $0.homeTeam == chosen.homeTeam && $0.awayTeam == chosen.awayTeam && $0.commence == chosen.commence}) else{return nil}
  return chosen
 }
}
