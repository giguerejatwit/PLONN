import Foundation
import Combine

/// In-memory, one-account page cache. Reuse never renews evidence or quote times.
@MainActor final class AnalyticsPageCache {
 static let shared=AnalyticsPageCache()
 struct Snapshot {
  let candidates:[AnalyticsFeedCandidate]
  let results:[String:AnalyticsHistoryResult]
  let historyDates:[String:Date]
  let warnings:[String]
  let discoveredAt:Date
 }
 private var scopeID=""
 private var lease=UUID()
 private var snapshots:[String:Snapshot]=[:]
 func activate(scopeID:String)->UUID {
  if self.scopeID != scopeID {self.scopeID=scopeID;lease=UUID();snapshots=[:]}
  return lease
 }
 func isCurrent(scopeID:String,lease:UUID)->Bool {self.scopeID == scopeID && self.lease == lease}
 func load(scopeID:String,lease:UUID,now:Date,sport:String?=nil)->Snapshot? {
  guard isCurrent(scopeID:scopeID,lease:lease),let snapshot=snapshots[sport ?? "all"],
        (0..<300).contains(now.timeIntervalSince(snapshot.discoveredAt)) else{return nil}
  let results=Self.retained(from:snapshot,candidates:snapshot.candidates,now:now)
  return Snapshot(candidates:snapshot.candidates,results:results,
   historyDates:snapshot.historyDates.filter{results[$0.key] != nil},warnings:snapshot.warnings,discoveredAt:snapshot.discoveredAt)
 }
 func save(_ value:Snapshot,scopeID:String,lease:UUID,sport:String?=nil) {
  guard isCurrent(scopeID:scopeID,lease:lease),value.candidates.count<=AnalyticsFeedCore.browsingCapacity,
        sport == nil || ["NFL","NBA","MLB","NHL"].contains(sport!),
        sport == nil || value.candidates.allSatisfy({$0.market.sport == sport}),
        value.results.values.allSatisfy({$0.observations.count<=2000}),
        snapshots[sport ?? "all"].map({$0.discoveredAt<=value.discoveredAt}) ?? true else{return}
  snapshots[sport ?? "all"]=value
 }
 /// Retain historical calculations only for the same verified subject/stat/event.
 /// New quotes and new status always come from the new discovery response.
 static func retained(from old:Snapshot,candidates:[AnalyticsFeedCandidate],now:Date)->[String:AnalyticsHistoryResult] {
  var results:[String:AnalyticsHistoryResult]=[:]
  let previousByID=Dictionary(old.candidates.map{($0.id,$0)},uniquingKeysWith:{first,_ in first})
  for candidate in candidates {
   guard let previous=previousByID[candidate.id],
         previous.subject == candidate.subject,previous.market == candidate.market,
         let before=previous.verifiedSubject,let after=candidate.verifiedSubject,
         before.providerID == after.providerID,before.sport == after.sport,before.subject == after.subject,
         before.name == after.name,before.teamID == after.teamID,before.teamName == after.teamName,
         before.sourceURL == after.sourceURL,
         sameEvent(before.event,after.event),
         let date=old.historyDates[candidate.id],(0..<900).contains(now.timeIntervalSince(date)),
         let history=old.results[candidate.id],(history.statistics?.valid ?? 0)>0 else{continue}
   results[candidate.id]=history
  }
  return results
 }
 private static func sameEvent(_ lhs:AnalyticsEvent?,_ rhs:AnalyticsEvent?)->Bool {
  guard let lhs,let rhs else{return lhs == nil && rhs == nil}
  return lhs.sport == rhs.sport && lhs.providerID == rhs.providerID && lhs.start == rhs.start &&
   lhs.home == rhs.home && lhs.away == rhs.away && lhs.homeID == rhs.homeID && lhs.awayID == rhs.awayID && lhs.seasonType == rhs.seasonType
 }
}

/// Tasks retain their slots through completion, including cancelled shared reads.
@MainActor final class AnalyticsFeedController:ObservableObject {
 @Published private(set) var scopeID=""
 @Published private(set) var sport:String?
 @Published private(set) var candidates:[AnalyticsFeedCandidate]=[]
 @Published private(set) var results:[String:AnalyticsHistoryResult]=[:]
 @Published private(set) var loadingIDs=Set<String>()
 @Published private(set) var warnings:[String]=[]
 @Published private(set) var discovering=false
 typealias Loader=([Market],AnalyticsFeedService.RecordedReader?) async -> AnalyticsFeedBuildResult
 typealias HistoryLoader=(AnalyticsFeedCandidate) async -> AnalyticsHistoryResult
 typealias StatusLoader=([AnalyticsFeedCandidate]) async -> AnalyticsFeedBuildResult
 private let cache:AnalyticsPageCache
 private let now:()->Date
 private let load:Loader?
 private let loadHistory:HistoryLoader
 private let loadStatus:StatusLoader
 private var cacheLease=UUID()
 private var historyDates:[String:Date]=[:]
 private var snapshots:[Market]=[]
 private var recordedRead:AnalyticsFeedService.RecordedReader?
 private var discoveredAt:Date?
 private var statusCheckedAt:Date?
 private var statusRefresh:Task<Void,Never>?
 @Published private(set) var statusWarnings:[String]=[]
 private var generation=UUID()
 private var active=false
 private var discovered=false
 private var discovery:Task<Void,Never>?
 private var discoveryTicket=UUID()
 private var tasks:[UUID:Task<Void,Never>]=[:]
 private var runningCandidates:[UUID:String]=[:]
 private var preferredID:String?
 init(cache:AnalyticsPageCache?=nil,now:@escaping ()->Date={Date()},
      load:Loader?=nil,
      history:@escaping HistoryLoader={await AnalyticsFeedService.shared.history(candidate:$0,limit:10)},
      status:@escaping StatusLoader={await AnalyticsFeedService.shared.refreshStatuses(candidates:$0)}) {
  self.cache=cache ?? .shared;self.now=now;self.load=load;self.loadHistory=history;self.loadStatus=status
 }
 private var page:AnalyticsPageCache.Snapshot? {
  guard let discoveredAt else{return nil}
  return .init(candidates:candidates,results:results,historyDates:historyDates,warnings:warnings,discoveredAt:discoveredAt)
 }
 private func savePage() {
  guard let page else{return}
  cache.save(page,scopeID:scopeID,lease:cacheLease,sport:sport)
 }
 func configure(scopeID:String,snapshots:[Market],read:AnalyticsFeedService.RecordedReader? = nil,sport:String?=nil) {
  self.recordedRead=read
  guard self.scopeID != scopeID || self.sport != sport || !cache.isCurrent(scopeID:scopeID,lease:cacheLease) else{return}
  pause();generation=UUID();self.scopeID=scopeID;self.snapshots=snapshots;self.sport=sport
  statusCheckedAt=nil;statusWarnings=[]
  candidates=[];results=[:];historyDates=[:];warnings=[];loadingIDs=[];discovered=false;discovering=false;discoveredAt=nil
  cacheLease=cache.activate(scopeID:scopeID)
  if let cached=cache.load(scopeID:scopeID,lease:cacheLease,now:now(),sport:sport) {
   candidates=cached.candidates;results=cached.results;historyDates=cached.historyDates
   warnings=cached.warnings;discoveredAt=cached.discoveredAt;discovered=true
  }
 }
 func resume() {
  guard !scopeID.isEmpty else{return}
  active=true
  if !discovered {discover()} else if discoveredAt.map({now().timeIntervalSince($0)>=300}) == true {retryDiscovery()} else {schedule()}
 }
 func pause() {
  active=false;generation=UUID();discoveryTicket=UUID();discovery?.cancel();discovery=nil;discovering=false
  for task in tasks.values {task.cancel()}
  loadingIDs=[];statusRefresh?.cancel();statusRefresh=nil;statusCheckedAt=nil
 }
 func retryDiscovery() {
  guard active,!discovering else{return}
  pause();active=true;discovered=false;discover()
 }
 func refreshIfNeeded() {
  guard active,discovered,!discovering,discoveredAt.map({now().timeIntervalSince($0)>=300}) == true else{return}
  retryDiscovery()
 }
 func refreshStatusIfNeeded() {
  guard active,discovered,!discovering,statusRefresh == nil,
        candidates.contains(where:{$0.recordedSnapshot}),
        statusCheckedAt.map({now().timeIntervalSince($0)>=30}) ?? true else{return}
  let token=generation,scope=scopeID,original=candidates
  statusCheckedAt=now()
  statusRefresh=Task {
   let refreshed=await self.loadStatus(original)
   guard !Task.isCancelled,self.active,self.generation == token,self.scopeID == scope else{return}
   self.candidates=refreshed.candidates;self.statusWarnings=refreshed.warnings;self.statusRefresh=nil;self.savePage()
  }
 }
 func prioritize(_ id:String?) {
  preferredID=id; schedule()
 }
 func retry(_ id:String) {
  guard active,!loadingIDs.contains(id) else{return}
  results[id]=nil;historyDates[id]=nil;preferredID=id;savePage();schedule()
 }
 private func discover() {
  guard active,discovery == nil else{return}
  let token=generation,ticket=UUID(),scope=scopeID,seeds=snapshots,read=recordedRead,previous=page,selectedSport=sport
  discoveryTicket=ticket;discovering=true
  discovery=Task {
   let result:AnalyticsFeedBuildResult
   if let load=self.load {result=await load(seeds,read)}
   else {result=await AnalyticsFeedService.shared.load(snapshots:seeds,recordedRead:read,sport:selectedSport,publicStatisticsFallback:true)}
   guard !Task.isCancelled,self.active,self.generation == token,self.scopeID == scope,self.discoveryTicket == ticket else{return}
   self.candidates=Array(result.candidates.prefix(AnalyticsFeedCore.browsingCapacity));self.warnings=result.warnings
   let loadedAt=self.now()
   self.results=previous.map{AnalyticsPageCache.retained(from:$0,candidates:self.candidates,now:loadedAt)} ?? [:]
   self.historyDates=previous?.historyDates.filter{self.results[$0.key] != nil} ?? [:]
   self.discoveredAt=loadedAt;self.statusCheckedAt=nil;self.statusWarnings=[]
   self.discovered=true;self.discovering=false;self.discovery=nil;self.savePage();self.schedule()
  }
 }
 private func schedule() {
  guard active,discovered else{return}
  // Warm only the opening cards. All other histories load when selected,
  // so expanding the roster does not fan out into hundreds of requests.
  let opening=Array(candidates.prefix(2))
  let ordered=candidates.filter{$0.id == preferredID}+opening.filter{$0.id != preferredID}
  for candidate in ordered {
   guard tasks.count<2 else{break}
   guard results[candidate.id] == nil,!runningCandidates.values.contains(candidate.id) else{continue}
   let ticket=UUID(),token=generation,scope=scopeID,id=candidate.id
   loadingIDs.insert(id);runningCandidates[ticket]=id
   tasks[ticket]=Task {
    let result=await self.loadHistory(candidate)
    if !Task.isCancelled,self.active,self.generation == token,self.scopeID == scope {
     self.results[id]=result;self.historyDates[id]=self.now();self.savePage()
    }
    self.tasks[ticket]=nil;self.runningCandidates[ticket]=nil
    if self.generation == token {self.loadingIDs.remove(id)}
    self.schedule()
   }
  }
 }
}
