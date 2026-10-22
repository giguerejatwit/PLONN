import SwiftUI
import Combine
import Supabase

// Reversible launch scope. Archived implementations and persisted records remain intact.
// Add a league here to restore its Games, Analytics, and Models availability.
enum LaunchSports {
 static let supported = ["NBA", "MLB", "NFL", "NHL"]
 static func includes(_ sport: String) -> Bool { supported.contains(sport) }
}

struct OddsResponse: Codable { var markets: [Market]; var events:[RecordedEventMetadata]? = nil; var message: String; var cacheOnly:Bool? = nil; var stale:Bool? = nil; var fetchedAt:String? = nil; var cacheOnlyUntil:String? = nil; var quoteUpdatedAt:String? = nil; var servedAt:String? = nil; var recordedArchive:Bool? = nil; var pricesAreLive:Bool? = nil; var eventStatusVerified:Bool? = nil }
struct HistoryRow: Codable { var date: String; var pick_team: String; var opponent: String; var result: String; var profit: String }
let charcoalAccent = Color(uiColor: .label)
let performanceGreen = Color(uiColor: UIColor { traits in traits.userInterfaceStyle == .dark ? UIColor(red:0.30,green:0.88,blue:0.53,alpha:1):UIColor(red:0.05,green:0.43,blue:0.23,alpha:1) })
// Violet → indigo → fuchsia. Lighter ink stays readable on the dark canvas;
// color stays in quiet edge reflections instead of solid button fills.
enum PLONNPalette {
 static let violet = Color(red:124/255,green:58/255,blue:237/255)
 static let indigo = Color(red:79/255,green:70/255,blue:229/255)
 static let fuchsia = Color(red:162/255,green:28/255,blue:175/255)
 static let ink = Color(red:182/255,green:162/255,blue:255/255)
 static let gradient = LinearGradient(colors:[violet,indigo,fuchsia],startPoint:.topLeading,endPoint:.bottomTrailing)
 static let highlight = LinearGradient(colors:[ink,Color(red:165/255,green:180/255,blue:252/255),Color(red:232/255,green:121/255,blue:249/255)],startPoint:.leading,endPoint:.trailing)
}
let brandAccent = PLONNPalette.ink
let appBackground = UIColor { traits in traits.userInterfaceStyle == .dark ? UIColor(white:0.075,alpha:1) : .systemBackground }
let navy = Color(uiColor: appBackground)
let navySurface = Color(uiColor: UIColor { traits in traits.userInterfaceStyle == .dark ? UIColor(white:0.13,alpha:1) : .secondarySystemBackground })
struct PLONNBrandBackdrop: View {
 @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
 var body:some View {
  ZStack {
   navy
   if !reduceTransparency {
    LinearGradient(colors:[PLONNPalette.violet.opacity(0.10),PLONNPalette.indigo.opacity(0.05),.clear],startPoint:.topLeading,endPoint:.bottomTrailing)
    RadialGradient(colors:[PLONNPalette.fuchsia.opacity(0.06),.clear],center:.bottomTrailing,startRadius:0,endRadius:420)
   }
  }.ignoresSafeArea().allowsHitTesting(false).accessibilityHidden(true)
 }
}
struct LeagueBackdrop: View {
 let league:String
 var body:some View {
  PLONNBrandBackdrop().overlay(ModelLandscape(accent:brandAccent,showsBase:false).opacity(0.45)).ignoresSafeArea()
 }
}
func odds(_ n: Double?) -> String { guard let n, n.isFinite else { return "Unavailable" }; if UserDefaults.standard.string(forKey:"oddsFormat") == "decimal" { guard abs(n)>=100 else{return "Unavailable"};return String(format:"%.2f",n>0 ? 1+n/100:1+100/abs(n)) }; return String(format: "%+.0f", n) }
func probability(_ p: Double?) -> String { guard let p else { return "Unavailable" }; return String(format: "%.1f%%", p * 100) }
func sample(_ id: String, _ event: String, _ sport: String, _ name: String, _ opponent: String, _ label: String, _ p: Double, _ price: Double, _ kind: String, _ aliases: String, _ logs: [Double]? = nil) -> Market {
 Market(id:id,event:event,sport:sport,kind:kind,name:name,opponent:opponent,label:label,aliases:aliases,p:p,price:price,market:label,book:"Sample sportsbook",quotes:[Quote(book:"Sample sportsbook",price:price)],mode:"demo",logs:logs)
}
let samples = [sample("d1","nba-bos-nyk","NBA","Boston Celtics","New York Knicks","Boston moneyline",0.60,-125,"team","BOS NYK Boston Celtics"),sample("d2","nba-bos-nyk","NBA","Jayson Tatum","New York Knicks","Over 27.5 points",0.56,-110,"prop","BOS Boston Celtics points",[24,31,28,20,35,26,33,22,30,29]),sample("d3","mlb-nyy-bos","MLB","Aaron Judge","Boston Red Sox","Over 1.5 total bases",0.55,105,"prop","NYY New York Yankees total bases",[2,0,3,1,4,0,2,1,3,0]),sample("d4","soc-fra-por","Soccer","France","Portugal","France to win",0.44,145,"team","FRA national team")]

// Shared odds request state is independent of view lifetime and keyed by identity.
struct OddsScope: Hashable {
 let backend: String
 let owner: UUID?
}
struct OddsRequestKey: Hashable {
 let scope: OddsScope
 let sport: String
 let event: String?
 var cacheOnly: Bool = false
 var phase: NBARecordedOddsPhase? = nil
 var recordedArchive: Bool = false
}

struct OddsServiceError: Error {
 let status: Int
 let body: Data
}
enum AnalyticsScopedOddsError:Error {
 case notConfigured,signInRequired,invalidResponse
}
@MainActor final class SharedOddsCache<Value> {
 struct Snapshot {
  let value: Value
  let receivedAt: Date
 }
 private var snapshots: [OddsRequestKey: Snapshot] = [:]
 private var flights: [OddsRequestKey: (id: UUID, task: Task<Snapshot, Error>)] = [:]
 func cached(_ key: OddsRequestKey) -> Snapshot? { snapshots[key] }
 func cachedValues(in scope: OddsScope) -> [Snapshot] {
  snapshots.filter { $0.key.scope == scope }.values.sorted { $0.receivedAt > $1.receivedAt }
 }
 func invalidate(_ scope: OddsScope) {
  snapshots = snapshots.filter { $0.key.scope != scope }
  let keys = flights.keys.filter { $0.scope == scope }
  for key in keys { flights.removeValue(forKey:key)?.task.cancel() }
 }
 func fetch(_ key: OddsRequestKey, force: Bool, now: Date = Date(),
            operation: @escaping @MainActor () async throws -> Value) async throws -> Snapshot {
  // Even explicit refreshes share an already-running request for exactly this scope.
  if let flight = flights[key] { return try await flight.task.value }
  if !force, let cached = snapshots[key], now.timeIntervalSince(cached.receivedAt) < 30 { return cached }
  let id = UUID()
  let task = Task { @MainActor in
   defer { if self.flights[key]?.id == id { self.flights[key] = nil } }
   let value = try await operation()
   try Task.checkCancellation()
   guard self.flights[key]?.id == id else { throw CancellationError() }
   let snapshot = Snapshot(value:value, receivedAt:Date())
   self.snapshots[key] = snapshot
   if self.snapshots.count > 128,
      let oldest = self.snapshots.min(by: { $0.value.receivedAt < $1.value.receivedAt })?.key {
    self.snapshots[oldest] = nil
   }
   return snapshot
  }
  flights[key] = (id, task)
  return try await task.value
 }
}
// End shared odds request state.

@MainActor final class Store: ObservableObject {
 @Published var saved: [Market] = []
 @Published var legs: [Market] = [] {didSet {persistBuilderDraft()}}
 @Published private(set) var builderDraftIssue=""
 @Published private(set) var builderDraftBlocked=false
 @Published var live: [Market] = []
 @Published var status = "Sample data. Prices and model estimates are illustrative."
 @Published var loading = false
 @Published var history: [HistoryRow] = []
 @AppStorage("backendURL") var backendURL = ""
 private let watchlist = WatchlistStorage<Market>(defaults: .standard)
 private let builderDraft = BuilderDraftStorage(defaults:.standard)
 private var builderOwner:UUID?
 private var requestVersion = 0
 private static let oddsCache = SharedOddsCache<OddsResponse>()
 private static var scopedSnapshots:[String:AnalyticsScopedOddsResponse]=[:]
 private static var scopedFlights:[String:Task<AnalyticsScopedOddsResponse,Error>]=[:]
 private var accountObservation: AnyCancellable?
 private weak var observedAccount: AccountStore?
 private var liveContext: [String?] = []
 private(set) var oddsScope: OddsScope?
 @Published private(set) var oddsStale = false
 @Published private(set) var oddsCachedOnly = false
 @Published private(set) var oddsUnauthorized = false
 @Published private(set) var oddsLoadSucceeded = false
 @Published private(set) var oddsIssue: String?
 @Published private(set) var lastOddsRefresh: Date?
 func scope(for account: AccountStore) -> OddsScope {
  var backend = backendURL.trimmingCharacters(in:.whitespacesAndNewlines)
  #if DEBUG && targetEnvironment(simulator)
  if backend.isEmpty { backend = "http://127.0.0.1:8766" }
  #endif
  return OddsScope(backend:backend, owner:account.user?.id)
 }
 /// Analytics reads retained selections only; this path never refreshes paid odds.
 func analyticsSnapshots(for account: AccountStore) -> [Market] {
  #if DEBUG && targetEnvironment(simulator)
  if let fixture=ProcessInfo.processInfo.environment["PLONN_ANALYTICS_MARKETS_FIXTURE"],
     let data=fixture.data(using:.utf8),data.count<=512*1024,
     let markets=try? JSONDecoder().decode([Market].self,from:data),markets.count<=512 { return markets }
  #endif
  let current=scope(for:account)
  return (oddsScope == current ? live:[]) + Self.oddsCache.cachedValues(in:current).flatMap { $0.value.markets }
 }
 /// Analytics independently reads the authenticated shared snapshot. No live view state,
 /// provider refresh, paid fallback, or inferred NBA phase is used by this adapter.
 func recordedAnalyticsSnapshot(sport:String,phase:NBARecordedOddsPhase?,event:String?,account:AccountStore) async throws -> AnalyticsFeedService.RecordedSnapshot {
  guard account.user?.id != nil,account.user?.isAnonymous == false else {throw URLError(.userAuthenticationRequired)}
  let scope=scope(for:account)
  let key=OddsRequestKey(scope:scope,sport:sport,event:event,cacheOnly:true,phase:phase,recordedArchive:true)
  let snapshot=try await Self.oddsCache.fetch(key,force:false) {try await Self.fetchOdds(key,account:account)}
  try Task.checkCancellation()
  guard self.scope(for:account) == scope,account.user?.isAnonymous == false else {throw CancellationError()}
  return AnalyticsFeedService.RecordedSnapshot(markets:snapshot.value.markets,events:snapshot.value.events ?? [],cacheOnly:snapshot.value.cacheOnly == true,fetchedAt:snapshot.value.fetchedAt,servedAt:snapshot.value.servedAt,recordedArchive:snapshot.value.recordedArchive == true && snapshot.value.stale == true && snapshot.value.pricesAreLive == false && snapshot.value.eventStatusVerified == false)
 }
 /// Separate, versioned endpoint. The old broad feed is never a fallback.
 func scopedAnalyticsOdds(_ request:AnalyticsScopedOddsRequest,account:AccountStore) async throws -> AnalyticsScopedOddsResponse {
  guard Bundle.main.object(forInfoDictionaryKey:"PLONNScopedOddsEnabled") as? Bool == true else {throw AnalyticsScopedOddsError.notConfigured}
  guard let owner=account.user?.id,account.user?.isAnonymous == false else {throw AnalyticsScopedOddsError.signInRequired}
  let scope=scope(for:account)
  let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
  let data=try encoder.encode(request)
  guard data.count<=16384 else{throw AnalyticsScopedOddsError.invalidResponse}
  let key=owner.uuidString+"|"+scope.backend+"|"+String(decoding:data,as:UTF8.self)
  if let cached=Self.scopedSnapshots[key],let expires=sportsDate(cached.expiresAt),expires>Date(),!cached.stale {return cached}
  if let flight=Self.scopedFlights[key] {
   let result=try await flight.value
   guard account.user?.id==owner,self.scope(for:account)==scope,!Task.isCancelled else{throw CancellationError()}
   return result
  }
  // Capture a SDK-owned session for this exact account; never read auth storage manually.
  let flight=Task { @MainActor () throws -> AnalyticsScopedOddsResponse in
   defer{Self.scopedFlights.removeValue(forKey:key)}
   let session=try await account.client.auth.session
   guard account.user?.id==owner,self.scope(for:account)==scope,session.user.id==owner,!session.user.isAnonymous,!session.isExpired,!Task.isCancelled else{throw CancellationError()}
   let result:AnalyticsScopedOddsResponse=try await account.client.functions.invoke("odds-scoped",options:.init(headers:["Authorization":"Bearer "+session.accessToken],body:request,timeoutInterval:20)){data,_ in
    guard data.count<=1024*1024 else{throw AnalyticsScopedOddsError.invalidResponse}
    let parsed=try JSONDecoder().decode(AnalyticsScopedOddsResponse.self,from:data)
    guard parsed.schemaVersion==1,parsed.markets.count<=1000 else{throw AnalyticsScopedOddsError.invalidResponse}
    return parsed
   }
   guard account.user?.id==owner,self.scope(for:account)==scope,!Task.isCancelled else{throw CancellationError()}
   Self.scopedSnapshots[key]=result
   if Self.scopedSnapshots.count>32 {Self.scopedSnapshots=Self.scopedSnapshots.filter{sportsDate($0.value.expiresAt).map{$0>Date()} ?? false}}
   if Self.scopedSnapshots.count>32 {Self.scopedSnapshots.removeValue(forKey:Self.scopedSnapshots.keys.sorted().first!)}
   return result
  }
  Self.scopedFlights[key]=flight
  let result=try await flight.value
  guard account.user?.id==owner,self.scope(for:account)==scope,!Task.isCancelled else{throw CancellationError()}
  return result
 }
 private func resetOdds() {
  if let oddsScope { Self.oddsCache.invalidate(oddsScope) }
  requestVersion += 1; live = []; liveContext = []; oddsScope = nil
  loading = false; oddsStale = false; oddsCachedOnly = false; oddsUnauthorized = false; oddsLoadSucceeded = false; oddsIssue = nil; lastOddsRefresh = nil
  status = "Account or odds source changed. Refresh sportsbook prices."
 }
 init() {
  if let url=Bundle.main.url(forResource:"history",withExtension:"json"),let data=try? Data(contentsOf:url),let rows=try? JSONDecoder().decode([HistoryRow].self,from:data){history=rows}
 }
 func bindAccount(_ account:AccountStore) {
  guard observedAccount !== account else {return}
  observedAccount=account
  accountObservation=account.$user.map {user in
   AccountAccessPolicy.permits(ownerID:user?.id,isAnonymous:user?.isAnonymous) ? user?.id:nil
  }.removeDuplicates().sink { [weak self] owner in
   guard let self else {return}
   self.builderOwner=nil;self.builderDraftBlocked=false;self.builderDraftIssue=""
   self.resetOdds();self.saved=[];self.legs=[]
   if owner != nil {
    do {self.saved=try self.watchlist.load()}
    catch {self.status="Saved research could not be read. Try again in Settings."}
    do {
     let restored=BuilderLogic.sanitized(try self.builderDraft.load())
     self.legs=restored.legs;self.builderDraftIssue=restored.notices.joined(separator:"\n")
    } catch {self.builderDraftBlocked=true;self.builderDraftIssue="The saved draft could not be read. Reset it to resume saving."}
   }
   self.builderOwner=owner
  }
 }
 private func persistBuilderDraft() {
  guard builderOwner != nil,!builderDraftBlocked else{return}
  do {try builderDraft.save(legs)} catch {builderDraftIssue="Draft could not be saved on this device. Your current picks remain available."}
 }
 func resetBuilderDraft() {
  guard observedAccount?.hasSignedInAccount == true else{return}
  builderDraft.clear();builderDraftBlocked=false;builderDraftIssue="";persistBuilderDraft()
 }
 func clearBuilderDraft() {
  guard observedAccount?.hasSignedInAccount == true else{return}
  builderDraftBlocked=false;builderDraftIssue="";legs=[]
 }
 func toggleSave(_ m:Market){
  guard let account=observedAccount else{return}
  account.access.require(.savedResearch,ownerID:account.user?.id,isAnonymous:account.user?.isAnonymous) { [weak self,weak account] in
   guard let self,let account,account.hasSignedInAccount,self.observedAccount === account else{return}
   var next=self.saved
   if next.contains(where:{$0.id==m.id}) {next.removeAll{$0.id==m.id}} else {next.append(m)}
   do {try self.watchlist.save(next);self.saved=next} catch {self.status="Unable to save this research on your device."}
  }
 }
 func clearWatchlist(){
  guard let account=observedAccount else{return}
  account.access.require(.savedResearch,ownerID:account.user?.id,isAnonymous:account.user?.isAnonymous) { [weak self,weak account] in
   guard let self,account?.hasSignedInAccount == true else{return}
   self.watchlist.clear();self.saved=[]
  }
 }
 func add(_ m:Market){
  guard let account=observedAccount else{return}
  account.access.require(.savedResearch,ownerID:account.user?.id,isAnonymous:account.user?.isAnonymous) { [weak self,weak account] in
   guard let self,account?.hasSignedInAccount == true,!self.legs.contains(where:{$0.id==m.id}) else{return}
   guard self.legs.count<8 else{self.status="Maximum of eight selections.";return}
   guard self.legs.first?.mode==nil || self.legs.first?.mode==m.mode else{self.status="Sample and live selections cannot be combined.";return}
   self.legs.append(m)
  }
 }
 func setSample(){requestVersion += 1;live=[];liveContext=[];loading=false;oddsStale=false;oddsCachedOnly=false;oddsLoadSucceeded=false;oddsIssue=nil;status="Sample data. Prices and model estimates are illustrative."}
 /// Existing callers refresh by default; background reuse can opt into the 30-second cache.
 func load(_ sport:String, account:AccountStore, home:String? = nil, away:String? = nil, start:String? = nil, forceRefresh:Bool = true, cacheOnly:Bool = false, nbaSeasonTypes:[Int?] = []) async {
  bindAccount(account)
  guard account.hasSignedInAccount else {
   resetOdds();oddsUnauthorized=true;oddsIssue="Sign in to view sportsbook odds.";status=oddsIssue!;return
  }
  let scope = scope(for:account)
  if oddsScope != scope { resetOdds(); oddsScope = scope }
  let phase = NBARecordedOddsRouting.phase(sport:sport,seasonTypes:nbaSeasonTypes)
  let context: [String?] = [sport,home,away,start,String(cacheOnly),phase?.rawValue]
  if liveContext != context { live=[];lastOddsRefresh=nil;oddsStale=false;oddsCachedOnly=false;liveContext=context }
  requestVersion += 1; let version = requestVersion
  oddsLoadSucceeded=false;oddsUnauthorized=false;oddsIssue=nil;loading=true
  status=live.isEmpty ? "Loading sportsbook prices…":"Refreshing sportsbook prices · Showing previous snapshot…"
  defer { if version == requestVersion { loading=false } }
  func current() -> Bool {
   guard self.scope(for:account) == scope else {
    if oddsScope == scope { resetOdds() }
    return false
   }
   return version == requestVersion && !Task.isCancelled
  }
  func staleStatus(_ message:String) -> String {
   guard !live.isEmpty else { return message }
   let time=lastOddsRefresh.map { " · Snapshot timestamp " + $0.formatted(date:.abbreviated,time:.shortened) } ?? ""
   return "Stale sportsbook prices" + time + " · " + message
  }
  guard NBARecordedOddsRouting.canRead(phase,cacheOnly:cacheOnly,preseasonHandlerEnabled:CloudConfiguration.nbaPreseasonCacheEnabled) else {
   live=[];lastOddsRefresh=nil;oddsStale=false;oddsCachedOnly=false
   status=phase == .preseason ? "Recorded preseason odds are not available in this build." : "Verified NBA season information is unavailable for this selection."
   oddsIssue=status;return
  }
  let slateKey=OddsRequestKey(scope:scope,sport:sport,event:nil,cacheOnly:cacheOnly,phase:phase)
  do {
   let slate=try await Self.oddsCache.fetch(slateKey,force:forceRefresh) {
    try await Self.fetchOdds(slateKey,account:account)
   }
   guard current() else { return }
   let markets=slate.value.markets.filter { $0.sport == sport && $0.mode == "live" }
   var next=markets
   var message=slate.value.message
   var stale=slate.value.stale == true || slate.value.cacheOnly == true
   var cachedOnly=slate.value.cacheOnly == true
   var receivedAt=Self.recordedQuoteTime(slate.value) ?? (cachedOnly ? nil:slate.value.fetchedAt.flatMap(sportsDate) ?? slate.receivedAt)
   if let home,let away,let start {
    guard let target=sportsDate(start) else { throw URLError(.badURL) }
    let matching=markets.filter { market in
     Self.matchesRecordedEvent(market,sport:sport,home:home,away:away,start:target,candidates:markets)
    }
    next=matching
    let metadata=RecordedEventSelection.select(slate.value.events ?? [],sport:sport,phase:phase,home:home,away:away,start:target,now:Date())
    let marketEvents=Set(matching.map(\.event))
    let selectedEvent:String? = (slate.value.events?.isEmpty == false) ? metadata?.id : (marketEvents.count == 1 ? marketEvents.first:nil)
    let selectedStarts:[(String,Date)] = metadata.flatMap { row in sportsDate(row.commence).map {[(row.id,$0)]} } ?? []
    if let event=selectedEvent {
     let key=OddsRequestKey(scope:scope,sport:sport,event:event,cacheOnly:cacheOnly,phase:phase)
     do {
      let detail=try await Self.oddsCache.fetch(key,force:forceRefresh) { try await Self.fetchOdds(key,account:account) }
      guard current() else { return }
      next=matching.filter { $0.kind != "prop" } + detail.value.markets.filter { $0.event == event && $0.kind == "prop" && Self.matchesRecordedEvent($0,sport:sport,home:home,away:away,start:target,candidates:markets,metadataCandidates:selectedStarts) }
      message=detail.value.message
      stale = stale || detail.value.stale == true || detail.value.cacheOnly == true
      cachedOnly = cachedOnly || detail.value.cacheOnly == true
      let detailTime=Self.recordedQuoteTime(detail.value) ?? (detail.value.cacheOnly == true ? nil:detail.value.fetchedAt.flatMap(sportsDate) ?? detail.receivedAt)
      if let detailTime {receivedAt=receivedAt.map{min($0,detailTime)} ?? detailTime}
     } catch {
      guard current() else { return }
      if Self.isAuthorizationFailure(error) { throw error }
      let previous=Self.oddsCache.cached(key)
      let props=(previous?.value.markets ?? live).filter { $0.event == event && $0.kind == "prop" && Self.matchesRecordedEvent($0,sport:sport,home:home,away:away,start:target,candidates:markets,metadataCandidates:selectedStarts) }
      next=matching.filter { $0.kind != "prop" } + props
      stale=true
      if !props.isEmpty, let previous, let time=Self.recordedQuoteTime(previous.value) ?? (previous.value.cacheOnly == true ? nil:previous.value.fetchedAt.flatMap(sportsDate) ?? previous.receivedAt) {receivedAt=receivedAt.map{min($0,time)} ?? time}
      let gameStatus = cachedOnly ? "Recorded game odds retained" : "Game odds refreshed"
      message=(props.isEmpty ? gameStatus+" · Player props unavailable. ":"Stale player props retained · "+gameStatus+". ")+Self.failureMessage(error,backend:scope.backend)
      oddsIssue=message
     }
    } else if matching.isEmpty { message=cachedOnly ? "No cached markets for this matchup.":"No sportsbook markets are currently offered for this matchup." }
   }
   live=next;lastOddsRefresh=receivedAt;oddsStale=stale;oddsCachedOnly=cachedOnly;oddsLoadSucceeded = !stale
   if cachedOnly {
    let time=receivedAt.map{Self.oddsAsOf($0)} ?? "As-of time unavailable"
    message="Cached odds · "+time+" · "+message
    oddsIssue=message
   }
   status=message.isEmpty ? "Sportsbook snapshots · Prices may change.":message
  } catch {
   guard current() else { return }
   if Self.isAuthorizationFailure(error) {
    Self.oddsCache.invalidate(scope);live=[];lastOddsRefresh=nil
    oddsStale=false;oddsCachedOnly=false;oddsUnauthorized=true;status="Sign in under Settings > Account to refresh sportsbook odds."
   } else {
    if live.isEmpty, home == nil, away == nil, start == nil, let cached=Self.oddsCache.cached(slateKey) {
     live=cached.value.markets.filter { $0.sport == sport && $0.mode == "live" };lastOddsRefresh=Self.recordedQuoteTime(cached.value) ?? (cached.value.cacheOnly == true ? nil:cached.value.fetchedAt.flatMap(sportsDate) ?? cached.receivedAt);oddsCachedOnly=cached.value.cacheOnly == true
    }
    oddsStale = !live.isEmpty
    status=staleStatus(Self.failureMessage(error,backend:scope.backend))
   }
   oddsIssue=status
  }
 }
 /// Require the complete provider team pair and scheduled instant; never substring-match teams.
 static func matchesRecordedEvent(_ market:Market,sport:String,home:String,away:String,start:Date,candidates:[Market] = [],metadataCandidates:[(String,Date)] = [],now:Date = Date()) -> Bool {
  func key(_ value:String)->String { RecordedTeamIdentity.key(value) }
  guard start > now, market.sport == sport, market.mode == "live", !market.event.isEmpty,
        !key(home).isEmpty, !key(away).isEmpty, key(home) != key(away),
        let label=market.eventLabel, key(label)==key(away+" at "+home),
        let quoteStart=market.commence.flatMap(sportsDate) else {return false}
  // Only NHL needs the full candidate set for its unique-event time tolerance.
  // Other sports use the exact scheduled instant; rescanning every quote here
  // made a matchup filter quadratic on the main thread.
  if sport != "NHL" {
   return RecordedStartPolicy.accepts(eventID:market.event,quoteStart:quoteStart,start:start,sport:sport,candidates:[],now:now)
  }
  let pair=candidates.filter {$0.sport == sport && $0.mode == "live" && !$0.event.isEmpty && key($0.eventLabel ?? "") == key(away+" at "+home)}.compactMap {row -> (String,Date)? in
   guard let date=row.commence.flatMap(sportsDate) else{return nil};return (row.event,date)
  }
  return RecordedStartPolicy.accepts(eventID:market.event,quoteStart:quoteStart,start:start,sport:sport,candidates:pair + metadataCandidates,now:now)
 }
 static func recordedMoneyline(_ team:String,markets:[Market],sport:String,home:String,away:String,start:Date,now:Date = Date())->Market? {
  markets.first { market in
   market.market == "h2h" && RecordedTeamIdentity.matches(market.name,team) &&
    matchesRecordedEvent(market,sport:sport,home:home,away:away,start:start,candidates:markets,now:now)
  }
 }
 /// A recorded quote timestamp is distinct from cache fetch or response delivery time.
 static func recordedQuoteTime(_ response:OddsResponse,now:Date = Date()) -> Date? {
  let times=response.markets.compactMap{$0.updated.flatMap(sportsDate)} + [response.quoteUpdatedAt.flatMap(sportsDate)].compactMap{$0}
  return times.filter{$0 <= now}.min()
 }
 static func oddsAsOf(_ date:Date) -> String {
  let formatter=DateFormatter();formatter.locale=Locale(identifier:"en_US_POSIX");formatter.timeZone=TimeZone(identifier:"America/New_York");formatter.dateFormat="MMM d, h:mm:ss a z"
  return "Quote as of "+formatter.string(from:date)
 }
 /// Render only recognized service failures, never arbitrary server response text.
 static func failureMessage(_ error:Error, backend:String) -> String {
  var code:Int?,body:Data?
  if case FunctionsError.httpError(let status,let data) = error {code=status;body=data}
  if let service=error as? OddsServiceError {code=service.status;body=service.body}
  let message=body.flatMap {try? JSONSerialization.jsonObject(with:$0) as? [String:Any]}?["error"] as? String
  switch message {
  case "Live odds are awaiting provider configuration", "Odds API key is not configured on the server.":
   return "Live odds are temporarily unavailable. Try again later."
  case "Today's beta odds refresh allowance has been reached":
   return "Today's odds refresh allowance has been reached."
  case "Local odds refresh allowance has been reached. Existing snapshots may be stale.":
   return "Odds refresh allowance reached. Existing snapshots may be stale."
  case "Prices are refreshing. Retry shortly":
   return "Prices are refreshing. Pull to retry shortly."
  case "Odds provider returned HTTP 401. Check plan, quota, and key configuration.", "Odds provider returned HTTP 403. Check plan, quota, and key configuration.":
   return "Live odds are temporarily unavailable. Try again later."
  case "Odds provider returned HTTP 402. Check plan, quota, and key configuration.":
   return "The sportsbook provider's usage allowance has been reached."
  case "This competition is not configured for odds yet.", "This competition is not configured for odds":
   return "Sportsbook odds are unavailable for this competition."
  default:break
  }
  if code == 429 {return "Odds refresh limit reached. Wait a minute, then pull to retry."}
  if let network=error as? URLError {
   #if DEBUG
   let host=URLComponents(string:backend)?.host
   if ["127.0.0.1","localhost","::1"].contains(host ?? ""),[.cannotConnectToHost,.cannotFindHost,.timedOut].contains(network.code) {
    return "Local odds service is offline. Start the local odds server, then pull to retry."
   }
   if network.code == .badURL {return "The odds service address is invalid. Check its configuration."}
   #endif
   if network.code == .notConnectedToInternet {return "You are offline. Reconnect, then pull to refresh odds."}
  }
  return "Odds service unavailable. Pull to retry."
 }
 private static func isAuthorizationFailure(_ error:Error) -> Bool {
  if case FunctionsError.httpError(let code, _) = error { return code == 401 || code == 403 }
  return (error as? URLError)?.code == .userAuthenticationRequired
 }
 private static func fetchOdds(_ key:OddsRequestKey, account:AccountStore) async throws -> OddsResponse {
  guard account.hasSignedInAccount else {throw URLError(.userAuthenticationRequired)}
  guard account.user?.id == key.scope.owner else { throw CancellationError() }
  guard !key.recordedArchive || key.cacheOnly else {throw URLError(.unsupportedURL)}
  guard NBARecordedOddsRouting.canRead(key.phase,cacheOnly:key.cacheOnly,preseasonHandlerEnabled:CloudConfiguration.nbaPreseasonCacheEnabled),key.phase == nil || key.sport == "NBA" else {throw URLError(.unsupportedURL)}
  #if DEBUG && targetEnvironment(simulator)
  // Launch-only offline fixtures exercise the actual discovery/cache lifecycle.
  // Release apps cannot read these inputs and fixtures never enter preferences.
  if let text=ProcessInfo.processInfo.environment["PLONN_EXPLORE_MARKETS_FIXTURE"],let data=text.data(using:.utf8) {
   let rows=try JSONDecoder().decode([Market].self,from:data)
   if let delay=ProcessInfo.processInfo.environment["PLONN_EXPLORE_ODDS_DELAY_MS"].flatMap(Int.init),delay>0 {
    try await Task.sleep(for:.milliseconds(min(delay,2000)))
   }
   try Task.checkCancellation()
   return OddsResponse(markets:rows.filter{$0.sport == key.sport && (key.event == nil || $0.event == key.event)},message:"Offline stability fixture · No provider request")
  }
  #endif
  if key.cacheOnly || key.scope.backend.isEmpty {
   guard account.user != nil else { throw URLError(.userAuthenticationRequired) }
   // Recorded reads always use the authenticated cache-only endpoint, never a local/provider fallback.
   let body=OddsRequestBody(sport:key.sport,event:key.event,cacheOnly:key.cacheOnly,phase:key.phase == .preseason ? "preseason" : nil,recordedArchive:key.recordedArchive ? true:nil)
   return try await account.client.functions.invoke("odds",options:.init(body:body))
  }
  guard var url=URLComponents(string:key.scope.backend),url.host != nil,url.user == nil,url.password == nil else { throw URLError(.badURL) }
  var allowed=url.scheme == "https"
  #if DEBUG && targetEnvironment(simulator)
  allowed = allowed || (url.scheme == "http" && url.host == "127.0.0.1" && url.port == 8766)
  #endif
  guard allowed else { throw URLError(.badURL) }
  url.path="/api/odds";url.fragment=nil;url.queryItems=[URLQueryItem(name:"sport",value:key.sport)]
  if let event=key.event { url.queryItems?.append(URLQueryItem(name:"event",value:event)) }
  guard let endpoint=url.url else { throw URLError(.badURL) }
  var request=URLRequest(url:endpoint);request.timeoutInterval=45;request.cachePolicy = .reloadIgnoringLocalCacheData
  let (data,response)=try await URLSession.shared.data(for:request)
  guard let response=response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
  if response.statusCode == 401 || response.statusCode == 403 { throw URLError(.userAuthenticationRequired) }
  guard response.statusCode == 200 else { throw OddsServiceError(status:response.statusCode,body:data) }
  return try JSONDecoder().decode(OddsResponse.self,from:data)
 }
 var combined:(Double?,String){
  parlayEstimate(legs.map { ProbabilityLeg(event: $0.event, probability: $0.p, mode: $0.mode) })
 }
}

@main struct PLONNApp: App {
  @StateObject private var store=Store();@StateObject private var account=AccountStore();@StateObject private var purchases=PurchaseStore();var body:some Scene{WindowGroup{launchRoot.buttonStyle(PLONNGlassButtonStyle()).fontDesign(.monospaced).environmentObject(store).environmentObject(account).environmentObject(account.access).environmentObject(purchases).task {
 #if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
 if LocalStoreKitLab.active {return}
 #endif
 #if DEBUG && targetEnvironment(simulator)
 if ProcessInfo.processInfo.environment["PLONN_PRO_PREVIEW"] == "1" || MatchupOfflineFixture.active || SharedReportOfflineFixture.active || LabGamePickerFixture.active {return}
 #endif
 await account.observe() }.onOpenURL { url in Task { await account.handleLoginURL(url) } }.preferredColorScheme(.dark).tint(brandAccent)}} }
extension PLONNApp {
 @ViewBuilder var launchRoot: some View {
  #if DEBUG && targetEnvironment(simulator) && PLONN_STOREKIT_LAB
  LocalStoreKitLabView()
  #elseif DEBUG && targetEnvironment(simulator)
  if ProcessInfo.processInfo.environment["PLONN_PRO_PREVIEW"] == "1" {NavigationStack {ProPaywallView()}} else if LabGamePickerFixture.active {LabGamePickerFixtureView()} else if LiveMatchupPreview.active {LiveMatchupPreviewView()} else if SharedReportOfflineFixture.active {SharedReportOfflineFixtureView()} else if MatchupOfflineFixture.active {MatchupOfflineFixtureView()} else {PreviewAccessView()}
  #else
  PreviewAccessView()
  #endif
 }
}
struct LaunchExperience: View {
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 @State private var showingIntro=true
 @State private var appeared=false
 @State private var minimumElapsed=false
 @State private var scheduleReady=false
 @State private var oddsReady=false
 private func finishIntro() {
  guard minimumElapsed else{return}
  withAnimation(.easeInOut(duration:reduceMotion ? 0:0.65)) {showingIntro=false}
 }
 var body:some View {
  ZStack {
   RootView().scaleEffect(showingIntro && !reduceMotion ? 0.97:1).blur(radius:showingIntro && !reduceMotion ? 5:0).allowsHitTesting(!showingIntro).accessibilityHidden(showingIntro)
   if showingIntro {
    ZStack {
     ModelLandscape(accent:brandAccent)
     VStack(spacing:22) {
      ZStack {
       Circle().stroke(charcoalAccent.opacity(0.08),lineWidth:1).frame(width:176,height:176)
        .scaleEffect(appeared || reduceMotion ? 1:0.5).opacity(appeared ? 1:0)
        .animation(reduceMotion ? nil:.easeOut(duration:0.8).delay(0.35),value:appeared)
       PLONNTrident().frame(width:82,height:104)
        .scaleEffect(appeared || reduceMotion ? 1:0.65)
        .blur(radius:appeared || reduceMotion ? 0:12)
        .animation(reduceMotion ? nil:.spring(response:0.8,dampingFraction:0.72),value:appeared)
        .offset(y:appeared || reduceMotion ? 0:28)
        .shadow(color:charcoalAccent.opacity(0.3),radius:20)
      }.frame(width:280,height:280)
      VStack(spacing:10) {
       PLONNWordmark(size:42)
        .mask(alignment:.leading) {
         Rectangle().frame(width:appeared ? 42*5.9:0)
          .animation(reduceMotion ? .linear(duration:0):.linear(duration:1.15).delay(0.65),value:appeared)
        }
       TerminalReveal(text:"STICK TO THE PLONN").foregroundStyle(.secondary)
        .opacity(appeared ? 1:0).offset(y:appeared || reduceMotion ? 0:10)
        .animation(reduceMotion ? nil:.easeOut(duration:0.6).delay(1.1),value:appeared)
       PLONNLoader().controlSize(.small).tint(charcoalAccent).padding(.top,18)
        .accessibilityLabel("Loading games and odds")
      }
     }.opacity(appeared ? 1:0).offset(y:appeared || reduceMotion ? 0:10)
    }.transition(.opacity)
   }
  }.onReceive(NotificationCenter.default.publisher(for:Notification.Name("PLONNInitialScheduleReady"))) {_ in
   scheduleReady=true
   if minimumElapsed && oddsReady {finishIntro()}
  }.onReceive(NotificationCenter.default.publisher(for:Notification.Name("PLONNInitialOddsReady"))) {_ in
   oddsReady=true
   if minimumElapsed && scheduleReady {finishIntro()}
  }.task {
   withAnimation(reduceMotion ? .linear(duration:0):.spring(response:0.7,dampingFraction:0.7)) {appeared=true}
   do {try await Task.sleep(for:.seconds(5))} catch {return}
   minimumElapsed=true
   if scheduleReady && oddsReady {finishIntro();return}
   do {try await Task.sleep(for:.seconds(10))} catch {return}
   finishIntro()
  }
 }
}
struct RootView: View {
 @State private var selectedTab="Games"
 @EnvironmentObject private var account:AccountStore
 @EnvironmentObject private var purchases:PurchaseStore
 @EnvironmentObject private var store:Store
 @Environment(\.scenePhase) private var scenePhase
 init() {
  let background=appBackground
  UIRefreshControl.appearance().tintColor = .clear
  let navigation=UINavigationBarAppearance()
  if #available(iOS 26.0, *) {
   navigation.configureWithDefaultBackground()
  } else {
   navigation.configureWithOpaqueBackground();navigation.backgroundColor=background
  }
  navigation.titleTextAttributes=[.font:UIFont.monospacedSystemFont(ofSize:17,weight:.semibold)]
  navigation.largeTitleTextAttributes=[.font:UIFont.monospacedSystemFont(ofSize:34,weight:.bold)]
  let buttonFont:[NSAttributedString.Key:Any] = [.font:UIFont.monospacedSystemFont(ofSize:17,weight:.regular)]
  navigation.buttonAppearance.normal.titleTextAttributes=buttonFont
  navigation.backButtonAppearance.normal.titleTextAttributes=buttonFont
  UIBarButtonItem.appearance().setTitleTextAttributes(buttonFont,for:.normal)
  UISegmentedControl.appearance().setTitleTextAttributes([.font:UIFont.monospacedSystemFont(ofSize:13,weight:.medium)],for:.normal)
  UISegmentedControl.appearance().setTitleTextAttributes([.font:UIFont.monospacedSystemFont(ofSize:13,weight:.semibold)],for:.selected)
  UINavigationBar.appearance().standardAppearance=navigation;UINavigationBar.appearance().scrollEdgeAppearance=navigation;UINavigationBar.appearance().compactAppearance=navigation;UINavigationBar.appearance().compactScrollEdgeAppearance=navigation
  let tabs=UITabBarAppearance()
  if #available(iOS 26.0, *) {
   tabs.configureWithDefaultBackground()
  } else {
   tabs.configureWithOpaqueBackground();tabs.backgroundColor=background
  }
  let accent=UIColor(brandAccent)
  for item in [tabs.stackedLayoutAppearance,tabs.inlineLayoutAppearance,tabs.compactInlineLayoutAppearance] { item.selected.iconColor=accent;item.normal.titleTextAttributes=[.font:UIFont.monospacedSystemFont(ofSize:10,weight:.medium)];item.selected.titleTextAttributes=[.foregroundColor:accent,.font:UIFont.monospacedSystemFont(ofSize:10,weight:.semibold)] }
  UITabBar.appearance().standardAppearance=tabs;UITabBar.appearance().scrollEdgeAppearance=tabs
 }
 var body: some View {
  TabView(selection:$selectedTab) {
   NavigationStack { TodayView() }.tabItem { Label("Games",systemImage:"calendar") }.tag("Games")
   ExploreView().safeAreaInset(edge:.bottom,spacing:8) {
    if account.hasSignedInAccount,!store.legs.isEmpty {
     Button {selectedTab="Builder"} label: {
      HStack(spacing:12) {
       Image(systemName:"square.stack.3d.up.fill")
       Text("\(store.legs.count) \(store.legs.count == 1 ? "pick":"picks")").monospacedDigit()
       Spacer()
       Text("View Builder").fontWeight(.semibold)
       Image(systemName:"arrow.right")
      }.font(.subheadline).foregroundStyle(.white).padding(.horizontal,18).frame(minHeight:54)
       .background(PLONNPalette.gradient.opacity(0.3),in:Capsule())
       .plonnGlassSurface(selected:true,radius:28)
     }.buttonStyle(.plain).padding(.horizontal,16).padding(.bottom,4)
      .accessibilityIdentifier("analytics-builder-bar")
      .accessibilityLabel("View Builder, \(store.legs.count) picks")
    }
   }.tabItem { Label("Analytics",systemImage:"chart.xyaxis.line") }.tag("Analytics")
   NavigationStack { AILabView() }.id("Models|"+(account.user?.id.uuidString ?? "guest")).tint(brandAccent).tabItem { Label("Models",systemImage:"flask") }.tag("Models")
   NavigationStack { AskPLONNView() }.id("LLM|"+(account.user?.id.uuidString ?? "guest")).tabItem { Label("Ask PLONN",systemImage:"bubble.left.and.bubble.right") }.tag("LLM")
   BuilderView(openAnalytics:{selectedTab="Analytics"}).id("Builder|"+(account.user?.id.uuidString ?? "guest")).tint(brandAccent).tabItem { Label("Builder",systemImage:"square.stack.3d.up") }.tag("Builder")
  }
  .tint(brandAccent)
  .onReceive(NotificationCenter.default.publisher(for:Notification.Name("PLONNOpenAnalytics"))) {_ in selectedTab="Analytics"}
  .task(id:account.user?.id) {await purchases.bind(account.user?.id, accountStore: account)}
  .task {await purchases.observeTransactions()}
  .task {await purchases.observePurchaseIntents()}
  .sheet(item:$purchases.purchaseRequest) {request in
   NavigationStack {ProPaywallView(initialProductID:request.productID)}.tint(brandAccent)
  }
  .onChange(of:scenePhase) {_,phase in if phase == .active {Task {await purchases.refreshEntitlements()}}}
  .task {store.bindAccount(account)}
 }
}

struct Brand:View{var body:some View{HStack{PLONNTrident().frame(width:30,height:34);PLONNWordmark(size:20)}}}
struct Panel<Content:View>:View{let content:Content;init(@ViewBuilder content:()->Content){self.content=content()};var body:some View{content.padding().frame(maxWidth:.infinity,alignment:.leading).background(navySurface,in:RoundedRectangle(cornerRadius:18))}}
struct Metric:View{let title:String;let value:String;var body:some View{VStack(alignment:.leading,spacing:7){StatLabel(title:title).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary);Text(value).font(.system(.title3, design: .monospaced).weight(.semibold)).monospacedDigit().foregroundStyle(charcoalAccent).minimumScaleFactor(0.7)}.frame(maxWidth:.infinity,alignment:.leading)}}
struct MarketCard:View{@EnvironmentObject var store:Store;let m:Market;var body:some View {AccountRequiredView(feature:.odds){protectedBody}}
 private var protectedBody:some View {Panel{VStack(alignment:.leading,spacing:12){HStack{Text(m.sport+" / "+(m.kind=="prop" ? "PLAYER PROP":"TEAM")).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary);Spacer();Text(m.mode=="demo" ? "SAMPLE":"ODDS SNAPSHOT").font(.system(.caption2, design: .monospaced)).foregroundStyle(charcoalAccent)};HStack(spacing:12){if m.kind == "prop" { PlayerPortrait(market:m) };Text(m.name).font(.system(.title3, design: .monospaced).bold())};Text(m.eventLabel ?? ("vs "+m.opponent)).font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary);Text(m.label);HStack{Metric(title:"Sportsbook",value:odds(m.price));Metric(title:"Model",value:probability(m.p));Metric(title:"Model fair",value:odds(fair(m.p)))};Text(m.book+" · "+(m.updated ?? "Illustrative prices")).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary);HStack{NavigationLink("Analysis"){DetailView(m:m)};Spacer();Button(store.legs.contains(where:{$0.id==m.id}) ? "Added":"+ Parlay"){store.add(m)};Button{store.toggleSave(m)}label:{Image(systemName:store.saved.contains(where:{$0.id==m.id}) ? "star.fill":"star")}.accessibilityLabel("Save "+m.name)}}}}}
// Keep existing deep links pointing at the new historical analytics surface.
struct ExploreView: View {
 let initialSport:String
 let initialQuery:String
 init(initialSport:String="All",initialQuery:String="",live:Bool=true) {
  self.initialSport=initialSport;self.initialQuery=initialQuery
 }
 var body:some View {AnalyticsView(initialSport:initialSport,initialQuery:initialQuery)}
}

private struct ExploreMatchupMarkets: View {
 @EnvironmentObject private var store:Store
 let markets:[Market]
 private var sections:[String] {Array(Set(markets.map(\.market))).sorted {a,b in
  let order=["h2h","spreads","totals"]
  return (order.firstIndex(of:a) ?? 99) < (order.firstIndex(of:b) ?? 99)
 }}
 private func title(_ key:String)->String {
  switch key {case "h2h":return "Moneyline";case "spreads":return markets.first?.sport == "MLB" ? "Run line":"Spread";case "totals":return "Total";default:return key.replacingOccurrences(of:"_",with:" ").capitalized}
 }
 private func label(_ market:Market)->String {
  let side=market.side ?? market.name
  guard let line=market.line else{return side}
  let number=line.formatted(.number.precision(.fractionLength(0...1)))
  return side + " " + (market.market == "spreads" && line > 0 ? "+":"") + number
 }
 var body:some View {
  VStack(alignment:.leading,spacing:16) {
   if let first=markets.first {
    Text(first.sport).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
    Text(first.eventLabel ?? first.name+" vs "+first.opponent).font(.system(.headline,design:.monospaced)).accessibilityAddTraits(.isHeader)
   }
   ForEach(sections,id:\.self) { key in
    VStack(alignment:.leading,spacing:10) {
     Text(title(key)).font(.system(.subheadline,design:.monospaced).bold())
     ForEach(markets.filter{$0.market == key}.sorted{$0.name == $1.name ? ($0.line ?? 0)<($1.line ?? 0):$0.name<$1.name}) { market in
      HStack(spacing:10) {
       NavigationLink {DetailView(m:market)} label: {
        VStack(alignment:.leading,spacing:4) {
         Text(label(market)).font(.system(.subheadline,design:.monospaced))
         Text(market.book).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
        }.frame(maxWidth:.infinity,alignment:.leading)
        VStack(alignment:.trailing,spacing:4) {
         Text(odds(market.price)).font(.system(.subheadline,design:.monospaced).bold())
         Text("Implied "+marketImpliedProbability(market.price)).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
        }
       }.buttonStyle(.plain)
       Button {store.add(market)} label: {Image(systemName:store.legs.contains{$0.id == market.id} ? "checkmark.circle.fill":"plus.circle").frame(width:44,height:44)}.plonnGlassButton().accessibilityLabel("Add "+label(market)+" to selections")
      }
     }
    }
    if key != sections.last {Divider()}
   }
   Text("Sportsbook snapshots").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
  }.padding(14).background(navySurface,in:RoundedRectangle(cornerRadius:16))
 }
}

private struct ExploreMarketRow: View {
 @EnvironmentObject var store: Store
 let m: Market

 var body:some View {AccountRequiredView(feature:.odds){protectedBody}}
 private var protectedBody:some View {
  VStack(alignment: .leading, spacing: 12) {
   NavigationLink {
    DetailView(m: m)
   } label: {
    VStack(alignment: .leading, spacing: 8) {
     HStack {
      Text(m.kind == "prop" ? "PLAYER PROP" : "TEAM MARKET")
      Spacer()
      Text(m.mode == "demo" ? "SAMPLE" : "ODDS SNAPSHOT")
     }.font(.system(.caption2, design: .monospaced).weight(.medium)).foregroundStyle(.secondary)
     HStack(alignment: .center, spacing: 12) {
      ExploreIdentity(m:m)
      Text(m.name).font(.system(.headline, design: .monospaced).weight(.semibold)).frame(maxWidth: .infinity, alignment: .leading)
      VStack(alignment: .trailing, spacing: 3) {
       Text("ODDS").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
       Text(m.price.isFinite ? odds(m.price) : "—").font(.system(.headline, design: .monospaced).weight(.semibold)).monospacedDigit()
      }
      Image(systemName: "chevron.right").font(.system(.caption, design: .monospaced).weight(.semibold)).foregroundStyle(.tertiary)
     }
     Text(m.label).font(.system(.subheadline, design: .monospaced).weight(.semibold)).monospacedDigit()
     Text("Implied win probability · "+marketImpliedProbability(m.price)).font(.system(.caption, design: .monospaced).weight(.medium)).monospacedDigit()
     Text(m.eventLabel ?? "vs " + m.opponent).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
    }.contentShape(Rectangle())
   }.buttonStyle(.plain)

   HStack(alignment: .center, spacing: 12) {
    VStack(alignment: .leading, spacing: 4) {
     Text(m.book).font(.system(.caption, design: .monospaced))
     Text(m.mode == "demo" ? "Illustrative prices" : (m.updated ?? "Snapshot time unavailable"))
      .font(.system(.caption2, design: .monospaced))
    }.foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
    Button { store.add(m) } label: {
     Image(systemName: store.legs.contains(where: { $0.id == m.id }) ? "checkmark.circle.fill" : "plus.circle")
      .font(.system(.title3, design: .monospaced)).frame(minWidth: 44, minHeight: 44)
    }.accessibilityLabel(store.legs.contains(where: { $0.id == m.id }) ? "Added to selections" : "Add " + m.name + " to selections")
    Button { store.toggleSave(m) } label: {
     Image(systemName: store.saved.contains(where: { $0.id == m.id }) ? "bookmark.fill" : "bookmark")
      .font(.system(.body, design: .monospaced)).frame(minWidth: 44, minHeight: 44)
    }.accessibilityLabel((store.saved.contains(where: { $0.id == m.id }) ? "Unsave " : "Save ") + m.name)
   }.plonnGlassButton().plonnGlassGroup()
  }.padding(14).background(ExploreTeamSurface(m:m))
 }
}
struct DetailView:View {
 @EnvironmentObject var store:Store
 let m:Market
 var body:some View {AccountRequiredView(feature:.odds){protectedBody}}
 private var protectedBody:some View {
  ScrollView {
   VStack(alignment:.leading,spacing:16) {
    HStack(spacing:14) {
     if m.kind == "prop" { PlayerPortrait(market:m,size:64) }
     VStack(alignment:.leading,spacing:5) {
      Text(m.name).font(.system(.title2, design: .monospaced).weight(.semibold))
      Text(m.eventLabel ?? "vs "+m.opponent).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
     }
    }
    HStack(alignment:.top,spacing:16) {
     VStack(alignment:.leading,spacing:7) {
      Text("PROP LINE").font(.system(.caption2, design: .monospaced).weight(.semibold)).tracking(0.8).foregroundStyle(.secondary)
      Text(m.kind == "prop" ? compactPropLabel(m) : m.label).font(.system(size:18,weight:.semibold,design:.monospaced)).monospacedDigit()
     }.frame(maxWidth:.infinity,alignment:.leading)
     VStack(alignment:.trailing,spacing:7) {
      Text("ODDS").font(.system(.caption2, design: .monospaced).weight(.semibold)).tracking(0.8).foregroundStyle(.secondary)
      Text(odds(m.price)).font(.system(size:22,weight:.bold,design:.monospaced)).monospacedDigit()
      Text(m.book).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     }
    }.padding(16).background(PlayerTeamBackground(market:m))
    if m.mode == "demo" { Text("Sample data · Illustrative prices and results").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
    TimelineView(.periodic(from:Date(),by:5)) { context in
     SportsbookPriceCard(comparison:AskPLONNResearch.priceComparison(market:m,markets:[m]+store.live+store.saved+store.legs,now:context.date))
    }
    if m.kind == "prop" && m.mode == "live" { PlayerHistoryView(m:m) }
    NavigationLink { AskPLONNView(initialMarket:m) } label: {
     Label("Research this selection with LLM",systemImage:"bubble.left.and.bubble.right").font(.system(.subheadline,design:.monospaced))
    }.plonnGlassButton()
    if let logs=m.logs { Panel { VStack(alignment:.leading,spacing:8) { Text("Illustrative results").font(.system(.headline, design: .monospaced));Text(logs.map{String(format:"%g",$0)}.joined(separator:" · "));Text("Sample values, not actual player results.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) } } }
    if let p=m.p { DisclosureGroup("Model estimate") { HStack { Metric(title:"Probability",value:probability(p));Metric(title:"Fair odds",value:odds(fair(p))) }.padding(.top,12) } }
    Text("Research only · No bets placed in PLONN").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
   }.padding(20)
  }.background(navy).navigationTitle("Player analysis").navigationBarTitleDisplayMode(.inline)
   .safeAreaInset(edge:.bottom) {
    Button { store.add(m) } label: { Text(store.legs.contains(where:{$0.id==m.id}) ? "Added to selections":"Add selection").font(.system(.headline, design: .monospaced)).frame(maxWidth:.infinity).padding(.vertical,8) }.plonnGlassButton(prominent:true).foregroundStyle(.primary).padding(.horizontal,20).padding(.vertical,10).background(navy)
   }
 }
}
struct WatchView:View{@EnvironmentObject var store:Store;var body:some View {AccountRequiredView(feature:.savedResearch){protectedBody}}
 private var protectedBody:some View {NavigationStack{ScrollView{LazyVStack(spacing:16){if store.saved.isEmpty{ContentUnavailableView("Nothing saved yet",systemImage:"star",description:Text("Choose a team or player in Analytics."))};ForEach(store.saved){MarketCard(m:$0)}}.padding()}.background(navy).navigationTitle("Watchlist")}}}
struct BuilderView: View {
 let openAnalytics:()->Void
 @EnvironmentObject private var account:AccountStore
 @EnvironmentObject private var store:Store
 private var notice:String {store.builderDraftIssue}
 @State private var confirmClear=false
 var body:some View {AccountRequiredView(feature:.savedResearch){protectedBody}}
 private var protectedBody:some View {
  NavigationStack {
   ScrollView {
    VStack(alignment:.leading,spacing:20) {
     HStack(alignment:.firstTextBaseline) {
      Text("Builder").font(.system(.title2,design:.monospaced,weight:.semibold))
      Spacer()
      Text("\(store.legs.count)/8 legs").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
     }
     if !notice.isEmpty {
      Label(notice,systemImage:"info.circle").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).accessibilityIdentifier("builder-notice")
     }
     if store.builderDraftBlocked {
      Button("Reset unreadable saved draft",role:.destructive) {store.resetBuilderDraft()}
     }
     if store.legs.isEmpty { emptyState }
     else {
      TimelineView(.periodic(from:.now,by:30)) { context in
       VStack(alignment:.leading,spacing:14) {
        BuilderPricePanel(legs:store.legs,now:context.date)
        ForEach(store.legs) { market in
         BuilderLegCard(market:market,now:context.date) {store.legs.removeAll{$0.id==market.id}}
        }
       }
      }
      HStack(spacing:12) {
       Button(action:openAnalytics) {Label("Add picks",systemImage:"plus").frame(minHeight:44)}
        .plonnGlassButton().disabled(store.legs.count>=BuilderLogic.limit).accessibilityIdentifier("builder-open-analytics")
       Spacer(minLength:0)
       Button("Clear draft",role:.destructive) {confirmClear=true}.frame(minHeight:44)
      }
      Text("Saved on this device · No bets placed in PLONN")
       .font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     }
    }.padding(16)
   }.scrollDismissesKeyboard(.interactively).background(LeagueBackdrop(league:"All")).navigationTitle("PLONN").navigationBarTitleDisplayMode(.inline).toolbar { ToolbarItem(placement:.principal) {Brand()} }
    .confirmationDialog("Clear all selections and the local draft?",isPresented:$confirmClear,titleVisibility:.visible) {
     Button("Clear draft",role:.destructive) {store.clearBuilderDraft()}
    }
  }
 }
 private var emptyState:some View {
  VStack(alignment:.leading,spacing:16) {
   Image(systemName:"square.stack").font(.system(size:30,weight:.light,design:.monospaced)).foregroundStyle(.secondary)
   Text("Build your parlay").font(.system(.title3,design:.monospaced,weight:.semibold))
   Text("Add picks from Analytics to see combined odds and your potential return.")
    .font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary)
   Button(action:openAnalytics) {Label("Find picks in Analytics",systemImage:"plus").frame(maxWidth:.infinity,minHeight:44)}
    .plonnGlassButton(prominent:true).foregroundStyle(.primary).accessibilityIdentifier("builder-open-analytics")
  }.padding(20).frame(maxWidth:.infinity,alignment:.leading).background(navySurface,in:RoundedRectangle(cornerRadius:18))
 }
}
private struct BuilderLegCard:View {
 @AppStorage("oddsFormat") private var oddsFormat="american"
 let market:Market
 let now:Date
 let remove:()->Void
 private var modelText:String {
  guard let p=market.p,p.isFinite,p>0,p<1 else{return "Model estimate unavailable"}
  if market.mode == "demo" {return "Illustrative model: "+probability(p)}
  guard market.modelValidated == true,let version=market.modelVersion,!version.isEmpty,
        let updated=market.modelUpdated.flatMap(sportsDate),(0...120).contains(now.timeIntervalSince(updated)) else{return "Current model estimate unavailable"}
  return "Model estimate: "+probability(p)+" · "+version
 }
 var body:some View {
  VStack(alignment:.leading,spacing:12) {
   HStack(spacing:8) {
    Text(market.sport+" · "+(market.mode == "demo" ? "SAMPLE":"SNAPSHOT"))
     .font(.system(.caption2,design:.monospaced,weight:.semibold)).foregroundStyle(.secondary)
    Spacer()
    Button(action:remove) {Image(systemName:"xmark").font(.system(.caption, design: .monospaced).weight(.semibold)).frame(width:44,height:44)}
     .plonnGlassButton().accessibilityLabel("Remove "+market.name+" "+market.label)
   }
   NavigationLink {DetailView(m:market)} label: {
    HStack(alignment:.top,spacing:12) {
     ExploreIdentity(m:market).accessibilityHidden(true)
     VStack(alignment:.leading,spacing:6) {
      Text(market.name).font(.system(.headline,design:.monospaced))
      Text(market.label).font(.system(.subheadline,design:.monospaced,weight:.semibold))
      Text(market.eventLabel ?? "vs "+market.opponent).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
     }.frame(maxWidth:.infinity,alignment:.leading)
     Image(systemName:"chevron.right").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
    }.contentShape(Rectangle())
   }.buttonStyle(.plain).accessibilityHint("Opens the original selection analysis")
   HStack(alignment:.top) {
    VStack(alignment:.leading,spacing:4) {
     Text(market.book.isEmpty ? "Book unavailable":market.book).font(.system(.caption, design: .monospaced).weight(.semibold))
     Text(BuilderLogic.quoteAge(market,now:now).label).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     if market.mode != "demo",let time=market.updated.flatMap(sportsDate) {
      Text(time.formatted(date:.abbreviated,time:.shortened)).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     }
    }.frame(maxWidth:.infinity,alignment:.leading)
    VStack(alignment:.trailing,spacing:4) {
     Text(BuilderLogic.decimalOdds(market.price) == nil ? "—":odds(market.price))
      .accessibilityLabel((oddsFormat == "decimal" ? "Decimal odds: ":"American odds: ") + (BuilderLogic.decimalOdds(market.price) == nil ? "Unavailable":odds(market.price)))
      .font(.system(.title3,design:.monospaced,weight:.semibold)).monospacedDigit()
     Text(BuilderLogic.decimalOdds(market.price).map{String(format:"%.3f decimal",$0)} ?? "Price unavailable")
      .font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
    }
   }
   if market.p != nil {Text(modelText).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)}
  }.padding(14).background(ExploreTeamSurface(m:market))
 }
}
private struct BuilderPricePanel:View {
 @AppStorage("oddsFormat") private var oddsFormat="american"
 @State private var stakeText=""
 @FocusState private var stakeFocused:Bool
 let legs:[Market]
 let now:Date
 private func money(_ value:Double)->String {value.formatted(.currency(code:"USD"))}
 var body:some View {
  let result=BuilderLogic.assess(legs,now:now)
  let payout=BuilderLogic.returnEstimate(stakeText:stakeText,estimate:result.estimate)
  VStack(alignment:.leading,spacing:16) {
   if let estimate=result.estimate {
    HStack(alignment:.top) {
     VStack(alignment:.leading,spacing:5) {
      Text("Combined odds").font(.caption).foregroundStyle(.secondary)
      Text(odds(estimate.american)).font(.title2.bold()).monospacedDigit().accessibilityIdentifier("builder-combined-odds")
     }.frame(maxWidth:.infinity,alignment:.leading)
     VStack(alignment:.leading,spacing:5) {
      Text("Implied probability").font(.caption).foregroundStyle(.secondary)
      Text(estimate.priceImpliedProbability.formatted(.percent.precision(.fractionLength(1)))).font(.title2.bold()).monospacedDigit().accessibilityIdentifier("builder-implied-probability")
     }.frame(maxWidth:.infinity,alignment:.leading)
    }
    Text("Estimate from recorded prices · Confirm the final quote at your sportsbook.").font(.caption2).foregroundStyle(.secondary)
   } else {
    Text("Combined odds unavailable").font(.headline)
    Text(result.blockingReason ?? "Review your picks.").font(.caption).foregroundStyle(.secondary)
   }
   HStack {
    Text("Stake").font(.subheadline)
    Spacer()
    Text("$").foregroundStyle(.secondary)
    TextField("0.00",text:$stakeText).keyboardType(.decimalPad).focused($stakeFocused)
     .multilineTextAlignment(.trailing).frame(width:120).frame(minHeight:44)
     .accessibilityLabel("Stake in US dollars").accessibilityIdentifier("builder-stake")
   }.padding(.horizontal,12).plonnGlassSurface(radius:12,interactive:false)
   HStack(alignment:.top) {
    VStack(alignment:.leading,spacing:5) {
     Text("Potential return").font(.caption).foregroundStyle(.secondary)
     Text(payout.map {money($0.totalReturn)} ?? "—").font(.title2.bold()).foregroundStyle(brandAccent).monospacedDigit().accessibilityIdentifier("builder-total-return")
     Text("Includes your stake").font(.caption2).foregroundStyle(.secondary)
    }.frame(maxWidth:.infinity,alignment:.leading)
    VStack(alignment:.leading,spacing:5) {
     Text("Profit").font(.caption).foregroundStyle(.secondary)
     Text(payout.map {money($0.profit)} ?? "—").font(.title2.bold()).monospacedDigit().accessibilityIdentifier("builder-profit")
    }.frame(maxWidth:.infinity,alignment:.leading)
   }
   if !stakeText.isEmpty,BuilderLogic.stakeAmount(stakeText) == nil {
    Text("Enter a positive dollar amount with up to two decimal places.").font(.caption).foregroundStyle(.red)
   }
   DisclosureGroup("Calculation details") {
    VStack(alignment:.leading,spacing:10) {
     Text("Potential return assumes every pick wins. Odds are multiplied for different games; actual sportsbook pricing can differ. Implied probability comes from those prices, includes bookmaker margin and is not a model prediction.")
     ForEach(result.warnings,id:\.self) {Text($0)}
    }.font(.caption).foregroundStyle(.secondary).padding(.top,8)
   }
   if result.mixedBooks {Text("Picks use different sportsbooks.").font(.caption).foregroundStyle(.orange)}
  }.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(navySurface,in:RoundedRectangle(cornerRadius:16))
   .toolbar {ToolbarItemGroup(placement:.keyboard) {Spacer();Button("Done") {stakeFocused=false}}}
 }
}


#if DEBUG
struct ResultsView:View{@EnvironmentObject var store:Store;var settled:[HistoryRow]{store.history.filter{["W","L"].contains($0.result) && Double($0.profit) != nil}};var body:some View{NavigationStack{List{Section{Text("Imported local ledger, unaudited. Includes losses. Not evidence validating sample predictions.").font(.system(.caption, design: .monospaced));HStack{Metric(title:"Settled",value:String(settled.count));Metric(title:"Wins",value:String(settled.filter{$0.result=="W"}.count));Metric(title:"Net recorded",value:String(format:"%.2f",settled.reduce(0){$0+(Double($1.profit) ?? 0)}))}};Section("All imported records"){ForEach(Array(store.history.enumerated()),id:\.offset){_,r in HStack{VStack(alignment:.leading){Text(r.pick_team+" vs "+r.opponent);Text(r.date).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)};Spacer();Text(r.result).foregroundStyle(r.result=="W" ? charcoalAccent:Color.red);Text(r.profit)}}}}.navigationTitle("Recorded results")}}}
#endif
struct SettingsView:View{@AppStorage("oddsFormat") private var oddsFormat="american";@AppStorage("preferredSportsbook") private var preferredBook="Ask each time";@EnvironmentObject var store:Store;var body:some View{NavigationStack{Form{Section("Odds"){Picker("Odds format",selection:$oddsFormat){Text("American (+150)").tag("american");Text("Decimal (2.50)").tag("decimal")}.pickerStyle(.segmented)};Section("Account"){SportsbookSelector(selection:$preferredBook,options:["Ask each time","DraftKings","FanDuel","Hard Rock Bet","BetMGM","Caesars","Fanatics","BetRivers"]);Text("Preferred sportsbook for model lines. Saved on this device.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary);NavigationLink("Account"){AccountView()}
#if DEBUG
NavigationLink("Subscription"){SubscriptionView()}
#else
if PurchaseConfiguration.salesEnabled {NavigationLink("Subscription"){SubscriptionView()}}
#endif
NavigationLink("Saved research"){WatchView()}
#if DEBUG
NavigationLink("Recorded results"){ResultsView()}
#endif
};ProSubscriptionSection();Section("Privacy"){NavigationLink("Privacy & data"){PrivacyDataView()};Text("Your saved watchlist stays on this device unless you choose to sync it through your account. Live odds requests include the selected sport. Research drafts stay on this device.");Button("Clear saved watchlist"){store.clearWatchlist()}}}.navigationTitle("Settings")}}}

private actor ExploreTeams {
 static let shared=ExploreTeams()
 struct Team:Sendable {let name:String;let logo:String;let color:String}
 private var cache:[String:[Team]]=[:]
 func team(_ market:Market) async -> Team? {
  if cache[market.sport] == nil {
   do {
    let url=URL(string:"https://site.api.espn.com/apis/site/v2/sports/"+sportsFeedPath(market.sport)+"/teams?limit=1000")!
    let(data,_)=try await URLSession.shared.data(from:url)
    let root=try JSONSerialization.jsonObject(with:data) as? [String:Any]
    let sports=root?["sports"] as? [[String:Any]]
    let leagues=sports?.first?["leagues"] as? [[String:Any]]
    cache[market.sport]=(leagues?.first?["teams"] as? [[String:Any]] ?? []).compactMap {entry in
     guard let t=entry["team"] as? [String:Any],let name=t["displayName"] as? String else{return nil}
     return Team(name:name,logo:(t["logos"] as? [[String:Any]])?.first?["href"] as? String ?? "",color:t["color"] as? String ?? "")
    }
   } catch {return nil}
  }
  return cache[market.sport]?.first{$0.name.caseInsensitiveCompare(market.name) == .orderedSame}
 }
}
private struct ExploreIdentity:View {
 let m:Market
 @State private var team:ExploreTeams.Team?
 var body:some View {
  Group {
   if m.kind == "prop" {PlayerPortrait(market:m,size:42)}
   else {AsyncImage(url:URL(string:team?.logo ?? "")) {image in image.resizable().scaledToFit()} placeholder:{Image(systemName:"sportscourt").foregroundStyle(.secondary)}.frame(width:42,height:42)}
  }.task(id:m.id) {if m.kind != "prop" {team=await ExploreTeams.shared.team(m)}}
 }
}

private struct ExploreTeamSurface:View {
 let m:Market
 @Environment(\.colorScheme) private var scheme
 @State private var team:ExploreTeams.Team?
 var body:some View {
  Group {
   if m.kind == "prop" {PlayerTeamBackground(market:m)}
   else {
    LinearGradient(colors:[matchupColor(team?.color ?? "").opacity(scheme == .dark ? 0.50:0.24),navySurface],startPoint:.topLeading,endPoint:.bottomTrailing).clipShape(RoundedRectangle(cornerRadius:14))
   }
  }.task(id:m.id) {if m.kind != "prop" {team=await ExploreTeams.shared.team(m)}}
 }
}

private struct ExploreTrend:Identifiable {var id:String {market.id};let market:Market;let hits:Int;let total:Int;let average:Double}

private func estimatedParlayOdds(_ legs:[Market])->String {
 guard Set(legs.map(\.event)).count == legs.count else{return "Quote needed"}
 guard !legs.isEmpty,legs.allSatisfy({$0.price.isFinite && abs($0.price)>=100}) else{return "Unavailable"}
 let decimal=legs.reduce(1.0) {total,m in total*(m.price>0 ? 1+m.price/100:1+100/abs(m.price))}
 guard decimal.isFinite,decimal>1 else{return "Unavailable"}
 return odds(decimal>=2 ? (decimal-1)*100 : -100/(decimal-1))
}
private struct ExploreParlayDetail:View {
 @EnvironmentObject var store:Store
 let legs:[Market]
 @ObservedObject var evidence:ParlayEvidenceStore
 var body:some View {AccountRequiredView(feature:.savedResearch){protectedBody}}
 private var protectedBody:some View {
  ScrollView {VStack(alignment:.leading,spacing:20) {
   Text("Estimated combined odds").font(.system(.headline, design: .monospaced))
   Text(estimatedParlayOdds(legs)).font(.system(size:38,weight:.bold,design:.monospaced)).monospacedDigit()
   Text("Estimated implied probability · "+parlayImpliedProbability(legs)).font(.system(.subheadline, design: .monospaced).weight(.medium)).monospacedDigit()
   Text("Implied probability comes from the price, includes the bookmaker’s margin, and is not a model forecast.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   Text(Set(legs.map(\.event)).count<legs.count ? "Same-game legs may be correlated. Combined odds and probability need a sportsbook quote or a validated joint model; neither is connected yet.":"Calculated by multiplying the displayed decimal odds. Actual sportsbook pricing may differ. This is not a model probability.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   if Set(legs.map(\.book)).count>1 {Text("These selections use different sportsbooks. Confirm both prices at one book before comparing its parlay quote.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
   Text("Why consider these legs?").font(.system(.title3, design: .monospaced).bold())
   Text("Each leg is evaluated separately. These combinations are research ideas, not validated recommendations.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   ForEach(legs) {m in
    VStack(alignment:.leading,spacing:12){ExploreMarketRow(m:m)
     if let finding=evidence.rows[m.id] {
      Text(finding.headline).font(.system(.headline, design: .monospaced))
      Text(finding.detail).font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary)
      Text(finding.source).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     } else {PLONNLoader("Checking recent games…")}
     NavigationLink("View full analysis"){DetailView(m:m)}.font(.system(.subheadline, design: .monospaced))
    }.padding().background(navySurface,in:RoundedRectangle(cornerRadius:14))
   }
   Text("Historical results do not establish an edge at today’s odds. Opponent context, injuries, and correlation still need review.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   Button {for m in legs {store.add(m)}} label:{Text("Add both to selections").font(.system(.headline, design: .monospaced)).frame(maxWidth:.infinity).padding(14)}.plonnGlassButton(prominent:true).foregroundStyle(.primary)
  }.padding(16)}.background(navy).navigationTitle(Set(legs.map(\.event)).count<legs.count ? "Same-game parlay":"2-leg parlay").navigationBarTitleDisplayMode(.inline).task {for m in legs {await evidence.load(m)}}
 }
}

// Original trident asset; alternate P exploration is deferred.
struct PLONNTrident:View {
 var body:some View {
  Image("trident").renderingMode(.original).resizable().scaledToFit().accessibilityLabel("PLONN")
 }
}
private func brandPolygon(_ points:[[CGFloat]],size:CGSize)->Path {
 Path { path in
  for (i,p) in points.enumerated() {
   let pt=CGPoint(x:p[0]*size.width/100,y:p[1]*size.height/100)
   if i==0 {path.move(to:pt)} else {path.addLine(to:pt)}
  }
  path.closeSubpath()
 }
}
struct PLONNWordmark:View {
 var size:CGFloat=20
 var body:some View {
  GeometryReader { g in
   Path { path in
    let letters:[[[[CGFloat]]]] = [
     [[[0,0],[19,0],[16,9],[6,13],[6,20],[0,20],[0,10],[12,6],[12,5],[0,5]]],
     [[[0,0],[6,0],[6,15],[19,15],[19,20],[0,20]]],
     [[[0,0],[19,0],[19,20],[0,20]],[[6,5],[6,15],[13,15],[13,5]]],
     [[[0,20],[0,0],[6,0],[13,11],[13,0],[19,0],[19,20],[13,20],[6,9],[6,20]]],
     [[[0,20],[0,0],[6,0],[13,11],[13,0],[19,0],[19,20],[13,20],[6,9],[6,20]]]]
    for (index,contours) in letters.enumerated() {
     for points in contours {
      for (i,p) in points.enumerated() {
       let pt=CGPoint(x:(CGFloat(index)*23+p[0]+(20-p[1])*0.32)*g.size.width/118,y:p[1]*g.size.height/20)
       if i==0 {path.move(to:pt)} else {path.addLine(to:pt)}
      }
      path.closeSubpath()
     }
    }
   }.fill(PLONNPalette.highlight,style:FillStyle(eoFill:true))
  }.frame(width:size*5.9,height:size).accessibilityLabel("PLONN")
 }
}

struct PrivacyDataView:View {
 var body:some View {
  List {
   Section("Your account") {Text("When you create an account, PLONN sends your email and display name to its Supabase account service. Password authentication is handled by Supabase. Your session is stored in the iOS Keychain.")}
   Section("Saved research") {Text("Watchlists, device-only setups and app preferences are saved on this device. Selections you explicitly copy to your account and private model configurations you save in My Models are stored in Supabase. These model versions are not shared publicly. Removing cloud records does not remove device copies.")}
   Section("Sports data") {Text("PLONN requests scores, rosters, statistics, and images from sports-data services, and sportsbook prices through its backend. Network requests expose your IP address to the service receiving them. PLONN does not send your account password to sports-data providers.")}
   if CloudConfiguration.servicePolicy.chatAvailable {
    Section("AI chat") {Text("AI chat is optional. After you enable it for your signed-in account, your questions, recent conversation, and attached research are sent to PLONN’s AI service. Signing out clears this chat permission. Avoid including sensitive personal information.")}
   }
   if CloudConfiguration.servicePolicy.publicPagesApproved {
    Section("PLONN information") {Link("PLONN privacy policy",destination:PLONNServicePolicy.privacyURL);Link("PLONN support",destination:PLONNServicePolicy.supportURL)}
   }
   Section("Tracking and payments") {Text("This build contains no advertising or analytics SDK and does not accept wagers. In-app purchases are disabled until App Store products are configured. If enabled, Apple processes payment and receives an account token linking the transaction to your PLONN account.")}
   Section("Delete your data") {Text("Use Account → Delete account to remove your account, saved cloud research, and hosted runs and artifacts. Account-linked usage is removed; anonymous shared-service totals may remain. Clear the device watchlist and device-only setups separately. Deleting your account does not cancel an Apple subscription. Signing out does not delete your account or your device selections.")}
  }.navigationTitle("Privacy & data").navigationBarTitleDisplayMode(.inline)
 }
}

private struct ParlayFinding {let headline:String;let detail:String;let source:String;var supported:Bool = true}
@MainActor private final class ParlayEvidenceStore:ObservableObject {
 @Published var rows:[String:ParlayFinding]=[:]
 private(set) var checkedAt:[String:Date]=[:]
 private var contexts:[String:String]=[:]
 private var pending=Set<String>()
 private func context(_ market:Market)->String {[market.sport,market.event,market.name,market.market,market.side ?? "",market.line.map(String.init(describing:)) ?? "",market.commence ?? ""].joined(separator:"|")}
 func finding(for market:Market)->ParlayFinding? {contexts[market.id] == context(market) ? rows[market.id]:nil}
 func load(_ market:Market) async {
  if contexts[market.id] != context(market) {rows.removeValue(forKey:market.id)}
  guard rows[market.id] == nil,pending.insert(market.id).inserted else{return}
  contexts[market.id]=context(market)
  defer{checkedAt[market.id]=Date();pending.remove(market.id)}
  if market.kind == "prop",let line=market.line, let side=market.side?.lowercased(),["over","under"].contains(side) {
   let history=PlayerHistory();await history.load(market)
   guard !Task.isCancelled,contexts[market.id] == context(market) else{return}
   let games=Array(history.games.prefix(10))
   guard !games.isEmpty else {unavailable(market);return}
   let hits=games.filter{side == "under" ? $0.value<line:$0.value>line}.count
   let pushes=games.filter{$0.value==line}.count
   let average=games.map(\.value).reduce(0,+)/Double(games.count)
   rows[market.id]=ParlayFinding(headline:"\(hits) of \(games.count) cleared \(market.label)",detail:String(format:"Average: %.1f · Line: %.1f. ",average,line)+"\(pushes) pushes. Prior games compared with this quoted line, not each game’s historical line."+(games.count<5 ? " Small sample.":""),source:"\(market.sport == "MLB" ? "MLB":"ESPN") game logs · Last \(games.count) appearances · Historical frequency, not win probability")
   return
  }
  guard market.kind == "team",market.market == "h2h" else {unavailable(market);return}
  do {
   let path=sportsFeedPath(market.sport)
   let directory=try await json("https://site.api.espn.com/apis/site/v2/sports/\(path)/teams?limit=1000")
   guard !Task.isCancelled,contexts[market.id] == context(market) else{return}
   let sports=directory["sports"] as? [[String:Any]] ?? []
   let leagues=sports.first?["leagues"] as? [[String:Any]] ?? []
   let teams=leagues.first?["teams"] as? [[String:Any]] ?? []
   let matches=teams.compactMap{$0["team"] as? [String:Any]}.filter{($0["displayName"] as? String)?.lowercased()==market.name.lowercased()}
   guard matches.count==1,let id=matches.first?["id"] as? String else {unavailable(market);return}
   let schedule=try await json("https://site.api.espn.com/apis/site/v2/sports/\(path)/teams/\(id)/schedule")
   guard !Task.isCancelled,contexts[market.id] == context(market) else{return}
   let events=schedule["events"] as? [[String:Any]] ?? []
   let cutoff=min(Date(),market.commence.flatMap(sportsDate) ?? Date())
   var results:[(Date,Double,Double)]=[]
   for event in events {
    guard let raw=event["date"] as? String,let date=sportsDate(raw),date<cutoff,
     let competition=(event["competitions"] as? [[String:Any]])?.first,
     let status=competition["status"] as? [String:Any],let type=status["type"] as? [String:Any],type["completed"] as? Bool == true,
     let competitors=competition["competitors"] as? [[String:Any]],
     let own=competitors.first(where:{$0["id"] as? String == id}),let other=competitors.first(where:{$0["id"] as? String != id}),
     let ownScore=own["score"] as? [String:Any],let opponentScore=other["score"] as? [String:Any],
     let scored=ownScore["value"] as? Double,let allowed=opponentScore["value"] as? Double else {continue}
    results.append((date,scored,allowed))
   }
   results=Array(results.sorted{$0.0>$1.0}.prefix(10))
   guard !results.isEmpty else {unavailable(market);return}
   let wins=results.filter{$0.1>$0.2}.count,losses=results.filter{$0.1<$0.2}.count,draws=results.filter{$0.1==$0.2}.count
   let unit=market.sport == "MLB" ? "runs":(["Soccer","NHL"].contains(market.sport) ? "goals":"points")
   rows[market.id]=ParlayFinding(headline:"Recent form: \(wins)W–\(losses)L"+(draws>0 ? "–\(draws)D":""),detail:String(format:"%.1f %@ scored and %.1f allowed per game. ",results.map{$0.1}.reduce(0,+)/Double(results.count),unit,results.map{$0.2}.reduce(0,+)/Double(results.count))+"This describes team form; it does not prove this moneyline is good value."+(results.count<5 ? " Small sample.":""),source:"ESPN completed games · Last \(results.count) · Before this matchup")
  } catch {if !Task.isCancelled,contexts[market.id] == context(market) {unavailable(market)}}
 }
 private func unavailable(_ market:Market){rows[market.id]=ParlayFinding(headline:"Supporting evidence unavailable",detail:"This selection has a price, but no verified recent-stat analysis is connected. It is not a recommended pick.",source:"Sportsbook snapshot only",supported:false)}
 private func json(_ value:String) async throws->[String:Any] {
  var request=URLRequest(url:URL(string:value)!);request.timeoutInterval=15
  let(data,response)=try await URLSession.shared.data(for:request)
  guard (response as? HTTPURLResponse)?.statusCode==200 else{throw URLError(.badServerResponse)}
  return try JSONSerialization.jsonObject(with:data) as? [String:Any] ?? [:]
 }
}

func marketImpliedProbability(_ price:Double)->String {
 guard price.isFinite,abs(price)>=100 else{return "Unavailable"}
 return probability(implied(price))
}
private func parlayImpliedProbability(_ legs:[Market])->String {
 guard Set(legs.map(\.event)).count == legs.count else{return "Quote needed"}
 guard !legs.isEmpty,legs.allSatisfy({$0.price.isFinite && abs($0.price)>=100}) else{return "Unavailable"}
 return probability(legs.reduce(1.0){$0*implied($1.price)})
}

// Native motion inspired by terminal interfaces; no web runtime required.
struct TerminalReveal:View {
 let text:String
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 @State private var displayed=""
 var body:some View {
  Text(displayed.isEmpty ? String(repeating:" ",count:text.count):displayed)
   .font(.system(.subheadline,design:.monospaced,weight:.medium)).tracking(2)
   .accessibilityLabel(text)
   .task(id:reduceMotion) {
    guard !reduceMotion else {displayed=text;return}
    let letters=Array(text),noise=Array("01/<>_#")
    do {
     try await Task.sleep(for:.milliseconds(850))
     for tick in 0...(letters.count+6) {
      try Task.checkCancellation()
      displayed=String(letters.enumerated().map {index,char in
       char == " " || index < tick-3 ? char:noise[(tick+index)%noise.count]
      })
      try await Task.sleep(for:.milliseconds(45))
     }
     displayed=text
    } catch {}
   }
 }
}

struct AnimatedProbability:View {
 let value:Double?
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 @State private var shown:Double=0
 var body:some View {
  Text(value == nil ? "— %":probability(shown)).monospacedDigit()
   .contentTransition(.numericText())
   .accessibilityLabel(value.map{probability($0)} ?? "Not available")
   .task(id:value) {
    guard let value else {shown=0;return}
    guard !reduceMotion else {shown=value;return}
    shown=0
    do {
     for step in 1...24 {
      try await Task.sleep(for:.milliseconds(25));try Task.checkCancellation()
      let t=Double(step)/24
      shown=value*(1-pow(1-t,3))
     }
     shown=value
    } catch {}
   }
 }
}

// Native decorative square field, shared by every main tab.
struct BlinkingSquares:View {
 var accent:Color = .primary
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 @Environment(\.scenePhase) private var scenePhase
 var body:some View {
  TimelineView(.animation(minimumInterval:1.0/12.0,paused:reduceMotion || scenePhase != .active)) { timeline in
   Canvas { context,size in
    let time = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate
    let pitch:CGFloat = 30
    for row in 0...Int(size.height/pitch) {
     for column in 0...Int(size.width/pitch) {
      let seed = (row*73+column*37)%101
      let rect = CGRect(x:CGFloat(column)*pitch+3,y:CGFloat(row)*pitch+3,width:23,height:23)
      context.stroke(Path(rect),with:.color(accent.opacity(0.035)),lineWidth:0.5)
      if seed%4 == 0 {
       let wave = (sin(time*0.9+Double(seed)*0.7)+1)/2
       let opacity = reduceMotion ? 0.045 : 0.015+pow(wave,4)*0.10
       context.fill(Path(rect),with:.color(accent.opacity(opacity)))
      }
     }
    }
   }
  }.allowsHitTesting(false).accessibilityHidden(true)
 }
}
// Original native terrain preview inspired by the Landscape visual direction.
struct ModelLandscape:View {
 var accent:Color = brandAccent
 var showsBase:Bool = true
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 @Environment(\.scenePhase) private var scenePhase
 @Environment(\.colorScheme) private var scheme
 @State private var visible = false
 var body:some View {
  GeometryReader { geo in
   ZStack {
    if showsBase { navy }
    TimelineView(.animation(minimumInterval:1.0/30.0,paused:reduceMotion || !visible || scenePhase != .active)) { timeline in
     Canvas(rendersAsynchronously:true) { context,size in
      let time = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate*0.12
      let horizon=size.height*0.36
      for row in 0..<24 {
       let depth=Double(row)/23
       let perspective=pow(depth,1.7)
       var ridge=Path()
       for column in 0...48 {
        let x=Double(column)/48
        let world=x*7
        let z=depth*9+time
        let terrain=sin(world*1.4+z)*0.48+sin(world*3.1-z*0.7)*0.22+cos(world*0.7+z*1.5)*0.3
        let y=horizon+CGFloat(perspective)*(size.height-horizon+80)-CGFloat(terrain)*(16+CGFloat(perspective)*85)
        let point=CGPoint(x:CGFloat(x)*size.width,y:y)
        if column==0 {ridge.move(to:point)} else {ridge.addLine(to:point)}
       }
       var fill=ridge
       fill.addLine(to:CGPoint(x:size.width,y:size.height+160))
       fill.addLine(to:CGPoint(x:0,y:size.height+160));fill.closeSubpath()
       context.fill(fill,with:.color(navy))
       let scan=(sin(depth*14-time*2)+1)/2
       let strength=(scheme == .dark ? 0.18:0.12)+depth*0.18+pow(scan,8)*0.16
       context.stroke(ridge,with:.color(accent.opacity(strength)),lineWidth:0.7+depth*0.6)
      }
     }
    }
    if showsBase { LinearGradient(colors:[navy,navy.opacity(0.85),.clear],startPoint:.top,endPoint:.bottom) }
   }
  }.ignoresSafeArea().allowsHitTesting(false).accessibilityHidden(true)
  .onAppear { visible = true }.onDisappear { visible = false }
 }
}

struct ModelAurora:View {
 var body:some View { ModelLandscape() }
}

// Liquid Vortex motion adapted from zzzzshawn/matrix dotm-hex-10 for this app.
struct PLONNLoader:View {
 private let label:String?
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 @Environment(\.scenePhase) private var scenePhase
 init(_ label:String? = nil) {self.label=label}

 private func vortexRipple(_ value:Double,width:Double)->Double {
  let wrapped=value-floor(value)
  return max(0,1-min(wrapped,1-wrapped)/width)
 }
 var body:some View {
  HStack(spacing:10) {
   TimelineView(.animation(minimumInterval:1.0/60.0,paused:reduceMotion || scenePhase != .active)) { context in
    let phase=reduceMotion ? 0.14:context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy:1.85)/1.85
    Canvas { canvas,size in
     let counts=[3,4,5,4,3],pitch=size.width/5
     for row in 0..<5 {
      for col in 0..<counts[row] {
       let x=Double(col)-Double(counts[row]-1)/2
       let y=Double(row-2)*sqrt(3)/2
       let radius=sqrt(x*x+y*y),angle=phase*2*Double.pi
       let lens=max(0,1-abs(sin(angle)*1.15-x*0.88-y*0.16)/0.78)
       let front=vortexRipple(phase+x*0.12-y*0.045+radius*0.07,width:0.16)
       let back=vortexRipple(phase+0.34+x*0.09+y*0.035+radius*0.05,width:0.2)*0.34
       let compression=max(0,1-abs(cos(angle)*1.18-y*1.25)/1.1)*0.18
       let sheen=(0.5+0.5*sin(angle-radius*1.9))*(radius>1.35 ? 0.16:0.06)
       let core=radius<0.1 ? 0.34+sin(angle)*0.1:0
       var dot=canvas;dot.opacity=min(0.98,0.09+lens*0.72+front*0.38+back+compression+sheen+core)
       let rect=CGRect(x:size.width/2+x*pitch-2,y:size.height/2+y*pitch-2,width:4,height:4)
       dot.fill(Path(ellipseIn:rect),with:.foreground)
      }
     }
    }.frame(width:30,height:30)
   }.accessibilityHidden(true)
   if let label {Text(label).font(.system(.subheadline, design: .monospaced))}
  }.accessibilityElement(children:.ignore).accessibilityLabel(label ?? "Loading")
 }
}


private struct PLONNRefreshModifier:ViewModifier {
 let action:@Sendable () async -> Void
 @State private var refreshing=false
 func body(content:Content)->some View {
  content.refreshable {
   guard !refreshing else{return}
   refreshing=true
   defer {refreshing=false}
   await action()
  }.overlay(alignment:.top) {
   if refreshing {PLONNLoader("Refreshing").padding(10).background(.regularMaterial,in:RoundedRectangle(cornerRadius:12)).padding(.top,8).allowsHitTesting(false)}
  }
 }
}
extension View {
 func plonnRefreshable(action:@escaping @Sendable () async -> Void)->some View {
  modifier(PLONNRefreshModifier(action:action))
 }
}


private struct PLONNGlassButton: ViewModifier {
 let prominent: Bool
 func body(content: Content) -> some View {
  content.buttonStyle(PLONNGlassButtonStyle(prominent:prominent))
 }
}
/// Shared by app actions; system navigation and menus retain their native glass.
struct PLONNGlassButtonStyle: ButtonStyle {
 var prominent = false
 @Environment(\.isEnabled) private var isEnabled
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 func makeBody(configuration:Configuration) -> some View {
  configuration.label
   .fontWeight(prominent ? .semibold:nil)
   .foregroundStyle(configuration.role == .destructive ? Color.red:Color.primary)
   .padding(.horizontal,14).padding(.vertical,8).frame(minHeight:44)
   .plonnGlassSurface(selected:prominent,radius:18)
   .contentShape(RoundedRectangle(cornerRadius:18))
   .opacity(isEnabled ? 1:0.45)
   .scaleEffect(configuration.isPressed && !reduceMotion ? 0.98:1)
   .animation(reduceMotion ? nil:.easeOut(duration:0.16),value:configuration.isPressed)
 }
}
private struct PLONNGlassSurface: ViewModifier {
 let selected: Bool
 let radius: CGFloat
 let interactive: Bool
 @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
 @Environment(\.accessibilityReduceMotion) private var reduceMotion
 @Environment(\.colorSchemeContrast) private var contrast
 @ViewBuilder func body(content: Content) -> some View {
  if reduceTransparency || contrast == .increased {
   content.background(selected ? Color(red:0.21,green:0.16,blue:0.35):navySurface,in:RoundedRectangle(cornerRadius:radius))
    .overlay(RoundedRectangle(cornerRadius:radius).strokeBorder(selected ? brandAccent:Color.primary.opacity(0.35)).allowsHitTesting(false))
  } else if #available(iOS 26.0, *) {
   content.background(PLONNPalette.gradient.opacity(selected ? 0.06:0),in:RoundedRectangle(cornerRadius:radius))
    .glassEffect(.regular.interactive(interactive && !reduceMotion),in:.rect(cornerRadius:radius))
    .overlay(RoundedRectangle(cornerRadius:radius).strokeBorder(PLONNPalette.highlight.opacity(selected ? 0.34:0.10),lineWidth:1).allowsHitTesting(false))
  } else {
   content.background(.regularMaterial,in:RoundedRectangle(cornerRadius:radius))
    .background(PLONNPalette.gradient.opacity(selected ? 0.06:0),in:RoundedRectangle(cornerRadius:radius))
    .overlay(RoundedRectangle(cornerRadius:radius).strokeBorder(PLONNPalette.highlight.opacity(selected ? 0.34:0.10)).allowsHitTesting(false))
  }
 }
}
private struct PLONNGlassGroup: ViewModifier {
 @ViewBuilder func body(content: Content) -> some View {
  if #available(iOS 26.0, *) { GlassEffectContainer(spacing:8) { content } }
  else { content }
 }
}
extension View {
 func plonnGlassButton(prominent:Bool=false) -> some View {modifier(PLONNGlassButton(prominent:prominent))}
 func plonnGlassSurface(selected:Bool=false,radius:CGFloat=12,interactive:Bool=true) -> some View {modifier(PLONNGlassSurface(selected:selected,radius:radius,interactive:interactive))}
 func plonnGlassGroup() -> some View {modifier(PLONNGlassGroup())}
}
