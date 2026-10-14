import SwiftUI

/// Recorded research is populated independently of paid odds and model estimates.
struct AnalyticsView:View {
 @EnvironmentObject private var store:Store
 @EnvironmentObject private var account:AccountStore
 @Environment(\.scenePhase) private var scenePhase
 @Environment(\.dynamicTypeSize) private var dynamicTypeSize
 @State private var selectedSport:String
 @State private var query:String
 @State private var visible=false
 @State private var evaluatedAt=Date()
 @StateObject private var feed=AnalyticsFeedController()
 init(initialSport:String="All",initialQuery:String="") {
  _selectedSport=State(initialValue:["NFL","NBA","MLB","NHL"].contains(initialSport) ? initialSport:"NFL")
  _query=State(initialValue:initialQuery)
 }
 private var scopeID:String {
  let scope=store.scope(for:account)
  return (scope.owner?.uuidString ?? "guest")+"|"+scope.backend
 }
 private var searching:Bool {!query.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty}
 private var visibleCandidates:[AnalyticsFeedCandidate] {
  guard feed.scopeID == scopeID else{return []}
  return AnalyticsFeedCore.visibleCandidates(feed.candidates,sport:selectedSport,query:query,histories:feed.results,now:evaluatedAt)
 }
 var body:some View {
  NavigationStack {
   VStack(spacing:0) {
    leagueSelectors
    ScrollView {
    LazyVStack(alignment:.leading,spacing:12) {
     searchField
     VStack(alignment:.leading,spacing:4) {
      Text("Team & player trends").font(.system(.title2,design:.monospaced).bold()).accessibilityIdentifier("analytics-heading")
      Text(searching ? selectedSport+" statistics":selectedSport+" · Recorded prices").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
     }
     fixtureBanner
     feedStatus
     if !visibleCandidates.isEmpty {
      trendCarousel
      Text("Recent results & prices").font(.system(.headline,design:.monospaced)).padding(.top,4)
      resultsCarousel(subject:.team,title:"Team Props",identifier:"analytics-team-props")
      resultsCarousel(subject:.player,title:"Player Props",identifier:"analytics-player-props")
     }
     if !feed.discovering,!feed.candidates.isEmpty,visibleCandidates.isEmpty {
      if searching {
       Text("No \(selectedSport) teams or players match this search.").font(.system(.subheadline,design:.monospaced)).foregroundStyle(.secondary)
      } else if feed.candidates.filter({$0.market.sport == selectedSport}).contains(where:{feed.results[$0.id] == nil}) {
       HStack{ProgressView();Text("Checking recorded statistics and cached lines…").font(.system(.caption,design:.monospaced))}
      } else {
       Text("No \(selectedSport) results with matching recorded prices right now. Search this league to explore its recorded statistics.").font(.system(.subheadline,design:.monospaced)).foregroundStyle(.secondary).accessibilityIdentifier("analytics-no-matching-odds")
      }
     }
     ForEach(feed.statusWarnings.filter{$0.hasPrefix(selectedSport+":")},id:\.self) {warning in
      Text(warning).font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("analytics-event-status-warning")
     }
     if !feed.candidates.isEmpty,!feed.warnings.isEmpty {
      Text("Some trends are unavailable or incomplete. Showing the verified results.").font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("analytics-partial-feed")
     }
    }.padding(16)
    }.refreshable{feed.retryDiscovery()}.scrollDismissesKeyboard(.interactively)
   }.background(LeagueBackdrop(league:selectedSport))
    .navigationTitle("Analytics").navigationBarTitleDisplayMode(.inline)
    .toolbar{ToolbarItem(placement:.principal){Brand()}}
    .toolbarBackground(.hidden,for:.navigationBar)
  }
  .task(id:scopeID) {
   feed.configure(scopeID:scopeID,snapshots:store.analyticsSnapshots(for:account),read:{[store,account] sport,phase,event in
    try await store.recordedAnalyticsSnapshot(sport:sport,phase:phase,event:event,account:account)
   })
   if visible,scenePhase == .active {feed.resume()}
  }
  .task(id:visible && scenePhase == .active) {
   guard visible,scenePhase == .active else{return}
   while !Task.isCancelled {
    evaluatedAt=Date();feed.refreshIfNeeded();feed.refreshStatusIfNeeded()
    do {try await Task.sleep(for:.seconds(1))} catch{return}
   }
  }
  .onDisappear{visible=false;feed.pause()}
  .onAppear{visible=true;if scenePhase == .active {feed.resume()}}
  .onChange(of:scenePhase){_,phase in if visible,phase == .active {feed.resume()} else {feed.pause()}}
 }
 private var carouselPages:[AnalyticsCarouselPage] {
  let groups=visibleCandidates.enumerated().map {entry -> [AnalyticsCarouselPage] in
   let candidate=entry.element
   let history=feed.results[candidate.id]
   let loading=feed.loadingIDs.contains(candidate.id)
   let insights=history.map{AnalyticsTrendInsights.make(candidate:candidate,history:$0)} ?? []
   if insights.isEmpty {return [AnalyticsCarouselPage(candidate:candidate,history:history,insight:nil,loading:loading)]}
   let rotation=entry.offset % insights.count
   let ordered=Array(insights.dropFirst(rotation))+Array(insights.prefix(rotation))
   return ordered.map{AnalyticsCarouselPage(candidate:candidate,history:history,insight:$0,loading:loading)}
  }
  var pages:[AnalyticsCarouselPage]=[]
  let depth=groups.map(\.count).max() ?? 0
  for index in 0..<depth {
   for group in groups where index<group.count {pages.append(group[index])}
  }
  return Array(pages.prefix(36))
 }
 private var trendCarousel:some View {
  VStack(alignment:.leading,spacing:7) {
   ScrollView(.horizontal,showsIndicators:false) {
    LazyHStack(alignment:.top,spacing:12) {
     ForEach(carouselPages) {page in
      AnalyticsInsightCard(page:page)
       .containerRelativeFrame(.horizontal){width,_ in max(250,width-52)}
     }
    }.scrollTargetLayout()
   }.scrollTargetBehavior(.viewAligned).accessibilityIdentifier("analytics-trend-carousel").id("trends|"+selectedSport)
   HStack{Text("Swipe through recorded trends");Image(systemName:"arrow.right")}.font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
  }
 }
 private func resultsCarousel(subject:AnalyticsFeedSubject,title:String,identifier:String)->some View {
  let candidates=visibleCandidates.filter{$0.subject == subject}
  return VStack(alignment:.leading,spacing:8) {
   HStack(alignment:.firstTextBaseline) {
    Text(title).font(.system(.headline,design:.monospaced)).accessibilityAddTraits(.isHeader).accessibilityIdentifier(identifier+"-heading")
    Spacer(minLength:8)
    Text("\(candidates.count)").font(.system(.caption,design:.monospaced)).monospacedDigit().foregroundStyle(.secondary)
     .accessibilityLabel("\(candidates.count) results").accessibilityIdentifier(identifier+"-count")
   }
   if candidates.isEmpty {
    Text(searching ? "No \(subject.rawValue) results match this search.":"No \(subject.rawValue) results with matching recorded odds right now.")
     .font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary).accessibilityIdentifier(identifier+"-empty")
   } else {
    ScrollView(.horizontal,showsIndicators:false) {
     HStack(alignment:.top,spacing:12) {
      ForEach(candidates) {candidate in
       AnalyticsFeedCard(candidate:candidate,result:feed.results[candidate.id],loading:feed.loadingIDs.contains(candidate.id),retry:{feed.retry(candidate.id)})
        .containerRelativeFrame(.horizontal){width,_ in
         dynamicTypeSize.isAccessibilitySize || candidates.count == 1 ? width:min(460,max(width-24,width*0.9))
        }
        .fixedSize(horizontal:false,vertical:true)
        .id(scopeID+"|"+candidate.id)
      }
     }.scrollTargetLayout()
    }.scrollTargetBehavior(.viewAligned).accessibilityLabel(title).accessibilityIdentifier(identifier+"-carousel").id(identifier+"|"+selectedSport)
   }
  }.padding(.top,8)
 }
 private var leagueSelectors:some View {
  ScrollView(.horizontal,showsIndicators:false) {
   HStack(spacing:20) {
    ForEach(["NFL","NBA","MLB","NHL"],id:\.self) {league in
     Button {selectedSport=league;query=""} label: {
      VStack(spacing:5) {LeagueMark(league:league);Text(league).font(.system(.caption,design:.monospaced).weight(.semibold))}
       .foregroundStyle(selectedSport == league ? .primary:.secondary).frame(minWidth:48).frame(height:58)
       .overlay(alignment:.bottom){if selectedSport == league {Rectangle().fill(charcoalAccent).frame(height:2)}}
     }.buttonStyle(.plain).accessibilityLabel(league).accessibilityAddTraits(selectedSport == league ? .isSelected:[]).accessibilityIdentifier("analytics-league-"+league)
    }
    Spacer(minLength:0)
   }.padding(.horizontal,16)
  }.overlay(alignment:.bottom){Divider().padding(.horizontal,16)}.accessibilityIdentifier("analytics-leagues")
 }
 private var searchField:some View {
  HStack(spacing:9) {
   Image(systemName:"magnifyingglass").foregroundStyle(.secondary)
   TextField("Search "+selectedSport+" teams, players, stats",text:$query).textInputAutocapitalization(.never).autocorrectionDisabled().submitLabel(.search).accessibilityIdentifier("analytics-search")
   if !query.isEmpty {Button{query=""}label:{Image(systemName:"xmark.circle.fill").foregroundStyle(.secondary)}.accessibilityLabel("Clear search")}
  }.padding(12).background(navySurface,in:RoundedRectangle(cornerRadius:12))
 }
 @ViewBuilder private var fixtureBanner:some View {
  #if DEBUG && targetEnvironment(simulator)
  if ProcessInfo.processInfo.environment["PLONN_ANALYTICS_HISTORY_FIXTURE"] != nil {
   Text("Offline preview · fixture statistics and cached quotes").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary).accessibilityIdentifier("analytics-fixture-banner")
  }
  #endif
 }
 @ViewBuilder private var feedStatus:some View {
  if feed.discovering || feed.scopeID != scopeID {
   HStack{ProgressView();Text("Loading recorded trends…").font(.system(.caption,design:.monospaced))}.accessibilityIdentifier("analytics-feed-status")
  } else if feed.candidates.isEmpty {
   VStack(alignment:.leading,spacing:8) {
    Text("No verified \(selectedSport) results with recorded prices are available right now.").font(.system(.subheadline,design:.monospaced)).accessibilityIdentifier("analytics-feed-status")
    if let warning=feed.warnings.first(where:{$0.hasPrefix(selectedSport+":") || $0.hasPrefix(selectedSport+" ")}) {Text(warning).font(.caption).foregroundStyle(.secondary)}
    Button("Retry trends"){feed.retryDiscovery()}.accessibilityIdentifier("analytics-retry-feed").frame(minHeight:44)
   }
  }
 }
}

/// Tasks retain their slots through completion, including cancelled shared reads.
@MainActor private final class AnalyticsFeedController:ObservableObject {
 @Published private(set) var scopeID=""
 @Published private(set) var candidates:[AnalyticsFeedCandidate]=[]
 @Published private(set) var results:[String:AnalyticsHistoryResult]=[:]
 @Published private(set) var loadingIDs=Set<String>()
 @Published private(set) var warnings:[String]=[]
 @Published private(set) var discovering=false
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
 func configure(scopeID:String,snapshots:[Market],read:@escaping AnalyticsFeedService.RecordedReader) {
  self.recordedRead=read
  guard self.scopeID != scopeID else{return}
  pause();generation=UUID();self.scopeID=scopeID;self.snapshots=snapshots
  statusCheckedAt=nil;statusWarnings=[]
  candidates=[];results=[:];warnings=[];loadingIDs=[];discovered=false;discovering=false
 }
 func resume() {
  guard !scopeID.isEmpty else{return}
  active=true
  if !discovered {discover()} else if discoveredAt.map({Date().timeIntervalSince($0)>300}) == true {retryDiscovery()} else {schedule()}
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
  guard active,discovered,!discovering,discoveredAt.map({Date().timeIntervalSince($0)>300}) == true else{return}
  retryDiscovery()
 }
 func refreshStatusIfNeeded() {
  guard active,discovered,!discovering,statusRefresh == nil,
        candidates.contains(where:{$0.recordedSnapshot}),
        statusCheckedAt.map({Date().timeIntervalSince($0)>=30}) ?? true else{return}
  let token=generation,scope=scopeID,original=candidates
  statusCheckedAt=Date()
  statusRefresh=Task {
   let refreshed=await AnalyticsFeedService.shared.refreshStatuses(candidates:original)
   guard !Task.isCancelled,self.active,self.generation == token,self.scopeID == scope else{return}
   self.candidates=refreshed.candidates;self.statusWarnings=refreshed.warnings;self.statusRefresh=nil
  }
 }
 func retry(_ id:String) {
  guard active,!loadingIDs.contains(id) else{return}
  results[id]=nil;schedule()
 }
 private func discover() {
  guard active,discovery == nil else{return}
  let token=generation,ticket=UUID(),scope=scopeID,seeds=snapshots,read=recordedRead
  discoveryTicket=ticket;discovering=true
  discovery=Task {
   let result=await AnalyticsFeedService.shared.load(snapshots:seeds,recordedRead:read)
   guard !Task.isCancelled,self.active,self.generation == token,self.scopeID == scope,self.discoveryTicket == ticket else{return}
   self.candidates=Array(result.candidates.prefix(24));self.warnings=result.warnings
   self.results=[:];self.discoveredAt=Date();self.statusCheckedAt=nil;self.statusWarnings=[]
   self.discovered=true;self.discovering=false;self.discovery=nil;self.schedule()
  }
 }
 private func schedule() {
  guard active,discovered else{return}
  for candidate in candidates {
   guard tasks.count<2 else{break}
   guard results[candidate.id] == nil,!runningCandidates.values.contains(candidate.id) else{continue}
   let ticket=UUID(),token=generation,scope=scopeID,id=candidate.id
   loadingIDs.insert(id);runningCandidates[ticket]=id
   tasks[ticket]=Task {
    let result=await AnalyticsFeedService.shared.history(candidate:candidate,limit:10)
    if !Task.isCancelled,self.active,self.generation == token,self.scopeID == scope {
     self.results[id]=result
    }
    self.tasks[ticket]=nil;self.runningCandidates[ticket]=nil
    if self.generation == token {self.loadingIDs.remove(id)}
    self.schedule()
   }
  }
 }
}

private struct AnalyticsCarouselPage:Identifiable {
 let candidate:AnalyticsFeedCandidate
 let history:AnalyticsHistoryResult?
 let insight:AnalyticsTrendInsight?
 let loading:Bool
 var id:String {candidate.id+"|"+(insight?.id ?? "recorded-statistics")}
}

private struct AnalyticsInsightCard:View {
 let page:AnalyticsCarouselPage
 private var market:Market {page.candidate.market}
 private var symbol:String {
  guard let insight=page.insight else{return "chart.bar.xaxis"}
  switch insight.kind {
  case .recentBaseline:return "chart.line.uptrend.xyaxis"
  case .consistency:return "chart.bar.xaxis"
  case .homeAway:return "house"
  case .seasonBaseline:return "calendar"
  case .opponent:return "person.2"
  }
 }
 private var statTitle:String {
  switch market.market {
  case "h2h","spreads":return "Score difference"
  case "totals":return "Game total"
  case "team_totals":return "Team total"
  default:return propTitle(AnalyticsHistoryEngine.statisticKey(sport: market.sport, marketKey: market.market) ?? market.market)
  }
 }
 var body:some View {
  VStack(alignment:.leading,spacing:12) {
   HStack {
    Image(systemName:symbol)
    Text(page.insight?.title ?? "Recorded statistics").textCase(.uppercase).tracking(0.5)
   }.font(.system(.caption2,design:.monospaced).weight(.semibold)).foregroundStyle(.secondary)
   HStack(spacing:12) {
    AnalyticsSubjectArtwork(candidate:page.candidate,identifier:"analytics-insight-imagery-"+page.id)
    VStack(alignment:.leading,spacing:4) {
     Text(market.name).font(.system(.title3,design:.monospaced).bold()).fixedSize(horizontal:false,vertical:true)
     Text(market.sport+" · "+statTitle).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
    }
   }
   if let insight=page.insight {
    Text(insight.headline).font(.system(.title2,design:.monospaced).weight(.semibold)).accessibilityIdentifier("analytics-insight-headline-"+page.id)
    Text(insight.detail).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
    Spacer(minLength:0)
    Text("\(insight.sample) games in this comparison").font(.system(.caption2,design:.monospaced)).accessibilityIdentifier("analytics-insight-sample-"+page.id)
    Text(insight.source).font(.caption).foregroundStyle(.secondary)
    if let availability=insight.availability {Text(availability).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)}
   } else {fallback}
  }.padding(18).frame(minHeight:300,alignment:.topLeading)
   .background(navySurface,in:RoundedRectangle(cornerRadius:20))
   .overlay(alignment:.topLeading){RoundedRectangle(cornerRadius:20).stroke(charcoalAccent.opacity(0.12),lineWidth:1)}
   .accessibilityElement(children:.contain).accessibilityIdentifier("analytics-insight-"+page.id)
 }
 @ViewBuilder private var fallback:some View {
  if let history=page.history {
   if let stats=history.statistics,let mean=stats.mean {
    Text("Average "+String(format:"%.1f",mean)).font(.system(.title2,design:.monospaced).weight(.semibold))
    if let low=stats.minimum,let high=stats.maximum {Text("Lowest to highest "+String(format:"%g – %g",low,high)).font(.system(.subheadline,design:.monospaced))}
    Text("More recorded games are needed for a comparison.").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
    Spacer(minLength:0)
    Text("Last \(stats.n) recorded games · \(stats.valid) with usable statistics").font(.system(.caption2,design:.monospaced))
    Text(history.source).font(.caption).foregroundStyle(.secondary)
   } else {Text(history.availability).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)}
  } else {
   if page.loading {ProgressView()}
   Text(page.loading ? "Loading recorded context…":"Recorded context queued…").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
  }
 }
}

private struct AnalyticsFeedCard:View {
 let candidate:AnalyticsFeedCandidate
 let result:AnalyticsHistoryResult?
 let loading:Bool
 let retry:()->Void
 @State private var count=10
 private var market:Market {candidate.market}
 private var statsResult:AnalyticsHistoryResult? {result.map{AnalyticsHistoryEngine.statistics(market:market,observations:$0.observations,limit:count)}}
 private var stats:AnalyticsStatisticsSummary? {statsResult?.statistics}
 private var comparisons:[Market] {AnalyticsFeedCore.eligibleComparisons(candidate:candidate,history:result)}
 private var title:String {
  switch market.market {
  case "h2h","spreads":return "Score difference"
  case "totals":return "Game total"
  case "team_totals":return "Team total"
  default:return propTitle(market.market)
  }
 }
 var body:some View {
  Panel {
   VStack(alignment:.leading,spacing:9) {
    HStack(alignment:.top,spacing:12) {
     AnalyticsSubjectArtwork(candidate:candidate,identifier:"analytics-card-imagery-"+candidate.id)
     VStack(alignment:.leading,spacing:4) {
      Text(market.name).font(.system(.headline,design:.monospaced)).fixedSize(horizontal:false,vertical:true)
      Text(market.sport+" · "+title).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
     }.frame(maxWidth:.infinity,alignment:.leading)
    }
    if let event=candidate.verifiedSubject?.event {
     let status=event.displayLifecycle(now:Date())
     if status != .scheduled {Text(status.label).font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("analytics-event-status-"+candidate.id)}
     Text(event.away+" at "+event.home+" · "+analyticsFeedDate(event.start)+" · Eastern").font(.caption).foregroundStyle(.secondary)
     if event.sport == "NBA",event.seasonType == 1 {Text("Preseason matchup · History shows verified regular-season and postseason games.").font(.caption2).foregroundStyle(.secondary)}
    }
    if ["h2h","spreads"].contains(market.market) {Text("Score difference = this team's score minus its opponent's. Positive means a win; negative means a loss.").font(.caption).foregroundStyle(.secondary)}
    if let result {
     if let stats,stats.valid>0 {
      statistics(stats)
      quoteComparisons
      detail
     } else {
      Text(result.availability).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
      Button("Retry recorded games",action:retry).accessibilityIdentifier("analytics-retry-history-"+candidate.id).frame(minHeight:44)
     }
    } else {
     HStack{if loading {ProgressView()};Text(loading ? "Loading recorded games…":"Recorded games queued…").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)}
    }
   }
  }.accessibilityElement(children:.contain).accessibilityIdentifier("analytics-card-"+candidate.id)
 }
 @ViewBuilder private func statistics(_ value:AnalyticsStatisticsSummary)->some View {
  ViewThatFits(in:.horizontal) {
   HStack(alignment:.top,spacing:16){metrics(value)}
   VStack(alignment:.leading,spacing:8){metrics(value)}
  }
  Text("Last \(value.n) recorded games · \(value.valid) with usable statistics").font(.system(.caption2,design:.monospaced)).accessibilityIdentifier("analytics-sample-"+candidate.id)
  if value.valid<5 {Text("Small sample · fewer than 5 games with usable statistics.").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)}
  if let source=result?.source {Text(source).font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("analytics-provenance-"+candidate.id)}
 }
 @ViewBuilder private func metrics(_ value:AnalyticsStatisticsSummary)->some View {
  if let mean=value.mean {metric("Average",number:String(format:"%.1f",mean),id:"analytics-mean-"+candidate.id)}
  if let low=value.minimum,let high=value.maximum {metric("Low–high",number:String(format:"%g – %g",low,high),id:"analytics-range-"+candidate.id)}
  if let wins=value.teamWins,let losses=value.teamLosses,let ties=value.teamTies {metric("W–L–T",number:"\(wins)–\(losses)–\(ties)",id:"analytics-team-results-"+candidate.id)}
 }
 private func metric(_ label:String,number:String,id:String)->some View {
  VStack(alignment:.leading,spacing:2) {
   Text(label).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
   Text(number).font(.system(.subheadline,design:.monospaced).weight(.semibold)).monospacedDigit().fixedSize().accessibilityIdentifier(id)
  }
 }
 @ViewBuilder private var quoteComparisons:some View {
  VStack(alignment:.leading,spacing:7) {
   if comparisons.isEmpty {
    Text("Odds unavailable · recorded statistics remain available.").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
   } else {
    ForEach(Array(comparisons.prefix(3))) {quote in quoteRow(quote)}
    if comparisons.count>3 {Text("More prices in recent results").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)}
   }
  }.accessibilityIdentifier("analytics-odds-"+candidate.id)
 }
 private func quoteRow(_ quote:Market)->some View {
  let comparison=result.map{AnalyticsHistoryEngine.evaluate(market:quote,observations:$0.observations,limit:count)}
  let summary=comparison?.summary
  let side=quote.side?.capitalized ?? quote.name
  let line=quote.line.map{String(format:quote.market == "spreads" ? "%+g":"%g",$0)} ?? ""
  let contract=[side,line].filter{!$0.isEmpty}.joined(separator:" ")
  let priceText=odds(quote.price)
  return VStack(alignment:.leading,spacing:3) {
   ViewThatFits(in:.horizontal) {
    HStack(alignment:.firstTextBaseline) {
     Text(contract).font(.system(.caption,design:.monospaced).weight(.semibold)).fixedSize()
     Spacer(minLength:6)
     if let frequency=summary?.hitFrequency {Text(String(format:"%.0f%%",frequency*100)+" observed hits").font(.system(.subheadline,design:.monospaced).weight(.semibold)).monospacedDigit().fixedSize()}
    }
    VStack(alignment:.leading,spacing:3) {
     Text(contract).font(.system(.caption,design:.monospaced).weight(.semibold)).fixedSize(horizontal:false,vertical:true)
     if let frequency=summary?.hitFrequency {Text(String(format:"%.0f%%",frequency*100)+" observed hits").font(.system(.subheadline,design:.monospaced).weight(.semibold)).monospacedDigit().fixedSize(horizontal:false,vertical:true)}
    }
   }
   if let summary {
    Text("\(summary.hits) hits / \(summary.hits+summary.misses) decisions").font(.system(.caption2,design:.monospaced)).accessibilityIdentifier("analytics-denominator-"+quote.id)
    Text("Pushes \(summary.pushes) · Voids \(summary.voids) · Missing \(summary.missing)").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
   }
   Text(quote.book+" · "+priceText).font(.caption).foregroundStyle(.secondary)
   Text("Recorded · "+(quote.updated.flatMap(sportsDate).map {Store.oddsAsOf($0)} ?? "Price time unavailable")).font(.caption).foregroundStyle(.secondary)
  }.accessibilityElement(children:.contain).accessibilityIdentifier("analytics-quote-"+quote.id)
 }
 private var detail:some View {
  DisclosureGroup("Sample & recent results") {
   VStack(alignment:.leading,spacing:9) {
    Picker("Sample window",selection:$count){Text("Last 5 games").tag(5);Text("Last 10 games").tag(10)}.pickerStyle(.segmented)
    if let stats {Text("Valid \(stats.valid) · Voids \(stats.voids) · Missing \(stats.missing)").font(.system(.caption2,design:.monospaced))}
    Text("Hit rate counts games that met this line, divided by games above or below it. Ties at the line, voids and missing values are excluded. These comparisons use this recorded line; past sportsbook lines and settlements are unavailable.").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
    if let stats,stats.valid<5 {Text("Small sample · fewer than 5 games with usable statistics.").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)}
    if !(result?.warnings ?? []).isEmpty || !(statsResult?.warnings ?? []).isEmpty {
     Text("Some results could not be verified. This sample may be incomplete.").font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("analytics-partial-sample-"+candidate.id)
    }
    ForEach(statsResult?.statsRows ?? []) {row in observationRow(row)}
    ForEach(Array(comparisons.dropFirst(3))) {quote in quoteRow(quote)}
    Button("Retry recorded games",action:retry).accessibilityIdentifier("analytics-retry-history-"+candidate.id).frame(minHeight:44)
   }.padding(.top,8)
  }.font(.system(.caption,design:.monospaced)).accessibilityIdentifier("analytics-history-detail-"+candidate.id)
 }
 private func observationRow(_ row:AnalyticsObservation)->some View {
  let value=row.status == .completed ? row.value.map{String(format:"%g",$0)}:nil
  let status=value ?? (row.status == .void ? "Void marker":"Unavailable")
  return HStack {Text(analyticsFeedDate(row.date,compact:true)+" · "+row.opponent);Spacer();Text(status)}.font(.system(.caption2,design:.monospaced))
 }
}

private func analyticsFeedDate(_ date:Date,compact:Bool=false)->String {
 let formatter=DateFormatter();formatter.locale=Locale(identifier:"en_US_POSIX");formatter.timeZone=TimeZone(identifier:"America/New_York")
 formatter.dateFormat=compact ? "MMM d":"MMM d, yyyy"
 return formatter.string(from:date)
}

/// Bounded, shared image cache avoids redownloading a subject on each trend page.
@MainActor private final class AnalyticsArtworkCache {
 static let shared = AnalyticsArtworkCache()
 private let images = NSCache<NSURL, UIImage>()
 private var pending: [URL: Task<UIImage?, Never>] = [:]
 init() { images.countLimit = 64; images.totalCostLimit = 16 * 1024 * 1024 }
 func image(_ url: URL) async -> UIImage? {
  if let image = images.object(forKey:url as NSURL) { return image }
  if let task = pending[url] { return await task.value }
  let task = Task<UIImage?, Never> {
   var request = URLRequest(url:url,cachePolicy:.returnCacheDataElseLoad,timeoutInterval:15)
   request.httpShouldHandleCookies = false
   guard let (data,response) = try? await URLSession.shared.data(for:request),
    let http = response as? HTTPURLResponse, http.statusCode == 200,
    AnalyticsFeedCore.imageURL(http.url?.absoluteString) != nil,
    data.count <= 4 * 1024 * 1024 else { return nil }
   return UIImage(data:data)
  }
  pending[url] = task
  let image = await task.value
  pending.removeValue(forKey:url)
  if let image { images.setObject(image,forKey:url as NSURL,cost:Int(image.size.width * image.size.height * 4)) }
  return image
 }
}
private struct AnalyticsRemoteArtwork: View {
 let url: URL?
 let symbol: String
 let label: String
 let identifier: String
 let size: CGFloat
 @State private var loaded: UIImage?
 @State private var loadedURL: URL?
 private var currentImage: UIImage? { loadedURL == url ? loaded : nil }
 var body: some View {
  Group {
   if let image = currentImage { Image(uiImage:image).resizable().scaledToFit() }
   else { Image(systemName:symbol).font(.system(size:size * 0.42)).foregroundStyle(.secondary) }
  }.frame(width:size,height:size)
   .background(charcoalAccent.opacity(0.08),in:RoundedRectangle(cornerRadius:size * 0.22))
   .clipShape(RoundedRectangle(cornerRadius:size * 0.22))
   .accessibilityElement(children:.ignore)
   .accessibilityLabel(label)
   .accessibilityValue(currentImage == nil ? "Image unavailable" : "Image loaded")
   .accessibilityIdentifier(identifier)
   .task(id:url) {
    loaded = nil; loadedURL = nil
    guard let url else { return }
    let image = await AnalyticsArtworkCache.shared.image(url)
    guard !Task.isCancelled else { return }
    loadedURL = url; loaded = image
   }
 }
}
private struct AnalyticsSubjectArtwork: View {
 let candidate: AnalyticsFeedCandidate
 let identifier: String
 var body: some View {
  let proof = candidate.verifiedSubject
  let player = candidate.subject == .player
  AnalyticsRemoteArtwork(url:AnalyticsFeedCore.imageURL(proof?.imageURL),symbol:player ? "person.fill" : "sportscourt",label:candidate.market.name + (player ? " headshot" : " team logo"),identifier:identifier,size:56)
   .overlay(alignment:.bottomTrailing) {
    if player, let team = proof?.teamName {
     AnalyticsRemoteArtwork(url:AnalyticsFeedCore.imageURL(proof?.teamLogoURL),symbol:"sportscourt",label:team + " team logo",identifier:identifier + "-team",size:22)
      .background(navySurface,in:RoundedRectangle(cornerRadius:5)).offset(x:4,y:4)
    }
   }.padding(.trailing,4).padding(.bottom,4)
 }
}
