import SwiftUI
import Charts

/// A focused research workspace. Statistics remain available independently of odds access.
struct AnalyticsView: View {
 @EnvironmentObject private var store: Store
 @EnvironmentObject private var account: AccountStore
 @Environment(\.scenePhase) private var scenePhase
 @Environment(\.dynamicTypeSize) private var dynamicTypeSize
 @State private var selectedSport: String
 @State private var query: String
 @State private var subject = AnalyticsFeedSubject.player
 @State private var subjectID = ""
 @State private var showSearch = false
 @State private var statID = ""
 @State private var visible = false
 @State private var evaluatedAt = Date()
 @StateObject private var feed = AnalyticsFeedController()
 private var canReadOdds: Bool { account.hasSignedInAccount }
 init(initialSport: String = "All", initialQuery: String = "") {
  _selectedSport = State(initialValue: ["NFL", "NBA", "MLB", "NHL"].contains(initialSport) ? initialSport : "NFL")
  _query = State(initialValue: initialQuery)
 }
 private var scopeID: String {
  let scope = store.scope(for: account)
  return (scope.owner?.uuidString ?? "guest") + "|" + scope.backend + "|" + (canReadOdds ? "odds" : "statistics")
 }
 private var candidates: [AnalyticsFeedCandidate] {
  guard feed.scopeID == scopeID, feed.sport == selectedSport else { return [] }
  return AnalyticsFeedCore.visibleCandidates(feed.candidates, sport: selectedSport, query: query, histories: feed.results, now: evaluatedAt).filter { $0.subject == subject }
 }
 private struct ResearchSelection {
  let subjects: [AnalyticsFeedCandidate]
  let stats: [AnalyticsFeedCandidate]
  let candidate: AnalyticsFeedCandidate?
 }
 private var researchSelection: ResearchSelection {
  // Validate/filter the feed once per render, not again for every prop button.
  let visibleCandidates = candidates
  var seen = Set<String>()
  let subjects = visibleCandidates.filter { seen.insert(AnalyticsResearch.subjectID($0)).inserted }
  let activeSubject = subjects.first { AnalyticsResearch.subjectID($0) == subjectID } ?? subjects.first
  let selectedSubjectID = activeSubject.map(AnalyticsResearch.subjectID)
  let stats = visibleCandidates.filter { AnalyticsResearch.subjectID($0) == selectedSubjectID }
  return ResearchSelection(subjects:subjects,stats:stats,candidate:stats.first { $0.id == statID } ?? stats.first)
 }
 var body: some View {
  let selection = researchSelection
  let candidate = selection.candidate
  return NavigationStack {
   VStack(spacing: 0) {
    leagueSelectors
    ScrollView {
     VStack(alignment: .leading, spacing: 18) {
      researchControls(selection)
      fixtureBanner
      if let candidate {
       AnalyticsResearchDetail(candidate: candidate, history: feed.results[candidate.id],
        loading: feed.loadingIDs.contains(candidate.id), evaluatedAt: evaluatedAt,
        canReadOdds: canReadOdds, retry: { feed.retry(candidate.id) })
        .id(scopeID + "|" + candidate.id)
      } else { feedStatus }
      if !feed.statusWarnings.isEmpty {
       Text("Some game statuses couldn’t refresh. Unverified prices are hidden.").font(.caption).foregroundStyle(.secondary)
      }
      if !feed.candidates.isEmpty, !feed.warnings.isEmpty {
       Text("Coverage is incomplete. Showing the subjects and statistics we could verify.").font(.caption).foregroundStyle(.secondary)
      }
     }.padding(20)
    }.refreshable { feed.retryDiscovery() }.scrollDismissesKeyboard(.interactively)
   }.background(PLONNBrandBackdrop())
    .navigationTitle("Analytics").navigationBarTitleDisplayMode(.inline)
    .toolbar { ToolbarItem(placement: .principal) { Brand() } }
  }.tint(brandAccent).fontDesign(.default)
   .task(id: scopeID + "|" + selectedSport) { configureFeed() }
   .task(id: visible && scenePhase == .active) {
    guard visible, scenePhase == .active else { return }
    while !Task.isCancelled {
     evaluatedAt = Date(); feed.refreshIfNeeded(); feed.refreshStatusIfNeeded()
     do { try await Task.sleep(for: .seconds(5)) } catch { return }
    }
   }
   .onChange(of: candidate?.id, initial: true) { _, id in feed.prioritize(id) }
   .onDisappear { visible = false; feed.pause() }
   .onAppear { evaluatedAt = Date(); visible = true; if scenePhase == .active { feed.resume() } }
   .onChange(of: scenePhase) { _, phase in if visible, phase == .active { feed.resume() } else { feed.pause() } }
 }
 private func researchControls(_ selection: ResearchSelection) -> some View {
  VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 12) {
       Picker("Research", selection: $subject) {
        Text("Players").tag(AnalyticsFeedSubject.player)
        Text("Teams").tag(AnalyticsFeedSubject.team)
       }.pickerStyle(.segmented).accessibilityIdentifier("analytics-subject-filter")
       Button { showSearch.toggle(); if !showSearch { query = "" } } label: {
        Image(systemName: showSearch ? "xmark" : "magnifyingglass").font(.body.weight(.medium)).frame(width: 44, height: 44)
         .plonnGlassSurface(selected:showSearch,radius:22)
       }.buttonStyle(.plain).accessibilityLabel(showSearch ? "Close search" : "Search players, teams and stats")
      }.plonnGlassGroup()
      if showSearch || !query.isEmpty { searchField }
      if let candidate = selection.candidate {
       subjectSelector(candidate,subjects:selection.subjects)
       if selection.stats.count > 1 { statSelectors(selection.stats,selectedID:candidate.id) }
      }
  }
 }
 private func configureFeed() {
  let read: AnalyticsFeedService.RecordedReader?
  if canReadOdds {
   read = { [store, account] sport, phase, event in
    try await store.recordedAnalyticsSnapshot(sport: sport, phase: phase, event: event, account: account)
   }
  } else { read = nil }
  feed.configure(scopeID: scopeID, snapshots: store.analyticsSnapshots(for: account), read: read, sport: selectedSport)
  if visible, scenePhase == .active { feed.resume() }
 }
 private var leagueSelectors: some View {
  HStack(spacing: 8) {
   ForEach(["NFL", "NBA", "MLB", "NHL"], id: \.self) { league in
    Button { selectedSport = league; query = ""; subjectID = ""; statID = "" } label: {
     Text(league).font(.subheadline.weight(.semibold)).frame(maxWidth: .infinity).frame(minHeight: 44)
      .foregroundStyle(selectedSport == league ? Color.primary : Color.secondary)
      .plonnGlassSurface(selected:selectedSport == league,radius:22)
    }.buttonStyle(.plain).accessibilityAddTraits(selectedSport == league ? .isSelected : []).accessibilityIdentifier("analytics-league-" + league)
   }
  }.plonnGlassGroup().padding(.horizontal, 20).padding(.vertical, 10)
 }
 private var searchField: some View {
  HStack(spacing: 10) {
   Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
   TextField("Search " + selectedSport + " players, teams, stats", text: $query)
    .font(.subheadline).textInputAutocapitalization(.never).autocorrectionDisabled().submitLabel(.search).accessibilityIdentifier("analytics-search")
   if !query.isEmpty {
    Button { query = "" } label: { Image(systemName: "xmark.circle.fill").frame(width: 44, height: 44) }.accessibilityLabel("Clear search")
   }
  }.padding(.horizontal, 14).frame(minHeight: 48).plonnGlassSurface(radius:24,interactive:false)
 }
 private func subjectSelector(_ candidate: AnalyticsFeedCandidate,subjects:[AnalyticsFeedCandidate]) -> some View {
  ScrollViewReader { proxy in
   ScrollView(.horizontal, showsIndicators: false) {
    LazyHStack(spacing: 10) {
     ForEach(subjects) { item in
      let id = AnalyticsResearch.subjectID(item)
      let selected = id == AnalyticsResearch.subjectID(candidate)
      Button { subjectID = id; statID = "" } label: {
       HStack(spacing: 10) {
        AnalyticsSubjectArtwork(candidate: item, identifier: "analytics-subject-artwork-" + id)
        VStack(alignment: .leading, spacing: 4) {
         Text(item.market.name).font(.subheadline.bold()).foregroundStyle(.primary).lineLimit(2)
         if let team = item.verifiedSubject?.teamName {
          Text(team).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
         }
        }
        Spacer(minLength: 0)
        if selected { Image(systemName: "checkmark.circle.fill").foregroundStyle(brandAccent) }
       }.multilineTextAlignment(.leading).padding(12)
        .frame(width: dynamicTypeSize.isAccessibilitySize ? 300 : 252, alignment: .leading).frame(minHeight: 90)
        .plonnGlassSurface(selected:selected,radius:16)
        .contentShape(RoundedRectangle(cornerRadius: 16))
      }.buttonStyle(.plain).id(id)
       .accessibilityElement(children: .ignore)
       .accessibilityLabel(item.market.name + (item.verifiedSubject?.teamName.map { ", " + $0 } ?? ""))
       .accessibilityAddTraits(selected ? .isSelected : [])
       .accessibilityIdentifier("analytics-subject-" + id)
     }
    }.scrollTargetLayout().plonnGlassGroup()
   }.scrollTargetBehavior(.viewAligned)
    .frame(height: dynamicTypeSize.isAccessibilitySize ? 170 : 104)
    .accessibilityIdentifier("analytics-subject-carousel")
    .onChange(of: AnalyticsResearch.subjectID(candidate)) { _, id in proxy.scrollTo(id, anchor: .leading) }
  }
 }
 private func statSelectors(_ stats:[AnalyticsFeedCandidate],selectedID:String) -> some View {
  ScrollView(.horizontal, showsIndicators: false) {
   HStack(spacing: 8) {
    ForEach(stats) { item in
     Button { statID = item.id } label: {
      Text(AnalyticsResearch.statLabel(item.market)).font(.caption.weight(.semibold)).padding(.horizontal, 14).frame(minHeight: 44)
       .foregroundStyle(selectedID == item.id ? Color.primary : .secondary)
       .plonnGlassSurface(selected:selectedID == item.id,radius:22)
     }.buttonStyle(.plain).accessibilityLabel(AnalyticsCardCopy.title(item.market))
      .accessibilityAddTraits(selectedID == item.id ? .isSelected : [])
    }
   }.plonnGlassGroup()
  }.accessibilityIdentifier("analytics-stat-selector")
 }
 @ViewBuilder private var fixtureBanner: some View {
  #if DEBUG && targetEnvironment(simulator)
  if ProcessInfo.processInfo.environment["PLONN_ANALYTICS_HISTORY_FIXTURE"] != nil {
   Text("Offline preview · Fixture statistics and prices").font(.caption).foregroundStyle(.secondary)
  }
  #endif
 }
 @ViewBuilder private var feedStatus: some View {
  if feed.discovering || feed.scopeID != scopeID || feed.sport != selectedSport {
   VStack(alignment: .leading, spacing: 16) {
    HStack { ProgressView().tint(brandAccent); Text("Finding \(selectedSport) research…").font(.subheadline) }
    RoundedRectangle(cornerRadius: 16).fill(Color.white.opacity(0.04)).frame(height: 220)
   }.accessibilityIdentifier("analytics-feed-status")
  } else {
   VStack(alignment: .leading, spacing: 12) {
    Label(query.isEmpty ? "No verified \(subject == .player ? "player" : "team") statistics right now" : "No results in this view", systemImage: "chart.bar.xaxis").font(.headline)
    Text(query.isEmpty ? "Try another sport or refresh the available games." : "Try another name, or switch between Players and Teams.").font(.subheadline).foregroundStyle(.secondary)
    Button("Refresh research") { feed.retryDiscovery() }.frame(minHeight: 44)
   }.padding(18).researchSurface()
  }
 }
}

private struct AnalyticsResearchDetail: View {
 let candidate: AnalyticsFeedCandidate
 let history: AnalyticsHistoryResult?
 let loading: Bool
 let evaluatedAt: Date
 let canReadOdds: Bool
 let retry: () -> Void
 @EnvironmentObject private var account: AccountStore
 @EnvironmentObject private var builderStore: Store
 @AppStorage("preferredSportsbook") private var preferredBook = "Ask each time"
 @State private var builderNotice = ""
 @State private var recordedQuotes: [Market]?
 @State private var checkingQuotes = false
 @State private var recordedScope = ""
 @State private var window = AnalyticsResearchWindow.last10
 @State private var quoteID = ""
 @State private var inspectedID: String?
 @State private var expandedChart = false
 @State private var researchPage: Int? = 0
 @State private var carouselHeights: [Int: CGFloat] = [:]
 private var market: Market { candidate.market }
 private var quoteScope: String {
  let scope = builderStore.scope(for: account)
  return (scope.owner?.uuidString ?? "guest") + "|" + scope.backend + "|" + candidate.id + "|" + (candidate.verifiedSubject?.event?.providerID ?? "")
 }
 private var pricedCandidate: AnalyticsFeedCandidate {
  guard recordedScope == quoteScope, let recordedQuotes else { return candidate }
  return AnalyticsFeedCandidate(id: candidate.id, subject: candidate.subject, market: candidate.market,
   quotes: recordedQuotes, verifiedSubject: candidate.verifiedSubject, recordedSnapshot: true, recordedArchive: candidate.recordedArchive)
 }
 private var quotes: [Market] { canReadOdds ? AnalyticsFeedCore.cardOdds(candidate: pricedCandidate, history: history, now: evaluatedAt) : [] }
 private var quote: Market? {
  quotes.first { $0.id == quoteID } ?? quotes.sorted { left, right in
   if (left.book == preferredBook) != (right.book == preferredBook) { return left.book == preferredBook }
   let a = left.updated.flatMap(sportsDate) ?? .distantPast, b = right.updated.flatMap(sportsDate) ?? .distantPast
   if a != b { return a > b }
   return (left.book, left.id) < (right.book, right.id)
  }.first
 }
 private var sample: AnalyticsResearchSample? {
  history.map { AnalyticsResearch.sample(candidate: candidate, history: $0, window: window, quote: quote, now: evaluatedAt) }
 }
 var body: some View {
  VStack(alignment: .leading, spacing: 18) {
   if let event = candidate.verifiedSubject?.event {
    HStack(alignment: .top, spacing: 8) {
     Image(systemName: "calendar").foregroundStyle(brandAccent)
     VStack(alignment: .leading, spacing: 3) {
      Text(event.away + " at " + event.home).font(.caption.weight(.medium))
      Text(event.start, format: .dateTime.month(.abbreviated).day().hour().minute()).font(.caption2).foregroundStyle(.secondary)
      if event.displayLifecycle(now: evaluatedAt) != .scheduled { Text(event.displayLifecycle(now: evaluatedAt).label).font(.caption2).foregroundStyle(.secondary) }
      if event.seasonType == 1 { Text("Preseason matchup · Historical regular-season & playoff results").font(.caption2).foregroundStyle(.secondary) }
     }
    }
   }
   oddsPanel
   researchCarousel
   NavigationLink { UpcomingGamesView() } label: {
    HStack {
     Image(systemName: "sparkles")
     Text("Model an upcoming game").font(.subheadline.bold())
     Spacer(); Image(systemName: "arrow.up.right")
    }.padding(16).plonnGlassSurface(radius:18)
   }.buttonStyle(.plain)
  }.task(id: quoteScope) { await loadSelectedQuotes() }
 }
 @MainActor private func loadSelectedQuotes() async {
  let scope = quoteScope
  if recordedScope == scope, recordedQuotes != nil { return }
  recordedQuotes = nil; recordedScope = scope
  guard canReadOdds else { return }
  #if DEBUG && targetEnvironment(simulator)
  if ProcessInfo.processInfo.environment["PLONN_ANALYTICS_MARKETS_FIXTURE"] != nil { return }
  #endif
  checkingQuotes = true
  defer { if quoteScope == scope { checkingQuotes = false } }
  do {
   let selected = try await AnalyticsFeedService.recordedSelection(candidate, read: { [builderStore, account] sport, phase, event in
    try await builderStore.recordedAnalyticsSnapshot(sport: sport, phase: phase, event: event, account: account)
   })
   try Task.checkCancellation()
   guard quoteScope == scope, canReadOdds else { return }
   recordedQuotes = selected?.quotes
  } catch { /* Existing verified quotes and statistics remain available. */ }
 }
 private var researchPages: [(Int, String)] {
  var pages = [(0, "Recent form")]
  if candidate.subject == .player { pages.append((1, "Matchup")) }
  if sample?.statistics.statistics?.valid ?? 0 > 0 { pages.append((2, "Context")) }
  return pages
 }
 private var historicalDeviation: Double? {
  guard let observations = sample?.statistics.statsRows else { return nil }
  return AnalyticsResearch.standardDeviation(observations.filter { $0.status == .completed }.compactMap(\.value))
 }
 private var researchCarousel: some View {
  VStack(alignment: .leading, spacing: 10) {
   HStack(spacing: 8) {
    ForEach(researchPages, id: \.0) { page in
     Button { researchPage = page.0 } label: {
      Text(page.1).font(.caption.weight(.semibold)).frame(maxWidth: .infinity).frame(minHeight: 44)
       .foregroundStyle(researchPage == page.0 ? Color.primary : .secondary)
       .plonnGlassSurface(selected:researchPage == page.0,radius:22)
     }.buttonStyle(.plain).accessibilityAddTraits(researchPage == page.0 ? .isSelected : [])
    }
   }.plonnGlassGroup()
   ScrollView(.horizontal, showsIndicators: false) {
    HStack(alignment: .top, spacing: 14) {
     carouselPage(0) { performancePanel }
     if candidate.subject == .player { carouselPage(1) { PlayerMatchupCard(candidate: candidate, quote: quote) } }
     if let history, let sample, sample.statistics.statistics?.valid ?? 0 > 0 {
      carouselPage(2) { contextPanel(history, sample: sample) }
     }
    }.scrollTargetLayout()
   }.scrollTargetBehavior(.viewAligned).scrollPosition(id: $researchPage)
    .frame(height: carouselHeights.values.max() ?? 420)
    .onPreferenceChange(AnalyticsCarouselHeight.self) { heights in carouselHeights.merge(heights) { _, new in new } }
    .accessibilityIdentifier("analytics-research-carousel")
    .onChange(of: researchPages.map(\.0)) { _, pages in if !pages.contains(researchPage ?? 0) { researchPage = pages.first } }
   HStack(spacing: 7) {
    Spacer()
    ForEach(researchPages, id: \.0) { page in
     Button { researchPage = page.0 } label: {
      Capsule().fill(researchPage == page.0 ? brandAccent : Color.secondary.opacity(0.4)).frame(width: researchPage == page.0 ? 18 : 6, height: 6).frame(width:44,height:44).plonnGlassSurface(radius:22)
     }.buttonStyle(.plain).accessibilityLabel("Show " + page.1).accessibilityAddTraits(researchPage == page.0 ? .isSelected : [])
    }
    Spacer()
   }.plonnGlassGroup().accessibilityLabel("Research pages")
  }
 }
 private func carouselPage<Content: View>(_ index: Int, @ViewBuilder content: () -> Content) -> some View {
  content().containerRelativeFrame(.horizontal).fixedSize(horizontal: false, vertical: true)
   .background(GeometryReader { proxy in Color.clear.preference(key: AnalyticsCarouselHeight.self, value: [index: proxy.size.height]) })
   .id(index)
 }
 @ViewBuilder private var oddsPanel: some View {
  if let quote {
   Menu {
    ForEach(quotes) { item in
     Button { quoteID = item.id; inspectedID = nil } label: {
      Text(AnalyticsFeedCore.cardOddsContract(item) + " · " + odds(item.price) + " · " + item.book)
     }
    }
   } label: {
    HStack(alignment: .center) {
     VStack(alignment: .leading, spacing: 5) {
      Text(AnalyticsFeedCore.cardOddsContract(quote)).font(.headline).foregroundStyle(.primary).multilineTextAlignment(.leading)
      Text(quote.book + " · " + (quote.updated.flatMap(sportsDate).map { Store.oddsAsOf($0) } ?? "Time unavailable"))
       .font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.leading)
     }
     Spacer(minLength: 8)
     Text(odds(quote.price)).font(.title2.bold()).monospacedDigit().foregroundStyle(brandAccent)
     if quotes.count > 1 { Image(systemName: "chevron.down").font(.caption).foregroundStyle(.secondary) }
    }.padding(16).plonnGlassSurface(radius:18)
   }.buttonStyle(.plain).accessibilityIdentifier("analytics-recorded-line")
   Button {
    if builderStore.legs.contains(where:{$0.id == quote.id}) {builderStore.legs.removeAll {$0.id == quote.id};builderNotice=""}
    else if let reason = BuilderLogic.rejection(for:quote,existing:builderStore.legs) {builderNotice=reason}
    else {builderStore.add(quote);builderNotice=""}
   } label: {
    Label(builderStore.legs.contains {$0.id == quote.id} ? "Added to Builder":"Add to Builder",systemImage:builderStore.legs.contains {$0.id == quote.id} ? "checkmark":"plus")
   }.plonnGlassButton(prominent:builderStore.legs.contains {$0.id == quote.id})
    .accessibilityHint(builderStore.legs.contains {$0.id == quote.id} ? "Remove this pick from Builder":"Add this pick to Builder")
    .accessibilityIdentifier("analytics-add-to-builder")
   if !builderNotice.isEmpty {Text(builderNotice).font(.caption).foregroundStyle(.secondary)}
  } else if !canReadOdds {
   Button { account.access.present(feature: .odds) } label: {
    HStack(spacing: 10) {
     Image(systemName: "lock.fill").foregroundStyle(brandAccent)
     VStack(alignment: .leading, spacing: 3) {
      Text("Sign in to see odds").font(.subheadline.bold()).foregroundStyle(.primary)
     }
     Spacer(minLength: 4); Image(systemName: "chevron.right").font(.caption).foregroundStyle(.secondary)
    }.padding(14).plonnGlassSurface(radius:18)
   }.buttonStyle(.plain).accessibilityIdentifier("analytics-sign-in")
  } else {
   HStack(spacing: 8) {
    if checkingQuotes { ProgressView(); Text("Checking recorded lines…") }
    else { Image(systemName: "clock"); Text("No verified price for this stat right now.") }
   }.font(.caption).foregroundStyle(.secondary)
  }
 }
 private var performancePanel: some View {
  VStack(alignment: .leading, spacing: 16) {
   HStack(spacing: 8) {
    HStack(spacing: 4) {
     ForEach(AnalyticsResearchWindow.allCases) { choice in
      Button { window = choice; inspectedID = nil } label: {
       Text(choice.rawValue).font(.caption.weight(.semibold)).frame(maxWidth: .infinity).frame(minHeight: 44)
        .foregroundStyle(window == choice ? Color.primary : .secondary)
        .plonnGlassSurface(selected:window == choice,radius:10)
      }.buttonStyle(.plain).accessibilityLabel(choice.accessibilityLabel).accessibilityAddTraits(window == choice ? .isSelected : [])
     }
    }.plonnGlassGroup().padding(4)
    Button { expandedChart = true } label: {
     Image(systemName: "arrow.up.left.and.arrow.down.right").font(.caption.weight(.medium)).frame(width: 44, height: 44)
    }.accessibilityLabel("Expand " + AnalyticsCardCopy.title(market) + " chart")
   }.sheet(isPresented: $expandedChart) { expandedChartView }
   if let sample {
    if let stats = sample.statistics.statistics, stats.valid > 0 {
     summaryMetrics(stats, comparison: sample.comparison?.summary)
     if let deviation = historicalDeviation {
      Text("Historical variation ±" + AnalyticsCardCopy.number(deviation) + " " + AnalyticsCardCopy.perGameUnit(market).replacingOccurrences(of: "/GM", with: ""))
       .font(.caption.weight(.medium)).foregroundStyle(.secondary)
       .accessibilityLabel("Historical standard deviation " + AnalyticsCardCopy.number(deviation) + ". Not a future range.")
     }
     AnalyticsResultsChart(market: market, sample: sample, quote: quote, inspectedID: $inspectedID, compact: true)
     if stats.valid < stats.n { Text("\(stats.valid) of \(stats.n) games available").font(.caption2).foregroundStyle(.secondary) }
     if let comparison = sample.comparison?.summary, let quote, let exclusions = AnalyticsCardCopy.exclusions(comparison, market: quote) {
      Text(exclusions).font(.caption2).foregroundStyle(.secondary)
     }
     if stats.valid < 5 { Text("Small sample · Only \(stats.valid) usable games.").font(.caption).foregroundStyle(.orange) }
    } else {
     Text(sample.note ?? history?.availability ?? "No recorded results available.").font(.subheadline).foregroundStyle(.secondary).frame(maxWidth: .infinity, minHeight: 140, alignment: .leading)
     Button("Retry game history", action: retry).frame(minHeight: 44)
    }
   } else {
    VStack(alignment: .leading, spacing: 16) {
     HStack { ProgressView().tint(brandAccent); Text(loading ? "Loading game history…" : "Game history is up next…").font(.subheadline).foregroundStyle(.secondary) }
     HStack(alignment: .bottom, spacing: 10) {
      ForEach(0..<10) { index in RoundedRectangle(cornerRadius: 4).fill(Color.white.opacity(0.045)).frame(height: CGFloat(35 + (index * 37) % 90)) }
     }.frame(height: 150)
    }.accessibilityElement(children: .combine).accessibilityLabel("Loading game history")
   }
  }.padding(16).researchSurface().accessibilityIdentifier("analytics-performance")
 }
 private var expandedChartView: some View {
  NavigationStack {
   ScrollView {
    VStack(alignment: .leading, spacing: 20) {
     HStack(spacing: 12) {
      AnalyticsSubjectArtwork(candidate: candidate, identifier: "analytics-expanded-artwork")
      VStack(alignment: .leading, spacing: 4) {
       Text(market.name).font(.title2.bold())
       Text(AnalyticsCardCopy.title(market)).font(.subheadline).foregroundStyle(.secondary)
      }
     }
     oddsPanel
     Picker("Games", selection: $window) {
      ForEach(AnalyticsResearchWindow.allCases) { item in Text(item.rawValue).tag(item).accessibilityLabel(item.accessibilityLabel) }
     }.pickerStyle(.segmented)
     if let sample, let stats = sample.statistics.statistics, stats.valid > 0 {
      summaryMetrics(stats, comparison: sample.comparison?.summary)
      if let deviation = historicalDeviation {
       Text("Historical standard deviation: " + AnalyticsCardCopy.number(deviation) + ". This measures variation in the selected recorded games; it is not a future range.")
        .font(.caption).foregroundStyle(.secondary)
      }
      AnalyticsResultsChart(market: market, sample: sample, quote: quote, inspectedID: $inspectedID)
      Text(sample.label).font(.subheadline.weight(.medium))
      Text("\(stats.valid) of \(stats.n) recorded games with usable stats").font(.caption).foregroundStyle(.secondary)
      if let comparison = sample.comparison?.summary, let quote, let exclusions = AnalyticsCardCopy.exclusions(comparison, market: quote) { Text(exclusions).font(.caption).foregroundStyle(.secondary) }
      if quote != nil { Text("Historical results against this line, not a forecast.").font(.caption).foregroundStyle(.secondary) }
     } else { Text(sample?.note ?? "Game history is unavailable.").foregroundStyle(.secondary) }
    }.padding(20)
   }.background(navy).navigationTitle("Game history").navigationBarTitleDisplayMode(.inline)
    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { expandedChart = false } } }
  }.tint(brandAccent)
 }
 private func summaryMetrics(_ stats: AnalyticsStatisticsSummary, comparison: AnalyticsHistorySummary?) -> some View {
  ViewThatFits(in: .horizontal) {
   HStack(alignment: .top, spacing: 24) { averageMetric(stats); Spacer(minLength: 0); frequencyMetric(stats, comparison: comparison) }
   VStack(alignment: .leading, spacing: 16) { averageMetric(stats); frequencyMetric(stats, comparison: comparison) }
  }
 }
 private func averageMetric(_ stats: AnalyticsStatisticsSummary) -> some View {
  VStack(alignment: .leading, spacing: 4) {
   Text(AnalyticsCardCopy.isScoringMargin(market) ? "Avg. win / loss margin" : "Average").font(.caption).foregroundStyle(.secondary)
   Text(stats.mean.map { (AnalyticsCardCopy.isScoringMargin(market) && $0 > 0 ? "+" : "") + AnalyticsCardCopy.number($0) } ?? "—")
    .font(.system(size: 34, weight: .bold, design: .rounded)).monospacedDigit()
   Text(AnalyticsCardCopy.perGameUnit(market)).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
  }.accessibilityElement(children: .ignore).accessibilityLabel(stats.mean.map { AnalyticsCardCopy.average($0, market: market, abbreviated: false) } ?? "Average unavailable")
 }
 @ViewBuilder private func frequencyMetric(_ stats: AnalyticsStatisticsSummary, comparison: AnalyticsHistorySummary?) -> some View {
  if let comparison, let frequency = comparison.hitFrequency {
   VStack(alignment: .leading, spacing: 4) {
    Text("Met this line").font(.caption).foregroundStyle(.secondary)
    Text("\(Int((frequency * 100).rounded()))%").font(.system(size: 34, weight: .bold, design: .rounded)).monospacedDigit().foregroundStyle(brandAccent)
    Text("\(comparison.hits) of \(comparison.hits + comparison.misses) games").font(.caption2).foregroundStyle(.secondary)
   }
  } else if let wins = stats.teamWins, let losses = stats.teamLosses {
   VStack(alignment: .leading, spacing: 4) {
    Text("Team record").font(.caption).foregroundStyle(.secondary)
    Text("\(wins)–\(losses)").font(.system(size: 34, weight: .bold, design: .rounded)).monospacedDigit()
    Text("W–L" + ((stats.teamTies ?? 0) > 0 ? " · \(stats.teamTies!) ties" : "")).font(.caption2).foregroundStyle(.secondary)
   }
  } else if let high = stats.maximum {
   VStack(alignment: .leading, spacing: 4) {
    Text("Best in sample").font(.caption).foregroundStyle(.secondary)
    Text(AnalyticsCardCopy.number(high)).font(.system(size: 34, weight: .bold, design: .rounded)).monospacedDigit()
    Text(AnalyticsResearch.statLabel(market)).font(.caption2).foregroundStyle(.secondary)
   }
  }
 }
 private func contextPanel(_ history: AnalyticsHistoryResult, sample: AnalyticsResearchSample) -> some View {
  VStack(alignment: .leading, spacing: 16) {
   let insights = AnalyticsTrendInsights.make(candidate: candidate, history: history, now: evaluatedAt)
   if !insights.isEmpty {
    ForEach(Array(insights.prefix(2))) { insight in
     VStack(alignment: .leading, spacing: 5) {
      Text(insight.title).font(.caption.weight(.medium)).foregroundStyle(brandAccent)
      Text(insight.headline).font(.subheadline.bold())
      Text(insight.detail).font(.caption).foregroundStyle(.secondary)
     }
    }
   }
   DisclosureGroup("Game log") {
    VStack(alignment: .leading, spacing: 12) {
     ForEach(sample.statistics.statsRows) { row in
      HStack(alignment: .top) {
       VStack(alignment: .leading, spacing: 3) {
        Text(analyticsFeedDate(row.date, compact: true)).font(.caption)
        Text(row.opponent).font(.caption2).foregroundStyle(.secondary)
       }
       Spacer(minLength: 8)
       Text(row.status == .completed ? row.value.map { AnalyticsCardCopy.result($0, market: market) } ?? "Unavailable" : "Unavailable")
        .font(.caption.weight(.medium)).monospacedDigit().multilineTextAlignment(.trailing)
      }
     }
     Text("Source: " + history.source).font(.caption.weight(.medium))
     Text("Available appearances only. Missing games or DNP coverage may be incomplete.").font(.caption2).foregroundStyle(.secondary)
     if !history.warnings.isEmpty || !sample.statistics.warnings.isEmpty { Text("Some records could not be verified.").font(.caption2).foregroundStyle(.secondary) }
     Button("Refresh game history", action: retry).frame(minHeight: 44)
    }.padding(.top, 12)
   }.font(.subheadline)
  }.padding(16).researchSurface()
 }
}

private struct AnalyticsCarouselHeight: PreferenceKey {
 static var defaultValue: [Int: CGFloat] = [:]
 static func reduce(value: inout [Int: CGFloat], nextValue: () -> [Int: CGFloat]) { value.merge(nextValue()) { _, new in new } }
}

private struct AnalyticsResultsChart: View {
 let market: Market
 let sample: AnalyticsResearchSample
 let quote: Market?
 @Binding var inspectedID: String?
 var compact = false
 private var rows: [AnalyticsObservation] { Array(sample.statistics.statsRows.prefix(20).reversed()) }
 private var trend: [AnalyticsResearch.TrendPoint] { AnalyticsResearch.trendLine(rows) }
 private var threshold: Double? { sample.comparison?.summary != nil ? quote.flatMap(AnalyticsResearch.chartLine) : nil }
 private func outcome(_ row: AnalyticsObservation) -> AnalyticsOutcome? { sample.comparison?.rows.first { $0.id == row.id }?.outcome }
 private func color(_ row: AnalyticsObservation) -> Color {
  switch outcome(row) { case .hit: return brandAccent; case .miss: return Color(white: 0.42); case .push: return .orange; default: return brandAccent.opacity(0.75) }
 }
 private func barStyle(_ row: AnalyticsObservation) -> AnyShapeStyle {
  if outcome(row) == .miss || outcome(row) == .push { return AnyShapeStyle(color(row)) }
  return AnyShapeStyle(PLONNPalette.highlight)
 }
 private var domain: ClosedRange<Double> {
  let values = rows.filter { $0.status == .completed }.compactMap(\.value) + trend.map(\.value) + [threshold ?? 0, 0]
  let low = values.min() ?? 0, high = values.max() ?? 1, padding = max((high - low) * 0.18, 1)
  return (low < 0 ? low - padding : 0)...(high + padding)
 }
 var body: some View {
  VStack(alignment: .leading, spacing: 10) {
   chart.frame(height: compact ? 92 : 190).accessibilityIdentifier("analytics-results-chart")
   if !trend.isEmpty {
    HStack(spacing: 5) { Capsule().fill(Color.white.opacity(0.8)).frame(width: 16, height: 2); Text("Trend").font(.system(size: 10)) }.foregroundStyle(.secondary)
     .accessibilityLabel("Historical trend across displayed games")
   }
   if let row = rows.first(where: { $0.id == inspectedID }) {
    Text(selectedGameLabel(row))
     .font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("analytics-chart-selection")
   } else if !compact { Text("Tap a bar for game details").font(.caption2).foregroundStyle(.secondary) }
   if !compact, threshold != nil {
    HStack(spacing: 14) { legend("Met line", color: brandAccent); legend("Missed", color: Color(white: 0.55)); if sample.comparison?.summary?.pushes ?? 0 > 0 { legend("On line", color: .orange) } }
   }
   if !compact, sample.statistics.statsRows.count > 20 { Text("Chart shows the latest 20 of \(sample.statistics.statsRows.count) games. Summary uses the full selected sample.").font(.caption2).foregroundStyle(.secondary) }
   if !compact, rows.contains(where: { $0.status != .completed || $0.value == nil }) { Text("Gaps indicate unavailable results; they aren’t counted as zero.").font(.caption2).foregroundStyle(.secondary) }
  }
 }
 private func selectedGameLabel(_ row: AnalyticsObservation) -> String {
  let result: String
  if row.status == .completed, let value = row.value { result = AnalyticsCardCopy.result(value, market: market) }
  else { result = "Unavailable" }
  return [analyticsFeedDate(row.date, compact: true), row.opponent, result].joined(separator: " · ")
 }
 private var xDomain: ClosedRange<Double> { -0.6...(Double(max(rows.count - 1, 0)) + 0.6) }
 private var chart: some View {
  Chart { marks }
   .chartYScale(domain: domain).chartXScale(domain: xDomain)
   .chartYAxis { AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { _ in
    AxisGridLine().foregroundStyle(Color.white.opacity(0.06))
    AxisValueLabel().font(.system(size: 9))
   } }
   .chartXAxis { AxisMarks(values: Array(stride(from: 0, to: rows.count, by: max(1, rows.count / 5))).map(Double.init)) { value in
    AxisValueLabel { if let x = value.as(Double.self), rows.indices.contains(Int(x)) {
     Text(rows[Int(x)].date, format: .dateTime.month(.twoDigits).day(.twoDigits)).font(.system(size: 9))
    } }
   } }
   .chartOverlay { proxy in
    GeometryReader { geometry in
     Rectangle().fill(.clear).contentShape(Rectangle()).onTapGesture { point in
      if let frame = proxy.plotFrame {
       let x = point.x - geometry[frame].origin.x
       if let value: Double = proxy.value(atX: x), rows.indices.contains(Int(value.rounded())) { inspectedID = rows[Int(value.rounded())].id }
      }
     }
    }
   }
 }
 @ChartContentBuilder private var marks: some ChartContent {
    RuleMark(y: .value("Zero", 0.0)).foregroundStyle(Color.white.opacity(0.15))
    ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
     if row.status == .completed, let value = row.value {
      if value == 0 {
       PointMark(x: .value("Game", Double(index)), y: .value("Result", value)).symbolSize(25).foregroundStyle(color(row))
        .accessibilityLabel(chartLabel(row)).accessibilityValue(AnalyticsCardCopy.result(value, market: market))
      } else {
       BarMark(x: .value("Game", Double(index)), y: .value("Result", value), width: .fixed(rows.count <= 5 ? 28 : rows.count <= 10 ? 18 : 9)).cornerRadius(3).foregroundStyle(barStyle(row))
        .opacity(inspectedID == nil || inspectedID == row.id ? 1 : 0.5)
        .accessibilityLabel(chartLabel(row)).accessibilityValue(AnalyticsCardCopy.result(value, market: market))
      }
     }
    }
    ForEach(trend) { point in
     LineMark(x: .value("Game", Double(point.index)), y: .value("Historical trend", point.value))
      .interpolationMethod(.linear).lineStyle(StrokeStyle(lineWidth: 1.5))
      .foregroundStyle(Color.white.opacity(0.8))
      .accessibilityLabel("Historical trend").accessibilityValue(AnalyticsCardCopy.number(point.value))
    }
    if let threshold {
     RuleMark(y: .value("Selected line", threshold)).lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4])).foregroundStyle(.white.opacity(0.85))
      .annotation(position: .top, alignment: .trailing) { Text("Line " + String(format: "%g", threshold)).font(.system(size: 10, weight: .semibold)).padding(.horizontal, 4).background(navySurface) }
    }
 }
 private func chartLabel(_ row: AnalyticsObservation) -> String {
  analyticsFeedDate(row.date) + ", " + row.opponent + (outcome(row).map { ", " + ($0 == .hit ? "met the selected line" : $0 == .miss ? "missed the selected line" : "on the selected line") } ?? "")
 }
 private func legend(_ text: String, color: Color) -> some View {
  HStack(spacing: 5) { Circle().fill(color).frame(width: 6, height: 6); Text(text).font(.system(size: 10)) }.foregroundStyle(.secondary)
 }
}
private extension View {
 func researchSurface() -> some View {
  background(Color(white: 0.105), in: RoundedRectangle(cornerRadius: 18))
   .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(Color.white.opacity(0.065), lineWidth: 1))
 }
}
private func analyticsFeedDate(_ date: Date, compact: Bool = false) -> String {
 let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = TimeZone(identifier: "America/New_York")
 formatter.dateFormat = compact ? "MMM d" : "MMM d, yyyy"
 return formatter.string(from: date)
}

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
