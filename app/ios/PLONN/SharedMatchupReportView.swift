import SwiftUI

/// Public report reads are independent of private chat consent and generation.
struct SharedMatchupReportCard: View {
 let event: SharedMatchupEvent
 let client: SharedMatchupReportClient?
 let accessIdentity: String
 var narrativeClient: MatchupNarrativeClient? = nil
 var markets: [Market] = []
 var currentFacts: [MatchupOutlookSelection.Fact] = []
 var sourceLoading = false
 var ready = true
 var embedded = false
 var unavailableMessage = "Matchup outlook is unavailable."
 @Environment(\.scenePhase) private var scenePhase
 @State private var narrative: MatchupNarrativeV4.Result?
 @State private var narrativeNotice: String?
 @State private var outlookFacts: [MatchupOutlookSelection.Fact] = []
 @State private var response: SharedMatchupReportResponse?
 @State private var responseEvent: SharedMatchupEvent?
 @State private var responseReadKey: String?
 @State private var loading = false
 @State private var message: String?
 @State private var requestID = UUID()
 private var readKey: String { [event.sport,event.eventID,event.homeTeam,event.awayTeam,event.startISO,accessIdentity,String(ready),String(scenePhase == .active)].joined(separator:"\u{1F}") }
 private var currentResponse: SharedMatchupReportResponse? {
  responseEvent == event && responseReadKey == readKey ? response : nil
 }
 var body: some View {
  VStack(alignment:.leading,spacing:12) {
   Label("Matchup outlook",systemImage:"sparkles").font(.headline).accessibilityIdentifier("matchup-outlook-heading")
   if !currentFacts.isEmpty || (narrative != nil && responseEvent == event && responseReadKey == readKey) || currentResponse?.report != nil {
    MatchupOutlookLeadView(event:event,facts:currentFacts.isEmpty ? outlookFacts:currentFacts,markets:markets,clock:client?.now ?? {Date()}).id(readKey)
   }
   if let narrative, responseEvent == event, responseReadKey == readKey {
    DisclosureGroup("Supporting context") {
    MatchupNarrativeContent(result:narrative)
    }
   } else if let response=currentResponse, let report=response.report {
    DisclosureGroup("Supporting context") {
    if report.outlook.isEmpty {
     Text("No supported outlook is available from this evidence.").font(.subheadline)
    }
    let sections=MatchupOutlookPresentation.sections(response)
    ForEach(sections) { section in
     VStack(alignment:.leading,spacing:8) {
      Text(section.title).font(.subheadline.bold())
      Text(section.takeaway).font(.subheadline).accessibilityIdentifier("shared-report-outlook")
      ForEach(section.evidenceLines,id:\.self) {Text($0).font(.subheadline).accessibilityIdentifier("shared-report-rate")}
      if let comparison=section.comparison {Text(comparison).font(.subheadline).accessibilityIdentifier("shared-report-comparison")}
     }
    }
    ForEach(Array(Set(sections.map(\.uncertainty))).sorted(),id:\.self) {Text($0).font(.caption).foregroundStyle(.secondary)}
    if response.refreshing {Text("An updated outlook is being prepared.").font(.caption).foregroundStyle(.secondary)}
    }
   } else if !currentFacts.isEmpty {
    Text("Based on current season team statistics. Player availability may change; this is not a live prediction.").font(.caption).foregroundStyle(.secondary)
   } else {
    Text(loading || sourceLoading ? "Loading outlook…" : (message ?? unavailableMessage))
     .font(.subheadline).foregroundStyle(.secondary).accessibilityIdentifier("shared-report-status")
   }
  }.padding(embedded ? 0:14).background(embedded ? Color.clear:Color(uiColor:.systemBackground),in:RoundedRectangle(cornerRadius:18))
   .accessibilityElement(children:.contain).accessibilityIdentifier(embedded ? "shared-report-content":"shared-report-card")
   .task(id:readKey) {
    await readReport()
    let key=readKey
    while !Task.isCancelled, readKey==key {
     do {
      if let value=narrative {
       let now=Date()
       let next=try MatchupNarrativeCurrentState.nextCheck(value,now:now)
       try await Task.sleep(nanoseconds:UInt64(max(0.001,next.timeIntervalSince(now))*1_000_000_000))
       try Task.checkCancellation()
       try await Task.detached(priority:.utility) {try MatchupNarrativeCurrentState.validate(value,now:Date())}.value
       try Task.checkCancellation()
      } else if let value=response, value.report != nil {
       let now=client?.now() ?? Date()
       let next=try SharedMatchupDisplayClock.nextCheck(value,now:now)
       try await Task.sleep(nanoseconds:UInt64(max(0.001,next.timeIntervalSince(now))*1_000_000_000))
       try Task.checkCancellation()
       guard readKey==key else{return}
       let aged=try SharedMatchupDisplayClock.update(value,now:client?.now() ?? Date())
       response=aged
       if aged.status != .ready {outlookFacts=[]}
       if aged.reason == "event_started" {message="Pregame outlook is no longer available for this event.";return}
      } else {return}
     } catch is CancellationError {return}
     catch {
      guard readKey==key else{return}
      narrative=nil;response=nil;loading=false
      message="This outlook is out of date. Current player availability and game conditions need verification."
      return
     }
    }
   }
   .onDisappear {requestID=UUID()}
 }
 @MainActor private func readReport() async {
  let key=readKey
  let ticket=UUID();requestID=ticket;response=nil;responseEvent=nil;responseReadKey=nil;message=nil;loading=false;narrative=nil;narrativeNotice=nil;outlookFacts=[]
  guard ready,scenePhase == .active else{return}
  loading=true
  if let narrativeClient {
   do {
    if let value=try await narrativeClient.fetch(event:event) {
     let now=narrativeClient.now()
     let facts=await Task.detached(priority:.utility) {MatchupOutlookSelection.facts(value,now:now)}.value
     guard !Task.isCancelled,requestID==ticket,readKey==key else{return}
     outlookFacts=facts;narrative=value;responseEvent=event;responseReadKey=key;loading=false;return
    }
   } catch {
    guard !Task.isCancelled,requestID==ticket,readKey==key else{return}
    if case SharedMatchupReportError.authenticationRequired=error {
     message="Sign in to view this matchup’s outlook.";loading=false;return
    }
    narrativeNotice="The newer outlook could not be verified. Earlier recorded context is shown below."
   }
  }
  guard let client else {message=unavailableMessage;loading=false;return}
  do {
   let value=try await client.fetch(event:event)
   guard !Task.isCancelled,requestID==ticket,readKey==key else{return}
   response=value;responseEvent=event;responseReadKey=key
   outlookFacts=MatchupOutlookSelection.facts(value,now:client.now())
   if value.status == .missing {
    switch value.reason {
    case "event_started": message="Pregame outlook is no longer available for this event."
    case "invalid_cached_report": message="The outlook could not be verified."
    case "source_unavailable": message="Source data for this outlook is unavailable."
    case "generation_failed": message="The outlook could not be prepared."
    default: message="No outlook is available for this matchup yet."
    }
   }
  } catch {
   guard !Task.isCancelled,requestID==ticket,readKey==key else{return}
   if case SharedMatchupReportError.authenticationRequired=error {message="Sign in to view this matchup’s outlook."}
   else {message="The outlook could not be loaded or verified."}
  }
  guard requestID==ticket,readKey==key else{return};loading=false
 }

}

private struct MatchupOutlookLeadView: View {
 let event:SharedMatchupEvent
 let facts:[MatchupOutlookSelection.Fact]
 let markets:[Market]
 let clock:@Sendable ()->Date
 @State private var directions:[MatchupOutlookSelection.Direction]=[]
 @State private var settled=false
 private struct Input:Hashable {let event:SharedMatchupEvent;let facts:[MatchupOutlookSelection.Fact];let markets:[Market]}
 private var input:Input {Input(event:event,facts:facts,markets:markets)}
 var body:some View {
  VStack(alignment:.leading,spacing:8) {
   if let lead=directions.first?.lead {
    Text(lead.title).font(.subheadline.bold()).accessibilityIdentifier("matchup-outlook-lead-title")
    Text(lead.selection).font(.headline).accessibilityIdentifier("matchup-outlook-selection")
    Text(lead.explanation).font(.subheadline).accessibilityIdentifier("matchup-outlook-reason")
    if let date=lead.quoteDate {Text(MatchupOutlookSelection.quoteCaption(date)).font(.caption).foregroundStyle(.secondary)}
    Text(lead.qualification).font(.caption).foregroundStyle(.secondary)
    if directions.count > 1 {
     Divider()
     Text("Also supported").font(.subheadline.bold()).accessibilityIdentifier("matchup-outlook-alternatives-heading")
     ForEach(Array(directions.dropFirst().enumerated()),id:\.element.id) { index,direction in
      VStack(alignment:.leading,spacing:5) {
       Text("\(index+2). "+direction.lead.selection).font(.subheadline.bold()).accessibilityIdentifier("matchup-outlook-alternative")
       Text(direction.lead.shortReason).font(.caption).accessibilityIdentifier("matchup-outlook-alternative-reason")
       if let date=direction.lead.quoteDate {Text(MatchupOutlookSelection.quoteCaption(date)).font(.caption2).foregroundStyle(.secondary)}
       if direction.lead.edge == nil {Text("Historical direction; no measured betting edge.").font(.caption2).foregroundStyle(.secondary)}
       if let note=direction.correlationNote {Text(note).font(.caption2).foregroundStyle(.secondary).accessibilityIdentifier("matchup-outlook-correlation")}
      }.padding(.top,4)
     }
    }
   } else {
    Text(settled ? MatchupOutlookSelection.noDirection : "Comparing supported directions…").font(.subheadline).accessibilityIdentifier("matchup-outlook-no-edge")
   }
  }.task(id:input) {
   let value=input
   directions=[];settled=false
   while !Task.isCancelled {
    let now=clock()
    let selected=await Task.detached(priority:.utility) {MatchupOutlookSelection.directions(event:value.event,facts:value.facts,markets:value.markets,now:now)}.value
    guard !Task.isCancelled else{return}
    directions=selected;settled=true
    // Recompute on price/model expiry, event start, and clock changes without a fetch.
    var deadlines=[now.addingTimeInterval(30)]
    if let start=sportsDate(value.event.startISO) {deadlines.append(start)}
    for market in value.markets {
     if let stamp=market.modelUpdated.flatMap(sportsDate) {deadlines.append(stamp.addingTimeInterval(120.001))}
     for stamp in ([market.updated] + market.quotes.map(\.updated)).compactMap({$0.flatMap(sportsDate)}) {
      deadlines.append(stamp.addingTimeInterval(120.001));deadlines.append(stamp.addingTimeInterval(86400.001))
     }
    }
    let next=deadlines.filter{$0>now}.min() ?? now.addingTimeInterval(30)
    do {try await Task.sleep(nanoseconds:UInt64(max(0.001,next.timeIntervalSince(now))*1_000_000_000))} catch {return}
   }
  }
 }
}


private struct MatchupNarrativeContent:View {
 let result:MatchupNarrativeV4.Result
 var body:some View {
  VStack(alignment:.leading,spacing:12) {
   let starters=MatchupOutlookPresentation.starters(result.designations,event:result.event)
   if !starters.isEmpty {
    Text(result.event.sport == "MLB" ? "Starting pitchers":result.event.sport == "NHL" ? "Starting goalies":"Reported starters").font(.subheadline.bold())
    ForEach(starters) {starter in
     Text(starter.team+": "+starter.name+" · "+starter.status)
      .font(.subheadline).accessibilityIdentifier("matchup-outlook-starter")
    }
   }
   ForEach(result.claims,id:\.id) {claim in
    VStack(alignment:.leading,spacing:6) {
     Text(claim.label).font(.subheadline.bold())
     Text(MatchupOutlookPresentation.text(for:claim,sport:result.event.sport)).font(.subheadline).accessibilityIdentifier("matchup-narrative-claim")
    }
   }
   ForEach(Array(Set(result.claims.map {MatchupOutlookPresentation.uncertainty($0.uncertainty)})).sorted(),id:\.self) {Text($0).font(.caption).foregroundStyle(.secondary)}
   ForEach(MatchupOutlookPresentation.notes(for:result),id:\.self) {Text($0).font(.caption).foregroundStyle(.secondary)}
   ForEach(MatchupOutlookPresentation.roleLimitations(result),id:\.self) {Text($0).font(.caption).foregroundStyle(.secondary)}
   let injury=result.injuryContext
   let observations=MatchupOutlookPresentation.injuryObservations(result)
   if !observations.isEmpty {
    Text("Player updates").font(.subheadline.bold())
    ForEach(observations,id:\.canonical) {row in
     Text(MatchupOutlookPresentation.injuryText(row,context:injury))
      .font(.subheadline).accessibilityIdentifier("matchup-narrative-injury")
    }
   }
  }
 }
}

#if DEBUG && targetEnvironment(simulator)
enum SharedReportOfflineFixture {
 static var payload:String {ProcessInfo.processInfo.environment["PLONN_SHARED_REPORT_FIXTURE"] ?? ""}
 static var active:Bool {!payload.isEmpty}
 static var recorded:Bool {ProcessInfo.processInfo.environment["PLONN_SHARED_REPORT_RECORDED"] == "1"}
 static func responseData(_ root:[String:Any])->Data? {
  if let raw=root["responseUTF8"] as? String {return raw.data(using:.utf8)}
  guard let object=root["response"] as? [String:Any] else{return nil}
  return try? JSONSerialization.data(withJSONObject:object)
 }
}
struct SharedReportOfflineFixtureView:View {
 var body:some View {
  NavigationStack {ScrollView {VStack(alignment:.leading,spacing:16) {
   Text(SharedReportOfflineFixture.recorded ? "Recorded public report · Offline preview · No live request" : "Offline fixture · Synthetic data · No network or inference").font(.caption.bold()).accessibilityIdentifier("shared-report-offline-banner")
   if let data=SharedReportOfflineFixture.payload.data(using:.utf8),
      let root=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any],
      let body=SharedReportOfflineFixture.responseData(root),
      let object=(try? JSONSerialization.jsonObject(with:body)) as? [String:Any],let rawEvent=object["event"],
      let eventData=try? JSONSerialization.data(withJSONObject:rawEvent),
      let event=try? JSONDecoder().decode(SharedMatchupEvent.self,from:eventData) {
    let status=root["statusCode"] as? Int ?? 200
    let iso=ISO8601DateFormatter()
    let now=iso.date(from:root["now"] as? String ?? "2026-10-13T13:00:00Z")!
    let client=SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{"offline.fixture"},transport:{request in
     (body,HTTPURLResponse(url:request.url!,statusCode:status,httpVersion:nil,headerFields:nil)!)
    },now:{now})
    let markets=(root["markets"].flatMap {try? JSONSerialization.data(withJSONObject:$0)}).flatMap {try? JSONDecoder().decode([Market].self,from:$0)} ?? []
    Text(event.sport+" · "+event.awayTeam+" at "+event.homeTeam).font(.headline)
    SharedMatchupReportCard(event:event,client:client,accessIdentity:"offline-fixture",markets:markets)
   } else {Text("Invalid offline fixture")}
  }.padding()}.navigationTitle(SharedReportOfflineFixture.recorded ? "Matchup outlook" : "Shared report fixture")}
 }
}
#endif
