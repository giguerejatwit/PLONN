import SwiftUI

/// Public report reads are independent of private chat consent and generation.
struct SharedMatchupReportCard: View {
 let event: SharedMatchupEvent
 let client: SharedMatchupReportClient?
 let accessIdentity: String
 var narrativeClient: MatchupNarrativeClient? = nil
 var ready = true
 var embedded = false
 var unavailableMessage = "Matchup outlook is unavailable."
 @Environment(\.scenePhase) private var scenePhase
 @State private var narrative: MatchupNarrativeV4.Result?
 @State private var narrativeNotice: String?
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
   if let narrative, responseEvent == event, responseReadKey == readKey {
    MatchupNarrativeContent(result:narrative)
   } else if let response=currentResponse, let report=response.report {
    if let narrativeNotice {Text(narrativeNotice).font(.caption).foregroundStyle(.orange)}
    if response.status == .stale {
     Text("This outlook is out of date. Current conditions may differ.")
      .font(.subheadline).foregroundStyle(.orange).accessibilityIdentifier("shared-report-status")
    }
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
   } else {
    Text(loading ? "Loading outlook…" : (message ?? unavailableMessage))
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
  let ticket=UUID();requestID=ticket;response=nil;responseEvent=nil;responseReadKey=nil;message=nil;loading=false;narrative=nil;narrativeNotice=nil
  guard ready,scenePhase == .active else{return}
  loading=true
  if let narrativeClient {
   do {
    if let value=try await narrativeClient.fetch(event:event) {
     guard !Task.isCancelled,requestID==ticket,readKey==key else{return}
     narrative=value;responseEvent=event;responseReadKey=key;loading=false;return
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
    Text(event.sport+" · "+event.awayTeam+" at "+event.homeTeam).font(.headline)
    SharedMatchupReportCard(event:event,client:client,accessIdentity:"offline-fixture")
   } else {Text("Invalid offline fixture")}
  }.padding()}.navigationTitle(SharedReportOfflineFixture.recorded ? "Matchup outlook" : "Shared report fixture")}
 }
}
#endif
