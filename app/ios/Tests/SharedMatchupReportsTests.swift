import XCTest
@testable import PLONNCore
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

final class SharedMatchupReportsTests: XCTestCase {
    func testFrozenReleaseArchiveAgesPerEventWithoutChangingBytes() throws {
        let path="/Users/indoorgeo/Dev/PLONN/app/ios/PLONN/FrozenMatchupReports-20261013.json"
        let bundle=try Data(contentsOf:URL(fileURLWithPath:path))
        let records=try SharedMatchupFrozenArchive.load(bundle)
        XCTAssertEqual(records.count,91)
        let original=records
        let raw=try XCTUnwrap(records.first {data in (try? JSONDecoder().decode(SharedMatchupReportResponse.self,from:data).event.eventID)=="espn:401872995"})
        let saved=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
        let generated=try SharedMatchupReportValidator.date(XCTUnwrap(saved.generatedAt))
        let ready=try SharedMatchupFrozenArchive.read(records,event:saved.event,now:generated)
        XCTAssertEqual(ready.status,.ready)
        let stale=try SharedMatchupFrozenArchive.read(records,event:saved.event,now:generated.addingTimeInterval(86400))
        XCTAssertEqual(stale.status,.stale);XCTAssertEqual(stale.reason,"generation_expired")
        XCTAssertEqual(stale.report,saved.report);XCTAssertEqual(stale.snapshot,saved.snapshot);XCTAssertEqual(stale.generatedAt,saved.generatedAt)
        XCTAssertThrowsError(try SharedMatchupFrozenArchive.read(records,event:saved.event,now:generated.addingTimeInterval(-1)))
        let start=try SharedMatchupReportValidator.date(saved.event.startISO)
        let ended=try SharedMatchupFrozenArchive.read(records,event:saved.event,now:start)
        XCTAssertEqual(ended.reason,"event_started");XCTAssertNil(ended.report)
        let another=try XCTUnwrap(records.compactMap {try? JSONDecoder().decode(SharedMatchupReportResponse.self,from:$0)}.first { (try? SharedMatchupReportValidator.date($0.event.startISO)) ?? .distantPast > start})
        XCTAssertEqual(try SharedMatchupFrozenArchive.read(records,event:another.event,now:start).status,.stale)
        XCTAssertEqual(records,original)
        var corrupt=bundle;corrupt[0]=32;XCTAssertThrowsError(try SharedMatchupFrozenArchive.load(corrupt))
    }
    func testFrozenLocalReadStillRequiresAuthenticationAndNeverUsesNetwork() async throws {
        let value=try example(),requested=try event(value),bytes=try data(value)
        let now=fixtureNow
        let client=SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{throw SharedMatchupReportError.authenticationRequired},transport:{_ in XCTFail("No network permitted");throw SharedMatchupReportError.unavailable},now:{now},localReader:{_,_ in XCTFail("No unauthenticated local read");throw SharedMatchupReportError.invalidResponse})
        do {_ = try await client.fetch(event:requested);XCTFail("Must require auth")} catch {XCTAssertEqual(error as? SharedMatchupReportError,.authenticationRequired)}
        let allowed=SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{"unit-test-session"},transport:{_ in XCTFail("No network permitted");throw SharedMatchupReportError.unavailable},now:{now},localReader:{event,date in try SharedMatchupReportValidator.validate(bytes,event:event,now:date)})
        let read=try await allowed.fetch(event:requested);XCTAssertEqual(read.status,.ready)
    }

    func testPresentationAll91ExactWireProjections() throws {
        struct Item:Decodable {let rawUTF8:String;let validatedAt:String;let expected:SharedMatchupPresentation}
        struct Fixture:Decodable {let cases:[Item]}
        let path="/Users/indoorgeo/Dev/PLONN/app/contracts/shared-insights-presentation-v1.json"
        let cases=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:URL(fileURLWithPath:path))).cases
        XCTAssertEqual(cases.count,91)
        var count=0
        for item in cases {
            let value=try SharedMatchupPresentationCopy.project(Data(item.rawUTF8.utf8),now:SharedMatchupReportValidator.date(item.validatedAt))
            XCTAssertEqual(value,item.expected,item.expected.event.eventID)
            count += value.sections.count
        }
        XCTAssertEqual(count,118)
    }
    func testPresentationProofFallbackAndRounding() throws {
        XCTAssertEqual(SharedMatchupPresentationCopy.number(1.005),"1.01")
        XCTAssertEqual(SharedMatchupPresentationCopy.number(23.3164556962),"23.32")
        XCTAssertEqual(SharedMatchupPresentationCopy.number(21.0),"21")
        let proof=try XCTUnwrap(SharedMatchupPresentationCopy.proofs.first)
        XCTAssertTrue(SharedMatchupPresentationCopy.comparison(proof.referenceFact,sport:proof.sport,proofs:[]).contains("Full-league coverage is not verified"))
        XCTAssertTrue(SharedMatchupPresentationCopy.comparison(proof.referenceFact,sport:proof.sport,proofs:[proof]).contains("across all 32 teams"))
        let original=try example(), raw=try data(original)
        let response=try SharedMatchupReportValidator.validate(raw,event:event(original),now:fixtureNow)
        let o=try XCTUnwrap(response.report?.outlook.first),snapshot=try XCTUnwrap(response.snapshot)
        let unknown=SharedMatchupOutlook(id:"unknown",ruleID:o.ruleID,label:o.label,text:o.text,uncertainty:o.uncertainty,factIDs:o.factIDs,sourceIDs:o.sourceIDs)
        let result=SharedMatchupPresentationCopy.section(unknown,snapshot:snapshot,checkedAt:try XCTUnwrap(response.sourceCheckedAt))
        XCTAssertFalse(result.mapped);XCTAssertEqual(result.takeaway,o.text);XCTAssertNil(result.sampleWindow)
    }

    func testScoreboardMinutePrecisionAdapter() throws {
        XCTAssertEqual(SharedMatchupEvent.scoreboardStart("2026-10-16T00:15Z"),"2026-10-16T00:15:00Z")
        XCTAssertEqual(SharedMatchupEvent.scoreboardStart("2026-10-16T00:15:12Z"),"2026-10-16T00:15:12Z")
        XCTAssertEqual(SharedMatchupEvent.scoreboardStart("invalid"),"invalid")
        XCTAssertThrowsError(try SharedMatchupReportValidator.date("2026-10-16T00:15Z"))
        let original=try example();let e=try event(original)
        let adapted=SharedMatchupEvent(sport:e.sport,eventID:e.eventID,homeTeam:e.homeTeam,awayTeam:e.awayTeam,startISO:SharedMatchupEvent.scoreboardStart(String(e.startISO.prefix(16))+"Z"))
        XCTAssertEqual(try SharedMatchupReportValidator.validate(data(original),event:adapted,now:fixtureNow).status,.ready)
    }
    func testLocalCacheSeparatesAllFourSportsAndPreservesExactMatch() throws {
        let original=try example("NFL"),raw=try data(original),savedEvent=try event(original)
        XCTAssertEqual(try SharedMatchupLocalCache.response(recorded:raw,event:savedEvent,now:fixtureNow),raw)
        for sport in ["NBA","MLB","NHL"] {
            let requested=try event(example(sport))
            let bytes=try SharedMatchupLocalCache.response(recorded:raw,event:requested,now:fixtureNow)
            let value=try SharedMatchupReportValidator.validate(bytes,event:requested,now:fixtureNow)
            XCTAssertEqual(value.status,.missing);XCTAssertEqual(value.reason,"cache_miss")
            XCTAssertNil(value.report);XCTAssertNil(value.snapshot);XCTAssertNil(value.generatedAt)
            XCTAssertNil(value.sourceFingerprint)
        }
        let past=SharedMatchupEvent(sport:"NBA",eventID:"espn:123",homeTeam:"Boston Celtics",awayTeam:"New York Knicks",startISO:"2026-10-12T00:00:00Z")
        let bytes=try SharedMatchupLocalCache.response(recorded:raw,event:past,now:fixtureNow)
        XCTAssertEqual(try SharedMatchupReportValidator.validate(bytes,event:past,now:fixtureNow).reason,"event_started")
    }
    func testRawBundleExactEventIsolationAndBounds() throws {
        let raws=try ["NFL","NBA","MLB","NHL"].map {try data(example($0))}
        func bundle(_ values:[Data]) throws -> Data {try JSONEncoder().encode(values.map {$0.base64EncodedString()})}
        let loaded=try SharedMatchupLocalCache.records(bundle:bundle(raws),now:fixtureNow)
        XCTAssertEqual(loaded,raws)
        var started=try example("NHL")
        var startedEvent=started["event"] as! [String:Any]
        startedEvent["eventID"]="espn:999999";startedEvent["startISO"]="2026-10-12T00:00:00Z";started["event"]=startedEvent
        XCTAssertEqual(try SharedMatchupLocalCache.records(bundle:bundle(raws+[try data(started)]),now:fixtureNow),raws)
        for (index,sport) in ["NFL","NBA","MLB","NHL"].enumerated() {
            XCTAssertEqual(try SharedMatchupLocalCache.response(records:loaded,event:event(example(sport)),now:fixtureNow),raws[index])
        }
        XCTAssertThrowsError(try SharedMatchupLocalCache.records(bundle:bundle([raws[0],raws[0]]),now:fixtureNow))
        XCTAssertThrowsError(try SharedMatchupLocalCache.records(bundle:bundle([]),now:fixtureNow))
        XCTAssertThrowsError(try SharedMatchupLocalCache.records(bundle:bundle(Array(repeating:raws[0],count:95)),now:fixtureNow))
        XCTAssertThrowsError(try SharedMatchupLocalCache.records(bundle:Data(repeating:32,count:SharedMatchupLocalCache.maximumBundleBytes+1),now:fixtureNow))
        var changed=try example();changed["sourceFingerprint"]=String(repeating:"0",count:64)
        XCTAssertThrowsError(try SharedMatchupLocalCache.records(bundle:bundle([try data(changed)]),now:fixtureNow))
        let other=SharedMatchupEvent(sport:"NHL",eventID:"espn:98765",homeTeam:"Boston Bruins",awayTeam:"Seattle Kraken",startISO:"2026-10-16T00:00:00Z")
        let missing=try SharedMatchupLocalCache.response(records:loaded,event:other,now:fixtureNow)
        XCTAssertEqual(try SharedMatchupReportValidator.validate(missing,event:other,now:fixtureNow).status,.missing)
    }
    func testPublishedRawBundleWhenProvided() throws {
        guard let path=ProcessInfo.processInfo.environment["PLONN_VERIFY_REPORT_BUNDLE"] else {throw XCTSkip("No published bundle supplied")}
        let now=Date(),bundle=try Data(contentsOf:URL(fileURLWithPath:path))
        let originals=try JSONDecoder().decode([String].self,from:bundle)
        for encoded in originals {
            let raw=try XCTUnwrap(Data(base64Encoded:encoded))
            let envelope=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
            let generated=try SharedMatchupReportValidator.date(XCTUnwrap(envelope.generatedAt))
            _ = try SharedMatchupReportValidator.validate(raw,event:envelope.event,now:generated)
        }
        let raws=try SharedMatchupLocalCache.records(bundle:bundle,now:now)
        for raw in raws {
            let envelope=try JSONDecoder().decode(SharedMatchupReportResponse.self,from:raw)
            XCTAssertEqual(try SharedMatchupLocalCache.response(records:raws,event:envelope.event,now:now),raw)
            let accepted=try SharedMatchupReportValidator.validate(raw,event:envelope.event,now:now)
            XCTAssertNotNil(accepted.report)
            print("Validated published report: \(envelope.event.sport) \(envelope.event.eventID)")
        }
    }
    func testV3ActualCandidateParity() throws {
        struct Case:Decodable {let snapshot:SharedMatchupSnapshot;let expectedCandidates:[SharedMatchupOutlook]}
        struct Fixtures:Decodable {let cases:[Case]}
        let path="/Users/indoorgeo/Dev/PLONN/app/backend/local_lab/verification/shared-insights-live-20261013/paired-v3-review-fixtures.json"
        let cases=try JSONDecoder().decode(Fixtures.self,from:Data(contentsOf:URL(fileURLWithPath:path))).cases
        XCTAssertEqual(cases.count,50)
        for item in cases {
            let selection=SharedMatchupSelection(observationIDs:[item.snapshot.facts[0].id],outlookIDs:item.expectedCandidates.map(\.id))
            let rendered=try SharedMatchupReportValidator.render(selection:selection,snapshot:item.snapshot,rules:"paired-reference-v3")
            XCTAssertEqual(rendered.outlook,item.expectedCandidates,item.snapshot.event.eventID)
            if item.snapshot.facts.contains(where:{$0.period.lowercased().contains("preseason")}) {
                XCTAssertEqual(rendered.missingContext.first,"Preseason observations are a limited exhibition sample; rotations and player availability can differ from regular-season games.")
            }
            XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:selection,snapshot:item.snapshot),item.snapshot.event.eventID)
        }
    }
    func testV3PreservesExistingDirectionalPrecedence() throws {
        for sport in ["NFL","NBA","MLB","NHL"] {
            let original=try example(sport)
            let snapshot=try JSONDecoder().decode(SharedMatchupSnapshot.self,from:JSONSerialization.data(withJSONObject:original["snapshot"]!))
            let report=try JSONDecoder().decode(SharedMatchupReport.self,from:JSONSerialization.data(withJSONObject:original["report"]!))
            let v3=try SharedMatchupReportValidator.render(selection:report.selection,snapshot:snapshot,rules:"paired-reference-v3")
            XCTAssertEqual(v3.outlook.map(\.id),report.outlook.map(\.id))
            XCTAssertEqual(v3.outlook.map(\.text),report.outlook.map(\.text))
            XCTAssertTrue(v3.outlook.allSatisfy {$0.ruleID.hasPrefix("paired-reference-v3:")})
            for id in ["game:near-reference","game:paired-context","game:limited-sample"] {
                XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:SharedMatchupSelection(observationIDs:report.selection.observationIDs,outlookIDs:[id]),snapshot:snapshot,rules:"paired-reference-v3"))
            }
        }
    }
    private let fixtureNow = ISO8601DateFormatter().date(from:"2026-10-13T13:00:00Z")!
    private func example(_ sport:String = "NFL") throws -> [String:Any] {
        let file=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("contracts/shared-insights-v1.json")
        let root=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
        return try XCTUnwrap((root["examples"] as? [String:Any])?[sport] as? [String:Any])
    }
    private func data(_ object:[String:Any]) throws -> Data {try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys,.withoutEscapingSlashes])}
    private func event(_ object:[String:Any]) throws -> SharedMatchupEvent {try JSONDecoder().decode(SharedMatchupEvent.self,from:JSONSerialization.data(withJSONObject:object["event"]!))}
    private func validate(_ object:[String:Any], now:Date? = nil) throws -> SharedMatchupReportResponse {
        try SharedMatchupReportValidator.validate(data(object),event:event(object),now:now ?? fixtureNow)
    }
    func testAllFourBundledFixturesAndDerivedPremises() throws {
        for sport in ["NFL","NBA","MLB","NHL"] {
            let result=try validate(example(sport))
            XCTAssertEqual(result.status,.ready)
            XCTAssertEqual(result.report?.observations.count,3)
            XCTAssertEqual(result.report?.outlook.count,1)
            XCTAssertTrue(result.report!.provenance.contains("rule-based"))
            XCTAssertEqual(Set(result.report!.observations.map(\.factID)),Set(result.report!.outlook.flatMap(\.factIDs)))
        }
    }
    func testRejectsChangedIdentitySourceFingerprintVersionAndInventedProse() throws {
        let original=try example()
        for mutation in 0..<9 {
            var object=original
            switch mutation {
            case 0: object["sourceFingerprint"]=String(repeating:"0",count:64)
            case 1: var v=object["versions"] as! [String:Any];v["prompt"]="unknown";object["versions"]=v
            case 2: var r=object["report"] as! [String:Any];var o=r["outlook"] as! [[String:Any]];o[0]["text"]="Seattle Seahawks will win.";r["outlook"]=o;object["report"]=r
            case 3: var snap=object["snapshot"] as! [String:Any];var facts=snap["facts"] as! [[String:Any]];facts[0]["value"]=99;snap["facts"]=facts;object["snapshot"]=snap
            case 4: object["generatedAt"]="2026-10-14T00:00:00.000Z"
            case 5: object["sourceRetrievedAt"]="2026-10-13T12:30:00.000Z"
            case 6: object["unexpected"]="field"
            case 7: object["schemaVersion"]=true
            default: var snap=object["snapshot"] as! [String:Any];var e=snap["event"] as! [String:Any];e["homeTeam"]="Other Team";snap["event"]=e;object["snapshot"]=snap
            }
            XCTAssertThrowsError(try validate(object),"mutation \(mutation)")
        }
        let expected=try event(original)
        let wrong=SharedMatchupEvent(sport:expected.sport,eventID:"espn:99",homeTeam:expected.homeTeam,awayTeam:expected.awayTeam,startISO:expected.startISO)
        XCTAssertThrowsError(try SharedMatchupReportValidator.validate(data(original),event:wrong,now:fixtureNow))
    }
    func testFreshnessBoundaryMissingAndStartedEvent() throws {
        var object=try example()
        let boundary=try SharedMatchupReportValidator.date("2026-10-14T12:00:00.000Z")
        XCTAssertThrowsError(try validate(object,now:boundary))
        object["status"]="stale";object["reason"]="source_expired"
        XCTAssertEqual(try validate(object,now:boundary).status,.stale)
        object["status"]="missing";object["reason"]="cache_miss"
        for field in ["report","snapshot","generatedAt","sourceAsOf","sourceRetrievedAt","sourceCheckedAt","sourceFingerprint"] {object[field]=NSNull()}
        XCTAssertEqual(try validate(object).status,.missing)
        object["reason"]="private arbitrary error";XCTAssertThrowsError(try validate(object))
        object["reason"]="event_started"
        XCTAssertThrowsError(try validate(object))
        XCTAssertEqual(try validate(object,now:try SharedMatchupReportValidator.date("2026-10-16T00:00:00.000Z")).status,.missing)
    }
    func testRuleEligibilityRejectsInsufficientSampleAndWrongPair() throws {
        let object=try example();let response=try validate(object)
        var snapshot=object["snapshot"] as! [String:Any], facts=snapshot["facts"] as! [[String:Any]]
        facts[0]["sample"]=1;snapshot["facts"]=facts
        let typed=try JSONDecoder().decode(SharedMatchupSnapshot.self,from:JSONSerialization.data(withJSONObject:snapshot))
        XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:response.report!.selection,snapshot:typed))
        facts[0]["sample"]=12;facts[0]["team"]="Seattle Seahawks";snapshot["facts"]=facts
        let sameTeamDefense=try JSONDecoder().decode(SharedMatchupSnapshot.self,from:JSONSerialization.data(withJSONObject:snapshot))
        XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:response.report!.selection,snapshot:sameTeamDefense))
    }
    func testResistanceLowerEfficiencyAndPaceRulesUseAllPremises() throws {
        let base = try validate(example("NBA")).snapshot!
        func fact(_ id: String, _ team: String, _ role: String, _ metric: String, _ value: Double) -> SharedMatchupFact {
            SharedMatchupFact(id:id,team:team,role:role,metric:metric,unit:metric == "pace" ? "possessions per game" : "points per 100 possessions",period:"2026 regular season",windowStart:"2026-09-01T00:00:00.000Z",windowEnd:"2026-10-12T23:00:00.000Z",value:value,sample:12,sourceIDs:["espn-team-stats"])
        }
        let home = base.event.homeTeam, away = base.event.awayTeam
        let lower = [fact("h-o",home,"offense","offensiveEfficiency",90),fact("h-d",home,"defense","offensiveEfficiency",90),fact("a-o",away,"offense","offensiveEfficiency",90),fact("a-d",away,"defense","offensiveEfficiency",90),fact("r","league","reference","offensiveEfficiency",100),fact("h-p",home,"offense","pace",95),fact("a-p",away,"offense","pace",95),fact("r-p","league","reference","pace",100)]
        let snapshot = SharedMatchupSnapshot(schemaVersion:1,event:base.event,sources:base.sources,facts:lower)
        let report = try SharedMatchupReportValidator.render(selection:SharedMatchupSelection(observationIDs:["h-o"],outlookIDs:["game:lower-efficiency","game:pace:low"]),snapshot:snapshot)
        XCTAssertEqual(report.observations.count,8)
        XCTAssertTrue(report.outlook[0].text.contains("does not establish a low game total"))
        XCTAssertTrue(report.outlook[1].text.contains("does not establish scoring efficiency"))
        let resistance = SharedMatchupSnapshot(schemaVersion:1,event:base.event,sources:base.sources,facts:[fact("h-o",home,"offense","offensiveEfficiency",100),fact("a-d",away,"defense","offensiveEfficiency",90),fact("r","league","reference","offensiveEfficiency",100)])
        XCTAssertEqual(try SharedMatchupReportValidator.render(selection:SharedMatchupSelection(observationIDs:["h-o"],outlookIDs:["home:offensiveEfficiency:resistance"]),snapshot:resistance).outlook[0].label,"Opposing defensive resistance")
    }
    func testReviewedNFLMixedScoringPublicCandidateAndBoundaries() throws {
        let file=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("contracts/shared-insights-seahawks-broncos-20261013.json")
        let root=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
        XCTAssertTrue(root["generatedAt"] is NSNull)
        let base=try JSONDecoder().decode(SharedMatchupSnapshot.self,from:JSONSerialization.data(withJSONObject:root["snapshot"]!))
        let expected=try JSONDecoder().decode([SharedMatchupOutlook].self,from:JSONSerialization.data(withJSONObject:root["expectedCandidates"]!))
        let selection=SharedMatchupSelection(observationIDs:expected[0].factIDs,outlookIDs:["game:mixed-scoring"])
        XCTAssertEqual(try SharedMatchupReportValidator.render(selection:selection,snapshot:base).outlook,expected)
        XCTAssertEqual(SharedMatchupVersions.current.rules,"paired-reference-v2")
        func edited(_ changes:[String:Any], at:Int? = nil, sport:String? = nil)throws->SharedMatchupSnapshot {
            var object=root["snapshot"] as! [String:Any]
            var facts=object["facts"] as! [[String:Any]]
            for index in facts.indices where facts[index]["metric"] as? String == "pointsPerGame" {
                let team=facts[index]["team"] as! String,role=facts[index]["role"] as! String
                facts[index]["value"]=role == "reference" ? 10 : role == "offense" ? (team == "Seattle Seahawks" ? 11:9) : (team == "Seattle Seahawks" ? 9:11)
            }
            if let at {for (key,value) in changes {facts[at][key]=value}}
            object["facts"]=facts
            if let sport {var event=object["event"] as! [String:Any];event["sport"]=sport;object["event"]=event}
            return try JSONDecoder().decode(SharedMatchupSnapshot.self,from:JSONSerialization.data(withJSONObject:object))
        }
        let boundary=try edited([:])
        XCTAssertNoThrow(try SharedMatchupReportValidator.render(selection:selection,snapshot:boundary))
        let rawFacts=(root["snapshot"] as! [String:Any])["facts"] as! [[String:Any]]
        for (team,role,value) in [("Seattle Seahawks","offense",10.999999999),("Denver Broncos","offense",9.000000001),("Seattle Seahawks","defense",8.999999999),("Denver Broncos","defense",11.000000001)] {
            let index=rawFacts.firstIndex {$0["team"] as? String == team && $0["role"] as? String == role && $0["metric"] as? String == "pointsPerGame"}!
            XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:selection,snapshot:edited(["value":value],at:index)))
        }
        let defense=rawFacts.firstIndex {$0["team"] as? String == "Denver Broncos" && $0["role"] as? String == "defense" && $0["metric"] as? String == "pointsPerGame"}!
        for change:[String:Any] in [["unit":"runs per game"],["period":"2025 regular season"],["windowEnd":"2026-10-12T20:00:00Z"],["sample":2],["team":"Seattle Seahawks"]] {
            XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:selection,snapshot:edited(change,at:defense)))
        }
        for id in expected[0].factIDs {
            let missing=SharedMatchupSnapshot(schemaVersion:1,event:base.event,sources:base.sources,facts:base.facts.filter {$0.id != id})
            XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:selection,snapshot:missing))
        }
        XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:selection,snapshot:edited([:],sport:"NBA")))
    }
    func testWholeLeagueCollectorFixturesAndBounds() throws {
        let file=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("contracts/shared-insights-collector-v1.json")
        let root=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
        let examples=try XCTUnwrap(root["examples"] as? [String:[String:Any]])
        for sport in ["NFL","NBA","MLB","NHL"] {
            let object=try XCTUnwrap(examples[sport])
            let wireURL=URL(fileURLWithPath:#filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/shared-collector-"+sport+".json")
            let result=try SharedMatchupReportValidator.validate(Data(contentsOf:wireURL),event:event(object),now:fixtureNow)
            XCTAssertEqual(result.status,.ready)
            XCTAssertLessThanOrEqual(try data(object).count,65536)
            let snapshot=try XCTUnwrap(result.snapshot)
            XCTAssertLessThanOrEqual(try JSONEncoder().encode(snapshot).count,32768)
            if sport == "NFL" || sport == "NHL" {XCTAssertEqual(snapshot.facts.filter {$0.role == "reference"}.first?.sourceIDs.count,32)}
            if sport == "NBA" {XCTAssertEqual(snapshot.facts.filter {$0.role == "reference"}.first?.sourceIDs.count,30)}
            var oversized=try XCTUnwrap(object["snapshot"] as? [String:Any])
            let sources=try XCTUnwrap(oversized["sources"] as? [[String:Any]])
            oversized["sources"]=(0..<41).map {index -> [String:Any] in var value=sources[0];value["id"]="source-\(index)";return value}
            var changed=object;changed["snapshot"]=oversized
            XCTAssertThrowsError(try validate(changed))
            for role in ["reference","offense"] {
                var raw=object["snapshot"] as! [String:Any]
                var facts=raw["facts"] as! [[String:Any]]
                let index=try XCTUnwrap(facts.firstIndex {($0["role"] as? String)==role})
                facts[index]["sourceIDs"]=(0..<(role == "reference" ? 33:5)).map {"source-\($0)"}
                raw["facts"]=facts;changed=object;changed["snapshot"]=raw
                XCTAssertThrowsError(try validate(changed))
            }
            var padded=try data(object);padded.append(Data(repeating:32,count:65537-padded.count))
            XCTAssertThrowsError(try SharedMatchupReportValidator.validate(padded,event:event(object),now:fixtureNow))
        }
        let nbaWire=URL(fileURLWithPath:#filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/shared-collector-NBA.json")
        let nba=try SharedMatchupReportValidator.validate(Data(contentsOf:nbaWire),event:event(try XCTUnwrap(examples["NBA"])),now:fixtureNow).snapshot!
        let ppg=nba.facts.filter {$0.metric == "pointsPerGame"}
        let partial=SharedMatchupSnapshot(schemaVersion:1,event:nba.event,sources:nba.sources,facts:ppg)
        let report=try SharedMatchupReportValidator.render(selection:SharedMatchupSelection(observationIDs:[ppg[0].id],outlookIDs:[]),snapshot:partial)
        XCTAssertEqual(Array(report.missingContext.suffix(2)),["Verified pace data are not supplied in this snapshot.","Verified scoring efficiency data are not supplied in this snapshot."])
        XCTAssertTrue(report.missingContext[0].hasPrefix("No supported qualitative outlook"))
        XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:SharedMatchupSelection(observationIDs:[ppg[0].id],outlookIDs:["game:lower-efficiency"]),snapshot:partial))
    }
    func testPublicCollectorQueryAllowlist() throws {
        let base="https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/1/statistics"
        for query in ["", "?season=2026&seasontype=2&limit=1000", "?dates=20260901-20261012", "?stats=season&group=pitching&sportIds=1&gameType=R", "?season=%32%30%32%36"] {
            XCTAssertNoThrow(try SharedMatchupReportValidator.validateSourceURL(base+query))
        }
        for query in ["?token=secret","?season=2026&season=2025","?season=","?season","?season=2026&","?season=1999","?limit=100","?group=fielding","?gameType=P","?dates=2026-10-13","?season=2026%0A","?season=2026 ","?season=2026#fragment"] {
            XCTAssertThrowsError(try SharedMatchupReportValidator.validateSourceURL(base+query),query)
        }
    }
    func testVolumeDoesNotImplyShareOrDefensiveResistance() throws {
        for (sport,metric,unit,phrase) in [("NFL","passingAttemptsPerGame","pass attempts per game","a higher passing-volume profile"),("NFL","rushingAttemptsPerGame","rush attempts per game","a higher rushing-volume profile"),("NHL","shotsPerGame","shots per game","a higher shot-volume profile")] {
            let base = try validate(example(sport)).snapshot!
            let home=base.event.homeTeam, away=base.event.awayTeam
            func fact(_ id:String,_ team:String,_ role:String,_ value:Double)->SharedMatchupFact {
                SharedMatchupFact(id:id,team:team,role:role,metric:metric,unit:unit,period:"2026 regular season",windowStart:"2026-09-01T00:00:00.000Z",windowEnd:"2026-10-12T23:00:00.000Z",value:value,sample:12,sourceIDs:["espn-team-stats"])
            }
            for high in [true,false] {
                let snapshot=SharedMatchupSnapshot(schemaVersion:1,event:base.event,sources:base.sources,facts:[fact("own",home,"offense",40),fact("opp",away,"defense",high ? 40:20),fact("ref","league","reference",30)])
                let result=try SharedMatchupReportValidator.render(selection:SharedMatchupSelection(observationIDs:["own"],outlookIDs:["home:"+metric+(high ? ":high":":resistance")]),snapshot:snapshot).outlook[0]
                if high {XCTAssertTrue(result.text.contains(phrase))}
                else {
                    XCTAssertEqual(result.label,"Contrasting volume patterns")
                    XCTAssertEqual(result.text,"\(home) faces contrasting recorded volume against \(away): its offensive volume is at or above the league reference, while the opposing allowance is below it. These samples do not establish which pattern will persist.")
                }
                XCTAssertFalse(result.text.contains("heavy"))
            }
        }
    }
    func testExactDecimalThresholdAndTinyFiniteValuesDoNotRoundOrCrash() throws {
        let base = try validate(example()).snapshot!
        func replace(_ fact: SharedMatchupFact, _ value: Double) -> SharedMatchupFact {
            SharedMatchupFact(id:fact.id,team:fact.team,role:fact.role,metric:fact.metric,unit:fact.unit,period:fact.period,windowStart:fact.windowStart,windowEnd:fact.windowEnd,value:value,sample:fact.sample,sourceIDs:fact.sourceIDs)
        }
        for (own, reference, expected) in [(33.0,30.0,true),(32.99999999999999,30.0,false),(1.1e-200,1e-200,true)] {
            let facts = base.facts.map { replace($0,$0.role == "reference" ? reference:own) }
            let snapshot = SharedMatchupSnapshot(schemaVersion:1,event:base.event,sources:base.sources,facts:facts)
            let selection = SharedMatchupSelection(observationIDs:["home-offense"],outlookIDs:["home:passingAttemptsPerGame:high"])
            if expected { XCTAssertNoThrow(try SharedMatchupReportValidator.render(selection:selection,snapshot:snapshot)) }
            else { XCTAssertThrowsError(try SharedMatchupReportValidator.render(selection:selection,snapshot:snapshot)) }
        }
    }

    func testDuplicateKeysAndOversizedPayloadFailClosed() throws {
        let original=try example();let event=try event(original)
        var raw=String(data:try data(original),encoding:.utf8)!
        raw=raw.replacingOccurrences(of:"\"schemaVersion\":1",with:"\"schemaVersion\":1,\"schemaVersion\":1")
        XCTAssertThrowsError(try SharedMatchupReportValidator.validate(Data(raw.utf8),event:event,now:fixtureNow))
        XCTAssertThrowsError(try SharedMatchupReportValidator.validate(Data(repeating:32,count:65537),event:event,now:fixtureNow)) {
            XCTAssertEqual($0 as? SharedMatchupReportError,.oversizedPayload)
        }
    }
    func testClientUsesOnlyAuthenticatedIdentityGETAndFreshToken() async throws {
        let object=try example(), bytes=try data(object), event=try event(object), now=fixtureNow
        let client=SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{"synthetic-token"},transport:{request in
            XCTAssertEqual(request.httpMethod,"GET");XCTAssertNil(request.httpBody)
            XCTAssertEqual(request.value(forHTTPHeaderField:"Authorization"),"Bearer synthetic-token")
            XCTAssertEqual(Set(URLComponents(url:request.url!,resolvingAgainstBaseURL:false)!.queryItems!.map(\.name)),Set(["sport","eventID","homeTeam","awayTeam","startISO"]))
            return (bytes,HTTPURLResponse(url:request.url!,statusCode:200,httpVersion:nil,headerFields:nil)!)
        },now:{now})
        let result = try await client.fetch(event:event)
        XCTAssertEqual(result.status,.ready)
    }
    func testClientDoesNotRetryAuthErrorsOrAcceptRedirectedIdentity() async throws {
        let event=try event(example())
        for status in [401,503,302] {
            let client=SharedMatchupReportClient(endpoint:URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!,tokenProvider:{"synthetic-token"},transport:{request in
                (Data(),HTTPURLResponse(url:request.url!,statusCode:status,httpVersion:nil,headerFields:nil)!)
            })
            do {_ = try await client.fetch(event:event);XCTFail("Failure accepted")}
            catch {XCTAssertEqual(error as? SharedMatchupReportError,status==401 ? .authenticationRequired:.unavailable)}
        }
    }
    func testDelegateBoundsChunksAndRefusesRedirectWithoutOpeningSocket() async throws {
        let url=URL(string:"https://api.oakas.ai/plonn/v1/insights/report")!
        for oversized in [true,false] {
            let session=URLSession(configuration:.ephemeral);defer{session.invalidateAndCancel()}
            let task=session.dataTask(with:url),delegate=SharedMatchupReadDelegate(expectedURL:url)
            let response=HTTPURLResponse(url:url,statusCode:200,httpVersion:nil,headerFields:nil)!
            if oversized {
                delegate.urlSession(session,dataTask:task,didReceive:response){XCTAssertEqual($0,.allow)}
                delegate.urlSession(session,dataTask:task,didReceive:Data(repeating:32,count:65537))
            } else {
                delegate.urlSession(session,task:task,willPerformHTTPRedirection:response,newRequest:URLRequest(url:URL(string:"https://other.invalid")!)){XCTAssertNil($0)}
            }
            do {
                _ = try await withCheckedThrowingContinuation {delegate.start(session:session,request:URLRequest(url:url),continuation:$0)}
                XCTFail("Transport failure accepted")
            } catch {XCTAssertEqual(error as? SharedMatchupReportError,oversized ? .oversizedPayload:.invalidResponse)}
        }
    }
}
