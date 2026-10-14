import Foundation
import XCTest
@testable import PLONNCore

final class FullGameModelTests:XCTestCase {
    private var evidence:URL {
        URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("app/backend/local_lab/verification/game-integration-20260930")
    }
    private func fixture(_ league:String,_ market:String,_ model:String) throws -> ([String:Any],LabRecipe,LabCapabilities.Support) {
        let file=evidence.appendingPathComponent("\(league.lowercased())-\(market)-\(model).json")
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
        let setup=try XCTUnwrap(object["setup"] as? [String:Any])
        let result=try XCTUnwrap(object["result"] as? [String:Any])
        let stat=try XCTUnwrap(setup["stat"] as? String)
        let recipe=LabRecipe(id:UUID(),name:"\(league) \(stat)",player:setup["player"] as! String,quarterback:"",opponent:setup["opponent"] as! String,
            features:setup["features"] as! [String],created:Date(),line:setup["line"] as? Double,side:"Over",sport:league,stat:stat,
            targetKind:"Team",advanced:false,training:LabTrainingSettings(),modelType:model,book:"DraftKings",historicalEventID:setup["historical_event_id"] as? String)
        let capabilities=try JSONDecoder().decode(LabCapabilities.self,from:Data(contentsOf:evidence.appendingPathComponent("capabilities.json")))
        return (result,recipe,try XCTUnwrap(capabilities.match(league:league,stat:stat,target:"Team")))
    }
    func testAll48GenuineResultsAndPrivateRecipeRoundTrips() throws {
        for league in ["NBA","MLB","NFL","NHL"] {
            for market in ["total","spread","moneyline"] {
                for model in SavedModelType.allCases {
                    let (body,recipe,support)=try fixture(league,market,model.rawValue)
                    let version=AccountModelVersion(id:UUID(),model_id:UUID(),user_id:UUID(),version:2,name:recipe.name,configuration:recipe,created_at:"2026-09-30T00:00:00Z")
                    let reopened=try JSONDecoder().decode(AccountModelVersion.self,from:JSONEncoder().encode(version))
                    XCTAssertEqual(reopened.configuration.configurationFingerprint,recipe.configurationFingerprint)
                    let payload=try LabServiceContract.payload(reopened.configuration,support:support)
                    XCTAssertEqual(payload["targetKind"] as? String,"Team")
                    XCTAssertEqual(payload["line"] as? Double,recipe.line)
                    XCTAssertEqual(payload["historical_event_id"] as? String,recipe.historicalEventID)
                    XCTAssertEqual(payload["stat"] as? String,recipe.stat)
                    let result=try LabServiceContract.result(body,recipe:reopened.configuration)
                    XCTAssertNotNil(result.probability);XCTAssertNotNil(result.baseline)
                    XCTAssertGreaterThan(result.count ?? 0,9)
                    XCTAssertTrue(result.validation.lowercased().contains("experimental"))
                    if market == "moneyline" {XCTAssertTrue(LabServiceContract.probabilityLabel(recipe).contains("to win"))}
                    if market == "spread" {XCTAssertTrue(LabServiceContract.probabilityLabel(recipe).contains("to cover -"))}
                    if market == "moneyline" && league == "NFL" {XCTAssertNotNil(result.tie)} else {XCTAssertNil(result.tie)}
                    let probs=(body["inference"] as! [String:Any])["probabilities"] as! [String:Double]
                    XCTAssertEqual(result.probability,probs[market == "moneyline" ? "win":market == "spread" ? "cover":"over"])
                }
            }
        }
    }
    func testGameResultsRejectWrongTeamMarketEventAndExpandedFeatures() throws {
        let (body,recipe,_)=try fixture("NBA","spread","logistic_regression")
        for change in ["event","team","opponent","market","feature","probability"] {
            var wrong=body;var inference=body["inference"] as! [String:Any];var target=inference["target"] as! [String:Any]
            switch change {
            case "event":target["event_id"]="999"
            case "team":target["selected_team"]="home"
            case "opponent":target["home_team"]="BAD";target["home_team_name"]="BAD"
            case "market":wrong["market"]="moneyline"
            case "feature":inference["features"]=["recent":3.0]
            default:inference["probabilities"]=["over":0.6,"under":0.4]
            }
            inference["target"]=target;wrong["inference"]=inference
            XCTAssertThrowsError(try LabServiceContract.result(wrong,recipe:recipe),change)
        }
    }
    func testSignedAndNoLineRulesAndSeparatePushTieLabels() throws {
        XCTAssertTrue(LabServiceContract.validLine(-3.5,target:"Team",stat:"Spread"))
        XCTAssertTrue(LabServiceContract.validLine(3.5,target:"Team",stat:"Spread"))
        XCTAssertFalse(LabServiceContract.validLine(-3.5,target:"Team",stat:"Game total"))
        XCTAssertFalse(LabServiceContract.validLine(-3.5,target:"Player",stat:"Points"))
        XCTAssertFalse(LabServiceContract.validLine(1.2,target:"Team",stat:"Spread"))
        XCTAssertTrue(LabServiceContract.validLine(0,target:"Team",stat:"Moneyline"))
        XCTAssertTrue(LabServiceContract.validLine(nil,target:"Team",stat:"Moneyline"))
        XCTAssertFalse(LabServiceContract.validLine(3.5,target:"Team",stat:"Moneyline"))
        let (body,recipe,support)=try fixture("NBA","total","logistic_regression")
        var raw=try JSONSerialization.jsonObject(with:JSONEncoder().encode(recipe)) as! [String:Any]
        raw["line"]=224.0;raw["side"]="Under"
        let integer=try JSONDecoder().decode(LabRecipe.self,from:JSONSerialization.data(withJSONObject:raw))
        _=try LabServiceContract.payload(integer,support:support)
        var result=body;result["line"]=224.0;var inference=body["inference"] as! [String:Any]
        inference["threshold"]=224.0;inference["probabilities"]=["over":0.4,"under":0.5,"push":0.1];result["inference"]=inference
        let checked=try LabServiceContract.result(result,recipe:integer)
        XCTAssertEqual(checked.probability,0.5);XCTAssertEqual(checked.push,0.1);XCTAssertNil(checked.tie)
    }
    func testTeamCoverageNeverLeaksToPlayerAndMissingEventCannotRun() throws {
        let (_,recipe,support)=try fixture("NFL","moneyline","random_forest")
        var bad=recipe;bad.historicalEventID=nil
        XCTAssertThrowsError(try LabServiceContract.payload(bad,support:support))
        bad.historicalEventID="2024_17_KC_PIT"
        XCTAssertThrowsError(try LabServiceContract.payload(bad,support:support))
        let cap=try JSONDecoder().decode(LabCapabilities.self,from:Data(contentsOf:evidence.appendingPathComponent("capabilities.json")))
        XCTAssertNil(cap.match(league:"NBA",stat:"Game total",target:"Player"))
        XCTAssertNil(cap.match(league:"NBA",stat:"Points",target:"Team"))
        XCTAssertNil(cap.match(league:"NBA",stat:"Team points",target:"Team"))
    }
}
