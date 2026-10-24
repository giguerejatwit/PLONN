import XCTest
@testable import PLONNCore

final class LabServiceContractTests:XCTestCase {
    func testWorkerFailureExplainsRetryWithoutBlamingValidSelection() {
        let body: [String: Any] = ["reason": "training_worker_failed", "message": "private server details"]
        XCTAssertEqual(PlayerMatchupSetup.failure(body), "The model service stopped before finishing. Try again with the same selection.")
    }
    func testRetainedInferenceRemainsAvailableWhenTrainingIsDisabled() throws {
        var catalog=try capabilities()
        catalog.enabled=false;catalog.service_enabled=false;catalog.retained_artifacts=true
        XCTAssertNil(catalog.match(league:"MLB",stat:"Pitcher strikeouts",target:"Player"))
        XCTAssertNotNil(catalog.matchRetained(league:"MLB",stat:"Pitcher strikeouts",target:"Player"))
        XCTAssertNil(catalog.matchRetained(league:"MLB",stat:"Unknown target",target:"Player"))
        catalog.retained_artifacts=false
        XCTAssertNil(catalog.matchRetained(league:"MLB",stat:"Pitcher strikeouts",target:"Player"))
    }
    private func recipe(_ type:String="logistic_regression") -> LabRecipe {
        LabRecipe(id:UUID(),name:"Pitcher setup",player:"Fixture pitcher",quarterback:"",opponent:"Fixture team",features:["recent","season","workload"],created:Date(),line:5.5,side:"Over",sport:"MLB",stat:"Pitcher strikeouts",targetKind:"Player",advanced:false,training:LabTrainingSettings(),modelType:type,book:"DraftKings")
    }
    private func capabilities() throws -> LabCapabilities {
        try JSONDecoder().decode(LabCapabilities.self,from:Data(#"{"enabled":true,"supported":[{"league":"MLB","stat":"Pitcher strikeouts","features":["recent","season","workload","homeAway","rest"],"model_types":["logistic_regression","decision_tree","random_forest","neural_network"],"evaluation_mode":"fixed_threshold_retrospective"}]}"#.utf8))
    }
    private func response(_ type:String="logistic_regression") -> [String:Any] {
        ["model":["model_type":type],"target_prop":"Pitcher strikeouts","requested_book":"DraftKings",
         "inference":["status":"ok","threshold":5.5,"features":["recent":4.0,"season":5.0,"workload":6.0],"probabilities":["over":0.6,"under":0.4]],
         "validation":["status":"ok","sample_size":20,"metrics":["multiclass_brier":0.4],"label":"Retrospective fixed-threshold validation"]]
    }
    func testAllAdaptersSaveReopenAndSendExactConfiguration() throws {
        let support=try XCTUnwrap(capabilities().match(league:"MLB",stat:"Pitcher strikeouts",target:"Player"))
        for type in SavedModelType.allCases {
            let original=recipe(type.rawValue)
            let reopened=try JSONDecoder().decode(LabRecipe.self,from:JSONEncoder().encode(original))
            let payload=try LabServiceContract.payload(reopened,support:support)
            XCTAssertEqual(payload["model_type"] as? String,type.rawValue)
            XCTAssertEqual(payload["features"] as? [String],original.features.sorted())
            XCTAssertEqual(payload["evaluation_mode"] as? String,"fixed_threshold_retrospective")
            XCTAssertEqual(payload["book"] as? String,"DraftKings")
            XCTAssertEqual(payload["line"] as? Double,5.5)
            XCTAssertEqual(reopened.configurationFingerprint,original.configurationFingerprint)
            let result=try LabServiceContract.result(response(type.rawValue),recipe:reopened)
            XCTAssertEqual(result.probability,0.6);XCTAssertEqual(result.count,20);XCTAssertEqual(result.brier,0.4)
        }
    }
    func testPrivateVersionEnvelopeReopensExactSetupWithoutInventingArtifact() throws {
        let setup=recipe("random_forest")
        let owner=UUID(),model=UUID()
        let version=AccountModelVersion(id:UUID(),model_id:model,user_id:owner,version:3,name:setup.name,configuration:setup,created_at:"2026-09-28T12:00:00Z")
        let encoded=try JSONEncoder().encode(version)
        let reopened=try JSONDecoder().decode(AccountModelVersion.self,from:encoded)
        XCTAssertEqual(reopened.user_id,owner);XCTAssertEqual(reopened.model_id,model);XCTAssertEqual(reopened.version,3)
        XCTAssertEqual(reopened.configuration.configurationFingerprint,setup.configurationFingerprint)
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:encoded) as? [String:Any])
        XCTAssertNil(object["artifact"]);XCTAssertNil(object["performance"])
        let support=try capabilities().supported[0]
        XCTAssertEqual(try LabServiceContract.payload(reopened.configuration,support:support)["model_type"] as? String,"random_forest")
    }
    func testNFLFullCatalogRequiresExplicitHistoricalTarget() throws {
        let data=Data(#"{"supported":[],"supported_data_targets":[{"league":"NFL","stat":"Pass Yds","features":["recent","season","workload"],"model_types":["logistic_regression","decision_tree","random_forest","neural_network"],"evaluation_mode":"fixed_threshold_retrospective","historical_event_required":true},{"league":"NFL","stat":"Rec Yds","features":["recent","season","workload"],"model_types":["logistic_regression","decision_tree","random_forest","neural_network"],"evaluation_mode":"fixed_threshold_retrospective","historical_event_required":true}]}"#.utf8)
        let catalog=try JSONDecoder().decode(LabCapabilities.self,from:data)
        for stat in ["Pass Yds","Rec Yds"] {
            let support=try XCTUnwrap(catalog.match(league:"NFL",stat:stat,target:"Player"))
            for type in SavedModelType.allCases {
                var setup=LabRecipe(id:UUID(),name:"Historical fixture",player:"Fixture",quarterback:"",opponent:"Fixture opponent",features:["recent","season","workload"],created:Date(),line:5.5,side:"Over",sport:"NFL",stat:stat,targetKind:"Player",advanced:false,training:nil,modelType:type.rawValue,book:"DraftKings")
                XCTAssertThrowsError(try LabServiceContract.payload(setup,support:support))
                setup.historicalEventID="fixture-event"
                let restored=try JSONDecoder().decode(LabRecipe.self,from:JSONEncoder().encode(setup))
                XCTAssertEqual(try LabServiceContract.payload(restored,support:support)["historical_event_id"] as? String,"fixture-event")
                var body=response(type.rawValue);body["target_prop"]=stat;body["live_prediction"]=false
                var inference=try XCTUnwrap(body["inference"] as? [String:Any]);inference["target"]=["event_id":"fixture-event"];body["inference"]=inference
                XCTAssertEqual(try LabServiceContract.result(body,recipe:setup).probability,0.6)
                inference["target"]=["event_id":"other-event"];body["inference"]=inference
                XCTAssertThrowsError(try LabServiceContract.result(body,recipe:setup))
            }
        }
        XCTAssertNil(catalog.match(league:"NFL",stat:"Rush Yds",target:"Player"))
    }
    func testUnsupportedFeaturesAndUnselectedBookCannotRun() throws {
        let support=try capabilities().supported[0]
        XCTAssertEqual(LabServiceContract.defaultFeatures(support),Set(recipe().features))
        var invalid=recipe();invalid.book="Ask each time"
        XCTAssertThrowsError(try LabServiceContract.payload(invalid,support:support))
        let unsupported=LabRecipe(id:UUID(),name:"Bad",player:"Fixture",quarterback:"",opponent:"Team",features:["opponent"],created:Date(),line:5.5,side:"Over",sport:"MLB",stat:"Pitcher strikeouts",targetKind:"Player",advanced:false,training:nil,book:"DraftKings")
        XCTAssertThrowsError(try LabServiceContract.payload(unsupported,support:support))
        var disabled=try capabilities();disabled.enabled=false
        XCTAssertNil(disabled.match(league:"MLB",stat:"Pitcher strikeouts",target:"Player"))
    }
    func testDifferentTargetAdapterAndFeaturesWithheld() throws {
        for key in ["target_prop","requested_book"] {
            var wrong=response();wrong[key]="Different"
            XCTAssertThrowsError(try LabServiceContract.result(wrong,recipe:recipe()))
        }
        XCTAssertThrowsError(try LabServiceContract.result(response("random_forest"),recipe:recipe()))
        var wrong=response();var inference=try XCTUnwrap(wrong["inference"] as? [String:Any])
        inference["threshold"]=6.5;wrong["inference"]=inference
        XCTAssertThrowsError(try LabServiceContract.result(wrong,recipe:recipe()))
        inference["threshold"]=5.5;inference["features"]=["recent":2.0];wrong["inference"]=inference
        XCTAssertThrowsError(try LabServiceContract.result(wrong,recipe:recipe()))
    }
    func testMalformedProbabilityAndValidationDoNotDisplaySuccess() throws {
        for probabilities in [["over":0.8,"under":0.8],["over":Double.nan,"under":0.2],["over":0.5],["over":0.5,"under":0.5,"push":0.1]] {
            var body=response();var inference=try XCTUnwrap(body["inference"] as? [String:Any]);inference["probabilities"]=probabilities;body["inference"]=inference
            XCTAssertThrowsError(try LabServiceContract.result(body,recipe:recipe()))
        }
        var body=response();body["validation"]=["status":"ok","sample_size":0,"metrics":["multiclass_brier":0.4]]
        XCTAssertThrowsError(try LabServiceContract.result(body,recipe:recipe()))
        body["validation"]=["status":"unavailable","reason":"insufficient_holdout_rows"]
        let result=try LabServiceContract.result(body,recipe:recipe())
        XCTAssertEqual(result.probability,0.6);XCTAssertNil(result.count)
        XCTAssertTrue(result.validation.contains("insufficient_holdout_rows"))
    }
}
