import XCTest
@testable import PLONNCore
final class NationalTeamTests:XCTestCase {
    let setup=NationalTeamSetup(home:"France",away:"Argentina",cutoff:"2026-06-01",neutral:true,modelType:"logistic_regression")
    func testExactPayloadAndInvalidTargets() throws {
        let body=try setup.payload()
        XCTAssertEqual(Set(body.keys),Set(["sport","context","target_kind","home","away","cutoff","neutral","evaluation_mode","model_type"]))
        XCTAssertEqual(body["neutral"] as? Bool,true)
        for type in SavedModelType.allCases {var copy=setup;copy.modelType=type.rawValue;XCTAssertNoThrow(try copy.payload())}
        var copy=setup;copy.away=" france ";XCTAssertThrowsError(try copy.payload())
        copy=setup;copy.cutoff="2099-01-01";XCTAssertThrowsError(try copy.payload())
        copy.cutoff="2026-02-30";XCTAssertThrowsError(try copy.payload())
    }
    func testResponseIdentityAndProbabilityGuards() throws {
        var body:[String:Any] = ["status":"ok","evaluation_mode":NationalTeamSetup.mode,"model":["model_type":"logistic_regression","adapter":"soccer_national_team_elo_v1"],"inference":["status":"ok","live_prediction":false,"target":["home":"France","away":"Argentina","cutoff":"2026-06-01","neutral":true],"probabilities":["home_win":0.3,"draw":0.2,"away_win":0.5]],"validation":["status":"ok","sample_size":69,"metrics":["multiclass_brier":0.53,"baseline":["status":"ok","multiclass_brier":0.63]]]]
        let parsed=try setup.result(body);XCTAssertEqual(parsed.count,69);XCTAssertEqual(parsed.baseline,0.63)
        var other=setup;other.neutral=false;XCTAssertThrowsError(try other.result(body))
        other=setup;other.modelType="random_forest";XCTAssertThrowsError(try other.result(body))
        var inference=body["inference"] as! [String:Any];inference["probabilities"]=["home_win":0.8,"draw":0.2,"away_win":0.5];body["inference"]=inference
        XCTAssertThrowsError(try setup.result(body))
    }
}
