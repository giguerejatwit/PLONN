import XCTest
@testable import PLONNCore

final class LabCreationPolicyTests: XCTestCase {
    private func catalog(idempotency:Bool?=true,enabled:Bool=true) -> LabCapabilities {
        let support=LabCapabilities.Support(league:"NFL",stat:"Moneyline",features:["recent","season","homeAway","rest"],model_types:["logistic_regression","random_forest"],evaluation_mode:"fixed_threshold_retrospective",historical_event_required:true,target_kind:"team",market:"moneyline")
        return LabCapabilities(supported:[],supported_data_targets:[support],enabled:enabled,service_enabled:enabled,retained_artifacts:false,idempotency:idempotency)
    }
    func testUnavailableTrainingIsExplainedBeforeSubmission() {
        XCTAssertEqual(LabCreationPolicy.availability(nil,loading:true,support:nil),.checking)
        XCTAssertEqual(LabCreationPolicy.availability(nil,loading:false,support:nil),.offline)
        let paused=catalog(enabled:false)
        XCTAssertEqual(LabCreationPolicy.availability(paused,loading:false,support:nil),.paused)
        let old=catalog(idempotency:nil)
        XCTAssertEqual(LabCreationPolicy.availability(old,loading:false,support:old.supported_data_targets?.first),.needsUpdate)
        let connected=catalog()
        XCTAssertEqual(LabCreationPolicy.availability(connected,loading:false,support:nil),.unsupported)
        XCTAssertEqual(LabCreationPolicy.availability(connected,loading:false,support:connected.supported_data_targets?.first),.ready)
    }
    func testPlayerCatalogRemainsVisibleWhenTestingCoverageIsLimited() {
        XCTAssertEqual(LabCreationPolicy.questions(league:"NFL",target:"Team",capabilities:catalog()),["Moneyline"])
        for league in ["NBA","NFL","MLB","NHL"] {
            let expected=LabPropCatalog.options(league:league,target:"Player").filter{$0 != "Custom stat"}
            XCTAssertEqual(LabCreationPolicy.questions(league:league,target:"Player",capabilities:catalog()),expected)
            XCTAssertEqual(LabCreationPolicy.questions(league:league,target:"Player",capabilities:nil),expected)
            XCTAssertEqual(LabCreationPolicy.questions(league:league,target:"Player",capabilities:catalog(enabled:false)),expected)
        }
        XCTAssertEqual(LabCreationPolicy.questions(league:"NBA",target:"Team",capabilities:nil),["Moneyline","Spread","Game total"])
        XCTAssertEqual(LabCreationPolicy.questions(league:"NFL",target:"Team",capabilities:nil),["Spread","Game total","Moneyline"])
        XCTAssertFalse(LabCreationPolicy.questions(league:"NBA",target:"Player",capabilities:nil).contains("Custom stat"))
    }
    func testDefaultsOnlyUseConnectedFeaturesAndModels() {
        let support=catalog().supported_data_targets!.first!
        XCTAssertEqual(LabCreationPolicy.recommendedFeatures(support,fullGame:true),["recent","season","homeAway"])
        XCTAssertEqual(LabCreationPolicy.recommendedModel(support),"logistic_regression")
        let limited=LabCapabilities.Support(league:"NBA",stat:"Points",features:["rest"],model_types:["random_forest"],evaluation_mode:"fixed_threshold_retrospective")
        XCTAssertEqual(LabCreationPolicy.recommendedFeatures(limited,fullGame:false),["rest"])
        XCTAssertEqual(LabCreationPolicy.recommendedModel(limited),"random_forest")
    }
    func testGroupedCatalogIsCompleteAndSearchableAcrossSports() {
        let counts=["NBA":24,"NFL":33,"MLB":20,"NHL":13]
        for (league,count) in counts {
            let props=LabPropCatalog.playerGroups(league:league).flatMap(\.props)
            XCTAssertEqual(props.count,count)
            XCTAssertEqual(Set(props).count,count)
            XCTAssertEqual(Set(props),Set(LabPropCatalog.options(league:league,target:"Player").filter{$0 != "Custom stat"}))
        }
        XCTAssertTrue(LabPropCatalog.playerGroups(league:"NFL",search:"receiving").flatMap(\.props).contains("Rec Yds"))
        XCTAssertTrue(LabPropCatalog.playerGroups(league:"NBA",search:"combinations").flatMap(\.props).contains("Pts + Reb + Ast"))
        XCTAssertTrue(LabPropCatalog.playerGroups(league:"MLB",search:"pitch").flatMap(\.props).contains("Pitcher win"))
        XCTAssertEqual(LabPropCatalog.playerGroups(league:"NHL",search:"power play").flatMap(\.props),["Power play points"])
        XCTAssertTrue(LabPropCatalog.playerGroups(league:"NBA",search:"no-such-prop").isEmpty)
    }
    func testCatalogPresenceNeverClaimsTestingSupport() {
        let support=LabCapabilities.Support(league:"NBA",stat:"Points",features:["recent"],model_types:["logistic_regression"],evaluation_mode:"fixed_threshold_retrospective")
        let caps=LabCapabilities(supported:[],supported_data_targets:[support],enabled:true,service_enabled:true,idempotency:true)
        XCTAssertEqual(LabPropCatalog.testingStatus(league:"NBA",stat:"Points",capabilities:caps),"Testing available")
        XCTAssertEqual(LabPropCatalog.testingStatus(league:"NBA",stat:"Rebounds",capabilities:caps),"Save setup only")
        XCTAssertNil(caps.match(league:"NBA",stat:"Rebounds",target:"Player"))
        XCTAssertEqual(LabPropCatalog.testingStatus(league:"NBA",stat:"Points",capabilities:nil),"Availability unverified")
        XCTAssertEqual(LabPropCatalog.testingStatus(league:"NBA",stat:"Fantasy points",capabilities:caps),"Save setup only")
        XCTAssertNil(LabPropCatalog.configurationLimitation("Method of first basket"))
    }
    func testYesNoPropsUseBinaryOutcomesWithoutAnArbitraryLine() throws {
        for prop in ["Double-double","Anytime touchdown scorer","Pitcher win","First goal scorer"] {
            XCTAssertTrue(LabPropCatalog.isYesNo(prop))
            XCTAssertTrue(LabServiceContract.validLine(0.5,target:"Player",stat:prop))
            XCTAssertFalse(LabServiceContract.validLine(3.5,target:"Player",stat:prop))
        }
        XCTAssertFalse(LabPropCatalog.isYesNo("Touchdowns"))
        XCTAssertFalse(LabPropCatalog.isYesNo("Goals"))
    }
    func testFantasyAndFirstScoreOptionsRequireExplicitSavedRules() throws {
        XCTAssertFalse(LabPropCatalog.validOptions(stat:"Fantasy points",options:[:]))
        XCTAssertTrue(LabPropCatalog.validOptions(stat:"Fantasy points",options:["scoring_preset":"draftkings_classic"]))
        XCTAssertTrue(LabPropCatalog.validOptions(stat:"Batter fantasy points",options:["scoring_preset":"fanduel_classic"]))
        XCTAssertFalse(LabPropCatalog.validOptions(stat:"Points",options:["scoring_preset":"fanduel_classic"]))
        XCTAssertTrue(LabPropCatalog.isYesNo("Method of first basket"))
        XCTAssertTrue(LabPropCatalog.validOptions(stat:"Method of first basket",options:["first_basket_method":"free_throw"]))
        var recipe=LabRecipe(id:UUID(),name:"Fantasy test",player:"Test Player",quarterback:"",opponent:"TEST",features:["recent"],created:Date(),line:40.55,side:"Over",sport:"NBA",stat:"Fantasy points",targetKind:"Player",advanced:false,training:nil,book:"Research",historicalEventID:"123",propOptions:["scoring_preset":"draftkings_classic"])
        let support=LabCapabilities.Support(league:"NBA",stat:"Fantasy points",features:["recent"],model_types:["logistic_regression"],evaluation_mode:"fixed_threshold_retrospective",historical_event_required:true,historical_event_source:"espn")
        let before=recipe.configurationFingerprint
        let roundtrip=try JSONDecoder().decode(LabRecipe.self,from:JSONEncoder().encode(recipe))
        XCTAssertEqual(roundtrip.propOptions,recipe.propOptions)
        XCTAssertEqual(try LabServiceContract.payload(recipe,support:support)["prop_options"] as? [String:String],recipe.propOptions)
        recipe.propOptions=["scoring_preset":"fanduel_classic"]
        XCTAssertNotEqual(before,recipe.configurationFingerprint)
        recipe.propOptions=nil
        XCTAssertThrowsError(try LabServiceContract.payload(recipe,support:support))
        XCTAssertEqual(LabHistoricalGames.Source.source(league:"NFL",fullGame:false,propSource:"espn"),.espn)
        XCTAssertTrue(LabServiceContract.validLine(0.75,target:"Player",stat:"Sacks"))
    }
    func testResearchSetupDoesNotRequireInventingASportsbookQuote() throws {
        let support=catalog().supported_data_targets!.first!
        let recipe=LabRecipe(id:UUID(),name:"Team to win",player:"Denver Broncos",quarterback:"",opponent:"Seattle Seahawks",features:["recent","season","homeAway"],created:Date(),line:0,side:"Over",sport:"NFL",stat:"Moneyline",targetKind:"Team",advanced:false,training:nil,book:"Research",historicalEventID:"401671627")
        let body=try LabServiceContract.payload(recipe,support:support)
        XCTAssertEqual(body["book"] as? String,"Research")
        XCTAssertEqual(body["historical_event_id"] as? String,"401671627")
        XCTAssertEqual(body["evaluation_mode"] as? String,"fixed_threshold_retrospective")
    }
    func testInsufficientHistoryGivesAnActionableExplanation() {
        XCTAssertEqual(LabServiceContract.userFacingMessage(["training":["reason":"insufficient_game_training_class_coverage_tie"]]),"There aren’t enough historical ties to test this NFL win model. Try a spread or game total.")
        XCTAssertEqual(LabServiceContract.userFacingMessage(["reason":"insufficient_game_training_class_coverage_push"]),"There aren’t enough varied historical results to test this line. Try another line or matchup.")
    }
}
