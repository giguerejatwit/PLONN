import XCTest
@testable import PLONNCore

final class ModelSetupTests: XCTestCase {
    private func recipe(_ type:String = "logistic_regression") -> LabRecipe {
        LabRecipe(id:UUID(),name:"My setup",player:"Fixture",quarterback:"",opponent:"Opponent",features:["recent","season"],created:Date(),line:5.5,side:"Over",sport:"MLB",stat:"Pitcher strikeouts",targetKind:"Player",advanced:false,training:LabTrainingSettings(),modelType:type)
    }
    func testAllModelTypesRoundTripWithoutChangingFeatures() throws {
        for type in SavedModelType.allCases {
            let original=recipe(type.rawValue)
            let decoded=try JSONDecoder().decode(LabRecipe.self,from:JSONEncoder().encode(original))
            XCTAssertEqual(decoded.resolvedModelType,type.rawValue)
            XCTAssertEqual(decoded.features,original.features)
            XCTAssertEqual(decoded.training,original.training)
            XCTAssertEqual(decoded.configurationFingerprint,original.configurationFingerprint)
        }
    }
    func testLegacyDeviceRecipeDefaultsToLogistic() throws {
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(recipe())) as? [String:Any])
        object.removeValue(forKey:"modelType")
        let decoded=try JSONDecoder().decode(LabRecipe.self,from:JSONSerialization.data(withJSONObject:object))
        XCTAssertEqual(decoded.resolvedModelType,"logistic_regression")
    }
    func testConfigurationIdentityIncludesTypeAndTrainingNotDeviceTime() {
        XCTAssertEqual(recipe().configurationFingerprint,recipe().configurationFingerprint)
        XCTAssertNotEqual(recipe().configurationFingerprint,recipe("random_forest").configurationFingerprint)
    }
    func testCapabilityGateDoesNotInventOptionalAdapters() throws {
        let old=try JSONDecoder().decode(LabCapabilities.self,from:Data(#"{"supported":[{"league":"MLB","stat":"Pitcher strikeouts","features":["recent"]}]}"#.utf8))
        let entry=try XCTUnwrap(old.match(league:"MLB",stat:"Pitcher strikeouts",target:"Player"))
        XCTAssertTrue(entry.supports("logistic_regression"))
        XCTAssertFalse(entry.supports("neural_network"))
        XCTAssertNil(old.match(league:"NBA",stat:"Points",target:"Player"))
        XCTAssertNil(old.match(league:"MLB",stat:"Pitcher strikeouts",target:"Team"))
        let updated=try JSONDecoder().decode(LabCapabilities.self,from:Data(#"{"supported":[{"league":"MLB","stat":"Pitcher strikeouts","features":["recent"],"model_types":["logistic_regression","decision_tree","random_forest","neural_network"]}]}"#.utf8))
        for type in SavedModelType.allCases {XCTAssertTrue(updated.supported[0].supports(type.rawValue))}
    }
    func testRunIdentityRejectsDifferentAdapterOrFeatureSet() {
        let result:[String:Any] = ["model":["model_type":"random_forest"],"inference":["features":["recent":3.0,"season":4.0]]]
        XCTAssertTrue(labRunMatches(result,modelType:"random_forest",features:["recent","season"]))
        XCTAssertFalse(labRunMatches(result,modelType:"logistic_regression",features:["recent","season"]))
        XCTAssertFalse(labRunMatches(result,modelType:"random_forest",features:["recent"]))
        XCTAssertFalse(labRunMatches([:],modelType:"random_forest",features:["recent"]))
    }

    func testEveryCatalogEntryRequiresMatchingBackendEvidence() throws {
        let empty=LabCapabilities(supported:[])
        let fixture=try JSONDecoder().decode(LabCapabilities.self,from:Data(#"{"supported":[{"league":"MLB","stat":"Pitcher strikeouts","features":["recent"],"model_types":["logistic_regression"]}]}"#.utf8))
        for league in LabPropCatalog.leagues {
            for target in LabPropCatalog.targets {
                let options=LabPropCatalog.options(league:league,target:target)
                XCTAssertFalse(options.isEmpty)
                XCTAssertEqual(options.count,Set(options).count)
                for label in options {
                    XCTAssertNil(empty.match(league:league,stat:label,target:target),"Tile alone must not enable training: \(league)/\(target)/\(label)")
                    let matching=fixture.match(league:league,stat:label,target:target)
                    XCTAssertEqual(matching != nil,league == "MLB" && target == "Player" && label == "Pitcher strikeouts")
                    if let matching {XCTAssertFalse(matching.supports("random_forest"))}
                }
            }
        }
        XCTAssertTrue(LabPropCatalog.options(league:"Unknown",target:"Player").isEmpty)
        XCTAssertTrue(LabPropCatalog.options(league:"NBA",target:"Unknown").isEmpty)
        XCTAssertNil(LabPropCatalog.serviceStat(league:"MLB",target:"Player",label:"Custom stat"))
    }

    func testCoverageManifestCannotDriftFromAdvertisedCatalog() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try Data(contentsOf:root.appendingPathComponent("docs/AI_LAB_PROP_CATALOG.json"))
        let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any])
        let rows=try XCTUnwrap(manifest["rows"] as? [[String:Any]])
        XCTAssertEqual(manifest["model_types"] as? [String],SavedModelType.allCases.map(\.rawValue))
        var expected=Set<String>()
        for league in LabPropCatalog.leagues {for target in LabPropCatalog.targets {
            for label in LabPropCatalog.options(league:league,target:target) {expected.insert(league+"|"+target+"|"+label)}
        }}
        var actual=Set<String>()
        for row in rows {
            let league=try XCTUnwrap(row["league"] as? String)
            let target=try XCTUnwrap(row["targetKind"] as? String)
            let label=try XCTUnwrap(row["uiLabel"] as? String)
            XCTAssertTrue(actual.insert(league+"|"+target+"|"+label).inserted)
            XCTAssertEqual(row["serviceStat"] as? String,LabPropCatalog.serviceStat(league:league,target:target,label:label))
            XCTAssertEqual(row["isCustom"] as? Bool,label == "Custom stat")
        }
        XCTAssertEqual(actual,expected,"Every advertised choice must be present in the real-data coverage handoff.")
    }

}
