import XCTest
@testable import PLONNCore

final class BuilderParlayStorageTests:XCTestCase {
    func withStorage(_ work:(BuilderParlayStorage,UserDefaults)throws->Void) throws {
        let name="BuilderParlayTests."+UUID().uuidString
        let defaults=try XCTUnwrap(UserDefaults(suiteName:name))
        defer {defaults.removePersistentDomain(forName:name)}
        try work(BuilderParlayStorage(defaults:defaults),defaults)
    }
    func slip(_ number:Int)->SavedBuilderParlay {
        SavedBuilderParlay(id:UUID(),name:"Parlay \(number)",createdAt:Date(timeIntervalSince1970:1_800_000_000),legs:[BuilderTests().leg("a\(number)","one\(number)"),BuilderTests().leg("b\(number)","two\(number)")],stake:"25")
    }
    func testStackRoundTripIsOwnerScopedAndPreservesExistingDraft() throws {
        try withStorage {storage,defaults in
            let owner=UUID(),other=UUID(),first=slip(1),second=slip(2)
            let draft=BuilderDraftStorage(defaults:defaults)
            try draft.save(first.legs)
            let collection=BuilderParlayStorage.Collection(parlays:[first,second],draftStake:"10")
            try storage.save(collection,owner:owner)
            XCTAssertEqual(try storage.load(owner:owner),collection)
            XCTAssertEqual(try storage.load(owner:other),.init())
            XCTAssertEqual(try draft.load(),first.legs)
            try storage.save(.init(parlays:[second],draftStake:"5"),owner:owner)
            XCTAssertEqual(try storage.load(owner:owner).parlays,[second])
            XCTAssertEqual(try draft.load(),first.legs)
        }
    }
    func testInvalidSaveCannotReplaceGoodParlays() throws {
        try withStorage {storage,_ in
            let owner=UUID(),first=slip(1),valid=BuilderParlayStorage.Collection(parlays:[slip(2)])
            try storage.save(valid,owner:owner)
            var duplicateLeg=first;duplicateLeg.legs.append(first.legs[0])
            for collection in [BuilderParlayStorage.Collection(version:99),.init(parlays:[first,first]),.init(parlays:[duplicateLeg]),.init(draftStake:String(repeating:"1",count:41))] {
                XCTAssertThrowsError(try storage.save(collection,owner:owner))
                XCTAssertEqual(try storage.load(owner:owner),valid)
            }
        }
    }
    func testLargeDraftAndSavedStacksRoundTripWithoutTruncatingPicks() throws {
        try withStorage {storage,defaults in
            let owner=UUID(),helper=BuilderTests()
            let picks=(0..<40).map {helper.leg("pick-\($0)","event-\($0)")}
            let draft=BuilderDraftStorage(defaults:defaults)
            try draft.save(picks)
            XCTAssertEqual(try draft.load(),picks)
            var many=(0..<25).map(slip)
            many[0].legs=picks
            let collection=BuilderParlayStorage.Collection(parlays:many,draftStake:"100")
            try storage.save(collection,owner:owner)
            XCTAssertEqual(try storage.load(owner:owner),collection)
            XCTAssertEqual(BuilderLogic.sanitized(try draft.load()).legs,picks)
        }
    }
    func testUnreadableAndFutureCollectionsAreReportedWithoutRemoval() throws {
        try withStorage {storage,defaults in
            let owner=UUID(),key="builderParlays.v1."+owner.uuidString
            for bytes in [Data("broken".utf8),Data(#"{"version":99,"parlays":[],"draftStake":""}"#.utf8)] {
                defaults.set(bytes,forKey:key)
                XCTAssertThrowsError(try storage.load(owner:owner))
                XCTAssertEqual(defaults.data(forKey:key),bytes)
            }
        }
    }
}
