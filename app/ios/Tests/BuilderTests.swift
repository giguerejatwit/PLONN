import XCTest
@testable import PLONNCore

final class BuilderTests:XCTestCase {
    let now=Date(timeIntervalSince1970:1_800_000_000)
    func leg(_ id:String,_ event:String,price:Double = -110,book:String = "Fixture book") -> Market {
        Market(id:id,event:event,sport:"NBA",kind:"prop",name:"Fixture player "+id,opponent:"Fixture opponent",label:"Over 20.5 points",aliases:"",p:nil,price:price,market:"player_points",book:book,quotes:[],updated:ISO8601DateFormatter().string(from:now),mode:"live",line:20.5,eventLabel:"Fixture event "+event,side:"Over")
    }
    func testPriceArithmeticIsNotModelProbability() throws {
        var a=leg("a","first",price:100);a.p=0.1
        var b=leg("b","second",price:100);b.p=0.9
        let value=try XCTUnwrap(BuilderLogic.assess([a,b],now:now).estimate)
        XCTAssertEqual(value.decimal,4,accuracy:1e-12)
        XCTAssertEqual(value.american,300,accuracy:1e-12)
        XCTAssertEqual(value.priceImpliedProbability,0.25,accuracy:1e-12)
        XCTAssertNotEqual(value.priceImpliedProbability,0.1*0.9)
    }
    func testNegativeAmericanCombinedPriceAndInvalidOdds() throws {
        let value=try XCTUnwrap(BuilderLogic.assess([leg("a","first",price:-1000),leg("b","second",price:-1000)],now:now).estimate)
        XCTAssertEqual(value.decimal,1.21,accuracy:1e-12)
        XCTAssertEqual(value.american,-100/0.21,accuracy:1e-9)
        for invalid in [0,99,-99,Double.nan,Double.infinity] {
            XCTAssertNil(BuilderLogic.decimalOdds(invalid))
            XCTAssertNil(BuilderLogic.assess([leg("a","first",price:invalid),leg("b","second")],now:now).estimate)
        }
        XCTAssertNil(BuilderLogic.assess((0..<8).map{leg("\($0)","\($0)",price:1e100)},now:now).estimate)
    }
    func testSameGameAndUnknownIdentityNeverMultiply() {
        XCTAssertNil(BuilderLogic.assess([leg("a","same"),leg("b","same")],now:now).estimate)
        XCTAssertTrue(BuilderLogic.assess([leg("a","same"),leg("b","same")],now:now).blockingReason?.contains("real sportsbook quote") == true)
        XCTAssertNil(BuilderLogic.assess([leg("a"," "),leg("b","other")],now:now).estimate)
        var a=leg("a","provider-a"),b=leg("b","provider-b")
        a.eventLabel="Home at Away";b.eventLabel="Home at Away"
        a.commence=ISO8601DateFormatter().string(from:now);b.commence=a.commence
        XCTAssertNil(BuilderLogic.assess([a,b],now:now).estimate)
        b.commence=ISO8601DateFormatter().string(from:now.addingTimeInterval(86400))
        XCTAssertNotNil(BuilderLogic.assess([a,b],now:now).estimate)
    }
    func testMixedBooksAndFreshnessStayVisible() {
        var a=leg("a","one"),b=leg("b","two",book:"Other fixture book")
        a.updated=ISO8601DateFormatter().string(from:now.addingTimeInterval(-121));b.updated=nil
        let result=BuilderLogic.assess([a,b],now:now)
        XCTAssertNotNil(result.estimate)
        XCTAssertTrue(result.mixedBooks)
        XCTAssertTrue(result.warnings.contains{$0.contains("Mixed sportsbooks")})
        XCTAssertTrue(result.warnings.contains{$0.contains("two minutes")})
        XCTAssertTrue(result.warnings.contains{$0.contains("freshness")})
        XCTAssertEqual(BuilderLogic.quoteAge(a,now:now),.stale)
        XCTAssertEqual(BuilderLogic.quoteAge(b,now:now),.missing)
        b.updated=ISO8601DateFormatter().string(from:now.addingTimeInterval(60))
        XCTAssertEqual(BuilderLogic.quoteAge(b,now:now),.future)
        b.updated=ISO8601DateFormatter().string(from:now.addingTimeInterval(-120))
        XCTAssertEqual(BuilderLogic.quoteAge(b,now:now),.recent)
    }
    func testDuplicateOutcomeAcrossBooksIsRejected() {
        let a=leg("a","same")
        var copy=a;copy.id="different-provider-id";copy.book="Other book";copy.price=150
        XCTAssertNotNil(BuilderLogic.rejection(for:copy,existing:[a]))
        XCTAssertEqual(BuilderLogic.sanitized([a,a,copy]).legs.count,1)
    }
    func testConflictingOverUnderAndAlternativeLines() {
        let a=leg("a","same")
        var b=a;b.id="b";b.side="Under"
        XCTAssertNotNil(BuilderLogic.rejection(for:b,existing:[a]))
        b.line=19.5
        XCTAssertNotNil(BuilderLogic.rejection(for:b,existing:[a]))
        b.line=23.5 // A potential middle is not impossible, but still needs a real SGP quote.
        XCTAssertNil(BuilderLogic.rejection(for:b,existing:[a]))
        XCTAssertNil(BuilderLogic.assess([a,b],now:now).estimate)
        b.side="Over"
        XCTAssertNotNil(BuilderLogic.rejection(for:b,existing:[a]))
    }
    func testOpposingWinnersAndSpreads() {
        var a=leg("a","same"),b=leg("b","same")
        a.kind="team";b.kind="team";a.market="h2h";b.market="h2h";a.line=nil;b.line=nil
        a.side="Home";b.side="Away"
        XCTAssertNotNil(BuilderLogic.rejection(for:b,existing:[a]))
        a.market="spreads";b.market="spreads";a.line = -3.5;b.line=3.5
        XCTAssertNotNil(BuilderLogic.rejection(for:b,existing:[a]))
        a.line=5;b.line=5
        XCTAssertNil(BuilderLogic.rejection(for:b,existing:[a]))
        XCTAssertNil(BuilderLogic.assess([a,b],now:now).estimate)
    }
    func testSampleMixCapacityAndFinishedEvents() {
        var sample=leg("sample","s");sample.mode="demo"
        XCTAssertEqual(BuilderLogic.quoteAge(sample,now:now),.sample)
        XCTAssertNotNil(BuilderLogic.rejection(for:sample,existing:[leg("a","a")]))
        let nine=(0..<9).map{leg("\($0)","event\($0)")}
        XCTAssertEqual(BuilderLogic.sanitized(nine).legs.count,8)
        XCTAssertNil(BuilderLogic.assess(nine,now:now).estimate)
        var final=leg("done","done");final.eventState="post"
        XCTAssertNil(BuilderLogic.assess([final,leg("a","a")],now:now).estimate)
        XCTAssertNil(BuilderLogic.assess([],now:now).estimate)
        XCTAssertNil(BuilderLogic.assess([leg("a","a")],now:now).estimate)
    }
    func testLocalDraftRoundTripRemoveAndClear() throws {
        let suite="BuilderTests."+UUID().uuidString
        let defaults=try XCTUnwrap(UserDefaults(suiteName:suite))
        defer{defaults.removePersistentDomain(forName:suite)}
        let storage=BuilderDraftStorage(defaults:defaults)
        XCTAssertEqual(try storage.load(),[])
        let a=leg("a","a"),b=leg("b","b")
        try storage.save([a,b])
        let restored=BuilderDraftStorage(defaults:try XCTUnwrap(UserDefaults(suiteName:suite)))
        XCTAssertEqual(try restored.load(),[a,b])
        try restored.save([b]);XCTAssertEqual(try storage.load(),[b])
        storage.clear();XCTAssertEqual(try restored.load(),[])
    }
    func testFailedEncodingDoesNotEraseDraftAndCorruptionIsReported() throws {
        let suite="BuilderTests."+UUID().uuidString
        let defaults=try XCTUnwrap(UserDefaults(suiteName:suite))
        defer{defaults.removePersistentDomain(forName:suite)}
        let storage=BuilderDraftStorage(defaults:defaults)
        let a=leg("a","a");try storage.save([a])
        var invalid=a;invalid.price = .nan
        XCTAssertThrowsError(try storage.save([invalid]));XCTAssertEqual(try storage.load(),[a])
        defaults.set(Data("broken".utf8),forKey:BuilderDraftStorage.key)
        XCTAssertThrowsError(try storage.load())
        defaults.set(Data(#"{"version":99,"legs":[]}"#.utf8),forKey:BuilderDraftStorage.key)
        XCTAssertThrowsError(try storage.load())
    }
}
