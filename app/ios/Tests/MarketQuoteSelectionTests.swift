import XCTest
@testable import PLONNCore

final class MarketQuoteSelectionTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)
    private func quote(_ sport: String) -> Market {
        Market(id:"fixture",event:"event",sport:sport,kind:"prop",name:"Fixture Player",opponent:"Away",label:"Over 2.5",aliases:"",p:nil,price:-110,market:"player_points",book:"DraftKings",quotes:[],updated:stamp(now),mode:"live",line:2.5,eventState:"pre",commence:stamp(now.addingTimeInterval(3600)),side:"Over")
    }
    private func stamp(_ date: Date) -> String { ISO8601DateFormatter().string(from:date) }

    func testEveryLeagueSelectsOnlyItsPlayerAndBook() {
        for league in ["NBA","MLB","NFL","NHL"] {
            let original=quote(league)
            var otherPlayer=original;otherPlayer.name="Other Player"
            var otherLeague=original;otherLeague.sport="Other league"
            var sample=original;sample.mode="demo"
            var final=original;final.eventState="post"
            let offers=LabQuoteSelection.offers([original,otherPlayer,otherLeague,sample,final],league:league,player:" fixture player ",book:"draftkings")
            XCTAssertEqual(offers,[original],league)
            XCTAssertTrue(LabQuoteSelection.canUsePregameQuote(original,now:now),league)
            XCTAssertEqual(BuilderLogic.refreshedQuote(for:original,from:[original],now:now),original,league)
        }
    }
    func testSecondaryBookNeverInheritsPrimaryTimestamp() throws {
        var original=quote("NFL")
        original.quotes=[Quote(book:"FanDuel",price:120)]
        let selected=try XCTUnwrap(LabQuoteSelection.offers([original],league:"NFL",player:original.name,book:"FanDuel").first)
        XCTAssertEqual(selected.price,120)
        XCTAssertEqual(selected.book,"FanDuel")
        XCTAssertNil(selected.updated)
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(selected,now:now))
        XCTAssertTrue(LabQuoteSelection.offers([original],league:"NFL",player:original.name,book:"Missing book").isEmpty)
    }
    func testStaleUnknownFutureAndStartedQuotesCannotFillPregameThreshold() {
        let original=quote("NBA")
        for timestamp in [nil, "invalid", stamp(now.addingTimeInterval(-121)), stamp(now.addingTimeInterval(60))] as [String?] {
            var bad=original;bad.updated=timestamp
            XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(bad,now:now))
            XCTAssertNil(BuilderLogic.refreshedQuote(for:original,from:[bad],now:now))
        }
        var boundary=original;boundary.updated=stamp(now.addingTimeInterval(-120))
        XCTAssertTrue(LabQuoteSelection.canUsePregameQuote(boundary,now:now))
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(boundary,now:now.addingTimeInterval(1)))
        for state in ["in","post"] {
            var started=original;started.eventState=state
            XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(started,now:now))
        }
        var started=original;started.commence=stamp(now)
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(started,now:now))
        started.commence=nil
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(started,now:now))
    }
    func testIncompleteOrSampleQuoteCannotInventASelection() {
        let original=quote("NFL")
        var missing=original;missing.side=nil
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(missing,now:now))
        missing=original;missing.event=" "
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(missing,now:now))
        missing=original;missing.book=" "
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(missing,now:now))
        missing=original;missing.mode="demo"
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(missing,now:now))
        missing=original;missing.price=99
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(missing,now:now))
        missing=original;missing.line = .nan
        XCTAssertFalse(LabQuoteSelection.canUsePregameQuote(missing,now:now))
    }
    func testRefreshNeverChangesBookLineSideEventOrMarket() {
        let saved=quote("MLB")
        var fresh=saved;fresh.price=125;fresh.updated=stamp(now.addingTimeInterval(1));fresh.id="new-provider-id"
        XCTAssertEqual(BuilderLogic.refreshedQuote(for:saved,from:[fresh],now:now),fresh)
        var wrong=fresh;wrong.line=3.5
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[wrong],now:now))
        wrong=fresh;wrong.book="FanDuel"
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[wrong],now:now))
        wrong=fresh;wrong.side="Under"
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[wrong],now:now))
        wrong=fresh;wrong.event="next-day"
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[wrong],now:now))
        wrong=fresh;wrong.market="alternate_player_points"
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[wrong],now:now))
        wrong=fresh;wrong.price=99
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[wrong],now:now))
        wrong=fresh;wrong.eventState="post"
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[wrong],now:now))
    }
    func testRefreshRejectsRegressionAndConflictingLatestPrices() {
        let saved=quote("NHL")
        var older=saved;older.updated=stamp(now.addingTimeInterval(-10))
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[older],now:now))
        var first=saved;first.updated=stamp(now.addingTimeInterval(1));first.price=120
        var conflicting=first;conflicting.price=130
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[first,conflicting],now:now))
        conflicting.updated=stamp(now.addingTimeInterval(2))
        XCTAssertEqual(BuilderLogic.refreshedQuote(for:saved,from:[first,conflicting],now:now),conflicting)
        XCTAssertNil(BuilderLogic.refreshedQuote(for:saved,from:[],now:now))
    }
}
