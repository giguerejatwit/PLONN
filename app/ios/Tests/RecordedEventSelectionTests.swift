import Foundation
import XCTest
@testable import PLONNCore
final class RecordedEventSelectionTests:XCTestCase {
func testNBAScheduledTimeAndProviderTipOffResolveSameMatchup() throws {
 let start=sportsDate("2026-10-23T23:00:00Z")!,now=start.addingTimeInterval(-3600)
 let row=RecordedEventMetadata(id:"0d34c898f6cbb990077ae9f0a2773513",code:"basketball_nba",homeTeam:"Boston Celtics",awayTeam:"New York Knicks",commence:"2026-10-23T23:10:00Z")
 func select(_ rows:[RecordedEventMetadata])->RecordedEventMetadata? {
  RecordedEventSelection.select(rows,sport:"NBA",phase:.regular,home:"Boston Celtics",away:"New York Knicks",start:start,now:now)
 }
 XCTAssertEqual(select([row])?.id,row.id)
 XCTAssertNil(select([row,.init(id:"second",code:row.code,homeTeam:row.homeTeam,awayTeam:row.awayTeam,commence:row.commence)]))
 XCTAssertNil(select([.init(id:row.id,code:row.code,homeTeam:row.awayTeam,awayTeam:row.homeTeam,commence:row.commence)]))
}
func testStrictMetadataSelection() throws {
let f=ISO8601DateFormatter();let now=f.date(from:"2026-10-14T12:00:00Z")!;let start=f.date(from:"2026-10-14T23:30:00Z")!
func row(_ id:String="one",_ code:String="icehockey_nhl",_ home:String="Washington Capitals",_ away:String="Montréal Canadiens",_ time:String="2026-10-14T23:40:00Z")->RecordedEventMetadata{.init(id:id,code:code,homeTeam:home,awayTeam:away,commence:time)}
func select(_ rows:[RecordedEventMetadata],_ sport:String="NHL",_ phase:NBARecordedOddsPhase?=nil)->RecordedEventMetadata?{RecordedEventSelection.select(rows,sport:sport,phase:phase,home:"Washington Capitals",away:"Montreal Canadiens",start:start,now:now)}
var count=0
func check(_ value:Bool,_ name:String){XCTAssertTrue(value,name);count+=1}
check(select([row()])?.id == "one","props-only NHL exact names plus10min")
check(select([row(),row("two")]) == nil,"ambiguous IDs")
check(select([row(),row()])?.id == "one","identical duplicate")
check(select([row(),row("one","icehockey_nhl","Other")]) == nil,"conflicting duplicate identity")
check(select([row("one","icehockey_nhl","Washington Capitals","Montréal Canadiens","2026-10-14T23:40:01Z")]) == nil,"outside tolerance")
check(select([row("one","icehockey_nhl","Montreal Canadiens","Washington Capitals")]) == nil,"reversed teams")
check(select([row("one","icehockey_nhl","Washington Capitals","Montréal Canadiens","2026-10-13T23:30:00Z")]) == nil,"started event")
check(select([row("one","baseball_mlb")]) == nil,"wrong provider code")
check(select([]) == nil,"missing metadata")
check(select([row("")]) == nil,"empty ID")
check(select([row("one","basketball_nba_preseason","Washington Capitals","Montréal Canadiens","2026-10-14T23:30:00Z")],"NBA",.preseason)?.id == "one","preseason exact code")
check(select([row("one","basketball_nba_preseason")],"NBA",.regular) == nil,"regular cannot read preseason")
check(select([row("one","basketball_nba")],"NBA",.unknown) == nil,"unknown phase")
check(select([row("one","americanfootball_nfl")],"NFL") == nil,"NFL no10min tolerance")
check(select([row("one","baseball_mlb","Washington Capitals","Montréal Canadiens","2026-10-14T23:30:00Z")],"MLB") != nil,"MLB exact start")
let decoded=try JSONDecoder().decode(RecordedEventMetadata.self,from:Data(#"{"id":"real","code":"basketball_nba_preseason","homeTeam":"Lakers","awayTeam":"Warriors","commence":"2026-10-14T02:00:00Z","home_team":"Lakers"}"#.utf8))
check(decoded.id == "real","ignore provider extras")
print("PASS \(count) offline event selector checks")

}
}
