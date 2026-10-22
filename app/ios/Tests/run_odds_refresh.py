#!/usr/bin/env python3
"""Test the verbatim Store/cache code on macOS with a controlled cloud transport.
No project registration, simulator, external requests, or production edits required.
"""
from pathlib import Path
import subprocess
import tempfile
import os
import argparse

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--filter')
parser.add_argument('--source-dir',type=Path,default=root/'PLONN')
args=parser.parse_args()
source = (args.source_dir / 'PLONNApp.swift').read_text()
game_source = (args.source_dir / 'GameCenter.swift').read_text()
slate_start=game_source.index('@MainActor private final class AllGamesStore:')
slate_end=game_source.index('private struct AllGamesSlate:View',slate_start)
slate_source=game_source[slate_start:slate_end].replace('private final class AllGamesStore:', 'final class AllGamesStore:',1)
# This harness exercises the broad recorded-odds Store path only. The separate
# scoped Analytics endpoint is covered by AnalyticsOddsMatchingTests.
scoped_start = source.index(' /// Separate, versioned endpoint.')
scoped_end = source.index(' private func resetOdds()', scoped_start)
source = source[:scoped_start] + source[scoped_end:]
source = '\n'.join(line for line in source.splitlines() if not line.startswith((' private static var scopedSnapshots:', ' private static var scopedFlights:')))
start = source.index('// Shared odds request state')
end = source.index('@main struct PLONNApp:', start)
headers = '\n'.join(line for line in source.splitlines() if line.startswith(('struct OddsResponse:', 'struct HistoryRow:')))
stubs = '''
import SwiftUI
import Combine
struct TestUser { let id: UUID; var isAnonymous=false }
// Only the retained snapshot's value type is needed by the verbatim Store adapter.
enum AnalyticsFeedService {
 struct RecordedSnapshot {
  let markets:[Market]; let events:[RecordedEventMetadata]; let cacheOnly:Bool
  let fetchedAt:String?; let servedAt:String?; let recordedArchive:Bool
 }
}
enum FunctionsError: Error { case httpError(Int, Data) }
struct TestOptions {
 var body: [String:String]
 init(body:OddsRequestBody) {self.body=["sport":body.sport,"cacheOnly":String(body.cacheOnly)]; if let event=body.event {self.body["event"]=event};if let phase=body.phase {self.body["phase"]=phase} }
}
enum CloudConfiguration { static let nbaPreseasonCacheEnabled=true }
enum LaunchSports {static let supported=["NBA","MLB","NFL","NHL"]}
struct SportsItem:Sendable {var id:String;var name:String="Game";var live=false;var start:String;var seasonType:Int?=nil}
@MainActor final class SportsDirectory {
 static var operation:(String,Date) async -> ([SportsItem],String) = { _,_ in ([],"ESPN") }
 var items:[SportsItem]=[];var status=""
 func load(sport:String,view:String,date:Date) async {let value=await Self.operation(sport,date);items=value.0;status=value.1}
}
@MainActor final class TestFunctions {
 var operation: ([String:String]) async throws -> OddsResponse = { _ in throw URLError(.unsupportedURL) }
 func invoke(_ name: String, options: TestOptions) async throws -> OddsResponse { try await operation(options.body) }
}
@MainActor final class TestClient { let functions=TestFunctions() }
@MainActor final class AccountStore: ObservableObject {
 @Published var user: TestUser?
 let client=TestClient()
 init(owner: UUID) { user=TestUser(id:owner) }
}
'''
with tempfile.TemporaryDirectory(prefix='plonn-odds-tests-') as directory:
    package = Path(directory)
    core = package / 'Sources/OddsRefreshCore'
    tests = package / 'Tests/OddsRefreshCoreTests'
    core.mkdir(parents=True)
    tests.mkdir(parents=True)
    (core / 'Store.swift').write_text('import SwiftUI\nimport Combine\n' + headers + '\n' + source[start:end])
    (core / 'TransportDouble.swift').write_text(stubs)
    (core / 'AllGamesStore.swift').write_text('import SwiftUI\nimport Combine\n'+slate_source)
    for name in ['WatchlistStorage.swift', 'Probability.swift', 'PLONNServicePolicy.swift']:
        (core / name).write_text(((args.source_dir/name) if (args.source_dir/name).exists() else (root/'PLONN'/name)).read_text())
    for name in ['OddsRefreshTests.swift','AllGamesOddsTests.swift','RecordedOddsSnapshotTests.swift']:
        if (root/'Tests'/name).exists():
            (tests/name).write_text((root/'Tests'/name).read_text())
    fixture_dir=package/'Fixtures';fixture_dir.mkdir()
    for path in (root/'verification/odds-review-20261014').glob('*-snapshot-fixture.json'):
        (fixture_dir/path.name).write_bytes(path.read_bytes())
    (package / 'Package.swift').write_text('''// swift-tools-version: 5.9
import PackageDescription
let package=Package(name:"OddsRefreshCore", platforms:[.macOS(.v14)], targets:[
 .target(name:"OddsRefreshCore"),
 .testTarget(name:"OddsRefreshCoreTests",dependencies:["OddsRefreshCore"],swiftSettings:[.define("ODDS_STANDALONE_TESTS")])
])
''')
    command=['swift','test','--package-path',directory,'--jobs','2','--disable-sandbox']
    if args.filter:command+=['--filter',args.filter]
    environment=dict(os.environ,PLONN_ODDS_FIXTURES=str(fixture_dir),CLANG_MODULE_CACHE_PATH='/tmp/plonn-odds-module-cache',SWIFT_MODULECACHE_PATH='/tmp/plonn-odds-module-cache')
    subprocess.run(command,check=True,env=environment)
