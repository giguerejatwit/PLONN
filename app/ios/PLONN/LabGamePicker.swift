import SwiftUI

struct LabGamePicker: View {
    let league: String
    let source: LabHistoricalGames.Source
    var initialDate: Date? = nil
    var loader: (String, LabHistoricalGames.Source, Date) async throws -> [LabHistoricalGame] = LabHistoricalGames.load
    var choose: (LabHistoricalGame, LabHistoricalGame.Team) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var date = LabHistoricalGames.latestDate()
    @State private var games: [LabHistoricalGame] = []
    @State private var loading = true
    @State private var failed = false
    @State private var retry = 0
    @State private var search = ""
    @State private var didSearchInitial = false
    @State private var loadedKey: String?
    private var requestKey: String { "\(league)|\(source.rawValue)|\(LabHistoricalGames.day(date))|\(retry)" }
    private var visibleGames: [LabHistoricalGame] {
        search.isEmpty ? games : games.filter { LabNameSearch.matches(query:search,name:$0.title) }
    }
    var body: some View {
        NavigationStack {
            List {
                Section {
                    DatePicker("Game date", selection: $date, in: ...LabHistoricalGames.latestDate(), displayedComponents: .date)
                        .environment(\.timeZone, LabHistoricalGames.calendar.timeZone)
                } footer: {
                    Text("Choose a completed regular-season game, then the team you want to analyze. For player props, choose the player's team in that game.")
                }
                Section {
                    if loading { ProgressView("Loading games…") }
                    else if failed {
                        Text("Games couldn’t load. Check your connection and try again.")
                        Button("Try again") { retry += 1 }
                    } else if visibleGames.isEmpty {
                        Text(search.isEmpty ? "No regular-season games are ready for historical testing on this date. Try an earlier date." : "No matching teams on this date.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(visibleGames) { game in
                            VStack(alignment: .leading, spacing: 12) {
                                Text(game.title).font(.headline)
                                Text("\(LabHistoricalGames.displayDate(game.date)) · \(game.detail) · \(game.awayScore)–\(game.homeScore)")
                                    .font(.caption).foregroundStyle(.secondary)
                                Text("Choose team").font(.caption).foregroundStyle(.secondary)
                                teamButton(game.away, game: game)
                                teamButton(game.home, game: game)
                            }.padding(.vertical, 6)
                        }
                    }
                }
            }
            .plonnGlassButton()
        .searchable(text: $search, prompt: "Find a team")
            .searchSuggestions {
                if !search.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {
                    ForEach(Array(Set(games.flatMap {[$0.home.name,$0.away.name]})).filter {LabNameSearch.matches(query:search,name:$0)}.sorted {LabNameSearch.rank(query:search,name:$0)<LabNameSearch.rank(query:search,name:$1)},id:\.self) {name in
                        Label(name,systemImage:"sportscourt").searchCompletion(name)
                    }
                }
            }
            .navigationTitle("Choose \(league) game")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
            .onAppear { if let initialDate { date = min(initialDate, LabHistoricalGames.latestDate()) } }
            .task(id: requestKey) { await load() }
        }.preferredColorScheme(.dark)
    }
    private func teamButton(_ team: LabHistoricalGame.Team, game: LabHistoricalGame) -> some View {
        Button { choose(game, team); dismiss() } label: {
            HStack { Text(team.name); Spacer(); Image(systemName: "chevron.right") }
                .contentShape(Rectangle())
        }.plonnGlassButton().accessibilityIdentifier("lab-game-\(game.id)-\(team.requestName)")
    }
    @MainActor private func load() async {
        let key = requestKey
        guard loadedKey != key else { return }
        let findRecent = initialDate == nil && !didSearchInitial
        didSearchInitial = true
        loading = true; failed = false; games = []
        do {
            var candidate = date
            var result = try await loader(league, source, candidate)
            if findRecent {
                for _ in 0..<7 where result.isEmpty {
                    guard !Task.isCancelled, key == requestKey else { return }
                    candidate = LabHistoricalGames.calendar.date(byAdding: .day, value: -1, to: candidate)!
                    result = try await loader(league, source, candidate)
                }
            }
            guard !Task.isCancelled, key == requestKey else { return }
            if !result.isEmpty { date = candidate }
            loadedKey = requestKey
            games = result; loading = false
        } catch {
            guard !Task.isCancelled, key == requestKey else { return }
            failed = true; loading = false
        }
    }
}

#if DEBUG && targetEnvironment(simulator)
/// Isolated, offline picker preview. It never opens private models or authenticates an account.
enum LabGamePickerFixture {
    static var active: Bool { ProcessInfo.processInfo.environment["PLONN_LAB_GAME_PICKER_FIXTURE"] == "1" }
}
struct LabGamePickerFixtureView: View {
    @State private var showing = true
    @State private var selection = ""
    var body: some View {
        VStack(spacing: 20) {
            Text("Completed-game picker preview").font(.headline)
            Text(selection).accessibilityIdentifier("lab-picker-selection")
            Button("Choose completed game") { showing = true }
        }.padding().sheet(isPresented: $showing) {
            LabGamePicker(league: "NFL", source: .nflverse,
                initialDate: ISO8601DateFormatter().date(from: "2024-12-01T18:00:00Z"),
                loader: { _, _, date in
                    [LabHistoricalGame(id: "2024_13_PIT_CIN", date: date,
                        home: .init(name: "Cincinnati Bengals", requestName: "CIN"),
                        away: .init(name: "Pittsburgh Steelers", requestName: "PIT"),
                        homeScore: 38, awayScore: 44, detail: "Final")]
                }) { game, team in
                    selection = "Team: \(team.name)\nOpponent: \(game.opponent(for: team)!.name)"
                }
        }.preferredColorScheme(.dark)
    }
}
#endif
