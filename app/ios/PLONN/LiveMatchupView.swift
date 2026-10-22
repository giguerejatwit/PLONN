import SwiftUI

@MainActor final class LiveMatchupStore: ObservableObject {
    @Published private(set) var snapshot: LiveMatchupSnapshot?
    @Published private(set) var loading = false
    @Published private(set) var failed = false
    @Published var refreshRevision = 0
    private var ticket = UUID()

    func watch(event: SharedMatchupEvent, client: LiveMatchupClient = .init()) async {
        let run = UUID(); ticket = run
        if snapshot?.event != event { snapshot = nil }
        guard LiveMatchupClient.feedPath(event.sport) != nil else { failed = true; return }
        while !Task.isCancelled, ticket == run {
            loading = snapshot == nil
            do {
                let value = try await client.fetch(event: event)
                guard !Task.isCancelled, ticket == run else { return }
                snapshot = value; failed = false
            } catch {
                guard !Task.isCancelled, ticket == run else { return }
                // Retain the last observed facts with an explicit stale label during an outage.
                failed = true
            }
            loading = false
            guard let delay = LiveMatchupPolicy.nextRefresh(event: event, snapshot: snapshot, now: client.now(), failed: failed) else { return }
            do { try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000)) }
            catch { return }
        }
    }
}

struct LiveMatchupCard: View {
    let event: SharedMatchupEvent
    let snapshot: LiveMatchupSnapshot?
    var loading = false
    var failed = false
    var now = Date()
    private var currentSnapshot: LiveMatchupSnapshot? { snapshot?.event == event ? snapshot : nil }
    private var stale: Bool { failed || (currentSnapshot?.isStale(at: now) ?? false) }
    private var heading: String {
        guard let snapshot = currentSnapshot else { return "Matchup update" }
        if snapshot.phase == .final { return "Final matchup" }
        return snapshot.phase == .live && !stale ? "Live matchup" : "Matchup update"
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(heading, systemImage: currentSnapshot?.phase == .final ? "flag.checkered" : "waveform.path.ecg")
                .font(.headline).accessibilityIdentifier("live-matchup-heading")
            if let snapshot = currentSnapshot {
                Text((stale ? "Last reported: " : "") + snapshot.status)
                    .font(.subheadline.bold()).foregroundStyle(stale ? .orange : .secondary)
                    .accessibilityIdentifier("live-matchup-status")
                if let summary = snapshot.scoreSummary {
                    Text(summary).font(.headline).accessibilityIdentifier("live-matchup-score")
                }
                if !snapshot.stats.isEmpty {
                    Grid(horizontalSpacing: 14, verticalSpacing: 10) {
                        GridRow {
                            Text(snapshot.phase == .final ? "Game stats" : "Game so far").frame(maxWidth: .infinity, alignment: .leading)
                            Text(snapshot.away.abbreviation)
                            Text(snapshot.home.abbreviation)
                        }.font(.caption.bold()).foregroundStyle(.secondary)
                        ForEach(snapshot.stats) { stat in
                            GridRow {
                                Text(stat.label).frame(maxWidth: .infinity, alignment: .leading)
                                Text(stat.away ?? "—").monospacedDigit()
                                Text(stat.home ?? "—").monospacedDigit()
                            }.font(.subheadline)
                                .accessibilityElement(children: .ignore)
                                .accessibilityLabel("\(stat.label), \(snapshot.away.name): \(stat.away ?? "unavailable"), \(snapshot.home.name): \(stat.home ?? "unavailable")")
                                .accessibilityIdentifier("live-matchup-stat-" + stat.id)
                        }
                    }
                } else {
                    Text(snapshot.phase == .scheduled ? "Waiting for the game to begin. Stats appear when play starts." : "Team stats haven’t reached the feed yet. The game status is shown above.")
                        .font(.subheadline).foregroundStyle(.secondary)
                        .accessibilityIdentifier("live-matchup-empty-stats")
                }
                Text((stale ? "Updates paused · " : "") + "ESPN · Last checked " + snapshot.checkedAt.formatted(date: .omitted, time: .standard))
                    .font(.caption).foregroundStyle(stale ? .orange : .secondary)
                    .accessibilityIdentifier("live-matchup-freshness")
            } else {
                Text(loading ? "Loading game updates…" : "Game updates are temporarily unavailable. Pull to retry; team history remains available below.")
                    .font(.subheadline).foregroundStyle(.secondary).accessibilityIdentifier("live-matchup-unavailable")
            }
            if currentSnapshot?.phase != .final {
                Divider()
                Text("Live odds and betting directions aren’t available yet.").font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("live-matchup-odds-unavailable")
            }
        }.accessibilityElement(children: .contain).accessibilityIdentifier("live-matchup-content")
    }
}

extension SportsItem {
    func matchupEvent(sport: String) -> SharedMatchupEvent {
        .init(sport: sport, eventID: "espn:" + id, homeTeam: home, awayTeam: away, startISO: SharedMatchupEvent.scoreboardStart(start))
    }
}

#if DEBUG && targetEnvironment(simulator)
/// Launch-only preview; never changes authentication, persisted data, or transport configuration.
enum LiveMatchupPreview {
    static var payload: String { ProcessInfo.processInfo.environment["PLONN_LIVE_MATCHUP_PREVIEW"] ?? "" }
    static var active: Bool { !payload.isEmpty }
}
struct LiveMatchupPreviewView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                if let data = LiveMatchupPreview.payload.data(using: .utf8),
                   let root = try? JSONSerialization.jsonObject(with: data) as? [String:Any],
                   let rawEvent = root["event"], let eventData = try? JSONSerialization.data(withJSONObject: rawEvent),
                   let event = try? JSONDecoder().decode(SharedMatchupEvent.self, from: eventData),
                   let raw = root["summary"], let summaryData = try? JSONSerialization.data(withJSONObject: raw) {
                    let date = (root["checkedAt"] as? String).flatMap(sportsDate) ?? Date()
                    let snapshot = try? LiveMatchupClient.parse(summaryData, event: event, checkedAt: date)
                    VStack(alignment: .leading, spacing: 20) {
                        Text(root["recorded"] as? Bool == true ? "Recorded game · Preview" : "Offline test data · Preview").font(.caption).foregroundStyle(.secondary)
                        LiveMatchupCard(event: event, snapshot: snapshot, failed: root["failed"] as? Bool == true, now: date)
                            .padding(16).background(navySurface, in: RoundedRectangle(cornerRadius: 18))
                    }.padding(16)
                }
            }.background(Color(uiColor:.systemGroupedBackground)).navigationTitle("Matchup")
        }.preferredColorScheme(.dark)
    }
}
#endif
