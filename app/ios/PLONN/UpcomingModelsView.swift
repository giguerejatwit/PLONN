import SwiftUI

struct UpcomingGamesView: View {
    @State private var sport = "NFL"
    @State private var games: [UpcomingGameSetup] = []
    @State private var loadedSport: String?
    @State private var loadedAt: Date?
    @State private var loading = false
    @State private var message = ""
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Picker("Sport", selection: $sport) {
                    ForEach(LaunchSports.supported, id: \.self) { Text($0).tag($0) }
                }.pickerStyle(.segmented)
                Text("Next 7 days").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                if loading { PLONNLoader("Finding upcoming games…") }
                if !message.isEmpty {
                    Text(message).foregroundStyle(.secondary)
                    Button("Refresh games") { Task { await load() } }.disabled(loading)
                }
                ForEach(games) { game in
                    VStack(alignment: .leading, spacing: 12) {
                        Text(game.away).font(.headline)
                        Text("at " + game.home).font(.headline)
                        if let start = game.start { Text(start, format: .dateTime.weekday(.wide).month(.abbreviated).day().hour().minute()).font(.subheadline).foregroundStyle(.secondary) }
                        NavigationLink { UpcomingModelView(setup: game, autoStart: true) } label: {
                            Label("Build model", systemImage: "sparkles").frame(maxWidth: .infinity).padding(.vertical, 6)
                        }.plonnGlassButton(prominent: true).accessibilityIdentifier("upcoming-build-" + game.eventID)
                    }.padding(18).background(navySurface, in: RoundedRectangle(cornerRadius: 18))
                }
            }.padding(20)
        }.background(ModelLandscape()).navigationTitle("Upcoming games").navigationBarTitleDisplayMode(.inline)
            .task(id: sport) {
                if loadedSport != sport || Date().timeIntervalSince(loadedAt ?? .distantPast) > 60 { await load() }
            }
            .plonnRefreshable { await load() }
    }
    @MainActor private func load() async {
        let expected = sport
        if loadedSport != expected { games = [] }
        message = ""; loading = true
        defer { if expected == sport { loading = false } }
        do {
            let formatter = DateFormatter(); formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(secondsFromGMT: 0); formatter.dateFormat = "yyyyMMdd"
            let now = Date()
            let urls = (0...7).map { offset in
                URL(string: "https://site.api.espn.com/apis/site/v2/sports/" + sportsFeedPath(expected) + "/scoreboard?limit=1000&dates=" + formatter.string(from: now.addingTimeInterval(Double(offset) * 86400)))!
            }
            let fetched = await withTaskGroup(of: [UpcomingGameSetup]?.self, returning: [[UpcomingGameSetup]?].self) { group in
                for url in urls { group.addTask {
                    do {
                        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData); request.timeoutInterval = 20
                        let (data, response) = try await URLSession.shared.data(for: request)
                        guard (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
                        return try UpcomingGameSetup.scheduledGames(data, league: expected, now: now)
                    } catch { return nil }
                } }
                var days: [[UpcomingGameSetup]?] = []
                for await day in group { days.append(day) }
                return days
            }
            try Task.checkCancellation()
            guard expected == sport else { return }
            guard fetched.contains(where: { $0 != nil }) else { throw URLError(.badServerResponse) }
            var unique: [String: UpcomingGameSetup] = [:]
            for game in fetched.compactMap({ $0 }).flatMap({ $0 }) { unique[game.id] = game }
            games = unique.values.filter { ($0.start ?? .distantFuture) < now.addingTimeInterval(7 * 86400) }.sorted { ($0.start ?? .distantFuture) < ($1.start ?? .distantFuture) }
            loadedSport = expected; loadedAt = Date()
            if fetched.contains(where: { $0 == nil }) { message = "Some dates couldn’t refresh. Showing the games we could verify." }

            if games.isEmpty { message = "No scheduled regular-season or playoff games in the next 7 days. Check another sport or refresh later." }
        } catch is CancellationError { }
        catch { if expected == sport && !Task.isCancelled { message = "Upcoming games couldn’t load. Check your connection and try again." } }
    }
}

struct UpcomingModelView: View {
    let setup: UpcomingGameSetup
    var autoStart = false
    var artifactID: String? = nil
    @EnvironmentObject private var account: AccountStore
    @EnvironmentObject private var purchases: PurchaseStore
    @StateObject private var markets = Store()
    @State private var forecast: UpcomingGameSetup.Forecast?
    @State private var showAvailability = false
    @State private var message = ""
    @State private var running = false
    @State private var attempted = false
    @State private var showPro = false
    @State private var retryIsNewRun = false
    @State private var resultOwner: UUID?
    private var quoteRequestID: String {
        (account.user?.id.uuidString ?? "") + setup.id + (setup.league == "NBA" ? (forecast.map { $0.postseason ? "playoff" : "regular" } ?? "unverified") : "")
    }
    private var quote: Market? {
        guard account.hasSignedInAccount, let start = setup.start else { return nil }
        return markets.live.filter { market in
            guard let observed = market.updated.flatMap(sportsDate), observed <= Date(), Date().timeIntervalSince(observed) <= 86400 else { return false }
            return market.kind == "team" && market.market == "h2h" && market.name == setup.home &&
                market.mode == "live" && market.eventState != "in" && market.eventState != "post" &&
                BuilderLogic.decimalOdds(market.price) != nil &&
                Store.matchesRecordedEvent(market, sport: setup.league, home: setup.home, away: setup.away, start: start, candidates: markets.live)
        }.sorted { $0.book < $1.book }.first
    }
    var body: some View {
        AccountRequiredView(feature: .privateModels) { content }
            .sheet(isPresented: $showAvailability) {
                NavigationStack {
                    ScrollView {
                        if let snapshot = forecast?.availability, resultOwner == account.user?.id {
                            availabilityCard(snapshot).padding(20)
                        }
                    }.background(navySurface).navigationTitle("Availability").navigationBarTitleDisplayMode(.inline)
                        .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showAvailability = false } } }
                }.preferredColorScheme(.dark)
            }
            .sheet(isPresented: $showPro) { NavigationStack { ProPaywallView().toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showPro = false } } } } }
    }
    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Text(setup.league).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(setup.title).font(.title2.bold())
                if let start = setup.start { Text(start, format: .dateTime.weekday(.wide).month().day().hour().minute()).foregroundStyle(.secondary) }
                if running {
                    PLONNLoader(message.isEmpty ? "Building your model…" : message)
                    Text("Return to this matchup to check progress.").font(.caption).foregroundStyle(.secondary)
                } else if !message.isEmpty { Text(message).foregroundStyle(forecast == nil ? .orange : .secondary) }
                if let forecast, resultOwner == account.user?.id {
                    resultCard(forecast)
                }
                if !running {
                    if purchases.mayUseProFeatures {
                        if (setup.start ?? .distantPast) > Date() {
                            if setup.evaluationMode != UpcomingGameSetup.mode {
                                NavigationLink {
                                    UpcomingModelView(setup: UpcomingGameSetup(league: setup.league, eventID: setup.eventID, home: setup.home, away: setup.away, startsAt: setup.startsAt), autoStart: true)
                                } label: { Text("Build with opponent strength") }.plonnGlassButton(prominent: true)
                            } else {
                            Button(forecast == nil ? (attempted ? "Try again" : "Build model") : "Update forecast") {
                                Task { await run(retrain: forecast != nil || retryIsNewRun) }
                            }.plonnGlassButton(prominent: true).accessibilityIdentifier("upcoming-run")
                            if forecast != nil { Text("Updating uses another model run.").font(.caption).foregroundStyle(.secondary) }
                            }
                        }
                    } else {
                        Button("Unlock model creation with Pro") { showPro = true }.plonnGlassButton(prominent: true)
                    }
                }
                Text(setup.evaluationMode == UpcomingGameSetup.mode ? "League-wide model · Both teams’ form, opponent strength, schedule difficulty and venue. Same-day and previous-day results are excluded. Availability is checked separately; its impact on win chance is not trained yet." : "Earlier model · Recent scoring and venue only. Build an updated model to include opponent strength and availability checks.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(20)
        }.background(ModelLandscape()).navigationTitle("Matchup model").navigationBarTitleDisplayMode(.inline)
        .task(id: account.user?.id) {
            if resultOwner != account.user?.id { forecast = nil; resultOwner = nil; attempted = false }
            guard account.hasSignedInAccount else { return }
            if forecast == nil && (artifactID != nil || autoStart) { await run() }
        }
        .task(id: quoteRequestID) {
            guard account.hasSignedInAccount, setup.league != "NBA" || forecast != nil else { return }
            await markets.load(setup.league, account: account, home: setup.home, away: setup.away, start: setup.startsAt, cacheOnly: true, nbaSeasonTypes: [forecast.map { $0.postseason ? 3 : 2 }])
        }
    }
    private func resultCard(_ result: UpcomingGameSetup.Forecast) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            TimelineView(.periodic(from: .now, by: 30)) { context in
                Text((setup.start ?? .distantPast) <= context.date ? "SAVED PREGAME FORECAST" : "PREGAME FORECAST")
                    .font(.caption.weight(.semibold)).foregroundStyle(brandAccent)
            }
            if result.brier >= min(result.baselineBrier, result.ratingBrier ?? 2) || result.accuracy < max(result.baselineAccuracy, result.ratingAccuracy ?? 0) {
                Label("No clear edge found", systemImage: "info.circle").font(.headline).foregroundStyle(.orange)
                Text("This model did not consistently outperform the comparison models on games kept out of training.").font(.caption).foregroundStyle(.secondary)
            }
            if result.availability != nil {
                Label("Availability limits this forecast", systemImage: "exclamationmark.triangle.fill").font(.subheadline.bold()).foregroundStyle(.orange)
                Text("Injury and lineup effects aren’t included in this percentage yet.").font(.caption).foregroundStyle(.secondary)
                Button { showAvailability = true } label: {
                    Label("View injuries & lineups", systemImage: "person.2.fill")
                }.plonnGlassButton().accessibilityIdentifier("upcoming-availability")
            }
            Text(setup.home + " to win").font(.headline)
            Text(probability(result.win)).font(.system(size: 44, weight: .bold, design: .rounded)).monospacedDigit()
            Text(result.availability == nil ? "Estimated win chance · Experimental" : "Team-strength win chance · Experimental").font(.subheadline).foregroundStyle(.secondary)
            if setup.league == "NFL" { Text("The remaining chance includes an away win or a tie. Ties aren’t estimated separately.").font(.caption).foregroundStyle(.secondary) }
            TimelineView(.periodic(from: .now, by: 30)) { context in
                if (setup.start ?? .distantPast) > context.date, let quote {
                    Text("Recorded odds · " + odds(quote.price) + " · " + quote.book).font(.subheadline.bold())
                    if let updated = quote.updated.flatMap(sportsDate) { Text("Recorded " + updated.formatted(date: .abbreviated, time: .shortened) + " · Prices may have changed").font(.caption).foregroundStyle(.secondary) }
                } else if (setup.start ?? .distantPast) > context.date {
                    Text("No recent moneyline snapshot is available. The forecast doesn’t require a sportsbook price.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Divider()
            if !result.strengths.isEmpty {
                Text("Opponent strength").font(.headline)
                if let stronger = result.strengths.max(by: { $0.rating < $1.rating }) {
                    Text(stronger.team + " has the higher rating from prior results.").font(.subheadline)
                }
                Text("Ratings account for who each team faced. The model also compares five earlier regular-season games and schedule difficulty.").font(.caption).foregroundStyle(.secondary)
                Divider()
            }
            Text("Past performance").font(.headline)
            Text("Trained on \(result.trained) games · Tested on \(result.tested) newer games")
                .font(.subheadline).foregroundStyle(.secondary)
            Text("Correct picks: " + probability(result.accuracy)).font(.subheadline.bold())
            Text("Always choosing the more common training result: " + probability(result.baselineAccuracy)).font(.caption).foregroundStyle(.secondary)
            if let ratingAccuracy = result.ratingAccuracy {
                Text("Simple opponent-rating model: " + probability(ratingAccuracy) + " correct picks").font(.caption).foregroundStyle(.secondary)
            }
            Text(result.brier < min(result.baselineBrier, result.ratingBrier ?? 2) ? "Its probability estimates beat the comparison models in this test." : "Its probability estimates did not beat all comparison models in this test.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Past results use revised game data, not archived odds. They don’t establish future accuracy or betting profit.").font(.caption).foregroundStyle(.secondary)
            if result.postseason { Text("Playoff matchup · This starter was trained on regular-season games.").font(.caption).foregroundStyle(.secondary) }
            Label("Saved privately to My Models", systemImage: "checkmark.circle").font(.caption).foregroundStyle(brandAccent)
            Text("Created " + result.generated.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(.secondary)
        }.padding(20).background(navySurface, in: RoundedRectangle(cornerRadius: 18))
    }
    private func availabilityCard(_ snapshot: ModelAvailability) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Injuries & lineups").font(.headline)
            TimelineView(.periodic(from: .now, by: 30)) { context in
                if let expiry = snapshot.expires, context.date >= expiry {
                    Label("Saved check · Update before relying on it", systemImage: "clock.badge.exclamationmark").font(.caption).foregroundStyle(.orange)
                }
            }
            if let checked = snapshot.checked { Text("Checked " + checked.formatted(date: .abbreviated, time: .shortened) + " · ESPN").font(.caption).foregroundStyle(.secondary) }
            ForEach(snapshot.injuries) { team in
                VStack(alignment: .leading, spacing: 6) {
                    Text(team.team).font(.subheadline.bold())
                    if team.players.isEmpty {
                        Text(team.state == "stale" ? "Injury reports are out of date." : "No fresh injury report verified.").font(.caption).foregroundStyle(.orange)
                    } else {
                        ForEach(team.players) { player in
                            Text(player.name + " · Reported " + player.status).font(.caption)
                        }
                        Text("General reports; game-day availability is not confirmed.").font(.caption).foregroundStyle(.secondary)
                    }
                    if team.excluded > 0 { Text("\(team.excluded) older or unverified reports excluded.").font(.caption).foregroundStyle(.secondary) }
                    if let lineup = snapshot.lineups.first(where: { $0.team == team.team }) {
                        if let starter = lineup.starter, let name = starter.name {
                            Text((starter.kind == "starting_goalie" ? "Goalie: " : "Starting pitcher: ") + name + (starter.state == "confirmed" ? " · Provider confirmed" : " · Probable"))
                                .font(.caption.bold())
                        }
                        if !lineup.players.isEmpty { Text("Reported starters: " + lineup.players.joined(separator: ", ")).font(.caption) }
                        Text(lineup.state == "confirmed" ? "Starting five reported by ESPN." : "Full starting lineup not confirmed.").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.vertical, 4)
            }
            Text("Missing reports do not mean everyone is available. These checks are saved with the model and may change before game time.").font(.caption).foregroundStyle(.secondary)
        }
    }
    @MainActor private func request(_ path: String, owner: UUID, method: String = "GET", body: Data? = nil, key: String? = nil) async throws -> (Int, [String: Any]) {
        let session = try await account.client.auth.session
        guard account.user?.id == owner else { throw CancellationError() }
        var request = URLRequest(url: CloudConfiguration.labURL.appendingPathComponent(path))
        request.httpMethod = method; request.timeoutInterval = 25; request.httpBody = body
        request.setValue("Bearer " + session.accessToken, forHTTPHeaderField: "Authorization")
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        if let key { request.setValue(key, forHTTPHeaderField: "Idempotency-Key") }
        let (data, response) = try await URLSession.shared.data(for: request)
        try Task.checkCancellation()
        guard account.user?.id == owner else { throw CancellationError() }
        guard let code = (response as? HTTPURLResponse)?.statusCode,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw URLError(.cannotParseResponse) }
        return (code, object)
    }
    @MainActor private func run(retrain: Bool = false) async {
        guard !running, let owner = account.user?.id, account.hasSignedInAccount else { return }
        guard purchases.mayUseProFeatures else { showPro = true; return }
        running = true; attempted = true; message = "Checking your model…"
        defer { running = false }
        do {
            let data = try LabSubmissionStore.canonical(setup.payload()), submissions = LabSubmissionStore()
            // Resume outstanding work before resolving a saved forecast. An update
            // must not silently display an earlier result while its job runs.
            if !retrain && !submissions.hasOutstanding(owner: owner, payload: data) {
                let path = artifactID.map { "artifacts/" + $0 + "/infer" } ?? "artifacts/resolve"
                let (code, saved) = try await request(path, owner: owner, method: "POST", body: data)
                if code == 200 {
                    let body = (saved["result"] as? [String: Any]) ?? saved
                    forecast = try setup.forecast(body); resultOwner = owner; message = ""; return
                }
                if code != 404 { message = UpcomingGameSetup.failure(saved); return }
                if artifactID != nil { message = "This saved model is no longer available."; return }
            }
            guard (setup.start ?? .distantPast) > Date() else { message = "This game has started. Choose another upcoming matchup."; return }
            let record = try submissions.prepare(owner: owner, payload: data, retrain: retrain && !submissions.hasOutstanding(owner: owner, payload: data))
            var jobID = record.jobID
            if jobID == nil {
                let (code, accepted) = try await request("jobs", owner: owner, method: "POST", body: data, key: record.key)
                guard code == 202, let id = accepted["job_id"] as? String else { message = UpcomingGameSetup.failure(accepted); return }
                jobID = id; submissions.update(record, jobID: id)
            }
            guard let jobID else { throw URLError(.cannotParseResponse) }
            for attempt in 0..<100 {
                if attempt > 0 { try await Task.sleep(nanoseconds: 3_000_000_000) }
                let (code, job) = try await request("jobs/" + jobID, owner: owner)
                if code == 404 { submissions.update(record, terminal: true); retryIsNewRun = true; message = "This run has expired. Try again to build a new forecast."; return }
                guard code == 200 else { message = UpcomingGameSetup.failure(job); return }
                switch job["status"] as? String {
                case "completed":
                    submissions.update(record, terminal: true)
                    guard let body = job["result"] as? [String: Any] else { throw URLError(.cannotParseResponse) }
                    forecast = try setup.forecast(body); resultOwner = owner; message = ""; retryIsNewRun = false; return
                case "failed":
                    submissions.update(record, terminal: true); retryIsNewRun = true
                    message = UpcomingGameSetup.failure((job["result"] as? [String: Any]) ?? job); return
                case "queued": message = "Waiting for your model to start…"
                case "running": message = "Training and testing your model…"
                default: throw URLError(.cannotParseResponse)
                }
            }
            message = "Your model is still running. Try again to check its progress."
        } catch is CancellationError { }
        catch is LabServiceContract.ContractError { message = "This model result couldn’t be verified. Try again later." }
        catch { if !Task.isCancelled { message = "We couldn’t connect to the model service. Try again to resume your request." } }
    }
}

struct UpcomingModelLibrarySection: View {
    @EnvironmentObject private var account: AccountStore
    @State private var models: [SavedUpcomingModel] = []
    @State private var owner: UUID?
    @State private var message = ""
    @State private var deleting: SavedUpcomingModel?
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if !models.isEmpty, owner == account.user?.id {
                Text("Teams").font(.title3.bold())
                ForEach(models) { model in
                    HStack {
                        NavigationLink { UpcomingModelView(setup: model.setup, artifactID: model.id) } label: {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(model.setup.title).font(.headline)
                                Text(model.setup.league + " · Pregame forecast").font(.caption).foregroundStyle(.secondary)
                                if let start = model.setup.start { Text(start, format: .dateTime.month(.abbreviated).day().hour().minute()).font(.caption).foregroundStyle(.secondary) }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }.buttonStyle(.plain)
                        Button { deleting = model } label: { Image(systemName: "trash").frame(width: 44, height: 44) }.accessibilityLabel("Delete " + model.setup.title)
                    }.padding(16).background(navySurface, in: RoundedRectangle(cornerRadius: 14))
                }
            }
            if !message.isEmpty { Text(message).font(.caption).foregroundStyle(.secondary); Button("Retry matchup models") { Task { await load() } } }
        }.task(id: account.user?.id) { await load() }
        .confirmationDialog("Delete this matchup model?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            if let model = deleting { Button("Delete model", role: .destructive) { deleting = nil; Task { await remove(model) } } }
        }
    }
    @MainActor private func load() async {
        models = []; message = ""; owner = nil
        guard let expected = account.user?.id, account.hasSignedInAccount else { return }
        do {
            let session = try await account.client.auth.session
            var request = URLRequest(url: CloudConfiguration.labURL.appendingPathComponent("upcoming-models")); request.timeoutInterval = 20
            request.setValue("Bearer " + session.accessToken, forHTTPHeaderField: "Authorization")
            let (data, response) = try await URLSession.shared.data(for: request)
            guard account.user?.id == expected, !Task.isCancelled else { return }
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            struct Response: Decodable { let models: [SavedUpcomingModel] }
            models = try JSONDecoder().decode(Response.self, from: data).models; owner = expected
        } catch { if !Task.isCancelled { message = "Saved matchup models couldn’t load." } }
    }
    @MainActor private func remove(_ model: SavedUpcomingModel) async {
        guard let expected = owner, account.user?.id == expected else { return }
        do {
            let session = try await account.client.auth.session
            var request = URLRequest(url: CloudConfiguration.labURL.appendingPathComponent("upcoming-models/" + model.id)); request.httpMethod = "DELETE"; request.timeoutInterval = 20
            request.setValue("Bearer " + session.accessToken, forHTTPHeaderField: "Authorization")
            let (_, response) = try await URLSession.shared.data(for: request)
            guard account.user?.id == expected else { return }
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            // A deleted job must not be resumed by its old idempotency key.
            if let payload = try? LabSubmissionStore.canonical(model.setup.payload()),
               let record = LabSubmissionStore().record(owner: expected, payload: payload) { LabSubmissionStore().update(record, terminal: true); LabSubmissionStore().forgetCompleted(owner: expected, payload: payload) }
            await load()
        } catch { message = "The model couldn’t be deleted. Try again." }
    }
}
