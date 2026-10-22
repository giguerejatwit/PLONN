import SwiftUI

struct PlayerMatchupLibrarySection: View {
    @EnvironmentObject private var account: AccountStore
    @State private var models: [SavedPlayerMatchupModel] = []
    @State private var owner: UUID?
    @State private var message = ""
    @State private var deleting: SavedPlayerMatchupModel?
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if owner == account.user?.id, !models.isEmpty {
                Text("Players").font(.title3.bold())
                ForEach(models) { model in
                    HStack {
                        NavigationLink { PlayerMatchupView(setup: model.setup, artifactID: model.id) } label: {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(model.setup.player).font(.headline)
                                Text(LabPropCatalog.displayName(model.setup.stat)).font(.subheadline)
                                Text(model.setup.away + " at " + model.setup.home).font(.caption).foregroundStyle(.secondary)
                                if let start = model.setup.start { Text(start, format: .dateTime.month(.abbreviated).day().hour().minute()).font(.caption).foregroundStyle(.secondary) }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }.buttonStyle(.plain)
                        Button { deleting = model } label: { Image(systemName: "trash").frame(width: 44, height: 44) }.accessibilityLabel("Delete " + model.setup.player + " matchup model")
                    }.padding(16).background(navySurface, in: RoundedRectangle(cornerRadius: 14))
                }
            }
            if !message.isEmpty {
                Text(message).font(.caption).foregroundStyle(.secondary)
                Button("Retry player models") { Task { await load() } }.frame(minHeight: 44)
            }
        }.task(id: account.user?.id) { await load() }
        .confirmationDialog("Delete this player matchup model?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            if let model = deleting {
                Button("Delete model", role: .destructive) { deleting = nil; Task { await remove(model) } }
            }
        }
    }
    @MainActor private func request(_ path: String, owner expected: UUID, method: String = "GET") async throws -> Data {
        let session = try await account.client.auth.session
        guard account.user?.id == expected, session.user.id == expected, !session.user.isAnonymous else { throw CancellationError() }
        var request = URLRequest(url: CloudConfiguration.labURL.appendingPathComponent(path)); request.httpMethod = method; request.timeoutInterval = 20
        request.setValue("Bearer " + session.accessToken, forHTTPHeaderField: "Authorization")
        let (data, response) = try await URLSession.shared.data(for: request)
        try Task.checkCancellation()
        guard account.user?.id == expected else { throw CancellationError() }
        guard (response as? HTTPURLResponse)?.statusCode == 200, data.count <= 2_000_000 else { throw URLError(.badServerResponse) }
        return data
    }
    @MainActor private func load() async {
        models = []; message = ""; owner = nil
        guard let expected = account.user?.id, account.hasSignedInAccount else { return }
        do {
            let data = try await request("player-matchup-models", owner: expected)
            struct Response: Decodable { let models: [SavedPlayerMatchupModel] }
            let decoded = try JSONDecoder().decode(Response.self, from: data).models
            guard decoded.count <= 500, Set(decoded.map(\.id)).count == decoded.count,
                  decoded.allSatisfy({ $0.id.range(of: "^[0-9a-f]{32}$", options: .regularExpression) != nil && (try? $0.setup.payload()) != nil && sportsDate($0.generated_at) != nil }) else { throw URLError(.cannotParseResponse) }
            models = decoded; owner = expected
        } catch is CancellationError { }
        catch { if !Task.isCancelled { message = "Saved player models couldn’t load." } }
    }
    @MainActor private func remove(_ model: SavedPlayerMatchupModel) async {
        guard let expected = owner, account.user?.id == expected else { return }
        do {
            _ = try await request("player-matchup-models/" + model.id, owner: expected, method: "DELETE")
            if let payload = try? LabSubmissionStore.canonical(model.setup.payload()),
               let record = LabSubmissionStore().record(owner: expected, payload: payload) {
                LabSubmissionStore().update(record, terminal: true)
                LabSubmissionStore().forgetCompleted(owner: expected, payload: payload)
            }
            await load()
        } catch is CancellationError { }
        catch { if account.user?.id == expected { message = "This model couldn’t be deleted. Try again." } }
    }
}
