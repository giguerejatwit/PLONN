import SwiftUI
import Supabase

private struct SaveModelParameters: Encodable {
    let p_model_id: UUID?
    let p_name: String
    let p_configuration: LabRecipe
    enum CodingKeys: CodingKey { case p_model_id, p_name, p_configuration }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        // RPC requires an explicit null for a new model.
        try c.encode(p_model_id, forKey: .p_model_id)
        try c.encode(p_name, forKey: .p_name)
        try c.encode(p_configuration, forKey: .p_configuration)
    }
}
@MainActor final class ModelLibraryStore: ObservableObject {
    @Published private(set) var versions: [AccountModelVersion] = []
    @Published private(set) var busy = false
    @Published private(set) var message = ""
    private var generation = UUID()

    func refresh(_ account: AccountStore) async {
        let request = UUID(); generation = request
        versions = []; message = ""; busy = false
        guard let owner = account.user?.id else { return }
        busy = true
        defer { if generation == request { busy = false } }
        do {
            let rows: [AccountModelVersion] = try await account.client.from("model_versions").select()
                .eq("user_id", value: owner).order("created_at", ascending: false).limit(500).execute().value
            guard generation == request, account.user?.id == owner, !Task.isCancelled else { return }
            versions = rows
            if rows.count == 500 { message = "Showing the latest 500 saved versions." }
        } catch {
            if generation == request, account.user?.id == owner {
                message = "My Models could not load. Check your connection and sign-in, then retry."
            }
        }
    }
    func save(_ recipe: LabRecipe, modelID: UUID?, account: AccountStore) async -> AccountModelVersion? {
        guard !busy, let owner = account.user?.id else { return nil }
        let request = generation; busy = true; message = ""
        defer { if generation == request { busy = false } }
        do {
            let version: AccountModelVersion = try await account.client.rpc("save_model_version", params:
                SaveModelParameters(p_model_id: modelID, p_name: recipe.name, p_configuration: recipe)).single().execute().value
            guard generation == request, account.user?.id == owner, version.user_id == owner, version.configuration.configurationFingerprint == recipe.configurationFingerprint else { return nil }
            versions.insert(version, at: 0)
            message = "Saved privately · \(version.name) v\(version.version). Configuration only."
            return version
        } catch {
            if generation == request, account.user?.id == owner { message = "Save failed. Your setup remains in Models. Check the name, inputs and connection, then retry." }
            return nil
        }
    }
    func delete(_ version: AccountModelVersion, account: AccountStore) async {
        guard !busy, let owner = account.user?.id, owner == version.user_id else { return }
        let request = generation; busy = true
        defer { if generation == request { busy = false } }
        do {
            try await account.client.rpc("delete_saved_model", params: ["p_model_id": version.model_id]).execute()
            guard generation == request, account.user?.id == owner else { return }
            versions.removeAll { $0.model_id == version.model_id }; message = "Model and its private versions deleted."
        } catch {
            if generation == request, account.user?.id == owner { message = "Could not delete this model. Try again." }
        }
    }
}

struct MyModelsView: View {
    @EnvironmentObject private var account: AccountStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var library = ModelLibraryStore()
    var currentSetup: LabRecipe? = nil
    var setupOwner: UUID? = nil
    var onLoad: ((AccountModelVersion) -> Void)? = nil
    var initialModelID: UUID? = nil
    @State private var selectedModel: UUID?
    @State private var deleteTarget: AccountModelVersion?
    private var mine: [AccountModelVersion] { library.versions.filter { $0.user_id == account.user?.id } }
    private var latest: [AccountModelVersion] {
        Dictionary(grouping: mine, by: \.model_id).values.compactMap { $0.max { $0.version < $1.version } }
            .sorted { $0.created_at > $1.created_at }
    }
    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 18) {
                Text("My Models").font(.system(.title2, design: .monospaced).weight(.semibold))
                Text("Named, versioned configurations saved to your account. Reopen a trained setup to resolve its retained weights and frozen historical feature snapshot. Performance remains experimental.")
                    .font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary)
                if account.user == nil {
                    NavigationLink { AccountView() } label: { Label("Sign in to My Models", systemImage: "person.crop.circle") }
                    Text("Device-only setups remain in Models and are never uploaded automatically.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                } else {
                    if let setup = currentSetup, setupOwner == nil || setupOwner == account.user?.id {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Save “\(setup.name)”").font(.system(.headline, design: .monospaced))
                            Picker("Save to", selection: $selectedModel) {
                                Text("New model").tag(nil as UUID?)
                                ForEach(latest.filter { LaunchSports.includes($0.configuration.sport ?? "NFL") }) { item in Text("\(item.name) · next version").tag(Optional(item.model_id)) }
                            }
                            Button(selectedModel == nil ? "Save private model" : "Save new private version") {
                                Task {
                                    if let version = await library.save(setup, modelID: selectedModel, account: account) {
                                        onLoad?(version)
                                        if onLoad != nil { dismiss() }
                                    }
                                }
                            }.plonnGlassButton(prominent:true).foregroundStyle(.primary).disabled(library.busy)
                        }.padding(16).background(navySurface, in: RoundedRectangle(cornerRadius: 14))
                    }
                    if library.busy { PLONNLoader() }
                    if !library.message.isEmpty { Text(library.message).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
                    if mine.isEmpty && !library.busy { Text("No private models loaded. Save a setup from Models.").foregroundStyle(.secondary) }
                    ForEach(latest) { model in
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Text(model.name).font(.system(.headline, design: .monospaced))
                                Spacer()
                                Button { deleteTarget = model } label: { Image(systemName: "trash").frame(width: 44, height: 44) }
                                    .accessibilityLabel("Delete " + model.name).disabled(library.busy)
                            }
                            ForEach(mine.filter { $0.model_id == model.model_id }.sorted { $0.version > $1.version }) { version in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("v\(version.version) · \(SavedModelType(rawValue: version.configuration.resolvedModelType)?.title ?? "Unsupported model type")")
                                        .font(.system(.subheadline, design: .monospaced).weight(.semibold))
                                    Text("\(version.configuration.sport ?? "") · \(version.configuration.player) · \(version.configuration.stat ?? "") · \(version.configuration.features.count) features")
                                        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                                    Text("Saved \(String(version.created_at.prefix(10))) · Setup only").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                                    if !LaunchSports.includes(version.configuration.sport ?? "NFL") {
                                        Text("Archived sport · Saved version preserved").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                                    } else if let onLoad {
                                        Button("Use this version") { onLoad(version); dismiss() }
                                    } else {
                                        NavigationLink("Open this version in Models") { AILabView(initialVersion: version) }
                                    }
                                }.padding(.vertical, 6)
                            }
                        }.padding(16).background(navySurface, in: RoundedRectangle(cornerRadius: 14))
                    }
                }
            }.padding(20)
        }.background(ModelLandscape()).navigationTitle("PLONN").navigationBarTitleDisplayMode(.inline).toolbar {ToolbarItem(placement:.principal) {Brand()}}
        .task(id: account.user?.id) { selectedModel = setupOwner == account.user?.id ? initialModelID:nil; deleteTarget = nil; await library.refresh(account) }
        .plonnRefreshable { guard !library.busy else { return }; await library.refresh(account) }
        .confirmationDialog("Delete this model and all its saved versions?", isPresented: Binding(get: { deleteTarget != nil }, set: { if !$0 { deleteTarget = nil } }), titleVisibility: .visible) {
            if let target = deleteTarget { Button("Delete model", role: .destructive) { Task { await library.delete(target, account: account) }; deleteTarget = nil } }
        }
    }
}
