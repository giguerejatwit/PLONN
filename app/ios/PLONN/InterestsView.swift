import SwiftUI

struct InterestsView: View {
    @EnvironmentObject private var account: AccountStore
    @Environment(\.dismiss) private var dismiss
    var onboarding = false
    @State private var owner: UUID?
    @State private var draft = UserInterests()
    @State private var selectedSport = InterestSport.nfl
    @State private var query = ""
    @State private var directories: [InterestSport: [InterestTeam]] = [:]
    @State private var loadingSport: InterestSport?
    @State private var directoryError: String?
    @State private var initialized = false

    private var visibleTeams: [InterestTeam] {
        let search = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return (directories[selectedSport] ?? []).filter {
            search.isEmpty || $0.name.localizedCaseInsensitiveContains(search) || $0.abbreviation.localizedCaseInsensitiveContains(search)
        }
    }
    private var isCurrentOwner: Bool { owner != nil && owner == account.user?.id && account.hasSignedInAccount }
    var body: some View {
        Form {
            if account.interestsState.invalidMetadata {
                Section {
                    Label("Your saved interests couldn’t be read. Saving here will replace them with your selections below.", systemImage: "exclamationmark.triangle")
                        .font(.footnote).foregroundStyle(.orange)
                }
            }
            Section {
                LazyVGrid(columns: [.init(.flexible()), .init(.flexible())], spacing: 10) {
                    ForEach(InterestSport.allCases) { sport in
                        let chosen = draft.sports.contains(sport)
                        Button { draft.toggle(sport) } label: {
                            HStack {
                                Text(sport.rawValue).font(.headline)
                                Spacer()
                                Image(systemName: chosen ? "checkmark.circle.fill" : "circle")
                            }.padding(.horizontal, 12).frame(minHeight: 48)
                                .foregroundStyle(chosen ? brandAccent : Color.primary)
                                .plonnGlassSurface(selected:chosen,radius:12)
                        }.buttonStyle(.plain)
                            .accessibilityLabel("Follow \(sport.rawValue)")
                            .accessibilityValue(chosen ? "Selected" : "Not selected")
                            .accessibilityIdentifier("interest-sport-" + sport.rawValue)
                    }
                }.plonnGlassGroup().padding(.vertical, 4)
            } header: { Text("Favorite sports") } footer: {
                Text("You can follow a team without following its league.")
            }
            if !draft.teams.isEmpty {
                Section("Following · \(draft.teams.count) teams") {
                    ForEach(draft.teams.sorted { ($0.sport.rawValue, $0.name) < ($1.sport.rawValue, $1.name) }) { team in
                        Button { draft.toggle(team) } label: {
                            HStack(spacing: 10) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(team.name).foregroundStyle(.primary)
                                    Text(team.sport.rawValue).font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(brandAccent)
                            }.frame(minHeight: 44)
                        }.plonnGlassButton().accessibilityLabel("Unfollow \(team.name), \(team.sport.rawValue)")
                    }
                }
            }
            Section {
                Picker("Team league", selection: $selectedSport) {
                    ForEach(InterestSport.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented).padding(.vertical, 4)
                HStack {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Search \(selectedSport.rawValue) teams", text: $query)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                        .accessibilityIdentifier("interest-team-search")
                    if !query.isEmpty {
                        Button { query = "" } label: { Image(systemName: "xmark.circle.fill").frame(minWidth: 44, minHeight: 44) }
                            .plonnGlassButton().accessibilityLabel("Clear team search")
                    }
                }.frame(minHeight: 44)
                if loadingSport == selectedSport {
                    ProgressView("Loading teams…").frame(minHeight: 44)
                } else if let directoryError {
                    Text(directoryError).font(.footnote).foregroundStyle(.secondary)
                    Button("Retry team list") { Task { await loadTeams(selectedSport, force: true) } }.frame(minHeight: 44)
                } else if visibleTeams.isEmpty {
                    Text(query.isEmpty ? "The team list is unavailable right now." : "No teams match that search.")
                        .font(.footnote).foregroundStyle(.secondary)
                } else {
                    ForEach(visibleTeams) { team in
                        let chosen = draft.teams.contains { $0.id == team.id }
                        Button { draft.toggle(team) } label: {
                            HStack {
                                Text(team.name).foregroundStyle(.primary)
                                Spacer()
                                Image(systemName: chosen ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(chosen ? brandAccent : Color.secondary)
                            }.frame(minHeight: 44)
                        }.plonnGlassButton(prominent:chosen)
                            .accessibilityLabel("Follow \(team.name)")
                            .accessibilityValue(chosen ? "Selected" : "Not selected")
                            .accessibilityIdentifier("interest-team-" + team.id)
                    }
                }
            } header: { Text("Find teams") }
            Section {
                Label("Alerts are not available yet", systemImage: "bell.slash")
                    .font(.subheadline.bold())
                Text("Save what you’d like to receive. PLONN will ask for notification permission when alerts become available. These choices do not enable delivery yet.")
                    .font(.footnote).foregroundStyle(.secondary)
                Toggle("Model results ready", isOn: $draft.notifications.modelReady)
                    .accessibilityIdentifier("interest-alert-model-ready")
                Toggle("Before followed teams play", isOn: $draft.notifications.gameReminders)
                Toggle("Injury and lineup updates", isOn: $draft.notifications.injuryLineup)
                Text("Game, injury and lineup alerts will use your selected teams. Model alerts will be for your own research.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Daily summary", isOn: $draft.notifications.dailyDigest)
                if draft.notifications.dailyDigest {
                    DatePicker("Summary time", selection: minuteBinding(\.digestMinute), displayedComponents: .hourAndMinute)
                }
            } header: { Text("Alerts") }
            Section {
                Toggle("Quiet hours", isOn: $draft.notifications.quietHoursEnabled)
                if draft.notifications.quietHoursEnabled {
                    DatePicker("From", selection: minuteBinding(\.quietStartMinute), displayedComponents: .hourAndMinute)
                    DatePicker("Until", selection: minuteBinding(\.quietEndMinute), displayedComponents: .hourAndMinute)
                    if draft.notifications.quietStartMinute == draft.notifications.quietEndMinute {
                        Label("Choose different start and end times.", systemImage: "exclamationmark.triangle")
                            .font(.footnote).foregroundStyle(.red)
                    }
                }
                LabeledContent("Time zone", value: draft.notifications.timeZoneID.replacingOccurrences(of: "_", with: " "))
                    .font(.footnote)
                if draft.notifications.timeZoneID != TimeZone.current.identifier {
                    Button("Use this device’s time zone") { draft.notifications.timeZoneID = TimeZone.current.identifier }
                }
            } header: { Text("Schedule") } footer: {
                Text("Quiet hours also apply to the daily summary. Time-sensitive game reminders will be skipped if they would arrive after the game starts.")
            }
            if let message = account.interestsSaveMessage {
                Section { Label(message, systemImage: "exclamationmark.triangle").font(.footnote).foregroundStyle(.red) }
            }
        }
        .plonnGlassButton()
        .disabled(account.interestsSaving)
        .scrollContentBackground(.hidden).background(navy)
        .fontDesign(.default).tint(brandAccent)
        .environment(\.timeZone, TimeZone(identifier: draft.notifications.timeZoneID) ?? .current)
        .navigationTitle("Interests & alerts")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button {
                    Task { await save(skip: false) }
                } label: {
                    if account.interestsSaving { ProgressView() } else { Text(onboarding ? "Continue" : "Save").bold() }
                }.disabled(!initialized || !isCurrentOwner || account.busy || !draft.isValid)
                    .accessibilityIdentifier("interests-save")
            }
            if onboarding {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Skip") { Task { await save(skip: true) } }
                        .disabled(!isCurrentOwner || account.busy).accessibilityIdentifier("interests-skip")
                }
            }
        }
        .task {
            guard !initialized else { return }
            owner = account.user?.id
            draft = account.interestsState.preferences
            account.clearInterestsMessage()
            initialized = true
        }
        .task(id: selectedSport) { await loadTeams(selectedSport) }
        .onChange(of: selectedSport) { _, _ in query = "" }
        .onChange(of: account.user?.id) { _, next in
            if next != owner { draft = UserInterests(); directories = [:]; dismiss() }
        }
    }

    private func save(skip: Bool) async {
        guard isCurrentOwner, let owner else { return }
        var value = skip ? account.interestsState.preferences : draft
        value.onboardingCompleted = true
        if await account.saveInterests(value, owner: owner) { dismiss() }
    }
    private func loadTeams(_ sport: InterestSport, force: Bool = false) async {
        if !force, directories[sport] != nil { return }
        loadingSport = sport; directoryError = nil
        do {
            let teams = try await InterestsTeamDirectory().load(sport)
            try Task.checkCancellation()
            directories[sport] = teams
        } catch is CancellationError { return }
        catch { if sport == selectedSport { directoryError = "Couldn’t load \(sport.rawValue) teams. Your saved choices are unchanged." } }
        if loadingSport == sport { loadingSport = nil }
    }
    private func minuteBinding(_ keyPath: WritableKeyPath<InterestNotifications, Int>) -> Binding<Date> {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: draft.notifications.timeZoneID) ?? .current
        let fixedCalendar = calendar
        return Binding(get: {
            let minute = draft.notifications[keyPath: keyPath]
            return fixedCalendar.date(from: DateComponents(year: 2026, month: 1, day: 1, hour: minute / 60, minute: minute % 60)) ?? Date()
        }, set: { date in
            draft.notifications[keyPath: keyPath] = fixedCalendar.component(.hour, from: date) * 60 + fixedCalendar.component(.minute, from: date)
        })
    }
}
