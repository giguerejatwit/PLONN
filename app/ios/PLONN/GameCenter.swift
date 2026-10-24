import SwiftUI
import Charts

struct SportsItem: Identifiable, Codable {
    let id: String
    let name: String
    var subtitle = ""
    var home = ""
    var away = ""
    var homeScore = "—"
    var awayScore = "—"
    var live = false
    var state = ""
    var start = ""
    var homeLogo = ""
    var awayLogo = ""
    var homeColor = ""
    var awayColor = ""
    var homeAlternateColor = ""
    var awayAlternateColor = ""
    var battingTeam:String? = nil
    var pitchingTeam:String? = nil
    var listedPitchers:String? = nil
    var liveBatter:String? = nil
    var livePitcher:String? = nil
    var possession = ""
    var downDistance = ""
    var fieldPosition = ""
    var balls:Int? = nil
    var strikes:Int? = nil
    var outs:Int? = nil
    var firstBase:Bool? = nil
    var secondBase:Bool? = nil
    var thirdBase:Bool? = nil
    var seasonType:Int? = nil
}

@MainActor final class SportsDirectory: ObservableObject {
    @Published var items: [SportsItem] = []
    @Published var status = ""
    @Published var loading = false
    private struct DiskSnapshot:Codable {let fetched:Date;let items:[SportsItem];let status:String}
    private var version = 0
    private var cache: [String: (Date, [SportsItem], String)] = [:]
    private func json(_ url: String) async throws -> [String: Any] {
        var request = URLRequest(url: URL(string: url)!)
        request.timeoutInterval = 25
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw URLError(.cannotParseResponse) }
        return object
    }
    private func rows(_ value: Any?) -> [[String: Any]] { value as? [[String: Any]] ?? [] }
    private func text(_ value: Any?) -> String { value.map { String(describing: $0) } ?? "" }
    func load(sport: String, view: String, date: Date) async {
        version += 1
        let requestVersion = version
        let formatter = DateFormatter(); formatter.dateFormat = "yyyyMMdd"; formatter.locale = Locale(identifier: "en_US_POSIX")
        let day = formatter.string(from: date)
        let key = sport + view + (view == "Games" ? day : "")
        let diskKey="sportsSnapshot.v1."+key
        if cache[key] == nil,let data=UserDefaults.standard.data(forKey:diskKey),let snapshot=try? JSONDecoder().decode(DiskSnapshot.self,from:data),Date().timeIntervalSince(snapshot.fetched)<86400 {
            cache[key]=(snapshot.fetched,snapshot.items,snapshot.status)
        }
        let ttl: TimeInterval = view == "Games" ? 30 : 3600
        if let cached = cache[key], Date().timeIntervalSince(cached.0) < ttl {
            items = cached.1; status = cached.2; loading = false; return
        }
        items = cache[key]?.1 ?? []; loading = true; status = "Loading \(sport) \(view.lowercased())…"
        defer { if requestVersion == version { loading = false } }
        do {
            let base = "https://site.api.espn.com/apis/site/v2/sports/" + sportsFeedPath(sport)
            var result: [SportsItem] = []
            var source = "ESPN public preview feed"
            var missing = 0
            if view == "Games" {
                let response = try await json(base + "/scoreboard?limit=1000&dates=" + day)
                for event in rows(response["events"]) {
                    guard let competition = rows(event["competitions"]).first else { continue }
                    let status = competition["status"] as? [String: Any] ?? event["status"] as? [String: Any] ?? [:]
                    let type = status["type"] as? [String: Any] ?? [:]
                    let competitors = rows(competition["competitors"])
                    guard let home = competitors.first(where: { text($0["homeAway"]) == "home" }), let away = competitors.first(where: { text($0["homeAway"]) == "away" }) else { continue }
                    let homeTeam = home["team"] as? [String: Any] ?? [:], awayTeam = away["team"] as? [String: Any] ?? [:]
                    let situation=competition["situation"] as? [String:Any] ?? [:]
                    func livePlayer(_ key:String)->String? {
                        guard let player=situation[key] as? [String:Any],let athlete=player["athlete"] as? [String:Any] else{return nil}
                        let name=text(athlete["displayName"])
                        guard !name.isEmpty else{return nil}
                        let summary=text(player["summary"])
                        return name + (summary.isEmpty ? "" : " · " + summary)
                    }
                    func liveTeam(_ key:String)->String? {
                        guard let player=situation[key] as? [String:Any],let athlete=player["athlete"] as? [String:Any],let team=athlete["team"] as? [String:Any] else{return nil}
                        let id=text(team["id"])
                        guard let competitor=competitors.first(where:{text($0["id"]) == id}),let details=competitor["team"] as? [String:Any] else{return nil}
                        return text(details["displayName"])
                    }
                    let starters=competitors.compactMap { competitor -> String? in
                        guard let probable=rows(competitor["probables"]).first,let athlete=probable["athlete"] as? [String:Any],let team=competitor["team"] as? [String:Any] else{return nil}
                        return text(team["abbreviation"])+" · "+text(athlete["displayName"])
                    }.joined(separator:" / ")
                    let possessionID=text(situation["possession"])
                    let possession=competitors.first{ text(($0["team"] as? [String:Any])?["id"]) == possessionID }
                    let possessionTeam=possession?["team"] as? [String:Any] ?? [:]
                    let down=situation["down"] as? Int ?? 0
                    let distance=situation["distance"] as? Int
                    let downText=(1...4).contains(down) ? (text(situation["shortDownDistanceText"]).isEmpty ? "\(down)\(down == 1 ? "st":down == 2 ? "nd":down == 3 ? "rd":"th") & \(distance.map(String.init) ?? "—")":text(situation["shortDownDistanceText"])) : ""
                    result.append(SportsItem(id: text(event["id"]), name: text(event["name"]), subtitle: text(type["detail"]), home: text(homeTeam["displayName"]), away: text(awayTeam["displayName"]), homeScore: text(type["state"]) == "pre" ? "—" : text(home["score"]), awayScore: text(type["state"]) == "pre" ? "—" : text(away["score"]), live: text(type["state"]) == "in", state: text(type["state"]), start: text(event["date"]), homeLogo: text(homeTeam["logo"]), awayLogo: text(awayTeam["logo"]), homeColor: text(homeTeam["color"]), awayColor: text(awayTeam["color"]), homeAlternateColor:text(homeTeam["alternateColor"]), awayAlternateColor:text(awayTeam["alternateColor"]),battingTeam:liveTeam("batter"),pitchingTeam:liveTeam("pitcher"),listedPitchers:starters.isEmpty ? nil:starters,liveBatter:livePlayer("batter"),livePitcher:livePlayer("pitcher"),possession:text(possessionTeam["abbreviation"]),downDistance:downText,fieldPosition:(1...4).contains(down) ? text(situation["possessionText"]):"",balls:situation["balls"] as? Int,strikes:situation["strikes"] as? Int,outs:situation["outs"] as? Int,firstBase:situation["onFirst"] as? Bool,secondBase:situation["onSecond"] as? Bool,thirdBase:situation["onThird"] as? Bool,seasonType:NBARecordedOddsRouting.seasonType(event:event)))
                }
            } else if sport == "MLB" {
                source = "MLB Stats API"
                let year = Calendar.current.component(.year, from: Date())
                let directory = try await json("https://statsapi.mlb.com/api/v1/teams?sportId=1&season=\(year)")
                let teams = rows(directory["teams"])
                if view == "Teams" {
                    result = teams.map { SportsItem(id: text($0["id"]), name: text($0["name"]), subtitle: text($0["abbreviation"])) }
                } else {
                    let names = Dictionary(uniqueKeysWithValues: teams.map { (text($0["id"]), text($0["name"])) })
                    let roster = try await json("https://statsapi.mlb.com/api/v1/sports/1/players?season=\(year)")
                    result = rows(roster["people"]).map { player in
                        let team = player["currentTeam"] as? [String: Any] ?? [:], position = player["primaryPosition"] as? [String: Any] ?? [:]
                        return SportsItem(id: text(player["id"]), name: text(player["fullName"]), subtitle: (names[text(team["id"])] ?? "Unassigned / outside MLB roster") + " · " + text(position["abbreviation"]))
                    }
                    source += " · Season players; game availability unverified"
                }
            } else {
                let directory = try await json(base + "/teams?limit=1000")
                let teams = rows(directory["sports"]).flatMap { rows($0["leagues"]) }.flatMap { rows($0["teams"]) }.compactMap { $0["team"] as? [String: Any] }.filter { ($0["isAllStar"] as? Bool) != true }
                if view == "Teams" {
                    result = teams.map { SportsItem(id: text($0["id"]), name: text($0["displayName"]), subtitle: text($0["abbreviation"])) }
                } else {
                    for team in teams {
                        try Task.checkCancellation()
                        do {
                            let roster = try await json(base + "/teams/" + text(team["id"]) + "/roster")
                            let athletes = rows(roster["athletes"]).flatMap { group in group["items"] != nil ? rows(group["items"]) : [group] }
                            if athletes.isEmpty { missing += 1 }
                            result += athletes.map { player in
                                let position = player["position"] as? [String: Any] ?? [:]
                                return SportsItem(id: text(team["id"]) + ":" + text(player["id"]), name: text(player["displayName"]), subtitle: text(team["displayName"]) + " · " + text(position["abbreviation"]))
                            }
                        } catch is CancellationError { throw CancellationError() } catch { missing += 1 }
                        try Task.checkCancellation()
                        guard requestVersion == version else { return }
                        // Surface partial directory results as each roster arrives.
                        items = result
                    }
                    if missing == teams.count && !teams.isEmpty { throw URLError(.badServerResponse) }
                }
            }
            try Task.checkCancellation()
            guard requestVersion == version else { return }
            let fetched = Date()
            let message = "\(result.count) \(view.lowercased()) · Fetched \(fetched.formatted(date: .omitted, time: .standard)) · \(source)" + (missing > 0 ? " · \(missing) team rosters unavailable" : "")
            cache[key] = (fetched, result, message); items = result; status = message
            if let data=try? JSONEncoder().encode(DiskSnapshot(fetched:fetched,items:result,status:message)) {UserDefaults.standard.set(data,forKey:diskKey)}
        } catch {
            guard requestVersion == version else { return }
            status = cache[key].map { "STALE · " + $0.2 + ". Refresh failed." } ?? "Sports feed unavailable. Pull to retry."
        }
    }
}

struct GameCenterView: View {
    @StateObject private var directory = SportsDirectory()
    @Environment(\.scenePhase) private var scenePhase
    @State private var sport = "MLB"
    @State private var view = "Games"
    @State private var date = Date()
    @State private var query = ""
    private var filtered: [SportsItem] {
        let words = query.lowercased().split(separator: " ")
        return directory.items.filter { item in words.allSatisfy { (item.name + " " + item.subtitle).lowercased().contains($0) } }
    }
    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                Brand()
                Picker("League", selection: $sport) { Text("NBA").tag("NBA"); Text("MLB").tag("MLB"); Text("NFL").tag("NFL") }.pickerStyle(.segmented)
                Picker("Browse", selection: $view) { ForEach(["Games", "Teams", "Players"], id: \.self) { Text($0) } }.pickerStyle(.segmented)
                if view == "Games" { DatePicker("Game date", selection: $date, displayedComponents: .date) }
                TextField("Search teams and players", text: $query).textFieldStyle(.roundedBorder).accessibilityIdentifier("directory-search")
                Text(directory.status).font(.system(.caption, design: .monospaced)).foregroundStyle(.orange).accessibilityIdentifier("directory-status")
                if directory.loading { PLONNLoader() }
                if filtered.isEmpty && !directory.loading { Text("No \(view.lowercased()) found for this selection.").foregroundStyle(.secondary) }
                ForEach(filtered) { item in
                    Panel {
                        VStack(alignment: .leading, spacing: 12) {
                            if view == "Games" {
                                Text(item.subtitle).font(.system(.caption, design: .monospaced)).foregroundStyle(item.live ? charcoalAccent : .secondary)
                                HStack { Text(item.away); Spacer(); Text(item.awayScore).bold() }
                                HStack { Text(item.home); Spacer(); Text(item.homeScore).bold() }
                                NavigationLink("Team & player analytics") { ExploreView(initialSport: sport, initialQuery: item.home, live: true) }
                            } else {
                                Text(item.name).font(.system(.headline, design: .monospaced))
                                Text(item.subtitle).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                                if view == "Teams" { Button("View players") { query = item.name; view = "Players" } }
                                else { NavigationLink("Player analytics") { ExploreView(initialSport: sport, initialQuery: item.name, live: true) } }
                            }
                        }
                    }
                }
                Text("Scores refresh every 30 seconds while active. Provider delays may apply. Player directory membership does not confirm game availability. Sportsbook coverage depends on offered markets.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            }.padding()
        }.background(navy).navigationTitle("Game center")
        .task(id: sport + view + date.description + String(describing: scenePhase)) {
            guard scenePhase == .active else { return }
            repeat {
                await directory.load(sport: sport, view: view, date: date)
                guard view == "Games" else { return }
                do { try await Task.sleep(for: .seconds(30)) } catch { return }
            } while !Task.isCancelled
        }
        .plonnRefreshable { await directory.load(sport: sport, view: view, date: date) }
    }
}

private let todayCanvas = navy
private let todaySurface = navySurface

/// Readable identity fallback when a portrait or team logo is unavailable.
struct SubjectBadge: View {
    let name: String
    var size: CGFloat = 52
    var isPlayer = false
    private var initials: String { SubjectBadgeInitials.make(name) }
    var body: some View {
        Group {
            if initials.isEmpty {
                Image(systemName: isPlayer ? "person.fill" : "sportscourt")
                    .font(.system(size: size * 0.38))
            } else {
                Text(initials).font(.system(size: size * 0.28, weight: .semibold, design: .monospaced))
                    .lineLimit(1).minimumScaleFactor(0.7)
            }
        }.foregroundStyle(brandAccent.opacity(0.9))
            .frame(width: size, height: size)
            .background(brandAccent.opacity(0.10), in: Circle())
            .overlay(Circle().strokeBorder(brandAccent.opacity(0.18), lineWidth: 1))
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(name)
            .accessibilityValue((isPlayer ? "Player" : "Team") + (initials.isEmpty ? " icon" : " initials"))
    }
}

struct TeamMark: View {
    let url: String
    let name: String
    var body: some View {
        let asset = "team-" + name.lowercased().map { $0.isLetter || $0.isNumber ? String($0) : "-" }.joined()
        Group {
            if let bundled = UIImage(named: asset) {
                Image(uiImage: bundled).resizable().scaledToFit()
            } else {
                AsyncImage(url: URL(string: url)) { image in
                    image.resizable().scaledToFit()
                } placeholder: {
                    Text(name.split(separator: " ").prefix(3).compactMap(\.first).map(String.init).joined()).font(.system(size: 9, weight: .bold,design:.monospaced)).foregroundStyle(.secondary)
                }
            }
        }.frame(width: 28, height: 28).padding(2).background(Color.black.opacity(0.06), in: Circle()).accessibilityHidden(true)
    }
}

struct WeekDateStrip: View {
    @Binding var date:Date
    @State private var week=0
    private let anchor=Calendar.current.dateInterval(of:.weekOfYear,for:Date())!.start
    private func day(_ offset:Int,_ index:Int)->Date { Calendar.current.date(byAdding:.day,value:offset*7+index,to:anchor)! }
    var body:some View {
        VStack(spacing:2) {
            HStack {
                Text(day(week,0).formatted(.dateTime.month(.abbreviated).year())).font(.system(.caption, design: .monospaced).weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button("Today") { date=Date();withAnimation {week=0} }.font(.system(.caption, design: .monospaced).weight(.semibold))
            }.padding(.horizontal,20)
            TabView(selection:$week) {
                ForEach(-52...52,id:\.self) { offset in
                    HStack(spacing:0) {
                        ForEach(0..<7,id:\.self) { index in
                            let value=day(offset,index)
                            let selected=Calendar.current.isDate(value,inSameDayAs:date)
                            Button {date=value} label: {
                                VStack(spacing:5) {
                                    Text(value.formatted(.dateTime.weekday(.abbreviated))).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                                    Text(value.formatted(.dateTime.day())).font(.system(.subheadline, design: .monospaced).weight(.semibold)).monospacedDigit().frame(width:38,height:38)
                                        .foregroundStyle(.primary).plonnGlassSurface(selected:selected,radius:19)
                                }.frame(maxWidth:.infinity).padding(.vertical,4)
                            }.buttonStyle(.plain).accessibilityLabel(value.formatted(date:.complete,time:.omitted)).accessibilityAddTraits(selected ? .isSelected:[])
                        }
                    }.plonnGlassGroup().padding(.horizontal,12).tag(offset)
                }
            }.tabViewStyle(.page(indexDisplayMode:.never)).frame(height:72).accessibilityIdentifier("games-week-dates")
        }.padding(.top,10)
    }
}

private struct SlateScrollOffset: PreferenceKey {
    static var defaultValue:CGFloat = 0
    static func reduce(value:inout CGFloat,nextValue:()->CGFloat) {value=nextValue()}
}

struct TodayView: View {
    @StateObject private var moneylines = Store()
    @EnvironmentObject private var account: AccountStore
    @StateObject private var directory = SportsDirectory()
    @Environment(\.scenePhase) private var scenePhase
    @State private var sport = "All"
    @State private var soccerCompetition="fifa.friendly"
    @State private var tennisTour="ATP"
    private var feedSport:String {sport == "Soccer" ? "Soccer:"+soccerCompetition:sport == "Tennis" ? "Tennis:"+tennisTour:sport}
    private var eventSport:Bool {["Tennis","Golf","UFC/MMA","Boxing","F1","NASCAR"].contains(sport)}
    @State private var date = Date()
    @State private var query = ""
    @State private var showDate = false
    @State private var calendarCollapsed = false
    @State private var findingGameDay=false
    @State private var dateMessage=""
    private var selectedDay: String { date.formatted(.dateTime.year().month().day()) }
    private var filtered: [SportsItem] {
        directory.items.filter { query.isEmpty || $0.name.localizedCaseInsensitiveContains(query) }
    }
    var body: some View {
        VStack(spacing: 0) {
            header
            Text("Games").font(.system(.title2,design:.monospaced).bold()).frame(maxWidth:.infinity,alignment:.leading).padding(.horizontal,16)
            if !account.hasSignedInAccount {
                HStack {
                    Text("Preview · Games and statistics").foregroundStyle(.secondary)
                    Spacer()
                    Button("Sign in") {account.access.present(feature:.odds)}.fontWeight(.semibold)
                        .accessibilityIdentifier("games-sign-in")
                }.font(.caption).padding(.horizontal,16).padding(.vertical,6)
            }
            leagueSelectors
            WeekDateStrip(date:$date)
                .frame(height:calendarCollapsed ? 0 : 132,alignment:.top)
                .opacity(calendarCollapsed ? 0 : 1)
                .clipped()
                .allowsHitTesting(!calendarCollapsed)
                .accessibilityHidden(calendarCollapsed)
            if sport == "All" {AllGamesSlate(date:date,query:query)} else if eventSport {OtherSportEvents(sport:feedSport,date:date)} else { ScrollView {
                LazyVStack(alignment:.leading,spacing:12) {
                    GeometryReader { proxy in
                        Color.clear.preference(key:SlateScrollOffset.self,value:proxy.frame(in:.named("slateScroll")).minY)
                    }.frame(height:0)
                    if findingGameDay { PLONNLoader("Finding game day…") }
                    if !dateMessage.isEmpty { Text(dateMessage).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
                    if ["NBA","MLB","NFL","NHL"].contains(sport) && Calendar.current.isDateInToday(date) && filtered.contains(where: \.live) { HotLiveOddsCarousel(sport: sport, isToday: Calendar.current.isDateInToday(date), games:filtered) }
                    if directory.loading && directory.items.isEmpty { PLONNLoader("Loading games").frame(maxWidth:.infinity).padding() }
                    if directory.status.hasPrefix("STALE") || directory.status.contains("unavailable") { Text(directory.status).font(.system(.caption, design: .monospaced)).foregroundStyle(.orange) }
                    if let issue=moneylines.oddsIssue {OddsIssueNotice(message:issue)}
                    section("In progress", games: filtered.filter { $0.state == "in" })
                    section("Upcoming", games: filtered.filter { $0.state != "in" && $0.state != "post" })
                    section("Final", games: filtered.filter { $0.state == "post" })
                    if filtered.isEmpty && !directory.loading && !directory.status.contains("unavailable") && !directory.status.hasPrefix("STALE") {
                        VStack(spacing:18) {
                            Image(systemName:"calendar").font(.system(size:30,design:.monospaced)).foregroundStyle(.secondary)
                            Text(query.isEmpty ? "No \(sport) games \(Calendar.current.isDateInToday(date) ? "today":"on this date")":"No matching games").font(.system(.title3, design: .monospaced).weight(.semibold))
                            if query.isEmpty {
                                Button {moveDay(1)} label: {
                                    HStack(spacing:10) {Text("Next game day");Image(systemName:"arrow.right")}
                                        .font(.system(.subheadline, design: .monospaced).weight(.semibold)).padding(.horizontal,20).padding(.vertical,14)
                                        .foregroundStyle(.primary).plonnGlassSurface(selected:true,radius:18)
                                }.buttonStyle(.plain).disabled(findingGameDay).opacity(findingGameDay ? 0.5:1)
                            }
                        }.frame(maxWidth:.infinity).padding(.vertical,60)
                    }
                }.padding(16)
            }.coordinateSpace(name:"slateScroll")
            .onPreferenceChange(SlateScrollOffset.self) { offset in
                let collapsed = offset < -40
                if collapsed != calendarCollapsed {
                    withAnimation(.easeInOut(duration:0.22)) { calendarCollapsed=collapsed }
                }
            }
            .plonnRefreshable {
                await directory.load(sport:feedSport,view:"Games",date:date)
                if LaunchSports.includes(feedSport) {await moneylines.load(feedSport,account:account,cacheOnly:true,nbaSeasonTypes:directory.items.map(\.seasonType))}
            }
            }
        }.background(LeagueBackdrop(league:sport)).toolbar(.hidden,for:.navigationBar)
        .sheet(isPresented:$showDate) { VStack { DatePicker("Game date",selection:$date,displayedComponents:.date).datePickerStyle(.graphical);Button("Done"){showDate=false}.plonnGlassButton(prominent:true).foregroundStyle(.primary) }.padding().presentationDetents([.medium]) }
        .task(id:feedSport+selectedDay+directory.items.map {"\($0.id):\($0.seasonType ?? -1)"}.joined()+String(describing:scenePhase)+(account.user?.id.uuidString ?? "guest")) {
            guard scenePhase == .active else { return }
            guard LaunchSports.includes(feedSport) else{return}
            await moneylines.load(feedSport,account:account,forceRefresh:false,cacheOnly:true,nbaSeasonTypes:directory.items.map(\.seasonType))
            NotificationCenter.default.post(name:Notification.Name("PLONNInitialOddsReady"),object:nil)
        }
        .task(id:feedSport+selectedDay+String(describing:scenePhase)) {
            guard scenePhase == .active, !eventSport, sport != "All" else { return }
            repeat {
                await directory.load(sport:feedSport,view:"Games",date:date)
                NotificationCenter.default.post(name:Notification.Name("PLONNInitialScheduleReady"),object:nil)
                do { try await Task.sleep(for:.seconds(30)) } catch { return }
            } while !Task.isCancelled
        }
    }
    private var header:some View {
        VStack(spacing:8) {
            HStack {
                PLONNTrident().frame(width:24,height:28)
                PLONNWordmark(size:17)
                Spacer()
                NavigationLink { SettingsView() } label: {
                    Image(systemName:"person.crop.circle").font(.system(size:22,design:.monospaced))
                        .foregroundStyle(.primary).frame(width:44,height:44)
                        .plonnGlassSurface(radius:22)
                }.buttonStyle(.plain).accessibilityLabel("Account and settings")
            }
            HStack(spacing:10) {
                Image(systemName:"magnifyingglass").foregroundStyle(.secondary)
                TextField("Search today's teams",text:$query).font(.system(.body,design:.monospaced))
                    .submitLabel(.search).accessibilityIdentifier("today-search")
                if !query.isEmpty {
                    Button {query=""} label: {
                        Image(systemName:"xmark.circle.fill").foregroundStyle(.secondary).frame(width:44,height:44)
                    }.plonnGlassButton().accessibilityLabel("Clear search")
                }
            }.padding(.horizontal,14).frame(minHeight:48)
                .plonnGlassSurface(radius:24,interactive:false)
        }.plonnGlassGroup().padding(.horizontal,16).padding(.bottom,12)
    }
    @ViewBuilder private var leagueSelectors:some View {
            ScrollView(.horizontal,showsIndicators:false) { HStack(spacing: 8) {
                ForEach(["All"] + LaunchSports.supported, id: \.self) { league in
                    Button { sport = league; query = "" } label: {
                        VStack(spacing:4) {LeagueMark(league:league);Text(league).font(.system(.caption, design: .monospaced).weight(.semibold))}
                            .foregroundStyle(sport == league ? Color.primary : .secondary).frame(minWidth:62,minHeight:62)
                            .plonnGlassSurface(selected:sport == league,radius:20)
                    }.buttonStyle(.plain).accessibilityAddTraits(sport == league ? .isSelected : [])
                }
                Spacer(minLength: 0)
            }.plonnGlassGroup().padding(.horizontal,16).padding(.vertical,8) }
            if sport == "Tennis" {Picker("Tour",selection:$tennisTour) {Text("ATP").tag("ATP");Text("WTA").tag("WTA")}.pickerStyle(.segmented).padding(.horizontal,16)}
            if sport == "Soccer" {
                ScrollView(.horizontal,showsIndicators:false) {
                    HStack(spacing:12) {
                        ForEach([("fifa.friendly","Friendlies","globe"),("uefa.nations","Nations League","flag.fill"),("caf.nations","AFCON","globe.africa.fill"),("uefa.euro","Euros","trophy.fill"),("conmebol.america","Copa América","globe.americas.fill"),("eng.1","Premier League","crown.fill"),("uefa.champions","Champions League","star.circle.fill"),("usa.1","MLS","soccerball")],id:\.0) { item in
                            Button {soccerCompetition=item.0} label: {
                                VStack(spacing:7) {
                                    Image(systemName:item.2).font(.system(.title2, design: .monospaced)).frame(width:46,height:46)
                                        .foregroundStyle(.primary).plonnGlassSurface(selected:soccerCompetition == item.0,radius:12)
                                    Text(item.1).font(.system(.caption2, design: .monospaced).weight(.semibold)).multilineTextAlignment(.center).frame(width:76)
                                }.padding(.vertical,8)
                            }.buttonStyle(.plain).accessibilityAddTraits(soccerCompetition == item.0 ? .isSelected:[])
                        }
                    }.padding(.horizontal,16)
                }
            }
    }
    private func moveDay(_ offset:Int) {
        guard !findingGameDay else{return}
        let selectedSport=feedSport
        let selected=date
        findingGameDay=true;dateMessage=""
        Task { @MainActor in
            defer{findingGameDay=false}
            do {
                let calendar=Calendar.current
                let formatter=DateFormatter();formatter.dateFormat="yyyyMMdd"
                for distance in 1...28 {
                    try Task.checkCancellation()
                    let candidate=calendar.date(byAdding:.day,value:(offset>0 ? distance:-distance),to:selected)!
                    let url=URL(string:"https://site.api.espn.com/apis/site/v2/sports/"+sportsFeedPath(selectedSport)+"/scoreboard?limit=1000&dates="+formatter.string(from:candidate))!
                    let(data,response)=try await URLSession.shared.data(from:url)
                    guard (response as? HTTPURLResponse)?.statusCode==200,let json=try JSONSerialization.jsonObject(with:data) as? [String:Any] else{throw URLError(.badServerResponse)}
                    guard feedSport == selectedSport,date==selected else{return}
                    let hasGame=(json["events"] as? [[String:Any]] ?? []).contains { event in
                        guard let raw=event["date"] as? String,let start=sportsDate(raw) else{return false}
                        return calendar.isDate(start,inSameDayAs:candidate)
                    }
                    if hasGame {date=candidate;return}
                }
                dateMessage="No games found in the next four weeks."
            } catch {dateMessage="Couldn’t load the schedule. Try again."}
        }
    }
    @ViewBuilder private func section(_ title:String,games:[SportsItem]) -> some View {
        if !games.isEmpty {
            Text("\(title) · \(games.count)").font(.system(size:12,weight:.semibold,design:.monospaced)).textCase(.uppercase).tracking(1).foregroundStyle(.secondary).padding(.top,8)
            ForEach(games) { game in
                NavigationLink { if !["NBA","MLB","NFL","NHL"].contains(sport) {SoccerMatchupView(game:game,sport:feedSport)} else {TodayGameDetail(game:game,sport:sport)} } label: { scoreCard(game) }.buttonStyle(.plain)
            }
        }
    }
    @ViewBuilder private func scoreCard(_ game:SportsItem)->some View {
        if game.state == "in" || game.state == "post" { liveScoreCard(game) }
        else {ScheduledOddsCard(game:game,sport:sport,markets:moneylines.live,loading:moneylines.loading,cachedOnly:moneylines.oddsCachedOnly,asOf:moneylines.lastOddsRefresh)}
    }
    private func liveScoreCard(_ game:SportsItem)->some View {SharedLiveScoreCard(game:game,sport:sport)}
}

private struct ScheduledOddsCard:View {
    @EnvironmentObject private var account:AccountStore
    let game:SportsItem
    let sport:String
    let markets:[Market]
    let loading:Bool
    var cachedOnly=false
    var asOf:Date?=nil
    var body:some View {
            VStack(spacing:12) {
                HStack { Text("MONEYLINE").font(.system(.caption2, design: .monospaced).weight(.semibold)).foregroundStyle(.secondary);Spacer();Text("ODDS · IMPLIED %").font(.system(.caption2, design: .monospaced).weight(.semibold)).foregroundStyle(.secondary) }
                moneylineRow(game.away,logo:game.awayLogo,game:game)
                moneylineRow(game.home,logo:game.homeLogo,game:game)
                HStack {
                    if let start=sportsDate(game.start) { Text(start,style:.time) }
                    Spacer()
                    Text(!account.hasSignedInAccount ? "Games and statistics preview":(cachedOnly ? (asOf.map{Store.oddsAsOf($0)} ?? "Price time unavailable"):"Snapshot · Includes book margin"))
                }.font(.system(size:10,design:.monospaced)).foregroundStyle(.secondary)
            }.padding(14).background(TeamCardBackground(game:game))
    }
    private func moneyline(_ team:String,game:SportsItem)->Market? {
        guard account.hasSignedInAccount,let start=sportsDate(game.start) else {return nil}
        return Store.recordedMoneyline(team,markets:markets,sport:sport,home:game.home,away:game.away,start:start)
    }
    private func moneylineRow(_ name:String,logo:String,game:SportsItem)->some View {
        HStack(spacing:8) {
            TeamMark(url:logo,name:name)
            Text(name).font(.system(size:14,weight:.medium,design:.monospaced)).lineLimit(2)
            Spacer(minLength:4)
            if let market=moneyline(name,game:game) {
                VStack(alignment:.trailing,spacing:3) {
                    HStack(spacing:8) { Text(odds(market.price)).foregroundStyle(charcoalAccent);Text(probability(implied(market.price))) }.font(.system(size:14,weight:.semibold,design:.monospaced)).monospacedDigit()
                    Text(market.book).font(.system(size:9,design:.monospaced)).foregroundStyle(.secondary).lineLimit(1)
                }
            } else { Text(account.hasSignedInAccount ? (loading ? "Loading…":"Odds unavailable"):"Sign in for odds").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary) }
        }
    }
}

private struct OddsIssueNotice:View {
 @EnvironmentObject private var account:AccountStore
 let message:String
 var body:some View {
  let presentation=OddsIssuePresentation(message)
  if account.hasSignedInAccount {VStack(alignment:.leading,spacing:4) {
   if let asOf=presentation.asOf {
    Text(asOf).font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("odds-as-of")
   }
   if let status=presentation.status {
    Text(status).font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("odds-feed-status")
   }
   if let warning=presentation.warning {
    Label(warning,systemImage:"exclamationmark.triangle")
     .font(.system(.caption,design:.monospaced)).foregroundStyle(.orange)
     .accessibilityIdentifier("odds-feed-issue")
   }
  }}
 }
}

private struct SharedLiveScoreCard:View {
 let game:SportsItem
 let sport:String
 var body:some View {
        HStack(spacing:12) {
            VStack(spacing:10) {
                teamRow(game.away,logo:game.awayLogo,score:game.awayScore)
                teamRow(game.home,logo:game.homeLogo,score:game.homeScore)
            }
            Rectangle().fill(Color.black.opacity(0.08)).frame(width:1)
            VStack(alignment:.center,spacing:6) {

                Text(sport == "NFL" ? game.subtitle.replacingOccurrences(of:" Quarter",with:"").replacingOccurrences(of:" quarter",with:"") : game.subtitle).font(.system(size:12,weight:.semibold,design:.monospaced)).fixedSize(horizontal:false,vertical:true)
                if game.live {
                 if sport == "NFL" {
                  if !game.possession.isEmpty {HStack(spacing:4) {Text(game.possession);Image(systemName:"american.football.fill")}.font(.system(.caption2, design: .monospaced).bold()).accessibilityElement(children:.ignore).accessibilityLabel(game.possession+" possession")}
                  if !game.downDistance.isEmpty {Text(game.downDistance).font(.system(.caption2, design: .monospaced).bold())}
                  if !game.fieldPosition.isEmpty {Text(game.fieldPosition).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)}
                 } else if sport == "MLB" {
                  BaseOccupancy(game:game).frame(maxWidth:.infinity,alignment:.center)
                  if let balls=game.balls,let strikes=game.strikes {Text("\(balls) B · \(strikes) S").font(.system(.caption2, design: .monospaced).bold())}
                  if let outs=game.outs {Text("\(outs) out\(outs == 1 ? "":"s")").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)}
                 }
                }
            }.multilineTextAlignment(.center).frame(width:76,alignment:.center)
        }.padding(14).background(TeamCardBackground(game:game))

 }
    private func teamRow(_ name:String,logo:String,score:String) -> some View {
        VStack(alignment:.leading,spacing:5) {
        HStack(spacing:8) { TeamMark(url:logo,name:name);Text(name).font(.system(size:14,weight:.medium,design:.monospaced)).lineLimit(2);Spacer(minLength:2);Text(score.isEmpty ? "—":score).font(.system(size:18,weight:.semibold,design:.monospaced)).monospacedDigit() }
        if game.live && sport == "MLB" {
         if game.battingTeam == name,let batter=game.liveBatter {Text(batter).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)}
         if game.pitchingTeam == name,let pitcher=game.livePitcher {Text(pitcher).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)}
        }
        }

    }
}

private struct TodayGameDetail: View {
    let game: SportsItem
    let sport: String
    @StateObject private var markets = Store()
    @Environment(\.scenePhase) private var scenePhase
    @EnvironmentObject private var account: AccountStore
    @State private var category = "Overview"
    @StateObject private var research = MatchupResearch()
    @StateObject private var gameUpdates = LiveMatchupStore()
    private var event: SharedMatchupEvent { game.matchupEvent(sport:sport) }
    private var liveSnapshot: LiveMatchupSnapshot? { gameUpdates.snapshot?.event == event ? gameUpdates.snapshot : nil }
    @State private var propType = "All props"
    @State private var search = ""
    private var matchupMarkets: [Market] {
        guard !LiveMatchupPolicy.showsGameContext(event:event,state:game.state,snapshot:liveSnapshot,now:Date()),account.hasSignedInAccount,let start=sportsDate(game.start) else {return []}
        return Store.recordedEventMarkets(markets.live,sport:sport,home:game.home,away:game.away,start:start)
    }
    private var matching: [Market] {
        guard !LiveMatchupPolicy.showsGameContext(event:event,state:game.state,snapshot:liveSnapshot,now:Date()),account.hasSignedInAccount,let start=sportsDate(game.start) else {return []}
        return markets.live.filter { market in
            (category == "Player props" ? market.kind == "prop" : market.kind == "team")
                && (category != "Player props" || propType == "All props" || propTitle(market.market) == propType)
                && (category != "Player props" || search.isEmpty || (market.name + " " + market.label).localizedCaseInsensitiveContains(search))
                && Store.matchesRecordedEvent(market,sport:sport,home:game.home,away:game.away,start:start,candidates:markets.live)
        }
    }
    var body: some View {
        ScrollViewReader { scroll in
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                VStack(spacing: 14) {
                    HStack { TeamMark(url:game.awayLogo,name:game.away);Text(game.away);Spacer();Text(liveSnapshot?.phase == .scheduled ? game.awayScore : (liveSnapshot?.away.score ?? game.awayScore)).bold() }
                    HStack { TeamMark(url:game.homeLogo,name:game.home);Text(game.home);Spacer();Text(liveSnapshot?.phase == .scheduled ? game.homeScore : (liveSnapshot?.home.score ?? game.homeScore)).bold() }
                    Text(liveSnapshot?.status ?? game.subtitle).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)

                }.padding(16).background(TeamCardBackground(game:game)).id("matchup-top")
                if game.state == "pre", let start = sportsDate(game.start), start > Date(), [2,3].contains(game.seasonType ?? 2) {
                    NavigationLink {
                        UpcomingModelView(setup: .init(league: sport, eventID: game.id, home: game.home, away: game.away, startsAt: game.start), autoStart: true)
                    } label: {
                        Label("Build model", systemImage: "sparkles").frame(maxWidth: .infinity).padding(.vertical, 6)
                    }.plonnGlassButton(prominent: true).accessibilityIdentifier("matchup-one-click-model")
                }
                Picker("Market category",selection:$category) {
                    ForEach(["Overview","Team stats","Player props"],id: \.self) { Text($0) }
                }.pickerStyle(.segmented)
                if !account.hasSignedInAccount {
                    Button("Sign in to view odds") {account.access.present(feature:.odds)}
                        .accessibilityIdentifier("matchup-sign-in-odds")
                } else if let issue=markets.oddsIssue {OddsIssueNotice(message:issue)}
                if category != "Player props" { MatchupOverview(game:game,sport:sport,markets:matchupMarkets,research:research,statsOnly:category == "Team stats",gameUpdates:gameUpdates) }
                if category == "Player props" {
                    NavigationLink("Player stats & trends") {ExploreView(initialSport:sport,live:true)}
                    Text("Load a player's stats in Analytics before matching a selected line. No player odds are fetched automatically.").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
                    ScrollView(.horizontal, showsIndicators:false) {
                        HStack { ForEach(["All props"] + propCategories(sport).map { $0.1 },id: \.self) { title in
                            Button(title) { propType = title }.font(.system(.caption, design: .monospaced).weight(.semibold))
                                .padding(.horizontal,12).padding(.vertical,9)
                                .background(propType == title ? matchupColor(game.homeColor).opacity(0.15) : Color.black.opacity(0.06),in:Capsule())
                                .foregroundStyle(propType == title ? matchupColor(game.homeColor) : .primary)
                        } }
                    }
                }
                if category == "Player props" {
                    if propType != "All props" { StatLabel(title:propType).font(.system(.caption, design: .monospaced)) }
                    TextField("Find a player",text:$search).textFieldStyle(.roundedBorder)
                }
                if markets.loading { PLONNLoader("Loading game markets…") }
                if category == "Player props" {
                ForEach(matching) { market in
                    NavigationLink { DetailView(m:market).environment(\.matchupPalette,[matchupColor(game.awayColor),matchupColor(game.homeColor)]) } label: {
                        HStack(spacing:12) {
                            if market.kind == "prop" { PlayerPortrait(market:market,size:40) }
                            VStack(alignment:.leading,spacing:4) {
                                Text(market.kind == "prop" || market.market == "h2h" || market.market == "spreads" ? market.name : market.label).font(.system(.subheadline, design: .monospaced).weight(.semibold)).foregroundStyle(.primary)
                                if market.kind == "prop" || market.market == "spreads" { Text(compactPropLabel(market)).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary); StatLabel(title:propTitle(market.market)).font(.system(.caption2, design: .monospaced)) }
                                Text(market.book).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                            }
                            Spacer(minLength:8)
                            VStack(alignment:.trailing,spacing:4) {
                                Text(odds(market.price)).font(.system(size:18,weight:.bold,design:.monospaced)).monospacedDigit().foregroundStyle(charcoalAccent)
                                if market.market == "h2h" { Text(probability(implied(market.price))+" implied").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary) }
                            }
                            Image(systemName:"chevron.right").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                        }.padding(14).background(PlayerTeamBackground(market:market))
                    }.buttonStyle(.plain)
                }
                if account.hasSignedInAccount && !markets.loading && matching.isEmpty {
                    ContentUnavailableView(markets.oddsIssue == nil ? "No available markets":"Prices could not refresh",systemImage:"chart.bar.xaxis",description:Text(markets.oddsIssue ?? "No prices available for this market right now."))
                }
                }
                if account.hasSignedInAccount {
                    Text("Sportsbook snapshots · Prices may change").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                }
            }.padding(16)
        }.onAppear { scroll.scrollTo("matchup-top",anchor:.top) }
        .background(todayCanvas).navigationTitle("Matchup").navigationBarTitleDisplayMode(.inline)
        .environment(\.matchupPalette,[matchupColor(game.awayColor),matchupColor(game.homeColor)])
        .task(id: event.sport + event.eventID + event.homeTeam + event.awayTeam + event.startISO + String(describing:scenePhase) + String(gameUpdates.refreshRevision)) {
            guard scenePhase == .active else {return}
            await gameUpdates.watch(event:event)
        }
        .task(id: sport + game.id + String(describing:scenePhase) + (account.user?.id.uuidString ?? "guest")) {
            guard scenePhase == .active else{return}
            await research.load(game:game,sport:sport)
            await markets.load(sport,account:account,home:game.home,away:game.away,start:game.start,forceRefresh:false,cacheOnly:true,nbaSeasonTypes:[game.seasonType])
        }
        .plonnRefreshable {
            gameUpdates.refreshRevision += 1
            await research.load(game:game,sport:sport)
            await markets.load(sport,account:account,home:game.home,away:game.away,start:game.start,cacheOnly:true,nbaSeasonTypes:[game.seasonType])
        }
        }
    }
}


// Historical team context precedes any optional, scoped price matching.
struct HotLiveOddsCarousel:View {
 let sport:String
 let isToday:Bool
 let games:[SportsItem]
 @Environment(\.scenePhase) private var scenePhase
 @State private var rows:[LiveTeamStats]=[]
 @State private var loading=false
 @State private var generation=UUID()
 @State private var work:Task<Void,Never>?
 private struct LiveTeamStats:Identifiable {let id:String;let name:String;let history:AnalyticsHistoryResult}
 private var liveGames:[SportsItem] {Array(games.filter(\.live).prefix(2))}
 private var identity:String {sport+liveGames.map(\.id).joined(separator:"|")}
 private func cancel() {generation=UUID();work?.cancel();work=nil;loading=false}
 private func loadStats() {
  guard !loading,isToday,scenePhase == .active else{return}
  let ticket=UUID(),context=identity,selected=liveGames
  generation=ticket;loading=true
  work=Task { @MainActor in
   defer{if generation==ticket {loading=false;work=nil}}
   var found:[LiveTeamStats]=[]
   for game in selected {
    for name in [game.away,game.home] {
     guard !Task.isCancelled,generation==ticket,identity==context,scenePhase == .active else{return}
     let market=Market(id:"stats-"+game.id+name,event:game.id,sport:sport,kind:"team",name:name,opponent:"",label:"Recent team results",aliases:"",p:nil,price:0,market:"h2h",book:"",quotes:[],updated:nil,mode:"analytics",commence:game.start)
     let result=await AnalyticsHistoryService.shared.loadStats(market:market,limit:5)
     guard !Task.isCancelled,generation==ticket,identity==context,scenePhase == .active else{return}
     found.append(LiveTeamStats(id:market.id,name:name,history:result))
    }
   }
   rows=found
  }
 }
 var body:some View {
  VStack(alignment:.leading,spacing:12) {
   Button(loading ? "Loading team stats…":"View recent form",action:loadStats).disabled(loading || liveGames.isEmpty).font(.system(.subheadline,design:.monospaced).weight(.semibold))
   if !rows.isEmpty { Text("Last 5 completed games · Not live predictions").font(.caption).foregroundStyle(.secondary) }
   ForEach(rows) {row in
    Panel {
     Text(row.name).font(.system(.headline,design:.monospaced))
     if let stats=row.history.statistics {
      Text("\(stats.teamWins ?? 0) wins · \(stats.teamLosses ?? 0) losses · \(stats.teamTies ?? 0) ties").font(.system(.subheadline,design:.monospaced))
      Text("\(stats.valid) verified results / \(stats.n) observed games").font(.system(.caption,design:.monospaced))
      Text("Source: "+row.history.source+" · completed games before this matchup").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
     } else {Text(row.history.availability).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)}
    }
   }
   NavigationLink("Open Analytics") {ExploreView(initialSport:sport,live:true)}.font(.system(.subheadline,design:.monospaced))
  }.accessibilityIdentifier("live-team-trends")
   .onDisappear{cancel()}
   .onChange(of:identity){_,_ in cancel();rows=[]}
   .onChange(of:scenePhase){_,phase in if phase != .active {cancel()}}
 }
}

func propCategories(_ sport:String) -> [(String,String)] {
    if sport == "NHL" {return [("player_points","Points"),("player_assists","Assists"),("player_goals","Goals"),("player_shots_on_goal","Shots on goal"),("player_blocked_shots","Blocked shots"),("player_total_saves","Saves"),("player_power_play_points","Power-play points")]}
    if sport == "NFL" { return [("player_pass_yds","Pass Yds"),("player_pass_tds","Pass TDs"),("player_rush_yds","Rush Yds"),("player_reception_yds","Rec Yds"),("player_receptions","Rec"),("player_anytime_td","Anytime TD")] }
    return sport == "MLB" ? [("batter_hits","Hits"),("batter_rbis","RBI"),("batter_stolen_bases","Stolen bases"),("batter_home_runs","Home runs"),("batter_total_bases","Total bases"),("batter_runs_scored","Runs"),("pitcher_strikeouts","Strikeouts")] : [("player_points","Points"),("player_rebounds","Rebounds"),("player_assists","Assists"),("player_threes","Threes"),("player_steals","Steals"),("player_blocks","Blocks"),("player_points_rebounds_assists","PTS + REB + AST"),("player_points_rebounds","PTS + REB"),("player_points_assists","PTS + AST"),("player_rebounds_assists","REB + AST")]
}
func propTitle(_ key:String) -> String { (propCategories("MLB") + propCategories("NBA") + propCategories("NFL") + propCategories("NHL")).first { $0.0 == key }?.1 ?? key }
@MainActor final class PlayerHistory:ObservableObject {
    @Published var games:[PlayerGame] = []
    @Published var opponentGames:[PlayerGame] = []
    @Published var context:OpponentHistoryContext?
    @Published var loading = false
    @Published var message = ""
    private var generation=UUID()
    func load(_ market:Market,includeOpponent:Bool=false) async {
        let request=UUID();generation=request
        loading=true;games=[];opponentGames=[];context=nil;message=""
        let result=await PlayerHistoryService.shared.load(market:market,includeOpponent:includeOpponent)
        guard generation==request else{return}
        loading=false
        guard !Task.isCancelled else{return}
        games=result.games;opponentGames=result.opponentGames;context=result.context
        message=([result.source+" · "+result.availability]+result.warnings).joined(separator:"\n")
    }
}
private func playerHistorySignature(_ market:Market)->String {
    [market.id,market.sport,market.name,market.market,market.event,market.eventLabel ?? "",market.commence ?? ""].joined(separator:"|")
}
private func playerHistoryDate(_ date:Date,compact:Bool=false)->String {
    let formatter=DateFormatter();formatter.locale=Locale(identifier:"en_US_POSIX")
    formatter.timeZone=TimeZone(identifier:"America/New_York")
    formatter.dateFormat=compact ? "MM/dd":"MMM d, yyyy"
    return formatter.string(from:date)
}
struct PlayerHistoryView:View {
    @State private var teamPrimary:Color = .gray
    private var palette:[Color] {[teamPrimary,teamPrimary]}
    let m:Market
    @StateObject private var history=PlayerHistory()
    @State private var count=5
    @State private var againstOpponent=false
    private var summary:OpponentHistorySummary? {
        history.context.map {OpponentPropHistoryEngine.summarize(games:history.opponentGames,context:$0,line:m.line,displayCount:count)}
    }
    private var displayed:[PlayerGame] {againstOpponent ? (summary?.games ?? []):Array(history.games.prefix(count))}
    private var direction:String {(m.side ?? m.label.components(separatedBy:" ").first ?? "").lowercased()}
    var body:some View {
        Panel {VStack(alignment:.leading,spacing:16) {
            HStack {Text("Player history").font(.system(size:18,weight:.semibold,design:.monospaced));Spacer();StatLabel(title:propTitle(m.market)).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)}
            Picker("History scope",selection:$againstOpponent) {Text("Recent").tag(false);Text("Against opponent").tag(true)}.pickerStyle(.segmented).accessibilityIdentifier("player-history-scope")
            Picker("Games to show",selection:$count) {Text("Last 5").tag(5);Text("Last 10").tag(10)}.pickerStyle(.segmented)
            if history.loading {PLONNLoader("Loading player stats…")}
            else if againstOpponent {
                if let summary {
                    Text(summary.text).font(.system(.subheadline,design:.monospaced))
                    Text(summary.evidence).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
                } else {Text("Verified opponent history is unavailable for this player and prop.").font(.system(.subheadline,design:.monospaced)).foregroundStyle(.secondary)}
            }
            if !displayed.isEmpty {
                HStack {
                    Metric(title:"Average",value:String(format:"%.1f",displayed.map(\.value).reduce(0,+)/Double(displayed.count)))
                    if let line=m.line,line.isFinite,["over","under"].contains(direction) {
                        let aboveOrBelow=displayed.filter {direction == "over" ? $0.value > line:$0.value < line}.count
                        let equal=displayed.filter {$0.value == line}.count
                        Metric(title:direction == "over" ? "Above current line":"Below current line",value:"\(aboveOrBelow)/\(displayed.count)")
                        if equal > 0 {Metric(title:"Equal to line",value:String(equal))}
                    }
                }
                Chart {
                    ForEach(Array(displayed.reversed())) {game in
                        BarMark(x:.value("Game",playerHistoryDate(game.date,compact:true)+" · "+String(game.id.suffix(3))),y:.value("Stat",game.value)).foregroundStyle(LinearGradient(colors:palette,startPoint:.bottomLeading,endPoint:.topTrailing)).cornerRadius(4)
                    }
                    if let line=m.line,line.isFinite {RuleMark(y:.value("Current line",line)).foregroundStyle(Color.primary.opacity(0.55)).lineStyle(StrokeStyle(lineWidth:1,dash:[4])).annotation(position:.top,alignment:.trailing) {Text("Current line \(line.formatted())").font(.system(.caption2,design:.monospaced))}}
                }.frame(height:170).chartXAxis(.hidden)
                Text("Oldest → newest · \(displayed.count) verified games · dates in Eastern time").font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)
                DisclosureGroup("Game-by-game results") {
                    ForEach(displayed) {game in
                        HStack {VStack(alignment:.leading,spacing:4) {Text(game.opponent).font(.system(.subheadline,design:.monospaced));Text(playerHistoryDate(game.date)).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)};Spacer();Text(game.workload).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary);Text(game.value.formatted()).bold().monospacedDigit()}
                        Divider()
                    }
                }
                if !againstOpponent,let line=m.line,line.isFinite {DisclosureGroup("Sample statistics") {PropStatisticsView(values:displayed.map(\.value),line:line).environment(\.matchupPalette,palette)}}
                Text("Actual results compared with today’s line. Historical betting lines are unavailable; this is not a forecast.").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)
            }
            if !history.loading {
                if displayed.isEmpty {Text(history.message).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)}
                else {
                    Text(m.sport == "MLB" ? "MLB Stats API":"ESPN").font(.caption).foregroundStyle(.secondary)
                    if history.message.split(separator:"\n").count>1 {
                        Text("Some earlier games or opponent details could not be verified. The available sample may be incomplete.").font(.caption).foregroundStyle(.secondary)
                    }
                }
                if displayed.isEmpty {Button("Retry") {Task {await history.load(m,includeOpponent:true)}}}
            }
        }}.task(id:playerHistorySignature(m)) {await history.load(m,includeOpponent:true)}
        .task(id:m.sport+m.name+m.aliases) {teamPrimary = .gray;let colors=await PlayerTeamColors.shared.colors(m);if let primary=colors.first,!primary.isEmpty {teamPrimary=matchupColor(primary)}}
    }
}

private struct OpponentPropTrendView:View {
    let market:Market
    var showSourceDetails = true
    @StateObject private var history=PlayerHistory()
    private var summary:OpponentHistorySummary? {
        history.context.map {OpponentPropHistoryEngine.summarize(games:history.opponentGames,context:$0,line:market.line)}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:6) {
            Text(market.name+" · "+propTitle(market.market)).font(.system(.subheadline,design:.monospaced).bold())
            if history.loading {Text("Loading games against this opponent…").font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)}
            else if let summary {
                Text(summary.text).font(.system(.subheadline,design:.monospaced))
                if showSourceDetails {Text(summary.evidence).font(.system(.caption2,design:.monospaced)).foregroundStyle(.secondary)}
                else if summary.n > 0 {
                    let incomplete=history.context.map {!Set($0.requestedSeasons).isSubset(of:Set($0.loadedSeasons))} ?? false
                    Text(MatchupOutlookPresentation.opponentCaution(count:summary.n,incomplete:incomplete)).font(.caption).foregroundStyle(.secondary)
                }
                if summary.n > 0 {
                    NavigationLink {DetailView(m:market)} label: {HStack(spacing:6) {Text("View "+compactPropLabel(market));Image(systemName:"chevron.right")}.font(.system(.caption,design:.monospaced).bold())}
                        .buttonStyle(.plain).modifier(PlayerLinkAccent(market:market))
                        .accessibilityLabel("View "+market.name+" opponent history at "+market.book)
                }
            } else {Text(history.message).font(.system(.caption,design:.monospaced)).foregroundStyle(.secondary)}
        }.accessibilityIdentifier("opponent-prop-trend")
        .task(id:playerHistorySignature(market)) {await history.load(market,includeOpponent:true)}
    }
}

actor PlayerPortraits {
    static let shared=PlayerPortraits()
    private var requests:[String:Task<[String:[String]],Never>] = [:]
    private static func key(_ name:String)->String { name.folding(options:[.diacriticInsensitive,.caseInsensitive],locale:Locale(identifier:"en_US_POSIX")).filter { $0.isLetter || $0.isNumber } }
    private static func json(_ url:String) async throws -> [String:Any] {
        var request=URLRequest(url:URL(string:url)!);request.timeoutInterval=20
        let (data,response)=try await URLSession.shared.data(for:request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONSerialization.jsonObject(with:data) as? [String:Any] ?? [:]
    }
    func url(for market:Market) async -> URL? {
        let season=Calendar.current.component(.year,from:Date())
        let cacheKey=market.sport+String(season)+(market.sport != "MLB" ? market.aliases:"")
        if requests[cacheKey] == nil {
            requests[cacheKey]=Task {
                var result:[String:[String]] = [:]
                do {
                    if market.sport == "MLB" {
                        let data=try await Self.json("https://statsapi.mlb.com/api/v1/sports/1/players?season=\(season)")
                        for player in data["people"] as? [[String:Any]] ?? [] {
                            guard let name=player["fullName"] as? String,let id=player["id"] as? Int else { continue }
                            result[Self.key(name),default:[]].append("https://img.mlbstatic.com/mlb-photos/image/upload/w_200,q_auto:best/v1/people/\(id)/headshot/67/current")
                        }
                    } else if market.sport == "NBA" || market.sport == "NFL" || market.sport == "NHL" {
                        let league=sportsFeedPath(market.sport)
                        let data=try await Self.json("https://site.api.espn.com/apis/site/v2/sports/\(league)/teams?limit=1000")
                        let sports=data["sports"] as? [[String:Any]] ?? []
                        let leagues=sports.first?["leagues"] as? [[String:Any]] ?? []
                        let teams=leagues.first?["teams"] as? [[String:Any]] ?? []
                        for entry in teams {
                            guard let team=entry["team"] as? [String:Any],let name=team["displayName"] as? String,let id=team["id"] as? String,market.aliases.localizedCaseInsensitiveContains(name) else { continue }
                            let roster=try await Self.json("https://site.api.espn.com/apis/site/v2/sports/\(league)/teams/\(id)/roster")
                            let groups=roster["athletes"] as? [[String:Any]] ?? []
                            let athletes=groups.flatMap { $0["items"] as? [[String:Any]] ?? [$0] }
                            for player in athletes {
                                guard let name=player["displayName"] as? String,let photo=player["headshot"] as? [String:Any],let href=photo["href"] as? String,href.hasPrefix("https://") else { continue }
                                result[Self.key(name),default:[]].append(href)
                            }
                        }
                    }
                } catch { return [:] }
                return result
            }
        }
        let result=await requests[cacheKey]!.value
        if result.isEmpty { requests[cacheKey]=nil }
        let urls=Set(result[Self.key(market.name)] ?? [])
        guard urls.count == 1,let url=urls.first else { return nil }
        return URL(string:url)
    }
}
struct PlayerPortrait:View {
    let market:Market
    var size:CGFloat = 52
    @State private var url:URL?
    var body:some View {
        AsyncImage(url:url) { phase in
            if let image=phase.image { image.resizable().scaledToFill() }
            else { Text(market.name.split(separator:" ").prefix(2).compactMap(\.first).map(String.init).joined()).font(.system(size:size*0.28,weight:.semibold,design:.monospaced)).foregroundStyle(charcoalAccent).frame(maxWidth:.infinity,maxHeight:.infinity) }
        }.frame(width:size,height:size).background(charcoalAccent.opacity(0.1)).clipShape(Circle())
            .overlay(Circle().strokeBorder(charcoalAccent.opacity(0.2),lineWidth:1))
            .accessibilityLabel("\(market.name) player portrait")
            .task(id:market.sport+market.name+market.aliases) { url=nil;url=await PlayerPortraits.shared.url(for:market) }
    }
}

struct PropStatisticsView:View {
 @Environment(\.matchupPalette) private var palette
 let values:[Double]
 let line:Double
 @State private var method="Historical"
 var body:some View {
  if let stats=PropDistribution(values:values,line:line) {
   VStack(alignment:.leading,spacing:16) {
    Text("Distribution").font(.system(.headline, design: .monospaced))
    HStack { Metric(title:"Median",value:stats.median.formatted(.number.precision(.fractionLength(1))));Metric(title:"Std. deviation",value:stats.deviation.formatted(.number.precision(.fractionLength(1))));Metric(title:"Sample",value:"\(values.count) games") }
    let buckets=Dictionary(grouping:values,by:{$0}).map{(value:$0.key,count:$0.value.count)}.sorted{$0.value<$1.value}
    Chart {
     ForEach(buckets,id:\.value){bucket in BarMark(x:.value("Result",String(format:"%g",bucket.value)),y:.value("Games",bucket.count)).foregroundStyle(LinearGradient(colors:palette,startPoint:.bottomLeading,endPoint:.topTrailing)).cornerRadius(3)}
    }.frame(height:140).chartYAxisLabel("Games")
    Text("Observed results • Gaps between categories are not to scale.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
    Picker("Statistical method",selection:$method){Text("Historical").tag("Historical");Text("Poisson").tag("Poisson");Text("Normal").tag("Normal")}.pickerStyle(.segmented)
    let result=method == "Historical" ? stats.empirical : (method == "Poisson" ? stats.poisson:stats.normal)
    if let result {
     HStack { Metric(title:"Over \(line.formatted())",value:probability(result.over));Metric(title:"Under",value:probability(result.under));Metric(title:"Push",value:probability(result.push)) }
    } else { Text("This method cannot be fitted to these observations.").font(.system(.caption, design: .monospaced)) }
    if method == "Historical" {
     let interval=stats.interval(under:false)
     Text("Over frequency: 95% Wilson interval \(probability(interval.0))–\(probability(interval.1)). Assumes independent observations; this is not a forecast interval.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
    } else if method == "Poisson" {
     Text("Count baseline fitted to the average. Assumes independent events and equal mean and variance; playing time and opponent effects are not included.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
     if stats.mean > 0 && stats.variance/stats.mean > 1.5 { Text("Results vary more than the Poisson assumption allows. Treat this fit cautiously.").font(.system(.caption, design: .monospaced)).foregroundStyle(.orange) }
    } else {
     Text("Normal baseline fitted to the mean and sample deviation, with continuity correction for count stats. Symmetry and unbounded tails can fit low-count props poorly.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
    }
    Text("Exploratory estimates from only \(values.count) games. These methods are not calibrated betting predictions and do not account for injuries, matchups or role changes.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   }.padding(.top,12)
  }
 }
}


private struct TeamCardBackground:View {
    let game:SportsItem
    private func color(_ hex:String)->Color {
        let text=hex.trimmingCharacters(in:CharacterSet(charactersIn:"#"))
        guard text.count==6,let rgb=UInt32(text,radix:16) else{return Color.gray}
        return Color(red:Double((rgb >> 16)&255)/255,green:Double((rgb >> 8)&255)/255,blue:Double(rgb&255)/255)
    }
    var body:some View {
        RoundedRectangle(cornerRadius:14).fill(navySurface)
            .overlay { LinearGradient(stops:[.init(color:color(game.awayColor).opacity(0.28),location:0),.init(color:color(game.awayColor).opacity(0.10),location:0.4),.init(color:color(game.homeColor).opacity(0.10),location:0.6),.init(color:color(game.homeColor).opacity(0.28),location:1)],startPoint:.topLeading,endPoint:.bottomTrailing) }
            .clipShape(RoundedRectangle(cornerRadius:14))
            .overlay { RoundedRectangle(cornerRadius:14).strokeBorder(Color.black.opacity(0.07),lineWidth:1) }
    }
}


struct TeamRecentGame:Identifiable {
 let id:String;let date:Date;let opponent:String;let scored:Double;let allowed:Double;let home:Bool
 var won:Bool { scored > allowed }
}
@MainActor final class MatchupResearch:ObservableObject {
 @Published var logs:[String:[TeamRecentGame]] = [:]
 @Published var notes:[String] = []
 @Published var loading=false
 @Published var message=""
 private func rows(_ value:Any?)->[[String:Any]] { value as? [[String:Any]] ?? [] }
 private func json(_ url:String) async throws->[String:Any] {
  var request=URLRequest(url:URL(string:url)!);request.timeoutInterval=20
  let(data,response)=try await URLSession.shared.data(for:request)
  guard (response as? HTTPURLResponse)?.statusCode==200 else { throw URLError(.badServerResponse) }
  return try JSONSerialization.jsonObject(with:data) as? [String:Any] ?? [:]
 }
 func load(game:SportsItem,sport:String) async {
  loading=true;defer{loading=false};logs=[:];notes=[];message=""
  let path=sportsFeedPath(sport)
  do {
   let summary=try await json("https://site.api.espn.com/apis/site/v2/sports/\(path)/summary?event=\(game.id)")
   let header=summary["header"] as? [String:Any] ?? [:]
   let competition=rows(header["competitions"]).first ?? [:]
   if let venue=(summary["gameInfo"] as? [String:Any])?["venue"] as? [String:Any],let name=venue["fullName"] as? String { notes.append(name) }
   let cutoff=min(sportsDate(game.start) ?? Date(),Date())
   let season=(header["season"] as? [String:Any])?["year"] as? Int ?? Calendar.current.component(.year,from:cutoff)
   for competitor in rows(competition["competitors"]) {
    guard let team=competitor["team"] as? [String:Any],let id=team["id"] as? String,let name=team["displayName"] as? String else{continue}
    for probable in rows(competitor["probables"]) {
     if let athlete=probable["athlete"] as? [String:Any],let player=athlete["displayName"] as? String { notes.append("\(name): probable starter \(player)") }
    }
    do {
     // Include regular season history when ESPN defaults to the postseason.
     var events:[[String:Any]]=[]
     for seasonType in [2,3] {
      let schedule=try await json("https://site.api.espn.com/apis/site/v2/sports/\(path)/teams/\(id)/schedule?season=\(season)&seasontype=\(seasonType)")
      events += rows(schedule["events"])
     }
     var seen=Set<String>()
     var results:[TeamRecentGame]=[]
     for event in events {
      guard let eid=event["id"] as? String,eid != game.id,seen.insert(eid).inserted,let raw=event["date"] as? String,let date=sportsDate(raw),date < cutoff,
       let match=rows(event["competitions"]).first,let status=match["status"] as? [String:Any],let type=status["type"] as? [String:Any],type["completed"] as? Bool == true else{continue}
      let teams=rows(match["competitors"])
      guard let own=teams.first(where:{$0["id"] as? String == id}),let other=teams.first(where:{$0["id"] as? String != id}),
       let ownScore=own["score"] as? [String:Any],let otherScore=other["score"] as? [String:Any],let scored=ownScore["value"] as? Double,let allowed=otherScore["value"] as? Double else{continue}
      let opponent=(other["team"] as? [String:Any])?["abbreviation"] as? String ?? "Opponent"
      results.append(TeamRecentGame(id:eid,date:date,opponent:opponent,scored:scored,allowed:allowed,home:own["homeAway"] as? String == "home"))
     }
     logs[name]=Array(results.sorted{$0.date>$1.date}.prefix(10))
    } catch { message="Some team history could not be loaded. Pull to refresh." }
   }
   if logs.values.allSatisfy({$0.isEmpty}) { message="Recent completed games are unavailable for this matchup." }
  } catch { message="Matchup research is unavailable. Pull to refresh." }
 }
}

private struct MatchupOverview:View {
 let game:SportsItem;let sport:String;let markets:[Market]
 @ObservedObject var research:MatchupResearch
 @State private var count=5
 var statsOnly=false
 @ObservedObject var gameUpdates:LiveMatchupStore
 private var gameContext:Bool { LiveMatchupPolicy.showsGameContext(event:game.matchupEvent(sport:sport),state:game.state,snapshot:gameUpdates.snapshot,now:Date()) }
 private func recent(_ name:String)->[TeamRecentGame] { Array((research.logs[name] ?? []).prefix(count)) }
 private func average(_ games:[TeamRecentGame],scored:Bool)->String { guard !games.isEmpty else{return "—"};return String(format:"%.1f",games.map{scored ? $0.scored:$0.allowed}.reduce(0,+)/Double(games.count)) }
 var body:some View {
  VStack(alignment:.leading,spacing:18) {
   TimelineView(.periodic(from:.now,by:15)) { timeline in
   if !statsOnly && !LiveMatchupPolicy.showsGameContext(event:game.matchupEvent(sport:sport),state:game.state,snapshot:gameUpdates.snapshot,now:timeline.date) {
    Text("Odds").font(.system(.headline, design: .monospaced))
    VStack(spacing:12) {
     ForEach([game.away,game.home],id:\.self) { team in
      let ml=sportsDate(game.start).flatMap { Store.recordedMoneyline(team,markets:markets,sport:sport,home:game.home,away:game.away,start:$0) }
      HStack { Text(team).font(.system(.subheadline, design: .monospaced));Spacer();if let ml { VStack(alignment:.trailing,spacing:3){Text(odds(ml.price)).bold().foregroundStyle(charcoalAccent);Text(probability(implied(ml.price))+" implied · "+ml.book).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)} } else { Text("No price").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) } }
     }
     Divider()
     HStack(alignment:.top) {
      marketSummary("Spread",key:"spreads")
      Spacer()
      marketSummary("Total",key:"totals")
     }
    }.padding(16).background(navySurface,in:RoundedRectangle(cornerRadius:14))
    Text("Sportsbook snapshots · Implied probability includes margin").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
   }
   }
   if statsOnly {
    TimelineView(.periodic(from:.now,by:15)) { timeline in
     if LiveMatchupPolicy.showsGameContext(event:game.matchupEvent(sport:sport),state:game.state,snapshot:gameUpdates.snapshot,now:timeline.date) {
      LiveMatchupCard(event:game.matchupEvent(sport:sport),snapshot:gameUpdates.snapshot,loading:gameUpdates.loading,failed:gameUpdates.failed,now:timeline.date)
       .padding(16).background(navySurface,in:RoundedRectangle(cornerRadius:18))
     }
    }
   }
   MatchupAdvantagesView(game:game,sport:sport,logs:research.logs,markets:markets,showInsights:!statsOnly,researchLoading:research.loading,gameUpdates:gameUpdates)
   if sport == "MLB" && !statsOnly && !gameContext { StartingPitchersView(game:game,markets:markets) }
   HStack { Text(statsOnly ? "Team comparison":"Recent form").font(.system(.headline, design: .monospaced));Spacer();Picker("Games",selection:$count){Text("Last 5").tag(5);Text("Last 10").tag(10)}.pickerStyle(.segmented).frame(width:155) }
   if research.loading { PLONNLoader("Loading matchup…") }
   ForEach([game.away,game.home],id:\.self) { name in
    let games=recent(name)
    VStack(alignment:.leading,spacing:10) {
     HStack { Text(name).font(.system(.subheadline, design: .monospaced).bold());Spacer();Text(games.isEmpty ? "Unavailable":"\(games.filter(\.won).count)–\(games.filter{!$0.won}.count)").foregroundStyle(charcoalAccent).bold() }
     HStack(spacing:6) { ForEach(Array(games.reversed())) { match in Text(match.won ? "W":"L").font(.system(.caption2, design: .monospaced).bold()).frame(maxWidth:.infinity).padding(.vertical,7).foregroundStyle(Color.white).background((match.won ? Color(red:0.05,green:0.43,blue:0.23):Color(red:0.72,green:0.14,blue:0.19)).opacity(0.8),in:RoundedRectangle(cornerRadius:5)) } }
     Text("\(games.count) completed games · Oldest → newest").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     if statsOnly {
      HStack { Metric(title:sport == "MLB" ? "Runs / game":"Points / game",value:average(games,scored:true));Metric(title:"Allowed / game",value:average(games,scored:false)) }
      ForEach(games) { match in HStack { Text(match.date.formatted(.dateTime.month().day()));Text((match.home ? "vs ":"at ")+match.opponent);Spacer();Text("\(Int(match.scored))–\(Int(match.allowed))") }.font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
     }
    }.padding(16).background(TeamIdentityBackground(primary:name == game.home ? game.homeColor:game.awayColor,secondary:name == game.home ? game.homeAlternateColor:game.awayAlternateColor))
   }
   if !research.message.isEmpty { Text(research.message).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
   Text("Recent form describes past games and does not predict this matchup.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
  }
 }
 @ViewBuilder private func marketSummary(_ title:String,key:String)->some View {
  VStack(alignment:.leading,spacing:4) {
   StatLabel(title:title).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   if let row=markets.first(where:{$0.market==key}) { Text(row.label).font(.system(.caption, design: .monospaced).bold());Text(odds(row.price)+" · "+row.book).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary) } else { Text("Unavailable").font(.system(.caption, design: .monospaced)) }
  }
 }
}

private struct PitcherLine:Identifiable {
 let id:String;let date:String;let innings:String;let strikeouts:String;let runs:String;let walks:String
}
private struct StarterProfile:Identifiable {
 let id:Int;let name:String;let team:String;let hand:String;let era:String;let whip:String;let k9:String;let innings:String;let starts:[PitcherLine]
}
@MainActor private final class StartingPitchers:ObservableObject {
 @Published var players:[StarterProfile]=[]
 @Published var league:[String:Double]=[:]
 @Published var loading=false
 @Published var message=""
 private func json(_ path:String) async throws->[String:Any] {
  var request=URLRequest(url:URL(string:"https://statsapi.mlb.com/api/v1/"+path)!);request.timeoutInterval=20
  let(data,response)=try await URLSession.shared.data(for:request)
  guard (response as? HTTPURLResponse)?.statusCode==200 else{throw URLError(.badServerResponse)}
  return try JSONSerialization.jsonObject(with:data) as? [String:Any] ?? [:]
 }
 func load(_ game:SportsItem) async {
  loading=true;players=[];league=[:];message="";defer{loading=false}
  let start=sportsDate(game.start) ?? Date()
  let format=DateFormatter();format.dateFormat="yyyy-MM-dd";format.timeZone=TimeZone(identifier:"America/New_York");let day=format.string(from:start)
  do {
   let schedule=try await json("schedule?sportId=1&date=\(day)&hydrate=probablePitcher")
   let games=(schedule["dates"] as? [[String:Any]] ?? []).flatMap{$0["games"] as? [[String:Any]] ?? []}
   let selected=games.filter { row in
    guard let teams=row["teams"] as? [String:[String:Any]],let home=teams["home"]?["team"] as? [String:Any],let away=teams["away"]?["team"] as? [String:Any],let raw=row["gameDate"] as? String,let date=sportsDate(raw) else{return false}
    return home["name"] as? String == game.home && away["name"] as? String == game.away && abs(date.timeIntervalSince(start))<1800
   }
   guard selected.count==1,let teams=selected[0]["teams"] as? [String:[String:Any]] else{message="Starting pitchers have not been matched to this game.";return}
   let season=String(day.prefix(4))
   if let leagueData=try? await json("teams/stats?stats=season&group=pitching&sportIds=1&season=\(season)") {
    let groups=leagueData["stats"] as? [[String:Any]] ?? []
    let rows=groups.flatMap{$0["splits"] as? [[String:Any]] ?? []}
    let totals=rows.compactMap{$0["stat"] as? [String:Any]}
    func sum(_ key:String)->Double { totals.reduce(0){$0+($1[key] as? Double ?? 0)} }
    let outs=sum("outs")
    if totals.count==30 && outs>0 && totals.allSatisfy({$0["outs"] != nil && $0["earnedRuns"] != nil && $0["hits"] != nil && $0["baseOnBalls"] != nil && $0["strikeOuts"] != nil}) {
     league=["ERA":27*sum("earnedRuns")/outs,"WHIP":3*(sum("hits")+sum("baseOnBalls"))/outs,"K / 9":27*sum("strikeOuts")/outs]
    }
   }

   for side in ["away","home"] {
    let team=side == "away" ? game.away:game.home
    guard let pitcher=teams[side]?["probablePitcher"] as? [String:Any],let id=pitcher["id"] as? Int,let name=pitcher["fullName"] as? String else{message += "\(team): starter not announced. ";continue}
    do {
     let person=try await json("people/\(id)")
     let info=(person["people"] as? [[String:Any]])?.first ?? [:]
     let hand=(info["pitchHand"] as? [String:Any])?["description"] as? String ?? "Hand unavailable"
     let seasonData=try await json("people/\(id)/stats?stats=season&group=pitching&season=\(season)")
     let statGroups=seasonData["stats"] as? [[String:Any]] ?? []
     let splits=statGroups.flatMap{$0["splits"] as? [[String:Any]] ?? []}
     let stats=(splits.first?["stat"] as? [String:Any]) ?? [:]
     func value(_ key:String)->String { stats[key].map{String(describing:$0)} ?? "—" }
     let logs=try await json("people/\(id)/stats?stats=gameLog&group=pitching&season=\(season)")
     var starts:[PitcherLine]=[]
     for group in logs["stats"] as? [[String:Any]] ?? [] { for row in group["splits"] as? [[String:Any]] ?? [] {
      guard let date=row["date"] as? String,date<day,let stat=row["stat"] as? [String:Any],(stat["gamesStarted"] as? Int ?? 0)>0,let event=row["game"] as? [String:Any],let gid=event["gamePk"] as? Int else{continue}
      func text(_ key:String)->String { stat[key].map{String(describing:$0)} ?? "—" }
      starts.append(PitcherLine(id:String(gid),date:date,innings:text("inningsPitched"),strikeouts:text("strikeOuts"),runs:text("earnedRuns"),walks:text("baseOnBalls")))
     } }
     players.append(StarterProfile(id:id,name:name,team:team,hand:hand,era:value("era"),whip:value("whip"),k9:value("strikeoutsPer9Inn"),innings:value("inningsPitched"),starts:Array(starts.sorted{$0.date>$1.date}.prefix(5))))
    } catch {message += "Stats unavailable for \(name). "}
   }
  } catch {message="Pitcher data unavailable. Pull to refresh."}
 }
}
private struct StartingPitchersView:View {
 let game:SportsItem
 let markets:[Market]
 @StateObject private var data=StartingPitchers()
 private func pitcherMetric(_ title:String,value:String,lowerBetter:Bool)->some View {
  let baseline=data.league[title]
  let number=Double(value)
  let rounded=baseline.map{($0*100).rounded()/100}
  let comparable=number != nil && rounded != nil
  let equal=comparable && number! == rounded!
  let better=comparable && (lowerBetter ? number! < rounded! : number! > rounded!)
  let color:Color = !comparable || equal ? .secondary : better ? performanceGreen:Color(red:0.72,green:0.14,blue:0.19)
  return VStack(alignment:.leading,spacing:7) {
   StatLabel(title:title).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   Text(value).font(.system(.headline, design: .monospaced)).foregroundStyle(color)
  }.frame(maxWidth:.infinity,alignment:.leading)
 }
 var body:some View {
  VStack(alignment:.leading,spacing:12) {
   HStack { Text("Starting pitchers").font(.system(.headline, design: .monospaced));Spacer();Text("PROBABLE").font(.system(.caption2, design: .monospaced).bold()).foregroundStyle(.secondary) }
   if data.loading { PLONNLoader("Loading pitchers…") }
   ForEach(data.players) { player in
    VStack(alignment:.leading,spacing:14) {
     HStack(spacing:12) {
      AsyncImage(url:URL(string:"https://img.mlbstatic.com/mlb-photos/image/upload/w_200,q_auto:best/v1/people/\(player.id)/headshot/67/current")) { image in image.resizable().scaledToFill() } placeholder: { Image(systemName:"person.fill").foregroundStyle(.secondary) }.frame(width:52,height:52).background(charcoalAccent.opacity(0.1)).clipShape(Circle())
      VStack(alignment:.leading,spacing:4) { Text(player.name).font(.system(.headline, design: .monospaced));Text(player.team).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary);Text("Throws \(player.hand)").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary) }
     }
     HStack(alignment:.top) { pitcherMetric("ERA",value:player.era,lowerBetter:true);pitcherMetric("WHIP",value:player.whip,lowerBetter:true);pitcherMetric("K / 9",value:player.k9,lowerBetter:false);Metric(title:"IP",value:player.innings) }
     Text("Current season · MLB Stats").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     Text("Green: better than MLB · Red: worse · Gray: equal or unavailable. League includes starters and relievers; IP is workload.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
     DisclosureGroup("Last 5 starts") {
      VStack(spacing:10) {
       HStack { Text("Date");Spacer();StatLabel(title:"IP / K / ER / BB") }.font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
       ForEach(player.starts) { line in HStack { Text(line.date);Spacer();Text("\(line.innings) / \(line.strikeouts) / \(line.runs) / \(line.walks)") }.font(.system(.caption, design: .monospaced)).monospacedDigit() }
       if player.starts.isEmpty { Text("No earlier starts available this season.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
      }.padding(.top,10)
     }
     let props=markets.filter{$0.market=="pitcher_strikeouts" && $0.name.folding(options:[.diacriticInsensitive,.caseInsensitive],locale:.current)==player.name.folding(options:[.diacriticInsensitive,.caseInsensitive],locale:.current)}
     ForEach(props.prefix(2)) { prop in NavigationLink { DetailView(m:prop) } label: { HStack { Text(prop.label).font(.system(.caption, design: .monospaced));Spacer();Text(odds(prop.price)).bold() }.foregroundStyle(charcoalAccent) } }
    }.padding(16).background(navySurface,in:RoundedRectangle(cornerRadius:14))
   }
   if !data.message.isEmpty { Text(data.message).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
   Button("Refresh pitchers") { Task { await data.load(game) } }.font(.system(.caption, design: .monospaced)).disabled(data.loading)
   Text("Probable starters can change. Refresh to check updates. IP uses baseball innings notation.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
  }.task(id:game.id) { await data.load(game) }
 }
}

func sportsFeedPath(_ sport:String)->String {
 let additional=["WNBA":"basketball/wnba","NCAAF":"football/college-football","NCAAM":"basketball/mens-college-basketball","NCAAW":"basketball/womens-college-basketball","Tennis":"tennis/atp","Golf":"golf/pga","UFC/MMA":"mma/ufc","F1":"racing/f1","NASCAR":"racing/nascar-premier"]
 if let path=additional[sport] {return path}
 if sport == "Soccer" {return "soccer/uefa.nations"}
 if sport.hasPrefix("Tennis:") {return sport == "Tennis:WTA" ? "tennis/wta":"tennis/atp"}
 if sport.hasPrefix("Soccer:") {return "soccer/"+String(sport.dropFirst(7))}; return sport == "NFL" ? "football/nfl" : sport == "NBA" ? "basketball/nba" : sport == "NHL" ? "hockey/nhl" : "baseball/mlb" }

private struct PropHitRateBoard:View {
 let markets:[Market]
 @State private var histories:[String:[PlayerGame]]=[:]
 @State private var loading=false
 @State private var window=5
 @State private var requested=false
 private func key(_ m:Market)->String { m.sport+"|"+m.name+"|"+m.market }
 private var ranked:[(market:Market,hits:Int,total:Int,streak:Int,pushes:Int)] {
  markets.compactMap { m in
   guard let line=m.line,let side=m.side?.lowercased(),["over","under"].contains(side) else{return nil}
   let games=Array((histories[key(m)] ?? []).prefix(window));guard !games.isEmpty else{return nil}
   func hit(_ g:PlayerGame)->Bool { side == "over" ? g.value>line:g.value<line }
   return (m,games.filter(hit).count,games.count,games.prefix(while:hit).count,games.filter{$0.value==line}.count)
  }.sorted { a,b in
   let ar=Double(a.hits)/Double(a.total),br=Double(b.hits)/Double(b.total)
   if ar != br{return ar>br};if a.total != b.total{return a.total>b.total};if a.streak != b.streak{return a.streak>b.streak};return a.market.id<b.market.id
  }
 }
 var body:some View {
  VStack(alignment:.leading,spacing:12) {
   HStack { StatLabel(title:"Top hit rates").font(.system(.headline, design: .monospaced));Spacer();if loading { PLONNLoader() } }
   Picker("History window",selection:$window){Text("Last 3").tag(3);Text("Last 5").tag(5);Text("Last 10").tag(10)}.pickerStyle(.segmented)
   if requested {
    ForEach(Array(ranked.prefix(5)),id:\.market.id) { row in
     NavigationLink { DetailView(m:row.market) } label: {
      VStack(alignment:.leading,spacing:6) {
       HStack { PlayerPortrait(market:row.market,size:36);Text(row.market.name).font(.system(.subheadline, design: .monospaced).weight(.semibold));Spacer();Text("\(row.hits)/\(row.total) · \(Int((100*Double(row.hits)/Double(row.total)).rounded()))%").font(.system(.subheadline, design: .monospaced).weight(.semibold)).monospacedDigit().foregroundStyle(performanceGreen) }
       Text(compactPropLabel(row.market)).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
       if row.market.sport == "NFL" {
        let prior=priorSeasonCount(row.market)
        if prior>0 {Text("\(row.total-prior) this season · \(prior) previous season").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)}
       }
       Text("\(row.streak) straight clears · \(row.pushes) on the line").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
       if row.total < 3 { Text("Small sample · Only \(row.total) appearance\(row.total == 1 ? "":"s")").font(.system(.caption2, design: .monospaced).weight(.medium)).foregroundStyle(.secondary) }
      }.padding(10).background(PlayerTeamBackground(market:row.market))
     }.buttonStyle(.plain)
    }
    if !loading && ranked.isEmpty { Text(markets.isEmpty ? "Waiting for sportsbook prop lines to compare with player stats." : "No verified game history available for these props.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
   }
   Button(requested ? "Refresh hit rates":"Find top hit rates") { Task { histories=[:];await load() } }.font(.system(.subheadline, design: .monospaced).weight(.semibold)).disabled(loading || markets.isEmpty)
   Text("Historical clears of today’s line, not settled bet wins or predicted probability. Uses previous-season appearances when needed; 1–2 games is a small sample. Ties count separately. Ranked among loaded props only.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
  }.padding(16).background(navySurface,in:RoundedRectangle(cornerRadius:14))
  .task(id:markets.map(\.id).joined(separator:"|")) { await load() }
 }
 private func priorSeasonCount(_ market:Market)->Int {
  let calendar=Calendar.current
  let cutoff=min(market.commence.flatMap(sportsDate) ?? Date(),Date())
  let year=calendar.component(.year,from:cutoff)
  let season=calendar.component(.month,from:cutoff)<3 ? year-1:year
  return (histories[key(market)] ?? []).prefix(window).filter {game in
   let y=calendar.component(.year,from:game.date)
   let gameSeason=calendar.component(.month,from:game.date)<3 ? y-1:y
   return gameSeason<season
  }.count
 }
 @MainActor private func load() async {
  loading=true;requested=true;defer{loading=false}
  var seen=Set<String>()
  for market in markets where market.kind=="prop" && market.line != nil {
   let id=key(market);guard seen.insert(id).inserted else{continue}
   if histories[id] != nil {continue}
   guard !Task.isCancelled else {return}
   let source=PlayerHistory();await source.load(market);histories[id]=source.games
  }
 }
}

private struct MatchupEfficiency:Identifiable {
 var id:String{name}
 let name:String
 let sample:String
 let offense:[String:Double]
 let allowed:[String:Double]
 var primaryColor=""
 var secondaryColor=""
}
@MainActor private final class EfficiencyResearch:ObservableObject {
 @Published var teams:[MatchupEfficiency]=[]
 @Published var league:[MatchupEfficiency]=[]
 @Published var insightProfiles:[InsightTeamProfile]=[]
 @Published var loading=false
 @Published var message=""
 private static var cache:[String:(Date,[MatchupEfficiency],[InsightTeamProfile])]=[:]
 private var generation=UUID()
 private func rows(_ value:Any?)->[[String:Any]] {value as? [[String:Any]] ?? []}
 private func json(_ url:String) async throws->[String:Any] {
  var request=URLRequest(url:URL(string:url)!);request.timeoutInterval=20
  let(data,response)=try await URLSession.shared.data(for:request)
  guard (response as? HTTPURLResponse)?.statusCode==200 else{throw URLError(.badServerResponse)}
  return try JSONSerialization.jsonObject(with:data) as? [String:Any] ?? [:]
 }
 func load(game:SportsItem,sport:String,logs:[String:[TeamRecentGame]],force:Bool=false) async {
  let token=UUID();generation=token
  loading=true;teams=[];league=[];insightProfiles=[];message="";defer{if generation==token {loading=false}}
  let base="https://site.api.espn.com/apis/site/v2/sports/"+sportsFeedPath(sport)
  // These are explicitly season-to-date comparisons, not historical as-of snapshots.
  let now=Date(),year=Calendar.current.component(.year,from:now)
  let season=["NBA","NHL"].contains(sport) && Calendar.current.component(.month,from:now)>=10 ? year+1:year
  let phase=game.seasonType == 1 ? 1:2
  let cacheKey=sport+String(season)+"|"+String(phase)
  if !force,let cached=Self.cache[cacheKey],now.timeIntervalSince(cached.0)<900 {
   league=cached.1;teams=league.filter{$0.name==game.home || $0.name==game.away};insightProfiles=cached.2;return
  }
  if sport == "MLB" {
   do {
    var batting:[Int:(String,[String:Double])]=[:]
    var pitching:[Int:[String:Double]]=[:]
    var battingRaw:[Int:[String:Any]]=[:],pitchingRaw:[Int:[String:Any]]=[:]
    var teamNames:[Int:String]=[:]
    for group in ["hitting","pitching"] {
     guard let data=try? await json("https://statsapi.mlb.com/api/v1/teams/stats?stats=season&group=\(group)&sportIds=1&season=\(season)&gameType=R") else {continue}
     try Task.checkCancellation();guard generation==token else{return}
     for category in rows(data["stats"]) {for row in rows(category["splits"]) {
      guard let team=row["team"] as? [String:Any],let id=team["id"] as? Int,let name=team["name"] as? String,let stat=row["stat"] as? [String:Any] else{continue}
      var values:[String:Double]=[:]
      for key in ["ops","avg","obp","slg","era","whip","strikeoutsPer9Inn","gamesPlayed","runs","homeRuns","baseOnBalls","strikeOuts","plateAppearances"] {
       if let value=stat[key] {values[key]=Double(String(describing:value))}
      }
      if let games=values["gamesPlayed"],games>0 {for key in ["runs","homeRuns"] {if let v=values[key]{values[key+"PerGame"]=v/games}}}
      if let pa=values["plateAppearances"],pa>0 {for key in ["baseOnBalls","strikeOuts"] {if let v=values[key]{values[key+"Pct"]=100*v/pa}}}
      var source=stat;source["season"]=row["season"]
      teamNames[id]=name
      if group == "hitting" {batting[id]=(name,values);battingRaw[id]=source} else {pitching[id]=values;pitchingRaw[id]=source}
     }}
    }
    for id in Set(batting.keys).union(pitching.keys).sorted() {
     guard let name=teamNames[id] else{continue}
     let offense=batting[id]?.1 ?? [:]
     let defense=pitching[id] ?? [:]
     if var insight=MatchupInsightSources.mlb(team:name,requestedYear:season,batting:battingRaw[id],pitching:pitchingRaw[id]) {
      insight.through=Date();insightProfiles.append(insight)
     }
     var profile=MatchupEfficiency(name:name,sample:"\(season) season to date · batting \(Int(offense["gamesPlayed"] ?? 0)) games / pitching \(Int(defense["gamesPlayed"] ?? 0)) games",offense:offense,allowed:defense)
     if profile.name==game.home {profile.primaryColor=game.homeColor;profile.secondaryColor=game.homeAlternateColor}
     if profile.name==game.away {profile.primaryColor=game.awayColor;profile.secondaryColor=game.awayAlternateColor}
     league.append(profile)
     if profile.name==game.home || profile.name==game.away{teams.append(profile)}
    }
    if league.count==30{Self.cache[cacheKey]=(Date(),league,insightProfiles)}else{message="League coverage incomplete; unavailable ranks remain gray."}
   }catch{message="MLB comparison unavailable. Try again."}
   return
  }
  do {
   let directory=try await json(base+"/teams?limit=1000")
   try Task.checkCancellation();guard generation==token else{return}
   let all=rows(directory["sports"]).flatMap{rows($0["leagues"])}.flatMap{rows($0["teams"])}.compactMap{$0["team"] as? [String:Any]}.filter{($0["isAllStar"] as? Bool) != true}
   let ordered=all.sorted { a,b in
    let an=a["displayName"] as? String ?? "",bn=b["displayName"] as? String ?? ""
    return ([game.home,game.away].contains(an) ? 0:1) < ([game.home,game.away].contains(bn) ? 0:1)
   }
   for team in ordered {
    try Task.checkCancellation()
    guard let name=team["displayName"] as? String,let id=team["id"] as? String else{continue}
    guard let data=try? await json(base+"/teams/\(id)/statistics?season=\(season)&seasontype=\(phase)") else{continue}
    try Task.checkCancellation();guard generation==token else{return}
    var insight=MatchupInsightSources.espn(team:name,sport:sport,requestedYear:season,payload:data,seasonType:phase)
    insight?.through=Date()
    let results=data["results"] as? [String:Any] ?? [:]
    func extract(_ groups:Any?)->[String:Double] {
     var result:[String:Double]=[:]
     for category in rows(groups) { for stat in rows(category["stats"]) {
      if let key=stat["name"] as? String,let value=stat["value"] as? Double,value.isFinite{result[key]=value}
     } }
     return result
    }
    let own=extract((results["stats"] as? [String:Any])?["categories"]),opp=extract(results["opponent"])
    var offense:[String:Double]=[:],allowed:[String:Double]=[:]
    let games=own["teamGamesPlayed"] ?? own["gamesPlayed"] ?? own["games"] ?? 0
    guard games>0 else{continue}
    if sport == "NFL" {
     offense=own;allowed=opp
     // Preserve category meaning: passing sacks/INTs are conceded/thrown,
     // whereas defensive categories use the same field names for plays made.
     func passing(_ groups:Any?)->[String:Double] {extract(rows(groups).filter{$0["name"] as? String == "passing"})}
     let pass=passing((results["stats"] as? [String:Any])?["categories"]),against=passing(results["opponent"])
     for (source,isDefense) in [(pass,false),(against,true)] {
      var values=isDefense ? allowed:offense
      if let attempts=source["passingAttempts"],attempts>0 {
       if let interceptions=source["interceptions"] {values["interceptionRate"]=100*interceptions/attempts}
       if let sacks=source["sacks"] {values["sackRate"]=100*sacks/(attempts+sacks);values["sacksPerGame"]=sacks/games}
      }
      if let interceptions=source["interceptions"] {values["interceptionsPerGame"]=interceptions/games}
      if isDefense {allowed=values} else {offense=values}
     }
    } else if sport == "NHL" {
     offense=insight?.offense ?? [:];allowed=insight?.allowed ?? [:]
    } else {
     guard let fga=own["fieldGoalsAttempted"],let fta=own["freeThrowsAttempted"],let orb=own["offensiveRebounds"],let to=own["turnovers"],let points=own["points"] else{if let insight {insightProfiles.append(insight)};continue}
     let possessions=fga+0.44*fta-orb+to
     guard possessions>0 else{if let insight {insightProfiles.append(insight)};continue}
     offense=["rating":points/possessions*100,"possessions":possessions/games,"gamesPlayed":games,"estimatedPossessions":possessions,"points":points]
     for key in ["fieldGoalPct","threePointPct","freeThrowPct","avgPoints","avgAssists","avgRebounds","avgThreePointFieldGoalsMade","avgTurnovers","avgOffensiveRebounds","avgDefensiveRebounds","avgSteals","avgBlocks","assistTurnoverRatio"] {offense[key]=own[key]}
     if let made=own["fieldGoalsMade"],let threes=own["threePointFieldGoalsMade"],fga>0 {offense["efgPct"]=100*(made+0.5*threes)/fga}
     if fga+0.44*fta>0 {offense["tsPct"]=100*points/(2*(fga+0.44*fta))}
     offense["turnoverRate"]=100*to/possessions
     if let rebounds=own["totalRebounds"] ?? own["rebounds"]{offense["rebounds"]=rebounds/games}
     if let schedule=try? await json(base+"/teams/\(id)/schedule?season=\(season)&seasontype=\(phase)"),
        let average=MatchupInsightSources.nbaScheduleAllowance(teamID:id,payload:schedule,sample:Int(games),cutoff:min(sportsDate(game.start) ?? now,now),excludingEventID:game.id) {
      allowed["rating"]=average*games/possessions*100;allowed["points"]=average*games;allowed["avgPoints"]=average
      insight?.allowed["avgPoints"]=average;insight?.through=Date()
     }
    }
    try Task.checkCancellation();guard generation==token else{return}
    if let insight {insightProfiles.append(insight)}
    var profile=MatchupEfficiency(name:name,sample:insight?.source ?? "\(season) regular season to date · \(Int(games)) games",offense:offense,allowed:allowed)
    profile.primaryColor=team["color"] as? String ?? "";profile.secondaryColor=team["alternateColor"] as? String ?? ""
    league.append(profile)
    if name==game.home || name==game.away {teams.append(profile)}
   }
   if league.count == (["NFL","NHL"].contains(sport) ? 32:30) {Self.cache[cacheKey]=(Date(),league,insightProfiles)} else {message="League coverage incomplete; unavailable ranks remain gray."}
  } catch is CancellationError {message="Comparison cancelled."} catch {message="Efficiency data unavailable. Try again."}
 }
 static func lowerBetter(_ key:String,defense:Bool)->Bool {
  if ["sackRate","sacksPerGame","interceptionRate","interceptionsPerGame"].contains(key) {return !defense}
  if ["turnoverRate","avgTurnovers","strikeOutsPct"].contains(key) {return !defense}
  return defense && key != "strikeoutsPer9Inn"
 }
 func benchmark(_ key:String,defense:Bool,sport:String)->Double? {
  let expected=["NFL","NHL"].contains(sport) ? 32:30
  let values=league.compactMap{defense ? $0.allowed[key]:$0.offense[key]}
  guard values.count==expected else{return nil}
  // Equal-team mean, explicitly labeled in the UI.
  return values.reduce(0,+)/Double(expected)
 }
 func rank(_ value:Double,key:String,defense:Bool,sport:String)->Int? {
  let values=league.compactMap{defense ? $0.allowed[key]:$0.offense[key]}
  guard values.count==(["NFL","NHL"].contains(sport) ? 32:30) else{return nil}
  let rounded=(value*100).rounded()
  let lowerBetter=Self.lowerBetter(key,defense:defense)
  return 1+values.filter{lowerBetter ? ($0*100).rounded()<rounded:($0*100).rounded()>rounded}.count
 }
}
private struct MatchupAdvantagesView:View {
 let game:SportsItem;let sport:String;let logs:[String:[TeamRecentGame]]
 let markets:[Market]
 let showInsights:Bool
 var researchLoading = false
 @ObservedObject var gameUpdates:LiveMatchupStore
 @EnvironmentObject private var account:AccountStore
 @Environment(\.scenePhase) private var scenePhase
 @State private var verifiedPlayerTeams:[String:String]=[:]
 @StateObject private var data=EfficiencyResearch()
 @State private var expanded=false
 @State private var selectedGroup=""
 @State private var loadedResearchSignature=""
 @State private var loadedRosterSignature=""
 var body:some View {
  VStack(alignment:.leading,spacing:14) {
   if showInsights {
    VStack(alignment:.leading,spacing:12) {
     TimelineView(.periodic(from:.now,by:15)) { timeline in
     if LiveMatchupPolicy.showsGameContext(event:sharedReportEvent,state:game.state,snapshot:gameUpdates.snapshot,now:timeline.date) {
      LiveMatchupCard(event:sharedReportEvent,snapshot:gameUpdates.snapshot,loading:gameUpdates.loading,failed:gameUpdates.failed,now:timeline.date)
     } else if account.hasSignedInAccount {
     SharedMatchupReportCard(event:sharedReportEvent,client:account.sharedMatchupReportClient(),accessIdentity:account.user?.id.uuidString ?? "signed-out",narrativeClient:account.matchupNarrativeClient(),markets:markets,currentFacts:MatchupOutlookSelection.currentFacts(event:sharedReportEvent,profiles:data.insightProfiles,now:timeline.date),sourceLoading:data.loading,ready:outlookReady,embedded:true,unavailableMessage:(CloudConfiguration.sharedReportsEnabled || SharedMatchupFrozenArchive.bundled) ? "Sign in to view this matchup’s outlook." : "Matchup outlook is unavailable.")
     } else {AccountFeaturePrompt(feature:.aiResearch)}
     }
     if !contextualInsights.isEmpty {
      Divider()
      Text("Historical team context").font(.subheadline.bold())
     }
     ForEach(contextualInsights) { insight in
      VStack(alignment:.leading,spacing:5) {
       Text(insight.text).font(.subheadline).accessibilityIdentifier("matchup-outlook-supporting-evidence")
       if let marketID=insight.marketID,let market=markets.first(where:{$0.id==marketID}) {
        NavigationLink {DetailView(m:market)} label: {
         HStack(spacing:6) {Text("View "+compactPropLabel(market));Image(systemName:"chevron.right")}
          .font(.system(.caption, design: .monospaced).bold())
        }.buttonStyle(.plain).modifier(PlayerLinkAccent(market:market))
         .accessibilityLabel("View "+market.name+" "+compactPropLabel(market)+" at "+market.book)
       }
      }
     }
    if !contextualInsights.isEmpty {
     Text("Team history may cover different periods and does not predict an individual player’s result.").font(.caption).foregroundStyle(.secondary)
    }
    if !opponentHistoryMarkets.isEmpty {
     Divider()
     Text("Player vs opponent").font(.system(.subheadline,design:.monospaced).bold())
     ForEach(opponentHistoryMarkets) {market in OpponentPropTrendView(market:market,showSourceDetails:false)}
    }
    }.padding(16).background(Color(uiColor:.systemBackground),in:RoundedRectangle(cornerRadius:18))
     .accessibilityElement(children:.contain).accessibilityIdentifier("matchup-outlook-card")
   }
   Text("Team comparison").font(.system(.headline, design: .monospaced))
   ScrollView(.horizontal,showsIndicators:false) {
    HStack(spacing:8) {ForEach(groups,id:\.self) {group in
     Button {selectedGroup=group} label: {
      Text(group).font(.system(.caption,design:.monospaced).bold())
       .foregroundStyle(activeGroup==group ? brandAccent:Color.primary)
       .padding(.horizontal,12).frame(minHeight:44)
       .plonnGlassSurface(selected:activeGroup==group,radius:22)
     }.buttonStyle(.plain)
      .accessibilityAddTraits(activeGroup==group ? .isSelected:[])
    }}.plonnGlassGroup()
   }
   if researchLoading || data.loading || loadedResearchSignature != researchSignature {PLONNLoader("Comparing teams…")}
   ForEach(loadedResearchSignature == researchSignature ? data.teams:[]) { team in
    if let opponent=data.teams.first(where:{$0.name != team.name}) {
     if sport == "NFL" {nflComparison(offense:team,defense:opponent)}
     else if team.name == data.teams.first?.name {leagueComparison(team,opponent)}
    }
   }
   if loadedResearchSignature == researchSignature && !data.message.isEmpty{Text(data.message).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
   Text("Descriptive matchup comparison, not a prop projection. Opponent strength, injuries and expected game flow are not included.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
   if sport == "NBA" {Text("Possessions estimated as FGA + 0.44 × FTA − offensive rebounds + turnovers, using each team’s season totals. Defensive rating uses that same estimated denominator. Includes overtime; tempo is not pace per 48 minutes.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)}
   Button("Refresh comparison") {
    #if DEBUG && targetEnvironment(simulator)
    if MatchupOfflineFixture.active {return}
    #endif
    Task{await data.load(game:game,sport:sport,logs:logs,force:true)}
   }.font(.system(.caption, design: .monospaced)).disabled(data.loading)
  }.task(id:researchSignature) {
    guard !researchLoading,scenePhase == .active else{return}
    let signature=researchSignature
    #if DEBUG && targetEnvironment(simulator)
    if MatchupOfflineFixture.active { loadedResearchSignature=signature;return }
    #endif
    await data.load(game:game,sport:sport,logs:logs)
    guard !Task.isCancelled,researchSignature==signature else{return}
    loadedResearchSignature=signature
    while !Task.isCancelled,scenePhase == .active,researchSignature == signature {
     do {try await Task.sleep(for:.seconds(600))} catch {return}
     guard !Task.isCancelled,scenePhase == .active,researchSignature == signature else{return}
     await data.load(game:game,sport:sport,logs:logs,force:true)
    }
   }
   .task(id:rosterSignature) {
    let signature=rosterSignature
    await loadVerifiedPlayers()
    guard !Task.isCancelled,rosterSignature==signature else{return}
    loadedRosterSignature=signature
   }
 }
 private var sharedReportEvent:SharedMatchupEvent {
  SharedMatchupEvent(sport:sport,eventID:"espn:"+game.id,homeTeam:game.home,awayTeam:game.away,startISO:SharedMatchupEvent.scoreboardStart(game.start))
 }
 private var outlookReady:Bool {
  #if DEBUG && targetEnvironment(simulator)
  if MatchupOfflineFixture.active {return false}
  #endif
  return true
 }
 private var researchSignature:String {
  game.id+String(researchLoading)+String(scenePhase == .active)+logs.keys.sorted().map {team in
   team+(logs[team] ?? []).map {"\($0.id)|\($0.date.timeIntervalSince1970)|\($0.scored)|\($0.allowed)"}.joined(separator:";")
  }.joined(separator:"|")
 }
 private var rosterSignature:String {
  sport+game.id+game.home+game.away+markets.map {"\($0.id)|\($0.name)"}.sorted().joined(separator:";")
 }
 @MainActor private func loadVerifiedPlayers() async {
  verifiedPlayerTeams=[:]
  #if DEBUG && targetEnvironment(simulator)
  if MatchupOfflineFixture.active {return}
  #endif
  guard showInsights,!["post","final","completed"].contains(game.state.lowercased()) else{return}
  var assignments:[String:String]=[:]
  for market in markets where market.kind == "prop" && market.sport == sport && market.mode == "live" {
   guard !Task.isCancelled else{return}
   if let team=await PlayerTeamColors.shared.insightTeam(market,home:game.home,away:game.away) {assignments[market.id]=team}
   guard !Task.isCancelled else{return}
  }
  verifiedPlayerTeams=assignments
 }
 private var contextualInsights:[MatchupInsight] {
  guard !researchLoading,loadedResearchSignature == researchSignature else{return []}
  return MatchupOutlookPresentation.supportingInsights(candidateInsights)
 }
 private var candidateInsights:[MatchupInsight] {
  #if DEBUG && targetEnvironment(simulator)
  if MatchupOfflineFixture.active {return MatchupOfflineFixture.insights}
  #endif
  let matchup=InsightMatchup(sport:sport,eventID:game.id,homeTeam:game.home,awayTeam:game.away,commence:game.start)
  let cutoff=min(sportsDate(game.start) ?? Date(),Date())
  let profiles=data.insightProfiles.filter {$0.team==game.home || $0.team==game.away}
  var recentProfiles:[InsightTeamProfile]=[]
  if ["NBA","NHL"].contains(sport) {
   let key=sport == "NBA" ? "avgPoints":"avgGoals"
   let usable=profiles.filter {$0.through.map{$0<cutoff} ?? true}
   if usable.count != 2 || !usable.allSatisfy({$0.offense[key] != nil && $0.allowed[key] != nil}) {
    recentProfiles=[game.home,game.away].compactMap {team in
     MatchupInsightSources.recent(team:team,sport:sport,games:(logs[team] ?? []).map {InsightGameObservation(id:$0.id,date:$0.date,scored:$0.scored,allowed:$0.allowed)},cutoff:cutoff)
    }
   }
  }
  let props=markets.compactMap {market -> InsightPlayerProp? in
   guard let team=verifiedPlayerTeams[market.id] else{return nil}
   return InsightPlayerProp(market:market,verifiedPlayerTeam:team)
  }
  let all=MatchupInsightEngine.generate(matchup:matchup,profiles:profiles,props:props,recentProfiles:recentProfiles)
  let ordered=all.sorted {a,b in
   if (a.marketID != nil) != (b.marketID != nil) {return a.marketID != nil}
   if a.hasPairedEvidence != b.hasPairedEvidence {return a.hasPairedEvidence}
   return (all.firstIndex(where:{$0.id==a.id}) ?? 0)<(all.firstIndex(where:{$0.id==b.id}) ?? 0)
  }
  var selected:[MatchupInsight]=[]
  for team in [game.home,game.away] {if let insight=ordered.first(where:{$0.team==team}) {selected.append(insight)}}
  for insight in ordered where selected.count<GemmaMatchupInsightSession.maximumObservationCount && !selected.contains(where:{$0.id==insight.id}) {selected.append(insight)}
  return selected
 }
 private var opponentHistoryMarkets:[Market] {
  guard !researchLoading,loadedRosterSignature == rosterSignature else{return []}
  return OpponentPropHistoryEngine.insightMarkets(markets:markets,matchup:InsightMatchup(sport:sport,eventID:game.id,homeTeam:game.home,awayTeam:game.away,commence:game.start),verifiedPlayerTeams:verifiedPlayerTeams)
 }
 private var groups:[String] {MatchupInsightSources.comparisonGroups(sport:sport)}
 private var activeGroup:String {groups.contains(selectedGroup) ? selectedGroup:(groups.first ?? "")}
 private var metrics:[(String,String)] {
  switch activeGroup {
  case "Scoring":return [("thirdDownConvPct","Third-down conversion %"),("redzoneTouchdownPct","Red-zone TD %"),("yardsPerGame","Total yards / game")]
  case "Passing":return [("passingYardsPerGame","Passing yards / game"),("yardsPerPassAttempt","Passing yards / attempt"),("passingTouchdownPct","Passing TD %")]
  case "Rushing":return [("rushingYardsPerGame","Rushing yards / game"),("yardsPerRushAttempt","Rushing yards / carry")]
  case "Disruption":return [("sackRate","Sack rate %"),("sacksPerGame","Sacks / game"),("interceptionRate","Interception rate %"),("interceptionsPerGame","Interceptions / game")]
  case "Efficiency":return [("rating","Offensive rating"),("defensiveRating","Defensive rating"),("possessions","Estimated possessions / game")]
  case "Shooting":return [("efgPct","Effective FG %"),("tsPct","True shooting %"),("threePointPct","Three-point %"),("freeThrowPct","Free-throw %")]
  case "Possession":return [("turnoverRate","Turnover rate %"),("assistTurnoverRatio","Assist / turnover ratio"),("avgOffensiveRebounds","Offensive rebounds / game")]
  case "Defense":return [("defensiveRating","Defensive rating"),("avgDefensiveRebounds","Defensive rebounds / game"),("avgSteals","Steals / game"),("avgBlocks","Blocks / game")]
  case "Goals":return [("avgGoals","Goals / game")]
  case "Shots":return [("avgShots","Shots on goal / game")]
  case "Goal prevention":return [("avgGoals","Goals allowed / game"),("avgShots","Shots allowed / game")]
  case "Batting":return [("ops","OPS"),("avg","Batting average"),("obp","On-base percentage")]
  case "Power":return [("slg","Slugging percentage"),("homeRunsPerGame","Home runs / game"),("runsPerGame","Runs / game")]
  case "Discipline":return [("baseOnBallsPct","Walk rate %"),("strikeOutsPct","Strikeout rate %")]
  default:return [("era","ERA"),("whip","WHIP"),("strikeoutsPer9Inn","K / 9")]
  }
 }
 private func matchupExplanation(title:String,key:String,left:String,right:String,a:Double?,b:Double?,opposing:Bool)->String {
  guard let a,let b,a.isFinite,b.isFinite else {return "There isn’t enough data from both teams to explain this comparison yet."}
  let format=["ops","avg","obp","slg"].contains(key) ? "%.3f":"%.2f"
  let av=String(format:format,a),bv=String(format:format,b)
  let historical="These are historical rates against different opponents, not a forecast for this game."
  if opposing && key == "thirdDownConvPct" {
   return "\(left) has converted \(av)% of third downs—about \(Int(a.rounded())) out of 100 attempts. \(right) has allowed opponents to convert \(bv)%, about \(Int(b.rounded())) out of 100. This matchup tests whether \(left) can sustain drives against a defense that has allowed a \(b < a ? "lower":"higher") conversion rate than the offense has achieved. " + historical
  }
  let disruption=["sackRate","sacksPerGame","interceptionRate","interceptionsPerGame"].contains(key)
  if opposing {
   let action=disruption ? "concedes":"produces"
   let response=disruption ? "forces":"allows"
   let implication=disruption ? "This compares the offense’s exposure to lost plays or turnovers with the defense’s record of creating them. Higher defensive numbers mean more disruption, not more yards allowed." : "This compares the offense’s production with what the opposing defense has given up. The gap is a point to investigate, not an expected total or a guaranteed advantage."
   return "For \(title.lowercased()), \(left) \(action) \(av), while \(right) \(response) \(bv). \(implication) \(historical)"
  }
  let lower=EfficiencyResearch.lowerBetter(key == "defensiveRating" ? "rating":key,defense:activeGroup == "Pitching" || activeGroup == "Goal prevention" || key == "defensiveRating")
  let volume=["possessions","avgOffensiveRebounds","avgDefensiveRebounds"].contains(key)
  let interpretation=volume ? "This shows the teams’ recorded volume; pace and opportunities affect the comparison." : "\(lower ? "Lower":"Higher") values indicate stronger performance on this measure."
  return "\(left): \(av). \(right): \(bv). Both values measure \(title.lowercased()). \(interpretation) This compares the same team statistic on both sides, not offense against opponent allowance. \(historical)"
 }
 private func leagueComparison(_ left:MatchupEfficiency,_ right:MatchupEfficiency)->some View {
  VStack(alignment:.leading,spacing:12) {
   HStack(spacing:10) {nflIdentity(left,role:activeGroup.uppercased());nflIdentity(right,role:activeGroup.uppercased())}
   ForEach(metrics,id:\.0) {key,title in
    VStack(alignment:.leading,spacing:6) {
     StatLabel(title:title,matchupContext:matchupExplanation(title:title,key:key,left:left.name,right:right.name,a:(activeGroup == "Pitching" || activeGroup == "Goal prevention" || key == "defensiveRating") ? left.allowed[key == "defensiveRating" ? "rating":key]:left.offense[key],b:(activeGroup == "Pitching" || activeGroup == "Goal prevention" || key == "defensiveRating") ? right.allowed[key == "defensiveRating" ? "rating":key]:right.offense[key],opposing:false)).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
     HStack(spacing:10) {
      ForEach([left,right]) {team in
       let defense=activeGroup == "Pitching" || activeGroup == "Goal prevention" || key == "defensiveRating"
       let actualKey=key == "defensiveRating" ? "rating":key
       benchmarkMetric("",value:defense ? team.allowed[actualKey]:team.offense[actualKey],key:actualKey,defense:defense)
        .padding(12).background(TeamIdentityBackground(primary:team.primaryColor,secondary:team.secondaryColor))
      }
     }
    }
   }
  }.padding(14).background(navySurface,in:RoundedRectangle(cornerRadius:16))
 }
 private func nflComparison(offense:MatchupEfficiency,defense:MatchupEfficiency)->some View {
  VStack(alignment:.leading,spacing:12) {
   HStack(alignment:.top,spacing:10) {
    nflIdentity(offense,role:"OFFENSE")
    Text("vs").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).padding(.top,20)
    nflIdentity(defense,role:"DEFENSE")
   }
   ForEach(metrics,id:\.0) { key,title in
    VStack(alignment:.leading,spacing:7) {
     StatLabel(title:title,matchupContext:matchupExplanation(title:title,key:key,left:offense.name,right:defense.name,a:offense.offense[key],b:defense.allowed[key],opposing:true)).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
     HStack(alignment:.top,spacing:10) {
      benchmarkMetric(activeGroup == "Disruption" ? "Concedes":"Produces",value:offense.offense[key],key:key,defense:false)
       .padding(12).background(TeamIdentityBackground(primary:offense.primaryColor,secondary:offense.secondaryColor))
      benchmarkMetric(activeGroup == "Disruption" ? "Forces":"Allows",value:defense.allowed[key],key:key,defense:true)
       .padding(12).background(TeamIdentityBackground(primary:defense.primaryColor,secondary:defense.secondaryColor))
     }
    }
   }
   if activeGroup == "Disruption" {Text("Pressure rate unavailable from this feed.").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)}
  }.padding(14).background(navySurface,in:RoundedRectangle(cornerRadius:16))
 }
 private func nflIdentity(_ team:MatchupEfficiency,role:String)->some View {
  VStack(alignment:.leading,spacing:5) {
   Text(role).font(.system(size:10,weight:.bold,design:.monospaced)).foregroundStyle(.secondary)
   Text(team.name).font(.system(.subheadline, design: .monospaced).bold()).fixedSize(horizontal:false,vertical:true)
  }.frame(maxWidth:.infinity,alignment:.leading).frame(minHeight:58,alignment:.topLeading)
   .padding(10).background(TeamIdentityBackground(primary:team.primaryColor,secondary:team.secondaryColor))
 }
 private func comparison(_ title:String,key:String,offense:Double?,allowed:Double?)->some View {
  VStack(alignment:.leading,spacing:6) {
   StatLabel(title:title).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
   HStack(alignment:.top) { benchmarkMetric("Produced",value:offense,key:key,defense:false);benchmarkMetric("Opponent allows",value:allowed,key:key,defense:true) }
  }
 }
 private func benchmarkMetric(_ title:String,value:Double?,key:String,defense:Bool)->some View {
  let average=data.benchmark(key,defense:defense,sport:sport)
  let delta=(value ?? 0)-(average ?? 0)
  let comparable=value != nil && average != nil
  let equal=abs(delta)<0.005
  let better=EfficiencyResearch.lowerBetter(key,defense:defense) ? delta<0:delta>0
  let color:Color = !comparable || equal ? .secondary:better ? performanceGreen:Color(red:0.72,green:0.14,blue:0.19)
  return VStack(alignment:.leading,spacing:5) {
   if !title.isEmpty {StatLabel(title:title).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
   HStack(alignment:.firstTextBaseline,spacing:5) {
    Text(value.map{String(format:(key == "ops" || key == "avg") ? "%.3f":"%.2f",$0)} ?? "—").font(.system(.headline, design: .monospaced)).foregroundStyle(color)
    if let value,let rank=data.rank(value,key:key,defense:defense,sport:sport) {Text("#\(rank)/\(["NFL","NHL"].contains(sport) ? 32:30)").font(.system(size:10,weight:.medium,design:.monospaced)).foregroundStyle(.secondary)}
   }
  }.frame(maxWidth:.infinity,alignment:.leading)
 }

}


func matchupColor(_ hex:String)->Color {
 let clean=hex.trimmingCharacters(in:CharacterSet(charactersIn:"#"))
 guard clean.count==6,let value=UInt32(clean,radix:16) else{return charcoalAccent}
 return Color(red:Double((value>>16)&255)/255,green:Double((value>>8)&255)/255,blue:Double(value&255)/255)
}
private struct MatchupPaletteKey:EnvironmentKey {
 static let defaultValue:[Color]=[charcoalAccent,Color.gray]
}
private extension EnvironmentValues {
 var matchupPalette:[Color] {
  get {self[MatchupPaletteKey.self]}
  set {self[MatchupPaletteKey.self]=newValue}
 }
}

private struct TeamIdentityBackground:View {
 @Environment(\.colorScheme) private var colorScheme
 let primary:String;let secondary:String
 private var dark:Bool {colorScheme == .dark}
 var body:some View {
  RoundedRectangle(cornerRadius:14).fill(dark ? Color(white:0.22):navySurface)
   .overlay {LinearGradient(colors:[matchupColor(primary).opacity(dark ? 0.42:0.22),matchupColor(primary).opacity(dark ? 0.12:0.05),matchupColor(secondary.isEmpty ? primary:secondary).opacity(dark ? 0.42:0.22)],startPoint:.topLeading,endPoint:.bottomTrailing)}
   .clipShape(RoundedRectangle(cornerRadius:14))
   .overlay {RoundedRectangle(cornerRadius:14).strokeBorder(dark ? Color.white.opacity(0.22):Color.black.opacity(0.06),lineWidth:1)}
   .shadow(color:Color.black.opacity(dark ? 0.2:0),radius:4,y:2)
 }
}


private actor PlayerTeamColors {
 static let shared=PlayerTeamColors()
 private var tasks:[String:Task<[String:[String]],Never>]=[:]
 private var insightRosters:[String:(Date,[String:[String]])]=[:]
 private static func key(_ name:String)->String { name.folding(options:[.diacriticInsensitive,.caseInsensitive],locale:Locale(identifier:"en_US_POSIX")).filter{$0.isLetter || $0.isNumber} }
 private static func json(_ url:String) async throws->[String:Any] {
  var request=URLRequest(url:URL(string:url)!);request.timeoutInterval=20
  let(data,response)=try await URLSession.shared.data(for:request)
  guard (response as? HTTPURLResponse)?.statusCode==200 else {throw URLError(.badServerResponse)}
  return try JSONSerialization.jsonObject(with:data) as? [String:Any] ?? [:]
 }
 func insightTeam(_ market:Market,home:String,away:String) async->String? {
  guard LaunchSports.supported.contains(market.sport),!home.isEmpty,!away.isEmpty,home != away else{return nil}
  let cacheKey=market.sport+"|"+home+"|"+away
  if insightRosters[cacheKey].map({Date().timeIntervalSince($0.0)>=900}) ?? true {
   do {
    let base="https://site.api.espn.com/apis/site/v2/sports/"+sportsFeedPath(market.sport)
    let data=try await Self.json(base+"/teams?limit=1000")
    let sports=data["sports"] as? [[String:Any]] ?? []
    let leagues=sports.first?["leagues"] as? [[String:Any]] ?? []
    let entries=leagues.first?["teams"] as? [[String:Any]] ?? []
    var rosters:[String:[String]]=[:]
    for expected in [home,away] {
     let matching=entries.compactMap{$0["team"] as? [String:Any]}.filter{$0["displayName"] as? String == expected}
     guard matching.count==1,let id=matching.first?["id"] as? String else{return nil}
     let roster=try await Self.json(base+"/teams/"+id+"/roster")
     try Task.checkCancellation()
     guard let groups=roster["athletes"] as? [[String:Any]],!groups.isEmpty else{return nil}
     rosters[expected]=groups.flatMap{$0["items"] as? [[String:Any]] ?? [$0]}.compactMap{$0["displayName"] as? String}
    }
    insightRosters[cacheKey]=(Date(),rosters)
   } catch {return nil}
  }
  guard let rosters=insightRosters[cacheKey]?.1 else{return nil}
  return MatchupInsightSources.uniqueRosterTeam(player:market.name,rosters:rosters,expectedTeams:[home,away])
 }
 func colors(_ market:Market) async->[String] {
  let cacheKey=market.sport+market.aliases
  if tasks[cacheKey]==nil {
   tasks[cacheKey]=Task {
    var result:[String:[String]]=[:]
    do {
     let base="https://site.api.espn.com/apis/site/v2/sports/"+sportsFeedPath(market.sport)
     let data=try await Self.json(base+"/teams?limit=1000")
     let sports=data["sports"] as? [[String:Any]] ?? []
     let leagues=sports.first?["leagues"] as? [[String:Any]] ?? []
     for entry in leagues.first?["teams"] as? [[String:Any]] ?? [] {
      guard let team=entry["team"] as? [String:Any],let name=team["displayName"] as? String,market.aliases.localizedCaseInsensitiveContains(name),let id=team["id"] as? String else {continue}
      let roster=try await Self.json(base+"/teams/"+id+"/roster")
      let groups=roster["athletes"] as? [[String:Any]] ?? []
      for player in groups.flatMap({$0["items"] as? [[String:Any]] ?? [$0]}) {
       guard let name=player["displayName"] as? String else {continue}
       let k=Self.key(name)
       if result[k] != nil {result[k]=[]} else {result[k]=[team["color"] as? String ?? "",team["alternateColor"] as? String ?? "",team["displayName"] as? String ?? ""]}
      }
     }
    } catch {return [:]}
    return result
   }
  }
  let result=await tasks[cacheKey]!.value
  if result.isEmpty {tasks[cacheKey]=nil}
  return result[Self.key(market.name)] ?? []
 }
}
struct PlayerTeamBackground:View {
 let market:Market
 @State private var colors:[String]=[]
 var body:some View {
  TeamIdentityBackground(primary:colors.first ?? "",secondary:colors.count>1 ? colors[1]:"")
   .task(id:market.sport+market.aliases+market.name) {colors=await PlayerTeamColors.shared.colors(market)}
 }
}


struct StatLabel:View {
 let title:String
 var matchupContext:String? = nil
 @State private var showing=false
 private var explanation:String? {StatGlossary.definition(title)}
 var body:some View {
  HStack(spacing:4) {
   Text(title)
   if let explanation {
    Button {showing=true} label: {
     Image(systemName:"questionmark").font(.system(.caption,design:.monospaced))
      .frame(width:28,height:28).plonnGlassSurface(radius:14)
      .frame(width:44,height:44).contentShape(Rectangle())
    }.buttonStyle(.plain).accessibilityLabel("Define "+title)
     .sheet(isPresented:$showing) {
      NavigationStack {
       ScrollView {VStack(alignment:.leading,spacing:16) {Text(explanation);if let matchupContext {Text("Matchup context").font(.system(.headline, design: .monospaced)).foregroundStyle(brandAccent);Text(matchupContext)}}.font(.system(.body, design: .monospaced)).frame(maxWidth:.infinity,alignment:.leading).padding(24)}
        .navigationTitle(title).navigationBarTitleDisplayMode(.inline)
        .toolbar {ToolbarItem(placement:.confirmationAction){Button("Done"){showing=false}}}
      }.presentationDetents([.medium,.large])
     }
   }
  }
 }
}
private enum StatGlossary {
 static func definition(_ title:String)->String? {
  if title.hasPrefix("Over ") || title.hasPrefix("Under ") {return "The number or share of games finishing strictly above or below the displayed line. A result exactly on the line is counted separately as a push. Historical results are not a prediction."}
  let definitions:[String:String]=[
   "Third-down conversion %":"Successful third downs divided by third-down attempts, multiplied by 100. Higher is better for offense; lower allowed is better for defense.",
   "Red-zone TD %":"The percentage of trips inside the opponent’s 20-yard line ending in a touchdown. Higher is better for offense; lower allowed is better for defense.",
   "Total yards / game":"Total offensive yards divided by games played. The defensive column shows opponent production allowed.",
   "Passing yards / game":"Passing yards divided by games played. The defensive column shows opponent passing yards allowed.",
   "Passing TD %":"Passing touchdowns divided by pass attempts, multiplied by 100.",
   "Rushing yards / game":"Rushing yards divided by games played. The defensive column shows opponent rushing yards allowed.",
   "Sack rate %":"Sacks divided by pass attempts plus sacks, multiplied by 100. Lower conceded is better for offense; higher forced is better for defense. This is not pressure rate.",
   "Sacks / game":"Sacks conceded by the offense or recorded by the defense divided by games played.",
   "Interception rate %":"Interceptions divided by pass attempts, multiplied by 100. Lower thrown is better for offense; higher forced is better for defense.",
   "Interceptions / game":"Interceptions thrown by the offense or forced by the defense divided by games played.",
   "Offensive rating":"Points scored per 100 estimated possessions. Higher is better. Possessions use FGA + 0.44 × FTA − offensive rebounds + turnovers.",
   "Defensive rating":"Points allowed per 100 estimated possessions. Lower is better. The estimate uses the team’s own possession denominator.",
   "Estimated possessions / game":"Estimated possessions divided by games played. Includes overtime and is not pace normalized to 48 minutes.",
   "Effective FG %":"Field goals made plus half of three-pointers made, divided by field-goal attempts, multiplied by 100. Accounts for the extra value of threes.",
   "True shooting %":"Points divided by twice the sum of field-goal attempts and 0.44 times free-throw attempts, multiplied by 100. An estimate of overall scoring efficiency.",
   "Three-point %":"Three-point field goals made divided by attempts, multiplied by 100.",
   "Free-throw %":"Free throws made divided by attempts, multiplied by 100.",
   "Turnover rate %":"Turnovers per 100 estimated possessions. Lower is better.",
   "Assist / turnover ratio":"Assists divided by turnovers. Higher means more assists per turnover.",
   "Offensive rebounds / game":"Rebounds recovered after the team’s own missed shots, divided by games.",
   "Defensive rebounds / game":"Rebounds recovered after opponent misses, divided by games.",
   "Steals / game":"Officially credited steals divided by games played.",
   "Blocks / game":"Officially credited blocked shots divided by games played.",
   "On-base percentage":"How often a batter reaches on a hit, walk or hit-by-pitch. The denominator includes at-bats, walks, hit-by-pitches and sacrifice flies.",
   "Slugging percentage":"Total bases from hits divided by at-bats. Measures hitting for extra bases.",
   "Home runs / game":"Home runs divided by games played.",
   "Walk rate %":"Walks divided by plate appearances, multiplied by 100.",
   "Strikeout rate %":"Strikeouts divided by plate appearances, multiplied by 100. Lower is better for batters.",

   "OPS":"On-base plus slugging: on-base percentage added to slugging percentage. It combines how often a batter reaches base with how many bases they earn per at-bat. Higher is better for hitters; lower allowed is better for pitchers.",
   "Batting average":"Hits divided by at-bats. An average of .300 means three hits per ten at-bats. Walks are not at-bats. Higher is better for batting; lower allowed is better for pitching.",
   "ERA":"Earned run average: earned runs allowed per nine innings pitched. Lower is better. Runs caused by certain fielding errors are excluded.",
   "WHIP":"Walks plus hits divided by innings pitched. This measures how many runners a pitcher allows through walks and hits per inning. Lower is better.",
   "K / 9":"Strikeouts per nine innings: strikeouts divided by innings pitched, multiplied by nine. Higher means more strikeouts per inning.",
   "IP":"Innings pitched. Baseball uses outs rather than decimal fractions: 5.1 means five innings and one out; 5.2 means five innings and two outs.",
   "IP / K / ER / BB":"IP: innings pitched. K: strikeouts. ER: earned runs allowed. BB: walks allowed. In innings notation, .1 is one out and .2 is two outs.",
   "Rushing yards / carry":"Rushing yards divided by rushing attempts. It measures yards gained per run. Higher is better for an offense; lower allowed is better for a defense.",
   "Passing yards / attempt":"Passing yards divided by pass attempts. It measures passing production per attempt, including incomplete passes. Higher is better for an offense; lower allowed is better for a defense. It is not adjusted net yards per attempt.",
   "Points / 100 estimated possessions":"Points scored or allowed per 100 estimated possessions. This app estimates possessions using field-goal attempts + 0.44 × free-throw attempts − offensive rebounds + turnovers. Higher offense and lower defense are better.",
   "Rebounds / game":"Total rebounds divided by games played. Rebounds recover the ball after a missed shot. This includes offensive and defensive rebounds.",
   "Runs / game":"Runs scored divided by games in the selected sample.",
   "Points / game":"Points scored divided by games in the selected sample.",
   "Allowed / game":"Opponent runs or points divided by games in the selected sample. Lower means fewer scores conceded.",
   "Average":"The sum of the selected game values divided by the number of games in that sample.",
   "Median":"The middle value after sorting the sample. For an even sample, it is the average of the two middle values.",
   "Std. deviation":"A measure of how much game results vary around their average, in the same units as the stat. Larger values mean more variation.",
   "Sample":"The number of verified game appearances used in the calculation. Small samples can produce unstable percentages.",
   "Top hit rates":"The share of available recent appearances clearing the displayed prop line, ranked among loaded props. Over requires a value above the line; under requires a value below it. Ties count separately. These are historical clear rates, not predicted chances or settled betting win rates.",
   "On the line":"Games with a stat exactly equal to the prop line. They are separate from over and under hits.",
   "Push":"A result exactly equal to the displayed line. Settlement depends on the sportsbook and market rules.",
   "Under":"A result strictly below the displayed line.",
   "Hits":"Official base hits credited to a batter. Walks and reaching base on an error do not count as hits.",
   "RBI":"Runs batted in: runs credited to a batter's plate appearance under official scoring rules. Not every run scored earns an RBI.",
   "Stolen bases":"Bases advanced by a runner and officially scored as a steal. Advancement on a hit or wild pitch is not a stolen base.",
   "Home runs":"Hits on which the batter rounds all bases and scores, including inside-the-park home runs.",
   "Total bases":"Bases earned from hits: a single counts as one, a double two, a triple three, and a home run four. Walks and steals are excluded.",
   "Runs":"Times the player crosses home plate and is credited with a run.",
   "Strikeouts":"Batters a pitcher retires on a third strike, under official scoring rules.",
   "Pass yards":"Yards credited to a player's completed passes, including yards gained by the receiver after the catch.",
   "Pass TDs":"Completed passes that result in touchdowns credited to the passer.",
   "Rush yards":"Yards gained or lost on a player's rushing attempts, under official scoring rules.",
   "Receiving yards":"Yards credited to a receiver on completed catches, including yards after the catch.",
   "Receptions":"Completed passes caught by a player.",
   "Anytime TD":"A market on whether the player scores a touchdown. Passing a touchdown generally does not count as scoring one; sportsbook rules determine settlement.",
   "Points":"A player's scored points, including free throws, two-point field goals, and three-point field goals.",
   "Rebounds":"A player's offensive and defensive rebounds combined.",
   "Assists":"Passes officially credited as directly leading to a teammate's made basket.",
   "Threes":"Three-point field goals made, not attempted.",
   "Steals":"Defensive plays officially credited with taking possession from the opponent.",
   "Blocks":"Opponent field-goal attempts officially credited as blocked by the player.",
   "PTS + REB + AST":"The player's points, rebounds, and assists added together in the same game.",
   "PTS + REB":"The player's points and rebounds added together in the same game.",
   "PTS + AST":"The player's points and assists added together in the same game.",
   "REB + AST":"The player's rebounds and assists added together in the same game.",
   "Odds":"The sportsbook price. Positive American odds show profit on a $100 stake; negative odds show the stake needed for $100 profit. Prices include the bookmaker's margin.",
   "Sportsbook":"The sportsbook's quoted odds for this selection.",
   "Model":"The model's estimated probability, when available. It is an estimate rather than a guarantee.",
   "Model fair":"Odds converted from the model probability without adding a bookmaker margin.",
   "Fair":"Odds converted from an estimated probability without adding a bookmaker margin."
  ]
  let aliases=["Pass Yds":"Pass yards","Rush Yds":"Rush yards","Rec Yds":"Receiving yards","Rec":"Receptions"]
  let key=aliases[title] ?? title.replacingOccurrences(of:"Opponent ",with:"")
  return definitions[key == "K/9" ? "K / 9":key]
 }
}


private struct LinkedPlayerTrend:Identifiable {
 let market:Market
 let text:String
 var id:String {market.id}
}
private struct PlayerMatchupProps:View {
 let player:Market
 let markets:[Market]
 private var selections:[Market] {
  markets.filter{$0.kind == "prop" && $0.name == player.name && $0.sport == player.sport}
   .sorted{$0.market == $1.market ? $0.label < $1.label:$0.market < $1.market}
 }
 var body:some View {
  ScrollView {
   VStack(alignment:.leading,spacing:14) {
    HStack(spacing:12) {
     PlayerPortrait(market:player,size:58)
     VStack(alignment:.leading,spacing:4) {
      Text(player.name).font(.system(.title2, design: .monospaced).bold())
      Text(player.eventLabel ?? "Matchup props").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
     }
    }.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(PlayerTeamBackground(market:player))
    Text("Player props").font(.system(.headline, design: .monospaced))
    ForEach(selections) {market in
     NavigationLink {DetailView(m:market)} label: {
      HStack {
       VStack(alignment:.leading,spacing:5) {
        Text(compactPropLabel(market)).font(.system(.subheadline, design: .monospaced).bold())
        Text(market.book).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
       }
       Spacer()
       Text(odds(market.price)).font(.system(.headline, design: .monospaced))
       Image(systemName:"chevron.right").font(.system(.caption, design: .monospaced))
      }.padding(14).background(PlayerTeamBackground(market:market))
     }.buttonStyle(.plain)
    }
    if selections.isEmpty {Text("No props available for this player in this snapshot.").foregroundStyle(.secondary)}
    Text("Matchup odds snapshot · Prices may change").font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
   }.padding(16)
  }.background(navy).navigationTitle("Player props").navigationBarTitleDisplayMode(.inline)
 }
}

private struct PlayerLinkAccent:ViewModifier {
 let market:Market
 @State private var colors:[String]=[]
 func body(content:Content)->some View {
  content.foregroundStyle(colors.first.map{matchupColor($0)} ?? charcoalAccent)
   .task(id:market.sport+market.name+market.aliases){colors=await PlayerTeamColors.shared.colors(market)}
 }
}

private struct LiveBaseballPlayers:View {
 let game:SportsItem
 var body:some View {
  VStack(alignment:.leading,spacing:6) {
   if let batter=game.liveBatter {Text(batter)}
   if let pitcher=game.livePitcher {Text(pitcher)}
  }.font(.system(.caption,design:.monospaced)).frame(maxWidth:.infinity,alignment:.leading).padding(.horizontal,14)
 }
}

private struct BaseOccupancy:View {
 let game:SportsItem
 private func base(_ occupied:Bool?)->some View {
  Rectangle().fill(occupied == true ? Color.orange:Color.secondary.opacity(0.15))
   .frame(width:10,height:10).rotationEffect(.degrees(45))
   .overlay {if occupied == nil {Text("?").font(.system(size:7,design:.monospaced))}}
 }
 var body:some View {
  if game.firstBase != nil || game.secondBase != nil || game.thirdBase != nil {
   VStack(spacing:1) {base(game.secondBase);HStack(spacing:15){base(game.thirdBase);base(game.firstBase)}}
    .accessibilityElement(children:.ignore)
    .accessibilityLabel("Bases: first \(description(game.firstBase)), second \(description(game.secondBase)), third \(description(game.thirdBase))")
  }
 }
 private func description(_ value:Bool?)->String {value.map{$0 ? "occupied":"empty"} ?? "unavailable"}
}

func compactPropLabel(_ market:Market)->String {
 guard market.kind == "prop",let line=market.line,let side=market.side else{return market.label}
 return side.capitalized+" "+line.formatted()+" "+propTitle(market.market)
}

struct LiveFlameIcon: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        TimelineView(.animation(minimumInterval:1.0/24.0,paused:reduceMotion)) { context in
            let time = reduceMotion ? 0 : context.date.timeIntervalSinceReferenceDate
            ZStack {
                Image(systemName:"flame.fill").foregroundStyle(Color.red)
                    .scaleEffect(reduceMotion ? 1 : 1 + 0.035 * sin(time * 5),anchor:.bottom)
                if !reduceMotion {
                    Canvas { canvas, size in
                        for index in 0..<4 {
                            let phase = (time * 0.55 + Double(index) * 0.25).truncatingRemainder(dividingBy:1)
                            let x = size.width/2 + sin(time * 1.8 + Double(index)*2) * 5
                            let y = size.height * 0.35 - phase * 15
                            let rect = CGRect(x:x,y:y,width:2,height:2.5)
                            canvas.fill(Path(ellipseIn:rect),with:.color(.orange.opacity((1-phase)*0.8)))
                        }
                    }.allowsHitTesting(false)
                }
            }.frame(width:20,height:22).accessibilityHidden(true)
        }
    }
}

private struct SoccerMatchupView:View {
 let game:SportsItem
 let sport:String
 @StateObject private var oddsStore=Store()
 @StateObject private var gameUpdates=LiveMatchupStore()
 @Environment(\.scenePhase) private var scenePhase
 private var event:SharedMatchupEvent {game.matchupEvent(sport:sport)}
 private var liveSnapshot:LiveMatchupSnapshot? {gameUpdates.snapshot?.event == event ? gameUpdates.snapshot:nil}
 private var gameContext:Bool {LiveMatchupPolicy.showsGameContext(event:event,state:game.state,snapshot:gameUpdates.snapshot,now:Date())}
 @EnvironmentObject private var account:AccountStore
 private var supportsOdds:Bool {!["Soccer:fifa.friendly"].contains(sport)}
 private var gameMarkets:[Market] {
  oddsStore.live.filter { market in
   guard market.kind != "prop",let label=market.eventLabel,label.localizedCaseInsensitiveContains(game.home),label.localizedCaseInsensitiveContains(game.away),let start=sportsDate(game.start),let quoteStart=market.commence.flatMap(sportsDate) else{return false}
   return abs(start.timeIntervalSince(quoteStart))<1800
  }
 }
 @State private var sections:[FeedSection]=[]
 @State private var loading=false
 @State private var message=""
 var body:some View {
  ScrollView {VStack(spacing:20) {
   HStack {TeamMark(url:game.awayLogo,name:game.away);Text(game.away);Spacer();Text(liveSnapshot?.phase == .scheduled ? game.awayScore : (liveSnapshot?.away.score ?? game.awayScore)).bold()}
   HStack {TeamMark(url:game.homeLogo,name:game.home);Text(game.home);Spacer();Text(liveSnapshot?.phase == .scheduled ? game.homeScore : (liveSnapshot?.home.score ?? game.homeScore)).bold()}
   Text(liveSnapshot?.status ?? game.subtitle).font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary)
   TimelineView(.periodic(from:.now,by:15)) { timeline in
    if LiveMatchupPolicy.showsGameContext(event:event,state:game.state,snapshot:gameUpdates.snapshot,now:timeline.date) {
     LiveMatchupCard(event:event,snapshot:gameUpdates.snapshot,loading:gameUpdates.loading,failed:gameUpdates.failed,now:timeline.date)
      .padding(16).background(navySurface,in:RoundedRectangle(cornerRadius:18))
    }
   }
   if supportsOdds && !gameContext {
    VStack(alignment:.leading,spacing:12) {
     Text("Odds").font(.system(.headline, design: .monospaced))
     if oddsStore.loading {PLONNLoader("Loading prices…")}
     if let issue=oddsStore.oddsIssue {OddsIssueNotice(message:issue)}
     ForEach(gameMarkets) {market in
      HStack {VStack(alignment:.leading,spacing:3) {Text(market.name).font(.system(.subheadline, design: .monospaced).weight(.semibold));Text(market.label).font(.system(.caption, design: .monospaced));Text(market.book).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)};Spacer();Text(odds(market.price)).font(.system(.headline, design: .monospaced)).monospacedDigit()}
     }
     if !oddsStore.loading && gameMarkets.isEmpty {Text("No sportsbook prices available for this matchup.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
    }.padding(14).background(navySurface,in:RoundedRectangle(cornerRadius:14))
   }
   if loading {PLONNLoader("Loading matchup…")}
   ForEach(sections.filter { !gameContext || !$0.title.hasSuffix(" · Game stats") }) { section in
    VStack(alignment:.leading,spacing:12) {Text(section.title).font(.system(.headline, design: .monospaced));ForEach(Array(section.rows.enumerated()),id:\.offset) { _, row in HStack(alignment:.top) {Text(row.0).font(.system(.subheadline, design: .monospaced));Spacer();Text(row.1).font(.system(.subheadline, design: .monospaced).weight(.semibold)).monospacedDigit()}}}.padding(14).background(navySurface,in:RoundedRectangle(cornerRadius:14))
   }
   if !message.isEmpty {Text(message).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
  }.padding(20)}.background(LeagueBackdrop(league:sport)).navigationTitle("Matchup").navigationBarTitleDisplayMode(.inline).task(id:game.id) {await load();if supportsOdds && !gameContext {await oddsStore.load(sport,account:account,home:game.home,away:game.away,start:game.start)}}.task(id:event.sport+event.eventID+event.homeTeam+event.awayTeam+event.startISO+String(describing:scenePhase)+String(gameUpdates.refreshRevision)) {guard scenePhase == .active else{return};await gameUpdates.watch(event:event)}.plonnRefreshable {gameUpdates.refreshRevision += 1;await load();if supportsOdds && !gameContext {await oddsStore.load(sport,account:account,home:game.home,away:game.away,start:game.start)}}
 }
 struct FeedSection:Identifiable {let id=UUID();let title:String;let rows:[(String,String)]}
 @MainActor private func load() async {
  loading=true;message="";defer{loading=false}
  do {
   let url=URL(string:"https://site.api.espn.com/apis/site/v2/sports/"+sportsFeedPath(sport)+"/summary?event="+game.id)!
   let(data,response)=try await URLSession.shared.data(from:url)
   guard (response as? HTTPURLResponse)?.statusCode==200,let root=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {throw URLError(.badServerResponse)}
   let box=root["boxscore"] as? [String:Any] ?? [:]
   var result:[FeedSection]=[]
   for entry in box["teams"] as? [[String:Any]] ?? [] {
    let team=entry["team"] as? [String:Any] ?? [:]
    let rows=(entry["statistics"] as? [[String:Any]] ?? []).compactMap { stat -> (String,String)? in
     guard let name=stat["label"] as? String ?? stat["displayName"] as? String ?? stat["name"] as? String,let value=stat["displayValue"] as? String else{return nil};return(name,value)
    }
    if !rows.isEmpty {result.append(FeedSection(title:(team["displayName"] as? String ?? "Team")+" · Game stats",rows:rows))}
   }
   for entry in box["players"] as? [[String:Any]] ?? [] {
    let team=entry["team"] as? [String:Any] ?? [:]
    for group in entry["statistics"] as? [[String:Any]] ?? [] {
     let labels=group["labels"] as? [String] ?? []
     let rows=(group["athletes"] as? [[String:Any]] ?? []).compactMap { item -> (String,String)? in
      let athlete=item["athlete"] as? [String:Any] ?? [:];guard let name=athlete["displayName"] as? String,let stats=item["stats"] as? [String],!stats.isEmpty else{return nil}
      return(name,zip(labels,stats).map {"\($0.0) \($0.1)"}.joined(separator:" · "))
     }
     if !rows.isEmpty {result.append(FeedSection(title:(team["displayName"] as? String ?? "Players")+" · "+(group["name"] as? String ?? "Box score"),rows:rows))}
    }
   }
   for entry in root["rosters"] as? [[String:Any]] ?? [] {
    let team=entry["team"] as? [String:Any] ?? [:]
    let rows=(entry["roster"] as? [[String:Any]] ?? []).compactMap { item -> (String,String)? in
     let athlete=item["athlete"] as? [String:Any] ?? [:];guard let name=athlete["displayName"] as? String else{return nil};return(name,item["starter"] as? Bool == true ? "Starter":"Roster")
    }
    if !rows.isEmpty {result.append(FeedSection(title:(team["displayName"] as? String ?? "Team")+" · Players",rows:rows))}
   }
   sections=result
   if result.isEmpty {message="Game statistics will appear when the feed provides them."}
  } catch {message="Matchup data is unavailable. Pull to retry."}
 }

}

private struct OtherSportEvents:View {
 let sport:String
 let date:Date
 @EnvironmentObject private var account:AccountStore
 @StateObject private var marketStore=Store()
 @State private var events:[Event]=[]
 @State private var loading=false
 @State private var error=""
 struct Event:Identifiable {let id:String;let name:String;let detail:String}
 var body:some View {
  ScrollView {LazyVStack(alignment:.leading,spacing:14) {
   if !["F1","NASCAR"].contains(sport) {NavigationLink {OtherSportOdds(sport:sport.hasPrefix("Tennis") ? "Tennis":sport)} label:{Label("Sportsbook odds",systemImage:"chart.line.uptrend.xyaxis").font(.system(.headline, design: .monospaced)).padding(12)}}
   if loading {PLONNLoader("Loading events…").frame(maxWidth:.infinity)}
   if !error.isEmpty {Text(error).foregroundStyle(.secondary)}
   ForEach(events) {event in
    VStack(alignment:.leading,spacing:8) {Text(event.name).font(.system(.headline, design: .monospaced));Text(event.detail).font(.system(.subheadline, design: .monospaced)).foregroundStyle(.secondary)}.frame(maxWidth:.infinity,alignment:.leading).padding(16).background(navySurface,in:RoundedRectangle(cornerRadius:14))
   }
   if !loading && events.isEmpty && error.isEmpty {ContentUnavailableView("No events on this date",systemImage:"calendar")}
  }.padding(16)}.task(id:sport+date.formatted(date:.numeric,time:.omitted)) {await load()}.plonnRefreshable {await load()}
 }
 @MainActor private func load() async {
  events=[];error=""
  if sport == "Boxing" {
   loading=true;defer{loading=false}
   await marketStore.load("Boxing",account:account)
   var seen=Set<String>()
   events=marketStore.live.compactMap {market in
    guard seen.insert(market.event).inserted,let start=market.commence.flatMap(sportsDate),Calendar.current.isDate(start,inSameDayAs:date) else{return nil}
    return Event(id:market.event,name:market.eventLabel ?? market.name,detail:start.formatted(date:.abbreviated,time:.shortened)+" · Sportsbook schedule")
   }
   if marketStore.live.isEmpty && marketStore.status.localizedCaseInsensitiveContains("unavailable") {error="Boxing schedule is unavailable. Pull to retry."}
   return
  }
  loading=true;defer{loading=false}
  let selectedSport=sport
  do {
   let formatter=DateFormatter();formatter.dateFormat="yyyyMMdd"
   let url=URL(string:"https://site.api.espn.com/apis/site/v2/sports/"+sportsFeedPath(sport)+"/scoreboard?limit=1000&dates="+formatter.string(from:date))!
   let(data,response)=try await URLSession.shared.data(from:url)
   guard !Task.isCancelled,selectedSport==sport else{return}
   guard (response as? HTTPURLResponse)?.statusCode==200,let root=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {throw URLError(.badServerResponse)}
   let sourceEvents=(root["events"] as? [[String:Any]] ?? []).flatMap { event -> [[String:Any]] in
    guard let groups=event["groupings"] as? [[String:Any]] else{return [event]}
    return groups.flatMap { group -> [[String:Any]] in
     (group["competitions"] as? [[String:Any]] ?? []).compactMap { competition in
      guard let start=(competition["date"] as? String).flatMap(sportsDate),Calendar.current.isDate(start,inSameDayAs:date) else{return nil}
      var match=event;match["id"]=competition["id"];match["status"]=competition["status"];match["date"]=competition["date"];match["competitions"]=[competition];return match
     }
    }
   }
   events=sourceEvents.compactMap {item in
    guard let id=item["id"] as? String,let name=item["name"] as? String else{return nil}
    let status=item["status"] as? [String:Any];let type=status?["type"] as? [String:Any]
    let statusText=type?["detail"] as? String ?? (item["date"] as? String).flatMap(sportsDate)?.formatted(date:.abbreviated,time:.shortened) ?? ""
    let participants=(item["competitions"] as? [[String:Any]] ?? []).flatMap { competition in
     (competition["competitors"] as? [[String:Any]] ?? []).compactMap { competitor -> String? in
      let athlete=competitor["athlete"] as? [String:Any] ?? competitor["team"] as? [String:Any] ?? [:]
      guard let name=athlete["displayName"] as? String else{return nil}
      let score=competitor["score"] as? String ?? ""
      let sets=(competitor["linescores"] as? [[String:Any]] ?? []).compactMap {($0["value"] as? NSNumber)?.stringValue}.joined(separator:"–")
      let record=(competitor["records"] as? [[String:Any]] ?? []).first?["summary"] as? String ?? ""
      let stats=(competitor["statistics"] as? [[String:Any]] ?? []).compactMap { stat -> String? in
       guard let label=stat["displayName"] as? String ?? stat["name"] as? String,let value=stat["displayValue"] as? String else{return nil};return label+" "+value
      }.joined(separator:" · ")
      return ([name]+[score,sets,record,stats].filter{!$0.isEmpty}).joined(separator:" · ")
     }
    }
    return Event(id:id,name:name,detail:([statusText]+participants).joined(separator:"\n"))
   }
  } catch {if !Task.isCancelled {self.error="Events are unavailable. Pull to retry."}}
 }
}

private func sportIcon(_ sport:String)->String {
 switch sport {
 case "All":return "square.grid.2x2.fill"
 case "NBA","WNBA","NCAAM","NCAAW":return "basketball.fill"
 case "MLB":return "baseball.fill"
 case "NFL","NCAAF":return "american.football.fill"
 case "NHL":return "hockey.puck.fill"
 case "Soccer":return "soccerball"
 case "Tennis":return "tennisball.fill"
 case "Golf":return "flag.fill"
 case "UFC/MMA","Boxing":return "figure.boxing"
 default:return "flag.checkered"
 }
}

private struct OtherSportOdds:View {
 let sport:String
 @EnvironmentObject private var account:AccountStore
 @StateObject private var store=Store()
 var body:some View {
  ScrollView {LazyVStack(alignment:.leading,spacing:14){
   if store.loading {PLONNLoader("Loading odds…")}
   if !store.loading && store.live.isEmpty {ContentUnavailableView("No prices currently offered",systemImage:"chart.line.uptrend.xyaxis",description:Text(store.status))}
   ForEach(store.live){market in
    NavigationLink{DetailView(m:market)}label:{VStack(alignment:.leading,spacing:8){HStack{Text(market.name).font(.system(.headline, design: .monospaced));Spacer();Text(odds(market.price)).bold().monospacedDigit()};Text(market.label).font(.system(.subheadline, design: .monospaced));Text(market.eventLabel ?? market.opponent).font(.system(.caption, design: .monospaced));Text(market.book).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}.foregroundStyle(.primary).padding().background(navySurface,in:RoundedRectangle(cornerRadius:14))}
   }
  }.padding()}.background(navy).navigationTitle("Sportsbook odds").task{await store.load(sport,account:account)}.plonnRefreshable{await store.load(sport,account:account)}
 }
}

@MainActor private final class AllGamesStore:ObservableObject {
 @Published var groups:[String:[SportsItem]]=[:]
 @Published var markets:[String:[Market]]=[:]
 @Published var loading=false
 @Published var failures:[String]=[]
 @Published var oddsIssues:[String:String]=[:]
 @Published var cachedLeagues:Set<String>=[]
 @Published var quoteTimes:[String:Date]=[:]
 let leagues=LaunchSports.supported
 private var requestVersion=0
 private var loadedDay:Date?
 private let source=Store()
 private var loadedOddsScope:OddsScope?
 func load(_ date:Date,account:AccountStore) async {
  guard !Task.isCancelled else{return}
  requestVersion += 1;let version=requestVersion
  let day=Calendar.current.startOfDay(for:date)
  let scope=source.scope(for:account)
  if loadedOddsScope != scope {markets=[:];oddsIssues=[:];cachedLeagues=[];quoteTimes=[:];loadedOddsScope=scope}
  if loadedDay != day {groups=[:];loadedDay=day}
  loading=true;defer{if version == requestVersion {loading=false}};failures=[]
  await withTaskGroup(of:(String,[SportsItem],String).self) { group in
   for league in leagues {
    group.addTask { @MainActor in
     let feed=SportsDirectory();await feed.load(sport:league,view:"Games",date:date)
     return (league,feed.items,feed.status)
    }
   }
   for await (league,items,status) in group {
    if Task.isCancelled || version != requestVersion{return}
    if status.localizedCaseInsensitiveContains("unavailable") {failures.append(league)}
    else {groups[league]=items.sorted { a,b in a.live != b.live ? a.live : a.start < b.start }}
   }
  }
  // A superseded/cancelled schedule may finish after a newer slate's odds.
  // Reject it before clearing any of the newer results or starting requests.
  guard !Task.isCancelled,version == requestVersion,source.scope(for:account)==scope else{return}
  // Read genuine recorded slate caches only. No refresh claim, provider call, or props fanout.
  markets=[:];oddsIssues=[:];cachedLeagues=[];quoteTimes=[:]
  for league in leagues where !(groups[league] ?? []).isEmpty {
   guard !Task.isCancelled,version == requestVersion else{return}
   await source.load(league,account:account,forceRefresh:false,cacheOnly:true,nbaSeasonTypes:(groups[league] ?? []).map(\.seasonType))
   guard !Task.isCancelled,version == requestVersion,source.scope(for:account)==scope else{return}
   markets[league]=source.live
   oddsIssues[league]=source.oddsIssue
   if source.oddsCachedOnly {cachedLeagues.insert(league)}
   quoteTimes[league]=source.lastOddsRefresh
  }
  guard version == requestVersion else{return}
  NotificationCenter.default.post(name:Notification.Name("PLONNInitialScheduleReady"),object:nil)
 }
}
private struct AllGamesSlate:View {
 let date:Date
 let query:String
 @StateObject private var slate=AllGamesStore()
 @EnvironmentObject private var account:AccountStore
 @Environment(\.scenePhase) private var scenePhase
 private func title(_ league:String)->String {
  switch league {case "Soccer:eng.1":return "Premier League";case "Soccer:uefa.champions":return "Champions League";case "Soccer:usa.1":return "MLS";case "Soccer:uefa.nations":return "Nations League";default:return league}
 }
 var body:some View {
  ScrollView {
   LazyVStack(alignment:.leading,spacing:14) {
    if slate.loading {PLONNLoader("Loading all games…")}
    ForEach(slate.leagues.sorted { a,b in
     let aLive=slate.groups[a]?.contains(where: \.live) ?? false
     let bLive=slate.groups[b]?.contains(where: \.live) ?? false
     return aLive != bLive ? aLive : slate.leagues.firstIndex(of:a)! < slate.leagues.firstIndex(of:b)!
    },id:\.self) { league in
     let games=(slate.groups[league] ?? []).filter {query.isEmpty || $0.name.localizedCaseInsensitiveContains(query) || $0.home.localizedCaseInsensitiveContains(query) || $0.away.localizedCaseInsensitiveContains(query)}
     if !games.isEmpty {
      HStack(spacing:10) { LeagueMark(league:league);Text(title(league)).font(.system(.headline, design: .monospaced)) }.padding(.top,8)
      if let issue=slate.oddsIssues[league] {OddsIssueNotice(message:issue)}
      if ["MLB","NBA","NFL","NHL"].contains(league),Calendar.current.isDateInToday(date),games.contains(where: \.live) {
       HotLiveOddsCarousel(sport:league,isToday:Calendar.current.isDateInToday(date),games:games)
      }
      ForEach(games) { game in
       NavigationLink {
        if ["NBA","MLB","NFL","NHL"].contains(league) {TodayGameDetail(game:game,sport:league)} else {SoccerMatchupView(game:game,sport:league)}
       } label: {
        if game.live || game.state == "post" {SharedLiveScoreCard(game:game,sport:league)} else {
        ScheduledOddsCard(game:game,sport:league,markets:slate.markets[league] ?? [],loading:slate.loading,cachedOnly:slate.cachedLeagues.contains(league),asOf:slate.quoteTimes[league])
        }
       }.buttonStyle(.plain)
      }
     }
    }
    if !slate.loading && slate.failures.isEmpty && slate.groups.values.allSatisfy({$0.isEmpty}) {Text("No games available for this date").padding(.vertical,32)}
    if !slate.failures.isEmpty {Text("Some leagues could not refresh. Pull to retry.").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)}
   }.padding(16)
  }.task(id:String(describing:date)+String(describing:scenePhase)+(account.user?.id.uuidString ?? "guest")) {
   guard scenePhase == .active else{return}
   repeat {
    await slate.load(date,account:account)
    do {try await Task.sleep(for:.seconds(30))} catch{return}
   } while !Task.isCancelled
  }.plonnRefreshable {await slate.load(date,account:account)}
 }
}

struct LeagueMark:View {
 let league:String
 private let urls:[String:String]=["NBA":"https://a.espncdn.com/i/teamlogos/leagues/500/nba.png","MLB":"https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png","NFL":"https://a.espncdn.com/i/teamlogos/leagues/500/nfl.png","NHL":"https://a.espncdn.com/i/teamlogos/leagues/500/nhl.png","WNBA":"https://a.espncdn.com/i/teamlogos/leagues/500/wnba.png","NCAAF":"https://a.espncdn.com/redesign/assets/img/icons/ESPN-icon-football-college.png","NCAAM":"https://a.espncdn.com/redesign/assets/img/icons/ESPN-icon-basketball.png","NCAAW":"https://a.espncdn.com/redesign/assets/img/icons/ESPN-icon-basketball.png","Soccer:eng.1":"https://a.espncdn.com/i/leaguelogos/soccer/500/23.png","Soccer:uefa.champions":"https://a.espncdn.com/i/leaguelogos/soccer/500/2.png","Soccer:usa.1":"https://a.espncdn.com/i/leaguelogos/soccer/500/19.png","Soccer:uefa.nations":"https://a.espncdn.com/i/leaguelogos/soccer/500/2395.png"]
 var body:some View {
 Group {if league == "NHL" {Image("league-nhl").resizable().scaledToFit()} else if let url=urls[league] {AsyncImage(url:URL(string:url)) {image in image.resizable().scaledToFit()} placeholder:{Image(systemName:sportIcon(league)).font(.system(.title3, design: .monospaced))}} else {Image(systemName:sportIcon(league)).font(.system(.title3, design: .monospaced))}}.frame(width:28,height:28).accessibilityHidden(true)
 }
}

#if DEBUG && targetEnvironment(simulator)
/// Exercises the real matchup view and generation state using an offline hosted-transport fixture.
struct MatchupOfflineFixtureView: View {
 @StateObject private var gameUpdates=LiveMatchupStore()
 var body: some View {
  NavigationStack {
   ScrollView {
    VStack(alignment:.leading,spacing:16) {
     Text("Offline fixture · No network or model request").font(.caption.bold()).accessibilityIdentifier("matchup-offline-fixture-banner")
     Text("Seattle Seahawks\nvs Denver Broncos").font(.title2.bold())
     MatchupAdvantagesView(game:SportsItem(id:"offline-sea-den",name:"Seattle Seahawks vs Denver Broncos",home:MatchupOfflineFixture.home,away:MatchupOfflineFixture.away,state:"pre",start:"2026-10-15T20:00:00Z"),sport:"NFL",logs:[:],markets:[],showInsights:true,gameUpdates:gameUpdates)
    }.padding()
   }.navigationTitle("PLONN · Offline verification").navigationBarTitleDisplayMode(.inline)
  }
 }
}
#endif
