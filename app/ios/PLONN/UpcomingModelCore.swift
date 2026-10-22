import Foundation

struct UpcomingGameSetup: Codable, Equatable, Hashable, Identifiable {
    static let mode = "upcoming_game_v2"
    static let supportedModes: Set<String> = [mode, "upcoming_game_v1"]
    let league: String
    let eventID: String
    let home: String
    let away: String
    let startsAt: String
    var evaluationMode = Self.mode
    var id: String { league + ":" + eventID }
    var start: Date? { sportsDate(startsAt) }
    var title: String { away + " at " + home }
    enum CodingKeys: String, CodingKey {
        case league, eventID = "event_id", home = "home_team", away = "away_team"
        case startsAt = "starts_at", evaluationMode = "evaluation_mode"
    }
    func payload() throws -> [String: Any] {
        guard Self.supportedModes.contains(evaluationMode), LabPropCatalog.leagues.contains(league),
              (1...15).contains(eventID.count), eventID.allSatisfy({ $0.isASCII && $0.isNumber }),
              (1...120).contains(home.count), (1...120).contains(away.count), home != away, start != nil
        else { throw LabServiceContract.ContractError.invalidSetup }
        return ["evaluation_mode": evaluationMode, "league": league, "event_id": eventID,
                "home_team": home, "away_team": away, "starts_at": startsAt]
    }
    static func scheduledGames(_ data: Data, league: String, now: Date = Date()) throws -> [Self] {
        guard LabPropCatalog.leagues.contains(league),
              let body = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let events = body["events"] as? [[String: Any]], events.count <= 1000 else { throw URLError(.cannotParseResponse) }
        var games: [Self] = [], seen = Set<String>()
        for event in events {
            guard let id = event["id"] as? String, seen.insert(id).inserted,
                  let competitions = event["competitions"] as? [[String: Any]], competitions.count == 1,
                  let game = competitions.first, game["id"] as? String == id,
                  let status = game["status"] as? [String: Any], let type = status["type"] as? [String: Any],
                  type["state"] as? String == "pre", type["name"] as? String == "STATUS_SCHEDULED",
                  type["completed"] as? Bool == false,
                  let season = event["season"] as? [String: Any], let seasonType = season["type"] as? Int, [2, 3].contains(seasonType),
                  let startText = game["date"] as? String, let start = sportsDate(startText),
                  start > now, start <= now.addingTimeInterval(31 * 86400),
                  let teams = game["competitors"] as? [[String: Any]], teams.count == 2,
                  let home = teams.first(where: { $0["homeAway"] as? String == "home" })?["team"] as? [String: Any],
                  let away = teams.first(where: { $0["homeAway"] as? String == "away" })?["team"] as? [String: Any],
                  let homeName = home["displayName"] as? String, let awayName = away["displayName"] as? String else { continue }
            let setup = Self(league: league, eventID: id, home: homeName, away: awayName, startsAt: startText)
            guard (try? setup.payload()) != nil else { continue }
            games.append(setup)
        }
        return games.sorted { ($0.start ?? .distantFuture) < ($1.start ?? .distantFuture) }
    }
    struct Forecast {
        let win: Double, accuracy: Double, baselineAccuracy: Double, brier: Double, baselineBrier: Double
        let trained: Int, tested: Int
        let generated: Date
        let postseason: Bool
        let availability: ModelAvailability?
        let strengths: [ModelTeamStrength]
        let ratingBrier: Double?, ratingAccuracy: Double?
    }
    func forecast(_ body: [String: Any], now: Date = Date()) throws -> Forecast {
        guard body["status"] as? String == "ok", body["evaluation_mode"] as? String == evaluationMode,
              body["upcoming_prediction"] as? Bool == true, body["forecast_kind"] as? String == "home_win",
              let requested = body["setup"] as? [String: Any],
              let encoded = try? JSONSerialization.data(withJSONObject: requested),
              let decoded = try? JSONDecoder().decode(Self.self, from: encoded), decoded == self,
              let generatedText = body["generated_at"] as? String, let generated = sportsDate(generatedText),
              let start, generated < start, generated <= now.addingTimeInterval(60),
              let inference = body["inference"] as? [String: Any], inference["status"] as? String == "ok",
              let target = inference["target"] as? [String: Any], target["event_id"] as? String == eventID,
              target["league"] as? String == league, target["home_team_name"] as? String == home,
              target["away_team_name"] as? String == away, target["selected_team"] as? String == "home",
              let seasonType = target["season_type"] as? Int, [2, 3].contains(seasonType),
              (target["starts_at"] as? String).flatMap(sportsDate) == start,
              let p = inference["probabilities"] as? [String: Double], Set(p.keys) == ["win", "not_win"],
              p.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }), abs(p.values.reduce(0,+) - 1) < 1e-8,
              let training = body["training"] as? [String: Any], training["status"] as? String == "ok",
              let trained = training["sample_size"] as? Int, trained >= 30,
              let validation = body["validation"] as? [String: Any], validation["status"] as? String == "ok",
              validation["target_observed_outcome"] == nil,
              let tested = validation["sample_size"] as? Int, tested >= 10,
              let metrics = validation["metrics"] as? [String: Any],
              let accuracy = metrics["accuracy"] as? Double, accuracy.isFinite, (0...1).contains(accuracy),
              let brier = metrics["multiclass_brier"] as? Double, brier.isFinite, (0...2).contains(brier),
              let baseline = metrics["baseline"] as? [String: Any],
              let baselineAccuracy = baseline["accuracy"] as? Double, baselineAccuracy.isFinite, (0...1).contains(baselineAccuracy),
              let baselineBrier = baseline["multiclass_brier"] as? Double, baselineBrier.isFinite, (0...2).contains(baselineBrier)
        else { throw LabServiceContract.ContractError.mismatchedResult }
        var availability: ModelAvailability?, strengths: [ModelTeamStrength] = []
        var ratingBrier: Double?, ratingAccuracy: Double?
        if evaluationMode == Self.mode {
            guard let context = body["availability"] as? [String: Any],
                  let bytes = try? JSONSerialization.data(withJSONObject: context),
                  let decoded = try? JSONDecoder().decode(ModelAvailability.self, from: bytes),
                  decoded.isValid(for: self, generated: generated),
                  let strength = body["opponent_strength"] as? [String: Any], Set(strength.keys) == ["home", "away"],
                  let strengthData = try? JSONSerialization.data(withJSONObject: strength),
                  let pair = try? JSONDecoder().decode([String: ModelTeamStrength].self, from: strengthData),
                  let h = pair["home"], let a = pair["away"], h.team == home, a.team == away,
                  [h,a].allSatisfy({ $0.rating.isFinite && (0...4000).contains($0.rating) && $0.games >= 5 }),
                  let rating = metrics["rating_baseline"] as? [String: Any],
                  let rb = rating["multiclass_brier"] as? Double, rb.isFinite, (0...2).contains(rb),
                  let ra = rating["accuracy"] as? Double, ra.isFinite, (0...1).contains(ra)
            else { throw LabServiceContract.ContractError.mismatchedResult }
            availability = decoded; strengths = [h,a]; ratingBrier = rb; ratingAccuracy = ra
        }
        return Forecast(win: p["win"]!, accuracy: accuracy, baselineAccuracy: baselineAccuracy, brier: brier,
                        baselineBrier: baselineBrier, trained: trained, tested: tested, generated: generated,
                        postseason: seasonType == 3, availability: availability, strengths: strengths, ratingBrier: ratingBrier, ratingAccuracy: ratingAccuracy)
    }
    static func failure(_ body: [String: Any]) -> String {
        let reason = body["reason"] as? String ?? (body["training"] as? [String: Any])?["reason"] as? String ?? ""
        switch reason {
        case "game_no_longer_upcoming": return "This game has started or its schedule changed. Choose another upcoming matchup."
        case "upcoming_game_changed": return "The matchup changed. Refresh the game list before building your model."
        case "upcoming_season_unsupported": return "Preseason and exhibition games aren’t supported by this model."
        case "insufficient_league_history", "insufficient_opponent_history", "stale_opponent_history", "insufficient_history_minimum_40_feature_complete_games", "insufficient_game_training_or_holdout_rows", "insufficient_prior_games":
            return "There isn’t enough verified history to build and test this model yet. Try another matchup."
        case "pro_required": return "PLONN Pro is required to build a model."
        case "upcoming_source_unavailable", "upcoming_history_truncated", "upcoming_data_unavailable", "upcoming_game_incomplete", "game_public_source_unavailable":
            return "Game data couldn’t be verified right now. Try again shortly."
        default: return LabServiceContract.userFacingMessage(body)
        }
    }
}

struct SavedUpcomingModel: Decodable, Identifiable {
    let id: String
    let setup: UpcomingGameSetup
    let generated_at: String
}

struct ModelTeamStrength: Decodable, Identifiable {
    let team: String
    let rating: Double
    let games: Int
    let last_game: String
    var id: String { team }
}

struct ModelAvailability: Decodable {
    struct Player: Decodable, Identifiable {
        let name: String, status: String, updated_at: String, scope: String, player_id: String
        var id: String { player_id }
    }
    struct Injuries: Decodable, Identifiable {
        let team: String, state: String
        let players: [Player]
        let excluded: Int
        var id: String { team }
    }
    struct Starter: Decodable {
        let kind: String, state: String
        let name: String?
    }
    struct Lineup: Decodable, Identifiable {
        let team: String, state: String, detail: String
        let players: [String]
        let starter: Starter?
        var id: String { team }
    }
    let version: Int, event_id: String, checked_at: String, expires_at: String, state: String
    let probability_adjusted: Bool, historical_training_available: Bool
    let injuries: [Injuries], lineups: [Lineup]
    var checked: Date? { sportsDate(checked_at) }
    var expires: Date? { sportsDate(expires_at) }
    func isValid(for setup: UpcomingGameSetup, generated: Date) -> Bool {
        let teams: Set<String> = [setup.home, setup.away]
        guard version == 1, event_id == setup.eventID, state == "limited",
              !probability_adjusted, !historical_training_available,
              let checked, let expires, let start = setup.start,
              checked <= generated, generated.timeIntervalSince(checked) <= 60,
              expires > checked, expires <= checked.addingTimeInterval(1800), expires <= start,
              injuries.count == 2, lineups.count == 2,
              Set(injuries.map(\.team)) == teams, Set(lineups.map(\.team)) == teams else { return false }
        for team in injuries {
            guard ["partial", "stale", "missing"].contains(team.state), team.excluded >= 0,
                  team.players.count <= 48, Set(team.players.map(\.id)).count == team.players.count,
                  team.state == "partial" || team.players.isEmpty else { return false }
            for player in team.players {
                guard (1...100).contains(player.name.count), (1...80).contains(player.status.count),
                      player.scope == "general_report", let updated = sportsDate(player.updated_at),
                      updated <= checked else { return false }
            }
        }
        for lineup in lineups {
            guard ["unknown", "partial", "confirmed"].contains(lineup.state), lineup.players.count <= 30,
                  Set(lineup.players).count == lineup.players.count,
                  lineup.players.allSatisfy({ (1...100).contains($0.count) }) else { return false }
            if let starter = lineup.starter {
                guard ["starting_pitcher", "starting_goalie"].contains(starter.kind),
                      ["unknown", "probable", "confirmed"].contains(starter.state),
                      starter.state == "unknown" ? starter.name == nil : (1...100).contains(starter.name?.count ?? 0) else { return false }
            }
        }
        return true
    }
}
