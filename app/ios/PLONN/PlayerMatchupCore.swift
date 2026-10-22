import Foundation

/// A forecast is bound to a verified player, team, event and exact prop contract.
/// The backend re-verifies identity and owns training; the client never adjusts probabilities.
struct PlayerMatchupSetup: Codable, Equatable {
    static let mode = "upcoming_player_prop_v1"
    var league: String
    var eventID: String
    var home: String
    var away: String
    var startsAt: String
    var player: String
    var playerID: String
    var teamID: String
    var stat: String
    var line: Double
    var side: String
    var book: String
    var propOptions: [String: String]
    var evaluationMode = Self.mode
    var start: Date? { sportsDate(startsAt) }
    var game: UpcomingGameSetup { .init(league: league, eventID: eventID, home: home, away: away, startsAt: startsAt) }
    enum CodingKeys: String, CodingKey {
        case league, player, stat, line, side, book
        case eventID = "event_id", home = "home_team", away = "away_team", startsAt = "starts_at"
        case playerID = "player_id", teamID = "team_id", propOptions = "prop_options", evaluationMode = "evaluation_mode"
    }
    static func identifier(_ value: String) -> Bool {
        (1...15).contains(value.count) && value.allSatisfy { $0.isASCII && $0.isNumber } && UInt64(value).map { $0 > 0 } == true
    }
    static func clean(_ value: String, max: Int = 120) -> Bool {
        (1...max).contains(value.count) && value == value.trimmingCharacters(in: .whitespacesAndNewlines)
        && value.unicodeScalars.allSatisfy { !CharacterSet.controlCharacters.contains($0) }
    }
    func payload() throws -> [String: Any] {
        guard evaluationMode == Self.mode, LabPropCatalog.leagues.contains(league),
              [eventID, playerID, teamID].allSatisfy(Self.identifier),
              [home, away, player, book].allSatisfy({ Self.clean($0) }), home != away, start != nil,
              startsAt.count <= 40, ["Over", "Under"].contains(side), stat != "Custom stat",
              LabPropCatalog.options(league: league, target: "Player").contains(stat),
              line.isFinite, (-1000...10000).contains(line), abs(line * 100 - (line * 100).rounded()) < 1e-6,
              !LabPropCatalog.isYesNo(stat) || line == 0.5,
              LabPropCatalog.validOptions(stat: stat, options: propOptions)
        else { throw LabServiceContract.ContractError.invalidSetup }
        return ["evaluation_mode": evaluationMode, "league": league, "event_id": eventID,
                "home_team": home, "away_team": away, "starts_at": startsAt, "player": player,
                "player_id": playerID, "team_id": teamID, "stat": stat, "line": line,
                "side": side, "book": book, "prop_options": propOptions]
    }
    func forecast(_ body: [String: Any], now: Date = Date()) throws -> PlayerMatchupForecast {
        _ = try payload()
        guard JSONSerialization.isValidJSONObject(body),
              let bytes = try? JSONSerialization.data(withJSONObject: body), bytes.count <= 2_000_000,
              let result = try? JSONDecoder().decode(PlayerMatchupForecast.self, from: bytes),
              result.status == "ok", result.evaluationMode == Self.mode, result.upcomingPrediction,
              result.setup == self, let generated = result.generatedAt, let start,
              generated <= now.addingTimeInterval(60), generated < start,
              result.availability.isValid(for: game, generated: generated)
        else { throw LabServiceContract.ContractError.mismatchedResult }
        let target = result.inference.target, projection = result.inference.projection, validation = result.validation
        let expectedTeam = target.isHome ? home : away, expectedOpponent = target.isHome ? away : home
        guard target.eventID == eventID, target.league == league, sportsDate(target.startsAt) == start,
              target.playerID == playerID, target.playerName == player, target.teamID == teamID,
              target.team == expectedTeam, target.opponent == expectedOpponent,
              Self.identifier(target.opponentID), target.opponentID != teamID,
              result.inference.factors.count <= 40, result.inference.missingInputs.count <= 40,
              result.inference.missingInputs.allSatisfy({ Self.clean($0, max: 500) }),
              result.inference.factors.allSatisfy({ $0.isValid }),
              ["passed", "limited"].contains(validation.status),
              (0...10000).contains(validation.trainingCount), (0...10000).contains(validation.testCount),
              validation.reasons.count <= 40, validation.reasons.allSatisfy({ Self.clean($0, max: 500) }),
              [validation.brier, validation.baselineBrier].allSatisfy({ $0.map { $0.isFinite && (0...2).contains($0) } ?? true }),
              [validation.calibrationError, validation.intervalCoverage].allSatisfy({ $0.map { $0.isFinite && (0...1).contains($0) } ?? true }),
              result.coverage.isValid, projection.intervalLevel.isFinite,
              abs(projection.intervalLevel - 0.8) < 1e-8,
              ["empirical", "poisson", "negative_binomial", "binomial", "beta_binomial", "bernoulli", "unavailable"].contains(projection.distribution),
              ["none", "head_to_head", "opponent_defense"].contains(projection.adjustmentKind ?? "none")
        else { throw LabServiceContract.ContractError.mismatchedResult }
        if projection.available {
            guard validation.status == "passed", validation.trainingCount >= 20, validation.testCount >= 10,
                  !validation.gates.isEmpty, validation.gates.values.allSatisfy({ $0 }),
                  let mean = projection.mean, mean.isFinite, let lower = projection.lower, lower.isFinite,
                  let upper = projection.upper, upper.isFinite, lower <= upper,
                  let probabilities = projection.probabilities, probabilities.isValid,
                  let selected = projection.selectedProbability, selected.isFinite,
                  abs(selected - (side == "Over" ? probabilities.over : probabilities.under)) < 1e-8,
                  projection.standardDeviation.map({ $0.isFinite && $0 >= 0 }) ?? true,
                  projection.opponentAdjustment.map({ $0.isFinite }) ?? true,
                  validation.brier != nil, validation.baselineBrier != nil
            else { throw LabServiceContract.ContractError.mismatchedResult }
            // Binary outcomes and all nonnegative stats must retain their support.
            if LabPropCatalog.isYesNo(stat) {
                guard (0...1).contains(mean), (0...1).contains(lower), (0...1).contains(upper) else { throw LabServiceContract.ContractError.mismatchedResult }
            } else if !Self.signed(stat) {
                guard mean >= 0, lower >= 0 else { throw LabServiceContract.ContractError.mismatchedResult }
            }
        } else {
            guard projection.mean == nil, projection.lower == nil, projection.upper == nil,
                  projection.probabilities == nil, projection.selectedProbability == nil,
                  projection.opponentAdjustment == nil, projection.standardDeviation == nil,
                  validation.status == "limited", !validation.reasons.isEmpty
            else { throw LabServiceContract.ContractError.mismatchedResult }
        }
        return result
    }
    private static func signed(_ stat: String) -> Bool {
        stat.contains("Yds") || stat.hasPrefix("Longest") || stat == "1st-quarter passing yards" || LabPropCatalog.isFantasy(stat)
    }
    static func failure(_ body: [String: Any]) -> String {
        switch body["reason"] as? String {
        case "upcoming_player_roster_mismatch", "player_team_not_in_upcoming_game", "upcoming_roster_team_mismatch":
            return "This player’s current team couldn’t be matched to the game. Refresh the player list."
        case "upcoming_player_history_unavailable", "upcoming_player_data_unavailable", "upcoming_roster_unavailable":
            return "Player or matchup data couldn’t be verified. Try again shortly."
        case "nfl_source_schema_mismatch", "nba_source_schema_mismatch", "nhl_source_schema_mismatch", "mlb_source_schema_mismatch":
            return "We couldn’t read this player’s game history. Try again shortly."
        case "insufficient_player_history", "incomplete_player_event_history":
            return "There isn’t enough complete history to test this prop reliably yet. Try another prop or player."
        case "unsupported_player_prop": return "This prop isn’t available for this sport. Choose another stat."
        case "invalid_upcoming_player_setup", "invalid_upcoming_player_selection", "invalid_threshold":
            return "Check the player, prop and line, then try again."
        default: return UpcomingGameSetup.failure(body)
        }
    }
    static func stat(sport: String, marketKey: String) -> String? {
        let key = marketKey.hasSuffix("_alternate") ? String(marketKey.dropLast(10)) : marketKey
        let mapping: [String: String]
        switch sport {
        case "NBA": mapping = ["player_points":"Points", "player_rebounds":"Rebounds", "player_assists":"Assists", "player_steals":"Steals", "player_blocks":"Blocks", "player_threes":"3-pointers", "player_turnovers":"Turnovers", "player_points_rebounds_assists":"Pts + Reb + Ast", "player_points_rebounds":"Pts + Reb", "player_points_assists":"Pts + Ast", "player_rebounds_assists":"Reb + Ast", "player_blocks_steals":"Blocks + Steals", "player_double_double":"Double-double", "player_triple_double":"Triple-double", "player_field_goals":"Field goals made", "player_frees_made":"Free throws made", "player_frees_attempts":"Free throw attempts"]
        case "NFL": mapping = ["player_pass_yds":"Pass Yds", "player_pass_tds":"Pass TDs", "player_pass_attempts":"Pass attempts", "player_pass_completions":"Completions", "player_pass_interceptions":"Interceptions thrown", "player_rush_attempts":"Rush attempts", "player_rush_yds":"Rush Yds", "player_reception_yds":"Rec Yds", "player_receptions":"Receptions", "player_rush_reception_yds":"Rush + Rec Yds", "player_pass_rush_yds":"Pass + Rush Yds", "player_pass_rush_reception_yds":"Pass + Rush + Rec Yds", "player_rush_tds":"Rushing touchdowns", "player_reception_tds":"Receiving touchdowns", "player_anytime_td":"Anytime touchdown scorer", "player_tds_over":"Touchdowns", "player_sacks":"Sacks", "player_tackles_assists":"Tackles + Assists", "player_field_goals":"Field goals made", "player_kicking_points":"Kicking points"]
        case "MLB": mapping = ["batter_hits":"Hits", "batter_rbis":"RBI", "batter_stolen_bases":"Stolen bases", "batter_home_runs":"Home runs", "batter_total_bases":"Total bases", "batter_runs_scored":"Runs", "pitcher_strikeouts":"Pitcher strikeouts", "pitcher_outs":"Pitcher outs", "pitcher_earned_runs":"Earned runs", "pitcher_hits_allowed":"Hits allowed", "pitcher_walks":"Walks allowed", "batter_walks":"Walks", "batter_strikeouts":"Batter strikeouts", "batter_singles":"Singles", "batter_doubles":"Doubles", "batter_triples":"Triples", "batter_hits_runs_rbis":"Hits + Runs + RBI"]
        case "NHL": mapping = ["player_goals":"Goals", "player_assists":"Assists", "player_points":"Points", "player_shots_on_goal":"Shots on goal", "player_power_play_points":"Power play points", "player_total_saves":"Saves", "player_blocked_shots":"Blocked shots", "player_goal_scorer_anytime":"Anytime goal scorer"]
        default: return nil
        }
        return mapping[key]
    }
}

struct PlayerMatchupForecast: Decodable {
    struct Probabilities: Decodable {
        let over: Double, under: Double, push: Double
        var isValid: Bool { [over, under, push].allSatisfy { $0.isFinite && (0...1).contains($0) } && abs(over + under + push - 1) < 1e-8 }
    }
    struct Projection: Decodable {
        let available: Bool, distribution: String
        let mean: Double?, lower: Double?, upper: Double?, intervalLevel: Double
        let probabilities: Probabilities?, selectedProbability: Double?, opponentAdjustment: Double?, standardDeviation: Double?
        let adjustmentKind: String?
        enum CodingKeys: String, CodingKey {
            case available, distribution, mean, lower, upper, probabilities
            case intervalLevel = "interval_level", selectedProbability = "selected_probability", opponentAdjustment = "opponent_adjustment", standardDeviation = "standard_deviation"
            case adjustmentKind = "adjustment_kind"
        }
    }
    struct Factor: Decodable, Identifiable {
        let kind: String, title: String, detail: String, direction: String, sampleSize: Int
        var id: String { kind + "|" + title + "|" + detail }
        var isValid: Bool { PlayerMatchupSetup.clean(kind, max: 80) && PlayerMatchupSetup.clean(title, max: 160) && PlayerMatchupSetup.clean(detail, max: 2000) && ["supports", "opposes", "neutral"].contains(direction) && (0...10000).contains(sampleSize) }
        enum CodingKeys: String, CodingKey { case kind, title, detail, direction; case sampleSize = "sample_size" }
    }
    struct Target: Decodable {
        let eventID: String, date: String, startsAt: String, playerID: String, playerName: String
        let teamID: String, team: String, opponentID: String, opponent: String, league: String, isHome: Bool
        enum CodingKeys: String, CodingKey {
            case date, team, opponent, league
            case eventID = "event_id", startsAt = "starts_at", playerID = "player_id", playerName = "player_name"
            case teamID = "team_id", opponentID = "opponent_id", isHome = "is_home"
        }
    }
    struct Inference: Decodable {
        let target: Target, projection: Projection, factors: [Factor], missingInputs: [String]
        enum CodingKeys: String, CodingKey { case target, projection, factors; case missingInputs = "missing_inputs" }
    }
    struct Validation: Decodable {
        let status: String, trainingCount: Int, testCount: Int
        let brier: Double?, baselineBrier: Double?, calibrationError: Double?, intervalCoverage: Double?
        let gates: [String: Bool], reasons: [String]
        enum CodingKeys: String, CodingKey {
            case status, brier, gates, reasons
            case trainingCount = "training_count", testCount = "test_count", baselineBrier = "baseline_brier", calibrationError = "calibration_error", intervalCoverage = "interval_coverage"
        }
    }
    struct Coverage: Decodable {
        let usableRows: Int, excludedRows: Int, opponentGames: Int, workloadGames: Int
        var isValid: Bool { [usableRows, excludedRows, opponentGames, workloadGames].allSatisfy { (0...10000).contains($0) } }
        enum CodingKeys: String, CodingKey { case usableRows = "usable_rows", excludedRows = "excluded_rows", opponentGames = "opponent_games", workloadGames = "workload_games" }
    }
    let status: String, evaluationMode: String, upcomingPrediction: Bool, setup: PlayerMatchupSetup
    let generatedAtText: String, inference: Inference, validation: Validation, coverage: Coverage, availability: ModelAvailability
    var generatedAt: Date? { sportsDate(generatedAtText) }
    enum CodingKeys: String, CodingKey {
        case status, setup, inference, validation, coverage, availability
        case evaluationMode = "evaluation_mode", upcomingPrediction = "upcoming_prediction", generatedAtText = "generated_at"
    }
}

struct SavedPlayerMatchupModel: Decodable, Identifiable {
    let id: String, setup: PlayerMatchupSetup, generated_at: String
}
