import pandas as pd
import os
import re
import json
import numpy as np

def generate_detailed_tactical_summary(df, match_id, team_name):
    summary = {
        "match_id": match_id.replace("match_", ""),
        "team": team_name,
        "formation": None,
        "key_tactical_patterns": [],
        "detailed_tactical_analysis": {},
        "summary": ""
    }
    # === Tactical Lineup Parsing ===
    if 'tactics_lineup' in df.columns and not df['tactics_lineup'].dropna().empty:
        try:
            import ast
            raw_lineup = df['tactics_lineup'].dropna().iloc[0]
            parsed_lineup = ast.literal_eval(raw_lineup) if isinstance(raw_lineup, str) else raw_lineup

            # Mapping from position.id to standard abbreviation
            position_map = {
                1: 'GK', 2: 'RB', 3: 'RCB', 4: 'CB', 5: 'LCB', 6: 'LB', 7: 'RWB', 8: 'LWB',
                9: 'RDM', 10: 'CDM', 11: 'LDM', 12: 'RM', 13: 'RCM', 14: 'CM', 15: 'LCM',
                16: 'LM', 17: 'RW', 18: 'RAM', 19: 'CAM', 20: 'LAM', 21: 'LW',
                22: 'RCF', 23: 'ST', 24: 'LCF', 25: 'SS'
            }

            lineup_list = []
            for player in parsed_lineup:
                jersey = player.get("jersey_number")
                name = player.get("player.name")
                pos_id = player.get("position.id")
                pos_abbr = position_map.get(pos_id, f"Pos-{pos_id}")

                lineup_list.append({
                    "jersey": jersey,
                    "name": name,
                    "position": pos_abbr
                })

            # Optional: Sort by position ID or jersey number
            lineup_list = sorted(lineup_list, key=lambda x: x["jersey"])

            summary["lineup"] = lineup_list

        except Exception as e:
            summary["lineup"] = []
            print(f"⚠️ Failed to parse lineup for {team_name}: {e}")

        position_role_map = {
        "Goalkeeper": [1],
        "Defender": [2, 3, 4, 5, 6, 7, 8],
        "Midfielder": [9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20],
        "Forward": [17, 21, 22, 23, 24, 25]
    }
    lineup_list = ast.literal_eval(df['tactics_lineup'].dropna().iloc[0])  # safe parse

    # Separate player IDs by role
    goalkeeper_ids = [p['player.id'] for p in lineup_list if p['position.id'] in position_role_map['Goalkeeper']]
    defender_ids   = [p['player.id'] for p in lineup_list if p['position.id'] in position_role_map['Defender']]
    midfielder_ids = [p['player.id'] for p in lineup_list if p['position.id'] in position_role_map['Midfielder']]
    forward_ids    = [p['player.id'] for p in lineup_list if p['position.id'] in position_role_map['Forward']]
    patterns = []

    # === ZONE LABELING CLEAN & SAFE ===
    def get_best_x(row):
        if pd.notnull(row.get("end_x")):
            return row["end_x"]
        elif pd.notnull(row.get("carry_end_x")):
            return row["carry_end_x"]
        elif pd.notnull(row.get("x")):
            return row["x"]
        else:
            return np.nan

    def get_best_y(row):
        if pd.notnull(row.get("end_y")):
            return row["end_y"]
        elif pd.notnull(row.get("carry_end_y")):
            return row["carry_end_y"]
        elif pd.notnull(row.get("y")):
            return row["y"]
        else:
            return np.nan
    df["zone_x"] = df.apply(get_best_x, axis=1)
    df["zone_y"] = df.apply(get_best_y, axis=1)

# Defensive / Middle / Final third (based on end_x)
    df["pitch_third"] = pd.cut(
        df["zone_x"],
        bins=[-1, 40, 80, 120],
        labels=["Defensive Third", "Middle Third", "Final Third"]
    )

    df["pitch_side"] = pd.cut(
        df["zone_y"],
        bins=[-1, 22.67, 45.33, 80],
        labels=["Left", "Center", "Right"]
    )

    df["zone_label"] = df["pitch_third"].astype(str) + " - " + df["pitch_side"].astype(str)
    # Combine both into zone label
    # ✅ Final Third Entry Channels using zone_x and zone_y
    df['final_third_entry_side'] = np.select(
        [
            (df['zone_x'] >= 92.5) & (df['zone_y'] < 25.67),
            (df['zone_x'] >= 92.5) & (df['zone_y'] >= 25.67) & (df['zone_y'] < 53.33),
            (df['zone_x'] >= 92.5) & (df['zone_y'] >= 53.33)
        ],
        ['Left Channel', 'Center Channel', 'Right Channel'],
        default='Not Final Third'
    )

    # ✅ Zone 14 (Danger Zone) using zone_x and zone_y
    df['is_zone_14'] = (
        (df['zone_x'] >= 80) & (df['zone_x'] <= 98.54) &
        (df['zone_y'] >= 26.66) & (df['zone_y'] <= 53.32)
    )

    # === Helper Functions ===
    def safe_count(cond):
        """Safely count the number of True values."""
        return int(cond.sum()) if cond is not None else 0

    def calculate_percentage(part, whole):
        """Safely calculate percentage."""
        return round((part / whole) * 100, 2) if whole > 0 else 0

    # === 1. Formation Extraction ===
    if 'tactics_formation' in df.columns:
        formations = df['tactics_formation'].dropna().unique()
        if len(formations) > 0:
            summary["formation"] = str(int(float(formations[0])))

    # 2. Possession and Buildup Analysis
        # === 2. Possession and Buildup Analysis ===
        possession_analysis = {}

        required_cols = ['possession', 'pass_pass_cluster_label', 'pass_type_name']
        if all(col in df.columns for col in required_cols):
            total_possessions = df['possession'].nunique()

    # --- Pass Cluster Breakdown ---
    if 'pass_pass_cluster_label' in df.columns:
        pass_clusters = df['pass_pass_cluster_label'].dropna().value_counts()
        top_clusters = pass_clusters.head(3)
        possession_analysis['pass_clusters'] = top_clusters.to_dict()

    # --- Structured Pass Type Distribution ---
    pass_type_counts = df['pass_type_name'].dropna().value_counts()
    pass_types_grouped = {
        "Recovery": pass_type_counts.get("Recovery", 0),
        "Throw-in": pass_type_counts.get("Throw-in", 0),
        "Corner": pass_type_counts.get("Corner", 0),
        "Free Kick": pass_type_counts.get("Free Kick", 0),
        "Goal Kick": pass_type_counts.get("Goal Kick", 0)
    }

    # --- Cluster-Based Keywords: Long, Short, Defensive ---
    if 'pass_pass_cluster_label' in df.columns:
        cluster_labels = df['pass_pass_cluster_label'].dropna()
        pass_types_grouped.update({
            "Long": cluster_labels.str.contains("Long", case=False).sum(),
            "Short": cluster_labels.str.contains("Short", case=False).sum(),
            "Defensive": cluster_labels.str.contains("Defensive", case=False).sum(),
            "Center": cluster_labels.str.contains("Center", case=False).sum(),
            "Attacking": cluster_labels.str.contains("Attacking", case=False).sum(),


        })

    # Attach final result
    possession_analysis["pass_types"] = pass_types_grouped

    # Get all successful passes
# --- 1. Base pass filter ---
    pass_df = df[df['type_name'] == 'Pass']

    # Only successful passes
    acc_passes = pass_df[pass_df['pass_outcome_name'] == '1']

    # Incomplete passes
    inacc_passes = pass_df[pass_df['pass_outcome_name'] != '1']

    # Progressive passes (e.g., > 9.25 meters)
    prog_passes = pass_df[pass_df['prog_pass'] >= 9.25]

    # Progressive accurate
    prog_acc = prog_passes[prog_passes['pass_outcome_name'] == '1']

    # Progressive inaccurate
    prog_inacc = prog_passes[prog_passes['pass_outcome_name'] != '1']
    # === Passing Outcome Summary ===
    possession_analysis["pass_accuracy_breakdown"] = {
        "total_passes": len(pass_df),
        "accurate_passes": len(acc_passes),
        "inaccurate_passes": len(inacc_passes),
        "progressive_passes": len(prog_passes),
        "progressive_accurate": len(prog_acc),
        "progressive_inaccurate": len(prog_inacc)
    }

    # === 2. % of possessions with at least one progressive pass ===
    if 'possession' in prog_passes.columns:
        prog_possession_ids = prog_passes['possession'].dropna().unique()
        total_possessions = df['possession'].dropna().nunique()
        progressive_possession_rate = round((len(prog_possession_ids) / total_possessions) * 100, 2) if total_possessions > 0 else 0
    else:
        progressive_possession_rate = 0

    # === 3. Save to possession_analysis ===
    possession_analysis["progressive_pass_possession_percentage"] = progressive_possession_rate
    possession_analysis["progressive_passes_total"] = int(len(prog_passes))
    possession_analysis["progressive_passes_accurate"] = int(len(prog_acc))
    possession_analysis["progressive_passes_inaccurate"] = int(len(prog_inacc))



    # 3. Attack Patterns
       # === 3. Attack Patterns & Final Third Presence ===
    attack_analysis = {}

    attack_cols = ['pitch_third', 'zone_label', 'box_entry', 'is_goal_assist', 'is_shot_assist']
    if all(col in df.columns for col in attack_cols):

        # --- Final Third Zone Usage ---
        final_third_df = df[df['pitch_third'] == 'Final Third']
        if not final_third_df.empty:
            top_attack_zones = final_third_df['zone_label'].dropna().value_counts().head(3)
            attack_analysis['preferred_attack_zones'] = top_attack_zones.to_dict()

        # --- Chance Creation & Box Entry ---
        attack_analysis.update({
            "box_entries": safe_count(df['box_entry']),
            "goal_assists": safe_count(df['is_goal_assist']),
            "shot_assists": safe_count(df['is_shot_assist'])
        })


    # 4. Defensive Patterns
        # === 4. Defensive Actions & Pressure Metrics ===
    defensive_analysis = {}

    core_defense_cols = [
        'duel_outcome_is_won',
        'interception_outcome_is_success_in_play',
        'clearance_aerial_won',
        'ball_recovery_is_offensive',
        'duel_type_is_tackle',
        '50_50_outcome_is_won'
    ]

    # --- Core Defensive Events ---
    if all(col in df.columns for col in core_defense_cols):
        defensive_analysis.update({
            "successful_duels": safe_count(df['duel_outcome_is_won']),
            "successful_interceptions": safe_count(df['interception_outcome_is_success_in_play']),
            "aerial_clearances": safe_count(df['clearance_aerial_won']),
            "offensive_recoveries": safe_count(df['ball_recovery_is_offensive']),
            "successful_tackles": safe_count(df['duel_type_is_tackle']),
            "50_50_wins": safe_count(df['50_50_outcome_is_won'])
        })

    # --- Defensive Blocks ---
    if 'block_offensive' in df.columns:
        defensive_analysis["offensive_blocks"] = int(df['block_offensive'].sum())

    if 'block_save_block' in df.columns:
        defensive_analysis["goal_saving_blocks"] = int(df['block_save_block'].sum())

    # --- Defensive Headers on Shots ---
    if 'shot_aerial_won' in df.columns:
        defensive_analysis["headers_on_shots"] = int(df['shot_aerial_won'].sum())



        # === 5. Player Movement & Physical Intensity ===
   # === MOVEMENT ANALYSIS ===
    movement_analysis = {}

    if all(col in df.columns for col in ['count_sprint', 'count_high_acceleration', 'phase', 'player_name']):
        # ✅ Remove duplicate rows: keep last entry per player per phase
        df_unique = df.sort_values(["player_name", "phase"]).drop_duplicates(subset=["player_name", "phase"], keep="last")

        # 🧮 Sprint totals per half
        first_half_mask = df_unique['phase'].isin(["1'-15'", "16'-30'", "31'-45+",])
        second_half_mask = df_unique['phase'].isin(["46'-60'", "61'-75'", "76'-90+",])

        first_half_sprints = int(df_unique[first_half_mask]['count_sprint'].sum())
        second_half_sprints = int(df_unique[second_half_mask]['count_sprint'].sum())


        # 🧠 Sprint drop-off percentage
        sprint_drop_pct_raw = calculate_percentage(
            first_half_sprints - second_half_sprints, first_half_sprints
        )

        # 🏃 Total running distance (across all players/unique phase rows)
        total_distance = float(df_unique['total_distance'].dropna().sum())

        # 💨 Average intensity (meters per minute)
        avg_intensity = float(df_unique['m/min'].dropna().mean()) if 'm/min' in df_unique.columns else None

        # ✅ Store clean movement metrics
        movement_analysis = {
            "first_half_sprints": first_half_sprints,
            "second_half_sprints": second_half_sprints,
            "sprint_intensity_drop_pct": sprint_drop_pct_raw,
            "high_accelerations": int(df_unique['count_high_acceleration'].sum()),
            "total_running_distance": round(total_distance, 2),
            "under_pressure_actions": int(df_unique['under_pressure'].sum()) if 'under_pressure' in df_unique.columns else 0,
            "counterpress_actions": int(df_unique['counterpress'].sum()) if 'counterpress' in df_unique.columns else 0,
        }

        if avg_intensity is not None:
            movement_analysis["avg_intensity_m_per_min"] = round(avg_intensity, 2)

        # 📌 Tactical pattern flag
        if sprint_drop_pct_raw > 30:
            patterns.append("Notable drop in sprint intensity in second half")

        # 📝 Summary string versions
        movement_analysis["summary_text"] = {
            "first_half_sprints": f"{first_half_sprints} sprints",
            "second_half_sprints": f"{second_half_sprints} sprints",
            "sprint_intensity_drop": f"{sprint_drop_pct_raw:.1f}%",
            "total_distance": f"{round(total_distance, 2)} meters",
            "avg_intensity": f"{round(avg_intensity, 2)} meters/min" if avg_intensity else "N/A"
        }

    # === PER PHASE SUMMARY ===
    if all(col in df.columns for col in ["phase", "count_sprint", "total_distance", "m/min", "player_name"]):
    # Keep only last row per player per phase (to avoid duplicate accumulation)
        df_valid = df.dropna(subset=["phase", "player_name", "count_sprint", "total_distance", "m/min"])
        df_unique = df_valid.sort_values(["player_name", "phase"]).drop_duplicates(subset=["player_name", "phase"], keep="last")

    # Then group by phase
    phase_summary = df_unique.groupby("phase").agg({
        "count_sprint": "sum",
        "total_distance": "sum",
        "m/min": "mean",
        "player_name": "nunique"
    }).reset_index().rename(columns={"player_name": "players_on_pitch"})

    phase_summary["minutes_played_in_phase"] = 15

    phase_summary["sprints_per_minute"] = (
        phase_summary["count_sprint"] / phase_summary["minutes_played_in_phase"]
    ).round(2)

    phase_summary["sprints_per_player"] = (
        phase_summary["count_sprint"] / phase_summary["players_on_pitch"]
    ).round(2)

    phase_summary["distance_per_minute"] = (
        phase_summary["total_distance"] / phase_summary["minutes_played_in_phase"]
    ).round(2)

    phase_summary["avg_sprint_distance"] = (
        phase_summary["total_distance"] / phase_summary["count_sprint"].replace(0, np.nan)
    ).fillna(0).round(2)

    # Final export with units
    summary["physical_phases"] = [
        {
            "phase": str(row["phase"]),
            "count_sprint": f"{int(row['count_sprint'])} sprints",
            "total_distance": f"{round(float(row['total_distance']), 2)} meters",
            "m/min": f"{round(float(row['m/min']), 2)} meters/min",
            "sprints_per_minute": f"{round(float(row['sprints_per_minute']), 2)} sprints/min",
            "sprints_per_player": f"{round(float(row['sprints_per_player']), 2)} sprints/player",
            "distance_per_minute": f"{round(float(row['distance_per_minute']), 2)} meters/min",
            "avg_sprint_distance": f"{round(float(row['avg_sprint_distance']), 2)} meters"
        }
        for _, row in phase_summary.iterrows()
    ]
    special_play_analysis = {}

    special_cols = [
        'pass_from_corner', 'pass_from_free_kick', 'pass_from_throwin',
        'is_through_ball', 'pass_is_dummy', 'is_cutback', 'is_cross'
    ]

    if all(col in df.columns for col in special_cols):
        special_play_analysis = {
            "corner_passes": safe_count(df['pass_from_corner']),
            "free_kick_passes": safe_count(df['pass_from_free_kick']),
            "throw_in_passes": safe_count(df['pass_from_throwin']),
            "through_balls": safe_count(df['is_through_ball']),
            "dummy_passes": safe_count(df['pass_is_dummy']),
            "cutback_passes": safe_count(df['is_cutback']),
            "crosses": safe_count(df['is_cross'])
        }
        # === 6.5 Carry-Based Progression & Penetration ===
    carry_analysis = {}

    carry_cols = [
        'carry_length',
        'carry_into_final_third',
        'carry_into_penalty_box',
        'carry_into_zone_14',
        'carry_leads_to_shot',
        'carry_leads_to_goal',
        'carry_ends_in_dispossession'
    ]

    if all(col in df.columns for col in carry_cols):
        carry_df = df[df['type_name'] == 'Carry']

        carry_analysis = {
            "total_carry_distance": round(carry_df['carry_length'].dropna().sum(), 2),
            "into_final_third": int(carry_df['carry_into_final_third'].sum()),
            "into_penalty_box": int(carry_df['carry_into_penalty_box'].sum()),
            "into_zone_14": int(carry_df['carry_into_zone_14'].sum()),
            "leads_to_shot": int(carry_df['carry_leads_to_shot'].sum()),
            "leads_to_goal": int(carry_df['carry_leads_to_goal'].sum()),
            "dispossessed_at_end": int(carry_df['carry_ends_in_dispossession'].sum())
        }


    shot_analysis = {}

    if all(col in df.columns for col in ['shot_statsbomb_xg', 'shot_gk_save_difficulty_xg', 'shot_type_name', 'shot_outcome_name']):
        shot_analysis = {
            'xG_total': round(df['shot_statsbomb_xg'].dropna().sum(), 2),
            'xGOT_total': round(df['shot_gk_save_difficulty_xg'].dropna().sum(), 2),
            'shot_types': df['shot_type_name'].dropna().value_counts().to_dict(),
            'shot_outcomes': df['shot_outcome_name'].dropna().value_counts().to_dict()
        }


        if shot_analysis['xG_total'] > 1.5:
            patterns.append(f"Created {shot_analysis['xG_total']} xG")

    summary['detailed_tactical_analysis']['shots'] = shot_analysis


    # Compile Tactical Patterns
    if summary['formation']:
        spatial_third = spatial_analysis.get("action_by_third", {}).get("Middle Third", 0)
        patterns.append(f"Tactical shape: {summary['formation']} with {spatial_third} actions in the middle third")
    
    if possession_analysis.get('progressive_pass_percentage', 0) > 30:
        patterns.append(f"Progressive passing: {possession_analysis['progressive_pass_percentage']}% of possessions")
    
    if attack_analysis.get('box_entries', 0) > 10:
        box_entries = attack_analysis['box_entries']
        penalty_carries = carry_analysis.get("into_penalty_box", 0)
        patterns.append(f"Penalty box threat: {box_entries} entries and {penalty_carries} carries into area")
    
    if movement_analysis.get('counterpress_actions', 0) > 10:
        final_recoveries = defensive_analysis.get("offensive_recoveries", 0)
        patterns.append(f"Counterpressing: {movement_analysis['counterpress_actions']} counterpress actions and {final_recoveries} recoveries high up")

    if special_play_analysis.get('through_balls', 0) > 5:
        patterns.append(f"Aggressive through ball strategy: {special_play_analysis['through_balls']} attempts")
   
    if carry_analysis.get('into_final_third', 0) > 10:
        final_third_carries = carry_analysis['into_final_third']
        prog_passes = possession_analysis.get("progressive_passes_total", 0)
        patterns.append(f"Vertical progression: {final_third_carries} carries into final third and {prog_passes} progressive passes")

    if carry_analysis.get('leads_to_shot', 0) > 3:
        patterns.append(f"Carries led to {carry_analysis['leads_to_shot']} shot attempts")

    if carry_analysis.get('leads_to_goal', 0) > 1:
        patterns.append("Carries directly led to goals")
    if 'under_pressure' in df.columns:
        movement_analysis['under_pressure_actions'] = int(df['under_pressure'].sum())

    if 'counterpress' in df.columns:
        movement_analysis['counterpress_actions'] = int(df['counterpress'].sum())

    if movement_analysis.get('counterpress_actions', 0) > 10:
        patterns.append("High counterpressing volume")
    if 'pass_aerial_won' in df.columns:
        defensive_analysis['aerial_pass_duels_won'] = int(df['pass_aerial_won'].sum())

    if 'duel_type_name' in df.columns:
        duel_types = df['duel_type_name'].dropna().value_counts().head(3).to_dict()
        defensive_analysis['duel_types'] = duel_types
    goalkeeper_analysis = {}

    # Goalkeeper key outcome-based events
    goalkeeper_metrics = {
        'shot_saved_to_post': 'shots_saved_to_post',
        'goalkeeper_shot_saved_to_post': 'keeper_saved_to_post',
        'goalkeeper_lost_out': 'keeper_lost_out',
        'goalkeeper_lost_in_play': 'keeper_lost_in_play',
        'goalkeeper_success_out': 'keeper_success_out',
        'goalkeeper_success_in_play': 'keeper_success_in_play',
        'goalkeeper_shot_saved_off_target': 'saved_off_target',
        'goalkeeper_punched_out': 'punches'
    }

    for col, label in goalkeeper_metrics.items():
        if col in df.columns:
            goalkeeper_analysis[label] = int(df[col].sum())

    # Attach to summary
    summary["detailed_tactical_analysis"]["goalkeeper"] = goalkeeper_analysis
    if 'pass_technique_name' in df.columns:
        technique_counts = df['pass_technique_name'].dropna().value_counts().head(3).to_dict()
        special_play_analysis['pass_techniques'] = technique_counts

    if 'pass_inswinging' in df.columns:
        special_play_analysis['inswingers'] = int(df['pass_inswinging'].sum())
    # xG faced by the goalkeeper (i.e., total xG of shots against the team)

    if 'pass_cross' in df.columns:
        special_play_analysis['crosses_total'] = int(df['pass_cross'].sum())

    if special_play_analysis.get('crosses_total', 0) > 12:
        patterns.append("Cross-heavy attacking pattern")
    if 'interception_outcome_name' in df.columns:
        interception_outcomes = df['interception_outcome_name'].dropna().value_counts().to_dict()
        defensive_analysis['interception_outcomes'] = interception_outcomes
    if '50_50_outcome_name' in df.columns:
        fifties = df['50_50_outcome_name'].dropna().value_counts().to_dict()
        defensive_analysis['fifty_fifty_outcomes'] = fifties
    if 'foul_won_defensive' in df.columns:
        defensive_analysis['fouls_won_defensively'] = int(df['foul_won_defensive'].sum())
    # Continue in special_play_analysis section
    if 'pass_cut_back' in df.columns:
        special_play_analysis['cutbacks'] = int(df['pass_cut_back'].sum())

    if 'pass_straight' in df.columns:
        special_play_analysis['straight_passes'] = int(df['pass_straight'].sum())

    if 'ball_recovery_offensive' in df.columns:
        special_play_analysis['offensive_recoveries'] = int(df['ball_recovery_offensive'].sum())

    if 'shot_deflected' in df.columns:
        special_play_analysis['shots_deflected'] = int(df['shot_deflected'].sum())

    if 'shot_open_goal' in df.columns:
        special_play_analysis['open_goal_shots'] = int(df['shot_open_goal'].sum())
    if special_play_analysis.get('cutbacks', 0) > 5:
        patterns.append("Cutback-focused attacks")
    discipline_analysis = {}

    if 'foul_committed_advantage' in df.columns:
        discipline_analysis['fouls_played_through'] = int(df['foul_committed_advantage'].sum())

    if 'foul_won_advantage' in df.columns:
        discipline_analysis['fouls_won_advantage_played'] = int(df['foul_won_advantage'].sum())

    if 'foul_committed_offensive' in df.columns:
        discipline_analysis['offensive_fouls'] = int(df['foul_committed_offensive'].sum())

    if 'foul_committed_penalty' in df.columns:
        discipline_analysis['penalties_conceded'] = int(df['foul_committed_penalty'].sum())

    if 'foul_won_penalty' in df.columns:
        discipline_analysis['penalties_won'] = int(df['foul_won_penalty'].sum())

    if 'foul_committed_type_name' in df.columns:
        discipline_analysis['foul_types'] = df['foul_committed_type_name'].dropna().value_counts().to_dict()

    if 'bad_behaviour_card_name' in df.columns:
        discipline_analysis['cards'] = df['bad_behaviour_card_name'].dropna().value_counts().to_dict()

    summary['detailed_tactical_analysis']['defense'] = defensive_analysis
    summary['detailed_tactical_analysis']['movement'] = movement_analysis
    summary['detailed_tactical_analysis']['special_plays'] = special_play_analysis
    summary['detailed_tactical_analysis']['discipline'] = discipline_analysis
    if discipline_analysis.get('penalties_conceded', 0) >= 2:
        patterns.append("Disciplinary issues: multiple penalties conceded")
    def identify_best_players(df, lineup):
    # === Position ID to Role Mapping ===
        role_map = {
            "Goalkeeper": [1],
            "Defender": [2, 3, 4, 5, 6, 7, 8],
            "Midfielder": [9, 10, 11, 12, 13, 14, 15, 16],
            "Forward": [17, 18, 19, 20, 21, 22, 23, 24, 25]
        }

        # Convert stringified lineup to list of dicts
        if isinstance(lineup, str):
            import ast
            lineup = ast.literal_eval(lineup)

        player_roles = {}
        for player in lineup:
            pos_id = player.get("position.id")
            name = player.get("player.name")
            for role, ids in role_map.items():
                if pos_id in ids:
                    player_roles[name] = role
                    break

        # Ensure all expected columns exist
        df = df.copy()
        df["goalkeeper_save"] = df.get("goalkeeper_save", 0)
        df["foul_committed_type_name"] = df.get("foul_committed_type_name", None)
        df["is_goal_assist"] = df.get("is_goal_assist", 0)
        df["is_shot_assist"] = df.get("is_shot_assist", 0)

        # Aggregate player stats
        metrics = df.groupby("player_name").agg({
            "goalkeeper_save": "sum",
            "shot_outcome_name": lambda x: (x == "Goal").sum(),
            "shot_statsbomb_xg": "sum",
            "pass_outcome_name": lambda x: (x == "1").sum(),
            "pass_type_name": "count",
            "carry_into_penalty_box": "sum",
            "carry_into_final_third": "sum",
            "carry_length": "sum",
            "interception_outcome_is_success_in_play": "sum",
            "duel_outcome_is_won": "sum",
            "clearance_aerial_won": "sum",
            "foul_committed_type_name": lambda x: x.notna().sum(),
            "is_goal_assist": "sum",
            "is_shot_assist": "sum"
        }).reset_index()

        metrics["player_role"] = metrics["player_name"].map(player_roles)

        # === Role-specific stat collections with proper names ===
        display_names = {
            "goalkeeper_save": "Saves",
            "pass_outcome_name": "Completed Passes",
            "duel_outcome_is_won": "Duels Won",
            "clearance_aerial_won": "Aerial Clearances",
            "interception_outcome_is_success_in_play": "Interceptions",
            "foul_committed_type_name": "Fouls Committed",
            "shot_outcome_name": "Goals",
            "is_goal_assist": "Assists",
            "is_shot_assist": "Key Passes",
            "carry_into_penalty_box": "Carries into Box",
            "carry_into_final_third": "Carries into Final Third"
        }

        def extract_stats(row, fields):
            return {display_names.get(k, k): row.get(k, 0) for k in fields}

        role_fields = {
            "Goalkeeper": ["goalkeeper_save", "pass_outcome_name"],
            "Defender": ["duel_outcome_is_won", "clearance_aerial_won", "interception_outcome_is_success_in_play", "foul_committed_type_name"],
            "Midfielder": ["pass_outcome_name", "carry_into_final_third", "interception_outcome_is_success_in_play", "duel_outcome_is_won"],
            "Forward": ["shot_outcome_name", "is_goal_assist", "is_shot_assist", "carry_into_penalty_box"]
        }

        best_players = {}
        for role in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            role_df = metrics[metrics["player_role"] == role].copy()
            if not role_df.empty:
                top_player = role_df.iloc[0]
                stats = extract_stats(top_player, role_fields[role])
                best_players[role] = {
                    "name": top_player["player_name"],
                    "stats": stats
                }

        return best_players



    if 'tactics_lineup' in df.columns:
            best_players = identify_best_players(df, df['tactics_lineup'].iloc[0])
            summary['best_players_by_position'] = best_players
    if discipline_analysis.get('cards', {}).get('Red Card', 0) >= 1:
        patterns.append("Received red card(s)")
# === Basic GK Stats (from flags) ===
    flag_fields = [
        ('shot_saved_to_post', 'shots_saved_to_post'),
        ('goalkeeper_shot_saved_to_post', 'keeper_saved_to_post'),
        ('goalkeeper_lost_out', 'keeper_lost_out'),
        ('goalkeeper_lost_in_play', 'keeper_lost_in_play'),
        ('goalkeeper_success_out', 'keeper_success_out'),
        ('goalkeeper_success_in_play', 'keeper_success_in_play'),
    ]
# === Spatial Analysis ===

    if 'pitch_third' in df.columns:
        third_counts = df['pitch_third'].value_counts().to_dict()
        spatial_analysis['action_by_third'] = third_counts

    if 'pitch_side' in df.columns:
        side_counts = df['pitch_side'].value_counts().to_dict()
        spatial_analysis['action_by_side'] = side_counts

    if 'zone_label' in df.columns:
        zone_counts = df['zone_label'].value_counts().to_dict()
        spatial_analysis['zone_usage'] = zone_counts

    if 'final_third_entry_side' in df.columns:
        ft_entry = df['final_third_entry_side'].value_counts().to_dict()
        spatial_analysis['final_third_entries'] = ft_entry

    if 'is_zone_14' in df.columns:
        z14_entries = int(df['is_zone_14'].sum())
        spatial_analysis['zone_14_entries'] = z14_entries

        if z14_entries > 5:
            z14_passes = df[(df["is_zone_14"] == True) & (df.get("is_shot_assist", False))].shape[0]
            z14_goals = df[(df["is_zone_14"] == True) & (df.get("is_goal_assist", False))].shape[0]
            patterns.append(
                f"Zone 14 dominance: {z14_entries} zone actions, including {z14_passes} key passes and {z14_goals} assists"
            )

    # Pattern: Left-flank dominance
    left = spatial_analysis.get('action_by_side', {}).get('Left', 0)
    right = spatial_analysis.get('action_by_side', {}).get('Right', 0)
    if right > 0 and left > 1.5 * right:
        patterns.append(f"Left-flank dominance: {left} vs {right} actions")

    # Pattern: Final third dominance
    final_third = spatial_analysis.get('action_by_third', {}).get('Final Third', 0)
    if final_third > 0 and final_third / len(df) > 0.35:
        patterns.append(f"Final third territorial dominance: {final_third} actions in this area")

    # Save to summary
    summary['detailed_tactical_analysis']['spatial'] = spatial_analysis
    summary["key_tactical_patterns"] = patterns


    # Combine all tactical categories into one enriched dictionary
    summary["detailed_tactical_analysis"] = {
        "possession": possession_analysis,
        "attack": attack_analysis,
        "defense": defensive_analysis,
        "movement": movement_analysis,
        "special_plays": special_play_analysis,
        "carries": carry_analysis,
        "goalkeeper": goalkeeper_analysis if 'goalkeeper_analysis' in locals() else {},
        "discipline": discipline_analysis if 'discipline_analysis' in locals() else {},
        "shots": shot_analysis if 'shot_analysis' in locals() else {},
        "spatial": spatial_analysis if 'spatial_analysis' in locals() else {}
    }

    # Final Narrative Summary
    narrative = f"{team_name} displayed a {summary['formation']} formation. "
    if patterns:
        narrative += "Tactical highlights include: " + ", ".join(patterns) + "."
    else:
        narrative += "No dominant tactical patterns identified."

    summary["summary"] = narrative

    return summary
spatial_analysis = {}

# === PATH SETUP ===
input_dir = r"E:/Ai_com/app/TV_teams"
output_dir = r"E:/Ai_com/bot_knowledge/team_summaries"
os.makedirs(output_dir, exist_ok=True)

# === FILE LISTING ===
files = [f for f in os.listdir(input_dir) if f.endswith("_TV.parquet")]
print(f"📁 Found {len(files)} files in {input_dir}")

for file in files:
    print(f"\n🔍 Checking: {file}")

    match = re.match(r"(match_\d+)_+team_(.+)_TV\.parquet", file)
    if not match:
        print(f"❌ Skipped (invalid filename format): {file}")
        continue

    match_id, raw_team = match.groups()
    team_name = raw_team.replace("_", " ")
    file_path = os.path.join(input_dir, file)

    import traceback
    import numpy as np  # Place this at the top of your script

    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_)): return bool(obj)
        return str(obj)

    try:
        print(f"📥 Reading file: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded: {df.shape}")

        summary = generate_detailed_tactical_summary(df, match_id, team_name)
        if not isinstance(summary, dict):
            print(f"⚠️ Skipped: summary is not a dictionary for {team_name}")
            continue
        if "summary" not in summary:
            print(f"⚠️ Skipped: summary missing 'summary' field for {team_name}")
            continue

        out_path = os.path.join(output_dir, f"{match_id}__team_{raw_team}_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=default_serializer)

        print(f"💾 Saved to: {out_path}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
    traceback.print_exc()


print("\n🏁 Done! All enhanced team-level tactical summaries generated.")
