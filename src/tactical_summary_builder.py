
import json

def build_tactical_summary(team_name, team_summary):
    dta = team_summary.get("detailed_tactical_analysis", {})
    match_id = team_summary.get("match_id")
    team = team_summary.get("team")
    formation = team_summary.get("formation")
    summary = team_summary.get("summary")
    key_tactical_patterns = team_summary.get("key_tactical_patterns", [])
    pass_clusters = team_summary.get("detailed_tactical_analysis", {}).get("possession", {}).get("pass_clusters", {})
    mid_center_right = pass_clusters.get("Midfield third - Center - To right - Short - Ground Pass", 0)
    mid_center_left = pass_clusters.get("Midfield third - Center - To left - Short - Ground Pass", 0)
    mid_left_backwards = pass_clusters.get("Midfield third - Left - Backwards - Short - Ground Pass", 0)
    # Extract pass types
    pass_types = team_summary.get("detailed_tactical_analysis", {}).get("possession", {}).get("pass_types", {})

    recovery_passes = pass_types.get("Recovery", 0)
    throw_ins = pass_types.get("Throw-in", 0)
    corners = pass_types.get("Corner", 0)
    free_kicks = pass_types.get("Free Kick", 0)
    goal_kicks = pass_types.get("Goal Kick", 0)
    long_passes = pass_types.get("Long", 0)
    short_passes = pass_types.get("Short", 0)
    defensive_passes = pass_types.get("Defensive", 0)
    center_passes = pass_types.get("Center", 0)
    attacking_passes = pass_types.get("Attacking", 0)
    pass_accuracy = team_summary.get("detailed_tactical_analysis", {}).get("possession", {}).get("pass_accuracy_breakdown", {})
    total_passes = pass_accuracy.get("total_passes", 0)
    accurate_passes = pass_accuracy.get("accurate_passes", 0)
    inaccurate_passes = pass_accuracy.get("inaccurate_passes", 0)
    progressive_passes = pass_accuracy.get("progressive_passes", 0)
    progressive_accurate = pass_accuracy.get("progressive_accurate", 0)
    progressive_inaccurate = pass_accuracy.get("progressive_inaccurate", 0)

    attack = team_summary.get("detailed_tactical_analysis", {}).get("attack", {})
    preferred_zones = attack.get("preferred_attack_zones", {})
    right_attacks = preferred_zones.get("Final Third - Right", "N/A")
    center_attacks = preferred_zones.get("Final Third - Center", "N/A")
    left_attacks = preferred_zones.get("Final Third - Left", "N/A")
    box_entries = attack.get("box_entries", "N/A")
    goal_assists = attack.get("goal_assists", "N/A")
    shot_assists = attack.get("shot_assists", "N/A")
    defense = team_summary.get("detailed_tactical_analysis", {}).get("defense", {})

    # Direct values
    offensive_blocks = defense.get("offensive_blocks", "N/A")
    goal_saving_blocks = defense.get("goal_saving_blocks", "N/A")
    headers_on_shots = defense.get("headers_on_shots", "N/A")
    aerial_duels_won = defense.get("aerial_pass_duels_won", "N/A")
    fouls_won_defensively = defense.get("fouls_won_defensively", "N/A")

    # Nested dicts
    duel_types = defense.get("duel_types", {})
    aerial_lost = duel_types.get("Aerial Lost", "N/A")
    tackles = duel_types.get("Tackle", "N/A")

    interceptions = defense.get("interception_outcomes", {})
    interceptions_success = interceptions.get("Success In Play", "N/A")
    interceptions_won = interceptions.get("Won", "N/A")
    interceptions_lost_out = interceptions.get("Lost Out", "N/A")
    interceptions_lost_in_play = interceptions.get("Lost In Play", "N/A")

    fifty_fifty = defense.get("fifty_fifty_outcomes", {})
    fifty_fifty_oppo = fifty_fifty.get("Success To Opposition", "N/A")
    fifty_fifty_lost = fifty_fifty.get("Lost", "N/A")
    
    movement = team_summary.get("detailed_tactical_analysis", {}).get("movement", {})
    # Direct values
    first_half_sprints = movement.get("first_half_sprints", "N/A")
    second_half_sprints = movement.get("second_half_sprints", "N/A")
    sprint_drop_pct = movement.get("sprint_intensity_drop_pct", "N/A")
    high_accels = movement.get("high_accelerations", "N/A")
    total_distance = movement.get("total_running_distance", "N/A")
    under_pressure = movement.get("under_pressure_actions", "N/A")
    counterpress = movement.get("counterpress_actions", "N/A")
    intensity_per_min = movement.get("avg_intensity_m_per_min", "N/A")

    # Optional summary text
    movement_text = movement.get("summary_text", {})
    sprint_summary = movement_text.get("first_half_sprints", "")
    sprint2_summary = movement_text.get("second_half_sprints", "")
    intensity_summary = movement_text.get("avg_intensity", "")
    distance_summary = movement_text.get("total_distance", "")
    
    special = team_summary.get("detailed_tactical_analysis", {}).get("special_plays", {})

    corner_passes = special.get("corner_passes", "N/A")
    free_kick_passes = special.get("free_kick_passes", "N/A")
    throw_in_passes = special.get("throw_in_passes", "N/A")
    through_balls = special.get("through_balls", "N/A")
    dummy_passes = special.get("dummy_passes", "N/A")
    cutback_passes = special.get("cutback_passes", "N/A")
    crosses = special.get("crosses", "N/A")

    pass_techniques = special.get("pass_techniques", {})
    inswingers = pass_techniques.get("Inswinging", "N/A")
    outswingers = pass_techniques.get("Outswinging", "N/A")
    through_ball_tech = pass_techniques.get("Through Ball", "N/A")
    inswingers = special.get("inswingers", "N/A")
    crosses_total = special.get("crosses_total", "N/A")
    cutbacks = special.get("cutbacks", "N/A")
    straight_passes = special.get("straight_passes", "N/A")
    offensive_recoveries = special.get("offensive_recoveries", "N/A")
    shots_deflected = special.get("shots_deflected", "N/A")
    open_goal_shots = special.get("open_goal_shots", "N/A")
    # === Carries ===
    carries = dta.get("carries", {})
    total_carry_distance = carries.get("total_carry_distance", "N/A")
    into_final_third = carries.get("into_final_third", "N/A")
    into_penalty_box = carries.get("into_penalty_box", "N/A")
    into_zone_14 = carries.get("into_zone_14", "N/A")
    leads_to_shot = carries.get("leads_to_shot", "N/A")
    leads_to_goal = carries.get("leads_to_goal", "N/A")
    dispossessed_at_end = carries.get("dispossessed_at_end", "N/A")
    
    # === Goalkeeper Actions ===
    gk = dta.get("goalkeeper", {})
    shots_saved_to_post = gk.get("shots_saved_to_post", "N/A")
    keeper_saved_to_post = gk.get("keeper_saved_to_post", "N/A")
    keeper_lost_out = gk.get("keeper_lost_out", "N/A")
    keeper_lost_in_play = gk.get("keeper_lost_in_play", "N/A")
    keeper_success_out = gk.get("keeper_success_out", "N/A")
    keeper_success_in_play = gk.get("keeper_success_in_play", "N/A")
    saved_off_target = gk.get("saved_off_target", "N/A")
    punches = gk.get("punches", "N/A")
    
    # === Discipline Metrics ===
    discipline = dta.get("discipline", {})
    fouls_played_through = discipline.get("fouls_played_through", "N/A")
    fouls_won_advantage_played = discipline.get("fouls_won_advantage_played", "N/A")
    offensive_fouls = discipline.get("offensive_fouls", "N/A")
    penalties_conceded = discipline.get("penalties_conceded", "N/A")
    penalties_won = discipline.get("penalties_won", "N/A")
    foul_types = discipline.get("foul_types", {})
    cards = discipline.get("cards", {})
    # === High-Level Tactical Summary ===
    summary_text = team_summary.get("summary", "N/A")
    # === Lineup Info ===
    lineup = team_summary.get("lineup", [])
    lineup_text = "\n".join(
        [f"{p['jersey']:>2} - {p['name']} ({p['position']})" for p in lineup]
    ) if lineup else "N/A"

    # === Shots Metrics ===
    shots = dta.get("shots", {})
    xg_total = shots.get("xG_total", "N/A")
    xgot_total = shots.get("xGOT_total", "N/A")
    shot_types = shots.get("shot_types", {})
    shot_outcomes = shots.get("shot_outcomes", {})
    # Format tactical summary
    special_summary = (
        f"Corner: {corner_passes}, FK: {free_kick_passes}, Throw-ins: {throw_in_passes} | "
        f"Through Balls: {through_balls}, Dummies: {dummy_passes}, Cutbacks: {cutback_passes} | "
        f"Crosses: {crosses} (Inswinging: {inswingers}, Outswinging: {outswingers})"
    )
    
    # === Physical Phases ===
    physical_phases = team_summary.get("physical_phases", [])
    physical_summary_lines = []

    for phase in physical_phases:
        phase_desc = phase.get("phase", "N/A")
        sprints = phase.get("count_sprint", "N/A")
        distance = phase.get("total_distance", "N/A")
        meters_per_min = phase.get("m/min", "N/A")
        sprints_per_min = phase.get("sprints_per_minute", "N/A")
        sprints_per_player = phase.get("sprints_per_player", "N/A")
        dist_per_min = phase.get("distance_per_minute", "N/A")
        avg_sprint_dist = phase.get("avg_sprint_distance", "N/A")

        physical_summary_lines.append(
            f"📊 Phase {phase_desc} — {sprints}, Distance: {distance}, Intensity: {meters_per_min}, "
            f"Sprints/min: {sprints_per_min}, Per Player: {sprints_per_player}, "
            f"Dist/min: {dist_per_min}, Avg Sprint: {avg_sprint_dist}"
        )

        # === Best Players by Position ===
        best_players = team_summary.get("best_players_by_position", {})
        player_highlights = []

        for role, info in best_players.items():
            name = info.get("name", "Unknown")
            stats = info.get("stats", {})
            stats_summary = ", ".join(f"{k}: {v}" for k, v in stats.items())
            player_highlights.append(f"⭐ {role}: {name} → {stats_summary}")

        best_players_summary = "\n".join(player_highlights) if player_highlights else "N/A"
        lineup_summary = ", ".join(
        f"{player['name']} ({player['position']})"
        for player in lineup
    )
    physical_summary = "\n".join(physical_summary_lines) if physical_summary_lines else "N/A"
    key_tactical_summary = "\n- ".join(key_tactical_patterns) if key_tactical_patterns else "N/A"


    summary = f"""
    Team: {team}
                    Formation: {formation}
                    Tactical Summary: {summary_text}

                    🎯 Key Tactical Patterns:
                    - - {key_tactical_summary}


                    🧠 Possession:
                    - Total Passes: {pass_accuracy.get("total_passes", "N/A")}, Accurate: {pass_accuracy.get("accurate_passes", "N/A")}, Inaccurate: {pass_accuracy.get("inaccurate_passes", "N/A")}
                    - Pass Types: {json.dumps(pass_types, indent=2)}
                    - Pass Clusters: {json.dumps(pass_clusters, indent=2)}
                    - Key Pass Clusters:
                    • Mid-Center → Right: {pass_clusters.get("Midfield third - Center - To right - Short - Ground Pass", 0)}
                    • Mid-Center → Left: {pass_clusters.get("Midfield third - Center - To left - Short - Ground Pass", 0)}
                    • Mid-Left → Backwards: {pass_clusters.get("Midfield third - Left - Backwards - Short - Ground Pass", 0)}
                    - Pass Types: {json.dumps(pass_types, indent=2)}
                    - Recovery Passes: {recovery_passes}
                    - Throw-ins: {throw_ins}
                    - Corners: {corners}
                    - Free Kicks: {free_kicks}
                    - Goal Kicks: {goal_kicks}
                    - Long Passes: {long_passes}
                    - Short Passes: {short_passes}
                    - Defensive Passes: {defensive_passes}
                    - Center Passes: {center_passes}
                    - Total Passes: {total_passes}
                    • Accurate: {accurate_passes} | Inaccurate: {inaccurate_passes}
                    - Progressive Passes: {progressive_passes}
                    • Accurate: {progressive_accurate} | Inaccurate: {progressive_inaccurate}
                    - Attacking Passes: {attacking_passes}
                    ⚔️ Preferred Attack Zones:
                    - Right Side: {right_attacks} actions
                    - Center: {center_attacks} actions
                    - Left Side: {left_attacks} actions
                    ⚔️ Attack:
                    - Box Entries: {box_entries}
                    - Shot Assists: {shot_assists}, Goal Assists: {goal_assists}

                    🛡️ Defense:
                    - Offensive Blocks: {offensive_blocks}, Goal-Saving Blocks: {goal_saving_blocks}
                    - Headers on Shots: {headers_on_shots},
                    - Duels: {json.dumps(duel_types, indent=2)}
                    - Fouls Won Defensively: {fouls_won_defensively}
                    🛡️ Defensive Duels:
                    - Aerial Duels Won: {aerial_duels_won}
                    - Aerial Duels Lost: {aerial_lost}
                    - Tackles Made: {tackles}
                    - Successful in Play: {interceptions_success}
                    - Interceptions Won: {interceptions_won}
                    - Lost Out: {interceptions_lost_out}
                    - Lost In Play: {interceptions_lost_in_play}
                    🤜 50/50 Duel Outcomes:
                    - Success to Opposition: {fifty_fifty_oppo}
                    - Lost: {fifty_fifty_lost}

                    💨 Movement & Physical:
                    - First Half Sprints: {first_half_sprints}, Second Half: {second_half_sprints},
                    🏃 Movement Summary:
                    - Sprint Intensity Drop: {sprint_drop_pct}%
                    - High Accelerations: {high_accels}
                    - Total Running Distance: {total_distance} meters
                    - Under Pressure Actions: {under_pressure}
                    - Counterpressing Actions: {counterpress}
                    - Avg Intensity (m/min): {intensity_per_min}
                    📊 Sprint & Intensity Breakdown:
                    - First Half Sprints: {sprint_summary}
                    - Second Half Sprints: {sprint2_summary}
                    - Total Distance: {distance_summary}
                    - Average Intensity: {intensity_summary}
                    🎯 Special Plays:
                    - Corners: {corner_passes}, Free Kicks: {free_kick_passes}, Throw-ins: {throw_in_passes}
                    - Through Balls: {through_balls}, Cutbacks: {cutback_passes}, Crosses: {crosses_total}
                    - Pass Techniques: {json.dumps(pass_techniques, indent=2)}
                    - Through Balls (technique): {through_ball_tech}
                    - Cutbacks: {cutbacks}
                    - Straight Passes: {straight_passes}
                    - Offensive Recoveries: {offensive_recoveries}
                    - Shots Deflected: {shots_deflected}
                    - Open Goal Shots: {open_goal_shots}

                    📦 Carries:
                    - Total Carry Distance: {total_carry_distance}
                    - Into Final Third: {into_final_third}, Penalty Box: {into_penalty_box}, Zone 14: {into_zone_14}
                    - Led to Shot: {leads_to_shot}
                    - Led to Goal: {leads_to_goal}
                    - Dispossessed at End: {dispossessed_at_end}
                    🧤 Goalkeeping:
                    - Shots Saved to Post: {shots_saved_to_post}
                    - Keeper Success (Out): {keeper_success_out}, (In Play): {keeper_success_in_play}
                    - Saved to Post: {keeper_saved_to_post}
                    - Lost Out: {keeper_lost_out}
                    - Lost In Play: {keeper_lost_in_play}
                    - Saved Off Target: {saved_off_target}
                    - Punches: {punches}

                    🚨 Discipline:
                    - Fouls Played Through: {fouls_played_through}, Won Advantage: {fouls_won_advantage_played}
                    - Offensive Fouls: {offensive_fouls}, Penalties Conceded: {penalties_conceded}, Won: {penalties_won}
                    - Fouls: {foul_types}
                    - cards {cards}

                    🎯 Shots:
                    - Shot Types: {json.dumps(shot_types, indent=2)}
                    - Shot Outcomes: {json.dumps(shot_outcomes, indent=2)}
                    xgot_total: {xgot_total}
                    xg_total: {xg_total}

                    👥 Lineup:
                    {lineup_summary}

                    📊 Physical Phases:
                    {json.dumps(physical_phases, indent=2)}
                    physical_summary:
                    {physical_summary}

                    🏅 Best Players by Position:
                    {best_players_summary}
    """
    return summary.strip()
