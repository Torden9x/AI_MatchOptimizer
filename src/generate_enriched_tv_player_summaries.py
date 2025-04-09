# generate_enriched_tv_player_summaries.py
import pandas as pd
import os
import re
import json
clustering_df = pd.read_csv("E:/Ai_com/app/Final Player Clustering.csv")
input_dir = r"E:/Ai_com/app/TV_players"
output_dir = r"E:/Ai_com/bot_knowledge/summaries"
def get_player_style(player_id):
    try:
        style = clustering_df.loc[clustering_df['player_id'] == player_id, 'player_style'].values
        return style[0] if len(style) > 0 else None
    except:
        return None
os.makedirs(output_dir, exist_ok=True)
files = [f for f in os.listdir(input_dir) if f.endswith("_TV.parquet")]
print(f"Found {len(files)} player TV files.")
for file in files:
    match = re.match(r"(match_\d+)___player_(.+)_TV.parquet", file)
    if not match:
        continue

    match_id, raw_name = match.groups()
    player_name = raw_name.replace("_", " ")
    file_path = os.path.join(input_dir, file)
    print(f"🔍 Processing file: {file}")


    try:
        df = pd.read_parquet(file_path)
        team = df['team_name'].dropna().unique()[0] if 'team_name' in df.columns else "Unknown"
        pos = df['position_name'].dropna().unique()[0] if 'position_name' in df.columns else "Unknown"

        summary = {
            "match_id": match_id.replace("match_", ""),
            "player_name": player_name,
            "team": team,
            "position": pos,
            "key_actions": [],
            "fatigue": "",
            "summary": ""
        }

        def safe_count(cond):
            return int(cond.sum()) if cond is not None else 0

        def safe_unique_count(col):
            return df[col].dropna().nunique()

        def safe_avg(col):
            return round(df[col].dropna().mean(), 2) if col in df.columns else None

        # Core actions
        summary["key_actions"].append(f"{safe_count(df['type_name'] == 'Pass')} passes")
        summary["key_actions"].append(f"{safe_count((df['type_name'] == 'Pass') & (df['pass_outcome_name'] == '1'))} accurate passes")
        summary["key_actions"].append(f"{safe_count(df['is_progressive_carry'])} progressive carries")
        summary["key_actions"].append(f"{safe_count(df['is_cross'])} crosses")
        summary["key_actions"].append(f"{safe_count(df['is_shot_assist'])} key passes")
        summary["key_actions"].append(f"{safe_count(df['type_name'] == 'Shot')} shots")
        summary["key_actions"].append(f"{safe_count(df['shot_outcome_name'] == 'Goal')} goals")
        summary["key_actions"].append(f"{round(df['shot_statsbomb_xg'].sum(), 2)} xG")
        summary["key_actions"].append(f"{round(df['xT'].sum(), 2)} xT")
        summary["key_actions"].append(f"{safe_count(df['type_name'] == 'Duel')} duels")
        summary["key_actions"].append(f"{safe_count(df['duel_type_name'].astype(str).str.contains('Tackle'))} tackles")
        summary["key_actions"].append(f"{safe_count(df['type_name'] == 'Interception')} interceptions")
        summary["key_actions"].append(f"{safe_count(df['type_name'] == 'Clearance')} clearances")
        summary["key_actions"].append(f"{safe_count(df['type_name'] == 'Ball Recovery')} recoveries")

        # Spatial + tactical zones
        summary["key_actions"].append(f"{safe_count(df['box_entry'])} box entries")
        summary["key_actions"].append(f"{safe_count(df['is_zone_14'])} zone 14 entries")
        summary["key_actions"].append(f"{safe_unique_count('final_third_entry_side')} channels used to enter final third")
        summary["key_actions"].append(f"{safe_unique_count('zone_label')} unique zones occupied")

        # Tactical signal flags
        if 'pass_pass_success_probability' in df.columns:
            pass_acc = safe_avg('pass_pass_success_probability')
            if pass_acc:
                summary["key_actions"].append(f"{pass_acc} avg. pass success %")

        if 'pass_pass_cluster_label' in df.columns:
            top_cluster = df['pass_pass_cluster_label'].dropna().mode()
            if not top_cluster.empty:
                summary["key_actions"].append(f"favored pass type: {top_cluster[0]}")

        if safe_count(df['under_pressure']) > 0:
            summary["key_actions"].append(f"{safe_count(df['under_pressure'])} actions under pressure")

        if safe_count(df['pass_switch']) > 0:
            summary["key_actions"].append(f"{safe_count(df['pass_switch'])} switch passes")

        if 'shot_technique_name' in df.columns:
            shot_styles = df['shot_technique_name'].dropna().value_counts().to_dict()
            for style, count in shot_styles.items():
                summary["key_actions"].append(f"{count} shots by {style.lower()}")

        if 'clearance_body_part_name' in df.columns:
            body_parts = df['clearance_body_part_name'].dropna().value_counts().to_dict()
            for part, count in body_parts.items():
                summary["key_actions"].append(f"{count} clearances with {part.lower()}")
        if 'pass_aerial_won' in df.columns:
            aerial_wins = int(df['pass_aerial_won'].sum())
            if aerial_wins > 0:
                summary["key_actions"].append(f"{aerial_wins} aerial pass wins")
        if 'counterpress' in df.columns:
            cp = int(df['counterpress'].sum())
            if cp > 0:
                summary["key_actions"].append(f"{cp} counterpressing actions")
        if 'pass_switch' in df.columns:
            switches = int(df['pass_switch'].sum())
            if switches > 0:
                summary["key_actions"].append(f"{switches} switch passes")
        if 'clearance_body_part_name' in df.columns:
            body_part_counts = df['clearance_body_part_name'].dropna().value_counts().to_dict()
            for part, count in body_part_counts.items():
                summary["key_actions"].append(f"{count} clearances with {part.lower()}")
        if 'clearance_head' in df.columns:
            head_clearances = int(df['clearance_head'].sum())
            if head_clearances > 0:
                summary["key_actions"].append(f"{head_clearances} clearances with head")
        if 'shot_technique_name' in df.columns:
            technique_counts = df['shot_technique_name'].dropna().value_counts().to_dict()
            for tech, count in technique_counts.items():
                summary["key_actions"].append(f"{count} shots by {tech.lower()}")
        if 'obv_total_net' in df.columns:
            obv_net = round(df['obv_total_net'].sum(), 2)
            summary["key_actions"].append(f"{obv_net} OBV net value")
        if 'pass_length' in df.columns:
            avg_length = round(df['pass_length'].dropna().mean(), 2)
            summary["key_actions"].append(f"{avg_length} avg. pass length")
        if 'interception_outcome_name' in df.columns:
            intc_types = df['interception_outcome_name'].dropna().value_counts().to_dict()
            for outcome, count in intc_types.items():
                summary["key_actions"].append(f"{count} interceptions - {outcome.lower()}")
        if 'duel_outcome_name' in df.columns:
            duel_outcomes = df['duel_outcome_name'].dropna().value_counts().to_dict()
            for outcome, count in duel_outcomes.items():
                summary["key_actions"].append(f"{count} duels - {outcome.lower()}")
        if 'pass_through_ball' in df.columns:
            throughs = int(df['pass_through_ball'].sum())
            if throughs > 0:
                summary["key_actions"].append(f"{throughs} through balls")
        if 'pass_cut_back' in df.columns:
            cutbacks = int(df['pass_cut_back'].sum())
            if cutbacks > 0:
                summary["key_actions"].append(f"{cutbacks} cutbacks")
        if 'ball_recovery_offensive' in df.columns:
            offensive_rec = int(df['ball_recovery_offensive'].sum())
            if offensive_rec > 0:
                summary["key_actions"].append(f"{offensive_rec} offensive ball recoveries")
        if 'prog_pass' in df.columns:
            prog_pass_sum = round(df['prog_pass'].dropna().sum(), 2)
            summary["key_actions"].append(f"{prog_pass_sum} yds progressive passing")

        if 'prog_carry' in df.columns:
            prog_carry_sum = round(df['prog_carry'].dropna().sum(), 2)
            summary["key_actions"].append(f"{prog_carry_sum} yds progressive carrying")
        if 'carry_into_final_third' in df.columns:
            fth = int(df['carry_into_final_third'].sum())
            if fth > 0:
                summary["key_actions"].append(f"{fth} carries into final third")
        if 'carry_leads_to_shot' in df.columns:
            if int(df['carry_leads_to_shot'].sum()) > 0:
                summary["key_actions"].append(f"{int(df['carry_leads_to_shot'].sum())} carries led to shots")

        if 'carry_leads_to_goal' in df.columns:
            if int(df['carry_leads_to_goal'].sum()) > 0:
                summary["key_actions"].append(f"{int(df['carry_leads_to_goal'].sum())} carries led to goals")
        if 'carry_ends_in_dispossession' in df.columns:
            dispossessed = int(df['carry_ends_in_dispossession'].sum())
            if dispossessed > 0:
                summary["key_actions"].append(f"{dispossessed} carries ended in dispossession")
        if 'dribble_successful' in df.columns and 'dribble_attempted' in df.columns:
            total_attempts = int(df['dribble_attempted'].sum())
            total_success = int(df['dribble_successful'].sum())
        if total_attempts > 0:
            drb_rate = round((total_success / total_attempts) * 100, 1)
            summary["key_actions"].append(f"{drb_rate}% dribble success rate")

        if 'dribble_successful' in df.columns and 'dribble_attempted' in df.columns:
            drb_succ = int(df[df['dribble_attempted'] == 1]['dribble_successful'].sum())

        if 'type_name' in df.columns:
            clean_rec = df[(df['type_name'] == 'Ball Recovery') & (df['ball_recovery_is_complete'] == 1)]
            summary["key_actions"].append(f"{len(clean_rec)} clean ball recoveries")

        player_id = df['player_id'].dropna().unique()[0] if 'player_id' in df.columns else None
        player_style = get_player_style(player_id)
        if player_style:
            summary["player_style"] = player_style

        
        # Fatigue detection
        fatigue_note = ""
        if 'phase' in df.columns and 'count_sprint' in df.columns:
            phases = df.groupby("phase")['count_sprint'].sum()
            if '1st Half' in phases and '2nd Half' in phases:
                s1, s2 = phases['1st Half'], phases['2nd Half']
                if s1 > 0:
                    drop = (s1 - s2) / s1
                    if drop >= 0.5:
                        fatigue_note = f"Heavy fatigue: sprint drop-off of {int(drop*100)}%"
                    elif drop >= 0.3:
                        fatigue_note = f"Moderate fatigue: sprint drop-off of {int(drop*100)}%"
                    elif drop >= 0.15:
                        fatigue_note = f"Slight fatigue: {int(drop*100)}% fewer sprints in 2nd half"

        # === PHYSICAL METRICS ===
        physical_note = []

        def safe_phase_sum(metric, phase_label):
            if metric in df.columns and 'phase' in df.columns:
                return df[df['phase'] == phase_label][metric].sum()
            return 0

        if 'total_distance' in df.columns:
            total_distance = df['total_distance'].sum()
            physical_note.append(f"🛣️ Total distance: {int(total_distance)} meters")

        if 'sprinting_distance' in df.columns:
            sprint_distance = df['sprinting_distance'].sum()
            physical_note.append(f"⚡ Sprinting distance: {int(sprint_distance)} meters")

        if 'count_hsr' in df.columns:
            hsr = int(df['count_hsr'].sum())
            physical_note.append(f"🏃‍♂️ High-speed runs: {hsr} bursts")

        if 'max_speed' in df.columns:
            max_speed = df['max_speed'].max()
            if not pd.isna(max_speed):
                physical_note.append(f"🚀 Max speed: {round(max_speed, 2)} m/s")

        if 'm/min' in df.columns:
            intensity = df['m/min'].dropna().mean()
            if not pd.isna(intensity):
                physical_note.append(f"💨 Avg intensity: {int(round(intensity))} meters/min")

        d1 = safe_phase_sum('total_distance', '1st Half')
        d2 = safe_phase_sum('total_distance', '2nd Half')
        if d1 and d2:
            dist_diff = d1 - d2
            dist_drop_pct = round((dist_diff / d1) * 100, 1) if d1 > 0 else 0
            if dist_drop_pct > 5:
                physical_note.append(f"📉 Distance drop: {int(d1)} → {int(d2)} meters ({dist_drop_pct}% less in 2nd half)")

        # === ADD TO JSON ===
        if physical_note:
            summary["physical_summary"] = " ".join(physical_note)
        if fatigue_note:
            summary["fatigue"] = fatigue_note

        # === FINAL SUMMARY TEXT ===
        summary["summary"] = (
            f"{pos} for {team} — involved in {len(summary['key_actions'])} key actions: "
            + ", ".join(summary["key_actions"]) + "."
        )

        if fatigue_note:
            summary["summary"] += f" {fatigue_note}"

        if physical_note:
            summary["summary"] += " Physical load: " + ". ".join(physical_note) + "."

        # ✅ EXPORT AFTER EVERYTHING IS READY
               # ✅ EXPORT AFTER EVERYTHING IS READY
        out_name = f"{match_id}__player_{raw_name}_summary.json"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print("✅ Fully enriched JSON summaries for TV_players generated.")
