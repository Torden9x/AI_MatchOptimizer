# vector_builder.py — Full Tactical Vector Builder with Technique Enrichment
import pandas as pd
import numpy as np
import os
match_id = "3925226"

event_df = pd.read_csv(r"E:\Ai_com\match_enriched\new\match_3925226_.csv")

# ======================= PITCH ZONES =======================
vertical_conditions = [(event_df['end_x'] < 40), (event_df['end_x'] >= 40) & (event_df['end_x'] < 80), (event_df['end_x'] >= 80)]
vertical_labels = ['Defensive Third', 'Middle Third', 'Final Third']
event_df['pitch_third'] = np.select(vertical_conditions, vertical_labels)

horizontal_conditions = [(event_df['end_y'] < 22.67), (event_df['end_y'] >= 22.67) & (event_df['end_y'] < 45.33), (event_df['end_y'] >= 45.33)]
horizontal_labels = ['Left', 'Center', 'Right']
event_df['pitch_side'] = np.select(horizontal_conditions, horizontal_labels)
event_df['zone_label'] = event_df['pitch_third'] + ' - ' + event_df['pitch_side']

is_pass_or_carry = ((event_df['type_name'] == 'Pass') & (event_df['pass_outcome_name'] == '1')) | (event_df['type_name'] == 'Carry')
ends_in_box = (((event_df['type_name'] == 'Pass') & (event_df['end_x'] >= 88.5) & (event_df['end_y'].between(16, 64.2))) | ((event_df['type_name'] == 'Carry') & (event_df['carry_end_x'] >= 88.5) & (event_df['carry_end_y'].between(16, 64.2))))
not_already_in_box = ~((event_df['x'] >= 101.1) & (event_df['y'].between(16, 64.2)))
not_set_piece = ~event_df['play_pattern_name'].astype(str).str.contains('From Corner|From Free Kick|From Throw In', na=False)
event_df['box_entry'] = is_pass_or_carry & ends_in_box & not_already_in_box & not_set_piece

is_final_third = event_df['end_x'] >= 92.5
left_entry = (event_df['end_y'] < 25.67)
middle_entry = (event_df['end_y'] >= 25.67) & (event_df['end_y'] < 53.33)
right_entry = (event_df['end_y'] >= 53.33)
event_df['final_third_entry_side'] = np.select(
    [left_entry & is_final_third, middle_entry & is_final_third, right_entry & is_final_third],
    ['Left Channel', 'Center Channel', 'Right Channel'],
    default='Not Final Third')

def is_zone_14(row):
    return row['end_x'] >= 80 and row['end_x'] <= 98.54 and row['end_y'] >= 26.66 and row['end_y'] <= 53.32

event_df['is_zone_14'] = event_df.apply(is_zone_14, axis=1)

# ======================= PASS TYPE ENRICHMENT =======================
event_df['is_cross'] = event_df['pass_cross'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_cutback'] = event_df['pass_cut_back'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_switch'] = event_df['pass_switch'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_through_ball'] = event_df['pass_technique_name'].astype(str).str.contains('Through Ball', case=False, na=False)
event_df['is_goal_assist'] = event_df['pass_goal_assist'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_shot_assist'] = event_df['pass_shot_assist'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_deflected_pass'] = event_df['pass_deflected'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_dummy_pass'] = event_df['pass_no_touch'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_miscommunication'] = event_df['pass_miscommunication'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['is_long_pass'] = event_df['pass_pass_cluster_label'].astype(str).str.contains('Long', case=False, na=False)
event_df['is_high_pass'] = event_df['pass_height_name'] == 'High Pass'
event_df['is_ground_pass'] = event_df['pass_height_name'] == 'Ground Pass'
event_df['is_low_pass'] = event_df['pass_height_name'] == 'Low Pass'

def classify_pass_direction(angle):
    if pd.isna(angle): return 'unknown'
    if -85 <= angle <= 85: return 'forward'
    elif angle > 95 or angle < -95: return 'backward'
    else: return 'sideways'
event_df['pass_direction'] = event_df['pass_angle'].apply(classify_pass_direction)

# ==== Pass Body Part ====
event_df['pass_with_head'] = event_df['pass_body_part_name'] == 'Head'
event_df['pass_with_right_foot'] = event_df['pass_body_part_name'] == 'Right Foot'
event_df['pass_with_left_foot'] = event_df['pass_body_part_name'] == 'Left Foot'
event_df['pass_with_keeper_arm'] = event_df['pass_body_part_name'] == 'Keeper Arm'
event_df['pass_with_drop_kick'] = event_df['pass_body_part_name'] == 'Drop Kick'
event_df['pass_with_other_body'] = event_df['pass_body_part_name'] == 'Other'
event_df['pass_is_dummy'] = event_df['pass_body_part_name'] == 'No Touch'

# ==== Pass Type Category ====
event_df['pass_from_corner'] = event_df['pass_type_name'].astype(str).str.contains('Corner', case=False, na=False)
event_df['pass_from_free_kick'] = event_df['pass_type_name'].astype(str).str.contains('Free Kick', case=False, na=False)
event_df['pass_from_goal_kick'] = event_df['pass_type_name'].astype(str).str.contains('Goal Kick', case=False, na=False)
event_df['pass_from_throwin'] = event_df['pass_type_name'].astype(str).str.contains('Throw', case=False, na=False)
event_df['pass_from_open_play'] = event_df['pass_type_name'].astype(str).str.contains('Open Play', case=False, na=False)
event_df['pass_from_kickoff'] = event_df['pass_type_name'].astype(str).str.contains('Kick Off', case=False, na=False)
event_df['pass_from_interception'] = event_df['pass_type_name'].astype(str).str.contains('Interception', case=False, na=False)
event_df['pass_from_recovery'] = event_df['pass_type_name'].astype(str).str.contains('Recovery', case=False, na=False)
event_df['pass_is_first_time'] = event_df['pass_type_name'].astype(str).str.contains('First Time', case=False, na=False)

# ==== Pass Technique ====
event_df['pass_is_inswinging'] = event_df['pass_technique_name'].astype(str).str.contains('Inswinging', case=False, na=False)
event_df['pass_is_outswinging'] = event_df['pass_technique_name'].astype(str).str.contains('Outswinging', case=False, na=False)
event_df['pass_is_straight'] = event_df['pass_technique_name'].astype(str).str.contains('Straight', case=False, na=False)
event_df['pass_is_through_ball'] = event_df['pass_technique_name'].astype(str).str.contains('Through Ball', case=False, na=False)
# ======================= CARRY ENRICHMENT =======================
carry_mask = event_df['type_name'] == 'Carry'
event_df['is_progressive_carry'] = (event_df['prog_carry'] >= 9.25)
event_df['carry_length'] = np.where(carry_mask, np.sqrt((event_df['carry_end_x'] - event_df['x'])**2 + (event_df['carry_end_y'] - event_df['y'])**2), np.nan)
event_df['carry_into_final_third'] = (event_df['carry_end_x'] >= 80)
event_df['carry_into_penalty_box'] = (event_df['carry_end_x'] >= 88.5) & (event_df['carry_end_y'].between(16, 64.2))
event_df['carry_into_zone_14'] = (event_df['carry_end_x'] >= 80) & (event_df['carry_end_x'] <= 98.54) & (event_df['carry_end_y'] >= 26.66) & (event_df['carry_end_y'] <= 53.32)
event_df['carry_leads_to_shot'] = (event_df['type_name'] == 'Carry') & (event_df['shot_outcome_name'].shift(-1).isin(['Saved', 'Shot']))
event_df['carry_leads_to_goal'] = (event_df['type_name'] == 'Carry') & ((event_df['shot_outcome_name'].shift(-1) == 'Goal') | (event_df['pass_goal_assist'].shift(-1) == 'TRUE'))
event_df['carry_ends_in_dispossession'] = (event_df['type_name'] == 'Carry') & (event_df['type_name'].shift(-1) == 'Dispossessed')
# ======================= DRIBBLE METRICS =======================
dribble_df = event_df[event_df['type_name'] == 'Dribble']
dribble_stats = dribble_df.groupby('player_name')['dribble_outcome_name'].value_counts().unstack().fillna(0)
dribble_stats['dribble_attempted'] = dribble_stats.sum(axis=1)
dribble_stats['dribble_successful'] = dribble_stats.get('Complete', 0)
dribble_stats['dribble_failed'] = dribble_stats.get('Incomplete', 0)
dribble_stats['dribble_success_rate'] = np.where(dribble_stats['dribble_attempted'] > 0, (dribble_stats['dribble_successful'] / dribble_stats['dribble_attempted']) * 100, 0)
for stat in ['dribble_attempted', 'dribble_successful', 'dribble_failed', 'dribble_success_rate']:
    event_df = event_df.merge(dribble_stats[[stat]], left_on='player_name', right_index=True, how='left')
# ======================= SHOT BODY PART, TECHNIQUE & OUTCOME =======================
def classify_goal_zone(row):
    try:
        x, y, z = row['shot_end_x'], row['shot_end_y'], row['shot_end_z']
        if pd.isna(x) or pd.isna(y) or pd.isna(z): return 'Unknown'
        if x < 120: return 'Outside Goal'
        if 38 <= y <= 42:
            return 'Center High' if z > 1.5 else 'Center Low'
        if y < 40:
            return 'Top Left' if z > 1.5 else 'Bottom Left'
        if y > 40:
            return 'Top Right' if z > 1.5 else 'Bottom Right'
        return 'Unknown'
    except:
        return 'Unknown'

event_df['shot_goal_zone'] = event_df.apply(classify_goal_zone, axis=1)
shot_mask = event_df['type_name'] == 'Shot'

# Shot Body Part
event_df['shot_with_head'] = event_df['shot_body_part_name'] == 'Head'
event_df['shot_with_right_foot'] = event_df['shot_body_part_name'] == 'Right Foot'
event_df['shot_with_left_foot'] = event_df['shot_body_part_name'] == 'Left Foot'
event_df['shot_with_other_body'] = event_df['shot_body_part_name'] == 'Other'

# Shot Technique
techniques = ['Backheel', 'Diving Header', 'Half Volley', 'Lob', 'Normal', 'Overhead Kick', 'Volley']
for tech in techniques:
    col_name = f"shot_is_{tech.lower().replace(' ', '_').replace('-', '_')}"
    event_df[col_name] = event_df['shot_technique_name'].astype(str).str.contains(tech, case=False, na=False)

# Shot Outcome
outcomes = ['Blocked', 'Fail', 'Goal', 'Off T', 'Post', 'Saved', 'Saved Off T', 'Saved To Post', 'Wayward', 'Won']
for outcome in outcomes:
    col_name = f"shot_outcome_is_{outcome.lower().replace(' ', '_').replace('-', '_')}"
    event_df[col_name] = event_df['shot_outcome_name'].astype(str).str.lower() == outcome.lower()

# ======================= INTERCEPTION OUTCOME ENRICHMENT + GOALKEEPER =======================
interception_outcomes = [
    'Lost In Play', 'Lost Out', 'Success In Play', 'Success Out', 'Won'
]
for outcome in interception_outcomes:
    col_name = f"interception_outcome_is_{outcome.lower().replace(' ', '_').replace('-', '_')}"
    event_df[col_name] = event_df['interception_outcome_name'].astype(str).str.lower() == outcome.lower()

# ======================= GOALKEEPER ACTION ENRICHMENT =======================


# Goalkeeper Position
gk_positions = ['Moving', 'Prone', 'Set']
for pos in gk_positions:
    col = f"gk_position_is_{pos.lower()}"
    event_df[col] = event_df['goalkeeper_position_name'].astype(str).str.lower() == pos.lower()

# Goalkeeper Technique
gk_techniques = ['Diving', 'Standing']
for tech in gk_techniques:
    col = f"gk_technique_is_{tech.lower()}"
    event_df[col] = event_df['goalkeeper_technique_name'].astype(str).str.lower() == tech.lower()

# Goalkeeper Body Part
gk_parts = ['Both Hands', 'Chest', 'Head', 'Left Foot', 'Left Hand', 'Right Foot', 'Right Hand']
for part in gk_parts:
    col = f"gk_used_{part.lower().replace(' ', '_')}"
    event_df[col] = event_df['goalkeeper_body_part_name'].astype(str).str.lower() == part.lower()

# Goalkeeper Type
gk_types = [
    'Collected', 'Goal Conceded', 'Keeper Sweeper', 'Penalty Conceded', 'Penalty Saved',
    'Penalty Saved To Post', 'Punch', 'Save', 'Shot Faced', 'Shot Saved',
    'Shot Saved Off Target', 'Shot Saved To Post', 'Smother'
]
for typ in gk_types:
    col = f"gk_type_is_{typ.lower().replace(' ', '_')}"
    event_df[col] = event_df['goalkeeper_type_name'].astype(str).str.lower() == typ.lower()

# Goalkeeper Outcome
gk_outcomes = [
    'Claim', 'Clear', 'Fail', 'Lost In Play', 'Lost Out', 'No Touch', 'Punched Out',
    'Saved Twice', 'Second Effort', 'Success', 'Success In Play', 'Touched In',
    'Touched Out', 'Won'
]
for outcome in gk_outcomes:
    col = f"gk_outcome_is_{outcome.lower().replace(' ', '_')}"
    event_df[col] = event_df['goalkeeper_outcome_name'].astype(str).str.lower() == outcome.lower()

# ======================= FOUL WON ENRICHMENT =======================

event_df['foul_won_advantage'] = event_df['foul_won_advantage'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['foul_won_defensive'] = event_df['foul_won_defensive'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['foul_won_penalty'] = event_df['foul_won_penalty'].astype(str).str.contains('TRUE', case=False, na=False)

# ======================= FOUL COMMITTED ENRICHMENT =======================


# Foul committed: boolean flags
event_df['foul_committed_offensive'] = event_df['foul_committed_offensive'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['foul_committed_advantage'] = event_df['foul_committed_advantage'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['foul_committed_penalty'] = event_df['foul_committed_penalty'].astype(str).str.contains('TRUE', case=False, na=False)

# Foul committed: type enrichment
foul_types = [
    'Backpass Pick', 'Dangerous Play', 'Dive', 'Foul Out', 'Handball', 'Offside',
    'Open Play', 'Regular', 'Six Seconds'
]
for foul in foul_types:
    col = f"foul_type_is_{foul.lower().replace(' ', '_')}"
    event_df[col] = event_df['foul_committed_type_name'].astype(str).str.lower() == foul.lower()

# Foul committed: card enrichment
card_types = ['No Card', 'Red Card', 'Second Yellow', 'Yellow Card']
for card in card_types:
    col = f"foul_card_is_{card.lower().replace(' ', '_')}"
    event_df[col] = event_df['foul_committed_card_name'].astype(str).str.lower() == card.lower()

# ======================= DUEL ENRICHMENT =======================



# Duel types
duel_types = ['Aerial Lost', 'Tackle']
for duel_type in duel_types:
    col = f"duel_type_is_{duel_type.lower().replace(' ', '_')}"
    event_df[col] = event_df['duel_type_name'].astype(str).str.lower() == duel_type.lower()

# Duel outcomes
duel_outcomes = [
    'Lost', 'Won', 'Lost In Play', 'Lost Out', 'Success', 'Success In Play', 'Success Out'
]
for outcome in duel_outcomes:
    col = f"duel_outcome_is_{outcome.lower().replace(' ', '_').replace('-', '_')}"
    event_df[col] = event_df['duel_outcome_name'].astype(str).str.lower() == outcome.lower()

# ======================= FIFTY-FIFTY ENRICHMENT =======================

fifty_outcomes = [
    'Success To Opposition', 'Success To Team', 'Won'
]
for outcome in fifty_outcomes:
    col = f"fifty_outcome_is_{outcome.lower().replace(' ', '_')}"
    event_df[col] = event_df['50_50_outcome_name'].astype(str).str.lower() == outcome.lower()

# ======================= 50-50 ENRICHMENT =======================


fifty_fifty_outcomes = ['Success To Opposition', 'Success To Team', 'Won']
for outcome in fifty_fifty_outcomes:
    col = f"fifty_fifty_outcome_is_{outcome.lower().replace(' ', '_')}"
    event_df[col] = event_df['50_50_outcome_name'].astype(str).str.lower() == outcome.lower()

# ======================= DRIBBLE ENRICHMENT =======================

event_df['dribble_is_no_touch'] = event_df['dribble_no_touch'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['dribble_is_nutmeg'] = event_df['dribble_nutmeg'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['dribble_is_overrun'] = event_df['dribble_overrun'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['dribble_outcome_is_complete'] = event_df['dribble_outcome_name'].astype(str).str.lower() == 'complete'

# ======================= CLEARANCE ENRICHMENT =======================

clearance_parts = ['Head', 'Left Foot', 'Right Foot', 'Other']
for part in clearance_parts:
    col = f"clearance_with_{part.lower().replace(' ', '_')}"
    event_df[col] = event_df['clearance_body_part_name'].astype(str).str.lower() == part.lower()

# ======================= BLOCK ENRICHMENT =======================

event_df['block_is_deflection'] = event_df['block_deflection'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['block_is_offensive'] = event_df['block_offensive'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['block_is_save_block'] = event_df['block_save_block'].astype(str).str.contains('TRUE', case=False, na=False)

# ======================= BALL RECOVERY ENRICHMENT =======================

event_df['ball_recovery_is_offensive'] = event_df['ball_recovery_offensive'].astype(str).str.contains('TRUE', case=False, na=False)
event_df['ball_recovery_is_complete'] = event_df['ball_recovery_recovery_failure'].astype(str).str.lower() != 'TRUE'
event_df['ball_recovery_is_failure'] = event_df['ball_recovery_recovery_failure'].astype(str).str.contains('TRUE', case=False, na=False)

# ======================= BALL RECEIPT ENRICHMENT =======================

event_df['ball_receipt_is_complete'] = event_df['ball_receipt_outcome_name'].astype(str).str.lower() == 'complete'

# ======================= BAD BEHAVIOUR ENRICHMENT =======================

bad_behaviour_cards = ['No Card', 'Red Card', 'Second Yellow', 'Yellow Card']
for card in bad_behaviour_cards:
    col = f"bad_behaviour_card_is_{card.lower().replace(' ', '_')}"
    event_df[col] = event_df['bad_behaviour_card_name'].astype(str).str.lower() == card.lower()

# ======================= SUBSTITUTION ENRICHMENT =======================
event_df['substitution_has_replacement'] = event_df['substitution_replacement_id'].notna()
event_df['substitution_is_tactical_shift'] = event_df['substitution_outcome_name'].astype(str).str.contains('Tactical Shift', case=False, na=False)
event_df['substitution_formation'] = event_df['tactics_formation'].astype(str)

# ======================= (Rest of the code continues) =======================
output_path = f"match_{match_id}_TV.parquet"
event_df.to_parquet(output_path, index=False)
print(f"✅ Tactical Vectors fully enriched and saved to {output_path}")
