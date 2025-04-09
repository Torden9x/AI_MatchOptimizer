import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from mplsoccer import Pitch
import seaborn as sns

# Global styling variables
green = '#b7b943'
red = '#ff4b44'
blue = '#00a0de'
violet = '#a369ff'
bg_color = '#0d1117'
line_color = '#fafafa'
hcol = '#ff4b44'  # Home team color
acol = '#00a0de'  # Away team color
path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, hcol], N=20)
pearl_earring_cmapa = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, acol], N=20)
plt.rcParams['figure.facecolor'] = bg_color

def extract_team_names(df):
    """Extract home and away team names from Starting XI rows"""
    lineup_rows = df[df['type_name'] == 'Starting XI'].reset_index(drop=True)
    if len(lineup_rows) < 2:
        return "Unknown Home", "Unknown Away"
    
    hteamName = lineup_rows.loc[0, 'team_name']
    ateamName = lineup_rows.loc[1, 'team_name']
    return hteamName, ateamName

def extract_players_info(df):
    """Extract player information including starting XI and jersey numbers"""
    # Step 1: Get Starting XI rows
    lineup_rows = df[df['type_name'] == 'Starting XI'].reset_index(drop=True)
    
    # Step 2: Build a lookup dictionary {player_id: jersey_number}
    player_lineup_lookup = {}
    
    for _, row in lineup_rows.iterrows():
        lineup = ast.literal_eval(row['tactics_lineup'])  # safely parse list
        for player in lineup:
            pid = int(player['player.id'])
            jersey = player['jersey_number']
            player_lineup_lookup[pid] = jersey
    
    # Step 3: Define function to map values
    def get_starting_info(pid):
        try:
            pid = int(pid)
            if pid in player_lineup_lookup:
                return pd.Series({'isFirstEleven': True, 'jersey_number': player_lineup_lookup[pid]})
        except:
            pass
        return pd.Series({'isFirstEleven': False, 'jersey_number': None})
    
    # Step 4: Apply to the full DataFrame
    df[['isFirstEleven', 'jersey_number']] = df['player_id'].apply(get_starting_info)
    
    # Extract players data from lineups
    players_data = []
    
    for _, row in lineup_rows.iterrows():
        team_lineup = ast.literal_eval(row['tactics_lineup'])
        for player in team_lineup:
            players_data.append({
                'player_id': int(player['player.id']),
                'player_name': player['player.name'],
                'jersey_number': player['jersey_number'],
                'position_name': player['position.name'],
                'isFirstEleven': True,
                'team_name': row['team_name']
            })
    
    # Create players_df
    players_df = pd.DataFrame(players_data)
    
    return players_df, df

def get_passes_df(df):
    """Create passes_df from full match dataframe"""
    df_filtered = df[~df['type_name'].str.contains(
        'Starting XI|FormationChange|FormationSet|Card|Substitution|Player On|Bad Behaviour|Player Off|Half Start|Half End', 
        na=False)]
    return df_filtered[df_filtered['type_name'] == 'Pass'].copy()

def get_passes_between_df(team_name, passes_df, players_df, df):
    """Calculate passes between players for pass network"""
    passes_df = passes_df[passes_df["team_name"] == team_name].copy()
    dfteam = df[(df['team_name'] == team_name) & (~df['type_name'].str.contains(
        'Starting XI|FormationChange|FormationSet|Card|Substitution|Player On|Bad Behaviour|Player Off|Half Start|Half End',
        na=False))]

    passes_df = passes_df.merge(players_df[["player_id"]], on="player_id", how='left')

    average_locs_and_count_df = dfteam.groupby("player_id").agg({
        'x': 'median', 'y': ['median', 'count']
    })
    average_locs_and_count_df.columns = ['pass_avg_x', 'pass_avg_y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(
        players_df[["player_id", 'jersey_number', "position_name", 'isFirstEleven']],
        on="player_id", how='left'
    ).set_index("player_id")

    passes_player_ids_df = passes_df.loc[:, ['index', "player_id", "pass_recipient_id", 'team_name']].copy()
    passes_player_ids_df.dropna(subset=["player_id", "pass_recipient_id"], inplace=True)
    passes_player_ids_df["player_id"] = passes_player_ids_df["player_id"].astype(int)
    passes_player_ids_df["pass_recipient_id"] = passes_player_ids_df["pass_recipient_id"].astype(int)

    passes_player_ids_df['pos_min'] = passes_player_ids_df[['player_id', 'pass_recipient_id']].min(axis=1)
    passes_player_ids_df['pos_max'] = passes_player_ids_df[['player_id', 'pass_recipient_id']].max(axis=1)

    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max'])['index'].count().reset_index()
    passes_between_df.rename(columns={'index': 'pass_count'}, inplace=True)

    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True,
                                            suffixes=['', '_end'])

    # Get all unique player_id + name combinations
    all_players_info = df[['player_id', 'player_name']].dropna().drop_duplicates()

    return passes_between_df, average_locs_and_count_df, all_players_info

def pass_network_visualization(team_name, passes_between_df, average_locs_and_count_df, col, team_is_away, all_players_info,passes_df):
    """Create pass network visualization"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    
    MAX_LINE_WIDTH = 15
    MIN_TRANSPARENCY = 0.05
    MAX_TRANSPARENCY = 0.85

    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() * MAX_LINE_WIDTH)
    color = np.array(to_rgba(col))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 120)

    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y,
                passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
                lw=passes_between_df.width, color=color, zorder=1, ax=ax)

    average_locs_and_count_df = average_locs_and_count_df.reset_index().merge(
        all_players_info, on='player_id', how='left'
    ).set_index('player_id')

    for index, row in average_locs_and_count_df.iterrows():
        if row.get('isFirstEleven') == True and pd.notna(row.get("jersey_number")):
            label = int(row["jersey_number"])
            marker = 'o'
        elif pd.notna(row.get("player_name", None)):
            name_parts = str(row["player_name"]).split()
            label = '. '.join([n[0] for n in name_parts[:-1]]) + f' {name_parts[-1]}' if len(name_parts) > 1 else name_parts[0]
            marker = 's'
        else:
            label = "?"
            marker = 's'

        pitch.scatter(row.pass_avg_x, row.pass_avg_y, s=1000, marker=marker, color=bg_color,
                    edgecolor=line_color, linewidth=2, ax=ax)
        pitch.annotate(label, xy=(row.pass_avg_x, row.pass_avg_y), c=col, ha='center', va='center', size=8, ax=ax)

    avgph = round(average_locs_and_count_df['pass_avg_x'].median(), 2)
    ax.axvline(x=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)

    try:
        def_line_h = round(
            average_locs_and_count_df[average_locs_and_count_df["position_name"]
            .str.contains("Center Back", case=False, na=False)]['pass_avg_x'].median(), 2)
        ax.axvline(x=def_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)
    except:
        def_line_h = 0

    try:
        fwd_line_h = round(average_locs_and_count_df[average_locs_and_count_df.get('isFirstEleven') == 1]
                        .sort_values(by='pass_avg_x', ascending=False)
                        .head(2)['pass_avg_x'].mean(), 2)
        ax.axvline(x=fwd_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)
        ax.fill([def_line_h, fwd_line_h, fwd_line_h, def_line_h], [0, 0, 68, 68], col, alpha=0.1)
    except:
        fwd_line_h = 0

    # Calculate verticality if possible
    verticality = 0
    try:
        team_passes_df = passes_df[passes_df["team_name"] == team_name].copy()
        if 'pass_or_carry_angle' in team_passes_df.columns:
            team_passes_df['pass_or_carry_angle'] = team_passes_df['pass_or_carry_angle'].abs()
            team_passes_df = team_passes_df[(team_passes_df['pass_or_carry_angle'] >= 0) & 
                                           (team_passes_df['pass_or_carry_angle'] <= 90)]
            verticality = round((1 - team_passes_df['pass_or_carry_angle'].median() / 90) * 100, 2)
    except:
        verticality = 0

    # Get top pass combination
    try:
        passes_between_df = passes_between_df.merge(
            all_players_info.rename(columns={"player_id": "pos_min", "player_name": "player_name"}), 
            on="pos_min", how="left"
        )
        passes_between_df = passes_between_df.merge(
            all_players_info.rename(columns={"player_id": "pos_max", "player_name": "player_name_end"}), 
            on="pos_max", how="left"
        )

        top_pass = passes_between_df.sort_values(by='pass_count', ascending=False).head(1).reset_index(drop=True)
        most_pass_from = top_pass['player_name'][0] if 'player_name' in top_pass.columns else "?"
        most_pass_to = top_pass['player_name_end'][0] if 'player_name_end' in top_pass.columns else "?"
        most_pass_count = top_pass['pass_count'][0]
    except:
        most_pass_from = "?"
        most_pass_to = "?"
        most_pass_count = 0

    if team_is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(avgph-1, 81, f"{avgph}m", fontsize=12, color=line_color, ha='left')
        ax.text(120, 81, f"verticality: {verticality}%", fontsize=12, color=line_color, ha='left')
    else:
        ax.text(avgph-1, -1, f"{avgph}m", fontsize=12, color=line_color, ha='right')
        ax.text(120, -1, f"verticality: {verticality}%", fontsize=12, color=line_color, ha='right')

    ax.text(2, 81 if not team_is_away else -1, "circle = starter\nbox = sub",
            color=col, size=12, ha='left' if not team_is_away else 'right', va='top')
    ax.set_title(f"{team_name}\nPassing Network", color=line_color, size=20, fontweight='bold')

    return fig, {
        'Team_Name': team_name,
        'Defense_Line_Height': def_line_h,
        'Vericality_%': verticality,
        'Most_pass_combination_from': most_pass_from,
        'Most_pass_combination_to': most_pass_to,
        'Most_passes_in_combination': most_pass_count,
    }

def get_defensive_action_df(df):
    """Extract defensive actions from match data"""
    playerdf = df.copy()
    
    # Ball Wins
    ball_wins = playerdf[
        (playerdf["type_name"] == 'Interception') |
        (playerdf["type_name"] == 'Ball Recovery')
    ]

    # Tackles (total and unsuccessful)
    tk = playerdf[
        (playerdf["type_name"] == 'Duel') &
        (playerdf['duel_type_name'].str.contains('Tackle', case=False, na=False))
    ]

    tk_u = playerdf[
        (playerdf["type_name"] == 'Duel') &
        (playerdf['duel_type_name'].str.contains('Tackle', case=False, na=False)) &
        (playerdf['duel_outcome_name'].isin(['Lost In Play', 'Lost']))
    ]

    # Interceptions
    intc = playerdf[
        (playerdf['type_name'] == 'Interception') &
        (playerdf['interception_outcome_name'].isin(['Success In Play', 'Won']))
    ]

    # Ball Recoveries (excluding recovery failures)
    br = playerdf[
        (playerdf['type_name'] == 'Ball Recovery') &
        (playerdf['ball_recovery_recovery_failure'].astype(str) != 'True')
    ]

    # Clearances
    cl = playerdf[playerdf["type_name"] == 'Clearance']

    # Fouls
    fl = playerdf[playerdf["type_name"] == 'Foul Committed']

    # Aerial Duels (total and unsuccessful)
    ar = playerdf[
        (playerdf["type_name"] == 'Duel') &
        (playerdf['duel_type_name'].str.contains('Aerial', case=False, na=False))
    ]

    ar_u = playerdf[
        (playerdf["type_name"] == 'Duel') &
        (playerdf['duel_type_name'].str.contains('Aerial', case=False, na=False)) &
        (playerdf['duel_outcome_name'] == 'Aerial Lost')
    ]

    # Blocks
    pass_bl = playerdf[playerdf["type_name"] == 'Block']

    shot_bl = playerdf[
        (playerdf["type_name"] == 'Block') &
        (playerdf['block_save_block'].astype(str).str.contains('TRUE', case=False, na=False))
    ]

    # Dribbled Past
    drb_pst = playerdf[playerdf["type_name"] == 'Dribbled Past']

    # Dribble leading to tackle loss
    drb_tkl = df[
        (df["type_name"] == 'Duel') &
        (df['duel_type_name'].str.contains('Tackle', case=False, na=False)) &
        (df["duel_outcome_name"].isin(['Lost In Play', 'Lost'])) &
        (df["type_name"].shift(1) == 'Dribble')
    ]

    # Defensive Errors
    errors = df[ (
            ((df['type_name'] == 'Duel') &
             (df['duel_type_name'].str.contains('Tackle|Aerial', case=False, na=False)) &
             (df['duel_outcome_name'].isin(['Lost In Play', 'Lost', 'Aerial Lost']))) |
            (df['type_name'] == 'Foul Committed')
        )
    ]

    return pd.concat([
        ball_wins, tk, tk_u, intc, br, cl, fl, ar, ar_u, 
        pass_bl, shot_bl, drb_pst, drb_tkl, errors
    ], ignore_index=True)

def get_da_count_df(team_name, defensive_actions_df, players_df):
    """Calculate defensive action locations and counts by player"""
    defensive_actions_df = defensive_actions_df[defensive_actions_df['team_name'] == team_name]
    # Add column with first eleven players only
    defensive_actions_df = defensive_actions_df.merge(players_df[['player_id', "isFirstEleven"]], on='player_id', how='left')
    
    # Calculate mean positions for players
    average_locs_and_count_df = (defensive_actions_df.groupby('player_id').agg({'x': ['median'], 'y': ['median', 'count']}))
    average_locs_and_count_df.columns = ['x', 'y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(
        players_df[['player_id', 'player_name', 'jersey_number', 'position_name', 'isFirstEleven']], 
        on='player_id', how='left'
    )
    average_locs_and_count_df = average_locs_and_count_df.set_index('player_id')

    return average_locs_and_count_df

def defensive_block(team_name, average_locs_and_count_df, team_is_away, defensive_actions_df, col):
    """Create defensive block visualization"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    
    defensive_actions_team_df = defensive_actions_df[defensive_actions_df['team_name'] == team_name]
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=line_color, linewidth=2, 
                  line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_facecolor(bg_color)
    ax.set_xlim(-0.5, 120.5)

    # Using variable marker size for each player according to their defensive engagements
    MAX_MARKER_SIZE = 3500
    # Check if dataframe is not empty
    if not average_locs_and_count_df.empty:
        average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count'] / 
                                                  average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)
        
        # Create a defensive heatmap using KDE
        try:
            flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors", [bg_color, col], N=500)
            if 'x' in defensive_actions_team_df.columns and 'y' in defensive_actions_team_df.columns:
                kde = pitch.kdeplot(
                    defensive_actions_team_df.x, defensive_actions_team_df.y, 
                    ax=ax, fill=True, levels=5000, thresh=0.02, cut=4, cmap=flamingo_cmap
                )
        except Exception as e:
            print(f"KDE plot error: {e}")

        # Plot player positions
        average_locs_and_count_df = average_locs_and_count_df.reset_index(drop=True)
        for index, row in average_locs_and_count_df.iterrows():
            marker_size = row.get('marker_size', 500)  # Default size if not available
            if row.get('x') is not None and row.get('y') is not None:
                if row.get('isFirstEleven') == True:
                    da_nodes = pitch.scatter(
                        row['x'], row['y'], s=marker_size+100, marker='o', 
                        color=bg_color, edgecolor=line_color, linewidth=1, 
                        alpha=1, zorder=3, ax=ax
                    )
                else:
                    da_nodes = pitch.scatter(
                        row['x'], row['y'], s=marker_size+100, marker='s', 
                        color=bg_color, edgecolor=line_color, linewidth=1, 
                        alpha=1, zorder=3, ax=ax
                    )
                
                # Plot jersey number
                if pd.notna(row.get('jersey_number')):
                    pitch.annotate(
                        row['jersey_number'], xy=(row['x'], row['y']), 
                        c=line_color, ha='center', va='center', size=14, ax=ax
                    )
                if pd.notna(row.get("player_name", None)):
                    name_parts = str(row["player_name"]).split()
                    label = '. '.join([n[0] for n in name_parts[:-1]]) + f' {name_parts[-1]}' if len(name_parts) > 1 else name_parts[0]
                    marker = 's'
                    

        # Plot all defensive actions as tiny points
        if 'x' in defensive_actions_team_df.columns and 'y' in defensive_actions_team_df.columns:
            da_scatter = pitch.scatter(
                defensive_actions_team_df.x, defensive_actions_team_df.y, 
                s=10, marker='x', color='yellow', alpha=0.2, ax=ax
            )

        # Calculate and show defensive metrics
        try:
            # Defensive Actions Height
            dah = round(average_locs_and_count_df['x'].mean(), 2)
            dah_show = round((dah*1.20), 2)
            ax.axvline(x=dah, color='gray', linestyle='--', alpha=0.75, linewidth=2)

            # Defense line Height
            center_backs_height = average_locs_and_count_df[
                average_locs_and_count_df["position_name"].isin(['Center Back', 'Left Center Back', 'Right Center Back'])
            ]                
            def_line_h = round(center_backs_height['x'].median(), 2) if not center_backs_height.empty else 0
            if def_line_h > 0:
                ax.axvline(x=def_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)
            
            # Forward line Height
            Forwards_height = average_locs_and_count_df[average_locs_and_count_df['isFirstEleven'] == 1]
            Forwards_height = Forwards_height.sort_values(by='x', ascending=False)
            Forwards_height = Forwards_height.head(2)
            fwd_line_h = round(Forwards_height['x'].mean(), 2) if not Forwards_height.empty else 0
            if fwd_line_h > 0:
                ax.axvline(x=fwd_line_h, color='gray', linestyle='dotted', alpha=0.5, linewidth=2)

            # Calculate compactness
            if def_line_h > 0 and fwd_line_h > 0:
                compactness = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)
            else:
                compactness = 0
        except Exception as e:
            print(f"Error calculating defensive metrics: {e}")
            dah = 0
            dah_show = 0
            def_line_h = 0
            fwd_line_h = 0
            compactness = 0

    else:
        dah = 0
        dah_show = 0
        def_line_h = 0
        fwd_line_h = 0
        compactness = 0

    # Display metrics and labels
    if team_is_away:
        # Inverting the axis for away team
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(dah-1, 81, f"{dah_show}m", fontsize=15, color=line_color, ha='left', va='center')
        ax.text(105, 81, f'Compact:{compactness}%', fontsize=15, color=line_color, ha='left', va='center')
        ax.text(2, -2, "circle = starter\nbox = sub", color='gray', size=12, ha='right', va='top')
    else:
        ax.text(dah-1, -1, f"{dah_show}m", fontsize=15, color=line_color, ha='right', va='center')
        ax.text(105, -1, f'Compact:{compactness}%', fontsize=15, color=line_color, ha='right', va='center')
        ax.text(2, 81, "circle = starter\nbox = sub", color='gray', size=12, ha='left', va='top')

    ax.set_title(f"{team_name}\nDefensive Action Heatmap", color=line_color, fontsize=25, fontweight='bold')

    return fig, {
        'Team_Name': team_name,
        'Average_Defensive_Action_Height': dah,
        'Forward_Line_Pressing_Height': fwd_line_h,
        'Compactness': compactness
    }

def draw_progressive_pass_map(team_name, df, team_is_away, col,ateamName,hteamName):
    """Create progressive pass visualization"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
    
    # Filter for progressive passes
    dfpro = df[(df['team_name']==team_name) & 
               (df['prog_pass']>=9.25) & 
               (~df['play_pattern_name'].astype(str).str.contains('From Corner|From Goal Kick', case=False, na=False)) & 
               (df['x']>=40)]
    
    if 'pass_outcome_name' in df.columns:
        dfpro = dfpro[dfpro['pass_outcome_name']=='1']
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=line_color, linewidth=2,
                  corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 120.5)

    if team_is_away:
        ax.invert_xaxis()
        ax.invert_yaxis()

    pro_count = len(dfpro)

    # Calculate zone percentages
    left_pro = len(dfpro[dfpro['y']>=54.33]) if not dfpro.empty else 0
    mid_pro = len(dfpro[(dfpro['y']>=30.67) & (dfpro['y']<54.33)]) if not dfpro.empty else 0
    right_pro = len(dfpro[(dfpro['y']>=0) & (dfpro['y']<30.67)]) if not dfpro.empty else 0
    
    left_percentage = round((left_pro/pro_count)*100) if pro_count > 0 else 0
    mid_percentage = round((mid_pro/pro_count)*100) if pro_count > 0 else 0
    right_percentage = round((right_pro/pro_count)*100) if pro_count > 0 else 0

    # Draw zone dividing lines
    ax.hlines(22.67, xmin=0, xmax=120, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=120, colors=line_color, linestyle='dashed', alpha=0.35)

    # Display zone stats
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=col, fontsize=24, 
            va='center', ha='center', bbox=bbox_props)
    ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=col, fontsize=24, 
            va='center', ha='center', bbox=bbox_props)
    ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=col, fontsize=24, 
            va='center', ha='center', bbox=bbox_props)

    # Plot the passes
    if not dfpro.empty and 'end_x' in dfpro.columns and 'end_y' in dfpro.columns:
        pro_pass = pitch.lines(dfpro.x, dfpro.y, dfpro.end_x, dfpro.end_y, 
                              lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
        # Plot end points
        pro_pass_end = pitch.scatter(dfpro.end_x, dfpro.end_y, s=35, 
                                    edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)

    counttext = f"{pro_count} Progressive Carries"
    display_team = hteamName if col == hcol else ateamName
    ax.set_title(f"{display_team}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')

    # Return statistics
    return fig,{
        'Team_Name': team_name,
        'Total_Progressive_Carries': pro_count,
        'Progressive_Carries_From_Left': left_pro,
        'Progressive_Carries_From_Center': mid_pro,
        'Progressive_Carries_From_Right': right_pro
    }

def plot_shotmap(ax,df,hteamName,ateamName):
    """Create shot map visualization and return shooting statistics"""
    # Step 1: Select shot events + own goals
    # Filter all shot events + own goals
    shots_df = df[
        (df['type_name'] == 'Shot') |
        (df['type_name'].astype(str).str.contains('Own Goal Against', na=False)) |
        (df['type_name'].astype(str).str.contains('Own Goal For', na=False))
    ]

    # Step 2: Filter only meaningful outcomes
    shot_outcomes = ['Goal', 'Off T', 'Saved', 'Blocked', 'Post']
    if 'shot_outcome_name' in shots_df.columns:
        mask4 = shots_df['shot_outcome_name'].isin(shot_outcomes)
        Shotsdf = shots_df[mask4].reset_index(drop=True)
    else:
        # Fallback for different column naming
        mask4 = shots_df['shot_outcome'].isin(shot_outcomes) if 'shot_outcome' in shots_df.columns else pd.Series([True] * len(shots_df))
        Shotsdf = shots_df[mask4].reset_index(drop=True)

    # Step 3: Split by team
    hShotsdf = Shotsdf[Shotsdf['team_name'] == hteamName]
    aShotsdf = Shotsdf[Shotsdf['team_name'] == ateamName]

    # Step 4: Filter by outcome
    outcome_column = 'shot_outcome_name' if 'shot_outcome_name' in Shotsdf.columns else 'shot_outcome'
    hSavedf = hShotsdf[hShotsdf[outcome_column] == 'Saved']
    aSavedf = aShotsdf[aShotsdf[outcome_column] == 'Saved']

    # Step 5: Own goals only
    hogdf = hShotsdf[hShotsdf['type_name'].astype(str).str.contains('Own Goal For', na=False)]
    aogdf = aShotsdf[aShotsdf['type_name'].astype(str).str.contains('Own Goal For', na=False)]

    # Get goals and xG
    hgoal_count = len(hShotsdf[(hShotsdf['team_name']==hteamName) & (hShotsdf[outcome_column]=='Goal')])
    agoal_count = len(aShotsdf[(aShotsdf['team_name']==ateamName) & (aShotsdf[outcome_column]=='Goal')])
    
    # Handle different xG column names
    xg_column = 'shot_statsbomb_xg' if 'shot_statsbomb_xg' in df.columns else 'xg'
    xgot_column = 'shot_gk_save_difficulty_xg' if 'shot_gk_save_difficulty_xg' in df.columns else 'xgot'
    
    hxg = hShotsdf[xg_column].sum().round(2) if xg_column in hShotsdf.columns else 0
    axg = aShotsdf[xg_column].sum().round(2) if xg_column in aShotsdf.columns else 0
    hxgot = hShotsdf[xgot_column].sum().round(2) if xgot_column in hShotsdf.columns else 0
    axgot = aShotsdf[xgot_column].sum().round(2) if xgot_column in aShotsdf.columns else 0

    # Center Goal point for distance calculations
    given_point = (60, 40)
    
    # Calculate shot distances
    
        # Try with expected column names
    home_shot_distances = np.sqrt((hShotsdf['Shoter_x'] - given_point[0])**2 + (hShotsdf['Shoter_y'] - given_point[1])**2)
    away_shot_distances = np.sqrt((aShotsdf['Shoter_x'] - given_point[0])**2 + (aShotsdf['Shoter_y'] - given_point[1])**2)

    home_average_shot_distance = round(home_shot_distances.mean(), 2) if len(home_shot_distances) > 0 else 0
    away_average_shot_distance = round(away_shot_distances.mean(), 2) if len(away_shot_distances) > 0 else 0

    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, linewidth=2, line_color=line_color)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 80.5)
    ax.set_xlim(-0.5, 120.5)

    # Calculate shooting stats
    hTotalShots = len(hShotsdf)
    aTotalShots = len(aShotsdf)
    hShotsOnT = len(hSavedf) + hgoal_count
    aShotsOnT = len(aSavedf) + agoal_count
    hxGpSh = round(hxg / hTotalShots, 2) if hTotalShots > 0 else 0
    axGpSh = round(axg / aTotalShots, 2) if aTotalShots > 0 else 0

    # --- HOME TEAM ---
    # Filter shots by outcome
    hGoalData = hShotsdf[(hShotsdf[outcome_column] == 'Goal')]
    hPostData = hShotsdf[(hShotsdf[outcome_column] == 'Post')]
    hSaveData = hShotsdf[(hShotsdf[outcome_column] == 'Saved')]
    hMissData = hShotsdf[(hShotsdf[outcome_column] == 'Off T')]

    # Filter big chances (high xG)
    big_chance_threshold = 0.2
    Big_C_hGoalData = hShotsdf[
        (hShotsdf[outcome_column] == 'Goal') &
        (hShotsdf[xg_column] >= big_chance_threshold)
    ] if xg_column in hShotsdf.columns else pd.DataFrame()
    
    Big_C_hPostData = hShotsdf[
        (hShotsdf[outcome_column] == 'Post') &
        (hShotsdf[xg_column] >= big_chance_threshold)
    ] if xg_column in hShotsdf.columns else pd.DataFrame()
    
    Big_C_hSaveData = hShotsdf[
        (hShotsdf[outcome_column] == 'Saved') &
        (hShotsdf[xg_column] >= big_chance_threshold)
    ] if xg_column in hShotsdf.columns else pd.DataFrame()
    
    Big_C_hMissData = hShotsdf[
        (hShotsdf[outcome_column] == 'Off T') &
        (hShotsdf[xg_column] >= big_chance_threshold)
    ] if xg_column in hShotsdf.columns else pd.DataFrame()

    total_bigC_home = len(Big_C_hGoalData) + len(Big_C_hPostData) + len(Big_C_hSaveData) + len(Big_C_hMissData)
    bigC_miss_home = len(Big_C_hPostData) + len(Big_C_hSaveData) + len(Big_C_hMissData)

  
    pitch.scatter((120 - hPostData.Shoter_x), (80 - hPostData.Shoter_y), s=200, edgecolors=hcol, c=hcol, marker='o', ax=ax)
    pitch.scatter((120 - hSaveData.Shoter_x), (80 - hSaveData.Shoter_y), s=200, edgecolors=hcol, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter((120 - hMissData.Shoter_x), (80 - hMissData.Shoter_y), s=200, edgecolors=hcol, c='None', marker='o', ax=ax)
    pitch.scatter((120 - hGoalData.Shoter_x), (80 - hGoalData.Shoter_y), s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
    pitch.scatter((120 - hogdf.Shoter_x), (80 - hogdf.Shoter_y), s=350, edgecolors='orange', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)

    pitch.scatter((120 - Big_C_hPostData.Shoter_x), (80 - Big_C_hPostData.Shoter_y), s=500, edgecolors=hcol, c=hcol, marker='o', ax=ax)
    pitch.scatter((120 - Big_C_hSaveData.Shoter_x), (80 - Big_C_hSaveData.Shoter_y), s=500, edgecolors=hcol, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter((120 - Big_C_hMissData.Shoter_x), (80 - Big_C_hMissData.Shoter_y), s=500, edgecolors=hcol, c='None', marker='o', ax=ax)
    pitch.scatter((120 - Big_C_hGoalData.Shoter_x), (80 - Big_C_hGoalData.Shoter_y), s=650, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)
     # --- AWAY TEAM ---
    # Non-Big Chances (lower xG)
    aGoalData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Goal') &
        (aShotsdf['shot_statsbomb_xg'] < 0.3)
    ]

    aPostData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Post') &
        (aShotsdf['shot_statsbomb_xg'] < 0.3)
    ]

    aSaveData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Saved') &
        (aShotsdf['shot_statsbomb_xg'] < 0.3)
    ]

    aMissData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Off T') &
        (aShotsdf['shot_statsbomb_xg'] < 0.3)
    ]

# Big Chances (using xG threshold)
    Big_C_aGoalData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Goal') &
        (aShotsdf['shot_statsbomb_xg'] >= 0.3)
    ]

    Big_C_aPostData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Post') &
        (aShotsdf['shot_statsbomb_xg'] >= 0.3)
    ]

    Big_C_aSaveData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Saved') &
        (aShotsdf['shot_statsbomb_xg'] >= 0.3)
    ]

    Big_C_aMissData = aShotsdf[
        (aShotsdf['shot_outcome_name'] == 'Off T') &
        (aShotsdf['shot_statsbomb_xg'] >= 0.3)
    ]

# Totals
    total_bigC_away = (
        len(Big_C_aGoalData) +
        len(Big_C_aPostData) +
        len(Big_C_aSaveData) +
        len(Big_C_aMissData)
    )

    bigC_miss_away = (
        len(Big_C_aPostData) +
        len(Big_C_aSaveData) +
        len(Big_C_aMissData)
    )


    # normal shots scatter of away team
    pitch.scatter(aPostData.Shoter_x, aPostData.Shoter_y, s=200, edgecolors=acol, c=acol, marker='o', ax=ax)
    pitch.scatter(aSaveData.Shoter_x, aSaveData.Shoter_y, s=200, edgecolors=acol, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter(aMissData.Shoter_x, aMissData.Shoter_y, s=200, edgecolors=acol, c='None', marker='o', ax=ax)
    pitch.scatter(aGoalData.Shoter_x, aGoalData.Shoter_y, s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
    pitch.scatter(aogdf.Shoter_x, aogdf.Shoter_y, s=350, edgecolors='orange', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)

    pitch.scatter(Big_C_aPostData.Shoter_x, Big_C_aPostData.Shoter_y, s=700, edgecolors=acol, c=acol, marker='o', ax=ax)
    pitch.scatter(Big_C_aSaveData.Shoter_x, Big_C_aSaveData.Shoter_y, s=700, edgecolors=acol, c='None', hatch='///////', marker='o', ax=ax)
    pitch.scatter(Big_C_aMissData.Shoter_x, Big_C_aMissData.Shoter_y, s=700, edgecolors=acol, c='None', marker='o', ax=ax)
    pitch.scatter(Big_C_aGoalData.Shoter_x, Big_C_aGoalData.Shoter_y, s=850, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)

    # --- NORMALIZED STATS BAR ---
    if hgoal_count + agoal_count == 0:
        hgoal = agoal = 10
    else:
        hgoal = (hgoal_count / (hgoal_count + agoal_count)) * 20
        agoal = (agoal_count / (hgoal_count + agoal_count)) * 20

    if total_bigC_home + total_bigC_away == 0:
        total_bigC_home_n = total_bigC_away_n = 10
    else:
        total_bigC_home_n = (total_bigC_home / (total_bigC_home + total_bigC_away)) * 20
        total_bigC_away_n = (total_bigC_away / (total_bigC_home + total_bigC_away)) * 20

    if bigC_miss_home + bigC_miss_away == 0:
        bigC_miss_home_n = bigC_miss_away_n = 10
    else:
        bigC_miss_home_n = (bigC_miss_home / (bigC_miss_home + bigC_miss_away)) * 20
        bigC_miss_away_n = (bigC_miss_away / (bigC_miss_home + bigC_miss_away)) * 20

    if hShotsOnT + aShotsOnT == 0:
        hShotsOnT_n = aShotsOnT_n = 10
    else:
        hShotsOnT_n = (hShotsOnT / (hShotsOnT + aShotsOnT)) * 20
        aShotsOnT_n = (aShotsOnT / (hShotsOnT + aShotsOnT)) * 20

    if hxgot + axgot == 0:
        hxgot_n = axgot_n = 10
    else:
        hxgot_n = (hxgot / (hxgot + axgot)) * 20
        axgot_n = (axgot / (hxgot + axgot)) * 20

    shooting_stats_title = [62, 62-(1*7), 62-(2*7), 62-(3*7), 62-(4*7), 62-(5*7), 62-(6*7), 62-(7*7), 62-(8*7)]
    shooting_stats_home = [hgoal_count, hxg, hxgot, hTotalShots, hShotsOnT, hxGpSh, total_bigC_home, bigC_miss_home, home_average_shot_distance]
    shooting_stats_away = [agoal_count, axg, axgot, aTotalShots, aShotsOnT, axGpSh, total_bigC_away, bigC_miss_away, away_average_shot_distance]

    shooting_stats_normalized_home = [
        hgoal,
        (hxg / (hxg + axg)) * 20,
        hxgot_n,
        (hTotalShots / (hTotalShots + aTotalShots)) * 20,
        hShotsOnT_n,
        total_bigC_home_n,
        bigC_miss_home_n,
        (hxGpSh / (hxGpSh + axGpSh)) * 20,
        (home_average_shot_distance / (home_average_shot_distance + away_average_shot_distance)) * 20
    ]

    shooting_stats_normalized_away = [
        agoal,
        (axg / (hxg + axg)) * 20,
        axgot_n,
        (aTotalShots / (hTotalShots + aTotalShots)) * 20,
        aShotsOnT_n,
        total_bigC_away_n,
        bigC_miss_away_n,
        (axGpSh / (hxGpSh + axGpSh)) * 20,
        (away_average_shot_distance / (home_average_shot_distance + away_average_shot_distance)) * 20
    ]

    start_x = 42.5
    start_x_for_away = [x + 42.5 for x in shooting_stats_normalized_home]

    ax.barh(shooting_stats_title, shooting_stats_normalized_home, height=5, color=hcol, left=start_x)
    ax.barh(shooting_stats_title, shooting_stats_normalized_away, height=5, left=start_x_for_away, color=acol)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # plotting the texts
    ax.text(52.5, 62, "Goals", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(1*7), "xG", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(2*7), "xGOT", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(3*7), "Shots", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(4*7), "On Target", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(5*7), "BigChance", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(6*7), "BigC.Miss", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(7*7), "xG/Shot", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(52.5, 62-(8*7), "Avg.Dist.", color=bg_color, fontsize=12, ha='center', va='center', fontweight='bold')

    ax.text(41.5, 62, f"{hgoal_count}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(1*7), f"{hxg}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(2*7), f"{hxgot}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(3*7), f"{hTotalShots}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(4*7), f"{hShotsOnT}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(5*7), f"{total_bigC_home}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(6*7), f"{bigC_miss_home}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(7*7), f"{hxGpSh}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
    ax.text(41.5, 62-(8*7), f"{home_average_shot_distance}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')

    ax.text(63.5, 62, f"{agoal_count}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(1*7), f"{axg}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(2*7), f"{axgot}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(3*7), f"{aTotalShots}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(4*7), f"{aShotsOnT}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(5*7), f"{total_bigC_away}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(6*7), f"{bigC_miss_away}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(7*7), f"{axGpSh}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
    ax.text(63.5, 62-(8*7), f"{away_average_shot_distance}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')

    # Heading and other texts
    ax.text(0, 70, f"{hteamName}\n<---shots", color=hcol, size=20, ha='left', fontweight='bold')
    ax.text(120, 70, f"{ateamName}\nshots--->", color=acol, size=20, ha='right', fontweight='bold')

    home_data = {
        'Team_Name': hteamName,
        'Goals_Scored': hgoal_count,
        'xG': hxg,
        'xGOT': hxgot,
        'Total_Shots': hTotalShots,
        'Shots_On_Target': hShotsOnT,
        'BigChances': total_bigC_home,
        'BigChances_Missed': bigC_miss_home,
        'xG_per_Shot': hxGpSh,
        'Average_Shot_Distance': home_average_shot_distance
    }
    
    away_data = {
        'Team_Name': ateamName,
        'Goals_Scored': agoal_count,
        'xG': axg,
        'xGOT': axgot,
        'Total_Shots': aTotalShots,
        'Shots_On_Target': aShotsOnT,
        'BigChances': total_bigC_away,
        'BigChances_Missed': bigC_miss_away,
        'xG_per_Shot': axGpSh,
        'Average_Shot_Distance': away_average_shot_distance
    }
    
    return [home_data, away_data]

def match_stat(df,hteamName,ateamName):
        # -------------------- PASSING STATS --------------------
    hpossdf = df[(df['team_name']==hteamName) & (df['type_name']=='Pass')]
    apossdf = df[(df['team_name']==ateamName) & (df['type_name']=='Pass')]
    hposs = round((len(hpossdf)/(len(hpossdf)+len(apossdf)))*100,2)
    aposs = round((len(apossdf)/(len(hpossdf)+len(apossdf)))*100,2)
    hftdf = df[
        (df['team_name'] == hteamName) &
        (df['type_name'].isin(['Pass', 'Ball Receipt'])) &
        (df['pass_outcome_name'] == "1") &
        (df['x'] >= 80)
    ]

    aftdf = df[
        (df['team_name'] == ateamName) &
        (df['type_name'].isin(['Pass', 'Ball Receipt'])) &
        (df['pass_outcome_name'] == "1") &
        (df['x'] >= 80)
    ]

    hft = round((len(hftdf)/(len(hftdf)+len(aftdf)))*100, 2)
    aft = round((len(aftdf)/(len(hftdf)+len(aftdf)))*100, 2)
    dfpass_h = df[(df['type_name'] == 'Pass') & (df['team_name'] == hteamName)]
    acc_pass_h = dfpass_h[dfpass_h['pass_outcome_name'] == "1"]
    accurate_pass_perc_h = round((len(acc_pass_h) / len(dfpass_h)) * 100, 2) if len(dfpass_h) != 0 else 0

    pro_pass_h = acc_pass_h[(acc_pass_h['prog_pass'] >= 9.25) & (acc_pass_h['x'] >= 41) & 
        (~acc_pass_h['play_pattern_name'].astype(str).str.contains('From Corner|From Goal Kick', case=False, na=False))]

    Thr_ball_h = dfpass_h[dfpass_h['pass_technique_name'].astype(str).str.contains('Throughball', case=False, na=False)]
    Thr_ball_acc_h = Thr_ball_h[Thr_ball_h['pass_outcome_name'] == "1"]

    Lng_ball_h = dfpass_h[dfpass_h['pass_pass_cluster_label'].astype(str).str.contains('Long', case=False, na=False)]
    Lng_ball_acc_h = Lng_ball_h[Lng_ball_h['pass_outcome_name'] == "1"]

    Crs_pass_h = dfpass_h[dfpass_h['pass_cross'].astype(str).str.contains('TRUE', case=False, na=False)]
    Crs_pass_acc_h = Crs_pass_h[Crs_pass_h['pass_outcome_name'] == "1"]

    key_pass_h = dfpass_h[dfpass_h['pass_shot_assist'].astype(str).str.contains('TRUE', case=False, na=False)]
    g_assist_h = dfpass_h[dfpass_h['pass_goal_assist'].astype(str).str.contains('TRUE', case=False, na=False)]

    fnl_thd_h = acc_pass_h[acc_pass_h['end_x'] >= 92.5]
    pen_box_h = acc_pass_h[(acc_pass_h['end_x'] >= 102) & (acc_pass_h['end_y'].between(18.62, 62))]

    frwd_pass_h = dfpass_h[dfpass_h['pass_or_carry_angle'].between(-85, 85)]
    back_pass_h = dfpass_h[(dfpass_h['pass_or_carry_angle'] >= 95) | (dfpass_h['pass_or_carry_angle'] <= -95)]
    side_pass_h = dfpass_h[(dfpass_h['pass_or_carry_angle'].between(-95, -85)) | (dfpass_h['pass_or_carry_angle'].between(85, 95))]

    frwd_pass_acc_h = frwd_pass_h[frwd_pass_h['pass_outcome_name'] == "1"]
    back_pass_acc_h = back_pass_h[back_pass_h['pass_outcome_name'] == "1"]
    side_pass_acc_h = side_pass_h[side_pass_h['pass_outcome_name'] == "1"]

    corners_h = dfpass_h[dfpass_h['play_pattern_name'].astype(str).str.contains('From Corner', case=False, na=False)]
    corners_acc_h = corners_h[corners_h['pass_outcome_name'] == "1"]

    freekik_h = dfpass_h[dfpass_h['play_pattern_name'].astype(str).str.contains('From Free Kick', case=False, na=False)]
    freekik_acc_h = freekik_h[freekik_h['pass_outcome_name'] == "1"]

    thins_h = dfpass_h[dfpass_h['play_pattern_name'].astype(str).str.contains('From Throw In', case=False, na=False)]
    thins_acc_h = thins_h[thins_h['pass_outcome_name'] == "1"]

    # Away Team (repeat the same process)
    dfpass_a = df[(df['type_name'] == 'Pass') & (df['team_name'] == ateamName)]
    acc_pass_a = dfpass_a[dfpass_a['pass_outcome_name'] == "1"]
    accurate_pass_perc_a = round((len(acc_pass_a) / len(dfpass_a)) * 100, 2) if len(dfpass_a) != 0 else 0

    pro_pass_a = acc_pass_a[(acc_pass_a['prog_pass'] >= 9.25) & (acc_pass_a['x'] >= 41) & 
        (~acc_pass_a['play_pattern_name'].astype(str).str.contains('From Corner|From Goal Kick', case=False, na=False))]

    Thr_ball_a = dfpass_a[dfpass_a['pass_technique_name'].astype(str).str.contains('Throughball', case=False, na=False)]
    Thr_ball_acc_a = Thr_ball_a[Thr_ball_a['pass_outcome_name'] == "1"]

    Lng_ball_a = dfpass_a[dfpass_a['pass_pass_cluster_label'].astype(str).str.contains('Long', case=False, na=False)]
    Lng_ball_acc_a = Lng_ball_a[Lng_ball_a['pass_outcome_name'] == "1"]

    Crs_pass_a = dfpass_a[dfpass_a['pass_cross'].astype(str).str.contains('TRUE', case=False, na=False)]
    Crs_pass_acc_a = Crs_pass_a[Crs_pass_a['pass_outcome_name'] == "1"]

    key_pass_a = dfpass_a[dfpass_a['pass_shot_assist'].astype(str).str.contains('TRUE', case=False, na=False)]
    g_assist_a = dfpass_a[dfpass_a['pass_goal_assist'].astype(str).str.contains('TRUE', case=False, na=False)]

    fnl_thd_a = acc_pass_a[acc_pass_a['end_x'] >= 92.5]
    pen_box_a = acc_pass_a[(acc_pass_a['end_x'] >= 102) & (acc_pass_a['end_y'].between(18.62, 62))]

    frwd_pass_a = dfpass_a[dfpass_a['pass_or_carry_angle'].between(-85, 85)]
    back_pass_a = dfpass_a[(dfpass_a['pass_or_carry_angle'] >= 95) | (dfpass_a['pass_or_carry_angle'] <= -95)]
    side_pass_a = dfpass_a[(dfpass_a['pass_or_carry_angle'].between(-95, -85)) | (dfpass_a['pass_or_carry_angle'].between(85, 95))]

    frwd_pass_acc_a = frwd_pass_a[frwd_pass_a['pass_outcome_name'] == "1"]
    back_pass_acc_a = back_pass_a[back_pass_a['pass_outcome_name'] == "1"]
    side_pass_acc_a = side_pass_a[side_pass_a['pass_outcome_name'] == "1"]

    corners_a = dfpass_a[dfpass_a['play_pattern_name'].astype(str).str.contains('From Corner', case=False, na=False)]
    corners_acc_a = corners_a[corners_a['pass_outcome_name'] == "1"]

    freekik_a = dfpass_a[dfpass_a['play_pattern_name'].astype(str).str.contains('From Free Kick', case=False, na=False)]
    freekik_acc_a = freekik_a[freekik_a['pass_outcome_name'] == "1"]

    thins_a = dfpass_a[dfpass_a['play_pattern_name'].astype(str).str.contains('From Throw In', case=False, na=False)]
    thins_acc_a = thins_a[thins_a['pass_outcome_name'] == "1"]

    # -------------------- DEFENSIVE STATS --------------------
    df_home = df[df['team_name'] == hteamName]
    df_away = df[df['team_name'] == ateamName]

    # Home Defensive Stats
    # ---------------- HOME DEFENSIVE STATS ----------------
    # Tackles (Duel with type "Tackle")
    hTackles = len(df_home[(df_home["type_name"] == 'Duel') & (df_home['duel_type_name'].str.contains('Tackle', case=False, na=False))])

    # Unsuccessful Tackles
    hTackles_Lost = len(df_home[
        (df_home["type_name"] == 'Duel') &
        (df_home['duel_type_name'].str.contains('Tackle', case=False, na=False)) &
        (df_home['duel_outcome_name'].isin(['Lost In Play', 'Lost']))
    ])

    # Interceptions (successful only)
    hInterceptions = len(df_home[
        (df_home['type_name'] == 'Interception') &
        (df_home['interception_outcome_name'].isin(['Success In Play', 'Won']))
    ])

    # Ball Recoveries (excluding failed recoveries)
    hBall_Recoveries = len(df_home[
        (df_home['type_name'] == 'Ball Recovery') &
        (df_home['ball_recovery_recovery_failure'].astype(str) != 'True')
    ])

    # Clearances
    hClearances = len(df_home[df_home['type_name'] == 'Clearance'])

    # Aerial Duels
    hAerials = len(df_home[
        (df_home["type_name"] == 'Duel') &
        (df_home['duel_type_name'].str.contains('Aerial', case=False, na=False))
    ])

    # Aerials Lost
    hAerials_Lost = len(df_home[
        (df_home["type_name"] == 'Duel') &
        (df_home['duel_type_name'].str.contains('Aerial', case=False, na=False)) &
        (df_home['duel_outcome_name'] == 'Aerial Lost')
    ])

    # Blocked Passes
    hBlocked_Passes = len(df_home[df_home['type_name'] == 'BlockedPass'])

    # Defensive Blocks (not shot blocks)
    hBlocks = len(df_home[df_home['type_name'] == 'Block'])

    # Shot Blocks
    hShot_Blocks = len(df_home[
        (df_home['type_name'] == 'Block') &
        (df_home['block_save_block'].astype(str).str.contains('TRUE', case=False, na=False))
    ])

    # Dribbled Past
    hDribbled_Past = len(df_home[df_home["type_name"] == 'Dribbled Past'])

    # Defensive Errors
    # Events that resulted in shot or goal within next 2 rows
    hDef_Errors = 0
    hErrors_Lead_Goal = 0
    home_errors = df_home[
        ((df_home["type_name"] == 'Duel') &
        (df_home['duel_type_name'].str.contains('Tackle|Aerial', case=False, na=False)) &
        (df_home['duel_outcome_name'].isin(['Lost In Play', 'Lost', 'Aerial Lost']))) |
        (df_home["type_name"] == 'Foul Committed')
    ]
    for idx in home_errors.index:
        next_events = df_home.loc[idx + 1: idx + 2]
        if not next_events.empty:
            if 'Shot' in next_events['type_name'].values:
                hDef_Errors += 1
                if 'Goal' in next_events['shot_outcome_name'].values:
                    hErrors_Lead_Goal += 1

    # ---------------- AWAY DEFENSIVE STATS ----------------
    aTackles = len(df_away[(df_away["type_name"] == 'Duel') & (df_away['duel_type_name'].str.contains('Tackle', case=False, na=False))])

    aTackles_Lost = len(df_away[
        (df_away["type_name"] == 'Duel') &
        (df_away['duel_type_name'].str.contains('Tackle', case=False, na=False)) &
        (df_away['duel_outcome_name'].isin(['Lost In Play', 'Lost']))
    ])

    aInterceptions = len(df_away[
        (df_away['type_name'] == 'Interception') &
        (df_away['interception_outcome_name'].isin(['Success In Play', 'Won']))
    ])

    aBall_Recoveries = len(df_away[
        (df_away['type_name'] == 'Ball Recovery') &
        (df_away['ball_recovery_recovery_failure'].astype(str) != 'True')
    ])

    aClearances = len(df_away[df_away['type_name'] == 'Clearance'])

    aAerials = len(df_away[
        (df_away["type_name"] == 'Duel') &
        (df_away['duel_type_name'].str.contains('Aerial', case=False, na=False))
    ])

    aAerials_Lost = len(df_away[
        (df_away["type_name"] == 'Duel') &
        (df_away['duel_type_name'].str.contains('Aerial', case=False, na=False)) &
        (df_away['duel_outcome_name'] == 'Aerial Lost')
    ])

    aBlocked_Passes = len(df_away[df_away['type_name'] == 'BlockedPass'])

    aBlocks = len(df_away[df_away['type_name'] == 'Block'])

    aShot_Blocks = len(df_away[
        (df_away['type_name'] == 'Block') &
        (df_away['block_save_block'].astype(str).str.contains('TRUE', case=False, na=False))
    ])

    aDribbled_Past = len(df_away[df_away["type_name"] == 'Dribbled Past'])

    aDef_Errors = 0
    aErrors_Lead_Goal = 0
    away_errors = df_away[
        ((df_away["type_name"] == 'Duel') &
        (df_away['duel_type_name'].str.contains('Tackle|Aerial', case=False, na=False)) &
        (df_away['duel_outcome_name'].isin(['Lost In Play', 'Lost', 'Aerial Lost']))) |
        (df_away["type_name"] == 'Foul Committed')
    ]
    for idx in away_errors.index:
        next_events = df_away.loc[idx + 1: idx + 2]
        if not next_events.empty:
            if 'Shot' in next_events['type_name'].values:
                aDef_Errors += 1
                if 'Goal' in next_events['shot_outcome_name'].values:
                    aErrors_Lead_Goal += 1

    home_goalkick = df[(df['team_name']==hteamName) & (df['type_name']=='Pass') & (df['play_pattern_name'].str.contains('From Goal Kick'))& (df['position_name'].str.contains('Goalkeeper'))]
    away_goalkick = df[(df['team_name']==ateamName) & (df['type_name']=='Pass') & (df['play_pattern_name'].str.contains('From Goal Kick'))& (df['position_name'].str.contains('Goalkeeper'))]
    import ast
    if len(home_goalkick) != 0:
        home_goalkick['pass_length'] = home_goalkick['pass_length'].astype(float)
        hglkl = round(home_goalkick['pass_length'].mean(),2)

    else:
        hglkl = 0

    if len(away_goalkick) != 0:
        away_goalkick['pass_length'] = away_goalkick['pass_length'].astype(float)
        aglkl = round(away_goalkick['pass_length'].mean(),2)

    else:
        aglkl = 0




    # PPDA
    # -------------------- PPDA --------------------
    # Defensive actions in opponent half (x > 35)
    # HOME TEAM DEFENSIVE ACTIONS in HIGH ZONE
    home_def_acts = df[
        (df['team_name'] == hteamName) &
        (
            (df['type_name'].isin(['Interception', 'Foul Committed', 'BlockedPass', 'Dribbled Past'])) |
            ((df['type_name'] == 'Block') & (~df['block_save_block'].astype(str).str.contains('TRUE', case=False, na=False))) |
            ((df['type_name'] == 'Duel') & (df['duel_type_name'].str.contains('Tackle', case=False, na=False)))
        ) &
        (df['x'] > 35)
    ]

    # AWAY TEAM DEFENSIVE ACTIONS in HIGH ZONE
    away_def_acts = df[
        (df['team_name'] == ateamName) &
        (
            (df['type_name'].isin(['Interception', 'Foul Committed', 'BlockedPass', 'Dribbled Past'])) |
            ((df['type_name'] == 'Block') & (~df['block_save_block'].astype(str).str.contains('TRUE', case=False, na=False))) |
            ((df['type_name'] == 'Duel') & (df['duel_type_name'].str.contains('Tackle', case=False, na=False)))
        ) &
        (df['x'] > 35)
    ]
    # Successful passes by HOME team in build-up zone (<80)
    home_pass = df[
        (df['team_name'] == hteamName) &
        (df['type_name'] == 'Pass') &
        (df['pass_outcome_name'] == "1") &
        (df['x'] < 80)
    ]

    # Successful passes by AWAY team in build-up zone (<80)
    away_pass = df[
        (df['team_name'] == ateamName) &
        (df['type_name'] == 'Pass') &
        (df['pass_outcome_name'] == "1") &
        (df['x'] < 80)
    ]


    # Calculate PPDA
    home_ppda = round((len(away_pass) / len(home_def_acts)), 2) if len(home_def_acts) > 0 else 0
    away_ppda = round((len(home_pass) / len(away_def_acts)), 2) if len(away_def_acts) > 0 else 0

    # -------------------- Passes per Sequence (PPS) --------------------
    pass_df_home = df[(df['type_name'] == 'Pass') & (df['team_name'] == hteamName)]
    pass_df_away = df[(df['type_name'] == 'Pass') & (df['team_name'] == ateamName)]

    # PPS: Mean passes per possession
    pass_counts_home = pass_df_home.groupby('team_name').size()
    pass_counts_away = pass_df_away.groupby('team_name').size()

    PPS_home = round(pass_counts_home.mean(), 2)
    PPS_away = round(pass_counts_away.mean(), 2)

    # -------------------- 10+ Pass Sequences --------------------
    pass_seq_10_more_home = pass_counts_home[pass_counts_home >= 10].count()
    pass_seq_10_more_away = pass_counts_away[pass_counts_away >= 10].count()
    return (
        hposs, aposs, hft, aft,
        dfpass_h, acc_pass_h, Lng_ball_h, Lng_ball_acc_h,
        hTackles, hTackles_Lost, hInterceptions, hClearances, hAerials, hAerials_Lost,
        home_ppda, PPS_home, pass_seq_10_more_home,
        dfpass_a, acc_pass_a, Lng_ball_a, Lng_ball_acc_a,
        aTackles, aTackles_Lost, aInterceptions, aClearances, aAerials, aAerials_Lost,
        away_ppda, PPS_away, pass_seq_10_more_away
    )

path_eff1 = [path_effects.Stroke(linewidth=1.5, foreground=line_color), path_effects.Normal()]
def plotting_match_stats(ax, hteamName, ateamName, hposs, aposs, hft, aft,
                         dfpass_h, acc_pass_h, Lng_ball_h, Lng_ball_acc_h,
                         hTackles, hTackles_Lost, hInterceptions, hClearances, hAerials, hAerials_Lost,
                         home_ppda, PPS_home, pass_seq_10_more_home,
                         dfpass_a, acc_pass_a, Lng_ball_a, Lng_ball_acc_a,
                         aTackles, aTackles_Lost, aInterceptions, aClearances, aAerials, aAerials_Lost,
                         away_ppda, PPS_away, pass_seq_10_more_away,
                         bg_color, hcol, acol, path_eff1):
        pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
        pitch.draw(ax=ax)
        ax.set_xlim(-0.5, 120.5)
        ax.set_ylim(-5, 80.5)
        ax.text(60,76, "Match Stats", ha='center', va='center', color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff1)

        # plotting the headline box
        head_y = [72,80,80,72]
        head_x = [0,0,120,120]
        ax.fill(head_x, head_y, '#dd9137')
        # Stats bar diagram
        stats_title = [58, 58-(1*6), 58-(2*6), 58-(3*6), 58-(4*6), 58-(5*6), 58-(6*6), 58-(7*6), 58-(8*6), 58-(9*6), 58-(10*6)] # y co-ordinate values of the bars
    # Raw stats
        stats_home = [
            hposs,
            hft,
            len(dfpass_h),          # Total passes
            len(Lng_ball_h),       # Long balls
            hTackles,
            hInterceptions,
            hClearances,
            hAerials,
            home_ppda,
            PPS_home,
            pass_seq_10_more_home
        ]

        stats_away = [
            aposs,
            aft,
            len(dfpass_a),          # Total passes
            len(Lng_ball_a),       # Long balls
            aTackles,
            aInterceptions,
            aClearances,
            aAerials,
            away_ppda,
            PPS_away,
            pass_seq_10_more_away
        ]


        # Normalized stats (home bars go negative to point left)
        def safe_norm(home_val, away_val):
            total = home_val + away_val
            if total == 0:
                return 0, 0
            return -(home_val / total) * 50, (away_val / total) * 50

        stats_normalized_home = []
        stats_normalized_away = []

        for h_val, a_val in zip(stats_home, stats_away):
            h_norm, a_norm = safe_norm(h_val, a_val)
            stats_normalized_home.append(h_norm)
            stats_normalized_away.append(a_norm)


        start_x = 60
        ax.barh(stats_title, stats_normalized_home, height=4, color=hcol, left=start_x)
        ax.barh(stats_title, stats_normalized_away, height=4, left=start_x, color=acol)
        # Turn off axis-related elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plotting the texts
        ax.text(60, 58, "Possession", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(1*6), "Field Tilt", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(2*6), "Passes (Acc.)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(3*6), "LongBalls (Acc.)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(4*6), "Tackles (Wins)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(5*6), "Interceptions", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(6*6), "Clearence", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(7*6), "Aerials (Wins)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(8*6), "PPDA", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(9*6), "Pass/Sequence", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(60, 58-(10*6), "10+Pass Seq.", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        ax.text(30, 58, f"{round(hposs)}%", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(1*6), f"{round(hft)}%", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(2*6), f"{len(dfpass_h)} ({len(acc_pass_h)})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(3*6), f"{len(Lng_ball_h)} ({len(Lng_ball_acc_h)})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(4*6), f"{hTackles} ({hTackles - hTackles_Lost})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(5*6), f"{hInterceptions}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(6*6), f"{hClearances}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(7*6), f"{hAerials} ({hAerials - hAerials_Lost})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(8*6), f"{home_ppda}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(9*6), f"{int(PPS_home)}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        ax.text(30, 58-(10*6), f"{pass_seq_10_more_home}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')

        ax.text(90, 58, f"{round(aposs)}%", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(1*6), f"{round(aft)}%", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(2*6), f"{len(dfpass_a)} ({len(acc_pass_a)})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(3*6), f"{len(Lng_ball_a)} ({len(Lng_ball_acc_a)})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(4*6), f"{aTackles} ({aTackles - aTackles_Lost})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(5*6), f"{aInterceptions}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(6*6), f"{aClearances}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(7*6), f"{aAerials} ({aAerials - aAerials_Lost})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(8*6), f"{away_ppda}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(9*6), f"{int(PPS_away)}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        ax.text(90, 58-(10*6), f"{pass_seq_10_more_away}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
        # Text annotations for HOME team (left-aligned)
        stat_labels_y = [58 - (i * 6) for i in range(11)]

        home_data = {
            "Team_Name": hteamName,
            "Possession": hposs,
            "Field Tilt": hft,
            "Passes (Acc.)": f"{len(dfpass_h)} ({len(acc_pass_h)})",
            "LongBalls (Acc.)": f"{len(Lng_ball_h)} ({len(Lng_ball_acc_h)})",
            "Tackles (Wins)": f"{hTackles} ({hTackles - hTackles_Lost})",
            "Interceptions": hInterceptions,
            "Clearance": hClearances,
            "Aerials (Wins)": f"{hAerials} ({hAerials - hAerials_Lost})",
            "PPDA": home_ppda,
            "Pass/Sequence": PPS_home,
            "10+ Pass Seq.": pass_seq_10_more_home
        }

        away_data = {
            "Team_Name": ateamName,
            "Possession": aposs,
            "Field Tilt": aft,
            "Passes (Acc.)": f"{len(dfpass_a)} ({len(acc_pass_a)})",
            "LongBalls (Acc.)": f"{len(Lng_ball_a)} ({len(Lng_ball_acc_a)})",
            "Tackles (Wins)": f"{aTackles} ({aTackles - aTackles_Lost})",
            "Interceptions": aInterceptions,
            "Clearance": aClearances,
            "Aerials (Wins)": f"{aAerials} ({aAerials - aAerials_Lost})",
            "PPDA": away_ppda,
            "Pass/Sequence": PPS_away,
            "10+ Pass Seq.": pass_seq_10_more_away
        }

    # Create final DataFrame
        general_match_stats_df = pd.DataFrame([home_data, away_data])
        return general_match_stats_df
def Final_third_entry(ax, team_name, col,df,ateamName,hteamName):
    dfpass = df[(df['team_name']==team_name) & (df['type_name']=='Pass') & (df['x']<80) & (df['end_x']>=80) & (df['pass_outcome_name']=='1') &
                (~df['play_pattern_name'].str.contains('From Free Kick'))]
    dfcarry = df[(df['team_name']==team_name) & (df['type_name']=='Carry') & (df['x']<80) & (df['carry_end_x']>=80)]
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=line_color, linewidth=2,
                          corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 120.5)
    # ax.set_ylim(-0.5, 68.5)

    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()

    pass_count = len(dfpass) + len(dfcarry)

    # calculating the counts
    left_entry = len(dfpass[dfpass['y']>=53.33]) + len(dfcarry[dfcarry['y']>=53.33])
    mid_entry = len(dfpass[(dfpass['y']>25.67) & (dfpass['y']<53.33)]) + len(dfcarry[(dfcarry['y']>=25.67) & (dfcarry['y']<53.33)])
    right_entry = len(dfpass[(dfpass['y']>=0) & (dfpass['y']<25.67)]) + len(dfcarry[(dfcarry['y']>=0) & (dfcarry['y']<25.67)])
    left_percentage = round((left_entry/pass_count)*100)
    mid_percentage = round((mid_entry/pass_count)*100)
    right_percentage = round((right_entry/pass_count)*100)

    ax.hlines(22.67, xmin=0, xmax=80, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.hlines(55.33, xmin=0, xmax=80, colors=line_color, linestyle='dashed', alpha=0.35)
    ax.vlines(80, ymin=-2, ymax=80, colors=line_color, linestyle='dashed', alpha=0.55)

    # showing the texts in the pitch
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
    if col == hcol:
        ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
        ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
        ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=hcol, fontsize=24, va='center', ha='center', bbox=bbox_props)
    else:
        ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
        ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)
        ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=acol, fontsize=24, va='center', ha='center', bbox=bbox_props)

    # plotting the passes
    pro_pass = pitch.lines(dfpass.x, dfpass.y, dfpass.end_x, dfpass.end_y, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
    # plotting some scatters at the end of each pass
    pro_pass_end = pitch.scatter(dfpass.end_x, dfpass.end_y, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)
    # plotting carries
    for index, row in dfcarry.iterrows():
        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['carry_end_x'], row['carry_end_y']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                        alpha=1, linewidth=2, linestyle='--')
        ax.add_patch(arrow)

    counttext = f"{pass_count} Final Third Entries"

    # Heading and other texts
    if col == hcol:
        ax.set_title(f"{hteamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
        ax.text(99.5, 82, '<------------------ Final third ------------------>', color=line_color, ha='center', va='center')
        pitch.lines(53, -2, 73, -2, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
        ax.scatter(73,-2, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2)
        arrow = patches.FancyArrowPatch((83, -2), (103, -2), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                        alpha=1, linewidth=2, linestyle='--')
        ax.add_patch(arrow)
        ax.text(63, -3, f'Entry by Pass: {len(dfpass)}', fontsize=8, color=line_color, ha='center', va='center')
        ax.text(93, -3, f'Entry by Carry: {len(dfcarry)}', fontsize=8, color=line_color, ha='center', va='center')
        
    else:
        ax.set_title(f"{ateamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
        ax.text(99.5, -2.8, '<------------------ Final third ------------------>', color=line_color, ha='center', va='center')
        pitch.lines(53, 81, 73, 81, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
        ax.scatter(73,81, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2)
        arrow = patches.FancyArrowPatch((83, 81), (103, 81), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                        alpha=1, linewidth=2, linestyle='--')
        ax.add_patch(arrow)
        ax.text(63, 83, f'Entry by Pass: {len(dfpass)}', fontsize=8, color=line_color, ha='center', va='center')
        ax.text(93, 83, f'Entry by Carry: {len(dfcarry)}', fontsize=8, color=line_color, ha='center', va='center')

    return {
        'Team_Name': team_name,
        'Total_Final_Third_Entries': pass_count,
        'Final_Third_Entries_From_Left': left_entry,
        'Final_Third_Entries_From_Center': mid_entry,
        'Final_Third_Entries_From_Right': right_entry,
        'Entry_By_Pass': len(dfpass),
        'Entry_By_Carry': len(dfcarry)
    }
def zone14hs(ax, team_name, col,ateamName,hteamName,df):
    dfhp = df[(df['team_name']==team_name) & (df['type_name']=='Pass') & (df['pass_outcome_name']=='1') & 
              (~df['play_pattern_name'].str.contains('CornerTaken|Freekick'))]
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=line_color,  linewidth=2,
                          corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 120.5)
    ax.set_facecolor(bg_color)
    if team_name == ateamName:
      ax.invert_xaxis()
      ax.invert_yaxis()

    # setting the count varibale
    z14 = 0
    hs = 0
    lhs = 0
    rhs = 0

    path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
    # iterating ecah pass and according to the conditions plotting only zone14 and half spaces passes
    for index, row in dfhp.iterrows():
        if row['end_x'] >= 80 and row['end_x'] <= 98.54 and row['end_y'] >= 26.66 and row['end_y'] <= 53.32:
            pitch.lines(row['x'], row['y'], row['end_x'], row['end_y'], color='orange', comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
            ax.scatter(row['end_x'], row['end_y'], s=35, linewidth=1, color=bg_color, edgecolor='orange', zorder=4)
            z14 += 1
        if row['end_x'] >= 80 and row['end_y'] >= 13.33 and row['end_y'] <= 26.66:
            pitch.lines(row['x'], row['y'], row['end_x'], row['end_y'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
            ax.scatter(row['end_x'], row['end_y'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
            hs += 1
            rhs += 1
        if row['end_x'] >= 80 and row['end_y'] >= 53.32 and row['end_y'] <= 67.95:
            pitch.lines(row['x'], row['y'], row['end_x'], row['end_y'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
            ax.scatter(row['end_x'], row['end_y'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
            hs += 1
            lhs += 1

    # coloring those zones in the pitch
    y_z14 = [26.66, 26.66, 53.32, 53.32]
    x_z14 = [80, 88.54, 88.54, 80]
    ax.fill(x_z14, y_z14, 'orange', alpha=0.2, label='Zone14')

    y_rhs = [13.33, 13.33, 26.66, 26.66]
    x_rhs = [80, 120, 120, 80]
    ax.fill(x_rhs, y_rhs, col, alpha=0.2, label='HalfSpaces')

    y_lhs = [53.32, 53.32, 67.95, 67.95]
    x_lhs = [80, 120, 120, 80]
    ax.fill(x_lhs, y_lhs, col, alpha=0.2, label='HalfSpaces')

    # showing the counts in an attractive way
    z14name = "Zone14"
    hsname = "HalfSp"
    z14count = f"{z14}"
    hscount = f"{hs}"
    ax.scatter(16.46, 13.85, color=col, s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
    ax.scatter(16.46, 54.15, color='orange', s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
    ax.text(16.46, 13.85-4, hsname, fontsize=20, color=line_color, ha='center', va='center', path_effects=path_eff)
    ax.text(16.46, 54.15-4, z14name, fontsize=20, color=line_color, ha='center', va='center', path_effects=path_eff)
    ax.text(16.46, 13.85+2, hscount, fontsize=40, color=line_color, ha='center', va='center', path_effects=path_eff)
    ax.text(16.46, 54.15+2, z14count, fontsize=40, color=line_color, ha='center', va='center', path_effects=path_eff)

    # Headings and other texts
    if col == hcol:
      ax.set_title(f"{hteamName}\nZone14 & Halfsp. Pass", color=line_color, fontsize=25, fontweight='bold')
    else:
      ax.set_title(f"{ateamName}\nZone14 & Halfsp. Pass", color=line_color, fontsize=25, fontweight='bold')

    return {
        'Team_Name': team_name,
        'Total_Passes_Into_Zone14': z14,
        'Passes_Into_Halfspaces': hs,
        'Passes_Into_Left_Halfspaces': lhs,
        'Passes_Into_Right_Halfspaces': rhs
    }
def Pass_end_zone(ax, team_name, cm,df,ateamName,hteamName):
    pez = df[(df['team_name'] == team_name) & (df['type_name'] == 'Pass') & (df['pass_outcome_name'] == '1')]
    pitch = Pitch(pitch_type='statsbomb', line_color=line_color, goal_type='box', goal_alpha=.5, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 120.5)
    if team_name == ateamName:
      ax.invert_xaxis()
      ax.invert_yaxis()

    pearl_earring_cmap = "bone"
    # binning the data points
    # bin_statistic = pitch.bin_statistic_positional(df.end_x, df.end_y, statistic='count', positional='full', normalize=True)
    bin_statistic = pitch.bin_statistic(pez.end_x, pez.end_y, bins=(6, 5), normalize=True)
    pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors=bg_color)
    pitch.scatter(pez.end_x, pez.end_y, c='gray', alpha=0.5, s=5, ax=ax)
    labels = pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)

    # Headings and other texts
    if team_name == hteamName:
      ax.set_title(f"{hteamName}\nPass End Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
    else:
      ax.set_title(f"{ateamName}\nPass End Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
def pass_end_zone_counts(team_name,df):
    # Filter successful passes by team
    passes = df[(df['team_name'] == team_name) & (df['type_name'] == 'Pass') & (df['pass_outcome_name'] == '1')]

    # Create zones
    conditions = [
        (passes['end_x'] < 40),
        (passes['end_x'] >= 40) & (passes['end_x'] < 80),
        (passes['end_x'] >= 80)
    ]
    third_labels = ['Defensive Third', 'Middle Third', 'Final Third']
    passes['zone'] = np.select(conditions, third_labels)

    # Horizontal zones (left/mid/right)
    h_conditions = [
        (passes['end_y'] < 22.67),
        (passes['end_y'] >= 22.67) & (passes['end_y'] < 45.33),
        (passes['end_y'] >= 45.33)
    ]
    h_labels = ['Left', 'Center', 'Right']
    passes['side'] = np.select(h_conditions, h_labels)

    # Count number of passes per zone and side
    summary = passes.groupby(['zone', 'side']).size().unstack(fill_value=0)

    # Total per zone
    summary['Total'] = summary.sum(axis=1)

    return summary.reset_index()
def Chance_creating_zone(ax, team_name, cm, col,ateamName,hteamName,df):
    # Filter key passes
    ccp = df[
    (df['type_name'] == 'Pass') &
    (df['pass_outcome_name'] == '1') &
    (df['team_name'] == team_name) &
    (
        df['pass_shot_assist'].astype(str).str.contains('TRUE|True', case=False, na=False) |
        df['pass_goal_assist'].astype(str).str.contains('TRUE|True', case=False, na=False)
    ) &
    (~df['play_pattern_name'].astype(str).str.contains('From Corner|From Free Kick', case=False, na=False))
]
    print(ccp['pass_goal_assist'].dropna().unique())




    pitch = Pitch(pitch_type='statsbomb', line_color=line_color, corner_arcs=True,
                  line_zorder=2, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 120.5)

    if team_name == ateamName:
        ax.invert_xaxis()
        ax.invert_yaxis()

    cc = 0  # total chances
    pearl_earring_cmap = "bone"

    # Heatmap binning
    bin_statistic = pitch.bin_statistic(ccp['x'], ccp['y'], bins=(6, 5), statistic='count', normalize=False)
    pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors='#f8f8f8')

    # Draw lines and end points
    for _, row in ccp.iterrows():
        is_assist = row.get('pass_goal_assist') is True
        color = green if is_assist else violet
        pitch.lines(row['x'], row['y'], row['end_x'], row['end_y'], color=color, comet=True, lw=3, zorder=3, ax=ax)
        ax.scatter(row['end_x'], row['end_y'], s=35, linewidth=1, color=bg_color, edgecolor=color, zorder=4)
        cc += 1

    # Add value labels on heatmap
    pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax,
                        ha='center', va='center', str_format='{:.0f}', path_effects=path_eff)

    # Titles & Texts
    if col == hcol:
        ax.text(120, -3.5, "violet = key pass\ngreen = assist", color=hcol, size=10, ha='right', va='center')
        ax.text(60, 82, f"Total Chances Created = {cc}", color=col, fontsize=10, ha='center', va='center')
        ax.set_title(f"{hteamName}\nChance Creating Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
    else:
        ax.text(120, 83, "violet = key pass\ngreen = assist", color=acol, size=10, ha='left', va='center')
        ax.text(60, -3, f"Total Chances Created = {cc}", color=col, fontsize=10, ha='center', va='center')
        ax.set_title(f"{ateamName}\nChance Creating Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)

    return {
        'Team_Name': team_name,
        'Total_Chances_Created': cc
    }
def box_entry(ax,df,ateamName,hteamName,):
    bentry = df[
        (
            ((df['type_name'] == 'Pass') & (df['pass_outcome_name'] == '1')) |
            ((df['type_name'] == 'Carry'))
        ) &
        (
            (
                ((df['type_name'] == 'Pass') & (df['end_x'] >= 88.5) & (df['end_y'].between(16, 64.2))) |
                ((df['type_name'] == 'Carry') & (df['carry_end_x'] >= 88.5) & (df['carry_end_y'].between(16, 64.2)))
            )
        ) &
        ~(
            (df['x'] >= 101.1) & (df['y'].between(16, 64.2))
        ) &
        (~df['play_pattern_name'].astype(str).str.contains('From Corner|From Free Kick|From Throw In', na=False))
    ]
    hbentry = bentry[bentry['team_name']==hteamName]
    abentry = bentry[bentry['team_name']==ateamName]

    hrigt = hbentry[hbentry['y']<80/3]
    hcent = hbentry[(hbentry['y']>=80/3) & (hbentry['y']<=160/3)]
    hleft = hbentry[hbentry['y']>160/3]

    arigt = abentry[(abentry['y']<80/3)]
    acent = abentry[(abentry['y']>=80/3) & (abentry['y']<=160/3)]
    aleft = abentry[(abentry['y']>160/3)]

    pitch = Pitch(pitch_type='statsbomb', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 120.5)
    ax.set_ylim(-0.5, 80.5)

    for index, row in bentry.iterrows():
    # Determine team color and apply flip for home team
        if row['team_name'] == ateamName:
            color = acol
            flip = False
        elif row['team_name'] == hteamName:
            color = hcol
            flip = True
        else:
            continue  # Skip invalid teams

        # Select correct coordinates based on action type
        if row['type_name'] == 'Pass':
            x, y = row['x'], row['y']
            end_x, end_y = row['end_x'], row['end_y']
        elif row['type_name'] == 'Carry':
            x, y = row['x'], row['y']
            end_x, end_y = row['carry_end_x'], row['carry_end_y']
        else:
            continue  # Skip anything else

        # Flip coordinates for home team
        if flip:
            x, y = 120 - x, 80 - y
            end_x, end_y = 120 - end_x, 80 - end_y

        # Draw the actions
        if row['type_name'] == 'Pass':
            pitch.lines(x, y, end_x, end_y, lw=3.5, comet=True, color=color, ax=ax, alpha=0.5)
            pitch.scatter(end_x, end_y, s=35, edgecolor=color, linewidth=1, color=bg_color, zorder=2, ax=ax)
        elif row['type_name'] == 'Carry':
            arrow = patches.FancyArrowPatch(
                (x, y), (end_x, end_y),
                arrowstyle='->', color=color,
                zorder=4, mutation_scale=20,
                alpha=1, linewidth=2, linestyle='--'
            )
            ax.add_patch(arrow)


    
    ax.text(0, 69, f'{hteamName}\nBox Entries: {len(hbentry)}', color=hcol, fontsize=25, fontweight='bold', ha='left', va='bottom')
    ax.text(105, 69, f'{ateamName}\nBox Entries: {len(abentry)}', color=acol, fontsize=25, fontweight='bold', ha='right', va='bottom')

    ax.scatter(46, 6, s=2000, marker='s', color=hcol, zorder=3)
    ax.scatter(46, 34, s=2000, marker='s', color=hcol, zorder=3)
    ax.scatter(46, 62, s=2000, marker='s', color=hcol, zorder=3)
    ax.text(46, 6, f'{len(hleft)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
    ax.text(46, 34, f'{len(hcent)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
    ax.text(46, 62, f'{len(hrigt)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')

    ax.scatter(59.5, 6, s=2000, marker='s', color=acol, zorder=3)
    ax.scatter(59.5, 34, s=2000, marker='s', color=acol, zorder=3)
    ax.scatter(59.5, 62, s=2000, marker='s', color=acol, zorder=3)
    ax.text(59.5, 6, f'{len(arigt)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
    ax.text(59.5, 34, f'{len(acent)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
    ax.text(59.5, 62, f'{len(aleft)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')

    home_data = {
        'Team_Name': hteamName,
        'Total_Box_Entries': len(hbentry),
        'Box_Entry_From_Left': len(hleft),
        'Box_Entry_From_Center': len(hcent),
        'Box_Entry_From_Right': len(hrigt)
    }
    
    away_data = {
        'Team_Name': ateamName,
        'Total_Box_Entries': len(abentry),
        'Box_Entry_From_Left': len(aleft),
        'Box_Entry_From_Center': len(acent),
        'Box_Entry_From_Right': len(arigt)
    }
    
    return [home_data, away_data]
def Crosses(ax,df,ateamName,hteamName):
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 80.5)
    ax.set_xlim(-0.5, 120.5)

    # Filtering crosses correctly
    home_cross = df[
        (df['team_name'] == hteamName) &
        (df['type_name'] == 'Pass') &
        (df['pass_cross'].astype(str).str.lower() == 'true') &
        (~df['play_pattern_name'].astype(str).str.contains('From Corner', case=False, na=False))
    ]

    away_cross = df[
        (df['team_name'] == ateamName) &
        (df['type_name'] == 'Pass') &
        (df['pass_cross'].astype(str).str.lower() == 'true') &
        (~df['play_pattern_name'].astype(str).str.contains('From Corner', case=False, na=False))
    ]

    hsuc, hunsuc, asuc, aunsuc = 0, 0, 0, 0

    for _, row in home_cross.iterrows():
        arrow = patches.FancyArrowPatch(
            (120 - row['x'], 80 - row['y']), (120 - row['end_x'], 80 - row['end_y']),
            arrowstyle='->', mutation_scale=15 if row['pass_outcome_name'] == '1' else 10,
            color=hcol if row['pass_outcome_name'] == '1' else line_color,
            linewidth=1.5 if row['pass_outcome_name'] == '1' else 1,
            alpha=1 if row['pass_outcome_name'] == '1' else .25, zorder=3
        )
        ax.add_patch(arrow)
        hsuc += (row['pass_outcome_name'] == '1')
        hunsuc += (row['pass_outcome_name'] != '1')

    for _, row in away_cross.iterrows():
        arrow = patches.FancyArrowPatch(
            (row['x'], row['y']), (row['end_x'], row['end_y']),
            arrowstyle='->', mutation_scale=15 if row['pass_outcome_name'] == '1' else 10,
            color=acol if row['pass_outcome_name'] == '1' else line_color,
            linewidth=1.5 if row['pass_outcome_name'] == '1' else 1,
            alpha=1 if row['pass_outcome_name'] == '1' else .25, zorder=3
        )
        ax.add_patch(arrow)
        asuc += (row['pass_outcome_name'] == '1')
        aunsuc += (row['pass_outcome_name'] != '1')

    home_left = len(home_cross[home_cross['y'] >= 40])
    home_right = len(home_cross[home_cross['y'] < 40])
    away_left = len(away_cross[away_cross['y'] >= 40])
    away_right = len(away_cross[away_cross['y'] < 40])

    # Annotations
    ax.text(51, 2, f"Crosses from\nLeftwing: {home_left}", color=hcol, fontsize=15, va='bottom', ha='right')
    ax.text(51, 66, f"Crosses from\nRightwing: {home_right}", color=hcol, fontsize=15, va='top', ha='right')
    ax.text(54, 66, f"Crosses from\nLeftwing: {away_left}", color=acol, fontsize=15, va='top', ha='left')
    ax.text(54, 2, f"Crosses from\nRightwing: {away_right}", color=acol, fontsize=15, va='bottom', ha='left')

    ax.text(0, -2, f"Successful: {hsuc}", color=hcol, fontsize=20, ha='left', va='top')
    ax.text(0, -5.5, f"Unsuccessful: {hunsuc}", color=line_color, fontsize=20, ha='left', va='top')
    ax.text(105, -2, f"Successful: {asuc}", color=acol, fontsize=20, ha='right', va='top')
    ax.text(105, -5.5, f"Unsuccessful: {aunsuc}", color=line_color, fontsize=20, ha='right', va='top')

    ax.text(0, 70, f"{hteamName}\n<---Crosses", color=hcol, size=25, ha='left', fontweight='bold')
    ax.text(105, 70, f"{ateamName}\nCrosses--->", color=acol, size=25, ha='right', fontweight='bold')

    home_data = {
        'Team_Name': hteamName,
        'Total_Cross': hsuc + hunsuc,
        'Successful_Cross': hsuc,
        'Unsuccessful_Cross': hunsuc,
        'Cross_From_LeftWing': home_left,
        'Cross_From_RightWing': home_right
    }

    away_data = {
        'Team_Name': ateamName,
        'Total_Cross': asuc + aunsuc,
        'Successful_Cross': asuc,
        'Unsuccessful_Cross': aunsuc,
        'Cross_From_LeftWing': away_left,
        'Cross_From_RightWing': away_right
    }

    return [home_data, away_data]
def HighTO(ax,ateamName,hteamName,df):
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5, 80.5)
    ax.set_xlim(-0.5, 120.5)

    highTO = df.copy()
    highTO['Distance'] = ((highTO['x'] - 120) ** 2 + (highTO['y'] - 40) ** 2) ** 0.5

    def is_valid_turnover(row, team_name):
        return (
            row['team_name'] == team_name and
            row['Distance'] <= 40 and
            (
                row['type_name'] == 'Ball Recovery' or
                (row['type_name'] == 'Interception' and row.get('interception_outcome_name') in ['Success In Play', 'Won'])
            )
        )

    hgoal_count = agoal_count = hshot_count = ashot_count = hht_count = aht_count = 0

    def handle_turnovers(team_name, color, side='right'):
        nonlocal hgoal_count, agoal_count, hshot_count, ashot_count, hht_count, aht_count

        for i in range(len(highTO)):
            row = highTO.iloc[i]
            if is_valid_turnover(row, team_name):
                possession_id = row['team_name']

                # Check for goals
                for j in range(i + 1, len(highTO)):
                    next_row = highTO.iloc[j]
                    if next_row['team_name'] != possession_id or next_row['team_name'] != team_name:
                        break
                    if next_row['shot_outcome_name'] == 'Goal':
                        x, y = row['x'], row['y']
                        if side == 'left':
                            x, y = 120 - x, 80 - y
                        ax.scatter(x, y, s=600, marker='*', color='green', edgecolor='k', zorder=3)
                        if team_name == hteamName:
                            hgoal_count += 1
                        else:
                            agoal_count += 1
                        break

                # Check for shots
                for j in range(i + 1, len(highTO)):
                    next_row = highTO.iloc[j]
                    if next_row['team_name'] != possession_id or next_row['team_name'] != team_name:
                        break
                    if next_row['type_name'] == 'Shot':
                        x, y = row['x'], row['y']
                        if side == 'left':
                            x, y = 120 - x, 80 - y
                        ax.scatter(x, y, s=150, color=color, edgecolor=bg_color, zorder=2)
                        if team_name == hteamName:
                            hshot_count += 1
                        else:
                            ashot_count += 1
                        break

                # Basic turnover
                j = i + 1
                if (
                    j < len(highTO) and
                    highTO.iloc[j]['team_name'] == team_name and
                    highTO.iloc[j]['type_name'] not in ['Dispossessed']
                ):
                    x, y = row['x'], row['y']
                    if side == 'left':
                        x, y = 120 - x, 80 - y
                    ax.scatter(x, y, s=100, color='None', edgecolor=color)
                    if team_name == hteamName:
                        hht_count += 1
                    else:
                        aht_count += 1

    # Process for both teams
    handle_turnovers(hteamName, hcol, side='left')
    handle_turnovers(ateamName, acol, side='right')

    # Plotting the half circles
    ax.add_artist(plt.Circle((0, 40), 40, color=hcol, fill=True, alpha=0.25, linestyle='dashed'))
    ax.add_artist(plt.Circle((120, 40), 40, color=acol, fill=True, alpha=0.25, linestyle='dashed'))
    ax.set_aspect('equal', adjustable='box')

    # Headings and text
    ax.text(0, 70, f"{hteamName}\nHigh Turnover: {hht_count}", color=hcol, size=25, ha='left', fontweight='bold')
    ax.text(105, 70, f"{ateamName}\nHigh Turnover: {aht_count}", color=acol, size=25, ha='right', fontweight='bold')
    ax.text(0, -3, '<---Attacking Direction', color=hcol, fontsize=13, ha='left', va='center')
    ax.text(105, -3, 'Attacking Direction--->', color=acol, fontsize=13, ha='right', va='center')

    # Return stats as dictionary
    home_data = {
        'Team_Name': hteamName,
        'Total_High_Turnovers': hht_count,
        'Shot_Ending_High_Turnovers': hshot_count,
        'Goal_Ending_High_Turnovers': hgoal_count,
        'Opponent_Team_Name': ateamName
    }

    away_data = {
        'Team_Name': ateamName,
        'Total_High_Turnovers': aht_count,
        'Shot_Ending_High_Turnovers': ashot_count,
        'Goal_Ending_High_Turnovers': agoal_count,
        'Opponent_Team_Name': hteamName
    }

    return [home_data, away_data]
def plot_congestion(ax,ateamName,hteamName,df,ax_text):
    pcmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [acol, 'gray', hcol], N=20)
    valid_types = ['Ball Receipt*', 'Pass', 'Ball Recovery']
    df1 = df[
        (df['team_name'] == hteamName) &
        (df['type_name'].isin(valid_types)) &
        (~df['play_pattern_name'].str.contains('CornerTaken|Freekick|ThrowIn', na=False))
    ]
    df2 = df[
        (df['team_name'] == ateamName) &
        (df['type_name'].isin(valid_types)) &
        (~df['play_pattern_name'].str.contains('CornerTaken|Freekick|ThrowIn', na=False))
    ]
    df2['x'] = 120-df2['x']
    df2['y'] =  80-df2['y']
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=6)
    pitch.draw(ax=ax)
    ax.set_ylim(-0.5,80.5)
    ax.set_xlim(-0.5,120.5)

    bin_statistic1 = pitch.bin_statistic(df1.x, df1.y, bins=(6,5), statistic='count', normalize=False)
    bin_statistic2 = pitch.bin_statistic(df2.x, df2.y, bins=(6,5), statistic='count', normalize=False)
 # Original grid
    cx = np.array([
        [8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
        [8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
        [8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
        [8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
        [8.75, 26.25, 43.75, 61.25, 78.75, 96.25]
    ])

    cy = np.array([
        [61.2, 61.2, 61.2, 61.2, 61.2, 61.2],
        [47.6, 47.6, 47.6, 47.6, 47.6, 47.6],
        [34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        [20.4, 20.4, 20.4, 20.4, 20.4, 20.4],
        [6.8,  6.8,  6.8,  6.8,  6.8,  6.8]
    ])

    # Scale to 120x80 pitch
    scale_x = 120 / 105
    scale_y = 80 / 68

    cx = cx * scale_x
    cy = cy * scale_y


    # Flatten the arrays
    cx_flat = cx.flatten()
    cy_flat = cy.flatten()

    # Create a DataFrame
    df_cong = pd.DataFrame({'cx': cx_flat, 'cy': cy_flat})

    hd_values = []


    # Loop through the 2D arrays
    for i in range(bin_statistic1['statistic'].shape[0]):
        for j in range(bin_statistic1['statistic'].shape[1]):
            stat1 = bin_statistic1['statistic'][i, j]
            stat2 = bin_statistic2['statistic'][i, j]
        
            if (stat1 / (stat1 + stat2)) > 0.55:
                hd_values.append(1)
            elif (stat1 / (stat1 + stat2)) < 0.45:
                hd_values.append(0)
            else:
                hd_values.append(0.5)

    df_cong['hd']=hd_values
    bin_stat = pitch.bin_statistic(df_cong.cx, df_cong.cy, bins=(6,5), values=df_cong['hd'], statistic='sum', normalize=False)
    pitch.heatmap(bin_stat, ax=ax, cmap=pcmap, edgecolors='#000000', lw=0, zorder=3, alpha=0.85)

    ax_text(60, 82, s=f"<{hteamName}>  |  Contested  |  <{ateamName}>", highlight_textprops=[{'color':hcol}, {'color':acol}],
            color='gray', fontsize=18, ha='center', va='center', ax=ax)
    ax.set_title("Team's Dominating Zone", color=line_color, fontsize=30, fontweight='bold', y=1.075)
    ax.text(0,  -3, 'Attacking Direction--->', color=hcol, fontsize=13, ha='left', va='center')
    ax.text(120,-3, '<---Attacking Direction', color=acol, fontsize=13, ha='right', va='center')

    ax.vlines(1*(120/6), ymin=0, ymax=80, color=bg_color, lw=2, ls='--', zorder=5)
    ax.vlines(2*(120/6), ymin=0, ymax=80, color=bg_color, lw=2, ls='--', zorder=5)
    ax.vlines(3*(120/6), ymin=0, ymax=80, color=bg_color, lw=2, ls='--', zorder=5)
    ax.vlines(4*(120/6), ymin=0, ymax=80, color=bg_color, lw=2, ls='--', zorder=5)
    ax.vlines(5*(120/6), ymin=0, ymax=80, color=bg_color, lw=2, ls='--', zorder=5)

    ax.hlines(1*(80/5), xmin=0, xmax=120, color=bg_color, lw=2, ls='--', zorder=5)
    ax.hlines(2*(80/5), xmin=0, xmax=120, color=bg_color, lw=2, ls='--', zorder=5)
    ax.hlines(3*(80/5), xmin=0, xmax=120, color=bg_color, lw=2, ls='--', zorder=5)
    ax.hlines(4*(80/5), xmin=0, xmax=120, color=bg_color, lw=2, ls='--', zorder=5)
    
    return