# match_report.py
import streamlit as st
import pandas as pd
import os
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from visuals import (
    extract_team_names,
    extract_players_info,
    get_passes_df,
    get_passes_between_df,
    pass_network_visualization,
    get_defensive_action_df,
    get_da_count_df,
    defensive_block,
    draw_progressive_pass_map,
    plot_shotmap,
    match_stat,
    Final_third_entry,
    zone14hs,
    Pass_end_zone,
    plotting_match_stats,
    Chance_creating_zone,
    box_entry,
    Crosses,
    HighTO,
    plot_congestion
)
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
path_eff1 = [path_effects.Stroke(linewidth=1.5, foreground=line_color), path_effects.Normal()]
plt.rcParams['figure.facecolor'] = bg_color

event_data_dir = "Stat"
match_info_path = os.path.join(event_data_dir, "match_info.csv")
def load_match_info():
    df = pd.read_csv(match_info_path)
    df["match_file"] = df["match_id"].apply(lambda mid: f"match_{mid}_.csv")
    df["match_name"] = df.apply(
        lambda row: f'{row["home_team"]} vs {row["away_team"]} ({row["home_score"]}-{row["away_score"]})',
        axis=1
    )
    return df

# ---- SIDEBAR ----
st.sidebar.title("📊 Match Report Generator")

# Load match info
match_info_df = load_match_info()

# Match selection dropdown
selected_match_name = st.sidebar.selectbox("Select Match", match_info_df["match_name"].tolist())

# Retrieve corresponding file name
selected_match_file = match_info_df.loc[match_info_df["match_name"] == selected_match_name, "match_file"].values[0]

# Full path to selected file
selected_match_path = os.path.join(event_data_dir, selected_match_file)

if selected_match_path:
    df = pd.read_csv(os.path.join(event_data_dir, selected_match_path))
    st.sidebar.success("Match loaded successfully!")

    # Extract teams
    hteamName, ateamName = extract_team_names(df)
    st.title(f"📝 Match Report: {hteamName} vs {ateamName}")
    
    # Color config
    hcol, acol = '#ff4b44', '#00a0de'
    
    # Extract players
    players_df, df = extract_players_info(df)
    passes_df = get_passes_df(df)
    def_df = get_defensive_action_df(df)

    # -- MATCH STATISTICS --
    st.header("📈 General Match Statistics")

    # Run stat calculation
    (
        hposs, aposs, hft, aft,
        dfpass_h, acc_pass_h, Lng_ball_h, Lng_ball_acc_h,
        hTackles, hTackles_Lost, hInterceptions, hClearances, hAerials, hAerials_Lost,
        home_ppda, PPS_home, pass_seq_10_more_home,
        dfpass_a, acc_pass_a, Lng_ball_a, Lng_ball_acc_a,
        aTackles, aTackles_Lost, aInterceptions, aClearances, aAerials, aAerials_Lost,
        away_ppda, PPS_away, pass_seq_10_more_away
    ) = match_stat(df, hteamName, ateamName)

    fig, ax = plt.subplots(figsize=(12, 8))
    stat_df = plotting_match_stats(
        ax, hteamName, ateamName, hposs, aposs, hft, aft,
        dfpass_h, acc_pass_h, Lng_ball_h, Lng_ball_acc_h,
        hTackles, hTackles_Lost, hInterceptions, hClearances, hAerials, hAerials_Lost,
        home_ppda, PPS_home, pass_seq_10_more_home,
        dfpass_a, acc_pass_a, Lng_ball_a, Lng_ball_acc_a,
        aTackles, aTackles_Lost, aInterceptions, aClearances, aAerials, aAerials_Lost,
        away_ppda, PPS_away, pass_seq_10_more_away,
        bg_color, hcol, acol,path_eff1
    )

    st.pyplot(fig)
    st.dataframe(stat_df)



    # -- PASS NETWORKS --
    st.header("🔗 Passing Network")
    for team_name, color, is_away in [(hteamName, hcol, False), (ateamName, acol, True)]:
        passes_between, avg_locs, players_info = get_passes_between_df(team_name, passes_df, players_df, df)
        fig, summary = pass_network_visualization(team_name, passes_between, avg_locs, color, is_away, players_info, passes_df)
        st.pyplot(fig)
        st.json(summary)

    # -- DEFENSIVE BLOCK --
    st.header("🛡️ Defensive Block")
    for team_name, color, is_away in [(hteamName, hcol, False), (ateamName, acol, True)]:
        da_avg = get_da_count_df(team_name, def_df, players_df)
        fig, summary = defensive_block(team_name, da_avg, is_away, def_df, color)
        st.pyplot(fig)
        st.json(summary)

    # -- PROGRESSIVE PASSES --
    st.header("📤 Progressive Pass Map")
    for team_name, color, is_away in [(hteamName, hcol, False), (ateamName, acol, True)]:
        fig, stats = draw_progressive_pass_map(team_name, df, is_away, color, ateamName, hteamName)
        st.pyplot(fig)  # <-- THIS LINE DISPLAYS THE VISUAL
        st.json(stats)  # Optional: keep if you want to show stats as JSON


    # -- SHOTMAP & GOALPOST --
    st.header("🎯 Shot Map & Goalpost Analysis")
    fig, ax = plt.subplots(figsize=(16, 10))
    shot_summary = plot_shotmap(ax, df, hteamName, ateamName)
    st.pyplot(fig)
    st.json(shot_summary)

    # -- FINAL THIRD ENTRY --
    st.header("🚪 Final Third Entry")
    for team_name, color in [(hteamName, hcol), (ateamName, acol)]:
        fig, ax = plt.subplots(figsize=(12, 8))
        stats = Final_third_entry(ax, team_name, color, df, ateamName, hteamName)
        st.pyplot(fig)
        st.json(stats)

    # -- ZONE 14 & HALFSPACES --
    st.header("🎯 Zone14 & Halfspace Passes")
    for team_name, color in [(hteamName, hcol), (ateamName, acol)]:
        fig, ax = plt.subplots(figsize=(12, 8))
        stats = zone14hs(ax, team_name, color, ateamName, hteamName, df)
        st.pyplot(fig)
        st.json(stats)

    # -- PASS END ZONE HEATMAP --
    st.header("📌 Pass End Zones")
    for team_name, cm in [(hteamName, 'Reds'), (ateamName, 'Blues')]:
        fig, ax = plt.subplots(figsize=(12, 8))
        Pass_end_zone(ax, team_name, cm, df, ateamName, hteamName)
        st.pyplot(fig)

    # -- CHANCE CREATING ZONES --
    st.header("🎨 Chance Creating Zone")
    for team_name, color, cm in [(hteamName, hcol, 'Oranges'), (ateamName, acol, 'Purples')]:
        fig, ax = plt.subplots(figsize=(12, 8))
        stats = Chance_creating_zone(ax, team_name, cm, color, ateamName, hteamName, df)
        st.pyplot(fig)
        st.json(stats)

    # -- BOX ENTRIES --
    st.header("🧱 Box Entries")
    fig, ax = plt.subplots(figsize=(12, 8))
    stats = box_entry(ax, df, ateamName, hteamName)
    st.pyplot(fig)
    st.json(stats)

    # -- CROSSES --
    st.header("🛬 Cross Analysis")
    fig, ax = plt.subplots(figsize=(12, 8))
    stats = Crosses(ax, df, ateamName, hteamName)
    st.pyplot(fig)
    st.json(stats)

    # -- HIGH TURNOVERS --
    st.header("⚡ High Turnovers")
    fig, ax = plt.subplots(figsize=(12, 8))
    stats = HighTO(ax, ateamName, hteamName, df)
    st.pyplot(fig)
    st.json(stats)

    # -- CONGESTION ZONES --
    st.header("🧱 Congestion Zones")
    from highlight_text import ax_text  # for better text control
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_congestion(ax, ateamName, hteamName, df, ax_text)
    st.pyplot(fig)

