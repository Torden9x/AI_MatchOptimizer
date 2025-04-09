import re
import pandas as pd
from rapidfuzz import fuzz, process
from typing import Dict
# Load player names from CSV
PLAYER_CSV_PATH = "data/files1/J1 2024_players.csv"
TEAM_CSV_PATH = "data/files1/J1_teams_with_match_id.csv"
MATCH_INFO_PATH = "data/files1/match_info.csv"
J1_TEAMS = [
    "Urawa Reds", "Kawasaki Frontale", "Yokohama F. Marinos", "Sanfrecce Hiroshima",
    "Avispa Fukuoka", "Nagoya Grampus", "FC Tokyo", "Cerezo Osaka", "Kashima Antlers",
    "Gamba Osaka", "Consadole Sapporo", "Albirex Niigata", "Shonan Bellmare", "Kashiwa Reysol"
]
try:
    player_df = pd.read_csv(PLAYER_CSV_PATH)
    team_df = pd.read_csv(TEAM_CSV_PATH)
    match_info_df = pd.read_csv(MATCH_INFO_PATH)

    PLAYER_NAMES = player_df["Full name"].dropna().unique().tolist()
    TEAM_NAMES = team_df["Team"].dropna().unique().tolist()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    PLAYER_NAMES = []
    TEAM_NAMES = []
    match_info_df = pd.DataFrame()


def extract_entities(query: str) -> Dict[str, str]:
    query = query.strip()

    # Try to detect a match ID directly
    match_ids = re.findall(r"\b(?:match\s*)?(\d{6,7})\b", query)
    match_id = match_ids[0] if match_ids else None

    # Player name
    matched_player = process.extractOne(query, PLAYER_NAMES, scorer=fuzz.token_set_ratio)
    player_name = matched_player[0] if matched_player and matched_player[1] > 85 else None

    # Team names (fuzzy match)
    matched_teams = process.extract(query, TEAM_NAMES, scorer=fuzz.token_set_ratio, limit=2)
    team_names = [t[0] for t in matched_teams if t[1] > 60]

    team_1 = team_names[0] if len(team_names) > 0 else None
    team_2 = team_names[1] if len(team_names) > 1 else None

    # Try to infer match_id if two teams are found
    if not match_id and team_1 and team_2 and not match_info_df.empty:
        possible_matches = match_info_df[
            ((match_info_df["home_team"] == team_1) & (match_info_df["away_team"] == team_2)) |
            ((match_info_df["home_team"] == team_2) & (match_info_df["away_team"] == team_1))
        ].sort_values("match_id", ascending=False)
        if not possible_matches.empty:
            match_id = str(possible_matches.iloc[0]["match_id"])
        # Infer match from player + opponent team
    if not match_id and player_name and len(team_names) == 1 and not match_info_df.empty:
        player_team_row = player_df[player_df["Full name"] == player_name]
        if not player_team_row.empty:
            player_team = player_team_row.iloc[0]["Team"]
            opponent_team = team_names[0]

            possible_matches = match_info_df[
                ((match_info_df["home_team"] == player_team) & (match_info_df["away_team"] == opponent_team)) |
                ((match_info_df["away_team"] == player_team) & (match_info_df["home_team"] == opponent_team))
            ]

            if not possible_matches.empty:
                match_id = str(possible_matches.iloc[0]["match_id"])


    # Tactical intent detection
    tactical_keywords = ["beat", "how", "play", "strategy", "tactics"]
    is_tactical = any(word in query.lower() for word in tactical_keywords) and (team_1 or team_2)

    # Mark it chat only if we got nothing useful
    is_chat = not is_tactical and not player_name and not match_id and not team_names

    return {
        "is_chat": is_chat,
        "player_name": player_name,
        "team_1": team_1,
        "team_2": team_2,
        "match_id": match_id
    }
