import os
import json
import pandas as pd
from typing import Tuple
from src.extract_entities import extract_entities
from src.tactical_summary_builder import build_tactical_summary
from glob import glob

# Folder where all match files are stored
DATA_DIR = "\data\files1"


def read_json(file_path: str) -> str:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f"{os.path.basename(file_path)}:\n" + f.read()
    return ""


def read_csv(file_path: str) -> str:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return f"{os.path.basename(file_path)}:\n" + df.head(5).to_string(index=False)
    return ""


def get_last_n_team_matches(team_name: str, n=3):
    pattern = os.path.join(DATA_DIR, f"match_*__team_{team_name.replace(' ', '_')}_summary.json")
    files = sorted(glob(pattern), key=os.path.getmtime, reverse=True) 

    print(f"🔍 Found {len(files)} files for {team_name}")
    for f in files:
        print(f" - {f}")

    match_files = files[:n]
    summaries = []
    stat_files = []

    for f in match_files:
        try:
            with open(f, "r", encoding="utf-8") as j:
                data = json.load(j)
            match_id = os.path.basename(f).split("_")[1]
            summary = build_tactical_summary(team_name, data)
            summaries.append((os.path.basename(f), summary))

            stat_path = os.path.join(DATA_DIR, f"match_{match_id}_Team_stat.csv")
            if os.path.exists(stat_path):
                stat_files.append(stat_path)

        except Exception as e:
            print(f"❌ Error processing {f}: {e}")

    return summaries, stat_files




def build_prompt(query: str) -> Tuple[str, list]:
    entities = extract_entities(query)
    match_id = entities.get("match_id")
    player = entities.get("player_name")
    team1 = entities.get("team_1")
    team2 = entities.get("team_2")
    is_chat = entities.get("is_chat")

    context_parts = []
    loaded_files = []

    if match_id:
        summary_files = sorted(
            glob(os.path.join(DATA_DIR, f"match_{match_id}__team_*_summary.json"))
        )

        for summary_file in summary_files:
            team_name = os.path.basename(summary_file).split("__team_")[1].replace("_summary.json", "").replace("_", " ")
            with open(summary_file, "r", encoding="utf-8") as f:
                team_summary = json.load(f)
                tactical_summary = build_tactical_summary(team_name, team_summary)
                context_parts.append(f"[Tactical Overview: {team_name}]\n{tactical_summary}")
                loaded_files.append(summary_file)

        stats_file = os.path.join(DATA_DIR, f"match_{match_id}_Team_stat.csv")
        if os.path.exists(stats_file):
            context_parts.append(read_csv(stats_file))
            loaded_files.append(stats_file)

    elif player and match_id:
        player_file = os.path.join(DATA_DIR, f"match_{match_id}__player_{player.replace(' ', '_')}_summary.json")
        if os.path.exists(player_file):
            context_parts.append(read_json(player_file))
            loaded_files.append(player_file)

    elif is_chat and team1:
        summaries = get_last_n_team_matches(team1)
        for file_name, summary in summaries:
            context_parts.append(f"[Summary from {file_name}]\n{summary}")
            loaded_files.append(os.path.join(DATA_DIR, file_name))

        if summaries:
            last_match_id = summaries[-1][0].split("_")[1]
            stats_file = os.path.join(DATA_DIR, f"match_{last_match_id}_Team_stat.csv")
            if os.path.exists(stats_file):
                context_parts.append(read_csv(stats_file))
                loaded_files.append(stats_file)

    # 📜 System instructions for the LLM
    system_prompt = (
        "You are a professional football tactics analyst. "
        "Based on the match data provided, respond with a clear, confident, and structured analysis. "
        "Avoid speculation or personal phrases like 'I think' or 'maybe'. "
        "Explain what happened in clear tactical terms using the stats, summaries, and patterns. "
        "If multiple matches are shown, identify common patterns and explain how to beat or emulate the team."
    )

    context = "\n\n".join([p for p in context_parts if p.strip()])
    print("\n================= FINAL CONTEXT (TRUNCATED) =================")
    print(context[:1500])
    print("=============================================================\n")
    print(f"✅ Loaded files ({len(loaded_files)}):")
    for f in loaded_files:
        print(f" - {f}")

    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"
    return prompt, loaded_files

