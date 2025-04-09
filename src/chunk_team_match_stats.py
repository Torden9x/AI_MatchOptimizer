import pandas as pd
import uuid
from chromadb import Client
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the team match data CSV
df = pd.read_csv("E:/Ai_com/data/files1/J1_teams_with_match_id.csv")
df = df.fillna("N/A").astype(str)

# Initialize Chroma client
chroma_client = Client()
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prepare chunks
team_chunks = []

for _, row in df.iterrows():
    team = row["Team"]
    match_id = row["match_id"]
    match = row.get("Match", "Unknown Match")
    date = row.get("Date", "Unknown Date")
    opponent = "N/A"  # Optional enhancement: parse this from `Match` if it's like "Team A vs Team B"

    # Drop metadata columns for clean stats
    stats = row.drop(["Team", "Match", "Date", "match_id"], errors="ignore").to_dict()

    text = f"""
    Match Performance Summary:
    - Team: {team}
    - Match ID: {match_id}
    - Match: {match}
    - Date: {date}
    - Stats: {stats}
    """

    metadata = {
        "team_name": team,
        "match_id": match_id,
        "match_name": match,
        "match_date": date,
        "type": "team_match_stat",
        "source_file": "J1_teams_with_match_id.csv"
    }

    team_chunks.append({
        "id": f"{team}_{match_id}",
        "text": text.strip(),
        "metadata": metadata
    })

# Save chunks to Chroma
collection = chroma_client.get_or_create_collection("ai_matchoptimizer")
for chunk in team_chunks:
    collection.add(
        documents=[chunk["text"]],
        metadatas=[chunk["metadata"]],
        ids=[chunk["id"]]
    )

print("✅ Team match stats from J1_teams_with_match_id.csv successfully chunked and stored.")
