# chunking_faiss.py
import os
import json
import pandas as pd
import re
from typing import List, Dict
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Paths
DATA_FOLDER = "E:/Ai_com/data/files1"
FAISS_DIR = "E:/Ai_com/data/faiss_index"

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def parse_filename(filename: str) -> Dict[str, str]:
    metadata = {}
    if match := re.search(r'match_(\d+)', filename):
        metadata['match_id'] = match.group(1)
    if player := re.search(r'player_(.*?)_summary', filename):
        metadata['player_name'] = player.group(1)
    if team := re.search(r'team_(.*?)_summary', filename):
        metadata['team_name'] = team.group(1)
    return metadata

def process_player_summary(file_path: str) -> List[Document]:
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    filename = os.path.basename(file_path)
    metadata = parse_filename(filename) | {'type': 'player_summary'}
    summary = data.get("summary", "")
    actions = data.get("key_actions", [])
    content = f"Player Summary:\n{summary}\n\nKey Actions:\n" + "\n".join(f"- {a}" for a in actions)
    metadata.update({k: data[k] for k in ('player_name', 'team', 'position') if k in data})
    return [Document(page_content=content, metadata=metadata)]

def process_team_summary(file_path: str) -> List[Document]:
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    filename = os.path.basename(file_path)
    metadata = parse_filename(filename) | {'type': 'team_summary'}
    summary = data.get("summary", "")
    patterns = data.get("key_tactical_patterns", [])
    content = f"Team Summary:\n{summary}\n\nKey Tactical Patterns:\n" + "\n".join(f"- {p}" for p in patterns)
    metadata.update({k: data[k] for k in ('team', 'formation') if k in data})
    return [Document(page_content=content, metadata=metadata)]

def process_csv_stats(file_path: str, name_col: str, stat_type: str) -> List[Document]:
    docs = []
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            if name_col not in row or pd.isna(row[name_col]):
                continue
            name = row[name_col]
            stats = {k: v for k, v in row.items() if k != name_col and not pd.isna(v)}
            content = f"{stat_type.title()} Stats for {name}:\n" + "\n".join(f"{k}: {v}" for k, v in stats.items())
            meta = parse_filename(os.path.basename(file_path)) | {'type': stat_type, name_col.lower(): name}
            docs.append(Document(page_content=content, metadata=meta))
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
    return docs

def process_files() -> List[Document]:
    all_docs = []
    for root, _, files in os.walk(DATA_FOLDER):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith("_player_stats.csv"):
                all_docs += process_csv_stats(path, 'Name', 'player_stats')
            elif file.endswith("_Team_stat.csv"):
                all_docs += process_csv_stats(path, 'Team_Name', 'team_stats')
            elif file.endswith("_summary.json") and "__player_" in file:
                all_docs += process_player_summary(path)
            elif file.endswith("_summary.json") and "__team_" in file:
                all_docs += process_team_summary(path)
    return all_docs

def main():
    print("Processing files...")
    documents = process_files()
    print(f"Total documents: {len(documents)}")
    if not documents:
        print("No documents created.")
        return
    print("Saving to FAISS index...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_DIR)
    print("✅ FAISS index created and saved.")

if __name__ == "__main__":
    main()
