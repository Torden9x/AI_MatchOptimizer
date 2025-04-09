import os
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path

# Configuration
FAISS_PATH = r"E:/Ai_com/data/faiss_index"
PLAYER_CSV_PATH = r"E:\Ai_com\data\files1\J1 2024_players.csv"
TEAM_CSV_PATH = r"E:\Ai_com\data\files1\J1_teams_with_match_id.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Expected columns for validation
PLAYER_EXPECTED_COLUMNS = [
    "Wyscout id", 
    "Full name", 
    "Team", 
    "Primary position", 
    "player_style",
    "Age",
    "Birthday",
    "Market value",
    "Contract expires",
    "Matches played",
    "Minutes played",
    "Goals",
    "xG",
    "Assists",
    "xA",
    "Duels per 90",
    "Duels won, %",
    "Birth country",
    "Passport country",
    "Foot",
    "Height",
    "Weight",
    "On loan",
    "Successful defensive actions per 90",
    "Defensive duels per 90",
    "Defensive duels won, %",
    "Aerial duels per 90",
    "Aerial duels won, %",
    "Sliding tackles per 90",
    "PAdj Sliding tackles",
    "Shots blocked per 90",
    "Interceptions per 90",
    "PAdj Interceptions",
    "Fouls per 90",
    "Yellow cards",
    "Yellow cards per 90",
    "Red cards",
    "Red cards per 90",
    "Successful attacking actions per 90",
    "Goals per 90",
    "Non-penalty goals",
    "Non-penalty goals per 90",
    "xG per 90",
    "Head goals",
    "Head goals per 90",
    "Shots",
    "Shots per 90",
    "Shots on target, %",
    "Goal conversion, %",
    "Assists per 90",
    "Crosses per 90",
    "Accurate crosses, %",
    "Crosses from left flank per 90",
    "Accurate crosses from left flank, %",
    "Crosses from right flank per 90",
    "Accurate crosses from right flank, %",
    "Crosses to goalie box per 90",
    "Dribbles per 90",
    "Successful dribbles, %",
    "Offensive duels per 90",
    "Offensive duels won, %",
    "Touches in box per 90",
    "Progressive runs per 90",
    "Accelerations per 90",
    "Received passes per 90",
    "Received long passes per 90",
    "Fouls suffered per 90",
    "Passes per 90",
    "Accurate passes, %",
    "Forward passes per 90",
    "Accurate forward passes, %",
    "Back passes per 90",
    "Accurate back passes, %",
    "Short / medium passes per 90",
    "Accurate short / medium passes, %",
    "Long passes per 90",
    "Accurate long passes, %",
    "Average pass length, m",
    "Average long pass length, m",
    "xA per 90",
    "Shot assists per 90",
    "Second assists per 90",
    "Third assists per 90",
    "Smart passes per 90",
    "Accurate smart passes, %",
    "Key passes per 90",
    "Passes to final third per 90",
    "Accurate passes to final third, %",
    "Passes to penalty area per 90",
    "Accurate passes to penalty area, %",
    "Through passes per 90",
    "Accurate through passes, %",
    "Deep completions per 90",
    "Deep completed crosses per 90",
    "Progressive passes per 90",
    "Accurate progressive passes, %",
    "Accurate vertical passes, %",
    "Vertical passes per 90",
    "Conceded goals",
    "Conceded goals per 90",
    "Shots against",
    "Shots against per 90",
    "Clean sheets",
    "Save rate, %",
    "xG against",
    "xG against per 90",
    "Prevented goals",
    "Prevented goals per 90",
    "Back passes received as GK per 90",
    "Exits per 90",
    "Aerial duels per 90.1",
    "Free kicks per 90",
    "Direct free kicks per 90",
    "Direct free kicks on target, %",
    "Corners per 90",
    "Penalties taken",
    "Penalty conversion, %"
]

TEAM_EXPECTED_COLUMNS = [
    "match_id",
    "Team",
    "Date",
    "Match",
    "xG",
    "xGA",
    "xGD",
    "Open Play xG",
    "Open Play xGA",
    "Open Play xGD",
    "Set Piece xG",
    "Set Piece xGA",
    "Set Piece xGD",
    "npxG",
    "npxGA",
    "npxGD",
    "Goals",
    "Goals Conceded",
    "GD",
    "GD-xGD",
    "Possession",
    "Field Tilt",
    "Avg Pass Height",
    "xT",
    "xT Against",
    "Passes in Opposition Half",
    "Passes into Box",
    "Shots",
    "Shots Faced",
    "Shots per 1.0 xT",
    "Shots Faced per 1.0 xT Against",
    "PPDA",
    "High Recoveries",
    "High Recoveries Against",
    "Crosses",
    "Corners",
    "Fouls",
    "On-Ball Pressure",
    "On-Ball Pressure Share",
    "Off-Ball Pressure",
    "Off-Ball Pressure Share",
    "Game Control",
    "Game Control Share",
    "Throw-Ins into the Box"
]

def validate_csv(df: pd.DataFrame, expected_columns: List[str], file_name: str) -> bool:
    """Validate if CSV has the expected columns"""
    missing_cols = [col for col in expected_columns if col not in df.columns]
    
    if missing_cols:
        print(f"Error: {file_name} is missing columns: {missing_cols}")
        return False
    
    print(f"✓ {file_name} validated successfully")
    return True

def load_csv_data() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and validate CSV files for players and teams"""
    try:
        # Load player data
        player_df = pd.read_csv(PLAYER_CSV_PATH)
        if not validate_csv(player_df, PLAYER_EXPECTED_COLUMNS, PLAYER_CSV_PATH):
            return None, None
        
        # Load team match data
        team_df = pd.read_csv(TEAM_CSV_PATH)
        if not validate_csv(team_df, TEAM_EXPECTED_COLUMNS, TEAM_CSV_PATH):
            return None, None
            
        return player_df, team_df
    
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None, None

def create_player_documents(player_df: pd.DataFrame) -> List[Document]:
    """Create Document objects from player data"""
    documents = []
    
    for _, row in player_df.iterrows():
        # Create human-readable content for embedding
        content = f"""
        Player Profile: {row['Full name']}
        Team: {row['Team']}
        Position: {row['Primary position']}
        Player ID: {row['Wyscout id']}
        Playing Style: {row['player_style']}
        Age: {row.get('Age', 'N/A')}
        Height: {row.get('Height', 'N/A')} cm
        Weight: {row.get('Weight', 'N/A')} kg
        Preferred Foot: {row.get('Foot', 'N/A')}
        Market Value: {row.get('Market value', 'N/A')}

        Season Statistics:
        - Matches Played: {row.get('Matches played', 'N/A')}
        - Minutes Played: {row.get('Minutes played', 'N/A')}
        - Goals: {row.get('Goals', 'N/A')} (xG: {row.get('xG', 'N/A')})
        - Assists: {row.get('Assists', 'N/A')} (xA: {row.get('xA', 'N/A')})
        - Non-Penalty Goals: {row.get('Non-penalty goals', 'N/A')}
        - Shots: {row.get('Shots', 'N/A')} ({row.get('Shots on target, %', 'N/A')}% on target)
        - Key Passes: {row.get('Key passes per 90', 'N/A')} per 90
        - Dribbles: {row.get('Dribbles per 90', 'N/A')} per 90 ({row.get('Successful dribbles, %', 'N/A')}% success)

        Per 90 Metrics:
        - Goals: {row.get('Goals per 90', 'N/A')}
        - xG: {row.get('xG per 90', 'N/A')}
        - Assists: {row.get('Assists per 90', 'N/A')}
        - xA: {row.get('xA per 90', 'N/A')}
        - Touches in Box: {row.get('Touches in box per 90', 'N/A')}
        - Progressive Runs: {row.get('Progressive runs per 90', 'N/A')}

        Defensive Actions:
        - Defensive Duels: {row.get('Defensive duels per 90', 'N/A')} ({row.get('Defensive duels won, %', 'N/A')}% won)
        - Aerial Duels: {row.get('Aerial duels per 90', 'N/A')} ({row.get('Aerial duels won, %', 'N/A')}% won)
        - Interceptions: {row.get('Interceptions per 90', 'N/A')}
        - Tackles: {row.get('Sliding tackles per 90', 'N/A')}

        Passing Statistics:
        - Pass Accuracy: {row.get('Accurate passes, %', 'N/A')}%
        - Progressive Passes: {row.get('Progressive passes per 90', 'N/A')}
        - Key Passes: {row.get('Key passes per 90', 'N/A')}
        - Passes to Final Third: {row.get('Passes to final third per 90', 'N/A')}

        Additional Attributes:
        - Fouls Committed: {row.get('Fouls per 90', 'N/A')} per 90
        - Fouls Suffered: {row.get('Fouls suffered per 90', 'N/A')} per 90
        - Yellow Cards: {row.get('Yellow cards', 'N/A')}
        - Red Cards: {row.get('Red cards', 'N/A')}
        """
        # Create structured metadata for filtering
        metadata = {
            "type": "player_profile",
            "player_name": row['Full name'],
            "player_id": str(row['Wyscout id']),
            "team_name": row['Team'],
            "position": row['Primary position'],
            "style": row['player_style'],
            "source_file": PLAYER_CSV_PATH
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Created {len(documents)} player profile documents")
    return documents

def create_team_match_documents(team_df: pd.DataFrame) -> List[Document]:
    """Create Document objects from team match data"""
    documents = []
    
    for _, row in team_df.iterrows():
        # Create human-readable content for embedding
        content = f"""
Match Report for {row['Team']}
Match ID: {row['match_id']}
Date: {row['Date']}
Match: {row['Match']}

Match Statistics:
- Expected Goals (xG): {row.get('xG', 'N/A')}
- Expected Goals Against (xGA): {row.get('xGA', 'N/A')}
- xG Difference (xGD): {row.get('xGD', 'N/A')}
- Open Play xG: {row.get('Open Play xG', 'N/A')}
- Open Play xGA: {row.get('Open Play xGA', 'N/A')}
- Set Piece xG: {row.get('Set Piece xG', 'N/A')}
- Set Piece xGA: {row.get('Set Piece xGA', 'N/A')}
- Non-Penalty xG (npxG): {row.get('npxG', 'N/A')}
- Goals Scored: {row.get('Goals', 'N/A')}
- Goals Conceded: {row.get('Goals Conceded', 'N/A')}
- Goal Difference: {row.get('GD', 'N/A')}
- Possession: {row.get('Possession', 'N/A')}%
- Field Tilt: {row.get('Field Tilt', 'N/A')}%
- Passes in Opposition Half: {row.get('Passes in Opposition Half', 'N/A')}
- Passes into Box: {row.get('Passes into Box', 'N/A')}
- Shots Taken: {row.get('Shots', 'N/A')}
- Shots Faced: {row.get('Shots Faced', 'N/A')}
- Crosses: {row.get('Crosses', 'N/A')}
- Corners: {row.get('Corners', 'N/A')}
- Fouls Committed: {row.get('Fouls', 'N/A')}
- PPDA (Passes per Defensive Action): {row.get('PPDA', 'N/A')}
- High Recoveries: {row.get('High Recoveries', 'N/A')}
- Game Control: {row.get('Game Control', 'N/A')}%
- Expected Threat (xT): {row.get('xT', 'N/A')}
- xT Against: {row.get('xT Against', 'N/A')}

Advanced Metrics:
- On-Ball Pressure: {row.get('On-Ball Pressure', 'N/A')}
- Off-Ball Pressure: {row.get('Off-Ball Pressure', 'N/A')}
- Throw-Ins into the Box: {row.get('Throw-Ins into the Box', 'N/A')}
- Average Pass Height: {row.get('Avg Pass Height', 'N/A')}
"""
        # Create structured metadata for filtering
        metadata = {
            "type": "team_match_stat",
            "team_name": row['Team'],
            "match_id": str(row['match_id']),
            "date": row['Date'],
            "match": row['Match'], 
            "source_file": TEAM_CSV_PATH
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Created {len(documents)} team match documents")
    return documents

def store_documents_in_faiss(documents: List[Document], index_dir: str):
    """Store or merge documents into FAISS index"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        os.makedirs(index_dir, exist_ok=True)

        faiss_index_file = Path(index_dir) / "index.faiss"
        if faiss_index_file.exists():
            print("🌀 Existing FAISS index found, loading and merging...")
            vectorstore = FAISS.load_local(
                folder_path=index_dir,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(documents)
        else:
            print("🆕 Creating new FAISS index...")
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )

        vectorstore.save_local(index_dir)
        print(f"✅ FAISS index saved/merged at {index_dir}")
        return vectorstore

    except Exception as e:
        print(f"❌ Error saving to FAISS: {e}")
        return None

def main():
    print("🏆 Football Analytics Data Ingestion Started")
    
    # Load CSV data
    player_df, team_df = load_csv_data()
    if player_df is None or team_df is None:
        print("❌ Data loading failed. Exiting.")
        return
    
    # Create documents
    player_documents = create_player_documents(player_df)
    team_documents = create_team_match_documents(team_df)
    all_documents = player_documents + team_documents
    
    # Store in ChromaDB
    FAISS_DB_PATH = r"E:/Ai_com/data/faiss_index"
    vectorstore = store_documents_in_faiss(all_documents, FAISS_DB_PATH)
    if vectorstore:
        print(f"✅ Successfully processed and stored {len(all_documents)} documents")
        print(f"📊 Player profiles: {len(player_documents)}")
        print(f"📈 Team match stats: {len(team_documents)}")
        print(f"🔍 Ready for semantic search and LLM queries")
    else:
        print("❌ Data ingestion failed")

if __name__ == "__main__":
    main()