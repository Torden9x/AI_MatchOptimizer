# ⚽ AI MatchOptimizer

**AI MatchOptimizer** is a tactical intelligence system built with football data science. It uses **StatsBomb event data**, **Hudl physical metrics**, and **AI modeling** to suggest the best **match lineup**, **formation**, and **strategy** — personalized for each opponent.

---

## 💡 What It Does

This system helps clubs and analysts:
- 🔍 **Analyze players** based on physical, tactical, and technical performance.
- 🧠 **Compare squads** (your team vs. opponent) to highlight key mismatches.
- 🧬 **Select the best lineup** using role-based clustering and tactical fit.
- 📐 **Recommend formations and strategies** based on match context.
- 💬 **Answer tactical questions** in natural language using a smart LLM-powered chatbot.

---

## 🧠 Why It Matters

AI MatchOptimizer is built to assist **clubs, federations, and scouts** — especially in preparation for international tournaments like:
- 🇸🇦 **2034 FIFA World Cup** (Saudi Arabia)
- 🏆 **AFC Asian Cup 2027**

---

## 🗂️ Project Structure
📦 AI_MatchOptimizer/ 
│ ├── 📂data/ │
 ├── match_<match_id>*.csv → Match-level data │ 
 ├── match<id>_TV.parquet → Enriched StatsBomb-style events │ 
 ├── match<id>_team<name>summary.json → Team summary per match │
 ├── match<id>_player<name>_summary.json → Player summary per match │ 
 ├── match_info.csv → Maps match_id to team names and score │ 
 ├── J1 2024_players.csv → Full season player stats │ 
 ├── J1 2024_teams.csv → Full season team stats │ 
 ├── teams_mapping.csv / players_mapping.csv / matches_mapping.csv │ 
 ├── JP1_physical.csv → Physical metrics (Hudl) │ 
 ├── Final Player Clustering.csv → Role-based clustering output 
 │ └── Player_Role_Profile.csv → Dictionary of role descriptions │ 
 ├── 📂src/
│   ├── 📌 Core Scripts
│   │   ├── app.py                              → Streamlit chatbot + dashboard
│   │   ├── prompt_builder.py                   → Builds LLM-ready prompts from data
│   │   ├── retriever_interface.py              → Chunk & retrieve file-based context
│   │   ├── extract_entities.py                 → Extracts player/team/match from query
│   │   ├── tactical_summary_builder.py         → Tactical report generator
│
│   ├── 🧠 Model + Chunking
│   │   ├── chunk_play_info.py                  → Extracts key play moments
│   │   ├── chunk_team_match_stats.py           → Chunks team stats files
│   │   ├── chunking.py                         → Converts all files to chunked DB
│   │   ├── vector_builder.py                   → Builds vector DB using embeddings
│
│   ├── 📊 Visuals + Match Stats
│   │   ├── visuals.py                          → xG maps, pass networks, duel maps
│
│   ├── 📓 Notebooks
│   │   ├── generate_tactical_summaries.ipynb   → Creates summary JSONs
│   │   ├── generate_enriched_player_summaries.ipynb → Player-level JSONs
│   │   ├── Prossing_Team.ipynb                 → Enriches team data
│   │   ├── pre_prossing.ipynb                  → Data preprocessing
│   │   ├── clustering.ipynb  
 ├── 📂docs/ │ 
 ├── FREE Hudl Statsbomb Guide.pdf 
 │ └── StatsBomb Open Data Specification.pdf 
 │ └── README.md ← You're here.

 ---

## 🚀 How to Run the Project

### Option 1: 📡 **Streamlit Community Cloud **
https://ai-matchoptimizer.streamlit.app

**No local setup needed. Just click & run.**

---

### Option 2: 🖥️ Local Setup

#### ✅ Step 1: Clone the Repo

```bash
git clone https://github.com/<your-username>/AI_MatchOptimizer.git
cd AI_MatchOptimizer
pip install -r requirements.txt

✅ Step 3: Prepare the Data
Place all data files (match CSVs, player stats, TV.parquet, summary JSONs) inside the data/ folder.

✅ Step 4 (Optional): Build Vector DB

python src/chunking.py

✅ Step 5: Launch the Chatbot Interface

streamlit run src/streamlit_app.py
Open your browser at http://localhost:8501
```
#### 💬 Suggested Prompts
Type	                Example
Match Report	        how did Sanfrecce Hiroshima win aginst Urawa Reds?
Player Performance	    how did Samuel Gustafson played against Sanfrecce Hiroshima?
Tactical Breakdown	    What style did Urawa Red Diamonds use across their last 3 matches?
Opponent Scouting	    How can we beat Sanfrecce Hiroshima?
Formation Advice	    What’s the best lineup and formation to face Sanfrecce Hiroshima?

📚 Data Sources
StatsBomb Open Data
Format: JSON & Parquet
Ref: [StatsBomb Open Events Structure.pdf](docs/StatsBomb Open Data Specification.pdf)



#### Match Mapping Files
Used to connect StatsBomb IDs ↔ Hudl IDs

teams_mapping.csv

players_mapping.csv

matches_mapping.csv

#### 🎯 How It Works
Event Data Extraction
Full match-level data: passes, shots, carries, duels, GK actions.

Player Profiling & Clustering
Assigns each player a role using performance-based clustering.

Physical Metric Integration
Uses distance, sprint, speed, HI runs, acceleration data (from Hudl).

Prompt Building & Retrieval
Smart prompts crafted with entity detection (player, match, team).

LLM Chatbot
Responds to tactical questions using retrieved match data.

#### Designed for Saudi Football's Future
AI MatchOptimizer can power:

🧠 Tactical planning for the Saudi National Team

🏋️ Fitness profiling of key players

📊 Match analysis for opponents in Asian Cup 2027 or World Cup 2034

#### This project uses StatsBomb Open Data and Hudl public data under their respective usage terms.
Licensed under the MIT License.