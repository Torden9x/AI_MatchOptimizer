import os
import json
import pandas as pd
import streamlit as st
from src.extract_entities import extract_entities
from src.prompt_builder import build_prompt
from src.llm_interface import generate_answer

st.set_page_config(page_title="AI MatchOptimizer", layout="wide")
st.title("⚽ AI MatchOptimizer")
st.markdown("Ask your tactical football question:")

# User input
query = st.text_input("🧠 Your question", placeholder="e.g., How can we beat Sanfrecce Hiroshima?")

if query:
    prompt, loaded_files = build_prompt(query)
    if st.button("🧠 Generate Answer"):
        with st.spinner("Thinking like a coach..."):
            answer = generate_answer(prompt)
        st.markdown("### 💬 Answer")
        st.markdown(answer)
