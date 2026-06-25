#!/usr/bin/env python3
"""
Streamlit Web Interface for Redrob AI Candidate Ranking System
WITH Sentence-Transformers - Deploy on Hugging Face Spaces
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ranking modules
from src.data_loader import CandidateLoader
from src.jd_parser import JDParser
from src.scorer import CandidateScorer
from src.utils import setup_logging

setup_logging('INFO')

# Configure page
st.set_page_config(
    page_title="AI Candidate Ranking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #1f77b4; margin-bottom: 1rem; font-weight: 700; }
.sub-header { font-size: 1.2rem; color: #2c3e50; margin-bottom: 1.5rem; }
.score-high { color: #2ca02c; font-weight: bold; }
.score-medium { color: #ff7f0e; font-weight: bold; }
.score-low { color: #d62728; font-weight: bold; }
.candidate-card { padding:1rem; border-radius:0.5rem; border:1px solid #e0e0e0; margin-bottom:1rem; background:#f8f9fa; }
.reasoning-text { color:#555; font-size:0.9rem; padding:0.5rem; background:#f1f3f5; border-radius:0.25rem; border-left:3px solid #1f77b4; }
.metric-card { background-color:#f8f9fa; padding:1rem; border-radius:0.5rem; text-align:center; border:1px solid #e0e0e0; }
.metric-value { font-size:2rem; font-weight:bold; color:#1f77b4; }
.metric-label { color:#666; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ranked_candidates' not in st.session_state:
    st.session_state.ranked_candidates = None
if 'ranking_time' not in st.session_state:
    st.session_state.ranking_time = None
if 'status' not in st.session_state:
    st.session_state.status = "idle"
if 'candidate_count' not in st.session_state:
    st.session_state.candidate_count = 0


@st.cache_resource
def load_embedding_model():
    """Cache the embedding model - loads once."""
    from sentence_transformers import SentenceTransformer
    with st.spinner("Loading embedding model (may take 30-60 seconds)..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model


def display_candidate(candidate_data: Dict[str, Any], rank: int):
    """Display a single candidate in a formatted card."""
    candidate_id = candidate_data['candidate_id']
    score = candidate_data['score']
    reasoning = candidate_data.get('reasoning', 'No reasoning provided')
    components = candidate_data.get('components', {})
    
    if score >= 0.7:
        score_class = "score-high"
        score_emoji = "🌟"
    elif score >= 0.5:
        score_class = "score-medium"
        score_emoji = "⭐"
    else:
        score_class = "score-low"
        score_emoji = "📌"
    
    with st.container():
        st.markdown(f"""
        <div class="candidate-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div><h4>#{rank} - {candidate_id}</h4></div>
                <div><span class="{score_class}">{score_emoji} Score: {score:.3f}</span></div>
            </div>
            <div class="reasoning-text" style="margin-top: 0.5rem;">
                <strong>Reasoning:</strong> {reasoning}
            </div>
            <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <span style="background-color: #e9ecef; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;">
                    🔍 Semantic: {(components.get('semantic', 0) * 100):.0f}%
                </span>
                <span style="background-color: #e9ecef; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;">
                    🛠️ Skills: {(components.get('skill_match', 0) * 100):.0f}%
                </span>
                <span style="background-color: #e9ecef; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;">
                    📈 Experience: {(components.get('experience', 0) * 100):.0f}%
                </span>
                <span style="background-color: #e9ecef; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;">
                    👤 Behavioral: {(components.get('behavioral', 0) * 100):.0f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_metrics(ranked_candidates: List[Dict[str, Any]]):
    """Display summary metrics."""
    scores = [c['score'] for c in ranked_candidates]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(ranked_candidates)}</div>
            <div class="metric-label">Total Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{np.mean(scores):.3f}</div>
            <div class="metric-label">Avg Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{np.max(scores):.3f}</div>
            <div class="metric-label">Max Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{np.min(scores):.3f}</div>
            <div class="metric-label">Min Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{np.std(scores):.3f}</div>
            <div class="metric-label">Std Dev</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    st.markdown('<p class="main-header">🎯 AI-Powered Candidate Ranking System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using Sentence-Transformers for semantic understanding</p>',
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("📁 Data Input")
        
        candidates_file = st.file_uploader(
            "Upload candidates.jsonl",
            type=['jsonl', 'gz'],
            help="Upload the candidates file in JSONL or gzipped JSONL format"
        )
        jd_file = st.file_uploader(
            "Upload job description (jd.txt or .md)",
            type=['txt', 'md'],
            help="Upload the job description as a text file"
        )
        
        st.subheader("🎯 Ranking Parameters")
        
        top_k = st.slider(
            "Number of candidates to rank (Top K)",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Number of candidates to include in the final output"
        )
        
        st.subheader("📊 Sample Size")
        
        sample_size = st.slider(
            "Sample size (0 = all candidates)",
            min_value=0,
            max_value=100000,
            value=0,
            step=100,
            help="For testing, limit number of candidates processed. 0 = process all candidates."
        )
        
        if sample_size == 0:
            st.info("📊 Processing ALL candidates (100K)")
        else:
            st.info(f"📊 Processing {sample_size:,} candidates ({(sample_size/100000)*100:.1f}% of dataset)")
        
        if sample_size == 0:
            est_time = "4-5 minutes"
        elif sample_size <= 100:
            est_time = "~5 seconds"
        elif sample_size <= 1000:
            est_time = "~10 seconds"
        elif sample_size <= 5000:
            est_time = "~30 seconds"
        elif sample_size <= 10000:
            est_time = "~1 minute"
        elif sample_size <= 25000:
            est_time = "~2 minutes"
        elif sample_size <= 50000:
            est_time = "~3 minutes"
        else:
            est_time = "~4 minutes"
        
        st.caption(f"⏱️ Estimated time: {est_time}")
        
        run_button = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)
        
        if st.session_state.status != "idle":
            st.info(f"Status: {st.session_state.status}")
        if st.session_state.ranking_time:
            st.success(f"⏱️ Completed in {st.session_state.ranking_time:.2f}s")
        if st.session_state.candidate_count > 0:
            st.info(f"📊 Processed {st.session_state.candidate_count:,} candidates")
    
    # Main content - JD Display
    if jd_file:
        jd_text = jd_file.read().decode('utf-8')
        st.subheader("📄 Job Description")
        st.text_area("Job Description", value=jd_text, height=200, disabled=True)
    
    # Ranking execution
    if run_button:
        try:
            st.session_state.ranked_candidates = None
            st.session_state.status = "loading"
            
            if not candidates_file or not jd_file:
                st.error("❌ Please upload both candidates file and job description.")
                st.session_state.status = "idle"
            else:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl.gz', delete=False) as f:
                    f.write(candidates_file.getvalue())
                    candidates_path = f.name
                
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
                    f.write(jd_file.getvalue())
                    jd_path = f.name
                
                try:
                    with st.spinner("🔄 Loading candidates..."):
                        loader = CandidateLoader(candidates_path)
                        candidates = loader.load_all(
                            limit=sample_size if sample_size > 0 else None
                        )
                    
                    with st.spinner("🔄 Processing candidates..."):
                        start_time = time.time()
                        
                        with open(jd_path, 'r', encoding='utf-8') as f:
                            jd_text = f.read()
                        
                        model = load_embedding_model()
                        jd_parser = JDParser(jd_text)
                        jd_embedding = model.encode(jd_text, normalize_embeddings=True)
                        scorer = CandidateScorer(jd_parser, model, jd_embedding)
                        
                        scored_candidates = []
                        progress_bar = st.progress(0)
                        
                        for i, candidate in enumerate(candidates):
                            score, components, reasoning = scorer.score(candidate)
                            scored_candidates.append({
                                'candidate_id': candidate.get('candidate_id', f'CAND_{i:07d}'),
                                'score': score,
                                'reasoning': reasoning,
                                'components': components
                            })
                            progress_bar.progress((i + 1) / len(candidates))
                        
                        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
                        top_candidates = scored_candidates[:top_k]
                        for i, cand in enumerate(top_candidates, 1):
                            cand['rank'] = i
                        
                        ranking_time = time.time() - start_time
                        st.session_state.ranked_candidates = top_candidates
                        st.session_state.ranking_time = ranking_time
                        st.session_state.status = "completed"
                        st.session_state.candidate_count = len(candidates)
                        
                        st.success(f"✅ Ranked {len(candidates):,} candidates in {ranking_time:.2f}s")
                        
                finally:
                    try:
                        os.unlink(candidates_path)
                        os.unlink(jd_path)
                    except:
                        pass
            
        except Exception as e:
            st.error(f"❌ Error during ranking: {str(e)}")
            st.session_state.status = "error"
            import traceback
            st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.ranked_candidates:
        ranked = st.session_state.ranked_candidates
        
        st.divider()
        st.subheader("📊 Ranking Summary")
        display_metrics(ranked)
        
        st.subheader("📈 Score Distribution")
        scores = [c['score'] for c in ranked]
        score_df = pd.DataFrame({'Rank': list(range(1, len(scores) + 1)), 'Score': scores})
        st.line_chart(score_df.set_index('Rank'))
        
        st.subheader("📥 Export Results")
        
        # ✅ CORRECT COLUMN ORDER: candidate_id, rank, score, reasoning
        export_df = pd.DataFrame([{
            'candidate_id': c['candidate_id'],
            'rank': c['rank'],
            'score': c['score'],
            'reasoning': c['reasoning']
        } for c in ranked])
        
        # ✅ Ensure exact column order
        export_df = export_df[['candidate_id', 'rank', 'score', 'reasoning']]
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download submission.csv",
            data=csv,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # ✅ Show the column format
        st.caption("📋 CSV Format: candidate_id, rank, score, reasoning")
        
        # Show preview of the CSV
        with st.expander("📋 Preview CSV Content", expanded=False):
            st.dataframe(export_df.head(10), use_container_width=True)
        
        # Full results in expandable table
        with st.expander("📋 View All Results", expanded=False):
            full_df = pd.DataFrame([{
                'Rank': c['rank'],
                'Candidate ID': c['candidate_id'],
                'Score': f"{c['score']:.3f}",
                'Reasoning': c['reasoning']
            } for c in ranked])
            st.dataframe(full_df, use_container_width=True, height=400)
        
        st.subheader("👤 Candidate Details")
        
        page_size = 10
        total_pages = (len(ranked) + page_size - 1) // page_size
        
        if total_pages > 1:
            page = st.selectbox("Page", list(range(1, total_pages + 1)))
        else:
            page = 1
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(ranked))
        
        for i in range(start_idx, end_idx):
            display_candidate(ranked[i], ranked[i]['rank'])


if __name__ == '__main__':
    import numpy as np
    main()
