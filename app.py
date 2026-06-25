#!/usr/bin/env python3
"""
Streamlit Web Interface for Redrob AI Candidate Ranking System
Deploy on Streamlit Cloud for interactive candidate ranking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
import tempfile
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import base64
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ranking modules
from src.data_loader import CandidateLoader
from src.jd_parser import JDParser
from src.feature_extractor import FeatureExtractor
from src.scorer import CandidateScorer
from src.honeypot import HoneypotDetector
from src.reasoner import ReasoningGenerator
from src.utils import setup_logging, Timer

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
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }
    .score-high {
        color: #2ca02c;
        font-weight: bold;
    }
    .score-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .score-low {
        color: #d62728;
        font-weight: bold;
    }
    .candidate-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        transition: all 0.3s;
    }
    .candidate-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .honeypot-warning {
        background-color: #ffcccc;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ff0000;
    }
    .reasoning-text {
        color: #555;
        font-size: 0.9rem;
        padding: 0.5rem;
        background-color: #f1f3f5;
        border-radius: 0.25rem;
        border-left: 3px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
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
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


@st.cache_resource
def load_embedding_model():
    """Cache the embedding model."""
    from sentence_transformers import SentenceTransformer
    with st.spinner("Loading embedding model (this may take a few seconds)..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.model_loaded = True
        return model


@st.cache_data
def load_sample_candidates() -> List[Dict[str, Any]]:
    """Load sample candidates for demonstration."""
    sample_path = Path("data/sample_candidates.json")
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            return json.load(f)
    # Fallback: generate sample data
    return generate_sample_candidates()


def generate_sample_candidates() -> List[Dict[str, Any]]:
    """Generate sample candidates for testing."""
    candidates = []
    
    sample_profiles = [
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Senior AI Engineer",
                "headline": "AI/ML Engineer with 7 years experience",
                "summary": "Senior AI Engineer with 7 years building RAG systems and production ML systems. Strong background in NLP, retrieval, and vector search.",
                "location": "Bangalore, India",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": "Senior AI Engineer",
                "current_company": "TechCorp AI",
                "current_company_size": "1001-5000",
                "current_industry": "AI/ML"
            },
            "career_history": [
                {
                    "company": "TechCorp AI",
                    "title": "Senior AI Engineer",
                    "start_date": "2021-01-01",
                    "end_date": None,
                    "duration_months": 60,
                    "is_current": True,
                    "industry": "AI/ML",
                    "company_size": "1001-5000",
                    "description": "Built RAG systems and embedding-based retrieval for enterprise search. Designed vector search pipelines using Pinecone and FAISS."
                },
                {
                    "company": "DataLabs",
                    "title": "ML Engineer",
                    "start_date": "2018-01-01",
                    "end_date": "2020-12-31",
                    "duration_months": 36,
                    "is_current": False,
                    "industry": "AI/ML",
                    "company_size": "201-500",
                    "description": "Developed recommendation systems and ranking models for e-commerce platform."
                }
            ],
            "education": [
                {
                    "institution": "IIT Bombay",
                    "degree": "M.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2016,
                    "end_year": 2018,
                    "grade": "8.5 CGPA",
                    "tier": "tier_1"
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 48},
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
                {"name": "Pinecone", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
                {"name": "NLP", "proficiency": "expert", "endorsements": 40, "duration_months": 60},
                {"name": "RAG", "proficiency": "advanced", "endorsements": 35, "duration_months": 24}
            ],
            "redrob_signals": {
                "profile_completeness_score": 92.0,
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.85,
                "notice_period_days": 30,
                "last_active_date": "2026-06-20",
                "willing_to_relocate": True
            }
        },
        {
            "candidate_id": "CAND_0000002",
            "profile": {
                "anonymized_name": "Full Stack Developer",
                "headline": "Full Stack Developer with ML experience",
                "summary": "Full Stack Developer with 5 years experience and some ML exposure. Built web applications and APIs.",
                "location": "Hyderabad, India",
                "country": "India",
                "years_of_experience": 5.0,
                "current_title": "Full Stack Developer",
                "current_company": "WebTech Inc",
                "current_company_size": "201-500",
                "current_industry": "Software"
            },
            "career_history": [
                {
                    "company": "WebTech Inc",
                    "title": "Full Stack Developer",
                    "start_date": "2021-06-01",
                    "end_date": None,
                    "duration_months": 60,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "Built React frontend and Node.js backend for SaaS product."
                }
            ],
            "education": [
                {
                    "institution": "Local Engineering College",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2016,
                    "end_year": 2020,
                    "grade": "7.8 CGPA",
                    "tier": "tier_3"
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "intermediate", "endorsements": 10, "duration_months": 36},
                {"name": "React", "proficiency": "advanced", "endorsements": 15, "duration_months": 48},
                {"name": "Node.js", "proficiency": "advanced", "endorsements": 12, "duration_months": 42}
            ],
            "redrob_signals": {
                "profile_completeness_score": 78.0,
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.6,
                "notice_period_days": 60,
                "last_active_date": "2026-05-15",
                "willing_to_relocate": False
            }
        }
    ]
    
    # Generate 100 candidates with variations
    for i in range(100):
        base = sample_profiles[i % len(sample_profiles)]
        candidate = {
            "candidate_id": f"CAND_{i+1:07d}",
            "profile": {
                "anonymized_name": f"{base['profile']['anonymized_name']}_{i}",
                "headline": base['profile']['headline'],
                "summary": f"{base['profile']['summary']} - Variation {i}",
                "location": base['profile']['location'],
                "country": base['profile']['country'],
                "years_of_experience": base['profile']['years_of_experience'] + (i % 3),
                "current_title": base['profile']['current_title'],
                "current_company": base['profile']['current_company'],
                "current_company_size": base['profile']['current_company_size'],
                "current_industry": base['profile']['current_industry']
            },
            "career_history": base['career_history'],
            "education": base['education'],
            "skills": [
                {**s, "duration_months": s["duration_months"] + (i % 12)}
                for s in base['skills']
            ],
            "redrob_signals": {
                **base['redrob_signals'],
                "last_active_date": "2026-06-15"
            }
        }
        candidates.append(candidate)
    
    return candidates


def display_candidate(candidate_data: Dict[str, Any], rank: int):
    """Display a single candidate in a formatted card."""
    candidate_id = candidate_data['candidate_id']
    score = candidate_data['score']
    reasoning = candidate_data.get('reasoning', 'No reasoning provided')
    components = candidate_data.get('components', {})
    
    # Determine score color
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
                <div>
                    <h4>#{rank} - {candidate_id}</h4>
                </div>
                <div>
                    <span class="{score_class}">{score_emoji} Score: {score:.3f}</span>
                </div>
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
    
    # Header
    st.markdown('<p class="main-header">🎯 AI-Powered Candidate Ranking System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Rank candidates against job descriptions using AI understanding</p>',
                unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data input section
        st.subheader("📁 Data Input")
        
        # Option to use sample data
        use_sample = st.checkbox(
            "Use sample data (for testing)",
            value=True if not st.session_state.ranked_candidates else False
        )
        
        if use_sample:
            st.info("📊 Using built-in sample data for demonstration.")
            st.info("Upload your own files below for real ranking.")
        else:
            # File uploads
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
        
        # Ranking parameters
        st.subheader("🎯 Ranking Parameters")
        
        top_k = st.slider(
            "Number of candidates to rank",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Number of candidates to include in output"
        )
        
        sample_size = st.number_input(
            "Sample size (0 = all candidates)",
            min_value=0,
            max_value=100000,
            value=0,
            step=1000,
            help="For testing, limit number of candidates processed"
        )
        
        # Run ranking button
        run_button = st.button(
            "🚀 Rank Candidates",
            type="primary",
            use_container_width=True
        )
        
        # Status display
        if st.session_state.status != "idle":
            st.info(f"Status: {st.session_state.status}")
        
        if st.session_state.ranking_time:
            st.success(f"⏱️ Ranking completed in {st.session_state.ranking_time:.2f} seconds")
        
        if st.session_state.candidate_count > 0:
            st.info(f"📊 Processed {st.session_state.candidate_count} candidates")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Job Description Display
        st.subheader("📄 Job Description")
        
        if use_sample:
            # Use sample JD
            sample_jd = """
            # AI Engineer - Redrob
            
            We are looking for an experienced AI Engineer to join our team.
            
            ## Requirements:
            - 5+ years of experience in ML/AI
            - Strong Python programming skills
            - Experience with RAG systems and vector databases
            - Product mindset with focus on shipping
            - Experience with embeddings and retrieval systems
            
            ## Nice to have:
            - Experience with LLM fine-tuning
            - Knowledge of ranking and recommendation systems
            - Open source contributions
            
            ## Location:
            - Bangalore/Pune/Noida preferred
            - Hybrid work mode
            """
            st.text_area(
                "Job Description (sample)",
                value=sample_jd,
                height=200,
                disabled=True,
                key="sample_jd_display"
            )
        else:
            if jd_file:
                jd_text = jd_file.read().decode('utf-8')
                st.text_area(
                    "Job Description",
                    value=jd_text,
                    height=200,
                    disabled=True,
                    key="uploaded_jd_display"
                )
            else:
                st.warning("⚠️ Please upload a job description file.")
    
    # Ranking execution
    if run_button:
        try:
            # Reset previous results
            st.session_state.ranked_candidates = None
            st.session_state.status = "loading"
            
            # Handle sample data or uploaded files
            if use_sample:
                # Use sample data
                with st.spinner("🔄 Loading sample data..."):
                    candidates = load_sample_candidates()
                    jd_text = "AI Engineer with experience in RAG, vector databases, and Python"
                
                with st.spinner("🔄 Processing candidates..."):
                    start_time = time.time()
                    
                    # Load embedding model
                    model = load_embedding_model()
                    
                    # Parse JD
                    jd_parser = JDParser(jd_text)
                    jd_embedding = model.encode(jd_text, normalize_embeddings=True)
                    
                    # Initialize scorer
                    scorer = CandidateScorer(jd_parser, model, jd_embedding)
                    
                    # Score candidates
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
                    
                    # Sort and get top K
                    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
                    top_candidates = scored_candidates[:top_k]
                    
                    # Assign ranks
                    for i, cand in enumerate(top_candidates, 1):
                        cand['rank'] = i
                    
                    ranking_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.ranked_candidates = top_candidates
                    st.session_state.ranking_time = ranking_time
                    st.session_state.status = "completed"
                    st.session_state.candidate_count = len(candidates)
                    
                    st.success(f"✅ Ranking complete! Processed {len(candidates)} candidates in {ranking_time:.2f} seconds")
                    
            else:
                # Handle uploaded files
                if candidates_file and jd_file:
                    # Save uploaded files temporarily
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl.gz', delete=False) as f:
                        f.write(candidates_file.getvalue())
                        candidates_path = f.name
                    
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
                        f.write(jd_file.getvalue())
                        jd_path = f.name
                    
                    try:
                        with st.spinner("🔄 Loading candidates..."):
                            loader = CandidateLoader(candidates_path)
                            candidates = loader.load_all(limit=sample_size if sample_size > 0 else None)
                        
                        with st.spinner("🔄 Processing candidates..."):
                            start_time = time.time()
                            
                            # Load JD
                            with open(jd_path, 'r', encoding='utf-8') as f:
                                jd_text = f.read()
                            
                            # Load model
                            model = load_embedding_model()
                            
                            # Parse JD
                            jd_parser = JDParser(jd_text)
                            jd_embedding = model.encode(jd_text, normalize_embeddings=True)
                            
                            # Initialize scorer
                            scorer = CandidateScorer(jd_parser, model, jd_embedding)
                            
                            # Score candidates
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
                            
                            # Sort and get top K
                            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
                            top_candidates = scored_candidates[:top_k]
                            
                            # Assign ranks
                            for i, cand in enumerate(top_candidates, 1):
                                cand['rank'] = i
                            
                            ranking_time = time.time() - start_time
                            
                            # Store results
                            st.session_state.ranked_candidates = top_candidates
                            st.session_state.ranking_time = ranking_time
                            st.session_state.status = "completed"
                            st.session_state.candidate_count = len(candidates)
                            
                            st.success(f"✅ Ranking complete! Processed {len(candidates)} candidates in {ranking_time:.2f} seconds")
                            
                    finally:
                        # Cleanup temp files
                        try:
                            os.unlink(candidates_path)
                            os.unlink(jd_path)
                        except:
                            pass
                else:
                    st.error("❌ Please upload both candidates file and job description.")
            
        except Exception as e:
            st.error(f"❌ Error during ranking: {str(e)}")
            st.session_state.status = "error"
            import traceback
            st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.ranked_candidates:
        st.divider()
        
        ranked = st.session_state.ranked_candidates
        
        # Summary metrics
        st.subheader("📊 Ranking Summary")
        display_metrics(ranked)
        
        # Score distribution chart
        st.subheader("📈 Score Distribution")
        scores = [c['score'] for c in ranked]
        score_df = pd.DataFrame({
            'Rank': list(range(1, len(scores) + 1)),
            'Score': scores
        })
        st.line_chart(score_df.set_index('Rank'))
        
        # Export options
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Preview top candidates
            st.write("**🏆 Top 5 Candidates**")
            preview_data = []
            for c in ranked[:5]:
                preview_data.append({
                    'Rank': c['rank'],
                    'Candidate ID': c['candidate_id'],
                    'Score': f"{c['score']:.3f}",
                    'Reasoning': c['reasoning'][:100] + '...' if len(c['reasoning']) > 100 else c['reasoning']
                })
            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True)
        
        with col2:
            # Download CSV
            export_df = pd.DataFrame([{
                'candidate_id': c['candidate_id'],
                'rank': c['rank'],
                'score': c['score'],
                'reasoning': c['reasoning']
            } for c in ranked])
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"ranked_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Download as JSON
            json_data = json.dumps([{
                'candidate_id': c['candidate_id'],
                'rank': c['rank'],
                'score': c['score'],
                'reasoning': c['reasoning'],
                'components': c.get('components', {})
            } for c in ranked], indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"ranked_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Full results in expandable table
            with st.expander("📋 View All Results", expanded=False):
                full_df = pd.DataFrame([{
                    'Rank': c['rank'],
                    'Candidate ID': c['candidate_id'],
                    'Score': f"{c['score']:.3f}",
                    'Reasoning': c['reasoning']
                } for c in ranked])
                st.dataframe(full_df, use_container_width=True, height=400)
        
        # Display all candidates in cards
        st.subheader("👤 Candidate Details")
        
        # Pagination
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
    main()
