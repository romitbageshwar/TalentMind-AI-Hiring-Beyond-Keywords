#!/usr/bin/env python3
"""
Redrob Hackathon - AI Candidate Ranking System
Rank 100K candidates against a JD within 5 minutes on CPU.
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.data_loader import CandidateLoader
from src.jd_parser import JDParser
from src.feature_extractor import FeatureExtractor
from src.scorer import CandidateScorer
from src.utils import setup_logging, Timer

# Setup logging
setup_logging('INFO')
logger = logging.getLogger(__name__)


class CandidateRanker:
    """Main ranking system."""
    
    def __init__(self, jd_path: str):
        """
        Initialize ranker with job description.
        
        Args:
            jd_path: Path to job description file
        """
        self.jd_path = jd_path
        self.jd_text = self._load_jd()
        self.jd_parser = JDParser(self.jd_text)
        
        logger.info("Loading embedding model...")
        with Timer("Model loading"):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Generating JD embedding...")
        with Timer("JD embedding"):
            self.jd_embedding = self.model.encode(
                self.jd_text, 
                normalize_embeddings=True
            )
        
        self.scorer = CandidateScorer(
            self.jd_parser, 
            self.model, 
            self.jd_embedding
        )
        
        logger.info("Ranker initialized successfully")
        logger.info(f"JD parsed: {len(self.jd_parser.skills)} skills, "
                   f"{self.jd_parser.required_years} years required")
    
    def _load_jd(self) -> str:
        """Load job description from file."""
        with open(self.jd_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def rank(self, candidates: List[Dict[str, Any]], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Rank candidates and return top K.
        
        Args:
            candidates: List of candidate dictionaries
            top_k: Number of candidates to return
            
        Returns:
            List of ranked candidates with scores and reasoning
        """
        logger.info(f"Scoring {len(candidates)} candidates...")
        
        scored = []
        start_time = time.time()
        
        # Score candidates with progress bar
        with tqdm(total=len(candidates), desc="Scoring") as pbar:
            for candidate in candidates:
                candidate_id = candidate.get('candidate_id', '')
                
                # Skip candidates without profile
                if 'profile' not in candidate:
                    pbar.update(1)
                    continue
                
                try:
                    score, components, reasoning = self.scorer.score(candidate)
                    
                    scored.append({
                        'candidate_id': candidate_id,
                        'score': score,
                        'reasoning': reasoning[:250],
                        'components': components,
                    })
                except Exception as e:
                    logger.warning(f"Error scoring {candidate_id}: {e}")
                
                pbar.update(1)
        
        elapsed = time.time() - start_time
        logger.info(f"Scored {len(scored)} candidates in {elapsed:.2f}s")
        
        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top K
        top = scored[:top_k]
        
        # Assign ranks
        for i, cand in enumerate(top, 1):
            cand['rank'] = i
        
        return top
    
    def export_csv(self, ranked: List[Dict[str, Any]], output_path: str):
        """
        Export ranked candidates to CSV.
        
        Args:
            ranked: List of ranked candidates
            output_path: Output file path
        """
        data = []
        for cand in ranked:
            data.append({
                'candidate_id': cand['candidate_id'],
                'rank': cand['rank'],
                'score': round(cand['score'], 4),
                'reasoning': cand['reasoning']
            })
        
        df = pd.DataFrame(data)
        df = df[['candidate_id', 'rank', 'score', 'reasoning']]
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported {len(data)} candidates to {output_path}")
        
        # Print top 5
        logger.info("\n--- Top 5 Candidates ---")
        for i, cand in enumerate(ranked[:5], 1):
            logger.info(f"#{i}: {cand['candidate_id']} - Score: {cand['score']:.4f}")
            logger.info(f"   {cand['reasoning'][:150]}...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Rank candidates against job description for Redrob Hackathon'
    )
    parser.add_argument(
        '--candidates', 
        required=True, 
        help='Path to candidates.jsonl.gz file'
    )
    parser.add_argument(
        '--jd', 
        required=True, 
        help='Path to job description file'
    )
    parser.add_argument(
        '--output', 
        default='submission.csv', 
        help='Output CSV path (default: submission.csv)'
    )
    parser.add_argument(
        '--sample', 
        type=int, 
        default=None, 
        help='Sample size for testing'
    )
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=100, 
        help='Number of candidates to output (default: 100)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Redrob Candidate Ranking System")
    logger.info("=" * 60)
    
    # Load candidates
    loader = CandidateLoader(args.candidates)
    with Timer("Loading candidates"):
        candidates = loader.load_all(limit=args.sample)
    
    if not candidates:
        logger.error("No candidates loaded")
        sys.exit(1)
    
    logger.info(f"Loaded {len(candidates)} candidates")
    
    # Rank
    ranker = CandidateRanker(args.jd)
    with Timer("Ranking candidates"):
        ranked = ranker.rank(candidates, top_k=args.top_k)
    
    if len(ranked) < args.top_k:
        logger.warning(f"Only {len(ranked)} candidates scored, expected {args.top_k}")
    
    # Export
    ranker.export_csv(ranked, args.output)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Ranking complete! Output: {args.output}")
    if ranked:
        logger.info(f"   Top score: {ranked[0]['score']:.4f}")
        logger.info(f"   Bottom score: {ranked[-1]['score']:.4f}")
        logger.info(f"   Score range: {ranked[-1]['score']:.4f} - {ranked[0]['score']:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
