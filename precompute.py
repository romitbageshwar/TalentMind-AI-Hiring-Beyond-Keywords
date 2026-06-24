#!/usr/bin/env python3
"""
Pre-compute embeddings for faster ranking.
This is optional - the main ranker can compute on the fly.
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import orjson

from src.data_loader import CandidateLoader
from src.feature_extractor import FeatureExtractor
from src.utils import setup_logging, Timer

setup_logging('INFO')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Pre-compute candidate embeddings')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl.gz')
    parser.add_argument('--output-dir', default='models/', help='Output directory')
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2', help='Model name')
    parser.add_argument('--limit', type=int, default=None, help='Limit for testing')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading candidates...")
    loader = CandidateLoader(args.candidates)
    candidates = loader.load_all(limit=args.limit)
    logger.info(f"Loaded {len(candidates)} candidates")
    
    logger.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    logger.info("Generating embeddings...")
    embeddings = []
    candidate_ids = []
    
    with Timer("Embedding generation"):
        for candidate in tqdm(candidates, desc="Processing"):
            candidate_id = candidate.get('candidate_id', '')
            text = FeatureExtractor.extract_candidate_text(candidate)
            
            if text:
                embedding = model.encode(text, normalize_embeddings=True)
                embeddings.append(embedding)
                candidate_ids.append(candidate_id)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Save embeddings and IDs
    embedding_path = os.path.join(args.output_dir, 'embeddings.npy')
    ids_path = os.path.join(args.output_dir, 'candidate_ids.pkl')
    
    np.save(embedding_path, embeddings)
    with open(ids_path, 'wb') as f:
        pickle.dump(candidate_ids, f)
    
    logger.info(f"Saved embeddings ({embeddings.shape}) to {embedding_path}")
    logger.info(f"Saved candidate IDs ({len(candidate_ids)}) to {ids_path}")
    
    # Save metadata
    metadata = {
        'model_name': args.model_name,
        'num_candidates': len(candidate_ids),
        'embedding_dim': embeddings.shape[1],
        'embedding_path': embedding_path,
        'ids_path': ids_path
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Pre-computation complete!")


if __name__ == '__main__':
    main()
