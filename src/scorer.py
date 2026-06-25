"""Multi-component scoring for candidate ranking - WITH Sentence-Transformers."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple
from .feature_extractor import FeatureExtractor
from .honeypot import HoneypotDetector
from .reasoner import ReasoningGenerator


class CandidateScorer:
    """Score candidates using Sentence-Transformers for semantic understanding."""
    
    def __init__(self, jd_parser, model, jd_embedding: np.ndarray):
        """
        Initialize scorer with sentence-transformer model.
        
        Args:
            jd_parser: JDParser instance
            model: SentenceTransformer model
            jd_embedding: Pre-computed JD embedding
        """
        self.jd_parser = jd_parser
        self.model = model
        self.jd_embedding = jd_embedding
        self.honeypot_detector = HoneypotDetector()
        self.reasoner = ReasoningGenerator()
        
        # Weights for scoring components
        self.weights = {
            'semantic': 0.35,      # Higher weight for semantic understanding
            'skill_match': 0.25,
            'experience': 0.20,
            'behavioral': 0.15,
            'product_experience': 0.05,
        }

    def score(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
        components = {}
        details = {}

        # 1. Semantic similarity (using sentence-transformers)
        sem, sem_det = self._score_semantic(candidate)
        components['semantic'] = sem
        details['semantic'] = sem_det

        # 2. Skill match
        skill, skill_det = self._score_skills(candidate)
        components['skill_match'] = skill
        details['skill_match'] = skill_det

        # 3. Experience fit
        exp, exp_det = self._score_experience(candidate)
        components['experience'] = exp
        details['experience'] = exp_det

        # 4. Behavioral signals
        beh, beh_det = self._score_behavioral(candidate)
        components['behavioral'] = beh
        details['behavioral'] = beh_det

        # 5. Product experience
        prod, prod_det = self._score_product_experience(candidate)
        components['product_experience'] = prod
        details['product_experience'] = prod_det

        # 6. Honeypot detection
        honeypot_risk, flags = self.honeypot_detector.detect(candidate)
        components['honeypot_risk'] = honeypot_risk
        details['honeypot'] = {'risk': honeypot_risk, 'flags': flags}

        # Calculate composite score
        composite = self._calculate_composite(components)
        
        # Apply honeypot penalty
        composite *= (1.0 - honeypot_risk * 0.4)
        
        # Generate reasoning
        reasoning = self.reasoner.generate(
            candidate, 
            composite, 
            components, 
            details
        )
        
        return composite, components, reasoning

    def _score_semantic(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute semantic similarity using sentence-transformers.
        This understands MEANING, not just keywords!
        """
        text = FeatureExtractor.extract_candidate_text(candidate)
        if not text:
            return 0.0, {'error': 'No text extracted'}
        
        try:
            candidate_embedding = self.model.encode(text, normalize_embeddings=True)
            similarity = cosine_similarity(
                self.jd_embedding.reshape(1, -1),
                candidate_embedding.reshape(1, -1)
            )[0][0]
            
            return float(similarity), {
                'similarity': float(similarity),
                'text_length': len(text)
            }
        except Exception as e:
            return 0.0, {'error': str(e)}

    def _score_skills(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        jd_skills = set(self.jd_parser.skills)
        if not jd_skills:
            return 0.5, {'jd_skills': []}
        
        candidate_skills = FeatureExtractor.extract_skills(candidate)
        candidate_skill_names = {s['name'] for s in candidate_skills}
        
        # Exact matches
        matched = candidate_skill_names & jd_skills
        exact_score = len(matched) / len(jd_skills)
        
        # Endorsement bonus
        endorsement_bonus = 0.0
        proficiency_bonus = 0.0
        
        for skill in candidate_skills:
            if skill['name'] in jd_skills:
                endorsement_bonus += min(0.1, skill['endorsements'] / 100)
                if skill['proficiency'] == 'expert':
                    proficiency_bonus += 0.05
                elif skill['proficiency'] == 'advanced':
                    proficiency_bonus += 0.03
                elif skill['proficiency'] == 'intermediate':
                    proficiency_bonus += 0.01
        
        score = min(1.0, exact_score + endorsement_bonus + proficiency_bonus)
        
        return score, {
            'matched_skills': list(matched),
            'total_jd_skills': len(jd_skills),
            'candidate_skills': list(candidate_skill_names)[:10],
            'exact_score': exact_score,
            'endorsement_bonus': endorsement_bonus,
            'proficiency_bonus': proficiency_bonus
        }

    def _score_experience(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        required = self.jd_parser.required_years
        candidate_years = FeatureExtractor.extract_experience_years(candidate)
        
        if candidate_years == 0:
            return 0.0, {'candidate_years': 0, 'required_years': required}
        
        if candidate_years >= required:
            score = min(1.0, 1.0 + (candidate_years - required) * 0.05)
        else:
            score = candidate_years / required
        
        return score, {
            'candidate_years': candidate_years,
            'required_years': required,
            'raw_score': score
        }

    def _score_behavioral(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        signals = FeatureExtractor.extract_redrob_signals(candidate)
        score = 0.5
        details = {}
        
        # Recruiter response rate
        response_rate = signals.get('recruiter_response_rate', 0)
        if response_rate > 0:
            score += response_rate * 0.2
            details['response_rate'] = response_rate
        
        # Recency of activity
        last_active = signals.get('last_active_date', '')
        if last_active:
            try:
                from datetime import datetime
                last = datetime.strptime(last_active, '%Y-%m-%d')
                days_ago = (datetime.now() - last).days
                if days_ago < 30:
                    score += 0.2
                    details['activity_recency'] = 'very_recent'
                elif days_ago < 90:
                    score += 0.1
                    details['activity_recency'] = 'recent'
                elif days_ago > 180:
                    score -= 0.1
                    details['activity_recency'] = 'inactive'
            except:
                details['activity_recency'] = 'unknown'
        
        # Profile completeness
        completeness = signals.get('profile_completeness_score', 0)
        if completeness > 80:
            score += 0.1
            details['profile_completeness'] = 'high'
        elif completeness < 40:
            score -= 0.1
            details['profile_completeness'] = 'low'
        else:
            details['profile_completeness'] = 'medium'
        
        # Open to work
        if signals.get('open_to_work_flag', False):
            score += 0.1
            details['open_to_work'] = True
        
        # Notice period
        notice = signals.get('notice_period_days', 90)
        if notice <= 30:
            score += 0.1
            details['notice_period'] = 'short'
        elif notice > 90:
            score -= 0.1
            details['notice_period'] = 'long'
        else:
            details['notice_period'] = 'medium'
        
        # Search appearance
        search_count = signals.get('search_appearance_30d', 0)
        if search_count > 100:
            score += 0.05
            details['search_visibility'] = 'high'
        
        # Saved by recruiters
        saved_count = signals.get('saved_by_recruiters_30d', 0)
        if saved_count > 10:
            score += 0.05
            details['recruiter_interest'] = 'high'
        
        return max(0.0, min(1.0, score)), details

    def _score_product_experience(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        company_type = FeatureExtractor.extract_company_type(candidate)
        details = {'company_type': company_type}
        
        if company_type == 'product':
            product_years = 0
            if 'career_history' in candidate:
                for exp in candidate['career_history']:
                    if isinstance(exp, dict):
                        combined = f"{exp.get('company', '')} {exp.get('title', '')} {exp.get('description', '')}".lower()
                        product_indicators = ['product', 'platform', 'saas', 'app', 'fintech', 'e-commerce']
                        if any(ind in combined for ind in product_indicators):
                            product_years += exp.get('duration_months', 0) / 12
            
            details['product_years'] = product_years
            
            if product_years >= 3:
                return 1.0, details
            elif product_years >= 1:
                return 0.7, details
            else:
                return 0.4, details
        elif company_type == 'mixed':
            return 0.5, details
        else:
            return 0.2, details

    def _calculate_composite(self, components: Dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        
        for key, weight in self.weights.items():
            if key in components:
                total += components[key] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            return max(0.0, min(1.0, total / weight_sum))
        return 0.0
