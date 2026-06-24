import logging
from typing import Dict, Any, List, Optional
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class ReasoningGenerator:
    """Generate concise, specific reasoning for candidate rankings."""
    
    def generate(self, candidate: Dict[str, Any], score: float, 
                components: Dict[str, float], details: Dict[str, Any]) -> str:
        """
        Generate reasoning for a candidate.
        
        Args:
            candidate: Candidate dictionary
            score: Composite score
            components: Component scores
            details: Additional details from scoring
            
        Returns:
            Reasoning string
        """
        reasons = []
        
        # Start with candidate context
        context = self._get_candidate_context(candidate)
        if context:
            reasons.append(context)
        
        # Add score-based reasoning
        score_reasons = self._get_score_reasoning(score, components, details)
        reasons.extend(score_reasons)
        
        # Add specific details
        specific_reasons = self._get_specific_reasons(candidate, components, details)
        reasons.extend(specific_reasons)
        
        # Add concerns if any
        concerns = self._get_concerns(candidate, components, details)
        if concerns:
            reasons.append(concerns)
        
        # Combine and limit length
        reasoning = "; ".join(reasons)
        if len(reasoning) > 250:
            reasoning = reasoning[:247] + "..."
        
        return reasoning
    
    def _get_candidate_context(self, candidate: Dict[str, Any]) -> Optional[str]:
        """Get candidate context string."""
        parts = []
        
        profile = candidate.get('profile', {})
        
        # Title and experience
        title = profile.get('current_title', '')
        years = profile.get('years_of_experience', 0)
        if title and years:
            parts.append(f"{title} with {years:.1f} years experience")
        elif title:
            parts.append(title)
        
        # Company
        company = profile.get('current_company', '')
        if company:
            parts.append(f"at {company}")
        
        # Location
        location = profile.get('location', '')
        if location:
            parts.append(f"based in {location}")
        
        return ' '.join(parts) if parts else None
    
    def _get_score_reasoning(self, score: float, components: Dict[str, float], 
                            details: Dict[str, Any]) -> List[str]:
        """Get reasoning based on component scores."""
        reasons = []
        
        # Semantic match
        semantic = components.get('semantic', 0)
        if semantic > 0.7:
            reasons.append("strong semantic match with JD")
        elif semantic > 0.4:
            reasons.append("good semantic alignment")
        elif semantic < 0.2:
            reasons.append("limited semantic match")
        
        # Skill match
        skill = components.get('skill_match', 0)
        if skill > 0.6:
            reasons.append(f"strong skill match")
        elif skill > 0.3:
            reasons.append(f"decent skill match ({int(skill*100)}%)")
        else:
            reasons.append("limited skill overlap")
        
        # Experience
        exp = components.get('experience', 0)
        if exp > 0.7:
            reasons.append("experience well-aligned")
        elif exp > 0.4:
            reasons.append("relevant experience")
        else:
            reasons.append("experience below requirements")
        
        # Behavioral
        behavioral = components.get('behavioral', 0)
        if behavioral > 0.7:
            reasons.append("strong engagement signals")
        elif behavioral > 0.4:
            reasons.append("moderate engagement")
        else:
            reasons.append("low engagement")
        
        return reasons
    
    def _get_specific_reasons(self, candidate: Dict[str, Any], 
                             components: Dict[str, float], 
                             details: Dict[str, Any]) -> List[str]:
        """Get specific reasoning based on candidate details."""
        reasons = []
        
        # Check for specific skills
        skill_details = details.get('skill_match', {})
        matched_skills = skill_details.get('matched_skills', [])[:3]
        if matched_skills:
            reasons.append(f"has matching skills: {', '.join(matched_skills)}")
        
        # Check for product experience
        product_details = details.get('product_experience', {})
        product_type = product_details.get('company_type', '')
        if product_type == 'product':
            reasons.append("has product company experience")
        elif product_type == 'mixed':
            reasons.append("has some product-oriented experience")
        
        # Check for specific experience
        exp_details = details.get('experience', {})
        candidate_years = exp_details.get('candidate_years', 0)
        required_years = exp_details.get('required_years', 0)
        if candidate_years and required_years:
            if candidate_years >= required_years:
                reasons.append(f"{candidate_years:.1f} years experience meets {required_years} year requirement")
            elif candidate_years > 0:
                reasons.append(f"{candidate_years:.1f} years of relevant experience")
        
        # Check for behavioral signals
        behavioral_details = details.get('behavioral', {})
        if behavioral_details.get('open_to_work'):
            reasons.append("open to work")
        if behavioral_details.get('activity_recency') == 'very_recent':
            reasons.append("recently active")
        if behavioral_details.get('notice_period') == 'short':
            reasons.append("short notice period")
        if behavioral_details.get('willing_to_relocate'):
            reasons.append("willing to relocate")
        
        return reasons
    
    def _get_concerns(self, candidate: Dict[str, Any], 
                     components: Dict[str, float], 
                     details: Dict[str, Any]) -> Optional[str]:
        """Get concerns about the candidate."""
        concerns = []
        
        # Check honeypot
        honeypot_details = details.get('honeypot', {})
        risk = honeypot_details.get('risk', 0)
        if risk > 0.3:
            concerns.append("⚠️ some profile inconsistencies")
        
        # Check experience gap
        exp = components.get('experience', 0)
        if exp < 0.3:
            concerns.append("experience gap")
        
        # Check behavioral concerns
        behavioral = components.get('behavioral', 0)
        if behavioral < 0.2:
            concerns.append("low engagement")
        
        # Check location
        profile = candidate.get('profile', {})
        location = profile.get('location', '')
        if location and 'Pune' not in location and 'Noida' not in location:
            concerns.append(f"based in {location}, not Pune/Noida")
        
        # Check notice period
        signals = candidate.get('redrob_signals', {})
        notice = signals.get('notice_period_days', 0)
        if notice > 90:
            concerns.append(f"long notice period ({notice} days)")
        elif notice > 60:
            concerns.append(f"notice period {notice} days")
        
        return "Concerns: " + ", ".join(concerns) if concerns else None
