import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract structured features from candidate profiles."""
    
    @staticmethod
    def extract_candidate_text(candidate: Dict[str, Any]) -> str:
        """
        Extract text from candidate for embedding.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Concatenated text string
        """
        parts = []
        
        profile = candidate.get('profile', {})
        
        # Headline and summary
        if 'headline' in profile:
            parts.append(profile['headline'])
        if 'summary' in profile:
            parts.append(profile['summary'])
        
        # Current role info
        if 'current_title' in profile:
            parts.append(f"Current: {profile['current_title']}")
        if 'current_company' in profile:
            parts.append(f"at {profile['current_company']}")
        
        # Skills
        skills_text = FeatureExtractor._extract_skills_text(candidate)
        if skills_text:
            parts.append(skills_text)
        
        # Career history
        career_text = FeatureExtractor._extract_career_text(candidate)
        if career_text:
            parts.append(career_text)
        
        # Education
        edu_text = FeatureExtractor._extract_education_text(candidate)
        if edu_text:
            parts.append(edu_text)
        
        # Redrob signals
        signals_text = FeatureExtractor._extract_signals_text(candidate)
        if signals_text:
            parts.append(signals_text)
        
        return ' '.join(parts)
    
    @staticmethod
    def _extract_skills_text(candidate: Dict[str, Any]) -> str:
        """Extract skills as text."""
        skills = []
        
        if 'skills' in candidate:
            for skill in candidate['skills']:
                if isinstance(skill, dict):
                    name = skill.get('name', '')
                    prof = skill.get('proficiency', '')
                    if name:
                        skills.append(f"{name} ({prof})" if prof else name)
        
        return 'Skills: ' + ', '.join(skills) if skills else ''
    
    @staticmethod
    def _extract_career_text(candidate: Dict[str, Any]) -> str:
        """Extract career history as text."""
        parts = []
        
        if 'career_history' in candidate:
            for exp in candidate['career_history'][:3]:
                if isinstance(exp, dict):
                    title = exp.get('title', '')
                    company = exp.get('company', '')
                    desc = exp.get('description', '')
                    if title or company:
                        parts.append(f"{title} at {company}: {desc[:150]}")
        
        return ' '.join(parts) if parts else ''
    
    @staticmethod
    def _extract_education_text(candidate: Dict[str, Any]) -> str:
        """Extract education as text."""
        parts = []
        
        if 'education' in candidate:
            for edu in candidate['education'][:2]:
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    field = edu.get('field_of_study', '')
                    institution = edu.get('institution', '')
                    if degree or field:
                        parts.append(f"{degree} in {field} from {institution}".strip())
        
        return ' '.join(parts) if parts else ''
    
    @staticmethod
    def _extract_signals_text(candidate: Dict[str, Any]) -> str:
        """Extract redrob signals as text."""
        signals = candidate.get('redrob_signals', {})
        parts = []
        
        if signals.get('open_to_work_flag'):
            parts.append("Open to work")
        
        if signals.get('willing_to_relocate'):
            parts.append("Willing to relocate")
        
        if signals.get('preferred_work_mode'):
            parts.append(f"Preferred work mode: {signals['preferred_work_mode']}")
        
        if signals.get('profile_completeness_score', 0) > 80:
            parts.append("Complete profile")
        
        return ' '.join(parts) if parts else ''
    
    @staticmethod
    def extract_skills(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract skills with full details.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            List of skill dictionaries
        """
        skills = []
        
        if 'skills' in candidate:
            for skill in candidate['skills']:
                if isinstance(skill, dict):
                    skills.append({
                        'name': skill.get('name', '').strip().lower(),
                        'proficiency': skill.get('proficiency', 'beginner'),
                        'endorsements': skill.get('endorsements', 0),
                        'duration_months': skill.get('duration_months', 0)
                    })
        
        return skills
    
    @staticmethod
    def extract_skill_names(candidate: Dict[str, Any]) -> Set[str]:
        """Extract skill names as a set."""
        return {s['name'] for s in FeatureExtractor.extract_skills(candidate)}
    
    @staticmethod
    def extract_experience_years(candidate: Dict[str, Any]) -> float:
        """
        Extract total years of experience.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Total years of experience
        """
        # Try profile years first
        profile = candidate.get('profile', {})
        if 'years_of_experience' in profile:
            try:
                return float(profile['years_of_experience'])
            except (ValueError, TypeError):
                pass
        
        # Calculate from career history
        total_years = 0.0
        
        if 'career_history' in candidate:
            for exp in candidate['career_history']:
                if not isinstance(exp, dict):
                    continue
                
                if 'duration_months' in exp:
                    total_years += exp['duration_months'] / 12.0
                elif 'start_date' in exp and 'end_date' in exp:
                    try:
                        start = datetime.strptime(exp['start_date'], '%Y-%m-%d')
                        end = datetime.strptime(exp['end_date'], '%Y-%m-%d')
                        total_years += (end - start).days / 365.25
                    except (ValueError, TypeError):
                        pass
                elif 'start_date' in exp and exp.get('is_current', False):
                    try:
                        start = datetime.strptime(exp['start_date'], '%Y-%m-%d')
                        end = datetime.now()
                        total_years += (end - start).days / 365.25
                    except (ValueError, TypeError):
                        pass
        
        return round(total_years, 1)
    
    @staticmethod
    def extract_experience_by_industry(candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract experience years by industry."""
        industry_years = {}
        
        if 'career_history' in candidate:
            for exp in candidate['career_history']:
                if not isinstance(exp, dict):
                    continue
                
                industry = exp.get('industry', 'unknown')
                years = exp.get('duration_months', 0) / 12.0
                
                if industry in industry_years:
                    industry_years[industry] += years
                else:
                    industry_years[industry] = years
        
        return industry_years
    
    @staticmethod
    def extract_current_title(candidate: Dict[str, Any]) -> Optional[str]:
        """Extract current job title."""
        profile = candidate.get('profile', {})
        return profile.get('current_title')
    
    @staticmethod
    def extract_current_company(candidate: Dict[str, Any]) -> Optional[str]:
        """Extract current company."""
        profile = candidate.get('profile', {})
        return profile.get('current_company')
    
    @staticmethod
    def extract_location(candidate: Dict[str, Any]) -> Optional[str]:
        """Extract candidate location."""
        profile = candidate.get('profile', {})
        return profile.get('location')
    
    @staticmethod
    def extract_country(candidate: Dict[str, Any]) -> Optional[str]:
        """Extract candidate country."""
        profile = candidate.get('profile', {})
        return profile.get('country')
    
    @staticmethod
    def extract_company_type(candidate: Dict[str, Any]) -> str:
        """Determine if candidate has product company experience."""
        product_indicators = ['product', 'platform', 'saas', 'app', 'fintech', 'e-commerce']
        
        if 'career_history' in candidate:
            product_years = 0
            total_years = 0
            
            for exp in candidate['career_history']:
                if not isinstance(exp, dict):
                    continue
                
                company = exp.get('company', '').lower()
                title = exp.get('title', '').lower()
                desc = exp.get('description', '').lower()
                years = exp.get('duration_months', 0) / 12.0
                
                combined = f"{company} {title} {desc}"
                if any(ind in combined for ind in product_indicators):
                    product_years += years
                total_years += years
            
            if total_years > 0 and product_years / total_years > 0.5:
                return 'product'
            elif product_years > 0:
                return 'mixed'
        
        return 'services'
    
    @staticmethod
    def extract_redrob_signals(candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Redrob behavioral signals."""
        return candidate.get('redrob_signals', {})
    
    @staticmethod
    def get_education_tier(candidate: Dict[str, Any]) -> int:
        """Get education tier (1-4, lower is better)."""
        if 'education' in candidate and candidate['education']:
            for edu in candidate['education']:
                if isinstance(edu, dict):
                    tier = edu.get('tier', 'tier_4')
                    if tier == 'tier_1':
                        return 1
                    elif tier == 'tier_2':
                        return 2
                    elif tier == 'tier_3':
                        return 3
                    elif tier == 'tier_4':
                        return 4
        return 4
    
    @staticmethod
    def extract_highest_degree(candidate: Dict[str, Any]) -> Optional[str]:
        """Extract highest degree."""
        if 'education' in candidate and candidate['education']:
            # Sort by degree level
            degree_priority = {
                'phd': 5,
                'doctorate': 5,
                'master': 4,
                'mba': 4,
                'bachelor': 3,
                'bs': 3,
                'ba': 3,
                'associate': 2,
                'certificate': 1,
            }
            
            highest = None
            highest_priority = -1
            
            for edu in candidate['education']:
                if isinstance(edu, dict):
                    degree = edu.get('degree', '').lower()
                    for key, priority in degree_priority.items():
                        if key in degree and priority > highest_priority:
                            highest_priority = priority
                            highest = degree
            
            return highest
        
        return None
    
    @staticmethod
    def extract_keywords(candidate: Dict[str, Any]) -> List[str]:
        """Extract keywords from candidate profile."""
        text = FeatureExtractor.extract_candidate_text(candidate)
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return list(set(words))
    
    @staticmethod
    def calculate_experience_match(candidate: Dict[str, Any], required_years: int) -> float:
        """Calculate experience match score."""
        candidate_years = FeatureExtractor.extract_experience_years(candidate)
        
        if candidate_years == 0:
            return 0.0
        
        if candidate_years >= required_years:
            return min(1.0, 1.0 + (candidate_years - required_years) * 0.05)
        else:
            return candidate_years / required_years
    
    @staticmethod
    def calculate_skill_match(candidate: Dict[str, Any], required_skills: List[str]) -> float:
        """Calculate skill match score."""
        if not required_skills:
            return 0.5
        
        candidate_skills = FeatureExtractor.extract_skill_names(candidate)
        required_set = set(required_skills)
        
        matched = len(candidate_skills & required_set)
        return matched / len(required_set)
