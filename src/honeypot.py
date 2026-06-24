import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class HoneypotDetector:
    """Detect honeypot candidates with impossible profiles."""
    
    def __init__(self):
        self.suspicious_companies = {
            'nasa': 0.3,
            'spacex': 0.3,
            'tesla': 0.2,
            'openai': 0.3,
            'anthropic': 0.3,
            'google': 0.2,
            'facebook': 0.2,
            'apple': 0.2,
            'microsoft': 0.2,
            'amazon': 0.2,
            'netflix': 0.2,
            'uber': 0.2,
            'lyft': 0.2,
            'stripe': 0.2,
            'coinbase': 0.2,
        }
        
        self.skill_proficiency_weights = {
            'expert': 0.15,
            'advanced': 0.10,
            'intermediate': 0.05,
            'beginner': 0.02,
        }
    
    def detect(self, candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Detect honeypot indicators in candidate profile.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Tuple of (risk_score, red_flags)
        """
        risk_score = 0.0
        red_flags = []
        
        # Check 1: Too many expert/advanced skills with low duration
        skill_flags, skill_risk = self._check_skill_consistency(candidate)
        red_flags.extend(skill_flags)
        risk_score += skill_risk
        
        # Check 2: Suspicious company tenure
        company_flags, company_risk = self._check_company_tenure(candidate)
        red_flags.extend(company_flags)
        risk_score += company_risk
        
        # Check 3: Experience inconsistency
        exp_flags, exp_risk = self._check_experience_consistency(candidate)
        red_flags.extend(exp_flags)
        risk_score += exp_risk
        
        # Check 4: Education tier mismatch
        edu_flags, edu_risk = self._check_education_mismatch(candidate)
        red_flags.extend(edu_flags)
        risk_score += edu_risk
        
        # Check 5: Behavioral signal anomalies
        signal_flags, signal_risk = self._check_behavioral_anomalies(candidate)
        red_flags.extend(signal_flags)
        risk_score += signal_risk
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        return risk_score, red_flags
    
    def _check_skill_consistency(self, candidate: Dict[str, Any]) -> Tuple[List[str], float]:
        """Check for inconsistent skill proficiency claims."""
        flags = []
        risk = 0.0
        
        if 'skills' not in candidate:
            return flags, risk
        
        skills = candidate['skills']
        expert_count = 0
        advanced_count = 0
        no_duration_count = 0
        
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            
            proficiency = skill.get('proficiency', 'beginner')
            duration = skill.get('duration_months', 0)
            endorsements = skill.get('endorsements', 0)
            
            # Count expert/advanced skills
            if proficiency in ['expert', 'advanced']:
                if proficiency == 'expert':
                    expert_count += 1
                else:
                    advanced_count += 1
                
                # Suspicious: expert with low duration
                if proficiency == 'expert' and duration < 12:
                    no_duration_count += 1
                
                # Suspicious: many endorsements with low duration
                if duration < 6 and endorsements > 20:
                    no_duration_count += 1
        
        # Too many expert skills
        if expert_count > 10:
            risk += 0.3
            flags.append(f"Too many expert skills: {expert_count}")
        elif expert_count > 5:
            risk += 0.15
            flags.append(f"Suspicious number of expert skills: {expert_count}")
        
        # Too many advanced skills
        if advanced_count > 15:
            risk += 0.15
            flags.append(f"Too many advanced skills: {advanced_count}")
        
        # Skills with no duration but high proficiency
        if no_duration_count > 3:
            risk += 0.1
            flags.append(f"{no_duration_count} expert/advanced skills with low duration")
        
        return flags, min(risk, 0.4)
    
    def _check_company_tenure(self, candidate: Dict[str, Any]) -> Tuple[List[str], float]:
        """Check for suspicious company tenure patterns."""
        flags = []
        risk = 0.0
        
        if 'career_history' not in candidate:
            return flags, risk
        
        for exp in candidate['career_history']:
            if not isinstance(exp, dict):
                continue
            
            company = exp.get('company', '').lower()
            duration = exp.get('duration_months', 0)
            is_current = exp.get('is_current', False)
            
            # Check if at suspicious company
            for susp_company, weight in self.suspicious_companies.items():
                if susp_company in company:
                    # Too long at top company (rare)
                    if duration > 60:  # 5+ years
                        risk += 0.2
                        flags.append(f"Unusually long tenure at {susp_company}: {duration} months")
                    # Too short at top company (also suspicious)
                    elif duration < 6 and not is_current:
                        risk += 0.1
                        flags.append(f"Suspiciously short tenure at {susp_company}: {duration} months")
                    break
        
        return flags, min(risk, 0.3)
    
    def _check_experience_consistency(self, candidate: Dict[str, Any]) -> Tuple[List[str], float]:
        """Check for inconsistency in experience claims."""
        flags = []
        risk = 0.0
        
        profile = candidate.get('profile', {})
        stated_years = profile.get('years_of_experience', 0)
        
        # Calculate from career history
        calculated_years = 0.0
        if 'career_history' in candidate:
            for exp in candidate['career_history']:
                if isinstance(exp, dict):
                    if 'duration_months' in exp:
                        calculated_years += exp['duration_months'] / 12.0
                    elif 'start_date' in exp and 'end_date' in exp:
                        try:
                            start = datetime.strptime(exp['start_date'], '%Y-%m-%d')
                            end = datetime.strptime(exp['end_date'], '%Y-%m-%d')
                            calculated_years += (end - start).days / 365.25
                        except:
                            pass
        
        # Compare stated vs calculated
        if stated_years > 0 and calculated_years > 0:
            diff = abs(stated_years - calculated_years)
            if diff > 5:
                risk += 0.25
                flags.append(f"Inconsistent experience: stated {stated_years} years, calculated {calculated_years:.1f} years")
            elif diff > 3:
                risk += 0.1
                flags.append(f"Mildly inconsistent experience: stated {stated_years} vs calculated {calculated_years:.1f}")
        
        return flags, min(risk, 0.25)
    
    def _check_education_mismatch(self, candidate: Dict[str, Any]) -> Tuple[List[str], float]:
        """Check for education tier mismatch with experience."""
        flags = []
        risk = 0.0
        
        if 'education' not in candidate or not candidate['education']:
            return flags, risk
        
        # Get highest education tier
        highest_tier = 4
        for edu in candidate['education']:
            if isinstance(edu, dict):
                tier = edu.get('tier', 'tier_4')
                if tier == 'tier_1':
                    highest_tier = 1
                elif tier == 'tier_2' and highest_tier > 2:
                    highest_tier = 2
                elif tier == 'tier_3' and highest_tier > 3:
                    highest_tier = 3
        
        # Check if high education tier with low experience
        experience = FeatureExtractor.extract_experience_years(candidate)
        if highest_tier <= 2 and experience < 2:
            risk += 0.1
            flags.append(f"High education tier (tier_{highest_tier}) with low experience ({experience} years)")
        
        return flags, min(risk, 0.1)
    
    def _check_behavioral_anomalies(self, candidate: Dict[str, Any]) -> Tuple[List[str], float]:
        """Check for anomalous behavioral signals."""
        flags = []
        risk = 0.0
        
        signals = candidate.get('redrob_signals', {})
        
        # High profile completeness but low activity
        completeness = signals.get('profile_completeness_score', 0)
        last_active = signals.get('last_active_date', '')
        
        if completeness > 80 and last_active:
            try:
                last = datetime.strptime(last_active, '%Y-%m-%d')
                days_ago = (datetime.now() - last).days
                if days_ago > 180:
                    risk += 0.05
                    flags.append(f"High completeness ({completeness}%) but inactive ({days_ago} days)")
            except:
                pass
        
        # High search appearances but low response rate
        search_count = signals.get('search_appearance_30d', 0)
        response_rate = signals.get('recruiter_response_rate', 0)
        
        if search_count > 100 and response_rate < 0.1:
            risk += 0.05
            flags.append(f"High search appearances ({search_count}) but low response rate ({response_rate:.2f})")
        
        return flags, min(risk, 0.1)
    
    def get_risk_level(self, risk_score: float) -> str:
        """Get risk level from risk score."""
        if risk_score >= 0.7:
            return "Very High"
        elif risk_score >= 0.5:
            return "High"
        elif risk_score >= 0.3:
            return "Medium"
        elif risk_score >= 0.1:
            return "Low"
        else:
            return "Very Low"
