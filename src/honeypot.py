"""Honeypot detection for identifying fake/impossible candidate profiles."""

from typing import Dict, Any, List, Tuple
from datetime import datetime
from .feature_extractor import FeatureExtractor  # ← ADD THIS IMPORT

class HoneypotDetector:
    def __init__(self):
        self.suspicious_companies = {
            'nasa':0.3,'spacex':0.3,'tesla':0.2,'openai':0.3,
            'anthropic':0.3,'google':0.2,'facebook':0.2,'apple':0.2,
            'microsoft':0.2,'amazon':0.2,'netflix':0.2,'uber':0.2
        }

    def detect(self, candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
        risk = 0.0
        flags = []
        # 1. Skill consistency
        skill_flags, skill_risk = self._check_skill_consistency(candidate)
        flags.extend(skill_flags); risk += skill_risk
        # 2. Company tenure
        comp_flags, comp_risk = self._check_company_tenure(candidate)
        flags.extend(comp_flags); risk += comp_risk
        # 3. Experience inconsistency
        exp_flags, exp_risk = self._check_experience_consistency(candidate)
        flags.extend(exp_flags); risk += exp_risk
        # 4. Education mismatch
        edu_flags, edu_risk = self._check_education_mismatch(candidate)
        flags.extend(edu_flags); risk += edu_risk
        # 5. Behavioral anomalies
        beh_flags, beh_risk = self._check_behavioral_anomalies(candidate)
        flags.extend(beh_flags); risk += beh_risk
        return min(risk, 1.0), flags

    def _check_skill_consistency(self, candidate):
        flags=[]; risk=0.0
        skills = candidate.get('skills', [])
        expert_count = 0
        low_duration_expert = 0
        for s in skills:
            if isinstance(s, dict):
                prof = s.get('proficiency', '')
                duration = s.get('duration_months', 0)
                if prof in ['expert','advanced']:
                    if prof == 'expert':
                        expert_count += 1
                    if duration < 12:
                        low_duration_expert += 1
        if expert_count > 10:
            risk += 0.3; flags.append(f"Too many expert skills: {expert_count}")
        elif expert_count > 5:
            risk += 0.15; flags.append(f"Suspicious expert skills: {expert_count}")
        if low_duration_expert > 3:
            risk += 0.1; flags.append(f"{low_duration_expert} expert skills with low duration")
        return flags, min(risk, 0.4)

    def _check_company_tenure(self, candidate):
        flags=[]; risk=0.0
        for exp in candidate.get('career_history', []):
            if isinstance(exp, dict):
                company = exp.get('company', '').lower()
                duration = exp.get('duration_months', 0)
                is_current = exp.get('is_current', False)
                for susp, w in self.suspicious_companies.items():
                    if susp in company:
                        if duration > 60:
                            risk += 0.2; flags.append(f"Long tenure at {susp}: {duration}mo")
                        elif duration < 6 and not is_current:
                            risk += 0.1; flags.append(f"Short tenure at {susp}: {duration}mo")
                        break
        return flags, min(risk, 0.3)

    def _check_experience_consistency(self, candidate):
        flags=[]; risk=0.0
        profile = candidate.get('profile', {})
        stated = profile.get('years_of_experience', 0)
        calculated = FeatureExtractor.extract_experience_years(candidate)  # ← Uses imported FeatureExtractor
        if stated > 0 and calculated > 0:
            diff = abs(stated - calculated)
            if diff > 5:
                risk += 0.25; flags.append(f"Inconsistent experience: stated {stated}, calculated {calculated:.1f}")
            elif diff > 3:
                risk += 0.1
        return flags, min(risk, 0.25)

    def _check_education_mismatch(self, candidate):
        flags=[]; risk=0.0
        tier = FeatureExtractor.get_education_tier(candidate)  # ← Uses imported FeatureExtractor
        exp = FeatureExtractor.extract_experience_years(candidate)  # ← Uses imported FeatureExtractor
        if tier <= 2 and exp < 2:
            risk += 0.1; flags.append(f"High education tier ({tier}) with low experience ({exp})")
        return flags, min(risk, 0.1)

    def _check_behavioral_anomalies(self, candidate):
        flags=[]; risk=0.0
        signals = candidate.get('redrob_signals', {})
        completeness = signals.get('profile_completeness_score', 0)
        last_active = signals.get('last_active_date', '')
        if completeness > 80 and last_active:
            try:
                last = datetime.strptime(last_active, '%Y-%m-%d')
                days = (datetime.now() - last).days
                if days > 180:
                    risk += 0.05; flags.append(f"High completeness ({completeness}) but inactive ({days}d)")
            except: pass
        search = signals.get('search_appearance_30d', 0)
        response = signals.get('recruiter_response_rate', 0)
        if search > 100 and response < 0.1:
            risk += 0.05; flags.append(f"High search ({search}) low response ({response:.2f})")
        return flags, min(risk, 0.1)
