import re
import logging
from typing import List, Optional, Set, Dict, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class JDParser:
    """Parse job description to extract key requirements."""
    
    def __init__(self, jd_text: str):
        """
        Initialize parser with job description text.
        
        Args:
            jd_text: Raw job description text
        """
        self.jd_text = jd_text
        self.jd_lower = jd_text.lower()
        
        # Extract all features
        self.skills = self._extract_skills()
        self.required_years = self._extract_years_experience()
        self.role_keywords = self._extract_role_keywords()
        self.location_preferences = self._extract_location_preferences()
        self.must_have = self._extract_must_have_requirements()
        self.nice_to_have = self._extract_nice_to_have_requirements()
        self.is_product_focused = self._is_product_focused()
        self.is_ml_ai_role = self._is_ml_ai_role()
        self.has_vector_search = self._has_vector_search()
        self.has_ranking = self._has_ranking()
        self.has_evaluation = self._has_evaluation()
        self.company_preferences = self._extract_company_preferences()
        self.work_mode_preferences = self._extract_work_mode_preferences()
        self.bonus_keywords = self._extract_bonus_keywords()
        self.red_flags = self._extract_red_flags()
        
        logger.info(f"JD parsed: {len(self.skills)} skills, {self.required_years} years experience")
    
    def _extract_skills(self) -> List[str]:
        """Extract technical skills from JD."""
        skill_keywords = [
            # Programming Languages
            'python', 'java', 'scala', 'c++', 'c#', 'golang', 'go', 'rust',
            'javascript', 'typescript', 'ruby', 'php', 'swift', 'kotlin',
            
            # Web Frameworks
            'react', 'vue', 'angular', 'svelte', 'next.js', 'nuxt',
            'django', 'flask', 'fastapi', 'spring', 'spring boot',
            'node.js', 'express', 'rails', 'laravel',
            
            # Cloud & DevOps
            'docker', 'kubernetes', 'openshift', 'aws', 'gcp', 'azure',
            'terraform', 'ansible', 'puppet', 'chef', 'jenkins',
            'github actions', 'gitlab ci', 'circleci', 'argo',
            
            # ML/AI
            'tensorflow', 'pytorch', 'keras', 'jax', 'scikit-learn',
            'xgboost', 'lightgbm', 'catboost', 'mlflow', 'kubeflow',
            'machine learning', 'deep learning', 'neural network',
            'nlp', 'natural language processing', 'llm', 'large language model',
            'transformer', 'attention', 'bert', 'gpt', 'gemini', 'claude',
            'rag', 'retrieval augmented generation', 'vector database',
            'embedding', 'sentence transformer', 'fine-tuning',
            'prompt engineering', 'langchain', 'llamaindex',
            
            # Data & Databases
            'sql', 'postgresql', 'mysql', 'mongodb', 'cassandra',
            'redis', 'elasticsearch', 'opensearch', 'pinecone',
            'weaviate', 'qdrant', 'milvus', 'faiss', 'vespa',
            'spark', 'hadoop', 'kafka', 'rabbitmq', 'airflow',
            'databricks', 'snowflake', 'bigquery', 'redshift',
            
            # Search & Ranking
            'ranking', 'retrieval', 'search', 'recommendation',
            'learning to rank', 'ltr', 'bm25', 'tf-idf',
            'information retrieval', 'relevance', 'ndcg', 'mrr', 'map',
            
            # Frontend
            'html', 'css', 'tailwind', 'bootstrap', 'webpack',
            'redux', 'mobx', 'zustand', 'graphql', 'rest api',
            
            # Testing & QA
            'pytest', 'unittest', 'selenium', 'cypress', 'jest',
            'load testing', 'performance testing', 'integration testing',
            
            # Other
            'git', 'linux', 'bash', 'shell', 'ci/cd', 'mlops',
            'product management', 'agile', 'scrum', 'jira', 'confluence'
        ]
        
        found_skills = []
        jd = self.jd_lower
        
        for skill in skill_keywords:
            if skill in jd:
                found_skills.append(skill)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in found_skills:
            if skill not in seen:
                seen.add(skill)
                unique_skills.append(skill)
        
        return unique_skills
    
    def _extract_years_experience(self) -> int:
        """Extract minimum years of experience required."""
        patterns = [
            r'(\d+)\+?\s*(?:years|yrs)',
            r'minimum\s*(\d+)\s*(?:years|yrs)',
            r'at least\s*(\d+)\s*(?:years|yrs)',
            r'(\d+)\s*(?:years|yrs)\s*(?:experience|exp)',
            r'(\d+)\s*\+\s*(?:years|yrs)',
            r'(\d+)-(\d+)\s*(?:years|yrs)',
            r'experience.*?(\d+)\s*(?:years|yrs)',
        ]
        
        max_years = 3
        
        for pattern in patterns:
            matches = re.findall(pattern, self.jd_lower)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        try:
                            years = int(match[0])
                            max_years = max(max_years, years)
                        except:
                            pass
                else:
                    try:
                        years = int(match)
                        max_years = max(max_years, years)
                    except:
                        pass
        
        return min(max_years, 15)
    
    def _extract_role_keywords(self) -> List[str]:
        """Extract role-specific keywords."""
        keywords = [
            'senior', 'junior', 'lead', 'principal', 'staff', 'entry level',
            'full stack', 'frontend', 'backend', 'devops', 'mlops',
            'data engineer', 'data scientist', 'data analyst', 'research',
            'product', 'software engineer', 'developer', 'programmer',
            'architect', 'manager', 'director', 'vp', 'cto',
            'applied scientist', 'research scientist', 'machine learning engineer',
            'ai engineer', 'prompt engineer', 'sre', 'platform engineer'
        ]
        
        found = []
        for keyword in keywords:
            if keyword in self.jd_lower:
                found.append(keyword)
        
        return found
    
    def _extract_location_preferences(self) -> List[str]:
        """Extract location preferences from JD."""
        locations = {
            'pune': ['pune', 'pune,', 'pun'],
            'noida': ['noida', 'noida,'],
            'bangalore': ['bangalore', 'bengaluru', 'blr'],
            'hyderabad': ['hyderabad', 'hyd'],
            'mumbai': ['mumbai', 'bombay'],
            'delhi': ['delhi', 'new delhi', 'ncr'],
            'chennai': ['chennai', 'madras'],
            'kolkata': ['kolkata', 'calcutta'],
        }
        
        found = []
        for city, patterns in locations.items():
            for pattern in patterns:
                if pattern in self.jd_lower:
                    found.append(city)
                    break
        
        if 'tier-1' in self.jd_lower:
            found.append('tier1_city')
        
        return found
    
    def _extract_must_have_requirements(self) -> List[str]:
        """Extract 'must have' requirements from JD."""
        must_have = []
        
        patterns = [
            r'must have\s*(?:experience with|knowledge of|expertise in)?\s*([^.]+)',
            r'must\s*([^.]+)',
            r'required\s*(?:experience with|knowledge of|expertise in)?\s*([^.]+)',
            r'essential\s*(?:experience with|knowledge of|expertise in)?\s*([^.]+)',
            r'absolutely need\s*([^.]+)',
            r'non-negotiable\s*([^.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, self.jd_lower)
            for match in matches:
                clean = match.strip()
                if clean and len(clean) < 100:
                    must_have.append(clean)
        
        return must_have[:10]
    
    def _extract_nice_to_have_requirements(self) -> List[str]:
        """Extract 'nice to have' requirements from JD."""
        nice_to_have = []
        
        patterns = [
            r'nice to have\s*([^.]+)',
            r'bonus\s*(?:points|if)\s*([^.]+)',
            r'preferred\s*(?:experience|qualification)?\s*([^.]+)',
            r'would be (?:great|good|ideal|awesome)\s*(?:if|to have)?\s*([^.]+)',
            r'optional\s*([^.]+)',
            r'we\'d like you to have\s*([^.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, self.jd_lower)
            for match in matches:
                clean = match.strip()
                if clean and len(clean) < 100:
                    nice_to_have.append(clean)
        
        return nice_to_have[:10]
    
    def _is_product_focused(self) -> bool:
        """Check if JD emphasizes product company experience."""
        indicators = [
            'product company', 'product-company', 'product driven',
            'product mindset', 'shipped', 'product market', 'product-led',
            'building products', 'product development', 'product culture'
        ]
        return any(ind in self.jd_lower for ind in indicators)
    
    def _is_ml_ai_role(self) -> bool:
        """Check if this is an AI/ML role."""
        indicators = [
            'ai', 'ml', 'machine learning', 'deep learning',
            'neural', 'embedding', 'rag', 'llm', 'large language',
            'transformer', 'attention', 'fine-tuning', 'pytorch',
            'tensorflow', 'ranking', 'retrieval'
        ]
        return any(ind in self.jd_lower for ind in indicators)
    
    def _has_vector_search(self) -> bool:
        """Check if JD mentions vector search/embedding requirements."""
        indicators = [
            'vector', 'embedding', 'retrieval', 'rag',
            'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss',
            'vector database', 'sentence transformer'
        ]
        return any(ind in self.jd_lower for ind in indicators)
    
    def _has_ranking(self) -> bool:
        """Check if JD mentions ranking/recommendation."""
        indicators = [
            'ranking', 'rank', 'recommendation', 'recommend',
            'retrieval', 'search relevance', 'learning to rank'
        ]
        return any(ind in self.jd_lower for ind in indicators)
    
    def _has_evaluation(self) -> bool:
        """Check if JD mentions evaluation/measurement."""
        indicators = [
            'ndcg', 'mrr', 'map', 'precision', 'recall',
            'evaluation', 'benchmark', 'ab test', 'a/b test',
            'metric', 'offline', 'online', 'correlation'
        ]
        return any(ind in self.jd_lower for ind in indicators)
    
    def _extract_company_preferences(self) -> Dict[str, bool]:
        """Extract company type preferences."""
        return {
            'startup': 'startup' in self.jd_lower,
            'product': 'product' in self.jd_lower and 'services' not in self.jd_lower,
            'services': 'services' in self.jd_lower and 'product' not in self.jd_lower,
            'enterprise': 'enterprise' in self.jd_lower,
            'consulting': any(c in self.jd_lower for c in ['consulting', 'consultancy']),
        }
    
    def _extract_work_mode_preferences(self) -> Dict[str, bool]:
        """Extract work mode preferences."""
        return {
            'remote': 'remote' in self.jd_lower,
            'hybrid': 'hybrid' in self.jd_lower,
            'onsite': 'onsite' in self.jd_lower or 'in-office' in self.jd_lower,
            'flexible': 'flexible' in self.jd_lower,
        }
    
    def _extract_bonus_keywords(self) -> List[str]:
        """Extract bonus/extra keywords."""
        keywords = []
        patterns = [
            r'bonus\s*(?:points|if)?\s*([^.]+)',
            r'extra credit\s*([^.]+)',
            r'plus\s*([^.]+)',
            r'additional\s*(?:points|credit)?\s*([^.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, self.jd_lower)
            keywords.extend([m.strip() for m in matches if m.strip() and len(m.strip()) < 50])
        
        return keywords[:5]
    
    def _extract_red_flags(self) -> List[str]:
        """Extract red flags from JD."""
        flags = []
        
        # Check for explicit disqualifiers
        disqualifiers = [
            'not a fit', 'will not move forward', 'disqualify',
            'explicitly do not want', 'not looking for',
        ]
        
        for d in disqualifiers:
            if d in self.jd_lower:
                # Extract the surrounding text
                idx = self.jd_lower.find(d)
                start = max(0, idx - 50)
                end = min(len(self.jd_lower), idx + 100)
                flags.append(self.jd_lower[start:end].strip())
        
        return flags
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of JD parsing results."""
        return {
            'skills_count': len(self.skills),
            'required_years': self.required_years,
            'role_keywords': self.role_keywords,
            'locations': self.location_preferences,
            'must_have_count': len(self.must_have),
            'nice_to_have_count': len(self.nice_to_have),
            'is_product_focused': self.is_product_focused,
            'is_ml_ai_role': self.is_ml_ai_role,
            'has_vector_search': self.has_vector_search,
            'has_ranking': self.has_ranking,
            'has_evaluation': self.has_evaluation,
            'work_mode': self.work_mode_preferences,
            'company_type': self.company_preferences,
        }
