"""summarizer.py - refactored from user's code"""
import re
from typing import List, Dict
import spacy

class MedicalNLPSummarizer:
    def __init__(self, nlp_model: str = 'en_core_web_sm') -> None:
        try:
            # prefer clinical model if available
            self.nlp = spacy.load('en_core_sci_sm')
        except Exception:
            self.nlp = spacy.load(nlp_model)

        self._symptoms = [
            'pain','ache','discomfort','hurt','sore','stiff','trouble','difficulty',
            'shock','impact','injury','trauma'
        ]
        self._treatments = [
            'therapy','treatment','session','medication','painkiller','analgesic',
            'examination','x-ray','scan','advice'
        ]
        self._body_parts = [
            'neck','back','head','spine','muscle','shoulder','cervical','lumbar',
            'steering wheel'
        ]

    def _find_name(self, text: str) -> str:
        """Try honorifics first, then spaCy PERSON entities."""
        for pattern in [r'Ms\.\s+([A-Z][a-z]+)', r'Mr\.\s+([A-Z][a-z]+)', r'Mrs\.\s+([A-Z][a-z]+)']:
            m = re.search(pattern, text)
            if m:
                return m.group(1)
        for ent in self.nlp(text).ents:
            if ent.label_ == 'PERSON':
                return ent.text
        return 'Unknown'

    def _sentences(self, text: str) -> List[str]:
        return [s.text.strip() for s in self.nlp(text).sents]

    def extract_symptoms(self, text: str) -> List[str]:
        """Return a short list of symptom phrases found in text."""
        found = set()
        for s in self._sentences(text.lower()):
            if 'patient:' in s or any(k in s for k in ['i have','i feel','my','i had','i was']):
                for part in self._body_parts:
                    for sym in self._symptoms:
                        if part in s and sym in s:
                            if 'pain' in sym:
                                found.add(f"{part.title()} pain")
                            else:
                                found.add(f"{part.title()} {sym}")
        # extra checks
        if 'head' in text.lower() and ('hit' in text.lower() or 'impact' in text.lower()):
            found.add('Head impact')
        if 'trouble sleeping' in text.lower():
            found.add('Trouble sleeping')
        return sorted(found)

    def extract_diagnosis(self, text: str) -> str:
        if 'whiplash' in text.lower():
            return 'Whiplash injury'
        return 'Not specified'

    def extract_treatment(self, text: str) -> List[str]:
        t = text.lower()
        out = []
        if 'physiotherapy' in t:
            m = re.search(r'(\d+)\s+sessions?\s+of\s+physiotherapy|physiotherapy.*?(\d+)\s+sessions?', t)
            if m:
                num = m.group(1) or m.group(2)
                out.append(f"{num} physiotherapy sessions")
            else:
                out.append('Physiotherapy sessions')
        if 'painkiller' in t or 'analgesic' in t:
            out.append('Painkillers')
        if 'physical examination' in t or 'physical exam' in t:
            out.append('Physical examination')
        return out

    def extract_current_status(self, text: str) -> str:
        t = text.lower()
        if 'occasional' in t and 'back' in t:
            return 'Occasional backache'
        if 'doing better' in t or 'doing well' in t:
            return 'Improving, occasional discomfort'
        return 'Not specified'

    def extract_prognosis(self, text: str) -> str:
        t = text.lower()
        if 'full recovery' in t:
            m = re.search(r'within\s+(\d+\s+(?:days|weeks|months|years)|\w+\s+\w+)', t)
            if m:
                return f"Full recovery expected within {m.group(1)}"
            return 'Full recovery expected'
        return 'Not specified'

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        doc = self.nlp(text)
        candidates = []
        for chunk in doc.noun_chunks:
            ct = chunk.text.lower()
            if any(k in ct for k in self._symptoms + self._treatments + self._body_parts):
                if len(ct.split()) <= 5:
                    candidates.append(chunk.text.strip())
        for phrase in ['whiplash injury','car accident','physiotherapy sessions','physical examination','full recovery']:
            if phrase in text.lower():
                candidates.append(phrase.title())
        # dedupe preserving order
        seen = set(); out = []
        for c in candidates:
            if c not in seen:
                seen.add(c); out.append(c)
        return out[:top_n]

    def generate_summary(self, text: str) -> Dict:
        return {
            'Patient_Name': self._find_name(text),
            'Symptoms': self.extract_symptoms(text),
            'Diagnosis': self.extract_diagnosis(text),
            'Treatment': self.extract_treatment(text),
            'Current_Status': self.extract_current_status(text),
            'Prognosis': self.extract_prognosis(text),
            'Keywords': self.extract_keywords(text)
        }
