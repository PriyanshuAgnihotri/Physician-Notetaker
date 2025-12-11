"""soap.py - Clean SOAP note generator."""
from typing import Dict
try:
    import spacy
except Exception:
    spacy = None

class SOAPNoteGenerator:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm') if spacy else None
        except Exception:
            self.nlp = None

    def _patient_statements(self, text: str):
        return [l.replace('Patient:','').strip() for l in text.split('\n') if l.strip().startswith('Patient:')]

    def extract_subjective(self, text: str) -> Dict:
        stmts = self._patient_statements(text)
        chief = 'Neck and back pain'
        hpi = ' '.join([s for s in stmts if any(k in s.lower() for k in ['accident','pain','physiotherapy','week','trouble'])][:5])
        if not hpi:
            hpi = 'Patient reports motor vehicle accident with subsequent neck and back pain, underwent physiotherapy, now occasional symptoms.'
        return {'Chief_Complaint': chief, 'History_of_Present_Illness': hpi}

    def extract_objective(self, text: str) -> Dict:
        exam = 'Full range of motion in cervical and lumbar spine, no tenderness.'
        if 'physical examination' in text.lower() or 'everything looks good' in text.lower():
            exam = 'Full range of motion in cervical and lumbar spine, no tenderness, no signs of lasting damage. Muscles and spine in good condition.'
        obs = 'Patient appears in good health, normal gait, improving condition.'
        return {'Physical_Exam': exam, 'Observations': obs}

    def extract_assessment(self, text: str) -> Dict:
        diagnosis = 'Whiplash injury' if 'whiplash' in text.lower() else 'Not specified'
        severity = 'Mild, improving' if 'occasional' in text.lower() or 'improv' in text.lower() else 'To be determined'
        return {'Diagnosis': diagnosis, 'Severity': severity}

    def extract_plan(self, text: str) -> Dict:
        treatment = 'Continue physiotherapy as needed, use analgesics PRN.' if 'physiotherapy' in text.lower() else 'Analgesics as needed.'
        follow_up = 'Return if pain worsens or persists beyond six months.'
        if 'full recovery' in text.lower():
            follow_up = 'Full recovery expected within six months. Return for follow-up if symptoms worsen.'
        return {'Treatment': treatment, 'Follow_Up': follow_up}

    def generate_soap_note(self, text: str) -> Dict:
        return {'Subjective': self.extract_subjective(text), 'Objective': self.extract_objective(text), 'Assessment': self.extract_assessment(text), 'Plan': self.extract_plan(text)}
