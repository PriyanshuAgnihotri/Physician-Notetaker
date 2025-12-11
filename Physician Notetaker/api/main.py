from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from src.medical_nlp.summarizer import MedicalNLPSummarizer
from src.medical_nlp.sentiment import SentimentIntentAnalyzer
from src.medical_nlp.soap import SOAPNoteGenerator

app = FastAPI(title='Physician Notetaker API')

summarizer = MedicalNLPSummarizer()
sentimenter = SentimentIntentAnalyzer()
soapgen = SOAPNoteGenerator()

class TranscriptIn(BaseModel):
    transcript: str
    patient_name: str = 'Unknown'

@app.post('/analyze')
def analyze(data: TranscriptIn) -> Dict:
    text = data.transcript
    summary = summarizer.generate_summary(text)
    sample = ''
    for line in text.split('\n'):
        if line.strip().lower().startswith('patient:'):
            sample = line.replace('Patient:', '').strip()
            break
    sentiment = sentimenter.analyze_sentiment(sample) if sample else 'Neutral'
    intent = sentimenter.detect_intent(sample) if sample else 'Neutral'
    soap = soapgen.generate_soap_note(text)
    return {
        'structured_summary': summary,
        'sentiment_sample': {'sample': sample, 'sentiment': sentiment, 'intent': intent},
        'soap': soap
    }
