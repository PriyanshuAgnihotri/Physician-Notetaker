"""sentiment.py
Refactored SentimentIntentAnalyzer with clear structure and fallbacks.
"""
from typing import List, Dict
try:
    from transformers import pipeline
except Exception:
    pipeline = None

class SentimentIntentAnalyzer:
    def __init__(self):
        self._model = None
        try:
            if pipeline:
                self._model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        except Exception:
            self._model = None

    def extract_patient_utterances(self, text: str) -> List[str]:
        lines = text.split('\n')
        return [line.replace('Patient:', '').strip() for line in lines if line.strip().startswith('Patient:')]

    def analyze_sentiment(self, text: str) -> str:
        if not text or not text.strip():
            return 'Neutral'
        t = text.lower()
        anxiety = any(w in t for w in ['worried','concerned','anxious','nervous','scared','afraid'])
        reassurance = any(w in t for w in ['better','relief','good','great','thank','appreciate','relieved'])
        if anxiety and not reassurance:
            return 'Anxious'
        if reassurance and not anxiety:
            return 'Reassured'
        if self._model:
            try:
                res = self._model(text[:512])[0]
                label = res.get('label','').upper(); score = res.get('score',0.0)
                if label == 'NEGATIVE' and score > 0.75:
                    return 'Anxious'
                if label == 'POSITIVE' and score > 0.75:
                    return 'Reassured'
            except Exception:
                pass
        return 'Neutral'

    def detect_intent(self, text: str) -> str:
        t = (text or '').lower()
        if any(k in t for k in ['worried','concern','hope','?']):
            return 'Seeking reassurance'
        if any(k in t for k in ['feel','have','experiencing','pain','hurt']):
            return 'Reporting symptoms'
        if any(k in t for k in ['thank','appreciate']):
            return 'Expressing gratitude'
        if any(k in t for k in ['when','what','how','where','accident']):
            return 'Providing information'
        if any(k in t for k in ['better','improving']):
            return 'Reporting progress'
        return 'General communication'

    def analyze_conversation_sentiment(self, text: str) -> List[Dict]:
        utterances = self.extract_patient_utterances(text)
        return [{'Utterance': u if len(u)<=120 else u[:120]+'...', 'Sentiment': self.analyze_sentiment(u), 'Intent': self.detect_intent(u)} for u in utterances]

    def get_overall_sentiment(self, text: str) -> Dict:
        utterances = self.extract_patient_utterances(text)
        sentiments = [self.analyze_sentiment(u) for u in utterances]
        counts = {'Anxious': sentiments.count('Anxious'), 'Neutral': sentiments.count('Neutral'), 'Reassured': sentiments.count('Reassured')}
        overall = max(counts, key=counts.get) if utterances else 'Neutral'
        return {'Overall_Sentiment': overall, 'Sentiment_Distribution': counts, 'Total_Utterances': len(utterances)}
