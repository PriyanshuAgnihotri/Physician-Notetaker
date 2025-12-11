# Medical Transcription NLP Pipeline

A comprehensive NLP system for medical transcription analysis, including entity extraction, sentiment analysis, and automated SOAP note generation.

## 🎯 Overview

This project implements three core tasks:
1. **Medical NLP Summarization** - Extract symptoms, diagnosis, treatment, and prognosis from medical conversations
2. **Sentiment & Intent Analysis** - Classify patient sentiment (Anxious/Neutral/Reassured) and detect intent
3. **SOAP Note Generation** - Automatically generate structured clinical documentation

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Core Dependencies
```bash
spacy>=3.5.0
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

### Optional Medical NLP Libraries
```bash
medspacy>=1.0.0
scispacy>=0.5.1
```

## 🚀 Installation

### Step 1: Clone or Download the Repository
```bash
# If using git
git clone <repository-url>
cd medical-nlp-pipeline

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
# Install core packages
pip install spacy transformers torch scikit-learn pandas numpy

# Download spaCy language model
python -m spacy download en_core_web_sm

# Optional: Install medical NLP packages
pip install medspacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

### Step 4: Verify Installation
```python
import spacy
import transformers
print("✓ All packages installed successfully!")
```

## 💻 Usage

### Running the Complete Pipeline

1. **Open Jupyter Notebook**
```bash
jupyter notebook medical_transcription_nlp.ipynb
```

2. **Run All Cells**
   - The notebook will automatically process the sample conversation
   - Results will be displayed in structured JSON format

### Using Individual Components

#### Task 1: Medical Summarization
```python
from medical_transcription_nlp import MedicalNLPSummarizer

# Initialize
summarizer = MedicalNLPSummarizer()

# Process transcript
summary = summarizer.generate_summary(your_transcript_text)

# Output structured summary
print(json.dumps(summary, indent=2))
```

**Output Format:**
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["Ten physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months",
  "Keywords": ["whiplash injury", "car accident", ...]
}
```

#### Task 2: Sentiment Analysis
```python
from medical_transcription_nlp import SentimentIntentAnalyzer

# Initialize
analyzer = SentimentIntentAnalyzer()

# Analyze single utterance
utterance = "I'm worried about my back pain"
sentiment = analyzer.analyze_sentiment(utterance)
intent = analyzer.detect_intent(utterance)

print(f"Sentiment: {sentiment}")
print(f"Intent: {intent}")

# Analyze full conversation
overall = analyzer.get_overall_sentiment(conversation_text)
```

**Output Format:**
```json
{
  "Utterance": "I'm worried about my back pain...",
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance"
}
```

#### Task 3: SOAP Note Generation
```python
from medical_transcription_nlp import SOAPNoteGenerator

# Initialize
soap_gen = SOAPNoteGenerator()

# Generate SOAP note
soap_note = soap_gen.generate_soap_note(your_transcript_text)

print(json.dumps(soap_note, indent=2))
```

**Output Format:**
```json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "..."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion...",
    "Observations": "..."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy...",
    "Follow_Up": "..."
  }
}
```

### Processing Custom Transcripts

```python
# Process your own transcript
custom_transcript = """
Doctor: How are you feeling?
Patient: I have a headache and feel dizzy.
...
"""

# Get complete analysis
results = process_custom_transcript(custom_transcript)

# Save to file
save_results_to_json(results, "my_analysis.json")
```

## 🏗️ Architecture

### Component Overview

```
medical_transcription_nlp.py
├── MedicalNLPSummarizer
│   ├── extract_patient_name()
│   ├── extract_symptoms()
│   ├── extract_diagnosis()
│   ├── extract_treatment()
│   ├── extract_prognosis()
│   └── generate_summary()
│
├── SentimentIntentAnalyzer
│   ├── analyze_sentiment()
│   ├── detect_intent()
│   └── get_overall_sentiment()
│
└── SOAPNoteGenerator
    ├── extract_subjective()
    ├── extract_objective()
    ├── extract_assessment()
    ├── extract_plan()
    └── generate_soap_note()
```

## 🎓 Technical Approach

### Task 1: Medical NLP Summarization

**Approach:**
- **NER (Named Entity Recognition)**: Uses spaCy with medical entity patterns
- **Pattern Matching**: Custom regex for symptoms, treatments, and medical terms
- **Rule-Based Extraction**: Domain-specific rules for diagnosis and prognosis
- **Keyword Extraction**: Noun phrase extraction with medical filtering

**Handling Missing Data:**
- Default values for unspecified fields
- Fallback patterns for ambiguous information
- Context-aware extraction using sentence boundaries

**Models Used:**
- Primary: `en_core_sci_sm` (SciSpacy medical model)
- Fallback: `en_core_web_sm` (general English model)

### Task 2: Sentiment & Intent Analysis

**Approach:**
- **Transformer Model**: DistilBERT fine-tuned on sentiment classification
- **Rule-Based Augmentation**: Medical-specific sentiment keywords
- **Intent Classification**: Pattern-based intent detection
- **Hybrid System**: Combines deep learning with domain rules

**Model Selection:**
- `distilbert-base-uncased-finetuned-sst-2-english` for efficiency
- Can be replaced with medical-specific BERT models

**Fine-tuning Strategy:**
For medical sentiment:
1. Collect annotated medical conversation data
2. Label sentiments: Anxious, Neutral, Reassured
3. Fine-tune BERT on domain-specific data
4. Validate on held-out test set

**Recommended Datasets:**
- MIMIC-III Clinical Notes
- Medical Transcription Dataset (Kaggle)
- PubMed abstracts with sentiment annotations

### Task 3: SOAP Note Generation

**Approach:**
- **Section Mapping**: Rule-based extraction for SOAP sections
- **Utterance Classification**: Categorize statements by speaker
- **Information Aggregation**: Combine related information
- **Template Filling**: Structure extraction into SOAP format

**Techniques:**
- **Rule-Based**: Pattern matching for structured sections
- **Deep Learning Enhancement**: Could use sequence-to-sequence models
- **Hybrid Approach**: Rules for structure, ML for content

**Improvement Strategies:**
1. Fine-tune T5/BART on SOAP note generation
2. Use few-shot learning with GPT-based models
3. Implement section classifiers for better accuracy
4. Add clinical validation rules

## 📊 Performance Considerations

### Accuracy Metrics
- **NER**: Precision/Recall for entity extraction
- **Sentiment**: F1-score across sentiment classes
- **SOAP**: Structural accuracy and clinical validity

### Optimization Tips
1. **Batch Processing**: Process multiple transcripts together
2. **Model Caching**: Load models once, reuse for multiple documents
3. **GPU Acceleration**: Use CUDA for transformer models
4. **Parallel Processing**: Multi-thread for large datasets

## 🐛 Troubleshooting

### Common Issues

**1. spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

**2. CUDA Out of Memory (for large models)**
```python
# Use CPU instead
device = torch.device("cpu")
```

**3. Transformers Library Version Conflicts**
```bash
pip install --upgrade transformers
```

**4. Missing Dependencies**
```bash
pip install -r requirements.txt
```

## 🔬 Advanced Usage

### Custom Model Integration

```python
# Use custom medical BERT model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Integrate into SentimentIntentAnalyzer
analyzer = SentimentIntentAnalyzer()
analyzer.sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
```

### Extending the Pipeline

```python
# Add new entity extraction
class ExtendedMedicalSummarizer(MedicalNLPSummarizer):
    def extract_medications(self, text):
        # Your custom medication extraction logic
        pass
    
    def extract_vital_signs(self, text):
        # Extract blood pressure, heart rate, etc.
        pass
```

## 📚 References

- **spaCy**: https://spacy.io/
- **Transformers**: https://huggingface.co/docs/transformers
- **SciSpacy**: https://allenai.github.io/scispacy/
- **Clinical BERT**: https://github.com/EmilyAlsentzer/clinicalBERT

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Enhanced medical entity recognition
- Better sentiment classification for medical context
- More sophisticated SOAP note generation
- Support for multiple languages
- Integration with EHR systems


## 👥 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on the repository

## 🔄 Updates

**Version 1.0** (Current)
- Initial implementation of all three tasks
- Support for basic medical transcription analysis
- Hybrid rule-based and ML approach

**Planned Features:**
- Deep learning SOAP note generation
- Real-time transcription support
- Multi-language support
- API endpoint integration

---

**"Happy Coding!" ~Priyanshu🚀**

