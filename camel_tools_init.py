from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.sentiment import SentimentAnalyzer
from camel_tools.tagger.default import DefaultTagger
from transformers import AutoTokenizer, AutoModel
from camel_tools.dialectid import DialectIdentifier
import torch

_mle_disambiguator = None
_morph_analyzer = None
_sentiment_analyzer = None
_bert_tokenizer = None
_bert_model = None
_default_tagger = None
_dialect_id = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_disambiguator():
    """Get or initialize the MLE disambiguator."""
    global _mle_disambiguator
    if _mle_disambiguator is None:
        _mle_disambiguator = MLEDisambiguator.pretrained()
    return _mle_disambiguator

def get_analyzer():
    """Get or initialize the morphological analyzer."""
    global _morph_analyzer
    if _morph_analyzer is None:
        # Load the database and create an analyzer instance
        db = MorphologyDB.builtin_db()
        _morph_analyzer = Analyzer(db)
    return _morph_analyzer

# Modified getter functions with error handling
def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer.pretrained()
    return _sentiment_analyzer

def get_bert_model(model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix"):
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _bert_model is None:
        _bert_model = AutoModel.from_pretrained(model_name).to(device)
    return _bert_tokenizer, _bert_model

def get_tagger():
    global _default_tagger, _mle_disambiguator
    if _default_tagger is None:
        if _mle_disambiguator is None:
            _mle_disambiguator = get_disambiguator()
        if _mle_disambiguator is not None:
            _default_tagger = DefaultTagger(_mle_disambiguator, 'pos')  
    return _default_tagger

def get_dialect_id():
    global _dialect_id
    if _dialect_id is None:
        _dialect_id = DialectIdentifier.pretrained()
    return _dialect_id