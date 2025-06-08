from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

# Initialize global variables
_mle_disambiguator = None
_morph_analyzer = None

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