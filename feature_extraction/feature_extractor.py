import re
import math
import numpy as np
from nltk.tokenize import word_tokenize
from farasa.stemmer import FarasaStemmer
from farasa.pos import FarasaPOSTagger
import farasa.pos as pos
from collections import Counter
from spellchecker import SpellChecker #this is a more general spellchecker not specific for arabic text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download NLTK data for Arabic tokenization
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Compute lexical density
def lexical_density(essay):
    original_words = word_tokenize(re.sub(r'[^\w\s]', '', essay)) 
    tagger = FarasaPOSTagger(interactive=False)
    pos_tagged = (tagger.tag(essay)).replace("S/S", "").replace("E/E", "") #removed start and end symbols for a sentence

    #preprocess the produced tags (will not addect anything)
    pattern = r'(\w) \+(\w)'
    modified_pos_tagged = re.sub(pattern, r'\1\2', pos_tagged) 
    pattern = r'/([^\s]+)'
    tags = re.findall(pattern, modified_pos_tagged)
    content_words=0

    for tag in tags:
        if tag.startswith('NOUN') or tag.startswith('V') or tag.startswith("ADJ") or tag.startswith("ADV"):  
            content_words +=1 

    lexical_density = content_words/len(original_words)
    return {'lexical_density': lexical_density}

def extract_surface_features(essay):
    
    #words
    words = word_tokenize(re.sub(r'[^\w\s]', '', essay)) #this way I remove all punctuation marks (any char that is not a word \w or space \s)
    words_count = len(words)
    log_words_count = math.log10(words_count)
    unique_words = set(words)
    unique_words_count = len(unique_words) #the set() removes duplicates
    log_unique_words_count = math.log10(unique_words_count)
    total_word_length = sum(len(word) for word in unique_words)#for unique words
    average_word_length = total_word_length / unique_words_count if unique_words_count > 0 else 0
    max_length_word = max(len(word) for word in unique_words)
    min_length_word = min(len(word) for word in unique_words)
    squared_diffs_words = [(len(word) - average_word_length) ** 2 for word in unique_words]
    mean_squared_diffs_words = sum(squared_diffs_words) / len(squared_diffs_words)
    standard_deviation_words= math.sqrt(mean_squared_diffs_words) #the standard deviation as a way to understand how much individual values within a group differ from the average value of that group

    #General counts
    chars_count = len(essay.replace(" ", "")) #not counting spaces
    hmpz_count = len(re.findall(r'[أإءؤئ]', essay))# Number of <hmzp> (F22)
    
    #Paragraphs
    paragraphs = essay.split('\n\n')  
    paragraphs_count =len(paragraphs)  #num_paragraphs = len(essay.split('\n')) #Number of paragraphs (F3)
    is_first_paragraph_less_than_or_equal_to_10 = int(len(word_tokenize(re.sub(r'[^\w\s]', '', paragraphs[0]))) <= 10 )#(F16)
    paragraphs_lengths = [len(word_tokenize(re.sub(r'[^\w\s]', '', paragraph))) for paragraph in paragraphs] #length of each paragraph interms of words
    average_length_paragraph = sum(paragraphs_lengths)/ paragraphs_count# Average length of paragraph (F11)
    max_length_paragraph = max(paragraphs_lengths) # Maximum length of paragraph (F12)
    min_length_paragraph = min(paragraphs_lengths) # Minimum length of paragraph (F13)
    has_parentheses = int(any('(' in part or ')' in part for part in paragraphs))    # Paragraph contains parentheses (F19)
    has_colon = int(any(':' in part for part in paragraphs) )   # Paragraph contains colon (F20)
    has_question_mark = int(any('؟' in part for part in paragraphs))    # Paragraph contains question mark (F21)
   
    #Sentences
    sentences = list(filter(str.strip, re.split(r'[,،!؛:.؟]', essay))) #spliting based on these sentences separators
    sentences_count = len(sentences) # Number of sentences (F5)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    average_length_sentence = sum(sentence_lengths) / sentences_count    # Average length of sentence (F10)
    max_length_sentence = max(sentence_lengths) # Maximum length of sentence 
    min_length_sentence = min(sentence_lengths)# Minimum length of sentence 
    squared_diff_sentence = [(length - average_length_sentence) ** 2 for length in sentence_lengths]
    mean_squared_diff_sentence = np.mean(squared_diff_sentence)
    standard_deviation_sentence = np.sqrt(mean_squared_diff_sentence)   
    
    #Grouping the features into a list
    extracted_surface_features= [words_count,log_words_count,unique_words_count,log_unique_words_count,
    average_word_length,max_length_word,min_length_word,standard_deviation_words,chars_count,hmpz_count, 
    paragraphs_count,is_first_paragraph_less_than_or_equal_to_10,average_length_paragraph,
    max_length_paragraph, min_length_paragraph, has_parentheses, has_colon,
    has_question_mark, sentences_count, average_length_sentence,
    max_length_sentence, min_length_sentence,standard_deviation_sentence]

    features = {
        "words_count": words_count,
        "log_words_count": log_words_count,
        "unique_words_count": unique_words_count,
        "log_unique_words_count": log_unique_words_count,
        "average_word_length": average_word_length,
        "max_length_word": max_length_word,
        "min_length_word": min_length_word,
        "standard_deviation_words": standard_deviation_words,
        "chars_count": chars_count,
        "hmpz_count": hmpz_count,
        "paragraphs_count": paragraphs_count,
        "is_first_paragraph_less_than_or_equal_to_10": is_first_paragraph_less_than_or_equal_to_10,
        "average_length_paragraph": average_length_paragraph,
        "max_length_paragraph": max_length_paragraph,
        "min_length_paragraph": min_length_paragraph,
        "has_parentheses": has_parentheses,
        "has_colon": has_colon,
        "has_question_mark": has_question_mark,
        "sentences_count": sentences_count,
        "average_length_sentence": average_length_sentence,
        "max_length_sentence": max_length_sentence,
        "min_length_sentence": min_length_sentence,
        "standard_deviation_sentence": standard_deviation_sentence
    }
    
    return features



def extract_syntactic_features(essay):
  
  #spell checking (I need to find a way to do spell checking based on the context not just the words)
  spell = SpellChecker(language='ar')   # use the Arabic Dictionary
  original_words = word_tokenize(re.sub(r'[^\w\s]', '', essay)) 
  corrected_words = [spell.correction(word) for word in original_words]
  # Count the number of misspelled words (not 100% accurate)
  misspelled_count = sum(1 for orig, corrected in zip(original_words, corrected_words) if orig != corrected)
  


  #counting inna words and kaana words
  inna_words = ["أن", "إن", "كأن", "لكن", "ليت", "لعل"]
  kana_words =["كان", "أضحى", "مازال", "لیس", "ماظل", "أمسى", "مافتئ", "بات", "صار", "ظل", "ماانفك", "مابرح", "مادام", "أصبح"]
  inna_count= sum(1 for word in corrected_words if word in inna_words)
  kana_count = sum(1 for word in corrected_words if word in kana_words)
  
  # Use the FarasaPOSTagger to get the POS
  tagger = FarasaPOSTagger(interactive=False)
  pos_tagged = (tagger.tag(essay)).replace("S/S", "").replace("E/E", "") #Remove start and end symbols for a sentence
#   print(pos_tagged)
  
#   print("--------------------------------------")
  
  #Extract only the tags
  pattern = r'(?<=/)[^\s/]+'
  tags = re.findall(pattern, pos_tagged)
#   print(tags)
  
#   print("--------------------------------------")

  # count the extracted tags
  noun_count=0
  verb_count=0
  adj_count=0
  punc_count=0
  pron_count=0
  prep_count=0
  conj_count=0
  adv_count=0
  num_count=0

  for tag in tags:
    if tag.startswith(('NOUN', 'DET')):
      noun_count +=1 
    elif tag.startswith('V'):
      verb_count+=1
    elif tag.startswith("ADJ"):
      adj_count+=1
    elif tag.startswith("PUNC"):
      punc_count+=1
    elif tag.startswith("PRON"):
      pron_count+=1
    elif tag.startswith("PREP"):
      prep_count+=1
    elif tag.startswith("ADV"): #ظرف المكان 
      adv_count+=1
    elif tag.startswith("CONJ"):
      conj_count+=1
    elif tag.startswith("NUM"):
      num_count+=1
  
  extracted_syntactic_features=[noun_count, verb_count,adj_count,punc_count,pron_count,prep_count,adv_count, conj_count,num_count, misspelled_count,inna_count,kana_count]
  
  features = {
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adj_count": adj_count,
        "punc_count": punc_count,
        "pron_count": pron_count,
        "prep_count": prep_count,
        "adv_count": adv_count,
        "conj_count": conj_count,
        "num_count": num_count,
        "misspelled_count": misspelled_count,
        "inna_count": inna_count,
        "kana_count": kana_count
    }
  return features

def extract_lexical_features(essay):
    
    #Count of stop words and words without stop words
    stop_words = set(stopwords.words('arabic'))
    wordsList = word_tokenize(re.sub(r'[^\w\s]', '', essay))
    stop_words_count =  sum(1 for word in wordsList if word in stop_words)
    words_count_without_stopwords = sum(1 for word in wordsList if word not in stop_words)

    #Existence of introducing and concluding words
    paragraphs = essay.split('\n\n')  
    intro_keywords = ['نبدأ','بداية', 'نتحدث', 'نتكلم', 'نستعرض', 'الموضوع', 'في البداية', 'أولاً', 'أود أن أبدأ ب', 'أقدم', 'أعرض']
    conclusion_keywords = ['أختم','أرى', 'أخيراً', 'أرجو', 'وجهة نظر', 'أقترح', 'أتمنى', 'في الختام', 'ختاماً', 'أختاماً', 'خلاصة', 'باختصار']
    first_paragraph_has_intro_words = int(any(re.search(r'\b{}\b'.format(keyword), paragraphs[0]) for keyword in intro_keywords)  )
    last_paragraph_has_conclusion_words = int(any(re.search(r'\b{}\b'.format(keyword), paragraphs[-1]) for keyword in conclusion_keywords) )



    extracted_lexical_features = [stop_words_count, words_count_without_stopwords,first_paragraph_has_intro_words,last_paragraph_has_conclusion_words]
    
    features = {
        "stop_words_count": stop_words_count,
        "words_count_without_stopwords": words_count_without_stopwords,
        "first_paragraph_has_intro_words": first_paragraph_has_intro_words,
        "last_paragraph_has_conclusion_words": last_paragraph_has_conclusion_words
    }
    return features

def extract_all_features(essay):
    """
    Extracts and combines surface, syntactic, and lexical features from a single essay.
    Returns a dictionary of all features.
    """
    features = {}
    features.update(extract_surface_features(essay))
    features.update(extract_syntactic_features(essay))
    features.update(extract_lexical_features(essay))
    features.update(lexical_density(essay))
    return features

def compute_features(essay):
    # extract_all_features returns a dictionary of features
    return extract_all_features(essay)