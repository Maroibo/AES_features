from nltk.corpus import stopwords
from camel_tools_init import _sentiment_analyzer, _bert_tokenizer, _bert_model
from essay_proccessing import split_into_sentences
import torch
from transformers import pipeline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Arabic stopwords
ARABIC_STOPWORDS = set(stopwords.words('arabic'))


def calculate_sentiment_scores(essay):
    """
    Calculates sentiment scores and proportions for Arabic text.
    Returns default values if sentiment analyzer is not available.
    """    

    sentiment_analyzer = pipeline('sentiment-analysis', model='CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment')


    # Normalize and tokenize the text
    sentences = split_into_sentences(essay)
    total_sentences = len(sentences)
    # Initialize counters
    # positive_count = 0
    # negative_count = 0
    # neutral_count = 0

    positive_confidence = 0
    negative_confidence = 0
    neutral_confidence = 0
    
    # Calculate sentiment for each sentence in batches
    batch_size = 4  # Process 4 sentences at a time
    positive_scores = []
    negative_scores = []
    neutral_scores = []  # Added neutral scores list
    
    for i in range(0, total_sentences, batch_size):
        batch = sentences[i:i + batch_size]
        # Get sentiment predictions for the batch
        sentiments = sentiment_analyzer(batch)
        
        # Process each sentiment prediction
        for sentiment in sentiments:
            if sentiment['label'] == 'positive':
                positive_scores.append(1)
                negative_scores.append(0)
                neutral_scores.append(0)
                # positive_count += 1
                positive_confidence += sentiment['score']
            elif sentiment['label'] == 'negative':
                positive_scores.append(0)
                negative_scores.append(1)
                neutral_scores.append(0)
                negative_confidence += sentiment['score']
                # negative_count += 1
            else:  # neutral
                positive_scores.append(0)
                negative_scores.append(0)
                neutral_scores.append(1)
                neutral_confidence += sentiment['score']
                # neutral_count += 1   
    # Calculate overall scores (proportion of each sentiment type)
    positive_sentence_prop = sum(positive_scores) / total_sentences
    neutral_sentence_prop = sum(negative_scores) / total_sentences
    negative_sentence_prop = sum(neutral_scores) / total_sentences
    
    # Calculate proportions (should sum to 1.0)
    # positive_sentence_prop = positive_count / total_sentences
    # neutral_sentence_prop = neutral_count / total_sentences
    # negative_sentence_prop = negative_count / total_sentences

    overall_positivity = positive_confidence / total_sentences
    overall_negativity = negative_confidence / total_sentences
    overall_neutrality = neutral_confidence / total_sentences
    
    return {
        "overall_positivity": overall_positivity,
        "overall_negativity": overall_negativity,
        "overall_neutrality": overall_neutrality,
        "positive_sentence_prop": positive_sentence_prop,
        "neutral_sentence_prop": neutral_sentence_prop,
        "negative_sentence_prop": negative_sentence_prop
    }
    
def calculate_prompt_adherence_features(essay, prompt, model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix"):
    """
    Calculates prompt adherence features using sentence embeddings with GPU acceleration.
    """
    
    # Process texts in batches for better GPU utilization
    def get_embedding(texts, batch_size=8):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Tokenize batch
            batch_inputs = _bert_tokenizer(batch, return_tensors="pt", truncation=True, 
                                  max_length=512, padding=True)
            # Move to GPU
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            with torch.no_grad():
                outputs = _bert_model(**batch_inputs)
            # Get embeddings for batch
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu())
        return embeddings
    
    # Get prompt embedding
    prompt_embedding = get_embedding([prompt])
    

    sentences = split_into_sentences(essay)
    sentence_embeddings = get_embedding(sentences)
    
    # Move prompt embedding to CPU for calculations
    prompt_embedding = prompt_embedding[0].cpu()
    print(len(prompt_embedding))
    
    # Calculate dot scores using vectorized operations
    dot_scores = torch.stack([torch.dot(emb, prompt_embedding) for emb in sentence_embeddings])
    
    # Calculate features
    features = {
        "max_sentence_dot_score": dot_scores.max().item() ,
        "mean_sentence_dot_score": dot_scores.mean().item() ,
        "min_sentence_dot_score": dot_scores.min().item(),
        "dot_score": torch.dot(get_embedding([essay])[0].cpu(), prompt_embedding).item()
    }
    
    return features



