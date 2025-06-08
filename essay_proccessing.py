import re
def split_into_sentences(essay):
    _SENTENCE_SPLIT_REGEX = re.compile(r'[.!?;؟،:]\s+')
    return _SENTENCE_SPLIT_REGEX.split(essay)

def split_into_paragraphs(essay):
    # this method is just a naive implementation
    # this should be reimplemented
    if not essay or len(essay.strip()) == 0:
        return ["", "", ""]
    essay = essay.strip()
    total_length = len(essay)
    target_length = total_length // 3
    paragraphs = []
    start = 0
    for i in range(2):  # First two paragraphs
        end = start + target_length  
        # Try to find a word boundary near the target position
        # Look for space within a reasonable range
        search_range = min(50, target_length // 4)  # Search within 25% of target length 
        best_end = end
        for j in range(max(0, end - search_range), min(total_length, end + search_range)):
            if essay[j] == ' ':
                # Prefer positions closer to the target
                if abs(j - end) < abs(best_end - end):
                    best_end = j    
        paragraphs.append(essay[start:best_end].strip())
        start = best_end + 1 if best_end < total_length else best_end 
    # Add the remaining text as the third paragraph
    paragraphs.append(essay[start:].strip()) 
    return paragraphs