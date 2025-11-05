# Feature Engineering is not Dead: A Step Towards State of the Art for Arabic Automated Essay Scoring

This repository contains the codebase for the paper **"Feature Engineering is not Dead: A Step Towards State of the Art for Arabic Automated Essay Scoring"** by Marwan Sayed, Sohaila Eltanbouly, May Bashendy, and Tamer Elsayed.

## Paper Abstract

Automated Essay Scoring (AES) has shown significant advancements in educational assessment. However, under-resourced languages like Arabic have received limited attention. To bridge this gap and enable robust Arabic AES, this paper introduces the first publicly-available comprehensive set of engineered features tailored for Arabic AES, covering surface-level, readability, lexical, syntactic, and semantic features. Experiments are conducted on a dataset of 620 Arabic essays, each annotated with both holistic and trait-specific scores. Our findings demonstrate that the proposed feature set is effective across different models and competitive with recent NLP advances, including LLMs, establishing the state-of-the-art performance and providing strong baselines for future Arabic AES research. Moreover, the resulting feature set offers a reusable and foundational resource, contributing towards the development of more effective Arabic AES systems.

## Overview

This repository provides a comprehensive feature extraction pipeline for Arabic Automated Essay Scoring (AES). The pipeline extracts over 800 engineered features from Arabic essays, covering:

- **Surface-level features**: Word counts, character counts, sentence statistics, punctuation usage
- **POS (Part-of-Speech) features**: Noun, verb, adjective frequencies, POS bigrams
- **Readability features**: Flesch Reading Ease, SMOG Index, ARI, and other readability metrics
- **Semantic features**: Sentiment analysis, semantic similarities, prompt adherence
- **Syntactic features**: Grammar features, nominal/verbal sentences, conjunctions, dialect usage
- **Lexical features**: Vocabulary diversity, lemma features, word frequency analysis
- **Clause features**: Clause structure analysis (requires separate extraction step)

The feature set has been validated on a dataset of 620 Arabic essays and demonstrates competitive performance with state-of-the-art methods, including Large Language Models (LLMs).

## File Structure

```
arab_nlp_features/
├── extractor_script.py              # Main script for extracting features from essays
├── clause_extractor.py               # Adds clause features to essay feature set
├── camel_tools_init.py               # Initializes CAMeL Tools models (disambiguator, analyzer, etc.)
├── essay_proccessing.py              # Sentence and paragraph splitting utilities
├── pos_features.py                   # Part-of-speech feature extraction
├── readability_measures.py           # Readability score calculations
├── semantic_features.py              # Semantic and sentiment features
├── surface_level_features.py        # Surface-level text features
├── syntactic_features.py            # Syntactic and grammatical features
├── clause_features.py                # Clause analysis features
├── output_features/                  # Output directory for feature CSV files
│   └── example_full_feature_set.csv # Essay features output
└── requirements.txt                  # Python dependencies
```

## Requirements

### Dependencies

Install all required packages:

```bash
pip install --user -r requirements.txt
```

Then download required NLTK data (first run only):

```bash
python -m nltk.downloader stopwords punkt
```

Or install Python packages directly:

```bash
pip install --user --extra-index-url https://download.pytorch.org/whl/cu121 \
    camel-tools==1.5.6 \
    pandas==2.0.2 \
    numpy==1.24.3 \
    tqdm==4.65.0 \
    nltk==3.8.1 \
    torch==2.5.1+cu121 \
    transformers==4.29.2 \
    scipy==1.10.1 \
    scikit-learn \
    matplotlib seaborn
```

### External Dependencies

The scripts require:
- **CAMeL Tools**: Arabic NLP toolkit (installed via pip)
- **NLTK Arabic stopwords**: Will be downloaded automatically on first run
- **Hugging Face models**: Downloaded automatically:
  - `CAMeL-Lab/bert-base-arabic-camelbert-mix` (for semantic features)
  - `CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment` (for sentiment analysis)

- **CAMeL Parser (for clause features)**:
  - You MUST have CAMeL Parser downloaded and installed to extract clause features.
  - Repository: https://github.com/CAMeL-Lab/camel_parser.git
  - Clone CAMeL Parser and install its dependencies following its README.
  - Default path expected by `clause_features.py` is `/data/home/marwan/camel_parser`.
  - If you clone elsewhere, pass the path when constructing the analyzer, e.g.:
    ```python
    ClauseAnalyzer(camel_parser_root="/path/to/camel_parser")
    ```
  - Make sure its models are downloaded (e.g., CATiB) and GPU is configured if available.



## Input Data Structure

**Input CSV File** should contain the following columns:

| Column Name | Type | Description | Required |
|------------|------|-------------|----------|
| `essay_id` | string/int | Unique identifier for each essay | Yes |
| `prompt_id` | string/int | Identifier linking essay to prompt | Yes |
| `essay` | string | The Arabic essay text | Yes |
| `relevance` | float | Score (0-2) | Optional |
| `organization` | float | Score (0-5) | Optional |
| `vocabulary` | float | Score (0-5) | Optional |
| `style` | float | Score (0-5) | Optional |
| `development` | float | Score (0-5) | Optional |
| `mechanics` | float | Score (0-5) | Optional |
| `grammar` | float | Score (0-5) | Optional |
| `holistic` | float | Score (0-32) | Optional |

**Example CSV structure:**
```csv
essay_id,prompt_id,essay,relevance,organization,vocabulary,style,development,mechanics,grammar,holistic
1,1,"هذا مقال تجريبي...",2.0,4.0,3.5,4.0,3.5,4.0,3.5,28.0
```

**Prompt JSON File** (`all_prompts.json`) should have the following structure:

```json
[
  {
    "prompt_id": "1",
    "prompt_text": "اكتب مقالًا عن...",
    "prompt_type": "explanatory"
  },
  {
    "prompt_id": "2",
    "prompt_text": "هل تتفق أو تختلف...",
    "prompt_type": "persuasive"
  }
]
```

Where `prompt_type` can be:
- `"explanatory"` → encoded as `2`
- `"persuasive"` → encoded as `1`

## Configuration

Update these constants at the top of `extractor_script.py`:

```python
INPUT_FILE_PATH = '/path/to/your/essays.csv'           # Input CSV with essays
OUTPUT_FILE_PATH = './output_features/TAQAE_full_feature_set.csv'  # Output CSV
OUTPUT_DIR = './output_features'                       # Output directory
PROMPTS_JSON_PATH = '/path/to/all_prompts.json'        # Prompts JSON file
```

## Usage

### 1. Extract Features from Essays

Run the main feature extraction script:

```bash
python extractor_script.py
```

This will:
1. Load and initialize all NLP models (may take 1-2 minutes on first run)
2. Load essays from the input CSV
3. Load prompts from the JSON file
4. Extract features for each essay
5. Save results to `./output_features/TAQAE_full_feature_set.csv`

**Output**: CSV file with ~812 columns (essay_id, prompt_id, essay text, scores, and all features)

**Processing Time**: Approximately 2-5 seconds per essay (depends on GPU availability)

### 2. Add Clause Features 

Clause features require additional processing and are extracted separately:

```bash
python clause_extractor.py
```

This reads `./output_features/TAQAE_full_feature_set.csv` and adds the following clause features:
- `mean_clause`: Average number of tokens per clause
- `clause_per_s`: Clauses per sentence ratio
- `sent_ave_depth`: Average dependency tree depth per sentence
- `ave_leaf_depth`: Average depth of leaf nodes in dependency trees
- `max_clause_in_s`: Maximum number of clauses in a single sentence

**Note**: Clause extraction requires GPU and may take longer (processes one text at a time).

## Feature Categories

### Surface-Level Features

### POS Features

### Readability Features

### Semantic Features

### Syntactic Features

### Lexical Features

### Paragraph-Specific Features

### Clause Features

## Output Format

The output CSV contains:
- **Metadata columns**: `essay_id`, `prompt_id`, `essay`, `prompt`, `prompt_type`, `prompt_type_encoded`
- **Score columns**: `relevance`, `organization`, `vocabulary`, `style`, `development`, `mechanics`, `grammar`, `holistic` (if provided)
- **Feature columns**: All ~800+ extracted features

## Notes

- **Paragraph Splitting**: Essays are automatically split into intro/body/conclusion based on newline characters:
  - 1 paragraph → all goes to intro
  - 2 paragraphs → first = intro, second = conclusion
  - 3+ paragraphs → first = intro, last = conclusion, middle = body

- **Feature Extraction Order**: 
  1. First run `extractor_script.py` to extract all features
  2. Then run `clause_extractor.py` to add clause features

- **Feature Set Design**: The feature set is designed to be comprehensive yet efficient, covering multiple linguistic dimensions while maintaining computational feasibility.

## Acknowledgments

- **CAMeL Tools**: https://github.com/CAMeL-Lab/camel_tools


## Citation

```bibtex
@inproceedings{sayed-etal-2025-feature,

    title = "Feature Engineering is not Dead: A Step Towards State of the Art for {A}rabic Automated Essay Scoring",
    author = "Sayed, Marwan  and
      Eltanbouly, Sohaila  and
      Bashendy, May  and
      Elsayed, Tamer",
    editor = "Darwish, Kareem  and
      Ali, Ahmed  and
      Abu Farha, Ibrahim  and
      Touileb, Samia  and
      Zitouni, Imed  and
      Abdelali, Ahmed  and
      Al-Ghamdi, Sharefah  and
      Alkhereyf, Sakhar  and
      Zaghouani, Wajdi  and
      Khalifa, Salam  and
      AlKhamissi, Badr  and
      Almatham, Rawan  and
      Hamed, Injy  and
      Alyafeai, Zaid  and
      Alowisheq, Areeb  and
      Inoue, Go  and
      Mrini, Khalil  and
      Alshammari, Waad",
    booktitle = "Proceedings of The Third Arabic Natural Language Processing Conference",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.arabicnlp-main.19/",
    pages = "231--245",
    ISBN = "979-8-89176-352-4",
    abstract = "Automated Essay Scoring (AES) has shown significant advancements in educational assessment. However, under-resourced languages like Arabic have received limited attention. To bridge this gap and enable robust Arabic AES, this paper introduces the $\\textit{first publicly-available}$ comprehensive set of engineered features tailored for Arabic AES, covering surface-level, readability, lexical, syntactic, and semantic features. Experiments are conducted on a dataset of 620 Arabic essays, each annotated with both holistic and trait-specific scores. Our findings demonstrate that the proposed feature set is effective across different models and competitive with recent NLP advances including LLMs, establishing the state-of-the-art performance and providing strong baselines for future Arabic AES research. Moroever, the resulting feature set offers a reusable and foundational resource, contributing towards the development of more effective Arabic AES systems."
}
```

## Authors

- **Marwan Sayed** (me2104862@qu.edu.qa)
- **Sohaila Eltanbouly** (se1403101@qu.edu.qa)
- **May Bashendy** (ma1403845@qu.edu.qa)
- **Tamer Elsayed** (telsayed@qu.edu.qa)

Computer Science and Engineering Department, Qatar University, Doha, Qatar



## Contact

For questions or issues, please contact the authors or open an issue on this repository.
