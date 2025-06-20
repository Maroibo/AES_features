"""
A module for calculating clause-based syntactic features from Arabic text
using the CamelParser dependency parser.
"""
import logging

# Aggressively disable all logging by removing handlers from the root logger.
# This prevents any third-party library from creating log files.
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.setLevel(logging.CRITICAL)

from pathlib import Path
from camel_tools.utils.charmap import CharMapper
from pandas import read_csv
from collections import defaultdict
import re
from essay_proccessing import split_into_sentences
import sys

# Import CamelParser modules
# Ensure the camel_parser path is available
sys.path.append('/data/home/marwan/camel_parser')
from src.classes import TextParams, Token
from src.data_preparation import parse_text, get_tagset
from src.utils.model_downloader import get_model_name


class ClauseAnalyzer:
    """
    Analyzes clause features using CamelParser's dependency parsing.
    This class encapsulates the parsing and feature calculation logic,
    making it suitable for use as a modular component.
    """
    
    def __init__(self, camel_parser_root="/data/home/marwan/camel_parser"):
        """
        Initializes the CamelParser clause analyzer.
        
        Args:
            camel_parser_root (str): Path to the root of the camel_parser directory.
        """
        self.root_dir = Path(camel_parser_root)
        self.model_path = self.root_dir / "models"
        self.parse_model = "catib"
        
        self.arclean = CharMapper.builtin_mapper("arclean")
        
        clitic_feats_df = read_csv(self.root_dir / 'data/clitic_feats.csv')
        self.clitic_feats_df = clitic_feats_df.astype(str).astype(object)
        
        self.model_name = get_model_name(self.parse_model, model_path=self.model_path)
        self.tagset = get_tagset(self.parse_model)
        
        self.disambiguator_type = "mle"
        self.morphology_db_type = "r13"
        
    def parse_sentences(self, sentences):
        """
        Parses a list of sentences using the configured CamelParser.
        """
        try:
            file_type_params = TextParams(
                sentences, 
                self.model_path / self.model_name, 
                self.arclean, 
                self.disambiguator_type, 
                self.clitic_feats_df, 
                self.tagset, 
                self.morphology_db_type
            )
            return parse_text("text", file_type_params)
        except Exception:
            # Silently fail on parsing error
            return []
    
    def _extract_dependency_tree(self, sentence_tuples):
        """
        Internal method to extract a dependency tree from parsed token tuples.
        """
        tokens = []
        dependency_tree = defaultdict(list)
        
        for token_tuple in sentence_tuples:
            token = Token(
                ID=int(token_tuple[0]),
                FORM=str(token_tuple[1]),
                LEMMA=str(token_tuple[2]),
                UPOS=str(token_tuple[3]),
                XPOS=str(token_tuple[4]),
                FEATS=str(token_tuple[5]),
                HEAD=int(token_tuple[6]),
                DEPREL=str(token_tuple[7]),
                DEPS=str(token_tuple[8]),
                MISC=str(token_tuple[9])
            )
            tokens.append(token)
            
            if token.HEAD != 0:
                dependency_tree[str(token.HEAD)].append(str(token.ID))
        
        return {
            'tokens': tokens,
            'dependency_tree': dict(dependency_tree)
        }

    def _identify_clauses_from_dependencies(self, tree_info):
        """
        Internal method to identify all clauses from a dependency tree.
        A clause is defined as a verb and all its dependents.
        """
        tokens = tree_info['tokens']
        dependency_tree = tree_info['dependency_tree']
        
        clauses = []
        token_map = {str(t.ID): t for t in tokens}

        def get_all_dependents(start_token_id):
            """Iteratively get all dependents of a token."""
            dependents_ids = []
            q = [str(start_token_id)]
            visited = set()
            while q:
                current_id = q.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                if current_id in dependency_tree:
                    children = dependency_tree[current_id]
                    dependents_ids.extend(children)
                    q.extend(children)
            return dependents_ids
        
        root_verbs = [t for t in tokens if t.HEAD == 0]
        
        if not root_verbs:
            return []
        
        processed_verbs = set()

        # Identify root clauses
        for verb in root_verbs:
            verb_id_str = str(verb.ID)
            dependents = get_all_dependents(verb_id_str)
            clauses.append([verb_id_str] + dependents)
            processed_verbs.add(verb_id_str)
        
        # Identify non-root clauses headed by verbs linked to markers
        for verb in tokens:
            verb_id_str = str(verb.ID)
            if verb.UPOS == 'VRB' and verb_id_str not in processed_verbs:
                head_token = token_map.get(str(verb.HEAD))
                if head_token and head_token.UPOS == 'PRT' and head_token.FORM in ['و+', 'أن', 'ف+']:
                    dependents = get_all_dependents(verb_id_str)
                    clauses.append([verb_id_str] + dependents)
                    processed_verbs.add(verb_id_str)
        
        return clauses

    def _calculate_tree_depth(self, tree_info, node_id):
        """Internal method to calculate the depth of a node in the tree."""
        tokens = tree_info['tokens']
        token_map = {str(t.ID): t for t in tokens}
        
        def get_depth(current_id_str):
            if current_id_str == '0':
                return 0
            
            token = token_map.get(current_id_str)
            if not token or str(token.HEAD) not in token_map and str(token.HEAD) != '0':
                return 1
            
            return 1 + get_depth(str(token.HEAD))
            
        return get_depth(str(node_id))

    def calculate_features(self, essay_text):
        """
        Calculates clause-based syntactic features for a given essay.

        This is the main public method of the class.

        Args:
            essay_text (str): The input Arabic text.

        Returns:
            dict: A dictionary containing the five syntactic features:
                  'mean_clause', 'clause_per_s', 'sent_ave_depth',
                  'ave_leaf_depth', 'max_clause_in_s'.
        """
        sentences = split_into_sentences(essay_text)
        if not sentences:
            return {'mean_clause': 0, 'clause_per_s': 0, 'sent_ave_depth': 0, 'ave_leaf_depth': 0, 'max_clause_in_s': 0}
        
        clauses_per_sentence = []
        clause_lengths = []
        sentence_depths = []
        leaf_depths = []
        
        for sentence in sentences:
            parsed_tuples = self.parse_sentences([sentence])
            if not parsed_tuples or not parsed_tuples[0]:
                continue

            sentence_tuples = parsed_tuples[0]
            tree_info = self._extract_dependency_tree(sentence_tuples)
            
            clauses = self._identify_clauses_from_dependencies(tree_info)
            clauses_per_sentence.append(len(clauses))
            clause_lengths.extend([len(c) for c in clauses])
            
            if tree_info['tokens']:
                # Calculate max depth for the sentence tree
                max_depth = 0
                for token in tree_info['tokens']:
                    depth = self._calculate_tree_depth(tree_info, token.ID)
                    if depth > max_depth:
                        max_depth = depth
                sentence_depths.append(max_depth)

                # Calculate average leaf depth for the sentence
                leaf_nodes = [t for t in tree_info['tokens'] if str(t.ID) not in tree_info['dependency_tree']]
                if leaf_nodes:
                    current_leaf_depths = [self._calculate_tree_depth(tree_info, leaf.ID) for leaf in leaf_nodes]
                    avg_leaf_depth = sum(current_leaf_depths) / len(current_leaf_depths)
                    leaf_depths.append(avg_leaf_depth)
                else:
                    leaf_depths.append(max_depth) # Fallback for single-node trees

        total_clauses = sum(clauses_per_sentence)
        total_sentences = len(sentences)
        
        features = {
            'mean_clause': sum(clause_lengths) / total_clauses if total_clauses > 0 else 0,
            'clause_per_s': total_clauses / total_sentences if total_sentences > 0 else 0,
            'sent_ave_depth': sum(sentence_depths) / len(sentence_depths) if sentence_depths else 0,
            'ave_leaf_depth': sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0,
            'max_clause_in_s': max(clauses_per_sentence) if clauses_per_sentence else 0,
        }
        
        return features


# if __name__ == "__main__":
#     # Example usage: can be run as a standalone script for testing.
#     test_text = "فاتصل علي أحد أصدقائي وقال لي: إني متواجد أمام بيتك، فنزلت وسلمت عليه قبل أن أسافر، وبعدها جاء أصدقائي وذهبت معهم، كان في طريقي إلى المطار حادث وخشيت أن تفوتني الرحلة، وهل سيشرحونها؟"
    
#     # Instantiate the analyzer
#     analyzer = ClauseAnalyzer()
    
#     # Calculate and print the features
#     clause_features = analyzer.calculate_features(test_text)
#     print("Syntactic Features Calculated:")
#     for feature, value in clause_features.items():
#         print(f"- {feature}: {value:.2f}")