from bert_score import score as bert_score
import re
import string

'''
https://www.semanticscholar.org/paper/BERTScore:-Evaluating-Text-Generation-with-BERT-Zhang-Kishore/295065d942abca0711300b2b4c39829551060578
'''

def normalize_answer(s):
    """Normalize answer by removing articles, punctuation, and normalizing whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_bert_score(predictions, ground_truths, lang="en"):
    """
    Calculate BERTScore for a list of predictions and ground truths.
    Returns precision, recall, and F1 scores.
    """
    normalized_preds = [normalize_answer(pred) for pred in predictions]
    normalized_refs = [normalize_answer(ref) for ref in ground_truths]

    valid_pairs = [(p, r) for p, r in zip(normalized_preds, normalized_refs)
                  if len(p.strip()) > 0 and len(r.strip()) > 0]

    if not valid_pairs:
        return {'bert_precision': 0, 'bert_recall': 0, 'bert_f1': 0}

    valid_preds, valid_refs = zip(*valid_pairs)

    try:
        # Calculate BERTScore
        P, R, F1 = bert_score(valid_preds, valid_refs, lang=lang, verbose=False)

        # Convert from torch tensors to numpy arrays and then to Python scalars
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {'bert_precision': 0, 'bert_recall': 0, 'bert_f1': 0}
