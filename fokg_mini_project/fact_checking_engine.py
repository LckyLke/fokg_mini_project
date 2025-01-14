import torch
import numpy as np
from typing import List
from pykeen.triples import TriplesFactory
from sklearn.metrics import roc_auc_score, accuracy_score
from rdflib import Graph, Namespace, RDF

class FactChecker:
    def __init__(self, path_csv=None, path_pkl=None):
        self.path_pkl = path_pkl
        self.path_csv = path_csv

        # Load your model and triples factory
        self.model = torch.load(self.path_pkl)
        self.triples_factory = TriplesFactory.from_path(self.path_csv)
        self.entity_to_id = self.triples_factory.entity_to_id
        self.relation_to_id = self.triples_factory.relation_to_id

    def predict(self, h: List[str], r: List[str], t: List[str]) -> List[float]:
        """
        Predict scores for a list of triples (heads, relations, tails).
        Applies a sigmoid transform to map raw scores to [0, 1].
        """
        # Convert entity and relation labels to IDs
        h_ids = [self.entity_to_id[head] for head in h]
        r_ids = [self.relation_to_id[rel] for rel in r]
        t_ids = [self.entity_to_id[tail] for tail in t]

        # Create the batch of H, R, T IDs
        hrt_batch = []
        for hi, ri, ti in zip(h_ids, r_ids, t_ids):
            hrt_batch.append([hi, ri, ti])
        hrt_batch = torch.LongTensor(hrt_batch)

        # Get raw scores from the model (shape: (batch_size, 1))
        raw_scores = self.model.score_hrt(hrt_batch)

        # Apply a sigmoid transform to map the scores to (0, 1)
        normalized_scores = torch.sigmoid(raw_scores)

        # Convert scores to a 1D list of floats
        return normalized_scores.squeeze(dim=-1).tolist()


def parse_rdf_file(file_path: str):
    """
    Parses an N-Triples file to extract facts (statements).  
    Allows for both labeled (with hasTruthValue) and unlabeled data.  
    If a statement has no truth value, we store 'truth_value' as None.
    """
    g = Graph()
    g.parse(file_path, format='nt')  # Assumes the file is in N-Triples format

    SWC = Namespace("http://swc2017.aksw.org/")
    RDF_SYNTAX_NS = RDF
    statements = {}

    for s, p, o in g:
        if s not in statements:
            # Initialize the statement dictionary for s
            statements[s] = {
                'subject': None,
                'predicate': None,
                'object': None,
                'truth_value': None,
                'is_statement': False
            }
        
        if p == RDF_SYNTAX_NS.type and o == RDF_SYNTAX_NS.Statement:
            statements[s]['is_statement'] = True
        elif p == RDF_SYNTAX_NS.subject:
            statements[s]['subject'] = o
        elif p == RDF_SYNTAX_NS.predicate:
            statements[s]['predicate'] = o
        elif p == RDF_SYNTAX_NS.object:
            statements[s]['object'] = o
        elif p == SWC.hasTruthValue:
            # Convert to float
            truth_value = float(o)
            statements[s]['truth_value'] = truth_value

    # Gather all valid RDF statements (type=Statement and subject/predicate/object not None)
    facts = []
    for stmt_id, stmt in statements.items():
        if (
            stmt['is_statement'] and
            None not in [stmt['subject'], stmt['predicate'], stmt['object']]
        ):
            facts.append({
                'id': str(stmt_id),
                'subject': str(stmt['subject']),
                'predicate': str(stmt['predicate']),
                'object': str(stmt['object']),
                'truth_value': stmt['truth_value'],  # could be None or float
            })

    return facts


def evaluate_model(
    fact_checker: FactChecker, 
    facts: List[dict], 
    output_file: str = "result.ttl",
    test: bool = True
):
    """
    If test=True:
      - Compute metrics only for facts that have a non-None truth_value.
      - Write predictions for all facts to the TTL file.
    If test=False:
      - Skip metrics (we have unlabeled data).
      - Still write predictions for all facts to the TTL file (if desired).
    """
    # We'll store everything in these lists to do optional evaluation
    y_true = []
    y_scores = []

    # Prepare lines for output
    output_lines = []

    for fact in facts:
        stmt_id = fact['id']
        head = fact['subject']
        relation = fact['predicate']
        tail = fact['object']
        truth_value = fact['truth_value']  # might be None for unlabeled

        try:
            # Predict the score for this triple
            score = fact_checker.predict([head], [relation], [tail])[0]

            # Print out the details
            print(f"Scored fact: [head={head}, relation={relation}, tail={tail}] -> {score}")
            if truth_value is not None:
                print(f"Truth value (labeled): {truth_value}\n")
            else:
                print("Truth value: None (unlabeled)\n")

            # Collect for output
            ttl_line = f"<{stmt_id}> <http://swc2017.aksw.org/hasTruthValue> \"{score}\"^^<http://www.w3.org/2001/XMLSchema#double> ."
            output_lines.append(ttl_line)

            # If we are in test mode AND the fact is labeled, store for metrics
            if test and truth_value is not None:
                y_true.append(truth_value)
                y_scores.append(score)

        except Exception as e:
            print(f"Error scoring fact: {fact} - {e}")

    # Write the TTL lines to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    # If in test mode, compute metrics on the labeled subset
    if test:
        if len(y_true) == 0:
            print("Warning: No labeled facts found for evaluation.")
            return [], []

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Scores are already in [0,1] due to sigmoid
        y_scores_normalized = y_scores

        # Compute metrics
        y_pred = (y_scores_normalized > 0.8).astype(float)
        roc_auc = roc_auc_score(y_true, y_scores_normalized)
        roc_auc_pred = roc_auc_score(y_true, y_pred)
        # Example threshold
        accuracy = accuracy_score(y_true, y_pred)

        print(f"ROC AUC Score: {roc_auc}")
        print(f"ROC AUC Score (0.8 threshold): {roc_auc_pred}")
        print(f"Accuracy: {accuracy}\n")
        return y_true, y_scores_normalized
    else:
        print("Skipping evaluation (test=False).")
        return [], []

# Example usage:
if __name__ == "__main__":
    # Parse facts from the N-Triples file
    facts = parse_rdf_file("data/fokg-sw-train-2024.nt")

    # Initialize the fact checker
    fact_checker = FactChecker(
        path_csv="data/reference.csv",
        path_pkl="/home/lukef/.data/pykeen/checkpoints/best-model-weights-810410eb-8ee0-418d-a643-97b8a6939ee2.pt"
    )

    # Evaluate the model
    #   - test=True  => compute metrics on labeled data
    #   - test=False => skip metrics (for unlabeled or unknown data)
    evaluate_model(fact_checker, facts, output_file="result.ttl", test=True)
