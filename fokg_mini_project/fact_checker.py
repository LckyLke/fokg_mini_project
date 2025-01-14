import os
import torch
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, RDF
from typing import List, Optional
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FactChecker:
    def __init__(
        self,
        path_nt: Optional[str] = None,
        path_csv: Optional[str] = None,
        path_model: Optional[str] = None,
        output_dir: str = "data/models",
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        batch_size: int = 1024,
    ):
        """
        - path_nt: Path to an .nt file. If provided, we'll parse it, filter out certain predicates (like rdfs:label),
                   convert to CSV, and then train a model unless a model is already provided (path_model).
        - path_csv: Path to a CSV file with the format (subject, predicate, object), tab-separated. 
                    If provided (and path_nt is None), we train using that CSV, 
                    unless a model checkpoint is also provided.
        - path_model: Path to a .pt or .pkl or .ptk PyTorch checkpoint for a pretrained model. If provided, 
                      we load the model directly and skip training. 
        - output_dir: Directory where we save the trained model if we train one.
        - embedding_dim, learning_rate, num_epochs, batch_size: Training parameters.
        """
        self.model = None
        self.triples_factory = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.output_dir = output_dir

        # If a model checkpoint is provided, load it
        if path_model is not None and os.path.exists(path_model):
            self.model = torch.load(path_model, map_location=torch.device('cpu'))
            logger.info(f"Loaded model from {path_model}")

            default_ref_csv = os.path.join(self.output_dir, "reference.csv")
            if os.path.exists(default_ref_csv):
                self.triples_factory = TriplesFactory.from_path(default_ref_csv)
                logger.info(f"Loaded reference TriplesFactory from {default_ref_csv}")
            else:
                logger.warning("No reference.csv found alongside model. ID lookups may fail.")

        else:
            if path_nt is not None:
                # Convert .nt to .csv
                path_csv = self.convert_nt_to_csv(path_nt)
                logger.info(f"Converted {path_nt} to CSV at {path_csv}")

            if path_csv is not None and os.path.exists(path_csv):
                self.triples_factory = TriplesFactory.from_path(path_csv)
                logger.info(f"Loaded TriplesFactory from {path_csv}")
                # Train a new model if no pretrained model was given
                self.model = self.train_model(
                    triples_factory=self.triples_factory,
                    embedding_dim=embedding_dim,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    output_dir=self.output_dir
                )
                logger.info("Training complete. Model is ready.")
            else:
                raise ValueError(
                    "No valid CSV or NT provided for training, and no model checkpoint provided."
                )

        if self.triples_factory is not None:
            self.entity_to_id = self.triples_factory.entity_to_id
            self.relation_to_id = self.triples_factory.relation_to_id

    def convert_nt_to_csv(self, nt_path: str) -> str:
        """
        Parse the .nt file, filter out rdfs:label (if needed), and create a .csv file.
        Returns the path to the newly created CSV.
        """
        g = Graph()
        g.parse(nt_path, format="nt")

        # Convert RDF triples to a list of (subject, predicate, object)
        triples = [(str(s), str(p), str(o)) for s, p, o in g]

        df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
        # Example filter to remove rdfs:label
        df = df[df["predicate"] != "http://www.w3.org/2000/01/rdf-schema#label"]

        csv_path = os.path.join("data", "reference.csv")
        os.makedirs("data", exist_ok=True)
        df.to_csv(csv_path, sep="\t", index=False, header=False)
        return csv_path

    def train_model(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        output_dir: str
    ):
        """
        Train a PyKEEN model (ComplEx by default) on the given TriplesFactory
        and return the trained model.
        """
        # Splitting the data
        training, testing, validation = triples_factory.split([0.8, 0.1, 0.1])
        logger.info("Splitted data into training, testing, and validation sets.")

        # Run pipeline with default or user-specified model/hparams
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model='ComplEx',
            model_kwargs=dict(embedding_dim=embedding_dim),
            optimizer_kwargs=dict(lr=learning_rate),
            training_kwargs=dict(num_epochs=num_epochs, batch_size=batch_size),
			evaluation_kwargs=dict(batch_size=512, slice_size=256),
        )

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        result.save_to_directory(output_dir)
        model = result.model

        # Optionally print metrics
        metrics = result.metric_results
        logger.info(f"Hits@1: {metrics.get_metric('hits@1')}")
        logger.info(f"Hits@3: {metrics.get_metric('hits@3')}")
        logger.info(f"Hits@5: {metrics.get_metric('hits@5')}")
        logger.info(f"Hits@10: {metrics.get_metric('hits@10')}")
        logger.info(f"Mean Reciprocal Rank: {metrics.get_metric('mean_reciprocal_rank')}")

        # Plot losses if you want:
        plot = result.plot_losses()
        plot.figure.savefig(os.path.join(output_dir, "losses.png"))

        # Also save the reference CSV used for ID lookups
        reference_csv_path = os.path.join(output_dir, "reference.csv")
        # We assume that "triples_factory.path" is the original CSV:
        if triples_factory.path is not None:
            # Copy it or just record it
            df = pd.read_csv(triples_factory.path, sep="\t", header=None)
            df.to_csv(reference_csv_path, sep="\t", index=False, header=False)

        return model

    def parse_rdf_file(self, file_path: str):
        """
        Parses an N-Triples file to extract facts (statements).
        If the triple has a literal with 'hasTruthValue', store it as float in 'truth_value'.
        If no truth value is present, store None.
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
                statements[s]['truth_value'] = float(o)

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

    def predict(self, h: List[str], r: List[str], t: List[str]) -> List[float]:
        """
        Predict scores for a list of triples (heads, relations, tails).
        Applies a sigmoid transform to map raw scores to [0, 1].
        """
        if self.model is None or self.triples_factory is None:
            raise ValueError("Model or TriplesFactory not initialized.")

        # Convert entity and relation labels to IDs
        h_ids = [self.entity_to_id[head] for head in h]
        r_ids = [self.relation_to_id[rel] for rel in r]
        t_ids = [self.entity_to_id[tail] for tail in t]

        # Create the batch of H, R, T IDs
        hrt_batch = []
        for hi, ri, ti in zip(h_ids, r_ids, t_ids):
            hrt_batch.append([hi, ri, ti])
        hrt_batch = torch.LongTensor(hrt_batch)

        # Get raw scores from the model (shape: (batch_size,))
        raw_scores = self.model.score_hrt(hrt_batch)

        # Apply a sigmoid transform to map the scores to (0, 1)
        normalized_scores = torch.sigmoid(raw_scores)

        # Convert scores to a 1D list of floats
        return normalized_scores.squeeze(dim=-1).tolist()

    def evaluate_model(
        self,
        facts: List[dict],
        output_file: str = "result.ttl",
        test: bool = True,
        threshold: float = 0.8 # only important if test=True
    ):
        """
        If test=True:
          - Compute metrics only for facts that have a non-None truth_value.
          - Write predictions for all facts to the TTL file.
        If test=False:
          - Skip metrics (we have unlabeled data).
          - Still write predictions for all facts to the TTL file.
        """
        y_true = []
        y_scores = []
        output_lines = []

        for fact in facts:
            stmt_id = fact['id']
            head = fact['subject']
            relation = fact['predicate']
            tail = fact['object']
            truth_value = fact['truth_value']  # might be None for unlabeled

            try:
                # Predict the score for this triple
                score = self.predict([head], [relation], [tail])[0]

                logger.info(
                    f"Scored fact: [head={head}, relation={relation}, tail={tail}] -> {score}"
                )
                if truth_value is not None:
                    logger.info(f"Truth value (labeled): {truth_value}\n")
                else:
                    logger.info("Truth value: None (unlabeled)\n")

                # Collect for output
                ttl_line = (
                    f"<{stmt_id}> "
                    f"<http://swc2017.aksw.org/hasTruthValue> "
                    f"\"{score}\"^^<http://www.w3.org/2001/XMLSchema#double> ."
                )
                output_lines.append(ttl_line)

                # If we are in test mode AND the fact is labeled, store for metrics
                if test and truth_value is not None:
                    y_true.append(truth_value)
                    y_scores.append(score)

            except Exception as e:
                logger.error(f"Error scoring fact: {fact} - {e}")

        # Write the TTL lines to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")

        # If in test mode, compute metrics on the labeled subset
        if test:
            if len(y_true) == 0:
                logger.warning("No labeled facts found for evaluation.")
                return [], []

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)

            # Scores are in [0,1] due to sigmoid
            y_scores_normalized = y_scores

            # Compute metrics
            y_pred = (y_scores_normalized > threshold).astype(float)
            roc_auc = roc_auc_score(y_true, y_scores_normalized)
            roc_auc_pred = roc_auc_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            logger.info(f"ROC AUC Score: {roc_auc}")
            logger.info(f"ROC AUC Score (threshold={threshold}): {roc_auc_pred}")
            logger.info(f"Accuracy: {accuracy}\n")
            return y_true, y_scores_normalized
        else:
            logger.info("Skipping evaluation (test=False).")
            return [], []
