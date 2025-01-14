import os
import torch
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, RDF
from typing import List, Optional
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
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
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        num_epochs: int = 500,
        batch_size: int = 1024 * 8,
        data_dir: str = "data",
    ):
        """
        A fact checker class that can:
          - Load an existing model checkpoint (if `path_model` is provided),
          - OR train a new model from a CSV of triples (if `path_csv` is provided),
          - OR convert an NT file to CSV first (if `path_nt` is provided but no CSV is given),
            and then train a model.
       
        :param path_nt: Path to an .nt file.
        :param path_csv: Path to a CSV file with format (subject, predicate, object), tab-separated.
        :param path_model: Path to a .pt or .pkl checkpoint for a pretrained model.
        :param embedding_dim: Embedding dimension for the KGE model (if training).
        :param learning_rate: Learning rate for training.
        :param num_epochs: Number of epochs for training.
        :param batch_size: Batch size for training.
        :param data_dir: Directory to read/write all data (models + reference.csv).
        """
        self.model = None
        self.triples_factory = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)

        # 1) If a model is provided, load it and skip training
        if path_model is not None and os.path.isfile(path_model):
            self.model = torch.load(path_model, map_location=torch.device('cpu'))
            logger.info(f"Loaded existing model from {path_model}")

            # Check for an existing reference CSV in data_dir
            ref_csv_path = os.path.join(self.data_dir, "reference.csv")
            if os.path.isfile(ref_csv_path):
                # Load the triple factory from the existing reference CSV
                self.triples_factory = TriplesFactory.from_path(ref_csv_path)
                logger.info(f"Loaded reference TriplesFactory from {ref_csv_path}")
            else:
                # If there's no reference.csv but an NT is provided, we can convert it
                if path_nt is not None and os.path.isfile(path_nt):
                    logger.info("No reference.csv found. Converting NT to CSV to build TriplesFactory.")
                    ref_csv_path = self.convert_nt_to_csv(path_nt)
                    self.triples_factory = TriplesFactory.from_path(ref_csv_path)
                else:
                    logger.warning(
                        "No reference.csv found and no .nt provided. "
                        "You won't be able to do label-based predictions."
                    )
            # If we have a triples_factory, set up entity/relation ID dicts
            if self.triples_factory is not None:
                self.entity_to_id = self.triples_factory.entity_to_id
                self.relation_to_id = self.triples_factory.relation_to_id

        else:
            # 2) If we have a CSV provided, use it directly (skip NT)
            if path_csv is not None and os.path.isfile(path_csv):
                logger.info(f"Using provided CSV at {path_csv} for training.")
                self.triples_factory = TriplesFactory.from_path(path_csv)
                logger.info(f"Loaded TriplesFactory from {path_csv}")

            # 3) Otherwise, if we have only an NT file, convert it to CSV and train
            elif path_nt is not None and os.path.isfile(path_nt):
                logger.info(f"No CSV provided; converting NT file ({path_nt}) to CSV.")
                path_csv = self.convert_nt_to_csv(path_nt)
                self.triples_factory = TriplesFactory.from_path(path_csv)
                logger.info(f"Created CSV from NT and loaded TriplesFactory from {path_csv}")

            # 4) If neither CSV nor NT is provided, raise an error
            else:
                raise ValueError(
                    "No valid model checkpoint, CSV, or NT file provided. Cannot proceed."
                )

            # Train the model (since no path_model was provided)
            logger.info("Starting training of a new model...")
            self.model = self.train_model(
                triples_factory=self.triples_factory,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
            logger.info("Training complete. Model is ready.")

            # Once the model is trained, store the entity/relation mappings
            self.entity_to_id = self.triples_factory.entity_to_id
            self.relation_to_id = self.triples_factory.relation_to_id

    def convert_nt_to_csv(self, nt_path: str) -> str:
        """
        Parse the .nt file, filter out rdfs:label (if needed), and create a .csv file.
        Returns the path to the newly created CSV in `data_dir/reference.csv`.
        """
        g = Graph()
        g.parse(nt_path, format="nt")

        # Convert RDF triples to a list of (subject, predicate, object)
        triples = [(str(s), str(p), str(o)) for s, p, o in g]

        df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
        # Example filter to remove rdfs:label
        df = df[df["predicate"] != "http://www.w3.org/2000/01/rdf-schema#label"]

        csv_path = os.path.join(self.data_dir, "reference.csv")
        df.to_csv(csv_path, sep="\t", index=False, header=False)
        return csv_path

    def train_model(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
    ):
        """
        Train a PyKEEN model (ComplEx by default) on the given TriplesFactory
        and return the trained model. Saves both the model and the reference CSV to `data_dir`.
        """
        # Split the data
        training, testing, validation = triples_factory.split([0.8, 0.1, 0.1])
        logger.info("Split data into training (80%), validation (10%), and testing (10%).")

        # Run the pipeline with default ComplEx or user-specified arguments
        result = pipeline(
            training=triples_factory, # training on full data -> split was used for evaluation only -> best performance on full data
            testing=testing,
            validation=validation,
            model='ComplEx',
            model_kwargs=dict(embedding_dim=embedding_dim),
            optimizer_kwargs=dict(lr=learning_rate),
            training_kwargs=dict(num_epochs=num_epochs, batch_size=batch_size),
            evaluation_kwargs=dict(batch_size=512, slice_size=256),
        )

        # Save the trained model to data_dir
        model_file = os.path.join(self.data_dir, "trained_model.pt")
        torch.save(result.model, model_file)
        logger.info(f"Saved trained model to {model_file}")

        # Log some metrics
        metrics = result.metric_results
        logger.info(f"Hits@1: {metrics.get_metric('hits@1')}")
        logger.info(f"Hits@3: {metrics.get_metric('hits@3')}")
        logger.info(f"Hits@5: {metrics.get_metric('hits@5')}")
        logger.info(f"Hits@10: {metrics.get_metric('hits@10')}")
        logger.info(f"Mean Reciprocal Rank: {metrics.get_metric('mean_reciprocal_rank')}")

        # save a loss plot
        plot = result.plot_losses()
        losses_png = os.path.join(self.data_dir, "losses.png")
        plot.figure.savefig(losses_png)
        logger.info(f"Saved loss plot to {losses_png}")

        return result.model

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

        # Gather all valid RDF statements
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
        Applies a sigmoid transform to map raw scores into [0, 1].
        """
        if self.model is None or self.triples_factory is None:
            raise ValueError("Model or TriplesFactory not initialized. Cannot predict.")

        # Convert entity and relation labels to IDs
        try:
            h_ids = [self.entity_to_id[head] for head in h]
            r_ids = [self.relation_to_id[rel] for rel in r]
            t_ids = [self.entity_to_id[tail] for tail in t]
        except KeyError as e:
            raise ValueError(f"Entity or relation not found in the training dictionary: {e}")

        # Create the batch of H, R, T IDs
        hrt_batch = torch.LongTensor(list(zip(h_ids, r_ids, t_ids)))

        # Get raw scores from the model (shape: (batch_size,))
        raw_scores = self.model.score_hrt(hrt_batch)

        # Apply sigmoid to map the scores to (0, 1)
        normalized_scores = torch.sigmoid(raw_scores)

        # Return as a simple list of floats
        return normalized_scores.squeeze(dim=-1).tolist()

    def evaluate_model(
        self,
        facts: List[dict],
        output_file: str = "result.ttl",
        test: bool = True,
        threshold: float = 0.8
    ):
        """
        Evaluate and/or generate predictions for a list of fact dictionaries.
        
        If test=True:
          - We compute metrics only for facts that have a non-None truth_value.
          - All predictions (including unlabeled ones) are written to a TTL file.
        
        If test=False:
          - We skip computing metrics (since these are unlabeled or we're not in a testing context).
          - We write predictions to the TTL file.
        
        :param facts: A list of dicts with keys 'id', 'subject', 'predicate', 'object', 'truth_value'.
        :param output_file: Where to store the TTL output with predicted truth values.
        :param test: Whether to compute metrics (requires 'truth_value' to be present).
        :param threshold: The score threshold above which we classify as True (for accuracy metrics).
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
                # Collect for TTL output
                ttl_line = (
                    f"<{stmt_id}> <http://swc2017.aksw.org/hasTruthValue> "
                    f"\"{score}\"^^<http://www.w3.org/2001/XMLSchema#double> ."
                )
                output_lines.append(ttl_line)

                # If in test mode and the fact has a label, collect it for metrics
                if test and truth_value is not None: #NOTE: naming this test is misleading as this operates on the train dataset :v
                    y_true.append(truth_value)       #      in the run.py it is named is_labeled so it is not that bad ig
                    y_scores.append(score)

            except Exception as e:
                logger.error(f"Error scoring fact {fact}: {e}")

        # Write the TTL lines to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")

        # Compute metrics if in test mode
        if test:
            if len(y_true) == 0:
                logger.warning("No labeled facts found for evaluation.")
                return [], []

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)

            # Scores are in [0,1] due to sigmoid
            y_scores_normalized = y_scores

            y_pred = (y_scores_normalized > threshold).astype(float)
            roc_auc = roc_auc_score(y_true, y_scores_normalized)
            roc_auc_pred = roc_auc_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            print("\n")
            logger.info(f"ROC AUC (continuous scores): {roc_auc}")
            logger.info(f"Accuracy (threshold={threshold}): {accuracy}\n")

            return y_true, y_scores_normalized
        else:
            logger.info("Skipping evaluation (test=False).")
            return [], []
