
from typing import List, Tuple
from dicee.knowledge_graph_embeddings import KGE
import numpy as np
from torch import FloatTensor
from kg_utils import KGUtils
import os
from rdflib import Graph, Namespace, RDF, URIRef, Literal
import re
from sklearn.metrics import roc_auc_score, accuracy_score


class FactChecker:
    def __init__(self, path_kb=None, path_kbe=None):
        assert path_kb is not None or path_kbe is not None, "Please provide a path to the knowledge base or the knowledge base embeddings"
        self.path_kb = path_kb
        self.path_kbe = path_kbe

        if path_kbe:
            self.model = KGE(path=path_kbe)
        else:
            # Train the model
            from dicee.executer import Execute
            from dicee.config import Namespace
            args = Namespace()
            args.model = 'Keci'
            args.scoring_technique = 'KvsAll'
            args.path_single_kg = path_kb
            path_kb = path_kb.replace("/", "_")
            path_kb = path_kb.replace(".", "_")
            args.num_epochs = 50
            args.embedding_dim = 128 * 4
            args.num_core = 2
            args.batch_size = 1024 * 2
            args.backend = "rdflib"
            args.trainer = "PL"
            path_kb = f"{path_kb}_{args.model}_{args.embedding_dim}-dim_{args.num_epochs}-epochs"
            # check if model folder already exists
            assert not os.path.exists(path_kb), f"Model folder already exists at {path_kb}"			
            args.path_to_store_single_run = path_kb
            reports = Execute(args).start()
            self.path_kbe = reports["path_experiment_folder"]
            self.train_score = reports["Train"]["MRR"]
            self.model = KGE(path=self.path_kbe)
            print(f"Model trained with MRR: {self.train_score}")
            print(f"Model saved at: {self.path_kbe}")

    def predict(self, head=None, relation=None, tail=None):
        assert head is not None or tail is not None or relation is not None, "Please provide a head, tail or relation"

        if head is not None and relation is not None:
            topk = len(self.model.entity_to_idx)
            return (self.model.predict_topk(h=head, r=relation, topk=topk),topk)
        elif head is not None and tail is not None:
            topk = len(self.model.relation_to_idx)
            return (self.model.predict_topk(h=head, t=tail, topk=topk),topk)
        elif relation is not None and tail is not None:
            topk = len(self.model.entity_to_idx)
            return (self.model.predict_topk(r=relation, t=tail, topk=topk),topk)

    def triple_score(self, head: List[str] = None, relation: List[str] = None, tail: List[str] = None) -> FloatTensor:
        assert head is not None and relation is not None and tail is not None, "Please provide a head, relation and tail"
        return self.model.triple_score(h=head, r=relation, t=tail)


def parse_rdf_file(file_path):
    from rdflib import Graph, Namespace, RDF
    g = Graph()
    g.parse(file_path, format='nt')  # Assumes the file is in N-Triples format

    SWC = Namespace("http://swc2017.aksw.org/")  # Adjust the namespace if needed
    RDF_SYNTAX_NS = RDF
    statements = {}

    for s, p, o in g:
        if s not in statements:
            # Initialize the statement dictionary for s
            statements[s] = {'subject': None, 'predicate': None, 'object': None, 'truth_value': None, 'is_statement': False}
        
        if p == RDF_SYNTAX_NS.type and o == RDF_SYNTAX_NS.Statement:
            # Mark s as a statement
            statements[s]['is_statement'] = True
        elif p == RDF_SYNTAX_NS.subject:
            statements[s]['subject'] = o
        elif p == RDF_SYNTAX_NS.predicate:
            statements[s]['predicate'] = o
        elif p == RDF_SYNTAX_NS.object:
            statements[s]['object'] = o
        elif p == SWC.hasTruthValue:
            truth_value = float(o)
            statements[s]['truth_value'] = truth_value

    # Filter out entries that are not marked as statements
    facts = []
    for stmt_id, stmt in statements.items():
        if stmt['is_statement'] and None not in [stmt['subject'], stmt['predicate'], stmt['object'], stmt['truth_value']]:
            facts.append(stmt)

    return facts


print(parse_rdf_file("data/fokg-sw-train-2024.nt")[0])

def evaluate_model(fact_checker, facts):
    y_true = []
    y_scores = []

    for fact in facts:
        head = str(fact['subject'])
        relation = str(fact['predicate'])
        tail = str(fact['object'])
        truth_value = fact['truth_value']

        # Use triple_score method
        try:
            # The method expects lists of strings
            score_tensor = fact_checker.triple_score(
                head=[head], 
                relation=[relation], 
                tail=[tail]
            )
            score = score_tensor.item()  # Extract the scalar from the tensor
            
            y_true.append(truth_value)
            y_scores.append(score)
            print(f"Scored fact {fact}: {score}")
            print(f"Truth value: {truth_value}")
        except Exception as e:
            print(f"Error scoring fact {fact}: {e}")

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Normalize scores to [0, 1]
    y_scores_normalized = y_scores

    # Compute evaluation metrics
    roc_auc = roc_auc_score(y_true, y_scores_normalized)
    y_pred = (y_scores_normalized > 0.5).astype(float)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"ROC AUC Score: {roc_auc}")
    print(f"Accuracy: {accuracy}")

    return y_true, y_scores_normalized

if __name__ == "__main__":
    # Paths to  model and data
    path_kb = "data/reference-kg.nt"
    path_kbe = "model_keci"  # Adjust as needed
    rdf_file_path = "data/fokg-sw-train-2024.nt"  # Replace with your file path

    # Initialize the FactChecker
    fact_checker = FactChecker(path_kbe=path_kbe)

    # Parse the RDF data
    facts = parse_rdf_file(rdf_file_path)[0:5]  
    print(f"Parsed {len(facts)} facts.")

    # Evaluate the model
    y_true, y_scores = evaluate_model(fact_checker, facts)

# if __name__ == "__main__":
# 	path_kbe = "model_keci"
# 	path_train_facts = "data/fokg-sw-train-2024.nt"
# 	fact_checker = FactChecker(path_kbe=path_kbe)
# 	#KGUtils.enrich_kg_with_hierarchy(kg_file=path_kb, hierarchy_file="data/classHierarchy.nt", output_file="data/reference-kg-enriched.nt")
# 	# head = "http://rdf.freebase.com/ns/m.02wr6r" #982
# 	# relation = "http://rdf.freebase.com/ns/people.deceased_person.place_of_death"
# 	# tail = "http://rdf.freebase.com/ns/m.02_286"

# 	head = "http://rdf.freebase.com/ns/m.02mc79"
# 	relation = "http://rdf.freebase.com/ns/people.person.profession"
# 	tail = "http://rdf.freebase.com/ns/m.04s2z"
# 	relations = list(fact_checker.predict(head=head, tail=tail))
# 	count = 0
# 	while True:
# 		print(f"Relation: {relations[0][count]}")
        
# 		if relations[0][count][0] == relation:
# 			break
# 		count += 1
# 	print(f"Rank of the correct relation: {count+1} with topk: {relations[1]}")
# 	print(f"Veracity value: {1-(count/relations[1])}")


