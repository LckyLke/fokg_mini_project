import numpy as np
from rdflib import Graph
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import pandas as pd

# Step 1: Load the .nt file
g = Graph()
g.parse("data/reference-kg.nt", format="nt")

# Step 2: Convert RDF triples to a NumPy array
triples = [(str(s), str(p), str(o)) for s, p, o in g]
triples_array = np.array(triples, dtype=str)

df = pd.DataFrame({'subject': triples_array[:, 0], 'predicate': triples_array[:, 1], 'object': triples_array[:, 2]})
df.to_csv('data/reference.csv', sep='\t', index=False, header=False)


triples_factory = TriplesFactory.from_path("data/reference.csv")



training, testing, validation = triples_factory.split([0.8, 0.1, 0.1])



# d=training
# id_to_entity = {i: e for e, i in d.entity_to_id.items()}
# id_to_relation = {i: r for r, i in d.relation_to_id.items()}
 
result = pipeline(
	model='ComplEx',	
	training=triples_factory,
	testing=testing,
	validation=validation,
	model_kwargs=dict(embedding_dim=128*2),
	optimizer_kwargs=dict(lr=0.005),
	training_kwargs=dict(num_epochs=1000, batch_size=1024*32),
)

model = result.model

from pykeen.evaluation import RankBasedEvaluator

# Create an evaluator
evaluator = RankBasedEvaluator()

# Evaluate the model
metrics = evaluator.evaluate(result.model, testing.mapped_triples, additional_filter_triples=[training.mapped_triples, validation.mapped_triples])

# Print the metrics
print(f"Hits@1: {metrics.get_metric('hits@1')}")
print(f"Hits@3: {metrics.get_metric('hits@3')}")
print(f"Hits@5: {metrics.get_metric('hits@5')}")
print(f"Hits@10: {metrics.get_metric('hits@10')}")
print(f"Mean Reciprocal Rank: {metrics.get_metric('mean_reciprocal_rank')}")