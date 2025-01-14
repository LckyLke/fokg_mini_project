import numpy as np
from rdflib import Graph
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
import pandas as pd
import os

# Step 1: Load the .nt file
if not os.path.exists("data/reference.csv"):
	g = Graph()
	g.parse("data/reference-kg.nt", format="nt")

	# Step 2: Convert RDF triples to a NumPy array
	triples = [(str(s), str(p), str(o)) for s, p, o in g]
	triples_array = np.array(triples, dtype=str)

	df = pd.DataFrame({'subject': triples_array[:, 0], 'predicate': triples_array[:, 1], 'object': triples_array[:, 2]})
	filtered_df = df[df["predicate"] != "http://www.w3.org/2000/01/rdf-schema#label"]
	filtered_df.to_csv('data/reference.csv', sep='\t', index=False, header=False)


triples_factory = TriplesFactory.from_path("data/reference.csv")



training, testing, validation = triples_factory.split([0.8, 0.1, 0.1])

result = pipeline(
	training=triples_factory,
	testing=testing,
	validation=validation,
	model='ComplEx',
	model_kwargs=dict(embedding_dim=128),
	optimizer_kwargs=dict(lr=0.0005),
	training_kwargs=dict(num_epochs=100000, batch_size=1024*16),
	evaluation_kwargs=dict(),
	 stopper='early',

    stopper_kwargs=dict(frequency=1000, patience=2, relative_delta=0.001),
)
plot = result.plot_losses()
result.save_to_directory("data/model_complex_intense")
model = result.model

metrics = result.metric_results

print(f"Hits@1: {metrics.get_metric('hits@1')}")
print(f"Hits@3: {metrics.get_metric('hits@3')}")
print(f"Hits@5: {metrics.get_metric('hits@5')}")
print(f"Hits@10: {metrics.get_metric('hits@10')}")
print(f"Mean Reciprocal Rank: {metrics.get_metric('mean_reciprocal_rank')}")

plot.figure.savefig("data/losses.png")




