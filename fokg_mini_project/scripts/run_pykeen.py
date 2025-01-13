import numpy as np
from rdflib import Graph
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
import pandas as pd

# # Step 1: Load the .nt file
# g = Graph()
# g.parse("data/reference-kg.nt", format="nt")

# # Step 2: Convert RDF triples to a NumPy array
# triples = [(str(s), str(p), str(o)) for s, p, o in g]
# triples_array = np.array(triples, dtype=str)

# df = pd.DataFrame({'subject': triples_array[:, 0], 'predicate': triples_array[:, 1], 'object': triples_array[:, 2]})
# df.to_csv('data/reference.csv', sep='\t', index=False, header=False)


triples_factory = TriplesFactory.from_path("data/reference.csv")



training, testing, validation = triples_factory.split([0.8, 0.1, 0.1])

result = pipeline(
	training=triples_factory,
	testing=testing,
	model='ComplEx',
	model_kwargs=dict(embedding_dim=50),
	optimizer_kwargs=dict(lr=0.01),
	training_kwargs=dict(num_epochs=100, batch_size=1024 * 16),
	evaluation_kwargs=dict(),
)
plot = result.plot_losses()
model = result.model

evaluator = RankBasedEvaluator()

# Evaluate your model with not only testing triples,
# but also filter on validation triples
results = evaluator.evaluate(
    model=model,
    mapped_triples=dataset.testing.mapped_triples,
    additional_filter_triples=[
        training.mapped_triples,
        datasetvalidation.mapped_triples,
    ],
)

plot.figure.savefig("data/losses.png")




