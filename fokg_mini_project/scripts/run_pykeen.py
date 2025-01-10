import numpy as np
from rdflib import Graph
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# Step 1: Load the .nt file
g = Graph()
g.parse("/data/family.nt,", format="nt")

# Step 2: Convert RDF triples to a NumPy array
triples = [(str(s), str(p), str(o)) for s, p, o in g]
triples_array = np.array(triples, dtype=str)

# Step 3: Create a TriplesFactory
factory = TriplesFactory.from_labeled_triples(triples_array)

# Step 4: Split the data into training, validation, and testing
training, testing, validation = factory.split([0.8, 0.1, 0.1])

# Step 5: Train a KGE model using the pipeline
result = pipeline(
    training=training,
    validation=validation,
    testing=testing,
    model='ComplEx',  # or any other KGE model
    training_kwargs=dict(num_epochs=10, batch_size=1024),
	dimensions=50,
    optimizer_kwargs=dict(lr=0.01),
)

# Step 6: Print evaluation metrics
print("MRR:", result.get_metric('mean_reciprocal_rank'))
print("Hits@10:", result.get_metric('hits@1'))

# Step 7: Save the trained model
result.save_to_directory("output/trained_model")
