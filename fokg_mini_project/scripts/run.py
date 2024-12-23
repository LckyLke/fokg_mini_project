import os
from fokg_mini_project.kg import KG
from fokg_mini_project.trans_e import TransEForward
from fokg_mini_project.distmult import DistMultForward
from fokg_mini_project.complex import ComplexForward
from fokg_mini_project.kge import KGE
from fokg_mini_project.loss_functions import MarginLoss
import torch

# Define hyperparameter lists
embedding_dims = [150, 200]
learning_rates = [ 0.005, 0.001]
margins = [1, 2, 3]
num_epochs_list = [20, 30, 50]
from itertools import product
# Load the knowledge graph
kg = KG('data/reference-kg.nt')

# Open a file to write the results
results_file = "complex_hyperparameter_results.txt"
with open(results_file, "w") as f:
    f.write("Testing all hyperparameter combinations for ComplexForward\n")
    f.write("=" * 50 + "\n")
    
    # Generate all combinations of hyperparameters
    for i, (embedding_dim, lr, margin, num_epochs) in enumerate(
        product(embedding_dims, learning_rates, margins, num_epochs_list), 1
    ):
        # Log the current configuration
        params = {
            "embedding_dim": embedding_dim,
            "learning_rate": lr,
            "margin": margin,
            "num_epochs": num_epochs,
        }
        f.write(f"Configuration {i}: {params}\n")
        
        print(f"Testing configuration {i}: {params}")
        
        # Initialize the model with the current configuration
        model = KGE(
            kg,
            forward=ComplexForward(),
            loss_func=MarginLoss(margin=margin),
            embedding_dim=embedding_dim,
            lr=lr,
        )
        
        # Train the model
        model.train(num_epochs=num_epochs)
        
        # Evaluate the model
        hits_at_10 = model.eval_hits(hits_at=10)
        
        # Write the results to the file
        result = f"Configuration {i} - Hits@10: {hits_at_10}\n"
        f.write(result)
        print(result)
    
    f.write("=" * 50 + "\n")
    f.write("Testing complete.\n")

# Inform the user
print(f"Results saved to {results_file}")