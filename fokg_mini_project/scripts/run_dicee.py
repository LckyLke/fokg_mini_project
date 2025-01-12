from dicee.executer import Execute 
from dicee.config import Namespace

import random

def subset_file(input_file, output_file, fraction=0.01):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # Select a subset of lines
    num_subset = int(len(lines) * fraction)
    subset_lines = random.sample(lines, num_subset)
    
    with open(output_file, 'w') as outfile:
        outfile.writelines(subset_lines)

# Define the file paths and fraction
input_file = "data/reference-kg.nt"
output_file = "data/reference-kg-subset.nt"
subset_fraction = 0.1

subset_file(input_file, output_file, subset_fraction)
path_of_kb = "data/reference-kg-subset.nt"

args = Namespace()
args.model = 'Keci'
args.scoring_technique = 'KvsAll'
args.path_single_kg = path_of_kb
path_of_kb = path_of_kb.replace("/", "_")
path_of_kb = path_of_kb.replace(".", "_")
args.path_to_store_single_run = path_of_kb

args.num_epochs = 100
args.embedding_dim = 512
args.batch_size = 1024 * 2
args.backend = "rdflib"
args.trainer = "PL"
reports = Execute(args).start()
path_neural_embedding = reports["path_experiment_folder"]
print(reports["Train"]["MRR"]) 
print(reports["Test"]["MRR"]) 
