[Gerbil Result](http://gerbil-kbc.aksw.org/gerbil/experiment?id=202501140061)


# Fact Checking Project

This project enables fact-checking experiments on a knowledge graph using a pre-trained model or a new one. It supports `.nt` or `.csv` formats as input and outputs the results in `.ttl`.

## Setup Instructions

### Prerequisites

Ensure you have **conda** installed on your system. You can install Miniconda or Anaconda from their [respective websites](https://www.anaconda.com/download/success) if not already available.

### Step 1: Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/LckyLke/fokg_mini_project.git
cd fokg_mini_project
```

### Step 2: Create and Activate a Conda Environment

Create a new conda environment and activate it:

```bash
conda create --name fact-checking-env python=3.10.13 -y
conda activate fact-checking-env
```

### Step 3: Install Dependencies and Project

Install the required dependencies and set up the project using the `setup.py` file:

```bash
pip install -e .
```

This installs the project in editable mode and registers the `factCheck` command.

## Examples

1. Run fact-checking with a pre-trained model:
   ```bash
   factCheck --reference-file data/reference-kg.nt --fact-base-file data/fokg-sw-test-2024.nt --model-path data/model_complex/trained_model.pkl
   ```

2. Run fact-checking and compute evaluation metrics:
   ```bash
   factCheck --reference-file data/reference-kg.nt --fact-base-file data/fokg-sw-train-2024.nt --is-labeled
   ```

3. Specify a custom output file:
   ```bash
   factCheck --reference-file data/reference-kg.nt --fact-base-file fokg-sw-test-2024.nt --output-ttl results.ttl
   ```

## Usage Instructions

Once the project is set up, you can use the `factCheck` command to run fact-checking experiments.

### Command Syntax

```bash
factCheck --reference-file <REFERENCE_FILE> --fact-base-file <FACT_BASE_FILE> [OPTIONS]
```

### Required Arguments

- `--reference-file`: Path to the reference knowledge graph file (must be `.nt` or `.csv`).
- `--fact-base-file`: Path to the file containing the facts to check (in `.nt` format).

### Optional Arguments

- `--model-path`: Path to a pre-trained PyTorch model checkpoint (supported formats: `.pt`, `.pkl`, `.ptk`). If not provided, a new model is trained.
- `--is-labeled`: Flag indicating that the `fact-base-file` contains labeled facts with truth values (`hasTruthValue`). If set, metrics will be computed.
- `--output-ttl`: Path to the output file with predicted scores (default: `result.ttl`).

## Approach Description

The approach implements a fact-checking pipeline that leverages Knowledge Graph Embedding (KGE) models for evaluating the plausibility of triples. It operates in the following stages:

---

### **1. Data Preparation**
- The pipeline supports input files in two formats: **N-Triples (.nt)** and **CSV** (tab-separated triples).
- **Conversion**: If the input is an N-Triples file, it converts the data into a CSV format. The `rdfs:label` predicates are filtered out to avoid irrelevant data.
- The parsed triples are loaded into a **PyKEEN TriplesFactory**.

---

### **2. Model Initialization**
- The framework supports:
  1. **Pretrained Models**: If a model is provided, it can be loaded directly.
  2. **Training a New Model**: If no model is provided, a new KGE model (default: ComplEx) is trained.
- Training is performed on the provided triples, with a default hyperparameter setup.
- **Loss Plot**: A loss plot is generated for debugging and visualization.

---

### **3. Fact Parsing**
- A parser extracts facts from an .nt file. It identifies facts represented as `rdf:Statement` nodes and captures their subject, predicate, object, and optional `hasTruthValue` literal.
- These parsed facts are used for veracity value predictions or evaluation.

---

### **4. Prediction**
- Predicts plausibility scores for triples using the KGE model.
- Converts entity and relation labels to IDs, feeds them into the model, and retrieves raw scores. The scores are normalized to a [0,1] range using the sigmoid function.

---

### **5. Evaluation**
- Supports evaluation of **labeled facts** (those with `hasTruthValue`) to compute metrics:
  - **Accuracy**: Based on a score threshold (default: 0.8).
  - **ROC-AUC**: Continuous score-based.
- Writes **predictions** (truth values) to a Turtle file (`.ttl`) for GERBIL.

---

### **6. Outputs**
- **Predicted Scores**: Stored in a `.ttl` file, associating each fact with a computed truth value.
- **Evaluation Metrics**: If labeled facts are provided, outputs metrics (ROC-AUC, accuracy).
- **Loss Plot**: Visualizes the training process for debugging.


