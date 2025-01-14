

# Fact Checking Project

This project enables fact-checking experiments on a knowledge graph using a pre-trained model or a new one. It supports `.nt` or `.csv` formats as input and outputs the results in `.ttl`.

## Setup Instructions

### Prerequisites

Ensure you have **conda** installed on your system. You can install Miniconda or Anaconda from their respective websites if not already available.

### Step 1: Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 2: Create and Activate a Conda Environment

Create a new conda environment and activate it:

```bash
conda create --name fact-checking-env python=3.9 -y
conda activate fact-checking-env
```

### Step 3: Install Dependencies and Project

Install the required dependencies and set up the project using the `setup.py` file:

```bash
pip install -e .
```

This installs the project in editable mode and registers the `factCheck` command.

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

- `--model-checkpoint`: Path to a pre-trained PyTorch model checkpoint (supported formats: `.pt`, `.pkl`, `.ptk`). If not provided, a new model is trained.
- `--is-labeled`: Flag indicating that the `fact-base-file` contains labeled facts with truth values (`hasTruthValue`). If set, metrics will be computed.
- `--output-ttl`: Path to the output file with predicted scores (default: `result.ttl`).

### Examples

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
