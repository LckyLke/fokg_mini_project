import argparse
from fokg_mini_project.fact_checker import FactChecker


def main():
    parser = argparse.ArgumentParser(description="Run fact checking experiments.")
    parser.add_argument(
        "--reference-file",
        help="Path to the reference knowledge graph file (either .nt or .csv).",
        required=True,
    )
    parser.add_argument(
        "--fact-base-file",
        help="Path to the file containing the facts to check (in .nt format).",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        help="Path to a pre-trained PyTorch model (.pt, .pkl, or .ptk). "
             "If provided, we skip training and load directly.",
        default=None
    )
    parser.add_argument(
        "--is-labeled",
        action='store_true',
        help="Flag indicating that the `fact-base-file` .nt contains labeled facts (hasTruthValue). "
             "If set, metrics will be computed."
    )
    parser.add_argument(
        "--output-ttl",
        default="result.ttl",
        help="Path to the output TTL file with predicted scores."
    )

    args = parser.parse_args()

    # Initialize the FactChecker
    # We set path_nt or path_csv depending on the extension of reference-file
    path_nt = None
    path_csv = None

    if args.reference_file.lower().endswith(".nt"):
        path_nt = args.reference_file
    elif args.reference_file.lower().endswith(".csv"):
        path_csv = args.reference_file
    else:
        raise ValueError("reference-file must be either .nt or .csv")

    fact_checker = FactChecker(
        path_nt=path_nt,
        path_csv=path_csv,
        path_model=args.model_path,  # Could be None
    )

    # Parse the fact-base file (labeled or unlabeled .nt)
    facts = fact_checker.parse_rdf_file(args.fact_base_file)

    # Evaluate or just score
    y_true, y_scores = fact_checker.evaluate_model(
        facts,
        output_file=args.output_ttl,
        test=args.is_labeled
    )

    if args.is_labeled and len(y_true) > 0:
        print("Evaluation done. Metrics were computed.")
    else:
        print("Scoring done. No metrics computed.")


if __name__ == "__main__":
    main()
