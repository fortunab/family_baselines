import argparse
import os
import sys
from experiments.runner import run_experiment_suite

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Medical Classification Paper Experiment Suite")
    parser.add_argument('--run-all', action='store_true', help='Run full paper experiment suite across all platforms, models, and environments')
    parser.add_argument('--cross-validation', action='store_true', help='Perform 10-fold cross-validation')
    parser.add_argument('--output', type=str, default='experiment_results.json', help='Output JSON file path')

    args = parser.parse_args()

    # Change CWD to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print("Running Federated Learning Medical Classification Experiments...")
    run_experiment_suite(use_10fold=args.cross_validation, output_file=args.output)

if __name__ == '__main__':
    main()
