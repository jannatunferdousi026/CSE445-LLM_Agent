import json
from ml_tools import train_sklearn_model


DATASETS = [
    "iris",
    "wine",
]

ALGORITHMS = [
    "decision_tree",
    "logistic_regression",
    "random_forest",
]


def run_benchmark():
    results = []

    print("\n==============================================")
    print("CSE445 TASK 3 - MODEL COMPARISON BENCHMARK")
    print("==============================================")

    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            print(f"\nRunning: {algorithm} on {dataset}")

            raw_result = train_sklearn_model(
                dataset,
                algorithm,
                test_size=0.2,
            )

            result = json.loads(raw_result)

            if "error" in result:
                print(f"ERROR: {result['error']}")
                continue

            results.append(result)

            print(
                f"Test Accuracy: {result['test_accuracy']:.4f} | "
                f"CV Mean: {result['cv_mean_accuracy']:.4f} | "
                f"CV Std: {result['cv_std']:.4f}"
            )

    return results


def create_markdown_table(results):
    lines = [
        "# Experimental Summary",
        "",
        "| Dataset | Algorithm | Test Accuracy | CV Mean Accuracy | CV Std |",
        "|---|---|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            f"| {result['dataset']} "
            f"| {result['model']} "
            f"| {result['test_accuracy']:.4f} "
            f"| {result['cv_mean_accuracy']:.4f} "
            f"| {result['cv_std']:.4f} |"
        )

    return "\n".join(lines)


def main():
    results = run_benchmark()

    if not results:
        print("\nNo benchmark results were produced.")
        return

    markdown = create_markdown_table(results)

    with open("benchmark_results.md", "w", encoding="utf-8") as file:
        file.write(markdown + "\n")

    print("\n==============================================")
    print("MARKDOWN EXPERIMENTAL SUMMARY")
    print("==============================================")
    print(markdown)

    print("\nSaved to: benchmark_results.md")


if __name__ == "__main__":
    main()
