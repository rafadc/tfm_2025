import json

from benchmark_evaluator import BenchmarkEvaluator
from config import Config
from data_manager import DataManager
from prompt_generator import PromptGenerator
from visualizer import ResultsVisualizer


def main():
    Config.validate()

    data_manager = DataManager(Config.BASE_DIR)

    print("Step 1: Downloading MMLU benchmark data...")
    if not data_manager.download_and_extract_mmlu():
        print("Failed to download data. Exiting.")
        return

    results_summary = {"llm_models": {}}

    for llm_model in Config.LLM_MODELS:
        print(f"\n{'='*60}")
        print(f"Processing LLM: {llm_model}")
        print(f"{'='*60}")

        llm_name = llm_model.replace(":", "")

        try:
            print("\nStep 2: Preparing LLM-specific data directory...")
            llm_data_dir = data_manager.prepare_llm_data_dir(llm_name)

            test_file = data_manager.data_dir / "test" / Config.TEST_FILE
            prompts_file = llm_data_dir / "test" / Config.TEST_FILE.replace("_test.csv", "_test_prompts.csv")

            if not prompts_file.exists():
                print("\nStep 3: Generating prompt variants...")
                generator = PromptGenerator(llm_model, Config.OLLAMA_BASE_URL)
                generator.process_test_file(test_file, prompts_file)
            else:
                print(f"Prompt variants already exist at {prompts_file}")

            print("\nStep 4: Running benchmark evaluations...")
            evaluator = BenchmarkEvaluator(llm_model, Config.OLLAMA_BASE_URL)

            print("\n4.1: Running baseline evaluation...")
            baseline_output = Config.RESULTS_DIR / f"{llm_name}_baseline_results.json"

            # Check if baseline results already exist (caching)
            if baseline_output.exists():
                print(f"Loading cached baseline results from {baseline_output}")
                with open(baseline_output, "r") as f:
                    baseline_results = json.load(f)
                print(f"Cached baseline accuracy: {baseline_results['accuracy']:.2f}%")
            else:
                baseline_results = evaluator.run_baseline_evaluation(test_file)
                evaluator.save_results(baseline_results, baseline_output)

            print("\n4.2: Running optimized evaluation with prompt variants...")
            optimized_output = Config.RESULTS_DIR / f"{llm_name}_optimized_results.json"

            # Check if optimized results already exist (caching)
            if optimized_output.exists():
                print(f"Loading cached optimized results from {optimized_output}")
                with open(optimized_output, "r") as f:
                    optimized_results = json.load(f)
                print(f"Cached optimized accuracy: {optimized_results['accuracy']:.2f}%")
            else:
                optimized_results = evaluator.run_optimized_evaluation(test_file, prompts_file)
                evaluator.save_results(optimized_results, optimized_output)

            improvement = optimized_results["accuracy"] - baseline_results["accuracy"]

            results_summary["llm_models"][llm_model] = {
                "baseline_accuracy": baseline_results["accuracy"],
                "optimized_accuracy": optimized_results["accuracy"],
                "improvement": improvement,
                "baseline_correct": baseline_results["correct"],
                "optimized_correct": optimized_results["correct"],
                "total_questions": baseline_results["total"],
            }

            print(f"\nResults for {llm_model}:")
            print(f"  Baseline accuracy:  {baseline_results['accuracy']:.2f}%")
            print(f"  Optimized accuracy: {optimized_results['accuracy']:.2f}%")
            print(f"  Improvement:        {improvement:+.2f}%")

        except RuntimeError as e:
            print(f"\n{'='*60}")
            print(f"ERROR: Ollama call failed for model {llm_model}")
            print(f"Error details: {e}")
            print(f"{'='*60}")
            print("\nExperiment stopped due to Ollama failure.")
            print("Please check that Ollama is running and the model is available.")
            print(f"Test with: curl {Config.OLLAMA_BASE_URL}/api/generate "
                  f"-d '{{\"model\":\"{llm_model}\",\"prompt\":\"test\"}}'")
            return

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for model, results in results_summary["llm_models"].items():
        print(f"\n{model}:")
        baseline = results['baseline_accuracy']
        baseline_correct = results['baseline_correct']
        total_q = results['total_questions']
        print(f"  Baseline:  {baseline:.2f}% ({baseline_correct}/{total_q})")
        optimized = results['optimized_accuracy']
        optimized_correct = results['optimized_correct']
        print(f"  Optimized: {optimized:.2f}% ({optimized_correct}/{total_q})")
        print(f"  Improvement: {results['improvement']:+.2f}%")

    summary_file = Config.RESULTS_DIR / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nExperiment summary saved to {summary_file}")

    # Step 5: Generate visualizations and tables
    if results_summary["llm_models"]:
        visualizer = ResultsVisualizer(Config.RESULTS_DIR)
        visualizer.generate_all_visualizations(results_summary)


if __name__ == "__main__":
    main()

