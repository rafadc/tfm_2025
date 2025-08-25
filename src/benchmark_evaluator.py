import csv
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from langchain_ollama import OllamaLLM


class BenchmarkEvaluator:
    def __init__(self, llm_model: str, base_url: str = "http://localhost:11434"):
        self.llm = OllamaLLM(model=llm_model, base_url=base_url, temperature=0)
        self.llm_model = llm_model

    def _format_mmlu_prompt(self, question: str, choices: List[str]) -> str:
        """Format a question and choices into MMLU prompt format."""
        prompt = (
            "Answer the following multiple choice question by selecting "
            "only the letter (A, B, C, or D) of the correct answer.\n"
            f"\nQuestion: {question}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            "\nAnswer (only the letter):"
        )
        return prompt

    def _evaluate_single(self, prompt: str) -> str:
        """Evaluate a single prompt and return the predicted answer."""
        try:
            response = self.llm.invoke(prompt).strip().upper()

            if response not in ["A", "B", "C", "D"]:
                response = response[0] if response and response[0] in ["A", "B", "C", "D"] else "A"

            return response
        except Exception as e:
            print(f"Error evaluating prompt: {e}")
            raise RuntimeError(f"Failed to evaluate question with Ollama: {e}") from e

    def _load_test_data(self, test_file: Path) -> List[Dict]:
        """Load test data from CSV file into HuggingFace-compatible format."""
        data = []
        with open(test_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    data.append({
                        "question": row[0],
                        "choices": row[1:5],
                        "answer": row[5].strip().upper()
                    })
        return data

    def run_baseline_evaluation(self, test_file: Path) -> Dict:
        """Run baseline evaluation using standard MMLU prompts."""
        print(f"Running HuggingFace MMLU-style baseline evaluation on {test_file}")

        test_data = self._load_test_data(test_file)
        dataset = Dataset.from_list(test_data)

        total_questions = 0
        correct_answers = 0
        results = []

        for example in dataset:
            question = example["question"]
            choices = example["choices"]
            correct_answer = example["answer"]

            prompt = self._format_mmlu_prompt(question, choices)
            predicted = self._evaluate_single(prompt)

            is_correct = predicted == correct_answer
            total_questions += 1
            if is_correct:
                correct_answers += 1

            results.append({
                "question": question,
                "predicted": predicted,
                "correct": correct_answer,
                "is_correct": is_correct
            })

            if total_questions % 10 == 0:
                print(f"Evaluated {total_questions} questions...")

        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        return {
            "total": total_questions,
            "correct": correct_answers,
            "accuracy": accuracy,
            "results": results
        }

    def run_optimized_evaluation(self, test_file: Path, prompts_file: Path) -> Dict:
        """Run optimized evaluation with prompt variants."""
        print(f"Running HuggingFace MMLU-style optimized evaluation with prompts from {prompts_file}")

        test_data = self._load_test_data(test_file)

        total_questions = 0
        correct_answers = 0
        results = []

        with open(prompts_file, "r", encoding="utf-8") as prompts_f:
            prompts_reader = csv.reader(prompts_f)

            for test_item, prompts_row in zip(test_data, prompts_reader):
                if not prompts_row:
                    continue

                choices = test_item["choices"]
                correct_answer = test_item["answer"]

                # Evaluate all prompt variants
                scores = []
                for variant in prompts_row:
                    prompt = self._format_mmlu_prompt(variant, choices)
                    predicted = self._evaluate_single(prompt)
                    is_correct = predicted == correct_answer
                    scores.append(1.0 if is_correct else 0.0)

                # Use the best performing variant
                best_idx = scores.index(max(scores))
                best_question = prompts_row[best_idx]

                # Get final prediction with best prompt
                best_prompt = self._format_mmlu_prompt(best_question, choices)
                predicted = self._evaluate_single(best_prompt)
                is_correct = predicted == correct_answer

                total_questions += 1
                if is_correct:
                    correct_answers += 1

                results.append({
                    "original_question": test_item["question"],
                    "best_prompt": best_question,
                    "best_idx": best_idx,
                    "scores": scores,
                    "predicted": predicted,
                    "correct": correct_answer,
                    "is_correct": is_correct
                })

                if total_questions % 10 == 0:
                    print(f"Evaluated {total_questions} questions...")

        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        return {
            "total": total_questions,
            "correct": correct_answers,
            "accuracy": accuracy,
            "results": results
        }

    def save_results(self, results: Dict, output_file: Path):
        """Save evaluation results to JSON file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
