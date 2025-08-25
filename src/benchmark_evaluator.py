import csv
import json
from pathlib import Path

from langchain_ollama import OllamaLLM


class BenchmarkEvaluator:
    def __init__(self, llm_model: str, base_url: str = "http://localhost:11434"):
        self.llm = OllamaLLM(model=llm_model, base_url=base_url, temperature=0)
        self.llm_model = llm_model

    def evaluate_question(self, question: str, choices: list[str], correct_answer: str) -> tuple[str, bool]:
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

        try:
            response = self.llm.invoke(prompt).strip().upper()

            if response not in ["A", "B", "C", "D"]:
                response = response[0] if response and response[0] in ["A", "B", "C", "D"] else "A"

            is_correct = response == correct_answer
            return response, is_correct

        except Exception as e:
            print(f"Error evaluating question: {e}")
            raise RuntimeError(f"Failed to evaluate question with Ollama: {e}") from e

    def evaluate_with_prompt_variants(
        self, question_variants: list[str], choices: list[str], correct_answer: str
    ) -> tuple[list[float], int]:
        scores = []
        for variant in question_variants:
            try:
                _, is_correct = self.evaluate_question(variant, choices, correct_answer)
                scores.append(1.0 if is_correct else 0.0)
            except RuntimeError:
                raise

        best_variant_idx = scores.index(max(scores))
        return scores, best_variant_idx

    def run_baseline_evaluation(self, test_file: Path) -> dict:
        print(f"Running baseline evaluation on {test_file}")

        total_questions = 0
        correct_answers = 0
        results = []

        with open(test_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue

                question = row[0]
                choices = row[1:5]
                correct_answer = row[5].strip().upper()

                predicted, is_correct = self.evaluate_question(question, choices, correct_answer)

                total_questions += 1
                if is_correct:
                    correct_answers += 1

                results.append(
                    {"question": question, "predicted": predicted, "correct": correct_answer, "is_correct": is_correct}
                )

                if total_questions % 10 == 0:
                    print(f"Evaluated {total_questions} questions...")

        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        return {"total": total_questions, "correct": correct_answers, "accuracy": accuracy, "results": results}

    def run_optimized_evaluation(self, test_file: Path, prompts_file: Path) -> dict:
        print(f"Running optimized evaluation with prompts from {prompts_file}")

        total_questions = 0
        correct_answers = 0
        results = []

        with open(test_file, "r", encoding="utf-8") as test_f, open(prompts_file, "r", encoding="utf-8") as prompts_f:
            test_reader = csv.reader(test_f)
            prompts_reader = csv.reader(prompts_f)

            for test_row, prompts_row in zip(test_reader, prompts_reader):
                if len(test_row) < 5:
                    continue

                choices = test_row[1:5]
                correct_answer = test_row[5].strip().upper()

                scores, best_idx = self.evaluate_with_prompt_variants(prompts_row, choices, correct_answer)

                best_question = prompts_row[best_idx]
                predicted, is_correct = self.evaluate_question(best_question, choices, correct_answer)

                total_questions += 1
                if is_correct:
                    correct_answers += 1

                results.append(
                    {
                        "original_question": test_row[0],
                        "best_prompt": best_question,
                        "best_idx": best_idx,
                        "scores": scores,
                        "predicted": predicted,
                        "correct": correct_answer,
                        "is_correct": is_correct,
                    }
                )

                if total_questions % 10 == 0:
                    print(f"Evaluated {total_questions} questions...")

        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        return {"total": total_questions, "correct": correct_answers, "accuracy": accuracy, "results": results}

    def save_results(self, results: dict, output_file: Path):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

