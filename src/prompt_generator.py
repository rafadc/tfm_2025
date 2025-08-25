import csv
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


class PromptGenerator:
    def __init__(self, llm_model: str, base_url: str = "http://localhost:11434"):
        self.llm = OllamaLLM(model=llm_model, base_url=base_url)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in generating alternative formulations of questions. "
                    "Given a question, generate 10 different ways to ask the same question. "
                    "Each alternative should maintain the same meaning but use different words or structure. "
                    "Return only the 10 alternatives, separated by | character, without numbering or additional text.",
                ),
                ("user", "{question}"),
            ]
        )

    def generate_prompt_variants(self, question: str, num_variants: int = 10) -> list[str]:
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({"question": question})

            variants = [v.strip() for v in response.split("|")][:num_variants]

            while len(variants) < num_variants:
                variants.append(question)

            return variants

        except Exception as e:
            print(f"Error generating variants: {e}")
            return [question] * num_variants

    def process_test_file(self, input_file: Path, output_file: Path):
        print(f"Processing {input_file} -> {output_file}")

        prompts_data = []

        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if len(row) < 5:
                    continue

                question = row[0]
                print(f"Processing question {row_num}: {question[:50]}...")

                variants = self.generate_prompt_variants(question)
                prompts_data.append(variants)

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for variants in prompts_data:
                writer.writerow(variants)

        print(f"Generated {len(prompts_data)} rows of prompt variants")

