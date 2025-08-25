from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class ResultsVisualizer:
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir

    def create_comparison_table(self, results_summary: Dict) -> str:
        """Create a formatted comparison table of results."""
        table_data = []
        headers = ["LLM Model", "Baseline Accuracy (%)", "Optimized Accuracy (%)",
                   "Improvement (%)", "Baseline Correct", "Optimized Correct", "Total Questions"]

        for model, results in results_summary["llm_models"].items():
            table_data.append([
                model,
                f"{results['baseline_accuracy']:.2f}",
                f"{results['optimized_accuracy']:.2f}",
                f"{results['improvement']:+.2f}",
                results['baseline_correct'],
                results['optimized_correct'],
                results['total_questions']
            ])

        # Create formatted table
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")

        # Also save as CSV for further processing
        df = pd.DataFrame(table_data, columns=headers)
        csv_path = self.results_dir / "results_comparison_table.csv"
        df.to_csv(csv_path, index=False)

        # Save formatted table to text file
        table_path = self.results_dir / "results_comparison_table.txt"
        with open(table_path, "w") as f:
            f.write("MMLU Benchmark Results Comparison\n")
            f.write("=" * 80 + "\n\n")
            f.write(table_str)
            f.write("\n")

        print(f"\nTable saved to {table_path}")
        print(f"CSV saved to {csv_path}")

        return table_str

    def create_accuracy_comparison_chart(self, results_summary: Dict):
        """Create a bar chart comparing baseline vs optimized accuracy."""
        models = []
        baseline_scores = []
        optimized_scores = []

        for model, results in results_summary["llm_models"].items():
            models.append(model)
            baseline_scores.append(results['baseline_accuracy'])
            optimized_scores.append(results['optimized_accuracy'])

        # Create grouped bar chart
        x = range(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar([i - width/2 for i in x], baseline_scores, width,
                       label='Baseline', color='#3498db', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], optimized_scores, width,
                       label='Optimized', color='#2ecc71', alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('LLM Model', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('MMLU Benchmark: Baseline vs Optimized Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = self.results_dir / "accuracy_comparison_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Accuracy comparison chart saved to {chart_path}")

    def create_improvement_chart(self, results_summary: Dict):
        """Create a chart showing the improvement percentage for each model."""
        models = []
        improvements = []

        for model, results in results_summary["llm_models"].items():
            models.append(model)
            improvements.append(results['improvement'])

        # Create horizontal bar chart for improvements
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color bars based on positive/negative improvement
        colors = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in improvements]

        bars = ax.barh(models, improvements, color=colors, alpha=0.8)

        # Add value labels
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            label_x = width + 0.1 if width >= 0 else width - 0.1
            ha = 'left' if width >= 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{imp:+.2f}%', ha=ha, va='center', fontsize=11, fontweight='bold')

        ax.set_xlabel('Improvement (%)', fontsize=12)
        ax.set_ylabel('LLM Model', fontsize=12)
        ax.set_title('Performance Improvement with Optimized Prompts', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        improvement_path = self.results_dir / "improvement_chart.png"
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Improvement chart saved to {improvement_path}")

    def create_detailed_comparison_plot(self, results_summary: Dict):
        """Create a detailed comparison plot with multiple metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        models = list(results_summary["llm_models"].keys())

        # Plot 1: Accuracy comparison (line plot)
        ax1 = axes[0, 0]
        baseline_acc = [results_summary["llm_models"][m]['baseline_accuracy'] for m in models]
        optimized_acc = [results_summary["llm_models"][m]['optimized_accuracy'] for m in models]

        x_pos = range(len(models))
        ax1.plot(x_pos, baseline_acc, marker='o', label='Baseline', linewidth=2, markersize=8)
        ax1.plot(x_pos, optimized_acc, marker='s', label='Optimized', linewidth=2, markersize=8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Correct answers comparison
        ax2 = axes[0, 1]
        baseline_correct = [results_summary["llm_models"][m]['baseline_correct'] for m in models]
        optimized_correct = [results_summary["llm_models"][m]['optimized_correct'] for m in models]
        total_questions = [results_summary["llm_models"][m]['total_questions'] for m in models]

        width = 0.35
        x = range(len(models))
        ax2.bar([i - width/2 for i in x], baseline_correct, width, label='Baseline', alpha=0.8)
        ax2.bar([i + width/2 for i in x], optimized_correct, width, label='Optimized', alpha=0.8)

        # Add total questions line
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, total_questions, color='red', marker='D', linestyle='--',
                     label='Total Questions', linewidth=2)

        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Correct Answers')
        ax2_twin.set_ylabel('Total Questions', color='red')
        ax2.set_title('Correct Answers Comparison')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Improvement distribution
        ax3 = axes[1, 0]
        improvements = [results_summary["llm_models"][m]['improvement'] for m in models]
        colors = ['green' if imp >= 0 else 'red' for imp in improvements]
        ax3.bar(models, improvements, color=colors, alpha=0.7)
        ax3.set_xlabel('LLM Model')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Improvement Distribution')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Calculate summary statistics
        avg_baseline = sum(baseline_acc) / len(baseline_acc)
        avg_optimized = sum(optimized_acc) / len(optimized_acc)
        avg_improvement = sum(improvements) / len(improvements)
        max_improvement_model = models[improvements.index(max(improvements))]
        max_improvement_value = max(improvements)

        summary_text = f"""
        Summary Statistics
        {'='*30}

        Average Baseline Accuracy:  {avg_baseline:.2f}%
        Average Optimized Accuracy: {avg_optimized:.2f}%
        Average Improvement:        {avg_improvement:+.2f}%

        Best Improvement:
        Model: {max_improvement_model}
        Improvement: {max_improvement_value:+.2f}%

        Total Models Tested: {len(models)}
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')

        plt.suptitle('MMLU Benchmark Results - Detailed Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        detailed_path = self.results_dir / "detailed_comparison_plot.png"
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Detailed comparison plot saved to {detailed_path}")

    def generate_all_visualizations(self, results_summary: Dict):
        """Generate all visualizations and tables."""
        print("\n" + "="*60)
        print("STEP 5: Generating Visualizations and Tables")
        print("="*60)

        # Create comparison table
        print("\n1. Creating comparison table...")
        table = self.create_comparison_table(results_summary)
        print("\nResults Comparison Table:")
        print(table)

        # Create accuracy comparison chart
        print("\n2. Creating accuracy comparison chart...")
        self.create_accuracy_comparison_chart(results_summary)

        # Create improvement chart
        print("\n3. Creating improvement chart...")
        self.create_improvement_chart(results_summary)

        # Create detailed comparison plot
        print("\n4. Creating detailed comparison plot...")
        self.create_detailed_comparison_plot(results_summary)

        print("\n" + "="*60)
        print("All visualizations have been generated successfully!")
        print(f"Check the '{self.results_dir}' directory for saved files.")
        print("="*60)
