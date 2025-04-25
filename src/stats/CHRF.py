import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

class chrF:
    def __init__(self, logger, n_gram=6, beta=2):
        """
        Initialize chrF metric.

        Args:
            logger: Logger object for recording results
            n_gram: Maximum n-gram order (default: 6)
            beta: Weight of recall compared to precision (default: 2)
        """
        self.scores = []
        self.logger = logger
        self.n_gram = n_gram
        self.beta = beta

    def get_char_ngrams(self, text, n):
        """
        Extract character n-grams from text.

        Args:
            text (str): Input text
            n (int): n-gram size

        Returns:
            Counter: Dictionary of n-grams with their frequencies
        """
        return Counter([text[i:i+n] for i in range(len(text)-n+1)])

    def score(self, predictions, truths):
        """
        Calculate chrF scores between predictions and reference texts.
        Failed generations (marked with __FAILED_GENERATION__) are scored as 0.

        Args:
            predictions (list): List of predicted text strings
            truths (list): List of reference text strings

        Returns:
            float: Average chrF score
        """
        try:
            self.logger.log(f"Computing chrF scores (n_gram={self.n_gram}, beta={self.beta})...")
            scores = []

            for i, (pred, ref) in enumerate(zip(predictions, truths)):
                # Handle failed generations
                if pred == "__FAILED_GENERATION__" or ref == "__FAILED_GENERATION__":
                    scores.append(0.0)
                    print(f"Assigned zero chrF score to failed generation at index {i}")
                    self.logger.log(f"Assigned zero chrF score to failed generation at index {i}")
                    continue
                
                precision_scores = []
                recall_scores = []

                for n in range(1, self.n_gram + 1):
                    pred_ngrams = self.get_char_ngrams(pred, n)
                    ref_ngrams = self.get_char_ngrams(ref, n)

                    matched = sum((pred_ngrams & ref_ngrams).values())

                    precision = matched / sum(pred_ngrams.values()) if sum(pred_ngrams.values()) > 0 else 0
                    recall = matched / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0

                    precision_scores.append(precision)
                    recall_scores.append(recall)

                avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
                avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

                if avg_precision == 0 and avg_recall == 0:
                    chrf_score = 0
                else:
                    beta_squared = self.beta ** 2
                    chrf_score = (1 + beta_squared) * (avg_precision * avg_recall) / (beta_squared * avg_precision + avg_recall) if (beta_squared * avg_precision + avg_recall) > 0 else 0

                scores.append(chrf_score)

            self.scores = scores
            avg_score = sum(scores) / len(scores) if scores else 0
            self.logger.log(f"Average chrF score: {avg_score:.4f}")

            return avg_score
        except Exception as e:
            self.logger.log(f"Error in chrF scoring: {str(e)}")
            raise

    def plt(self, savepath):
        """
        Generate and save plots of chrF scores, similar to BLEURT style.

        Args:
            savepath (str): Path where the plots should be saved
        """
        try:
            if not self.scores:
                self.logger.log("No chrF scores available for plotting")
                return

            self.logger.log(f"Generating chrF score plots at {savepath}")

            os.makedirs(savepath, exist_ok=True)
            questions = [f"q{i+1}" for i in range(len(self.scores))]

            for i in range(0, len(questions), 50):
                plot_start = i
                plot_end = min(i + 50, len(questions))

                plot_questions = questions[plot_start:plot_end]
                plot_scores = self.scores[plot_start:plot_end]

                fig, ax = plt.subplots(figsize=(15, 8))
                ax.bar(plot_questions, plot_scores, color='green')
                ax.set_title(f'chrF Scores (Questions {plot_start+1}-{plot_end})')
                ax.set_xlabel('Questions')
                ax.set_ylabel('chrF Score')
                ax.set_ylim(0, 1)

                if len(plot_questions) > 20:
                    plt.setp(ax.get_xticklabels(), rotation=90)

                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'{savepath}/chrf_questions_{plot_start+1}_to_{plot_end}.png')
                plt.close()

                self.logger.log(f"Created plot for questions {plot_start+1} to {plot_end}")

            mean_score = np.mean(self.scores)
            min_score = np.min(self.scores)
            max_score = np.max(self.scores)
            std_score = np.std(self.scores)

            self.logger.log("\nOverall chrF Statistics:")
            self.logger.log(f"Average Score: {mean_score:.4f}")
            self.logger.log(f"Min Score: {min_score:.4f}")
            self.logger.log(f"Max Score: {max_score:.4f}")
            self.logger.log(f"Standard Deviation: {std_score:.4f}")

            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.scores, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(mean_score, color='red', linestyle='dashed', linewidth=2,
                      label=f'Mean: {mean_score:.4f}')
            ax.set_xlabel('chrF Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of chrF Scores')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'{savepath}/chrf_distribution.png')
            plt.close()

            # Create summary table
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            ax.axis('tight')

            table_data = [
                ['Metric', 'Mean', 'Min', 'Max', 'Std Dev'],
                ['chrF Score', f'{mean_score:.4f}', f'{min_score:.4f}', f'{max_score:.4f}', f'{std_score:.4f}']
            ]

            table = ax.table(cellText=table_data, loc='center', cellLoc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)

            for j in range(len(table_data[0])):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white', fontweight='bold')

            table[1, 0].set_facecolor('#D9E1F2')
            table[1, 0].set_text_props(fontweight='bold')

            plt.title('chrF Summary Statistics', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()

            plt.savefig(f'{savepath}/chrf_summary_table.png', bbox_inches='tight', dpi=300)
            plt.close()

            self.logger.log(f"chrF plots and statistics saved at {savepath}")

        except Exception as e:
            self.logger.log(f"Error generating chrF plots: {str(e)}")
            raise
