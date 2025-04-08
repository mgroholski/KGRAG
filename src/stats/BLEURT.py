import numpy as np
import matplotlib.pyplot as plt
from bleurt import score as bleurt_score
import os

class BLEURT:
    def __init__(self, logger):
        self.scores = []
        self.logger = logger

    def score(self, predictions, truths):
        """
        Calculate BLEURT scores between predictions and reference texts.

        Args:
            predictions (list): List of predicted text strings
            truths (list): List of reference text strings

        Returns:
            float: Average BLEURT score
        """
        try:
            self.logger.log("Computing BLEURT scores...")
            scorer = bleurt_score.BleurtScorer()
            scores = scorer.score(references=truths, candidates=predictions)
            self.scores = scores
            avg_score = sum(scores) / len(scores)
            self.logger.log(f"Average BLEURT score: {avg_score:.4f}")

            return avg_score
        except Exception as e:
            self.logger.log(f"Error in BLEURT scoring: {str(e)}")
            raise

    def plt(self, savepath):
        """
        Generate and save plots of BLEURT scores, similar to BERTScore style.

        Args:
            savepath (str): Path where the plots should be saved
        """
        try:
            if not self.scores:
                self.logger.log("No BLEURT scores available for plotting")
                return

            self.logger.log(f"Generating BLEURT score plots at {savepath}")

            os.makedirs(savepath, exist_ok=True)
            questions = [f"q{i+1}" for i in range(len(self.scores))]
            for i in range(0, len(questions), 50):
                plot_start = i
                plot_end = min(i + 50, len(questions))


                plot_questions = questions[plot_start:plot_end]
                plot_scores = self.scores[plot_start:plot_end]


                fig, ax = plt.subplots(figsize=(15, 8))
                ax.bar(plot_questions, plot_scores, color='blue')
                ax.set_title(f'BLEURT Scores (Questions {plot_start+1}-{plot_end})')
                ax.set_xlabel('Questions')
                ax.set_ylabel('BLEURT Score')
                ax.set_ylim(-1, 1)


                if len(plot_questions) > 20:
                    plt.setp(ax.get_xticklabels(), rotation=90)

                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'{savepath}/bleurt_questions_{plot_start+1}_to_{plot_end}.png')
                plt.close()

                self.logger.log(f"Created plot for questions {plot_start+1} to {plot_end}")


            mean_score = np.mean(self.scores)
            min_score = np.min(self.scores)
            max_score = np.max(self.scores)
            std_score = np.std(self.scores)

            self.logger.log("\nOverall BLEURT Statistics:")
            self.logger.log(f"Average Score: {mean_score:.4f}")
            self.logger.log(f"Min Score: {min_score:.4f}")
            self.logger.log(f"Max Score: {max_score:.4f}")
            self.logger.log(f"Standard Deviation: {std_score:.4f}")


            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(mean_score, color='red', linestyle='dashed', linewidth=2,
                      label=f'Mean: {mean_score:.4f}')
            ax.set_xlabel('BLEURT Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of BLEURT Scores')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'{savepath}/bleurt_distribution.png')
            plt.close()


            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            ax.axis('tight')


            table_data = [
                ['Metric', 'Mean', 'Min', 'Max', 'Std Dev'],
                ['BLEURT Score', f'{mean_score:.4f}', f'{min_score:.4f}', f'{max_score:.4f}', f'{std_score:.4f}']
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

            plt.title('BLEURT Summary Statistics', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()


            plt.savefig(f'{savepath}/bleurt_summary_table.png', bbox_inches='tight', dpi=300)
            plt.close()

            self.logger.log(f"BLEURT plots and statistics saved at {savepath}")

        except Exception as e:
            self.logger.log(f"Error generating BLEURT plots: {str(e)}")
            raise
