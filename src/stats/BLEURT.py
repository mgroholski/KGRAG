import numpy as np
import matplotlib.pyplot as plt
from bleurt import score as bleurt_score
import os
import subprocess
import sys

class BLEURT:
    def __init__(self, logger):
        self.scores = []
        self.logger = logger
        self.model_path = "./models/BLEURT-20"
        self._check_model()

    def _check_model(self):
        """
        Check if the BLEURT model exists and download it if it doesn't.
        """
        try:
            if not os.path.exists(self.model_path):
                self.logger.log(f"BLEURT model not found at {self.model_path}. Downloading...")

                # Create the models directory if it doesn't exist
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

                # Use pip to install the BLEURT model
                cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
                ]

                self.logger.log("Running command: " + " ".join(cmd))
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    self.logger.log(f"Error downloading BLEURT model: {result.stderr}")
                    raise Exception("Failed to download BLEURT model")

                self.logger.log("BLEURT model downloaded successfully")
            else:
                self.logger.log(f"BLEURT model found at {self.model_path}")
        except Exception as e:
            self.logger.log(f"Error checking/downloading BLEURT model: {str(e)}")
            raise

    def score(self, predictions, truths):
        """
        Calculate BLEURT scores between predictions and reference texts.
        Failed generations (marked with __FAILED_GENERATION__) are scored as 0.

        Args:
            predictions (list): List of predicted text strings
            truths (list): List of reference text strings

        Returns:
            float: Average BLEURT score
        """
        try:
            self.logger.log("Computing BLEURT scores...")
            
            # Create processed lists without failed generations
            processed_predictions = []
            processed_truths = []
            original_indices = []
            
            self.scores = [0.0] * len(predictions)  # Initialize all scores to 0
            
            # First pass: identify failed generations and track valid pairs
            for i, (pred, ref) in enumerate(zip(predictions, truths)):
                if pred == "__FAILED_GENERATION__" or ref == "__FAILED_GENERATION__":
                    print(f"Assigned zero BLEURT score to failed generation at index {i}")
                    self.logger.log(f"Assigned zero BLEURT score to failed generation at index {i}")
                else:
                    processed_predictions.append(pred)
                    processed_truths.append(ref)
                    original_indices.append(i)
            
            # Score only the valid pairs
            if processed_predictions and processed_truths:
                scorer = bleurt_score.BleurtScorer(self.model_path)
                valid_scores = scorer.score(references=processed_truths, candidates=processed_predictions)
                
                # Map scores back to their original positions
                for idx, score in zip(original_indices, valid_scores):
                    self.scores[idx] = score
            
            avg_score = sum(self.scores) / len(self.scores) if self.scores else 0
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
