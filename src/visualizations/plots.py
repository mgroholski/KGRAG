import matplotlib.pyplot as plt
import os
import numpy as np

def BERTScore_plts(results, savepath):
    # Create directories if they don't exist
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print(f"Created directory: {savepath}")

    precision_scores = [score['bert_precision'] for score in results]
    recall_scores = [score['bert_recall'] for score in results]
    f1_scores = [score['bert_f1'] for score in results]

    # Generate question labels (q1, q2, q3, ...)
    questions = [f"q{i+1}" for i in range(len(results))]

    # Create plots for every 50 questions
    for i in range(0, len(questions), 50):
        plot_start = i
        plot_end = min(i + 50, len(questions))

        # Get the relevant slice of data for this plot
        plot_questions = questions[plot_start:plot_end]
        plot_precision = precision_scores[plot_start:plot_end]
        plot_recall = recall_scores[plot_start:plot_end]
        plot_f1 = f1_scores[plot_start:plot_end]

        # Create a figure with three subplots for precision, recall, and F1
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Plot precision
        ax1.bar(plot_questions, plot_precision, color='blue')
        ax1.set_title(f'BERTScore Precision (Questions {plot_start+1}-{plot_end})')
        ax1.set_xlabel('Questions')
        ax1.set_ylabel('Precision Score')
        ax1.set_ylim(0, 1)

        # Plot recall
        ax2.bar(plot_questions, plot_recall, color='green')
        ax2.set_title(f'BERTScore Recall (Questions {plot_start+1}-{plot_end})')
        ax2.set_xlabel('Questions')
        ax2.set_ylabel('Recall Score')
        ax2.set_ylim(0, 1)

        # Plot F1
        ax3.bar(plot_questions, plot_f1, color='red')
        ax3.set_title(f'BERTScore F1 (Questions {plot_start+1}-{plot_end})')
        ax3.set_xlabel('Questions')
        ax3.set_ylabel('F1 Score')
        ax3.set_ylim(0, 1)

        # Rotate x-axis labels if there are many questions
        for ax in [ax1, ax2, ax3]:
            if len(plot_questions) > 20:
                plt.setp(ax.get_xticklabels(), rotation=90)

        plt.tight_layout()
        plt.savefig(f'{savepath}/bertscore_questions_{plot_start+1}_to_{plot_end}.png')
        print(f"Created plot for questions {plot_start+1} to {plot_end}")
        plt.close()

    # Calculate statistics
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    min_precision = np.min(precision_scores) if precision_scores else 0
    min_recall = np.min(recall_scores) if recall_scores else 0
    min_f1 = np.min(f1_scores) if f1_scores else 0

    max_precision = np.max(precision_scores) if precision_scores else 0
    max_recall = np.max(recall_scores) if recall_scores else 0
    max_f1 = np.max(f1_scores) if f1_scores else 0

    std_precision = np.std(precision_scores) if precision_scores else 0
    std_recall = np.std(recall_scores) if recall_scores else 0
    std_f1 = np.std(f1_scores) if f1_scores else 0

    print("\nOverall BERTScore Statistics:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    # Create a summary table figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')

    # Create data for the table
    table_data = [
        ['Metric', 'Mean', 'Min', 'Max', 'Std Dev'],
        ['Precision', f'{avg_precision:.4f}', f'{min_precision:.4f}', f'{max_precision:.4f}', f'{std_precision:.4f}'],
        ['Recall', f'{avg_recall:.4f}', f'{min_recall:.4f}', f'{max_recall:.4f}', f'{std_recall:.4f}'],
        ['F1 Score', f'{avg_f1:.4f}', f'{min_f1:.4f}', f'{max_f1:.4f}', f'{std_f1:.4f}']
    ]

    # Create the table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')

    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Style the header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style the metric column
    for i in range(1, len(table_data)):
        table[i, 0].set_facecolor('#D9E1F2')
        table[i, 0].set_text_props(fontweight='bold')

    plt.title('BERTScore Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save the table
    plt.savefig(f'{savepath}/bertscore_summary_table.png', bbox_inches='tight', dpi=300)
    print(f"Created summary statistics table at {savepath}/bertscore_summary_table.png")
    plt.close()
