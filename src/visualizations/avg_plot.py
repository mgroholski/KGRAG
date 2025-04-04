import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['Precision', 'Recall', 'F1 Score']
values = [0.8503, 0.8636, 0.8565]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar chart
bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#f39c12'], width=0.6)

# Add a horizontal line at y=1.0 to represent perfect score
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)

# Add text labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Customize the plot
ax.set_ylim(0, 1.1)  # Set y-axis limits with some padding
ax.set_title('Model Performance Metrics', fontsize=15, fontweight='bold')
ax.set_ylabel('Score (higher is better)', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a subtle background color
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#ffffff')

plt.tight_layout()
plt.savefig('./output/bertscore_summary_table.png', bbox_inches='tight')
