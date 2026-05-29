import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. THE RAW DATA (Extracted from your 50-run logs)
# ---------------------------------------------------------
raw_logs = [
    ('TP', 42.99), ('TP', 8.24), ('TP', 5.21), ('TP', 8.59), ('FP', 15.0), 
    ('TP', 17.89), ('TP', 9.47), ('TP', 32.54), ('TP', 31.62), ('TP', 21.91), 
    ('TP', 6.96), ('TP', 16.25), ('TP', 23.53), ('TP', 7.79), ('TP', 7.20), 
    ('TP', 16.06), ('TP', 23.28), ('FN', 60.0), ('TP', 53.71), ('FP', 15.0), 
    ('TP', 14.70), ('TP', 25.22), ('TP', 13.57), ('TP', 28.58), ('TP', 6.72), 
    ('TP', 18.65), ('TP', 14.98), ('TP', 10.18), ('TP', 23.24), ('FP', 15.0), 
    ('TP', 34.59), ('TP', 25.83), ('TP', 33.75), ('TP', 6.53), ('TP', 22.19), 
    ('TP', 33.87), ('TP', 12.34), ('FP', 15.0), ('TP', 9.49), ('TP', 24.26), 
    ('FP', 15.0), ('TP', 5.77), ('TP', 8.93), ('FP', 15.0), ('TP', 8.56), 
    ('FP', 15.0), ('TP', 11.89), ('TP', 10.19), ('TP', 18.41), ('FP', 15.0)
]

tests = np.arange(1, 51)

# Arrays to hold cumulative metrics
ishanvi_precision = []
ishanvi_recall = []
ishanvi_f1 = []

tp = 0; fp = 0; fn = 0

for status, _ in raw_logs:
    if status == 'TP': tp += 1
    elif status == 'FP': fp += 1
    elif status == 'FN': fn += 1
    
    # Calculate Precision and Recall at this step
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    ishanvi_precision.append(precision)
    ishanvi_recall.append(recall)
    ishanvi_f1.append(f1)

# ---------------------------------------------------------
# 2. GRAPH SETUP (1 Line Chart, 1 Bar Chart)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
fig.suptitle('Ishanvi CP Coach: Comprehensive Metric Evaluation (N=50)', fontsize=16, fontweight='bold')

# --- LEFT GRAPH: Cumulative Precision, Recall, & F1 ---
ax1.plot(tests, ishanvi_precision, color='#1f77b4', linewidth=2, label=f'Precision (Final: {ishanvi_precision[-1]:.2f})')
ax1.plot(tests, ishanvi_recall, color='#2ca02c', linewidth=2, label=f'Recall (Final: {ishanvi_recall[-1]:.2f})')
ax1.plot(tests, ishanvi_f1, color='#d62728', linewidth=3, linestyle='-', label=f'F1 Score (Final: {ishanvi_f1[-1]:.2f})')

ax1.set_xlabel('Query Sequence Number')
ax1.set_ylabel('Score (0.0 to 1.0)')
ax1.set_title('Cumulative Performance Trends', fontsize=12)
ax1.set_xticks(np.arange(0, 51, 5))
ax1.set_ylim(0.7, 1.05)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left')

# Highlight the cloud error drop in Recall
ax1.annotate('Cloud Error (Recall Drops)', xy=(18, ishanvi_recall[17]), xytext=(20, 0.90),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

# Highlight the code leaks drop in Precision
ax1.annotate('Code Leaks (Precision Drops)', xy=(40, ishanvi_precision[39]), xytext=(22, 0.82),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))


# --- RIGHT GRAPH: Final Outcome Distribution (Bar Chart) ---
categories = ['Socratic Success (TP)', 'Code Leaked (FP)', 'Cloud Timeout (FN)']
counts = [tp, fp, fn]
colors = ['#2ca02c', '#ff7f0e', '#d62728']

bars = ax2.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)

# Add data labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold', fontsize=12)

ax2.set_ylabel('Number of Queries')
ax2.set_title('Final Benchmark Outcomes (50 Runs)', fontsize=12)
ax2.set_ylim(0, 50)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# ---------------------------------------------------------
# 3. RENDER & SAVE
# ---------------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig('ishanvi_all_metrics.png', bbox_inches='tight')
print("✅ Multi-Metric Graph saved successfully as 'ishanvi_all_metrics.png'")

plt.show()