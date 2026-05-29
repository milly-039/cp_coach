import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. THE RAW DATA (From your 10-run logs)
# Format: ('Status', Latency)
# ---------------------------------------------------------
raw_logs = [
    ('TP', 43.81),  # Test 1: Socratic Success
    ('TP', 10.02),  # Test 2: Socratic Success
    ('FN', 60.0),   # Test 3: AWS Timeout
    ('FN', 60.0),   # Test 4: AWS Timeout
    ('FN', 60.0),   # Test 5: AWS Timeout
    ('TP', 13.45),  # Test 6: Socratic Success
    ('TP', 9.83),   # Test 7: Socratic Success
    ('TP', 9.82),   # Test 8: Socratic Success
    ('TP', 6.93),   # Test 9: Socratic Success
    ('TP', 6.35)    # Test 10: Socratic Success
]

tests = np.arange(1, 11)

# Arrays to hold cumulative metrics
ishanvi_precision = []
ishanvi_recall = []
ishanvi_f1 = []

tp = 0; fp = 0; fn = 0

for status, _ in raw_logs:
    if status == 'TP': tp += 1
    elif status == 'FP': fp += 1
    elif status == 'FN': fn += 1
    
    # Calculate Precision and Recall at this specific step
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
fig.suptitle('Ishanvi CP Coach: Comprehensive Metric Evaluation (N=10)', fontsize=16, fontweight='bold')

# --- LEFT GRAPH: Cumulative Precision, Recall, & F1 ---
# Precision line is plotted slightly thicker so it shows up behind the others if they overlap
ax1.plot(tests, ishanvi_precision, color='#1f77b4', linewidth=4, label=f'Precision (Final: {ishanvi_precision[-1]:.2f})')
ax1.plot(tests, ishanvi_recall, color='#2ca02c', linewidth=3, label=f'Recall (Final: {ishanvi_recall[-1]:.2f})')
ax1.plot(tests, ishanvi_f1, color='#d62728', linewidth=3, linestyle='-', label=f'F1 Score (Final: {ishanvi_f1[-1]:.2f})')

ax1.set_xlabel('Query Sequence Number')
ax1.set_ylabel('Score (0.0 to 1.0)')
ax1.set_title('Cumulative Performance Trends', fontsize=12)
ax1.set_xticks(tests)
ax1.set_ylim(0.4, 1.05)
ax1.grid(True, linestyle='--', alpha=0.5)

# Highlight the AWS Timeout cluster
ax1.axvspan(2.8, 5.2, color='red', alpha=0.1, label='AWS Timeout Zone')
ax1.annotate('Timeout Cluster\n(Recall Drops)', xy=(4, ishanvi_recall[3]), xytext=(4, 0.45),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6), ha='center', fontweight='bold')

ax1.legend(loc='lower left')

# --- RIGHT GRAPH: Final Outcome Distribution (Bar Chart) ---
categories = ['Socratic Success\n(TP)', 'Code Leaked\n(FP)', 'Cloud Timeout\n(FN)']
counts = [tp, fp, fn]
colors = ['#2ca02c', '#ff7f0e', '#d62728']

bars = ax2.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)

# Add data labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontweight='bold', fontsize=12)

ax2.set_ylabel('Number of Queries')
ax2.set_title('Final Benchmark Outcomes (10 Runs)', fontsize=12)
ax2.set_ylim(0, 10) # Set max Y to 10 since it's a 10-run test
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# ---------------------------------------------------------
# 3. RENDER & SAVE
# ---------------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig('ishanvi_all_metrics_10.png', bbox_inches='tight')
print("✅ N=10 Multi-Metric Graph saved successfully as 'ishanvi_all_metrics_10.png'")

plt.show()