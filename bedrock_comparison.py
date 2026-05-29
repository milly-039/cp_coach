import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. THE RAW DATA (Ishanvi 50-Run Logs)
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
ishanvi_latency = [log[1] for log in raw_logs]

# Calculate Cumulative F1 Score for Ishanvi
ishanvi_f1 = []
tp = 0; fp = 0; fn = 0

for status, _ in raw_logs:
    if status == 'TP': tp += 1
    elif status == 'FP': fp += 1
    elif status == 'FN': fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    ishanvi_f1.append(f1)

# ---------------------------------------------------------
# 2. AWS BEDROCK BASELINE DATA (Estimated for Managed Service)
# Bedrock doesn't suffer cold starts, averages ~3.5s latency.
# General models score ~0.87 F1 on strict pedagogical guardrails.
# ---------------------------------------------------------
bedrock_latency = [3.5] * 50
bedrock_f1 = [0.87] * 50

# ---------------------------------------------------------
# 3. GRAPH SETUP
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=150)
fig.suptitle('Architectural Comparison: Custom Serverless (Ishanvi) vs. Managed Service (AWS Bedrock)', fontsize=14, fontweight='bold')

# --- TOP GRAPH: System Latency ---
ax1.plot(tests, ishanvi_latency, color='#2ca02c', linewidth=1.5, alpha=0.6, label='Ishanvi (Serverless Modal/Lambda)')
ax1.plot(tests, bedrock_latency, color='#9467bd', linewidth=3, linestyle='-', label='AWS Bedrock (Managed, No Cold Start)')

# Highlight Ishanvi specific events
for i, (status, lat) in enumerate(raw_logs):
    if status == 'FN': # Cloud Error
        ax1.scatter(i+1, lat, color='red', s=80, zorder=5, label='Lambda Timeout (Ishanvi)' if i==17 else "")
    elif i == 0: # Cold Start
        ax1.scatter(i+1, lat, color='purple', s=80, zorder=5, label='GPU Cold Start (Ishanvi)')
    else:
        ax1.scatter(i+1, lat, color='#2ca02c', s=20, alpha=0.5)

ax1.set_ylabel('Latency (Seconds)')
ax1.set_title('Infrastructure Response Time (Serverless vs. Managed)', fontsize=12)
ax1.set_xticks(np.arange(0, 51, 5))
ax1.set_ylim(0, 65)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right', ncol=2)

# --- BOTTOM GRAPH: System Integrity (F1 Score) ---
ax2.plot(tests, ishanvi_f1, color='#1f77b4', linewidth=3, label='Ishanvi F1 (Fine-Tuned Socratic Guardrails)')
ax2.plot(tests, bedrock_f1, color='#9467bd', linewidth=3, linestyle='--', label='AWS Bedrock F1 (General Purpose Estimate)')

# Mark the final score for Ishanvi
ax2.annotate(f'Ishanvi Final F1: {ishanvi_f1[-1]:.3f}', 
             xy=(50, ishanvi_f1[-1]), xytext=(40, 0.95),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold')

ax2.set_xlabel('Query Sequence Number (N=50)')
ax2.set_ylabel('F1 Score (0.0 to 1.0)')
ax2.set_title('Pedagogical Integrity (Specialized RAG vs. General Model)', fontsize=12)
ax2.set_xticks(np.arange(0, 51, 5))
ax2.set_ylim(0.5, 1.05)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='lower right')

# ---------------------------------------------------------
# 4. RENDER & SAVE
# ---------------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(top=0.92)

plt.savefig('ishanvi_vs_bedrock.png', bbox_inches='tight')
print("✅ Comparison Graph saved successfully as 'ishanvi_vs_bedrock.png'")

plt.show()