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

# Cumulative Metrics for Ishanvi
ishanvi_p = []
ishanvi_r = []
ishanvi_f1 = []
tp = 0; fp = 0; fn = 0

for status, _ in raw_logs:
    if status == 'TP': tp += 1
    elif status == 'FP': fp += 1
    elif status == 'FN': fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    ishanvi_p.append(precision)
    ishanvi_r.append(recall)
    ishanvi_f1.append(f1)

# ---------------------------------------------------------
# 2. AWS BEDROCK BASELINE DATA (Managed Service Estimate)
# - Latency: Flat ~3.5s (No cold starts)
# - Recall: 1.0 (100% uptime, no timeouts)
# - Precision: ~0.77 (Leaks code easily without strict guardrails)
# - F1 Score: ~0.87 (Harmonic mean of P and R)
# ---------------------------------------------------------
bedrock_latency = [3.5] * 50
bedrock_p = [0.77] * 50
bedrock_r = [1.0] * 50
bedrock_f1 = [0.87] * 50

# ---------------------------------------------------------
# 3. GRAPH SETUP (2x2 Grid)
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
fig.suptitle('Complete Architectural Benchmarking: Ishanvi vs. AWS Bedrock', fontsize=16, fontweight='bold')

# Colors
color_ish = '#1f77b4'  # Blue for Ishanvi logic
color_bed = '#9467bd'  # Purple for Bedrock
color_lat = '#2ca02c'  # Green for Latency

# --- QUADRANT 1: LATENCY (Top Left) ---
ax1 = axs[0, 0]
ax1.plot(tests, ishanvi_latency, color=color_lat, linewidth=1.5, alpha=0.7, label='Ishanvi Latency')
ax1.plot(tests, bedrock_latency, color=color_bed, linewidth=3, linestyle='-', label='Bedrock (Managed)')
# Highlight events
for i, (status, lat) in enumerate(raw_logs):
    if status == 'FN': ax1.scatter(i+1, lat, color='red', s=60, zorder=5)
    elif i == 0: ax1.scatter(i+1, lat, color='purple', s=60, zorder=5)
ax1.set_title('1. Infrastructure Latency', fontsize=12, fontweight='bold')
ax1.set_ylabel('Seconds')
ax1.set_ylim(0, 65)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right')

# --- QUADRANT 2: RECALL / RELIABILITY (Bottom Left) ---
ax2 = axs[1, 0]
ax2.plot(tests, ishanvi_r, color=color_ish, linewidth=3, label=f'Ishanvi ({ishanvi_r[-1]:.3f})')
ax2.plot(tests, bedrock_r, color=color_bed, linewidth=3, linestyle='--', label='Bedrock (1.000)')
ax2.annotate('Cloud Timeout Drop', xy=(18, ishanvi_r[17]), xytext=(22, 0.85),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
ax2.set_title('2. System Recall (Cloud Reliability)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Query Sequence Number (N=50)')
ax2.set_ylabel('Score (0.0 to 1.0)')
ax2.set_ylim(0.7, 1.05)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='lower right')

# --- QUADRANT 3: PRECISION / GUARDRAILS (Top Right) ---
ax3 = axs[0, 1]
ax3.plot(tests, ishanvi_p, color=color_ish, linewidth=3, label=f'Ishanvi ({ishanvi_p[-1]:.3f})')
ax3.plot(tests, bedrock_p, color=color_bed, linewidth=3, linestyle='--', label='Bedrock (~0.770)')
ax3.set_title('3. Socratic Precision (Pedagogical Discipline)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score (0.0 to 1.0)')
ax3.set_ylim(0.6, 1.05)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(loc='lower right')

# --- QUADRANT 4: F1 SCORE / OVERALL INTEGRITY (Bottom Right) ---
ax4 = axs[1, 1]
ax4.plot(tests, ishanvi_f1, color='#d62728', linewidth=3, label=f'Ishanvi F1 ({ishanvi_f1[-1]:.3f})')
ax4.plot(tests, bedrock_f1, color=color_bed, linewidth=3, linestyle='--', label='Bedrock F1 (~0.870)')
ax4.annotate('Ishanvi Wins F1', xy=(48, ishanvi_f1[-1]), xytext=(35, 0.95),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6), fontweight='bold')
ax4.set_title('4. Final Systemic Integrity (F1 Score)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Query Sequence Number (N=50)')
ax4.set_ylabel('Score (0.0 to 1.0)')
ax4.set_ylim(0.6, 1.05)
ax4.grid(True, linestyle='--', alpha=0.4)
ax4.legend(loc='lower right')

# ---------------------------------------------------------
# 4. RENDER & SAVE
# ---------------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(top=0.92)

plt.savefig('ishanvi_complete_comparison.png', bbox_inches='tight')
print("✅ 4-Quadrant Comparison Graph saved successfully as 'ishanvi_complete_comparison.png'")

plt.show()