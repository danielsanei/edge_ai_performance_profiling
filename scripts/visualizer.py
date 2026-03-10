import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

# Setup styles for academic reporting
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 14})

DATA_DIR = "../data"
PLOT_DIR = "../data/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# CHANGE: explicit ordering by parameter count — used everywhere to enforce consistent layout
model_order = ['Llama-1B', 'Phi-4-mini', 'Mistral-7B', 'Llama-8B']
quant_order  = ['Q4_K_M', 'Q8_0']
model_full_order = [f"{m} ({q})" for m in model_order for q in quant_order]

def reorder_pivot(pivot, order, col='Model_Full'):
    """Reindex a pivot dataframe's rows to match the desired model order."""
    available = [m for m in order if m in pivot[col].values]
    return pivot.set_index(col).reindex(available).reset_index()

def analyze_results():
    # 1. Load Master Results
    # CHANGE: Explicitly specify all 18 column names — pipeline writes 18 values per row
    # but the CSV header only declared the original 12; the 6 hardware summary columns
    # (added later to the pipeline) were never included in the header row.
    col_names = [
        'Model', 'Quant', 'Context', 'Threads', 'Prompt_ID',
        'TPS', 'PPL', 'Total_Time', 'TTFT', 'Load_Duration',
        'Prompt_Tokens', 'Prompt_Eval_Rate',
        'Avg_Watts', 'Peak_Temp', 'Avg_CPU', 'Peak_RAM', 'Avg_Clock', 'Throttled'
    ]
    master_df = pd.read_csv(f"{DATA_DIR}/master_results.csv", names=col_names, header=0)

    # CHANGE: Shorten model names for cleaner plot labels throughout
    # NEW: Added Meta-Llama-3.1-8B mapping
    label_map = {
        'microsoft_Phi-4-mini-instruct': 'Phi-4-mini',
        'Llama-3.2-1B': 'Llama-1B',
        'Mistral-7B-v0.3': 'Mistral-7B',
        'Meta-Llama-3.1-8B': 'Llama-8B'
    }
    master_df['Model_Short'] = master_df['Model'].map(label_map).fillna(master_df['Model'])
    master_df['Model_Full']  = master_df['Model_Short'] + " (" + master_df['Quant'] + ")"

    # CHANGE: enforce parameter-size ordering on label columns so all plots are consistent
    master_df['Model_Short'] = pd.Categorical(master_df['Model_Short'], categories=model_order,      ordered=True)
    master_df['Model_Full']  = pd.Categorical(master_df['Model_Full'],  categories=model_full_order, ordered=True)

    # 2. Compute Averages per Model/Quant/Context
    # CHANGE: Added hardware metric columns to aggregation
    summary = master_df.groupby(
        ['Model', 'Model_Short', 'Quant', 'Context', 'Threads'], observed=True
    ).agg({
        'TPS': 'mean',
        'PPL': 'mean',
        'TTFT': 'mean',
        'Load_Duration': 'mean',
        'Prompt_Eval_Rate': 'mean',
        'Avg_Watts': 'mean',
        'Peak_Temp': 'mean',
        'Avg_CPU': 'mean',
        'Peak_RAM': 'mean',
        'Avg_Clock': 'mean',
        'Throttled': 'max'
    }).reset_index()

    summary['Model_Full'] = summary['Model_Short'].astype(str) + " (" + summary['Quant'] + ")"
    summary['Model_Full'] = pd.Categorical(summary['Model_Full'], categories=model_full_order, ordered=True)

    # CHANGE: sort summary by the explicit model size order for consistent downstream use
    summary = summary.sort_values(['Model_Short', 'Quant', 'Context', 'Threads']).reset_index(drop=True)

    print("\n" + "="*80)
    print(f"{'MODEL SUMMARY STATISTICS':^80}")
    print("="*80)
    print(summary.to_string(index=False))

    # --- PLOT 1: The Pareto Frontier (PPL vs TPS) ---
    # Average over prompts and threads; one point per model+quant+context combination
    pareto_df = summary.groupby(['Model_Short', 'Quant', 'Context'], observed=True).agg({
        'TPS': 'mean', 'PPL': 'mean'
    }).reset_index()
    pareto_df['Label'] = pareto_df['Model_Short'].astype(str) + " (" + pareto_df['Quant'] + ")"

    # CHANGE: hue_order follows parameter-size ordering
    label_order = [l for l in model_full_order if l in pareto_df['Label'].values]

    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=pareto_df, x='TPS', y='PPL', hue='Label', style='Quant',
                    s=180, palette='tab10', hue_order=label_order)

    for _, row in pareto_df.iterrows():
        plt.text(row['TPS'] + 0.05, row['PPL'], f"CTX={int(row['Context'])}", fontsize=7.5, alpha=0.85)

    plt.title("Pareto Frontier: Generative Fidelity vs. Throughput")
    plt.xlabel("Throughput (Tokens/sec) — Higher is Better")
    plt.ylabel("Perplexity (PPL) — Lower is Better")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pareto_frontier.png", dpi=300)
    plt.close()

    # --- PLOT 2 (NEW): TPS vs Context Window — reveals the performance cliff ---
    # Averaged over prompts; one line per model+quant combination; threads shown separately
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, thr in zip(axes, [2, 4]):
        tps_ctx = summary[summary['Threads'] == thr].groupby(
            ['Model_Short', 'Quant', 'Context'], observed=True
        )['TPS'].mean().reset_index()

        # CHANGE: iterate in explicit parameter-size order instead of pandas default
        for model in model_order:
            for quant in quant_order:
                grp = tps_ctx[(tps_ctx['Model_Short'] == model) & (tps_ctx['Quant'] == quant)]
                if grp.empty:
                    continue
                grp = grp.sort_values('Context')
                linestyle = '--' if quant == 'Q8_0' else '-'
                ax.plot(grp['Context'], grp['TPS'], marker='o', linestyle=linestyle,
                        label=f"{model} ({quant})", linewidth=2)

        ax.set_title(f"TPS vs. Context Window — {thr} Threads")
        ax.set_xlabel("Context Window Size (tokens)")
        ax.set_ylabel("Tokens per Second (TPS)")
        ax.set_xticks([512, 2048, 4096])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle("Throughput Degradation Across Context Window Sizes", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/tps_vs_context.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 3 (NEW): Thread Count Impact on TPS (2T vs 4T) ---
    # Memory-bandwidth-bound workloads should show diminishing returns; verify empirically.
    # Fixed to CTX=512 for the most complete dataset across all models.
    thread_df = summary[summary['Context'] == 512].groupby(
        ['Model_Full', 'Threads'], observed=True
    )['TPS'].mean().reset_index()
    thread_pivot = thread_df.pivot(index='Model_Full', columns='Threads', values='TPS').reset_index()

    # CHANGE: reorder rows by parameter size
    thread_pivot = reorder_pivot(thread_pivot, model_full_order)

    x = np.arange(len(thread_pivot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width/2, thread_pivot[2], width, label='2 Threads', color='steelblue')
    ax.bar(x + width/2, thread_pivot[4], width, label='4 Threads', color='coral')

    # annotate percent change
    for i, (v2, v4) in enumerate(zip(thread_pivot[2], thread_pivot[4])):
        if pd.isna(v2) or pd.isna(v4):
            continue
        pct = (v4 - v2) / v2 * 100
        sign = '+' if pct >= 0 else ''
        ax.text(i, max(v2, v4) + 0.1, f"{sign}{pct:.1f}%", ha='center', fontsize=8, color='dimgray')

    ax.set_xticks(x)
    ax.set_xticklabels(thread_pivot['Model_Full'], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Avg Tokens per Second")
    ax.set_title("Thread Count Impact on TPS (CTX=512)\nAnnotated with % Change 2T → 4T")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/thread_impact_tps.png", dpi=300)
    plt.close()

    # --- PLOT 4 (NEW): Q4 vs Q8 — Speed and Fidelity Trade-off (CTX=512, THR=2 baseline) ---
    # Side-by-side bars for TPS and PPL comparing Q4_K_M vs Q8_0 at baseline config.
    q_df = summary[(summary['Context'] == 512) & (summary['Threads'] == 2)].groupby(
        ['Model_Short', 'Quant'], observed=True
    ).agg({'TPS': 'mean', 'PPL': 'mean'}).reset_index()

    fig, (ax_tps, ax_ppl) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, ylabel, title_suffix in [
        (ax_tps, 'TPS', 'Tokens per Second', 'Throughput — Higher is Better'),
        (ax_ppl, 'PPL', 'Perplexity', 'Fidelity — Lower is Better')
    ]:
        pivot = q_df.pivot(index='Model_Short', columns='Quant', values=metric).reset_index()

        # CHANGE: reorder rows by parameter size
        available = [m for m in model_order if m in pivot['Model_Short'].values]
        pivot = pivot.set_index('Model_Short').reindex(available).reset_index()

        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w/2, pivot['Q4_K_M'], w, label='Q4_K_M', color='mediumseagreen')
        ax.bar(x + w/2, pivot['Q8_0'],   w, label='Q8_0',   color='mediumpurple')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot['Model_Short'], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Q4 vs Q8: {title_suffix}")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.suptitle("Quantization Trade-off: Q4_K_M vs Q8_0 (CTX=512, THR=2)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/quant_comparison.png", dpi=300)
    plt.close()

    # --- PLOT 5 (NEW): TTFT vs Context Window ---
    # Prefill cost scales with context; shows how KV cache setup time grows.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, thr in zip(axes, [2, 4]):
        ttft_ctx = summary[summary['Threads'] == thr].groupby(
            ['Model_Short', 'Quant', 'Context'], observed=True
        )['TTFT'].mean().reset_index()

        # CHANGE: iterate in explicit parameter-size order
        for model in model_order:
            for quant in quant_order:
                grp = ttft_ctx[(ttft_ctx['Model_Short'] == model) & (ttft_ctx['Quant'] == quant)]
                if grp.empty:
                    continue
                grp = grp.sort_values('Context')
                linestyle = '--' if quant == 'Q8_0' else '-'
                ax.plot(grp['Context'], grp['TTFT'], marker='s', linestyle=linestyle,
                        label=f"{model} ({quant})", linewidth=2)

        ax.set_title(f"TTFT vs. Context Window — {thr} Threads")
        ax.set_xlabel("Context Window Size (tokens)")
        ax.set_ylabel("Time to First Token (seconds)")
        ax.set_xticks([512, 2048, 4096])
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle("Prefill Latency (TTFT) Across Context Window Sizes", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/ttft_vs_context.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 6: Hardware Stress Test — Time-series Thermal & Power Profile ---
    # Pick the heaviest completed model run: Mistral-7B Q8_0 at CTX=512, T2, Prompt 1
    # CHANGE: check individual/ subfolder first (pipeline2 writes there), fall back to data root
    target_csv = glob.glob(f"{DATA_DIR}/individual/Mistral-7B-v0.3-Q8_0_C512_T2_P1.csv") or \
                 glob.glob(f"{DATA_DIR}/Mistral-7B-v0.3-Q8_0_C512_T2_P1.csv")
    if target_csv:
        telemetry_df = pd.read_csv(target_csv[0])
        telemetry_df['Relative_Time'] = telemetry_df['timestamp'] - telemetry_df['timestamp'].iloc[0]

        fig, ax1 = plt.subplots(figsize=(11, 5))

        # temperature axis
        ax1.set_xlabel('Time into Inference (s)')
        ax1.set_ylabel('Temperature (°C)', color='tab:red')
        ax1.plot(telemetry_df['Relative_Time'], telemetry_df['temp'],
                 color='tab:red', label='Temp (°C)', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.axhline(y=80, color='r', linestyle='--', label='Throttling Limit (80°C)', alpha=0.7)

        # power axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Power Draw (Watts)', color='tab:blue')
        ax2.plot(telemetry_df['Relative_Time'], telemetry_df['watts'],
                 color='tab:blue', label='Power (W)', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # NEW: add RAM on a third visual layer (right-side annotation style)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('RAM Used (GB)', color='tab:green')
        ax3.plot(telemetry_df['Relative_Time'], telemetry_df['ram'],
                 color='tab:green', label='RAM (GB)', alpha=0.6, linestyle=':')
        ax3.tick_params(axis='y', labelcolor='tab:green')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                   loc='lower right', fontsize=9)

        plt.title("Hardware Profile: Mistral-7B (Q8_0, CTX=512, THR=2) — Thermal, Power & RAM")
        fig.tight_layout()
        plt.savefig(f"{PLOT_DIR}/hardware_stress_test.png", dpi=300)
        plt.close()

    # --- PLOT 6b (NEW): Hardware Profile for Meta-Llama-3.1-8B ---
    # The 8B model is the only config that crosses the 80C throttling threshold during
    # inference; this profile captures the thermal ceiling event as a standalone finding.
    llama8b_csv = glob.glob(f"{DATA_DIR}/individual/Meta-Llama-3.1-8B-Q4_K_M_C512_T4_P3.csv")
    if llama8b_csv:
        tel8b = pd.read_csv(llama8b_csv[0])
        tel8b['Relative_Time'] = tel8b['timestamp'] - tel8b['timestamp'].iloc[0]

        fig, ax1 = plt.subplots(figsize=(11, 5))

        ax1.set_xlabel('Time into Inference (s)')
        ax1.set_ylabel('Temperature (°C)', color='tab:red')
        ax1.plot(tel8b['Relative_Time'], tel8b['temp'], color='tab:red', label='Temp (°C)', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.axhline(y=80, color='r', linestyle='--', label='Throttling Limit (80°C)', alpha=0.7)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Power Draw (Watts)', color='tab:blue')
        ax2.plot(tel8b['Relative_Time'], tel8b['watts'], color='tab:blue', label='Power (W)', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('RAM Used (GB)', color='tab:green')
        ax3.plot(tel8b['Relative_Time'], tel8b['ram'], color='tab:green', label='RAM (GB)', alpha=0.6, linestyle=':')
        ax3.tick_params(axis='y', labelcolor='tab:green')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower right', fontsize=9)

        plt.title("Hardware Profile: Llama-3.1-8B (Q4_K_M, CTX=512, THR=4) — Thermal Throttling Event")
        fig.tight_layout()
        plt.savefig(f"{PLOT_DIR}/hardware_profile_llama8b.png", dpi=300)
        plt.close()

    # --- PLOT 7 (NEW): Average Power Draw per Model Configuration ---
    # From master_results Avg_Watts — shows energy cost of each model/quant combination.
    hw_df = summary[summary['Context'] == 512].groupby(
        ['Model_Full', 'Threads'], observed=True
    ).agg({'Avg_Watts': 'mean', 'Peak_Temp': 'mean', 'Avg_Clock': 'mean'}).reset_index()

    hw_pivot_w = hw_df.pivot(index='Model_Full', columns='Threads', values='Avg_Watts').reset_index()
    hw_pivot_t = hw_df.pivot(index='Model_Full', columns='Threads', values='Peak_Temp').reset_index()

    # CHANGE: reorder rows by parameter size
    hw_pivot_w = reorder_pivot(hw_pivot_w, model_full_order)
    hw_pivot_t = reorder_pivot(hw_pivot_t, model_full_order)

    fig, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pivot, ylabel, title in [
        (ax_w, hw_pivot_w, 'Avg Power Draw (Watts)', 'Average Power Draw by Model Config'),
        (ax_t, hw_pivot_t, 'Peak Temperature (°C)',   'Peak Die Temperature by Model Config')
    ]:
        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w/2, pivot[2], w, label='2 Threads', color='steelblue')
        ax.bar(x + w/2, pivot[4], w, label='4 Threads', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot['Model_Full'], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.suptitle("Hardware Metrics by Model Configuration (CTX=512)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/hardware_power_temp.png", dpi=300)
    plt.close()

    # --- PLOT 8 (NEW): Energy Efficiency — Joules per Token ---
    # J/Token = Avg_Watts / TPS; lower is better. Uses hardware data directly from master_results.
    eff_df = summary[summary['Context'] == 512].copy()
    eff_df['J_per_Token'] = eff_df['Avg_Watts'] / eff_df['TPS']
    eff_summary = eff_df.groupby(['Model_Full', 'Threads'], observed=True)['J_per_Token'].mean().reset_index()
    eff_pivot = eff_summary.pivot(index='Model_Full', columns='Threads', values='J_per_Token').reset_index()

    # CHANGE: reorder rows by parameter size
    eff_pivot = reorder_pivot(eff_pivot, model_full_order)

    x = np.arange(len(eff_pivot))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, eff_pivot[2], w, label='2 Threads', color='steelblue')
    ax.bar(x + w/2, eff_pivot[4], w, label='4 Threads', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(eff_pivot['Model_Full'], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("Energy per Token (Joules/Token) — Lower is Better")
    ax.set_title("Energy Efficiency: Joules per Token by Model Configuration (CTX=512)")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/energy_efficiency.png", dpi=300)
    plt.close()

    # --- PLOT 9 (NEW): Edge Boundary Heatmap — TPS by Model × Context Window ---
    # Reveals where throughput collapses; the key "edge cliff" visualization.
    # Fixed to THR=2 as baseline; averaged over prompts.
    heatmap_df = summary[summary['Threads'] == 2].groupby(
        ['Model_Full', 'Context'], observed=True
    )['TPS'].mean().reset_index()
    heatmap_pivot = heatmap_df.pivot(index='Model_Full', columns='Context', values='TPS')

    # CHANGE: reorder rows by parameter size (small → large models, top to bottom)
    available_rows = [m for m in model_full_order if m in heatmap_pivot.index]
    heatmap_pivot = heatmap_pivot.loc[available_rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        heatmap_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
        linewidths=0.5, ax=ax, cbar_kws={'label': 'Tokens per Second'}
    )
    ax.set_title("Edge Boundary: TPS by Model × Context Window (THR=2)\nGreen = practical; Red = edge limit")
    ax.set_xlabel("Context Window Size (tokens)")
    ax.set_ylabel("Model Configuration")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/edge_boundary_heatmap.png", dpi=300)
    plt.close()

    # --- PLOT 10 (NEW): Clock Speed and CPU Load vs Model Config ---
    # ARM Cortex-A76 scales frequency under load; shows how hard each model pushes the CPU.
    clock_df = summary[summary['Context'] == 512].groupby(
        ['Model_Full', 'Threads'], observed=True
    ).agg({'Avg_Clock': 'mean', 'Avg_CPU': 'mean'}).reset_index()
    clock_pivot = clock_df.pivot(index='Model_Full', columns='Threads', values='Avg_Clock').reset_index()
    cpu_pivot   = clock_df.pivot(index='Model_Full', columns='Threads', values='Avg_CPU').reset_index()

    # CHANGE: reorder rows by parameter size
    clock_pivot = reorder_pivot(clock_pivot, model_full_order)
    cpu_pivot   = reorder_pivot(cpu_pivot,   model_full_order)

    fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pivot, ylabel, title in [
        (ax_c, clock_pivot, 'Avg ARM Clock Speed (MHz)', 'Avg ARM Clock Speed by Config'),
        (ax_u, cpu_pivot,   'Avg CPU Load (%)',           'Avg CPU Load by Config')
    ]:
        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w/2, pivot[2], w, label='2 Threads', color='steelblue')
        ax.bar(x + w/2, pivot[4], w, label='4 Threads', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot['Model_Full'], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.suptitle("CPU Utilization Metrics by Model Configuration (CTX=512)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cpu_clock_load.png", dpi=300)
    plt.close()

    # --- PRINTED SUMMARY: Efficiency Table ---
    print("\n" + "="*80)
    print(f"{'ENERGY EFFICIENCY (JOULES/TOKEN) — CTX=512':^80}")
    print("="*80)
    eff_print = eff_df.groupby(['Model_Full', 'Threads'], observed=True).agg(
        Avg_W=('Avg_Watts', 'mean'),
        TPS=('TPS', 'mean'),
        J_per_Token=('J_per_Token', 'mean')
    ).reset_index().sort_values('J_per_Token')
    print(eff_print.to_string(index=False))

    # --- PRINTED SUMMARY: Hardware Metrics Table ---
    print("\n" + "="*80)
    print(f"{'HARDWARE TELEMETRY SUMMARY (AVG ACROSS PROMPTS)':^80}")
    print("="*80)
    hw_print = summary[summary['Context'] == 512].groupby(
        ['Model_Full', 'Context', 'Threads'], observed=True
    ).agg(
        Avg_W=('Avg_Watts', 'mean'),
        Peak_T=('Peak_Temp', 'mean'),
        Avg_CPU=('Avg_CPU', 'mean'),
        Peak_RAM_GB=('Peak_RAM', 'mean'),
        Avg_MHz=('Avg_Clock', 'mean'),
        Ever_Throttled=('Throttled', 'max')
    ).reset_index()
    print(hw_print.to_string(index=False))

if __name__ == "__main__":
    analyze_results()
    print(f"\nAnalysis complete. Plots saved to: {PLOT_DIR}")
