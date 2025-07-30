import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from typing import Dict, List

def plot_all_summaries(
    summary_files: Dict[str, str],
    output_path: str,
    log_scale: bool = False,
    showfliers: bool = False
) -> None:
    """
    Load summary CSVs, compute RMSE, and plot boxplots of overall RMSE
    by model, alpha, and condition. Saves to output_path.
    """
    # --- 1. Load and concatenate data ---
    dfs: List[pd.DataFrame] = []
    for model, path_str in summary_files.items():
        p = Path(path_str)
        if not p.exists():
            print(f"⚠️ Warning: '{model}' file not found: {p}")
            continue
        df = pd.read_csv(p)
        df['model'] = model
        df['overall_rmse'] = np.sqrt(df['overall_mse'])
        dfs.append(df)

    if not dfs:
        print("❌ No summary files loaded. Aborting.")
        return
    data = pd.concat(dfs, ignore_index=True)

    # --- 2. Broadcast 'random'/'all' to all alphas for consistent plotting ---
    alpha_dependent = data[data['condition'].isin(['topk','bottomk'])]
    alpha_independent = data[data['condition'].isin(['random','all'])]
    alphas = sorted(alpha_dependent['alpha'].dropna().unique())
    if alphas and not alpha_independent.empty:
        replicated = pd.concat([
            alpha_independent.assign(alpha=a) for a in alphas
        ], ignore_index=True)
        final_df = pd.concat([alpha_dependent, replicated], ignore_index=True)
    else:
        final_df = data.copy()

    # --- 3. Setup orders and colors ---
    model_order = ['SVR', 'RF', 'DGM']
    cond_order  = ['all', 'topk', 'random', 'bottomk']
    color_map = {
        'all':    '#1f77b4',
        'topk':   '#ff7f0e',
        'random': '#2ca02c',
        'bottomk':'#d62728',
    }

    # --- 4. Create subplots ---
    n_models = len(model_order)
    fig, axes = plt.subplots(1, n_models, sharey=True,
                             figsize=(5*n_models, 6))

    for ax, model in zip(axes, model_order):
        df_model = final_df[final_df['model'] == model]
        n_cond = len(cond_order)

        # Collect data and positions
        box_data = []
        colors = []
        positions = []

        for i, a in enumerate(alphas):
            base = i * (n_cond + 1)
            for j, cond in enumerate(cond_order):
                vals = df_model.loc[
                    (df_model['alpha'] == a) &
                    (df_model['condition'] == cond),
                    'overall_rmse'
                ].values
                box_data.append(vals if vals.size else [np.nan])
                colors.append(color_map.get(cond, '#ccc'))
                positions.append(base + j)

        # Draw boxplot with custom median line
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=showfliers,
            whis=1.5,
            medianprops=dict(color='black', linewidth=2)
        )
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)

        # X-axis: ticks at group centers
        centers = [i*(n_cond+1) + (n_cond-1)/2 for i in range(len(alphas))]
        ax.set_xticks(centers)
        ax.set_xticklabels([f"α={a}" for a in alphas], rotation=0)
        ax.set_xlabel("Alpha")

        # Optional log scale
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(model)
        if ax is axes[0]:
            ax.set_ylabel("Overall RMSE")
        ax.grid(axis='y', linestyle='--', alpha=0.7)


    # --- 5. Add legend ---
    label_map = {
        'all':      'all',
        'topk':     'topk',
        'random':   'randomk',
        'bottomk':  'bottomk'
    }

    handles = [Patch(color=color_map[c], label=label_map[c]) for c in cond_order]
    fig.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=(0.1, 0.9),
        borderaxespad=0.,
        title_fontsize=18,
        fontsize=18,
    )

    # --- 6. Print statistics ---. Print statistics ---
    print("\nMean & Std of RMSE by model, condition, alpha:")
    stats = final_df.groupby(['model','condition','alpha'])['overall_rmse']
    stats = stats.agg(['mean','std']).round(3).reset_index()
    print(stats.to_string(index=False))

    # --- 7. Save and show ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    summary_files = {
        'DGM': 'outputs/dgm_summary.csv',
        'SVR': 'outputs/svr_summary.csv',
        'RF':  'outputs/rf_summary.csv'
    }
    plot_all_summaries(summary_files, 'outputs/comparison_plot.png')