import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

def plot_all_summaries(
    summary_files: Dict[str, str],
    output_path: str,
    log_scale: bool = False,
    showfliers: bool = False
) -> None:
    """
    複数のモデルのサマリーCSVを読み込み、比較用の箱ひげ図を生成する。
    'all'と'random'の結果を、各alphaの値に共通で表示する。

    Args:
        summary_files: モデル名をキー、CSVファイルパスを値とする辞書。
        output_path: グラフの保存先パス。
        log_scale: Y軸を対数スケールにするか。
        showfliers: 外れ値を表示するか。
    """
    # 1. データ読み込み
    all_dfs: List[pd.DataFrame] = []
    for model_name, path_str in summary_files.items():
        path = Path(path_str)
        if not path.exists():
            print(f"⚠️ Warning: File for {model_name} not found at '{path}'. Skipping.")
            continue
        df = pd.read_csv(path)
        df['model'] = model_name
        all_dfs.append(df)

    if not all_dfs:
        print("❌ Error: No summary files could be loaded.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 2. データ前処理
    combined_df['overall_rmse'] = np.sqrt(combined_df['overall_mse'])
    dep_df = combined_df[combined_df['condition'].isin(['topk','bottomk'])]
    indep_df = combined_df[combined_df['condition'].isin(['random','all'])]
    alphas = sorted(dep_df['alpha'].dropna().unique())
    if alphas and not indep_df.empty:
        indep_rep = pd.concat([indep_df.assign(alpha=a) for a in alphas], ignore_index=True)
        final_df = pd.concat([dep_df, indep_rep], ignore_index=True)
    else:
        final_df = combined_df

    # 3. カテゴリ順序設定
    model_order = ['SVR','RF','DGM']
    cond_order = ['all','topk','random','bottomk']
    final_df['model'] = pd.Categorical(final_df['model'], categories=model_order, ordered=True)
    final_df['condition'] = pd.Categorical(final_df['condition'], categories=cond_order, ordered=True)

    # 4. プロット
    sns.set_theme(style='whitegrid')
    g = sns.catplot(
        data=final_df,
        x='alpha',
        y='overall_rmse',
        hue='condition',
        col='model',
        kind='box',
        order=alphas,
        hue_order=cond_order,
        height=6,
        aspect=0.8,
        showfliers=showfliers,
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.1, 0.9), title=None)

    # 対数スケール設定
    if log_scale:
        for ax in g.axes.flatten():
            ax.set_yscale('log')

    g.figure.suptitle('Performance of Source Selection Strategies Across Models', y=1.03)
    g.set_axis_labels('Alpha', 'Overall RMSE')
    g.set_titles('Model: {col_name}')

    # 6. 保存 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison chart saved as '{output_path}'")
    plt.show()

if __name__ == '__main__':
    summary_files = {'DGM':'outputs/dgm_summary.csv','SVR':'outputs/svr_summary.csv','RF':'outputs/rf_summary.csv'}
    plot_all_summaries(summary_files, 'outputs/comparison_plot_unified.png', log_scale=False, showfliers=False)
