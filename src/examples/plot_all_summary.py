import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

def plot_all_summaries(
    summary_files: Dict[str, str],
    output_path: str
) -> None:
    """
    複数のモデルのサマリーCSVを読み込み、比較用の箱ひげ図を生成する。
    'all'と'random'の結果を、各alphaの値に共通で表示する。
    
    Args:
        summary_files: モデル名をキー、CSVファイルパスを値とする辞書。
        output_path: 生成したグラフの保存先パス。
    """
    all_dfs: List[pd.DataFrame] = []
    
    # --- 1. 各モデルのサマリーCSVを読み込み、結合する ---
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

    combined_df: pd.DataFrame = pd.concat(all_dfs, ignore_index=True)

    # --- 2. データ準備 ---
    print("🛠️  Processing data to broadcast 'all' and 'random' conditions...")
    combined_df['overall_rmse'] = np.sqrt(combined_df['overall_mse'])

    # a) Alphaに依存するデータ (topk, bottomk)
    alpha_dependent_df = combined_df[combined_df['condition'].isin(['topk', 'bottomk'])].copy()
    
    # b) Alphaに依存しないデータ (random, all)
    alpha_independent_df = combined_df[combined_df['condition'].isin(['random', 'all'])].copy()

    # c) 存在するalphaのユニークな値を取得
    alphas_to_broadcast = sorted(alpha_dependent_df['alpha'].dropna().unique())
    if not alphas_to_broadcast:
        print("⚠️ Warning: No specific alpha values found for 'topk'/'bottomk'. Cannot broadcast.")
        # alpha依存データがない場合は、元のDFをそのまま使う
        final_df = combined_df
    else:
        # d) 'random'と'all'のデータを各alpha値に複製
        broadcasted_dfs = []
        if not alpha_independent_df.empty:
            for alpha_val in alphas_to_broadcast:
                temp_df = alpha_independent_df.copy()
                temp_df['alpha'] = alpha_val
                broadcasted_dfs.append(temp_df)
            
            broadcasted_df = pd.concat(broadcasted_dfs, ignore_index=True)
            # e) データを再結合
            final_df = pd.concat([alpha_dependent_df, broadcasted_df], ignore_index=True)
        else:
            final_df = alpha_dependent_df

    # --- 3. グラフ描画のための最終準備 ---
    model_order: List[str] = ['SVR', 'RF', 'DGM']
    condition_order: List[str] = ['all', 'topk', 'random', 'bottomk']

    final_df['model'] = pd.Categorical(final_df['model'], categories=model_order, ordered=True)
    final_df['condition'] = pd.Categorical(final_df['condition'], categories=condition_order, ordered=True)

    # --- 4. Seabornを使ってグラフを描画 ---
    print("🎨 Generating plot...")
    sns.set_theme(style="whitegrid")
    
    g: sns.FacetGrid = sns.catplot(
        data=final_df,
        x='alpha',
        y='overall_rmse',
        hue='condition',
        col='model',
        kind='box',
        order=alphas_to_broadcast,
        hue_order=condition_order,
        height=6,
        aspect=0.8,
        legend_out=False  # ★★★ 1. 凡例をグラフの内側に配置する設定 ★★★
    )
    
    g.fig.suptitle('Performance of Source Selection Strategies Across Models', y=1.03)
    g.set_axis_labels("Alpha Parameter", "Overall Root Mean Squared Error (RMSE)")
    g.set_titles("Model: {col_name}")
    
    # ★★★ 2. 凡例を左上に移動し、タイトルを設定 ★★★
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.1, 0.8), title="Selection Condition")

    # --- 5. グラフを保存 ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison chart saved as '{output_path}'")
    plt.show()


if __name__ == '__main__':
    summary_files = {
        "DGM": "outputs/dgm_summary.csv",
        "SVR": "outputs/svr_summary.csv",
        "RF": "outputs/rf_summary.csv"
    }
    output_path = "outputs/comparison_plot_unified.png"

    plot_all_summaries(summary_files, output_path)