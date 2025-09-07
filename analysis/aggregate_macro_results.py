"""
グラフレベルの全ての実験結果を集計し、DataFrameに変換するスクリプト。
出力例：
  condition  alpha  seed  top_k  bottom_k  max_samples epochs batch_size    lr      mse_mean       mse_std   rmse_mean   rmse_std  n_targets_used
0   bottomk  100.0     3    100       100         5000   None       None  None  27775.570449  43900.157036  137.552714  94.308017             227
1   bottomk  100.0     4    100       100         5000   None       None  None  21905.409285  39198.641088  123.725535  81.403891             227
2   bottomk  100.0     5    100       100         5000   None       None  None  22900.944858  44793.001721  122.288713  89.339670             227
3   bottomk  100.0     2    100       100         5000   None       None  None  24664.293219  37614.628962  131.322878  86.321612             227
4   bottomk  100.0     9    100       100         5000   None       None  None  20890.908470  34995.127237  120.133783  80.544158             227

テストケースは227枚のグラフ
"""

import os
import glob
import json
import pandas as pd
import numpy as np


# ===== 共通ユーティリティ =====
def _finite(values):
    """数値かつ有限の値だけを残す。"""
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _mean_std_sample(arr: np.ndarray):
    """平均と標本標準偏差（ddof=1）。要素数<2のとき std は NaN。"""
    if arr.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan
    return mean, std


# ===== 集計本体 =====
def aggregate_graph_results_from_json(
    model_name: str,
    input_root_dir: str,
    output_csv_path: str,
) -> pd.DataFrame:
    """
    指定ディレクトリ以下の全 JSON（ターゲット別結果を含む）を走査し、
    グラフレベル（ターゲット毎）の MSE/RMSE を集計して CSV に保存する。

    Returns:
        pd.DataFrame: 集計結果の DataFrame
    """
    print(f"--- Starting graph-level aggregation for {model_name.upper()} from '{input_root_dir}' ---")

    # パラメータキー（JSON metadata から拾う列）
    param_keys = [
        'condition', 'alpha', 'seed', 'top_k', 'bottom_k',
        'max_samples', 'epochs', 'batch_size', 'lr'
    ]

    # JSON ファイル探索
    search_pattern = os.path.join(input_root_dir, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"❌ Error: No result JSON files found in '{input_root_dir}'.")
        return pd.DataFrame(columns=param_keys + [
            'mse_mean', 'mse_std', 'rmse_mean', 'rmse_std', 'n_targets_used'
        ])

    all_rows = []

    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        results = data.get("results", [])

        if not metadata or not isinstance(results, list):
            print(f"⚠️ Warning: Invalid JSON structure in {filepath}. Skipping.")
            continue

        # 1) ターゲットごとの MSE を抽出（欠損/NaN を除外）
        mse_list = _finite([r.get('mse', np.nan) for r in results])

        # 2) RMSE = sqrt(MSE) をターゲット単位で計算（負値は除外）
        rmse_list = _finite([np.sqrt(x) for x in mse_list if np.isfinite(x) and x >= 0.0])

        # 3) 平均と標本標準偏差
        mse_mean, mse_std = _mean_std_sample(mse_list)
        rmse_mean, rmse_std = _mean_std_sample(rmse_list)

        # 4) 行の構築
        row = {key: metadata.get(key) for key in param_keys}
        row['mse_mean'] = mse_mean
        row['mse_std'] = mse_std
        row['rmse_mean'] = rmse_mean
        row['rmse_std'] = rmse_std
        row['n_targets_used'] = int(mse_list.size)  # 使用できたターゲット数（例: 227）

        # 'random'/'all' では alpha を NaN に
        if row.get('condition') in ['random', 'all']:
            row['alpha'] = np.nan

        all_rows.append(row)

    # DataFrame 化
    df = pd.DataFrame(all_rows)

    # 重複があれば最後を残す
    df.drop_duplicates(subset=param_keys, keep='last', inplace=True)

    # 並び替え & index 整理
    df.sort_values(by=['condition', 'alpha', 'seed'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 保存
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"✅ Aggregation complete. Results saved to '{output_csv_path}'")
    print(f"--- Summary [{model_name.upper()}] ---")
    print(df.head())

    return df


if __name__ == '__main__':
    # 必要なモデルをここで指定
    models = ['svr', 'rf', 'dgm']

    for model in models:
        in_dir = os.path.join("results", model, "raw")
        out_csv = os.path.join("outputs", f"{model}_graph_summary.csv")
        aggregate_graph_results_from_json(model_name=model, input_root_dir=in_dir, output_csv_path=out_csv)