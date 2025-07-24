import os
import glob
import json
import pandas as pd
import numpy as np

def aggregate_results_from_json(
    model_name="svr",
    input_root_dir="results/model/options",
    output_csv_path="outputs/model_summary.csv"
):
    """
    指定されたディレクトリ以下の全てのJSON結果ファイルを集計
    micro mse を算出して単一のCSVファイルにまとめる
    """

    print(f"--- Starting aggregation for {model_name.upper()} from '{input_root_dir}' ---")

    param_keys = [
        'condition', 'alpha', 'seed', 'top_k', 'bottom_k', 
        'max_samples', 'epochs', 'batch_size', 'lr'
    ]

    # --- ステップ1: 全てのJSONファイルを検索 ---
    search_pattern = os.path.join(input_root_dir, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"❌ Error: No result JSON files found in '{input_root_dir}'.")
        return

    all_runs_data = []

    # --- ステップ2: 各JSONファイルを処理して加重平均を計算 ---
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # メタデータと結果を取得
        metadata = data.get("metadata", {})
        results = data.get("results", [])

        if not metadata or not results:
            print(f"⚠️ Warning: Invalid JSON structure in {filepath}. Skipping.")
            continue

        total_mse_product = 0
        total_test_samples = 0
        
        for r in results:
            mse = r.get("mse")
            test_samples = r.get("test_samples")
            if mse is not None and test_samples is not None and test_samples > 0:
                total_mse_product += mse * test_samples
                total_test_samples += test_samples

        if total_test_samples > 0:
            overall_mse = total_mse_product / total_test_samples
        else:
            overall_mse = np.nan

        run_data = {key: metadata.get(key) for key in param_keys}
        run_data['overall_mse'] = overall_mse

        # 'condition'が'random'または'all'の場合、alphaはダミーなのでNaNに統一する
        if run_data.get('condition') in ['random', 'all']:
            run_data['alpha'] = np.nan

        all_runs_data.append(run_data)

    if not all_runs_data:
        print(f"❌ Error: No data could be aggregated from '{input_root_dir}'.")
        return

    # --- ステップ3: 結果をDataFrameに変換して保存 ---
    df = pd.DataFrame(all_runs_data)

    # --- ステップ4: 重複行の削除 ---
    df.drop_duplicates(subset=param_keys, keep='last', inplace=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
    print(f"✅ Aggregation complete. Results are saved in '{output_csv_path}'")
    print(f"--- Summary for {model_name.upper()} ---")
    print(df.head())


if __name__ == '__main__':
    # SVR (raw)
    aggregate_results_from_json(
        model_name="svr_raw",
        input_root_dir="results/svr/raw",
        output_csv_path="outputs/svr_summary.csv"
    )
    
    # RF (raw)
    aggregate_results_from_json(
        model_name="rf_raw",
        input_root_dir="results/rf/raw",
        output_csv_path="outputs/rf_summary.csv"
    ) 

    aggregate_results_from_json(
        model_name="dgm",
        input_root_dir="results/dgm/raw",
        output_csv_path="outputs/dgm_summary.csv"
    )