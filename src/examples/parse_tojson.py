import os
import re
import glob
import json
from datetime import datetime

def convert_classical_logs_to_json(
    model_name="svr",
    input_root_dir="path/to/your/svr_results_txt",
    output_root_dir="path/to/your/svr_results_json"
):
    """
    SVR/RFのテキストログを、DGMの詳細なJSON形式に準拠する形で変換・保存する。
    ディレクトリ階層が異なっていても堅牢に動作するバージョン。
    """
    print(f"--- Starting detailed conversion for {model_name.upper()} ---")
    print(f"Input directory: {input_root_dir}")
    print(f"Output directory: {output_root_dir}")

    search_pattern = os.path.join(input_root_dir, "**", "*.txt")
    log_files = glob.glob(search_pattern, recursive=True)

    if not log_files:
        print(f"❌ Error: No result files (.txt) found in '{input_root_dir}'.")
        return

    results_pattern = re.compile(
        r"\[\d+/\d+\]\s+(\d+)\s*\n"
        r"MSE:\s+([\d.]+)\s+\(train=(\d+), test=(\d+)\)",
        re.MULTILINE
    )

    converted_count = 0
    skipped_archive_count = 0
    for filepath in log_files:
        try:
            parts = filepath.split(os.sep)
            
            # 'archive' ディレクトリ内のファイルはスキップする
            if 'archive' in parts:
                skipped_archive_count += 1
                continue

            # ▼▼▼▼▼ パスの解釈ロジックを堅牢化 ▼▼▼▼▼
            # パスの中から 'seed' ディレクトリを探す
            seed_index = -1
            for i, part in enumerate(parts):
                if re.match(r'seed\d+', part):
                    seed_index = i
                    break
            
            if seed_index == -1:
                raise ValueError("Could not find 'seed' directory in path")

            alpha_str = parts[seed_index - 2]
            condition = parts[seed_index - 1]
            seed_str = parts[seed_index]
            # End of file path interpretation logic

            alpha = int(re.search(r'alpha(\d+)', alpha_str).group(1))
            seed = int(re.search(r'seed(\d+)', seed_str).group(1))

        except (IndexError, AttributeError, ValueError) as e:
            print(f"⚠️ Warning: Could not parse parameters from path: {filepath}. Error: {e}. Skipping.")
            continue

        # (以下、ファイル内容のパースとJSON保存のロジックは変更なし)
        # ... (previous correct logic) ...
        with open(filepath, 'r') as f:
            content = f.read()

        matches = results_pattern.findall(content)

        if not matches:
            print(f"⚠️ Warning: No valid result lines found in {filepath}. Skipping.")
            continue

        results_list = []
        for match in matches:
            results_list.append({
                "target_id": match[0],
                "mse": float(match[1]),
                "train_samples": int(match[2]),
                "test_samples": int(match[3]),
                "status": "success"
            })
            
        datetime_match = re.search(r'_(\d{8}_\d{4})\.txt$', os.path.basename(filepath))
        execution_datetime = None
        if datetime_match:
            try:
                dt_obj = datetime.strptime(datetime_match.group(1), '%Y%m%d_%H%M')
                execution_datetime = dt_obj.isoformat()
            except ValueError:
                pass

        metadata = {
            "data_dir": "/work/hideki-h/jcomm/ComOD-dataset/data",
            "fgw_dir": "/work/hideki-h/jcomm/ComOD-dataset/outputs",
            "results_dir": "results",
            "model_output_dir": "outputs",
            "top_k": 100,
            "bottom_k": 100,
            "max_samples": 5000,
            "targets_path": f"source_target_lists/targets_seed{seed}.txt",
            "sources_path": f"source_target_lists/sources_seed{seed}.txt",
            "condition": condition,
            "alpha": alpha,
            "seed": seed,
            "epochs": None,
            "batch_size": None,
            "lr": None,
            "execution_datetime": execution_datetime
        }

        final_json = {
            "metadata": metadata,
            "results": results_list
        }

        output_path = filepath.replace(input_root_dir, output_root_dir).replace('.txt', '.json')
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)
        
        converted_count += 1

    if skipped_archive_count > 0:
        print(f"ℹ️ Info: Skipped {skipped_archive_count} files found in 'archive' directories.")
    print(f"\n✅ Conversion complete. {converted_count} files were converted and saved in '{output_root_dir}'.")


if __name__ == '__main__':
    # SVR/raw
    convert_classical_logs_to_json(
        model_name="svr",
        input_root_dir="results/svr/raw",
        output_root_dir="results/svr/raw_json"
    )
    # RF/raw
    convert_classical_logs_to_json(
        model_name="rf",
        input_root_dir="results/rf/raw",
        output_root_dir="results/rf/raw_json"
    )
    # SVR/opt
    convert_classical_logs_to_json(
        model_name="svr_opt",
        input_root_dir="results/svr/opt",
        output_root_dir="results/svr/opt_json"
    )