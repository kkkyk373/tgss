# 概要

グラフに対する各種回帰モデル（Random Forest、SVR、ANN など）を用いた選択的転移学習の実験

## Requirements

以下のパッケージが必要です（詳細は [requirements.txt](requirements.txt) を参照してください）:
- matplotlib
- POT
- torch
- tqdm
- scikit-learn
- numpy
- pandas

## Directory Structure

```
jcomm/
├── src/
│   ├── experiments/
│   │   ├── run_selective_dgm.py      # Deep Gravity Model を用いた実験
│   │   ├── run_selective_rf.py       # Random Forest を用いた実験
│   │   ├── run_selective_svr.py      # SVR を用いた実験
│   ├── models/
│   │   └── gravity.py                # DeepGravityReg の実装
│   ├── utils/
│   │   └── dataset.py                # CommutingODPairDataset の実装
├── scripts/                         # SLURMや実行用のスクリプト群
├── requirements.txt
└── README.md
```

## Usage

### 手元での実行例

Deep Gravity Model の実験は以下のように実行できます:

```bash
python src/experiments/run_selective_dgm.py \
    --data_dir "/path/to/data" \
    --fgw_dir "/path/to/fgw_outputs" \
    --targets_path "source_target_lists/targets_seed0.txt" \
    --sources_path "source_target_lists/sources_seed0.txt" \
    --condition topk \
    --alpha 50 \
    --max_samples 5000 \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --seed 42
```

同様に、SVRやRandom Forestの実験はそれぞれ `run_selective_svr.py` や `run_selective_rf.py` を実行してください。

### クラスター環境での実行

SLURM用のスクリプトが `scripts/` ディレクトリに用意されています。  
例:
```bash
sbatch scripts/slurm_dgm_ablation.sh
```

## Logs & Results

各実験の結果やログは、引数で指定した `--model_output_dir` や `--results_dir` 以下に保存されます。  
実験結果はJSON形式でまとめられ、解析ツール（例: [parse_tojson.py](src/examples/parse_tojson.py)）での変換も可能です。

## License

MIT License

---