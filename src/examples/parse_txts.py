"""
早期に実験を行った random forest と support vector regression の結果は平文でログファイルに出力されている。
このスクリプトは、これらのログファイルを読み込み、必要な情報を抽出して DataFrame に変換する。
"""

import glob
import re
import pathlib
import pandas as pd

# ---------- 設定 ----------
RESULT_DIR = pathlib.Path("results")
GLOB_EXPR  = "selective_rf_*_seed*_alpha*_topk100_max5000_*.txt"

# ファイル名に埋め込まれたメタ情報を抜く正規表現
fname_re = re.compile(
    r"selective_rf_(?P<method>topk|random|bottomk)_"
    r"seed(?P<seed>\d+)_alpha(?P<alpha>\d+)_topk100_max5000_.*\.txt$"
)

# ターゲット行   : [ 12/227] 40027
target_re = re.compile(r"\[\d+/\d+\]\s+(?P<area_id>\d{5})")

# MSE 行         : MSE: 7156.0501 (train=5000, test=400)
mse_re = re.compile(
    r"MSE:\s+(?P<mse>[\d\.eE+-]+).*test=(?P<test>\d+)\)"
)

rows = []
for path in RESULT_DIR.glob(GLOB_EXPR):
    m_fname = fname_re.match(path.name)
    if not m_fname:        # 想定外のファイルはスキップ
        continue

    meta = m_fname.groupdict()
    meta["model"] = "rf"
    meta["seed"]  = int(meta["seed"])
    meta["alpha"] = int(meta["alpha"])

    with path.open() as f:
        lines = iter(f)
        for line in lines:
            m_target = target_re.match(line)
            if not m_target:        # ターゲット行でなければ読み飛ばし
                continue

            # 次の行に MSE 情報がある
            mse_line = next(lines, "")
            m_mse    = mse_re.search(mse_line)
            if not m_mse:
                raise ValueError(f"MSE 行の形式が想定外です: {mse_line}")

            rows.append(
                {
                    "model":           meta["model"],
                    "method":          meta["method"],
                    "seed":            meta["seed"],
                    "fgw_alpha":       meta["alpha"],
                    "target_area_id":  m_target["area_id"],
                    "target_pair_count": int(m_mse["test"]),
                    "mse":             float(m_mse["mse"]),
                }
            )

# ----------  DataFrame  ----------
df = pd.DataFrame(rows, columns=[
    "model", "method", "seed", "fgw_alpha",
    "target_area_id", "target_pair_count", "mse"
])

# 必要なら行数チェック
expected = 227 * 90           # = 20 430
assert len(df) == expected, f"行数が想定外: {len(df)=}, 期待 {expected}"


# ---------- 保存例 ----------
df.to_csv("results/selective_results_rf.csv", index=False)

print("✅ DataFrame 生成完了:", df.shape)