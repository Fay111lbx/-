# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================
# 0) 你只需要改这里（目录批处理）
# =========================
# 源数据目录：里面放了所有城市的 CSV（如 北京_...csv、德州_...csv 等）
DATA_DIR = "/root/data/no-heating open/源数据"

# 输出基目录：与“源数据”同级（即 /root/data/treatedpass-heating）
OUT_BASE_DIR = os.path.dirname(DATA_DIR.rstrip("/"))

# 只处理包含该关键词的 CSV（更安全，避免把别的 csv 也跑了）
FILE_KEYWORD = "panel_vs_controls_weekly_2018_2022_with_year_week_fips_with_cov"
FILE_SUFFIX  = ".csv"

# ✅ 开启 GAN（生成 donor）
USE_GAN = True

# ✅ 生成样本数：1~5 倍（n_gen = ratio * 真实donor数）
GAN_RATIOS = [1, 2, 3, 4, 5]

# 训练随机种子
SEED = 2026

# =========================
# 1) 固定随机性
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2) 读数据 + 自动识别 treated 城市 + 政策时点
# =========================
COV_COLS = [
    "primary_industry_gdp_share_pct",
    "secondary_industry_gdp_share_pct",
    "gdp_per_capita_cny",
    "registered_population_10k",
]
REQUIRED = ["fips", "city", "week_end", "year", "week", "treated", "pm25"] + COV_COLS

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"缺少列：{miss}\n当前列：{list(df.columns)}")

    df = df.copy()
    df["fips"] = pd.to_numeric(df["fips"], errors="coerce").astype("Int64")
    df["treated"] = pd.to_numeric(df["treated"], errors="coerce").astype(int)
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

    df["week_end_dt"] = pd.to_datetime(df["week_end"], errors="coerce")
    if df["week_end_dt"].isna().any():
        raise ValueError("week_end 有无法解析的日期，请检查格式，如 2018/1/7")

    df["year_week"] = (df["year"].astype(int) * 100 + df["week"].astype(int)).astype(int)

    # 协变量转数值 + 对数变换（处理非正值）
    for c in COV_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        ser = df[c]
        if ser.notna().any():
            minv = ser.min(skipna=True)
            if pd.isna(minv):
                continue
            if minv <= 0:
                offset = abs(minv) + 1e-6
                df[c] = np.log(ser + offset)
            else:
                df[c] = np.log(ser)

    df = df.sort_values(["year_week", "fips"]).reset_index(drop=True)
    return df

def infer_treated_unit_and_t0(df: pd.DataFrame):
    treated_fips = (
        df.groupby("fips")["treated"].max()
        .loc[lambda s: s == 1]
        .index.tolist()
    )
    if len(treated_fips) != 1:
        raise ValueError(f"当前文件识别到 {len(treated_fips)} 个 treated 城市（应当只有1个）。识别结果：{treated_fips}")

    treated_fips = int(treated_fips[0])
    treated_city = df.loc[df["fips"] == treated_fips, "city"].dropna().iloc[0]

    t0 = (
        df.loc[(df["fips"] == treated_fips) & (df["treated"] == 1), "year_week"]
        .min()
    )
    if pd.isna(t0):
        raise ValueError("没找到 treated 城市 treated==1 的起点，检查 treated 列是否正确标注。")

    return treated_fips, treated_city, int(t0)

# =========================
# 3) 构造 deepSCM 的监督学习数据
# =========================
def build_matrices(df: pd.DataFrame, treated_fips: int, t0: int):
    donor_fips = sorted([int(x) for x in df["fips"].dropna().unique() if int(x) != treated_fips])
    wide_y = df.pivot(index="year_week", columns="fips", values="pm25").sort_index()

    y_treat = wide_y[treated_fips].copy()
    X_donors = wide_y[donor_fips].copy()

    treat_cov = (
        df.loc[df["fips"] == treated_fips, ["year_week"] + COV_COLS]
        .drop_duplicates("year_week")
        .set_index("year_week")
        .sort_index()
    )

    donor_cov_mean = (
        df.loc[df["fips"].isin(donor_fips), ["year_week"] + COV_COLS]
        .groupby("year_week")[COV_COLS]
        .mean()
        .sort_index()
    )

    common_times = wide_y.index.intersection(treat_cov.index).intersection(donor_cov_mean.index)
    y_treat = y_treat.loc[common_times]
    X_donors = X_donors.loc[common_times]
    treat_cov = treat_cov.loc[common_times]
    donor_cov_mean = donor_cov_mean.loc[common_times]

    keep_mask = (~y_treat.isna())
    keep_mask &= (~X_donors.isna().any(axis=1))
    keep_mask &= (~treat_cov.isna().any(axis=1))
    keep_mask &= (~donor_cov_mean.isna().any(axis=1))
    common_times = common_times[keep_mask.values]

    y_treat = y_treat.loc[common_times]
    X_donors = X_donors.loc[common_times]
    treat_cov = treat_cov.loc[common_times]
    donor_cov_mean = donor_cov_mean.loc[common_times]

    pre_times = common_times[common_times < t0]
    post_times = common_times[common_times >= t0]
    if len(pre_times) < 30:
        raise ValueError(f"政策前时间点太少（{len(pre_times)}），deepSCM 不稳。")

    def make_X(times):
        A = X_donors.loc[times].values
        B = donor_cov_mean.loc[times].values
        C = treat_cov.loc[times].values
        return np.concatenate([A, B, C], axis=1)

    X_pre = make_X(pre_times)
    y_pre = y_treat.loc[pre_times].values.reshape(-1, 1)

    X_all = make_X(common_times)
    y_all = y_treat.loc[common_times].values.reshape(-1, 1)

    meta = {
        "donor_fips": donor_fips,
        "common_times": common_times,
        "pre_times": pre_times,
        "post_times": post_times,
    }
    return X_pre, y_pre, X_all, y_all, X_donors, meta

# =========================
# 4) GAN（新增：仅用于扩充 donor）
#     参数保持你之前的：epochs=1500, batch_size=32, z_dim=64, lr=1e-4
# =========================
class Generator(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_gan_return_model(real_sequences: np.ndarray, epochs: int = 1500, batch_size: int = 32, z_dim: int = 64):
    """
    real_sequences: (n_donors, T_total)
    返回：训练好的 G + scaler + z_dim
    """
    n_donors, T_total = real_sequences.shape

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_sequences)

    G = Generator(z_dim, T_total).to(DEVICE)
    D = Discriminator(T_total).to(DEVICE)

    opt_g = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-4)
    bce = nn.BCELoss()

    ds = TensorDataset(torch.tensor(real_scaled, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    for ep in range(epochs):
        for (x_real,) in dl:
            x_real = x_real.to(DEVICE)
            bs = x_real.size(0)

            # ---- train D ----
            z = torch.randn(bs, z_dim, device=DEVICE)
            x_fake = G(z).detach()

            y_real = torch.ones(bs, 1, device=DEVICE) * 0.9
            y_fake = torch.zeros(bs, 1, device=DEVICE)

            d_real = D(x_real)
            d_fake = D(x_fake)

            loss_d = bce(d_real, y_real) + bce(d_fake, y_fake)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---- train G ----
            z = torch.randn(bs, z_dim, device=DEVICE)
            x_fake = G(z)
            d_fake = D(x_fake)
            loss_g = bce(d_fake, y_real)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if (ep + 1) % 300 == 0:
            print(f"[GAN] epoch {ep+1}/{epochs} | loss_d={loss_d.item():.4f} loss_g={loss_g.item():.4f}")

    return G, scaler, z_dim

@torch.no_grad()
def gan_generate(G: nn.Module, scaler: StandardScaler, n_gen: int, z_dim: int, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    z = torch.randn(n_gen, z_dim, device=DEVICE)
    gen_scaled = G(z).detach().cpu().numpy()
    gen = scaler.inverse_transform(gen_scaled)
    return gen  # (n_gen, T_total)

# =========================
# 5) DNN（保持原样）
# =========================
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

def train_regressor(X_pre, y_pre, X_all, epochs=2000, batch_size=64):
    scaler = StandardScaler()
    X_pre_s = scaler.fit_transform(X_pre)
    X_all_s = scaler.transform(X_all)

    X_pre_t = torch.tensor(X_pre_s, dtype=torch.float32)
    y_pre_t = torch.tensor(y_pre, dtype=torch.float32)
    ds = TensorDataset(X_pre_t, y_pre_t)
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model = MLPRegressor(in_dim=X_pre_s.shape[1], hidden=256, dropout=0.05).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience = 200
    patience_left = patience

    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if (ep + 1) % 200 == 0:
            print(f"[DNN] epoch {ep+1}/{epochs} | train_mse={mean_loss:.6f} | best={best_loss:.6f}")

        if patience_left <= 0:
            print(f"[DNN] early stop at epoch {ep+1}, best_mse={best_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        yhat_all = model(torch.tensor(X_all_s, dtype=torch.float32).to(DEVICE)).cpu().numpy().reshape(-1)

    return yhat_all, scaler

# =========================
# 6) 批处理：每个城市单独跑
# =========================
OUT_DIR = None

def get_city_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    if "_" in base:
        return base.split("_")[0]
    return os.path.splitext(base)[0]

def sanitize_name(s: str) -> str:
    return "".join(ch for ch in str(s) if ch not in r'\/:*?"<>|').strip()

def run_one_city_file(data_path: str):
    global OUT_DIR

    city_from_name = get_city_from_filename(data_path)
    OUT_DIR = os.path.join(OUT_BASE_DIR, city_from_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"[START] city_folder={city_from_name}")
    print(f"       data={data_path}")
    print(f"       out ={OUT_DIR}")
    print("=" * 90)

    set_seed(SEED)

    df = load_and_prepare(data_path)
    treated_fips, treated_city, t0 = infer_treated_unit_and_t0(df)

    print(f"识别处理城市：{treated_city} (fips={treated_fips})")
    print(f"识别政策起点 year_week = {t0}")

    X_pre0, y_pre, X_all0, y_all, X_donors_wide, meta = build_matrices(df, treated_fips, t0)
    donor_fips = meta["donor_fips"]
    common_times = meta["common_times"]
    pre_times = meta["pre_times"]

    safe_city = sanitize_name(treated_city)

    # ====== GAN：对该城市只训练一次，然后生成 1~5 倍分别跑 ======
    if not USE_GAN:
        raise ValueError("你要求开启 GAN，但当前 USE_GAN=False，请改为 True。")

    n_real = len(donor_fips)
    donors_full = X_donors_wide.loc[common_times].values.T  # (n_donors, T)

    print(f"GAN 开启：真实donor={n_real}，将分别运行倍率={GAN_RATIOS}")

    # GAN 训练（参数保持：epochs=1500, batch=32）
    G, gan_scaler, z_dim = train_gan_return_model(donors_full, epochs=1500, batch_size=32, z_dim=64)

    results_rows = []

    for ratio in GAN_RATIOS:
        n_gen = int(n_real * int(ratio))
        print("\n" + "-" * 90)
        print(f"[RUN] {treated_city} | GANx{ratio} | n_gen={n_gen}")
        print("-" * 90)

        # 生成 donor 序列（用不同 seed 保证可复现且倍率不同）
        gen_full = gan_generate(G, gan_scaler, n_gen=n_gen, z_dim=z_dim, seed=SEED + int(ratio)).T  # (T, n_gen)

        # 拼接：X = [real_donors, gen_donors, cov(8列)]
        X_pre_real = X_pre0[:, :n_real]
        X_all_real = X_all0[:, :n_real]

        idx_pre = [np.where(common_times == t)[0][0] for t in pre_times]
        idx_all = list(range(len(common_times)))

        gen_pre = gen_full[idx_pre, :]  # (T_pre, n_gen)
        gen_all = gen_full[idx_all, :]  # (T_all, n_gen)

        cov_pre = X_pre0[:, n_real:]
        cov_all = X_all0[:, n_real:]

        X_pre = np.concatenate([X_pre_real, gen_pre, cov_pre], axis=1)
        X_all = np.concatenate([X_all_real, gen_all, cov_all], axis=1)

        print(f"加入生成donor后：X_pre dim={X_pre.shape}, X_all dim={X_all.shape}")

        # DNN 学习 F 并预测反事实（参数保持：epochs=2000, batch_size=64）
        yhat_all, _ = train_regressor(X_pre, y_pre, X_all, epochs=2000, batch_size=64)

        out = pd.DataFrame({
            "year_week": common_times.astype(int),
            "pm25_obs": y_all.reshape(-1),
            "pm25_cf": yhat_all,
        })
        out["att"] = out["pm25_obs"] - out["pm25_cf"]
        out["period"] = np.where(out["year_week"] < t0, "pre", "post")

        pre_rmspe = np.sqrt(np.mean((out.loc[out["period"] == "pre", "pm25_obs"] - out.loc[out["period"] == "pre", "pm25_cf"]) ** 2))
        post_att_mean = out.loc[out["period"] == "post", "att"].mean()

        print(f"\n[诊断] 政策前 RMSPE = {pre_rmspe:.4f}")
        print(f"[结果] 政策后 平均ATT(周度均值) = {post_att_mean:.4f}  （obs - cf）")

        tag = f"GANx{int(ratio)}"

        out_csv = os.path.join(OUT_DIR, f"deepSCM_result_{safe_city}_{tag}.csv")
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"已保存：{out_csv}")

        # obs vs cf
        plt.figure(figsize=(12, 5))
        plt.plot(out["year_week"], out["pm25_obs"], label="Observed")
        plt.plot(out["year_week"], out["pm25_cf"], label=f"Counterfactual (deepSCM, {tag})")
        plt.axvline(t0, linestyle="--")
        plt.legend()
        plt.title(f"deepSCM: {treated_city} | {tag} | pre_RMSPE={pre_rmspe:.2f} | post_mean_ATT={post_att_mean:.2f}")
        plt.tight_layout()
        fig1 = os.path.join(OUT_DIR, f"deepSCM_obs_vs_cf_{safe_city}_{tag}.png")
        plt.savefig(fig1, dpi=200)
        plt.close()
        print(f"已保存：{fig1}")

        # ATT
        plt.figure(figsize=(12, 4))
        plt.plot(out["year_week"], out["att"], label="ATT = obs - cf")
        plt.axhline(0, linestyle="--")
        plt.axvline(t0, linestyle="--")
        plt.legend()
        plt.title(f"deepSCM ATT over time | {tag}")
        plt.tight_layout()
        fig2 = os.path.join(OUT_DIR, f"deepSCM_att_{safe_city}_{tag}.png")
        plt.savefig(fig2, dpi=200)
        plt.close()
        print(f"已保存：{fig2}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results_rows.append({
            "file": os.path.basename(data_path),
            "city_folder": city_from_name,
            "treated_city": treated_city,
            "treated_fips": treated_fips,
            "t0": t0,
            "gan_ratio": int(ratio),
            "n_real_donors": int(n_real),
            "n_gen": int(n_gen),
            "pre_rmspe": float(pre_rmspe),
            "post_att_mean": float(post_att_mean),
            "out_dir": OUT_DIR,
        })

    return results_rows  # ✅ 返回该城市的 5 行结果

# =========================
# 7) 主函数：遍历源数据目录下所有城市
# =========================
def main():
    if not os.path.isdir(DATA_DIR):
        raise ValueError(f"DATA_DIR 不存在：{DATA_DIR}")

    files = []
    for fn in os.listdir(DATA_DIR):
        if not fn.endswith(FILE_SUFFIX):
            continue
        if FILE_KEYWORD not in fn:
            continue
        files.append(os.path.join(DATA_DIR, fn))
    files = sorted(files)

    if not files:
        raise ValueError(f"在 {DATA_DIR} 未找到匹配文件：包含关键字 '{FILE_KEYWORD}' 且后缀 '{FILE_SUFFIX}'")

    print(f"[Batch] 共发现 {len(files)} 个城市文件，将输出到：{OUT_BASE_DIR}/城市名/")
    print(f"[Batch] GAN 已开启，倍率={GAN_RATIOS}")

    summary = []
    failed = []

    for p in files:
        try:
            rows = run_one_city_file(p)     # ✅ 一个城市返回 5 行
            summary.extend(rows)
        except Exception as e:
            print(f"[FAIL] {os.path.basename(p)} -> {e}")
            failed.append({"file": os.path.basename(p), "error": str(e)})

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(OUT_BASE_DIR, "deepSCM_batch_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n[Batch] 汇总结果已保存：{summary_csv}")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_csv = os.path.join(OUT_BASE_DIR, "deepSCM_batch_failed.csv")
        failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print(f"[Batch] 失败列表已保存：{failed_csv}")

if __name__ == "__main__":
    main()
