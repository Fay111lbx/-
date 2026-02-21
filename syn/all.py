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
# 0) 你只需要改这里（单文件：合成处理组）
# =========================
# 输入文件：由你之前生成的合成处理组数据
DATA_PATH = "/root/data/合成/combined_final.csv"

# donor 筛选（对照组）
# 说明：35 µg/m³ 是 PM2.5 国家二级标准的“年均值”阈值；这里用作周度序列的 donor 可比性筛选规则。
APPLY_DONOR_PRE_MEAN_FILTER = True
DONOR_PRE_MEAN_MIN_PM25 = 35.0

# donor 筛选口径：
# - "overall"：政策前所有周的整体均值
# - "yearly" ：政策前按“年(ISO year)”分组计算年均值；若某 donor 在政策前所有年份的年均都 < 阈值（等价于 max(年均) < 阈值）则剔除
DONOR_PRE_MEAN_FILTER_MODE = "yearly"

# 输出目录：为避免覆盖旧结果，开启 donor 筛选时输出到新目录
OUT_BASE_DIR = os.path.join(
    os.path.dirname(DATA_PATH),
    (
        "deepSCM_synthetic_filter35_yearly"
        if (APPLY_DONOR_PRE_MEAN_FILTER and DONOR_PRE_MEAN_FILTER_MODE == "yearly")
        else ("deepSCM_synthetic_filter35" if APPLY_DONOR_PRE_MEAN_FILTER else "deepSCM_synthetic")
    ),
)

# treated 单元：city == 该值
TREATED_CITY_NAME = "合成"

# 同时跑两套：不开 GAN + 开 GAN(1~5 倍)
RUN_NO_GAN = True
RUN_GAN = True

# GAN 生成样本数倍率：n_gen = ratio * 真实donor数
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
REQUIRED = ["city", "treated", "pm25", "time"]

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"缺少列：{miss}\n当前列：{list(df.columns)}")

    df = df.copy()
    df["treated"] = pd.to_numeric(df["treated"], errors="coerce").fillna(0).astype(int)
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        bad = df.loc[df["time"].isna()].head(5)
        raise ValueError(f"time 有无法解析的日期，示例行：\n{bad}")

    # 用 ISO 周来生成 year/week（如果文件里已有 year/week 也不依赖它们）
    iso = df["time"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["week"] = iso.week.astype(int)
    df["year_week"] = (df["year"] * 100 + df["week"]).astype(int)

    df = df.sort_values(["time", "city"]).reset_index(drop=True)
    return df

def infer_treated_unit_and_t0(df: pd.DataFrame, treated_city_name: str):
    if treated_city_name not in set(df["city"].astype(str)):
        raise ValueError(f"未找到 treated 城市：{treated_city_name}。请检查 city 列是否包含该值。")

    t0 = df.loc[(df["city"] == treated_city_name) & (df["treated"] == 1), "time"].min()
    if pd.isna(t0):
        raise ValueError("没找到 treated 城市 treated==1 的起点，检查 treated 列是否正确标注。")

    return treated_city_name, pd.Timestamp(t0)

# =========================
# 3) 构造 deepSCM 的监督学习数据
# =========================
def build_matrices(df: pd.DataFrame, treated_city: str, t0_time: pd.Timestamp):
    donor_cities = sorted([c for c in df["city"].dropna().astype(str).unique().tolist() if c != treated_city])
    wide_y = df.pivot(index="time", columns="city", values="pm25").sort_index()

    if treated_city not in wide_y.columns:
        raise ValueError(f"treated 城市 {treated_city} 不在透视表列中，无法建模")

    # donor 候选表
    donor_df = wide_y[donor_cities]

    # 1) 丢弃存在缺失值的 donor（否则城市多时很容易把时间点筛空）
    donor_complete = donor_df.columns[donor_df.notna().all(axis=0)].tolist()
    donor_df = donor_df[donor_complete]

    dropped_by_pre_mean = []
    donor_pre_filter_stats = None
    if APPLY_DONOR_PRE_MEAN_FILTER:
        pre_mask = donor_df.index < t0_time
        if pre_mask.sum() == 0:
            raise ValueError("donor 筛选失败：政策前时间点为 0，无法计算 pre 均值")

        thr = float(DONOR_PRE_MEAN_MIN_PM25)
        mode = str(DONOR_PRE_MEAN_FILTER_MODE).lower().strip()

        if mode == "overall":
            pre_means = donor_df.loc[pre_mask].mean(axis=0, skipna=True)
            keep = pre_means[pre_means >= thr].index.tolist()
            dropped_by_pre_mean = pre_means[pre_means < thr].index.tolist()
            donor_pre_filter_stats = pd.DataFrame({"pre_overall_mean_pm25": pre_means})
            donor_df = donor_df[keep]
        elif mode == "yearly":
            pre_df = donor_df.loc[pre_mask]
            pre_years = pd.Series(pre_df.index, index=pre_df.index).dt.isocalendar().year.astype(int)
            pre_year_means = pre_df.groupby(pre_years).mean(numeric_only=True)  # index=year, columns=donor
            max_pre_year_mean = pre_year_means.max(axis=0, skipna=True)
            keep = max_pre_year_mean[max_pre_year_mean >= thr].index.tolist()
            dropped_by_pre_mean = max_pre_year_mean[max_pre_year_mean < thr].index.tolist()
            donor_pre_filter_stats = pd.DataFrame({"pre_yearly_max_mean_pm25": max_pre_year_mean})
            donor_df = donor_df[keep]
        else:
            raise ValueError(f"未知 DONOR_PRE_MEAN_FILTER_MODE={DONOR_PRE_MEAN_FILTER_MODE!r}，仅支持 'overall' 或 'yearly'")

    if donor_df.shape[1] < 5:
        raise ValueError(f"可用 donor 城市太少（{donor_df.shape[1]}），请检查 pm25 是否有缺失")

    y_treat = wide_y[treated_city].copy()

    common_times = wide_y.index
    keep_mask = (~y_treat.isna()) & (~donor_df.isna().any(axis=1))
    common_times = common_times[keep_mask.values]

    y_treat = y_treat.loc[common_times]
    X_donors = donor_df.loc[common_times]

    pre_times = common_times[common_times < t0_time]
    post_times = common_times[common_times >= t0_time]
    if len(pre_times) < 30:
        raise ValueError(f"政策前时间点太少（{len(pre_times)}），deepSCM 不稳。")

    X_pre = X_donors.loc[pre_times].values
    y_pre = y_treat.loc[pre_times].values.reshape(-1, 1)

    X_all = X_donors.loc[common_times].values
    y_all = y_treat.loc[common_times].values.reshape(-1, 1)

    meta = {
        "donor_cities": donor_df.columns.tolist(),
        "donor_dropped_pre_mean": dropped_by_pre_mean,
        "donor_filter_pre_mean_min_pm25": float(DONOR_PRE_MEAN_MIN_PM25) if APPLY_DONOR_PRE_MEAN_FILTER else None,
        "donor_filter_pre_mean_mode": str(DONOR_PRE_MEAN_FILTER_MODE) if APPLY_DONOR_PRE_MEAN_FILTER else None,
        "donor_pre_filter_stats": donor_pre_filter_stats,
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
# 6) 单次运行：treated=“合成”，controls=其他城市（可选 GAN 扩充 donor）
# =========================
def sanitize_name(s: str) -> str:
    return "".join(ch for ch in str(s) if ch not in r'\/:*?"<>|').strip()

def _make_out_table(time_index, y_obs, y_cf, t0_time: pd.Timestamp):
    out = pd.DataFrame({
        "time": pd.to_datetime(pd.Series(time_index)),
        "pm25_obs": y_obs,
        "pm25_cf": y_cf,
    })
    out["att"] = out["pm25_obs"] - out["pm25_cf"]
    out["period"] = np.where(out["time"] < t0_time, "pre", "post")

    iso = out["time"].dt.isocalendar()
    out["year"] = iso.year.astype(int)
    out["week"] = iso.week.astype(int)
    out["year_week"] = (out["year"] * 100 + out["week"]).astype(int)
    return out


def run_synthetic(data_path: str):
    os.makedirs(OUT_BASE_DIR, exist_ok=True)
    set_seed(SEED)

    df = load_and_prepare(data_path)
    treated_city, t0_time = infer_treated_unit_and_t0(df, TREATED_CITY_NAME)

    print("\n" + "=" * 90)
    print(f"[START] treated_city={treated_city}")
    print(f"       data={data_path}")
    print(f"       out ={OUT_BASE_DIR}")
    print(f"       t0  ={t0_time.date()}")
    print("=" * 90)

    X_pre0, y_pre, X_all0, y_all, X_donors_wide, meta = build_matrices(df, treated_city, t0_time)
    donor_cities = meta["donor_cities"]
    common_times = meta["common_times"]
    pre_times = meta["pre_times"]

    safe_city = sanitize_name(treated_city)
    OUT_DIR = os.path.join(OUT_BASE_DIR, safe_city)
    os.makedirs(OUT_DIR, exist_ok=True)

    n_real = len(donor_cities)
    print(f"可用 donor 城市数: {n_real}")

    # 导出 donor 列表（保留/剔除）
    pd.Series(donor_cities, name="donor_city").to_csv(
        os.path.join(OUT_DIR, "donor_cities_used.csv"), index=False, encoding="utf-8-sig"
    )
    dropped = meta.get("donor_dropped_pre_mean") or []

    # 更可读的剔除清单（含可选的统计量）
    dropped_csv = os.path.join(
        OUT_DIR,
        "donor_cities_dropped_pre_%s_lt_%.1f.csv"
        % (sanitize_name(meta.get("donor_filter_pre_mean_mode") or "overall"), float(meta.get("donor_filter_pre_mean_min_pm25") or DONOR_PRE_MEAN_MIN_PM25)),
    )
    stats = meta.get("donor_pre_filter_stats")
    if isinstance(stats, pd.DataFrame) and (len(stats.columns) >= 1):
        out_stats = stats.copy()
        out_stats.index.name = "donor_city"
        out_stats = out_stats.reset_index()
        out_stats["dropped"] = out_stats["donor_city"].isin(set(dropped))
        out_stats.to_csv(os.path.join(OUT_DIR, "donor_pre_filter_stats.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame({"donor_city": dropped}).to_csv(dropped_csv, index=False, encoding="utf-8-sig")
    else:
        pd.Series(dropped, name="donor_city").to_csv(dropped_csv, index=False, encoding="utf-8-sig")

    results_rows = []

    # 1) 不开 GAN
    if RUN_NO_GAN:
        print("\n" + "-" * 90)
        print(f"[RUN] {treated_city} | noGAN")
        print("-" * 90)

        yhat_all, _ = train_regressor(X_pre0, y_pre, X_all0, epochs=2000, batch_size=64)
        out = _make_out_table(common_times, y_all.reshape(-1), yhat_all, t0_time)

        pre_rmspe = float(np.sqrt(np.mean((out.loc[out["period"] == "pre", "pm25_obs"] - out.loc[out["period"] == "pre", "pm25_cf"]) ** 2)))
        post_att_mean = float(out.loc[out["period"] == "post", "att"].mean())
        print(f"[诊断] 政策前 RMSPE = {pre_rmspe:.4f}")
        print(f"[结果] 政策后 平均ATT(周度均值) = {post_att_mean:.4f}  （obs - cf）")

        tag = "noGAN"
        out_csv = os.path.join(OUT_DIR, f"deepSCM_result_{safe_city}_{tag}.csv")
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"已保存：{out_csv}")

        plt.figure(figsize=(12, 5))
        plt.plot(out["time"], out["pm25_obs"], label="Observed")
        plt.plot(out["time"], out["pm25_cf"], label="Counterfactual (deepSCM, noGAN)")
        plt.axvline(t0_time, linestyle="--")
        plt.legend()
        plt.title(f"deepSCM: {treated_city} | noGAN | pre_RMSPE={pre_rmspe:.2f} | post_mean_ATT={post_att_mean:.2f}")
        plt.tight_layout()
        fig1 = os.path.join(OUT_DIR, f"deepSCM_obs_vs_cf_{safe_city}_{tag}.png")
        plt.savefig(fig1, dpi=200)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(out["time"], out["att"], label="ATT = obs - cf")
        plt.axhline(0, linestyle="--")
        plt.axvline(t0_time, linestyle="--")
        plt.legend()
        plt.title("deepSCM ATT over time | noGAN")
        plt.tight_layout()
        fig2 = os.path.join(OUT_DIR, f"deepSCM_att_{safe_city}_{tag}.png")
        plt.savefig(fig2, dpi=200)
        plt.close()

        results_rows.append({
            "data": os.path.basename(data_path),
            "treated_city": treated_city,
            "t0": str(t0_time.date()),
            "mode": "noGAN",
            "gan_ratio": 0,
            "n_real_donors": int(n_real),
            "n_gen": 0,
            "pre_rmspe": pre_rmspe,
            "post_att_mean": post_att_mean,
            "out_dir": OUT_DIR,
            "donor_filter_pre_mean_min_pm25": meta.get("donor_filter_pre_mean_min_pm25"),
            "donor_filter_pre_mean_mode": meta.get("donor_filter_pre_mean_mode"),
            "donor_dropped_pre_mean_n": int(len(dropped)),
        })

    # 2) 开 GAN（训练一次，倍率 1~5）
    if RUN_GAN:
        donors_full = X_donors_wide.loc[common_times].values.T  # (n_donors, T)
        print("\n" + "-" * 90)
        print(f"[GAN] train | real_donors={n_real} | ratios={GAN_RATIOS}")
        print("-" * 90)
        G, gan_scaler, z_dim = train_gan_return_model(donors_full, epochs=1500, batch_size=32, z_dim=64)

        # index mapping（DatetimeIndex -> row idx）
        time_to_idx = {t: i for i, t in enumerate(list(common_times))}
        idx_pre = [time_to_idx[t] for t in list(pre_times)]
        idx_all = list(range(len(common_times)))

        for ratio in GAN_RATIOS:
            n_gen = int(n_real * int(ratio))
            print("\n" + "-" * 90)
            print(f"[RUN] {treated_city} | GANx{ratio} | n_gen={n_gen}")
            print("-" * 90)

            gen_full = gan_generate(G, gan_scaler, n_gen=n_gen, z_dim=z_dim, seed=SEED + int(ratio)).T  # (T, n_gen)
            gen_pre = gen_full[idx_pre, :]
            gen_all = gen_full[idx_all, :]

            X_pre = np.concatenate([X_pre0, gen_pre], axis=1)
            X_all = np.concatenate([X_all0, gen_all], axis=1)
            print(f"加入生成donor后：X_pre dim={X_pre.shape}, X_all dim={X_all.shape}")

            yhat_all, _ = train_regressor(X_pre, y_pre, X_all, epochs=2000, batch_size=64)
            out = _make_out_table(common_times, y_all.reshape(-1), yhat_all, t0_time)

            pre_rmspe = float(np.sqrt(np.mean((out.loc[out["period"] == "pre", "pm25_obs"] - out.loc[out["period"] == "pre", "pm25_cf"]) ** 2)))
            post_att_mean = float(out.loc[out["period"] == "post", "att"].mean())

            print(f"[诊断] 政策前 RMSPE = {pre_rmspe:.4f}")
            print(f"[结果] 政策后 平均ATT(周度均值) = {post_att_mean:.4f}  （obs - cf）")

            tag = f"GANx{int(ratio)}"
            out_csv = os.path.join(OUT_DIR, f"deepSCM_result_{safe_city}_{tag}.csv")
            out.to_csv(out_csv, index=False, encoding="utf-8-sig")

            plt.figure(figsize=(12, 5))
            plt.plot(out["time"], out["pm25_obs"], label="Observed")
            plt.plot(out["time"], out["pm25_cf"], label=f"Counterfactual (deepSCM, {tag})")
            plt.axvline(t0_time, linestyle="--")
            plt.legend()
            plt.title(f"deepSCM: {treated_city} | {tag} | pre_RMSPE={pre_rmspe:.2f} | post_mean_ATT={post_att_mean:.2f}")
            plt.tight_layout()
            fig1 = os.path.join(OUT_DIR, f"deepSCM_obs_vs_cf_{safe_city}_{tag}.png")
            plt.savefig(fig1, dpi=200)
            plt.close()

            plt.figure(figsize=(12, 4))
            plt.plot(out["time"], out["att"], label="ATT = obs - cf")
            plt.axhline(0, linestyle="--")
            plt.axvline(t0_time, linestyle="--")
            plt.legend()
            plt.title(f"deepSCM ATT over time | {tag}")
            plt.tight_layout()
            fig2 = os.path.join(OUT_DIR, f"deepSCM_att_{safe_city}_{tag}.png")
            plt.savefig(fig2, dpi=200)
            plt.close()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results_rows.append({
                "data": os.path.basename(data_path),
                "treated_city": treated_city,
                "t0": str(t0_time.date()),
                "mode": "GAN",
                "gan_ratio": int(ratio),
                "n_real_donors": int(n_real),
                "n_gen": int(n_gen),
                "pre_rmspe": pre_rmspe,
                "post_att_mean": post_att_mean,
                "out_dir": OUT_DIR,
                "donor_filter_pre_mean_min_pm25": meta.get("donor_filter_pre_mean_min_pm25"),
                "donor_filter_pre_mean_mode": meta.get("donor_filter_pre_mean_mode"),
                "donor_dropped_pre_mean_n": int(len(dropped)),
            })

    return results_rows

# =========================
# 7) 主函数：对合成处理组跑一次
# =========================
def main():
    if not os.path.isfile(DATA_PATH):
        raise ValueError(f"DATA_PATH 不存在：{DATA_PATH}")

    print(f"[Single] 输入文件：{DATA_PATH}")
    print(f"[Single] treated 城市：{TREATED_CITY_NAME}")
    print(f"[Single] 输出目录：{OUT_BASE_DIR}")
    print(f"[Single] RUN_NO_GAN={RUN_NO_GAN} | RUN_GAN={RUN_GAN} | GAN_RATIOS={GAN_RATIOS}")

    summary = run_synthetic(DATA_PATH)
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(OUT_BASE_DIR, "deepSCM_synthetic_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n[Single] 汇总结果已保存：{summary_csv}")

if __name__ == "__main__":
    main()
