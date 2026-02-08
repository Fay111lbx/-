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
# 0) 你只需要改这里
# =========================
DATA_PATH = "/root/data/treatedpass-heating/源数据/临汾_panel_vs_controls_weekly_2018_2022_with_year_week_fips_with_cov.csv"
# 默认输出到项目根下以城市命名的文件夹
OUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "临汾")

# 是否启用 GAN 扩充 donor
# 启用 GAN 以生成额外 donor 序列
USE_GAN = True
N_GEN_RATIO = 5.0     # 生成 donor 数量 = N_GEN_RATIO * 真实donor数量（5.0=生成五倍数量）

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

    # week_end 作为时间索引（周末日期）
    df["week_end_dt"] = pd.to_datetime(df["week_end"], errors="coerce")
    if df["week_end_dt"].isna().any():
        raise ValueError("week_end 有无法解析的日期，请检查格式，如 2018/1/7")

    # year_week 用于排序/对齐（整数）
    df["year_week"] = (df["year"].astype(int) * 100 + df["week"].astype(int)).astype(int)

    # 协变量转数值
    # 协变量转数值，并对协变量做对数变换（处理非正值）
    for c in COV_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        ser = df[c]
        # 如果存在非空值则做对数变换；对非正值先加上偏移量
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
    # treated==1 出现过的城市就是处理城市（你这里应当是北京）
    treated_fips = (
        df.groupby("fips")["treated"].max()
        .loc[lambda s: s == 1]
        .index.tolist()
    )
    if len(treated_fips) != 1:
        raise ValueError(f"当前文件识别到 {len(treated_fips)} 个 treated 城市（应当只有北京一个）。识别结果：{treated_fips}")

    treated_fips = int(treated_fips[0])
    treated_city = df.loc[df["fips"] == treated_fips, "city"].dropna().iloc[0]

    # 政策时点：treated 城市第一次 treated==1 的时间
    t0 = (
        df.loc[(df["fips"] == treated_fips) & (df["treated"] == 1), "year_week"]
        .min()
    )
    if pd.isna(t0):
        raise ValueError("没找到 treated 城市 treated==1 的起点，检查 treated 列是否正确标注。")

    t0 = int(t0)
    return treated_fips, treated_city, t0


# =========================
# 3) 构造 deepSCM 的监督学习数据：
#    X_t = donors(t) + donor_cov_mean(t) + treated_cov(t)
#    y_t = treated_pm25(t)
# =========================
def build_matrices(df: pd.DataFrame, treated_fips: int, t0: int):
    # donor：除 treated 城市外的所有城市
    donor_fips = sorted([int(x) for x in df["fips"].dropna().unique() if int(x) != treated_fips])

    # wide: (time x city)
    wide_y = df.pivot(index="year_week", columns="fips", values="pm25").sort_index()

    # treated 序列
    y_treat = wide_y[treated_fips].copy()

    # donors 序列矩阵（time x n_donors）
    X_donors = wide_y[donor_fips].copy()

    # 协变量：treated 协变量 time x 4
    treat_cov = (
        df.loc[df["fips"] == treated_fips, ["year_week"] + COV_COLS]
        .drop_duplicates("year_week")
        .set_index("year_week")
        .sort_index()
    )

    # donor 协变量均值 time x 4
    donor_cov_mean = (
        df.loc[df["fips"].isin(donor_fips), ["year_week"] + COV_COLS]
        .groupby("year_week")[COV_COLS]
        .mean()
        .sort_index()
    )

    # 对齐公共时间轴（只保留三者都有的 time）
    common_times = wide_y.index.intersection(treat_cov.index).intersection(donor_cov_mean.index)
    y_treat = y_treat.loc[common_times]
    X_donors = X_donors.loc[common_times]
    treat_cov = treat_cov.loc[common_times]
    donor_cov_mean = donor_cov_mean.loc[common_times]

    # 如果有缺失，简单删除对应时间点（你前面清洗过一般不会有）
    keep_mask = (~y_treat.isna())
    keep_mask &= (~X_donors.isna().any(axis=1))
    keep_mask &= (~treat_cov.isna().any(axis=1))
    keep_mask &= (~donor_cov_mean.isna().any(axis=1))
    common_times = common_times[keep_mask.values]

    y_treat = y_treat.loc[common_times]
    X_donors = X_donors.loc[common_times]
    treat_cov = treat_cov.loc[common_times]
    donor_cov_mean = donor_cov_mean.loc[common_times]

    # 划分 pre/post
    pre_times = common_times[common_times < t0]
    post_times = common_times[common_times >= t0]

    if len(pre_times) < 30:
        raise ValueError(f"政策前时间点太少（{len(pre_times)}），deepSCM 不稳。")

    # 组装特征：donors(t) + donor_cov_mean(t) + treated_cov(t)
    def make_X(times):
        A = X_donors.loc[times].values  # (T, n_donors)
        B = donor_cov_mean.loc[times].values  # (T, 4)
        C = treat_cov.loc[times].values  # (T, 4)
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
# 4) GAN：学习 donor 全时期序列分布，生成更多 donor 序列（长度=全时期T）
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


def train_gan(real_sequences: np.ndarray, n_gen: int, epochs: int = 1500, batch_size: int = 32):
    """
    real_sequences: (n_donors, T_total) donor 城市的完整时间序列矩阵
    输出：gen_sequences: (n_gen, T_total)
    """
    n_donors, T_total = real_sequences.shape

    # 标准化到 GAN 更易训的尺度
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_sequences)

    z_dim = 64
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

            y_real = torch.ones(bs, 1, device=DEVICE) * 0.9  # label smoothing
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
            loss_g = bce(d_fake, y_real)  # 希望 D 认为它是 real

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if (ep + 1) % 300 == 0:
            print(f"[GAN] epoch {ep+1}/{epochs} | loss_d={loss_d.item():.4f} loss_g={loss_g.item():.4f}")

    # 生成 n_gen
    with torch.no_grad():
        z = torch.randn(n_gen, z_dim, device=DEVICE)
        gen_scaled = G(z).cpu().numpy()

    gen = scaler.inverse_transform(gen_scaled)
    return gen  # (n_gen, T_total)


# =========================
# 5) DNN：学习非线性映射 F，构造反事实
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


def train_regressor(X_pre, y_pre, X_all, epochs=4000, batch_size=64):
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


# ========== 7) 可复用的运行接口：对任意 treated_fips 和 t0 运行 deepSCM ==========
def run_deepscm_for(df, treated_fips, t0, save_prefix=None, use_gan=USE_GAN, n_gen_ratio=N_GEN_RATIO):
    X_pre, y_pre, X_all, y_all, X_donors_wide, meta = build_matrices(df, treated_fips, t0)
    donor_fips = meta["donor_fips"]
    common_times = meta["common_times"]
    pre_times = meta["pre_times"]
    post_times = meta["post_times"]

    # GAN 扩充
    if use_gan:
        donors_full = X_donors_wide.loc[common_times].values.T  # (n_donors, T)
        n_gen = int(len(donor_fips) * n_gen_ratio)
        print(f"GAN 扩充：真实donor={len(donor_fips)}，生成donor={n_gen}")
        gen_full = train_gan(donors_full, n_gen=n_gen, epochs=1500, batch_size=32)
        gen_full = gen_full.T  # (T, n_gen)

        n_real = len(donor_fips)
        X_pre_real = X_pre[:, :n_real]
        X_all_real = X_all[:, :n_real]

        idx_pre = [np.where(common_times == t)[0][0] for t in pre_times]
        idx_all = list(range(len(common_times)))

        gen_pre = gen_full[idx_pre, :]
        gen_all = gen_full[idx_all, :]

        cov_pre = X_pre[:, n_real:]
        cov_all = X_all[:, n_real:]

        X_pre = np.concatenate([X_pre_real, gen_pre, cov_pre], axis=1)
        X_all = np.concatenate([X_all_real, gen_all, cov_all], axis=1)

    # DNN 预测
    yhat_all, _ = train_regressor(X_pre, y_pre, X_all, epochs=4000, batch_size=64)

    out = pd.DataFrame({
        "year_week": common_times.astype(int),
        "pm25_obs": y_all.reshape(-1),
        "pm25_cf": yhat_all,
    })
    out["att"] = out["pm25_obs"] - out["pm25_cf"]
    out["period"] = np.where(out["year_week"] < t0, "pre", "post")

    pre_rmspe = np.sqrt(np.mean((out.loc[out["period"] == "pre", "pm25_obs"] - out.loc[out["period"] == "pre", "pm25_cf"]) ** 2))
    post_att_mean = out.loc[out["period"] == "post", "att"].mean()

    res = {
        "out": out,
        "pre_rmspe": pre_rmspe,
        "post_att_mean": post_att_mean,
        "donor_fips": donor_fips,
        "pre_times": pre_times,
        "post_times": post_times,
    }

    # 保存（可选前缀）
    if save_prefix is not None:
        csvp = os.path.join(OUT_DIR, f"{save_prefix}_result.csv")
        out.to_csv(csvp, index=False, encoding="utf-8-sig")
        print(f"已保存：{csvp}")

    return res


# ========== 8) 安慰剂检验：时间安慰剂（time-placebo）和个体安慰剂（unit-placebo） ==========
def time_placebo_tests(df, treated_fips, t0, n_placebos=4, n_gen_ratio=N_GEN_RATIO):
    # 选择若干个在原始 t0 之前的时间点作为伪政策时点
    X_pre, y_pre, X_all, y_all, X_donors_wide, meta = build_matrices(df, treated_fips, t0)
    common_times = meta["common_times"]
    idx0 = int(np.where(common_times == t0)[0][0])

    # 选取间隔为 4 的几个伪时点（如果可用）
    shifts = [4, 8, 12, 16][:n_placebos]
    placebo_t0s = []
    for s in shifts:
        idx = idx0 - s
        if idx >= 30:  # 保证伪处理前仍有足够的 pre 样本
            placebo_t0s.append(common_times[idx])

    results = []
    for pt in placebo_t0s:
        print(f"运行时间安慰剂 t0={pt}")
        r = run_deepscm_for(df, treated_fips, int(pt), save_prefix=f"time_placebo_{int(pt)}", n_gen_ratio=n_gen_ratio)
        results.append({"placebo_t0": int(pt), "post_att_mean": r["post_att_mean"], "pre_rmspe": r["pre_rmspe"]})

    df_res = pd.DataFrame(results)
    csvp = os.path.join(OUT_DIR, "deepSCM_time_placebo.csv")
    df_res.to_csv(csvp, index=False, encoding="utf-8-sig")
    print(f"时间安慰剂结果已保存：{csvp}")

    # 绘图：placebo t0 vs post_att
    plt.figure()
    plt.bar(df_res["placebo_t0"].astype(str), df_res["post_att_mean"])
    plt.axhline(0, color="k", linestyle="--")
    plt.title("Time-placebo: post ATT for placebo t0s")
    plt.tight_layout()
    figp = os.path.join(OUT_DIR, "deepSCM_time_placebo_att.png")
    plt.savefig(figp, dpi=200)
    plt.close()
    print(f"时间安慰剂图已保存：{figp}")

    return df_res


def unit_placebo_tests(df, treated_fips, t0, n_gen_ratio=N_GEN_RATIO):
    # 对每个 donor 当作 treated 运行一次
    X_pre, y_pre, X_all, y_all, X_donors_wide, meta = build_matrices(df, treated_fips, t0)
    donor_fips = meta["donor_fips"]

    results = []
    for dfips in donor_fips:
        try:
            print(f"运行个体安慰剂：treated_fips={dfips}")
            r = run_deepscm_for(df, int(dfips), t0, save_prefix=f"unit_placebo_{dfips}", n_gen_ratio=n_gen_ratio)
            results.append({"placebo_fips": int(dfips), "post_att_mean": r["post_att_mean"], "pre_rmspe": r["pre_rmspe"]})
        except Exception as e:
            print(f"跳过 {dfips}：{e}")

    df_res = pd.DataFrame(results)
    csvp = os.path.join(OUT_DIR, "deepSCM_unit_placebo.csv")
    df_res.to_csv(csvp, index=False, encoding="utf-8-sig")
    print(f"个体安慰剂结果已保存：{csvp}")

    # 绘图：直方图
    plt.figure()
    plt.hist(df_res["post_att_mean"].dropna(), bins=30)
    plt.axvline(0, color="k", linestyle="--")
    plt.title("Unit-placebo: distribution of post ATT")
    plt.tight_layout()
    figp = os.path.join(OUT_DIR, "deepSCM_unit_placebo_hist.png")
    plt.savefig(figp, dpi=200)
    plt.close()
    print(f"个体安慰剂图已保存：{figp}")

    return df_res


# =========================
# 6) 主函数：deepSCM 全流程
# =========================
def main():
    df = load_and_prepare(DATA_PATH)
    treated_fips, treated_city, t0 = infer_treated_unit_and_t0(df)
    print(f"识别处理城市：{treated_city} (fips={treated_fips})")
    print(f"识别政策起点 year_week = {t0}")

    # 构造矩阵
    X_pre, y_pre, X_all, y_all, X_donors_wide, meta = build_matrices(df, treated_fips, t0)
    donor_fips = meta["donor_fips"]
    common_times = meta["common_times"]
    pre_times = meta["pre_times"]
    post_times = meta["post_times"]

    # ========== GAN 扩充 donor（可选） ==========
    if USE_GAN:
        # donor 完整序列矩阵：shape (n_donors, T_total)
        donors_full = X_donors_wide.loc[common_times].values  # (T, n_donors)
        donors_full = donors_full.T  # (n_donors, T)

        n_gen = int(len(donor_fips) * N_GEN_RATIO)
        print(f"GAN 扩充开启：真实donor={len(donor_fips)}，生成donor={n_gen}，序列长度T={donors_full.shape[1]}")

        gen_full = train_gan(donors_full, n_gen=n_gen, epochs=1500, batch_size=32)  # (n_gen, T)
        gen_full = gen_full.T  # (T, n_gen)

        # 将生成 donor 的值拼到 donors(t) 后面，形成新的特征矩阵
        # 注意：我们只把 “生成donor的pm25” 加进去，不为生成donor提供协变量（否则需要联合生成协变量）
        # 所以：X_t = [real_donors(t), gen_donors(t), donor_cov_mean(t), treated_cov(t)]
        # ——我们这里直接在 X_pre / X_all 的左侧 donors 块追加 gen 块
        n_real = len(donor_fips)
        X_pre_real = X_pre[:, :n_real]
        X_all_real = X_all[:, :n_real]

        # 取对应时间的生成 donor 值
        idx_pre = [np.where(common_times == t)[0][0] for t in pre_times]
        idx_all = list(range(len(common_times)))

        gen_pre = gen_full[idx_pre, :]  # (T_pre, n_gen)
        gen_all = gen_full[idx_all, :]  # (T_all, n_gen)

        # 原 X 的后 8 列是 cov（donor_cov_mean 4 + treated_cov 4）
        cov_pre = X_pre[:, n_real:]
        cov_all = X_all[:, n_real:]

        X_pre = np.concatenate([X_pre_real, gen_pre, cov_pre], axis=1)
        X_all = np.concatenate([X_all_real, gen_all, cov_all], axis=1)

        print(f"加入生成donor后：X_pre dim={X_pre.shape}, X_all dim={X_all.shape}")
    else:
        print("GAN 扩充关闭：仅使用真实 donor + 协变量")

    # ========== DNN 学习 F 并预测反事实 ==========
    yhat_all, _ = train_regressor(X_pre, y_pre, X_all, epochs=2000, batch_size=64)

    # 输出结果表
    out = pd.DataFrame({
        "year_week": common_times.astype(int),
        "pm25_obs": y_all.reshape(-1),
        "pm25_cf": yhat_all,
    })
    out["att"] = out["pm25_obs"] - out["pm25_cf"]
    out["period"] = np.where(out["year_week"] < t0, "pre", "post")

    # pre RMSPE / post 平均ATT
    pre_rmspe = np.sqrt(np.mean((out.loc[out["period"] == "pre", "pm25_obs"] - out.loc[out["period"] == "pre", "pm25_cf"]) ** 2))
    post_att_mean = out.loc[out["period"] == "post", "att"].mean()

    print(f"\n[诊断] 政策前 RMSPE = {pre_rmspe:.4f}")
    print(f"[结果] 政策后 平均ATT(周度均值) = {post_att_mean:.4f}  （obs - cf）")

    # 保存 CSV
    out_csv = os.path.join(OUT_DIR, "deepSCM_result_linfen.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已保存：{out_csv}")

    # 画图：obs vs cf
    plt.figure(figsize=(12, 5))
    plt.plot(out["year_week"], out["pm25_obs"], label="Observed")
    plt.plot(out["year_week"], out["pm25_cf"], label="Counterfactual (deepSCM)")
    plt.axvline(t0, linestyle="--")
    plt.legend()
    plt.title(f"deepSCM: {treated_city} | pre_RMSPE={pre_rmspe:.2f} | post_mean_ATT={post_att_mean:.2f}")
    plt.tight_layout()
    fig1 = os.path.join(OUT_DIR, "deepSCM_obs_vs_cf.png")
    plt.savefig(fig1, dpi=200)
    plt.close()
    print(f"已保存：{fig1}")

    # 画图：ATT
    plt.figure(figsize=(12, 4))
    plt.plot(out["year_week"], out["att"], label="ATT = obs - cf")
    plt.axhline(0, linestyle="--")
    plt.axvline(t0, linestyle="--")
    plt.legend()
    plt.title("deepSCM ATT over time")
    plt.tight_layout()
    fig2 = os.path.join(OUT_DIR, "deepSCM_att.png")
    plt.savefig(fig2, dpi=200)
    plt.close()
    print(f"已保存：{fig2}")


if __name__ == "__main__":
    main()