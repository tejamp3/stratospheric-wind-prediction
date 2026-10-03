"""Audit experiments: measure the weaknesses instead of asserting them.

Runs against the existing repo's data and checkpoints and writes only
docs/audit_results.json, plus a printed report.

Usage:  python docs/audit_experiments.py      (about 2 minutes on CPU)
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = "M:/red"
sys.path.insert(0, ROOT + "/src")
import config as C  # noqa: E402
import data as D  # noqa: E402
from evaluate import load_model, predict, physical  # noqa: E402
from model import RidgeBaseline, WindLSTM  # noqa: E402

OUT = {}
rng = np.random.default_rng(0)


def spd(z):
    return np.hypot(z[..., 0], z[..., 1])


def srmse(p, t):
    return float(np.sqrt(np.mean((spd(p) - spd(t)) ** 2)))


def vrmse(p, t):
    return float(np.sqrt(np.mean(np.sum((p - t) ** 2, axis=-1))))


# ------------------------------------------------------------------ base data
raw = D.open_raw()
print("raw variables:", list(raw.data_vars), "| levels:", raw.pressure_level.values.tolist(),
      "| grid:", raw.sizes["latitude"], "x", raw.sizes["longitude"])
OUT["raw_variables"] = list(raw.data_vars)
OUT["raw_levels"] = [float(x) for x in raw.pressure_level.values]

df, _ = D.clean(D.add_derived(D.station_frame(raw)))
ds, sx, sy, _ = D.build_dataset(df=df, verbose=False)
model, ck = load_model("base")
pred, true, pers = physical(ds["test"], predict(model, ds["test"]["X"]), sx, sy)
ridge = sy.inverse(RidgeBaseline().fit(ds["train"]["X"], ds["train"]["Y"])
                   .predict(ds["test"]["X"]).reshape(-1, 2)).reshape(pred.shape)
t_test = ds["test"]["time"]
for name in ("train", "val", "test"):
    tt = ds[name]["time"]
    print(f"{name}: {tt[0]} -> {tt[-1]}  ({len(tt)} windows)")
OUT["splits"] = {k: [str(ds[k]["time"][0]), str(ds[k]["time"][-1]), int(len(ds[k]["time"]))]
                 for k in ("train", "val", "test")}

# --------------------------------------------- B. stronger simple baselines
tr_end = ds["train"]["time"][-1]
train_df = df.loc[:tr_end]
# hour-of-day climatology fitted on TRAIN only
hod = train_df[["u50", "v50"]].groupby(train_df.index.hour).mean()
hod_anom = hod - hod.mean()
month_hod = train_df[["u50", "v50"]].groupby([train_df.index.month, train_df.index.hour]).mean()

issue_hours = pd.DatetimeIndex(t_test).hour.to_numpy()
diurnal = np.empty_like(pers)
clim = np.empty_like(pers)
for k, h in enumerate(C.HORIZONS):
    tgt_times = pd.DatetimeIndex(t_test) + pd.Timedelta(hours=h)
    diurnal[:, k, :] = (pers[:, k, :]
                        + hod_anom.loc[tgt_times.hour].to_numpy()
                        - hod_anom.loc[issue_hours].to_numpy())
    clim[:, k, :] = month_hod.loc[list(zip(tgt_times.month, tgt_times.hour))].to_numpy()

rows = []
for k, h in enumerate(C.HORIZONS):
    p = srmse(pers[:, k], true[:, k])
    pv = vrmse(pers[:, k], true[:, k])
    r = {"h": h}
    for name, f in (("persistence", pers), ("diurnal_persistence", diurnal),
                    ("climatology_month_hour", clim), ("ridge", ridge), ("lstm", pred)):
        r[f"{name}_speed_rmse"] = srmse(f[:, k], true[:, k])
        r[f"{name}_vector_rmse"] = vrmse(f[:, k], true[:, k])
        r[f"{name}_vector_skill_pct"] = (pv - r[f"{name}_vector_rmse"]) / pv * 100
        bias = (f[:, k] - true[:, k]).mean(0)
        r[f"{name}_bias_ms"] = float(np.hypot(*bias))
    rows.append(r)
base_tbl = pd.DataFrame(rows)
OUT["baselines"] = base_tbl.to_dict("records")
print("\nVECTOR RMSE (m/s) by baseline")
print(base_tbl[["h"] + [c for c in base_tbl if c.endswith("vector_rmse")]].round(2).to_string(index=False))
print("\nVECTOR SKILL vs plain persistence (%)")
print(base_tbl[["h"] + [c for c in base_tbl if c.endswith("vector_skill_pct")]].round(1).to_string(index=False))
print("\nBIAS magnitude (m/s)")
print(base_tbl[["h"] + [c for c in base_tbl if c.endswith("bias_ms")]].round(2).to_string(index=False))

# ----------------------------------------------- D. block-bootstrap CIs
BLOCK = 8 * 8  # 8 days of 3-hourly windows; autocorrelation is days long
n = len(true)
nblocks = int(np.ceil(n / BLOCK))
starts_all = np.arange(0, n - BLOCK + 1)
B = 2000
ci = {}
for k, h in enumerate(C.HORIZONS):
    sk_l, d_lr, sk_r = [], [], []
    for _ in range(B):
        st = rng.choice(starts_all, nblocks)
        idx = (st[:, None] + np.arange(BLOCK)).ravel()[:n]
        pl, pr, pp = (srmse(f[idx, k], true[idx, k]) for f in (pred, ridge, pers))
        sk_l.append((pp - pl) / pp * 100)
        sk_r.append((pp - pr) / pp * 100)
        d_lr.append(pl - pr)
    ci[h] = {"lstm_skill_ci95": np.percentile(sk_l, [2.5, 97.5]).round(1).tolist(),
             "ridge_skill_ci95": np.percentile(sk_r, [2.5, 97.5]).round(1).tolist(),
             "lstm_minus_ridge_rmse_ci95": np.percentile(d_lr, [2.5, 97.5]).round(3).tolist()}
OUT["bootstrap"] = ci
OUT["effective_independent_blocks"] = nblocks
print(f"\nBLOCK BOOTSTRAP ({B} resamples, 8-day blocks, {nblocks} blocks in test)")
for h, v in ci.items():
    print(h, v)

# ------------------------------------------------------ E. seed variance
def train_seed(seed, epochs=100, patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = WindLSTM(n_features=9, hidden=128, num_layers=1, dropout=0.2, residual=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lf = nn.MSELoss()
    Xtr, Ytr = torch.from_numpy(ds["train"]["X"]), torch.from_numpy(ds["train"]["Y"])
    Xva, Yva = torch.from_numpy(ds["val"]["X"]), torch.from_numpy(ds["val"]["Y"])
    best, bad, best_state, best_ep = 1e9, 0, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = lf(m(Xtr[b]), Ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = lf(m(Xva), Yva).item()
        sched.step(vl)
        if vl < best - 1e-6:
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    p, _, _ = physical(ds["test"], predict(m, ds["test"]["X"]), sx, sy)
    return best, best_ep, p


seed_rows = []
t0 = time.time()
for seed in range(5):
    vl, ep, p = train_seed(seed)
    r = {"seed": seed, "val_loss": vl, "best_epoch": ep}
    for k, h in enumerate(C.HORIZONS):
        pp = srmse(pers[:, k], true[:, k])
        r[f"skill_{h}h"] = (pp - srmse(p[:, k], true[:, k])) / pp * 100
        r[f"bias_{h}h_kmday"] = float(np.hypot(*(p[:, k] - true[:, k]).mean(0)) * 86.4)
    seed_rows.append(r)
    print("seed", seed, {k: round(v, 3) for k, v in r.items()})
seeds = pd.DataFrame(seed_rows)
OUT["seeds"] = seed_rows
print(f"seed runs took {(time.time() - t0) / 60:.1f} min")

# ------------------------------------------- F. spatial generalisation
def site_frame(lat, lon):
    f, _ = D.clean(D.add_derived(D.station_frame(raw, lat=lat, lon=lon)))
    return f


sites = [(20.0, 80.0), (30.0, 70.0), (10.0, 90.0), (16.5, 80.5), (25.0, 95.0)]
space = []
for lat, lon in sites:
    f = site_frame(lat, lon)
    feats = f[D.FEATURES].to_numpy("float32")
    targs = f[D.TARGETS].to_numpy("float32")
    tr, va, te = D.time_splits(len(f))
    Xte, Yte = D.make_windows(sx.transform(feats[te]), sy.transform(targs[te]))
    p_tr, tt, pb = physical({"X": Xte, "Y": Yte}, predict(model, Xte), sx, sy)
    rid_tr = sy.inverse(RidgeBaseline().fit(ds["train"]["X"], ds["train"]["Y"])
                        .predict(Xte).reshape(-1, 2)).reshape(p_tr.shape)
    # locally refitted ridge
    dsl, sxl, syl, _ = D.build_dataset(df=f, verbose=False)
    rid_loc = syl.inverse(RidgeBaseline().fit(dsl["train"]["X"], dsl["train"]["Y"])
                          .predict(dsl["test"]["X"]).reshape(-1, 2)).reshape(p_tr.shape)
    _, tl, pl = physical(dsl["test"], dsl["test"]["Y"], sxl, syl)
    r = {"site": f"{lat:.1f}N {lon:.1f}E"}
    k = 0
    pp = vrmse(pb[:, k], tt[:, k])
    r["persist_vrmse_6h"] = pp
    r["lstm_transferred_skill_6h"] = (pp - vrmse(p_tr[:, k], tt[:, k])) / pp * 100
    r["ridge_transferred_skill_6h"] = (pp - vrmse(rid_tr[:, k], tt[:, k])) / pp * 100
    ppl = vrmse(pl[:, k], tl[:, k])
    r["ridge_local_skill_6h"] = (ppl - vrmse(rid_loc[:, k], tl[:, k])) / ppl * 100
    space.append(r)
sp = pd.DataFrame(space)
OUT["spatial"] = space
print("\nSPATIAL TRANSFER (6 h vector skill vs persistence, %)")
print(sp.round(1).to_string(index=False))

# ------------------------------- G. station keeping with position feedback
def track(w, plan, cap=12.0, feedback_tau_h=None, step_h=3):
    pos = np.zeros(2)
    out = [0.0]
    for i in range(len(w)):
        cmd = -plan[i].copy()
        if feedback_tau_h:
            cmd += -pos * 1000 / (feedback_tau_h * 3600)  # m/s back towards station
        mag = np.hypot(*cmd)
        if mag > cap:
            cmd *= cap / mag
        pos = pos + (w[i] + cmd) * step_h * 3.6
        out.append(np.hypot(*pos))
    return np.array(out)


k6 = 0
w6, l6, p6 = true[:, k6], pred[:, k6], pers[:, k6]
N5 = 40
res = {f"{c}_{fb}": [] for c in ("perfect", "lstm", "persist") for fb in ("open", "fb")}
for s0 in range(0, len(w6) - N5 - 1, 8):
    q = slice(s0, s0 + N5)
    for c, plan in (("perfect", w6), ("lstm", l6), ("persist", p6)):
        res[f"{c}_open"].append(track(w6[q], plan[q]).max())
        res[f"{c}_fb"].append(track(w6[q], plan[q], feedback_tau_h=6).max())
succ = {k: round(float((np.array(v) <= 200).mean() * 100), 1) for k, v in res.items()}
OUT["station_keeping_feedback"] = succ
print("\n5-DAY MISSIONS HELD WITHIN 200 km (%), open loop vs position feedback")
print(succ)

# --------------------------- H. is a single column enough for a free balloon?
u = raw["u"].sel(pressure_level=50).load().values  # (T, lat, lon)
v = raw["v"].sel(pressure_level=50).load().values
lats = raw.latitude.values
lons = raw.longitude.values
tt_all = pd.DatetimeIndex(raw.valid_time.values)
lat_desc = lats[0] > lats[-1]


def interp(field, ti, la, lo):
    # bilinear in space, nearest 3-hourly time step
    li = np.interp(la, lats[::-1] if lat_desc else lats, np.arange(len(lats))[::-1] if lat_desc else np.arange(len(lats)))
    lj = np.interp(lo, lons, np.arange(len(lons)))
    i0, j0 = int(np.floor(li)), int(np.floor(lj))
    i1, j1 = min(i0 + 1, len(lats) - 1), min(j0 + 1, len(lons) - 1)
    a, b = li - i0, lj - j0
    f = field[ti]
    return ((1 - a) * (1 - b) * f[i0, j0] + a * (1 - b) * f[i1, j0]
            + (1 - a) * b * f[i0, j1] + a * b * f[i1, j1])


exit_h, disp24 = [], []
starts = rng.choice(np.arange(len(tt_all) - 30), 300, replace=False)
for s in starts:
    la, lo = 20.0, 80.0
    left = None
    for step in range(24):  # up to 72 h in 3 h steps
        uu, vv = interp(u, s + step, la, lo), interp(v, s + step, la, lo)
        la += vv * 3 * 3600 / 111_000
        lo += uu * 3 * 3600 / (111_000 * np.cos(np.radians(la)))
        if step == 7:
            disp24.append(np.hypot((la - 20) * 111, (lo - 80) * 111 * np.cos(np.radians(20))))
        if not (5 <= la <= 35 and 60 <= lo <= 100):
            left = (step + 1) * 3
            break
    exit_h.append(left if left else np.inf)
exit_h = np.array(exit_h)
OUT["free_balloon"] = {
    "median_24h_displacement_km": float(np.median(disp24)),
    "p90_24h_displacement_km": float(np.percentile(disp24, 90)),
    "frac_exit_domain_within_72h": float(np.isfinite(exit_h).mean()),
    "median_speed_50hPa_at_station_ms": float(np.median(df["speed50"])),
}
print("\nFREE-DRIFTING BALLOON FROM 20N 80E AT 50 hPa (300 random starts)")
print(OUT["free_balloon"])

json.dump(OUT, open(ROOT + "/docs/audit_results.json", "w"), indent=1, default=str)
print("\nsaved audit_results.json")
