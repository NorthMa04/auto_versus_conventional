import numpy as np
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


# =========================
# Reproducibility
# =========================
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# I/O
# =========================
def load_matrix(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()

    if first.startswith("%%MatrixMarket"):
        from scipy.io import mmread
        A = mmread(path)
        return A.toarray().astype(float)
    else:
        return np.loadtxt(path).astype(float)


# =========================
# Normalization
# =========================
def minmax_01(M, eps=1e-12):
    mn, mx = M.min(), M.max()
    if abs(mx - mn) < eps:
        return np.zeros_like(M)
    return (M - mn) / (mx - mn)


def scale_signed(M, eps=1e-12):
    s = np.max(np.abs(M))
    return M / (s + eps)


# =========================
# Algorithm 1: X, Z, Q
# =========================
def compute_X(W, A, alpha=0.5, beta=0.5):
    n = W.shape[0]
    X = np.zeros((n, n), dtype=float)

    B = A @ A
    neighbors = [set(np.where(A[i] > 0)[0]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j] != 0:
                common = neighbors[i] & neighbors[j]
                W1 = sum(W[i, m] + W[m, j] for m in common)
                X[i, j] = X[j, i] = alpha * W[i, j] + beta * W1
            else:
                X[i, j] = X[j, i] = alpha * W[i, j]
    return X


def compute_Z(A):
    B = A @ A
    Z = 0.5 * A + B
    np.fill_diagonal(Z, 0)
    return Z


def compute_modularity_matrix(W):
    k = W.sum(axis=1)
    twoW = W.sum()
    Q = W - np.outer(k, k) / (twoW + 1e-12)
    np.fill_diagonal(Q, 0)
    return Q


# =========================
# Modularity score
# =========================
def modularity_score(W, labels):
    n = W.shape[0]
    m = W.sum() / 2.0
    deg = W.sum(axis=1)

    Qv = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Qv += W[i, j] - deg[i] * deg[j] / (2.0 * m + 1e-12)
    return Qv / (2.0 * m + 1e-12)


# =========================
# Autoencoders
# =========================
class SigmoidAE(nn.Module):
    """For X and Z: inputs in [0,1]."""
    def __init__(self, d_in, d_h):
        super().__init__()
        self.enc = nn.Linear(d_in, d_h)
        self.dec = nn.Linear(d_h, d_in)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.enc(x))
        x_hat = self.act(self.dec(h))
        return x_hat, h


class LinearDecAE(nn.Module):
    """For Q: keep signed semantics (linear decoder)."""
    def __init__(self, d_in, d_h):
        super().__init__()
        self.enc = nn.Linear(d_in, d_h)
        self.dec = nn.Linear(d_h, d_in)
        self.act = nn.Sigmoid()

    def forward(self, x):
        h = self.act(self.enc(x))
        x_hat = self.dec(h)
        return x_hat, h


def kl_div(rho, rho_hat):
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


# =========================
# One-layer training (X/Q/Z)
# =========================
def train_one_layer_three_way(
    X, Q, Z, d_h,
    epochs=600, batch_size=16, lr=1e-3,
    rho=0.10, lam_sparse=1e-3,
    w_x=1.0, w_q=1.0, w_z=1.0
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_in, n = X.shape
    Xc = torch.tensor(X.T, dtype=torch.float32, device=device)
    Qc = torch.tensor(Q.T, dtype=torch.float32, device=device)
    Zc = torch.tensor(Z.T, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(Xc, Qc, Zc),
        batch_size=batch_size,
        shuffle=True
    )

    ae_x = SigmoidAE(d_in, d_h).to(device)
    ae_q = LinearDecAE(d_in, d_h).to(device)
    ae_z = SigmoidAE(d_in, d_h).to(device)

    opt = torch.optim.Adam(
        list(ae_x.parameters()) +
        list(ae_q.parameters()) +
        list(ae_z.parameters()),
        lr=lr
    )

    mse = nn.MSELoss()

    for _ in range(epochs):
        for xb, qb, zb in loader:
            xh, hx = ae_x(xb)
            qh, hq = ae_q(qb)
            zh, hz = ae_z(zb)

            kl_term = (
                kl_div(rho, hx.mean(0)).sum() +
                kl_div(rho, hq.mean(0)).sum() +
                kl_div(rho, hz.mean(0)).sum()
            )

            loss = (
                w_x * mse(xh, xb) +
                w_q * mse(qh, qb) +
                w_z * mse(zh, zb) +
                lam_sparse * kl_term
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        _, HX = ae_x(Xc)
        _, HQ = ae_q(Qc)
        _, HZ = ae_z(Zc)

    return HX.T.cpu().numpy(), HQ.T.cpu().numpy(), HZ.T.cpu().numpy()


# =========================
# Representation diagnostics
# =========================
def repr_stats(H, sat_eps=0.01):
    H = np.asarray(H)
    sat = float(np.mean((H < sat_eps) | (H > 1 - sat_eps)))
    var = float(np.var(H))
    return sat, var


# =========================
# Dimension schedule:
# - Slow drop chain: 64-48-32-24-16
# - 3 layers per stage
# - Optional one-time 8-dim bottleneck (ONLY 1 layer)
# - After bottleneck, go back to 16 and keep refining (avoid collapse)
# =========================
def build_dim_schedule(T_max, stage_dims, layers_per_stage=3,
                       use_bottleneck8=True, bottleneck_after_dim=16,
                       bottleneck_dim=8, bottleneck_layers=1,
                       post_bottleneck_dim=16):
    schedule = []
    for d in stage_dims:
        schedule.extend([d] * layers_per_stage)

    bottleneck_used = False
    out = []
    for t in range(1, T_max + 1):
        if t <= len(schedule):
            out.append(schedule[t - 1])
            continue

        # after staged compression finished, we are at the last stage dim
        current_floor = stage_dims[-1]

        if use_bottleneck8 and (not bottleneck_used) and current_floor == bottleneck_after_dim:
            # insert one-time bottleneck
            for _ in range(bottleneck_layers):
                if len(out) >= T_max:
                    break
                out.append(bottleneck_dim)
            bottleneck_used = True
            # if we still need more layers after bottleneck, fill with post_bottleneck_dim
            while len(out) < T_max:
                out.append(post_bottleneck_dim)
            break
        else:
            out.append(current_floor)

    return out[:T_max]


# =========================
# Main
# =========================
def main():
    set_seed(0)

    CONFIG = {
        "alpha": 0.5,
        "beta": 0.5,

        # ---- schedule params ----
        "stage_dims": [64, 48, 32, 24, 16],
        "layers_per_stage": 4,
        "use_bottleneck8": True,
        "bottleneck_layers": 1,      # ONLY 1 layer at 8
        "post_bottleneck_dim": 16,   # go back to 16 to keep T large safely
        "T_max": 25,

        # ---- training params (reasonable for lesmis/football) ----
        "epochs": 600,
        "batch_size": 16,
        "lr": 1e-3,
        "rho": 0.10,
        "lam_sparse": 1e-3,
        "w_x": 1.0,
        "w_q": 1.0,
        "w_z": 1.0,

        # ---- clustering ----
        "k_min": 2,
        "k_max": 14,
        "k_n_init": 50,
    }

    # change to "lesmis.txt" / "football.txt"
    input_path = Path("celegans_edges.txt")
    dataset_name = input_path.stem
    out_dir = Path(f"wcd_threeway_schedule_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = build_dim_schedule(
        T_max=CONFIG["T_max"],
        stage_dims=CONFIG["stage_dims"],
        layers_per_stage=CONFIG["layers_per_stage"],
        use_bottleneck8=CONFIG["use_bottleneck8"],
        bottleneck_after_dim=CONFIG["stage_dims"][-1],
        bottleneck_dim=8,
        bottleneck_layers=CONFIG["bottleneck_layers"],
        post_bottleneck_dim=CONFIG["post_bottleneck_dim"],
    )

    t0 = time.perf_counter()

    W = load_matrix(input_path)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    A = (W > 0).astype(int)

    X0 = compute_X(W, A, CONFIG["alpha"], CONFIG["beta"])
    Z0 = compute_Z(A)
    Q0 = scale_signed(compute_modularity_matrix(W))

    X0 = minmax_01(X0)
    Z0 = minmax_01(Z0)

    X_t, Q_t, Z_t = X0, Q0, Z0
    results = []

    for T in range(1, CONFIG["T_max"] + 1):
        d_h = dims[T - 1]
        print(f"T={T}, d_h={d_h}")

        X_t, Q_t, Z_t = train_one_layer_three_way(
            X_t, Q_t, Z_t,
            d_h=d_h,
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            lr=CONFIG["lr"],
            rho=CONFIG["rho"],
            lam_sparse=CONFIG["lam_sparse"],
            w_x=CONFIG["w_x"],
            w_q=CONFIG["w_q"],
            w_z=CONFIG["w_z"],
        )

        H = X_t.T  # (n, d_h)
        sat, var = repr_stats(H, sat_eps=0.01)

        best_Q, best_k = -1, None
        modularity_by_k = {}

        for k in range(CONFIG["k_min"], CONFIG["k_max"] + 1):
            labels = KMeans(n_clusters=k, n_init=CONFIG["k_n_init"], random_state=0).fit_predict(H)
            Qv = modularity_score(W, labels)
            modularity_by_k[k] = Qv
            if Qv > best_Q:
                best_Q, best_k = Qv, k

        row = {
            "dataset": dataset_name,
            "T": T,
            "d_h": d_h,
            "best_k": best_k,
            "modularity": best_Q,
            "H_saturation_rate": sat,
            "H_var": var,
            "alpha": CONFIG["alpha"],
            "beta": CONFIG["beta"],
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "lr": CONFIG["lr"],
            "rho": CONFIG["rho"],
            "lam_sparse": CONFIG["lam_sparse"],
            "w_x": CONFIG["w_x"],
            "w_q": CONFIG["w_q"],
            "w_z": CONFIG["w_z"],
        }

        # add modularity for each k (wide format, easy to plot/inspect)
        for k, qv in modularity_by_k.items():
            row[f"Q_k{k}"] = qv

        results.append(row)

    df = pd.DataFrame(results)
    excel_path = out_dir / "compression_schedule_experiment.xlsx"
    df.to_excel(excel_path, index=False)

    t1 = time.perf_counter()
    with open(out_dir / "run_info.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset_name}\n")
        f.write(f"n={W.shape[0]}\n")
        f.write(f"edges={(W > 0).sum() // 2}\n")
        f.write(f"dims={dims}\n")
        f.write(f"time_seconds={t1 - t0:.6f}\n")

    print(f"Finished. Results saved to {excel_path}")


if __name__ == "__main__":
    main()
