import os
import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_residuals(
    case_dir,
    log_name,
    eps_phys=1e-4,
    out_png="residual_rms_analysis.png",
):
    """
    Parse training log, extract residual_rms_batch_mean vs epoch,
    compute first-epoch crossing of eps_phys (k*), and plot.
    """

    log_path = os.path.join(case_dir, log_name)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    epochs = []
    residuals = []

    # ------------------------------------------------------------------
    # 1) Adapt the regex to your actual log format
    #
    # Example assumed line:
    #   "Epoch 17 ... residual_rms_batch_mean=3.21e-04 ..."
    #
    # If your format differs, just tweak 'pattern'.
    # ------------------------------------------------------------------
    pattern = re.compile(
        r"Epoch\s+(\d+).*residual_rms_batch_mean\s*=?\s*([0-9.eE+-]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group(1))
                res   = float(m.group(2))
                epochs.append(epoch)
                residuals.append(res)

    if not epochs:
        raise RuntimeError("No residual entries found in log. Check regex / log format.")

    epochs = np.array(epochs)
    residuals = np.array(residuals)

    # ------------------------------------------------------------------
    # 2) Find first epoch that satisfies residual <= eps_phys
    # ------------------------------------------------------------------
    idx = np.where(residuals <= eps_phys)[0]
    if len(idx) > 0:
        k_star = int(epochs[idx[0]])
        res_at_k = residuals[idx[0]]
        print(f"[INFO] First epoch with residual <= {eps_phys:g}: k* = {k_star}, residual = {res_at_k:g}")
    else:
        k_star = None
        print(f"[INFO] Residual never reached tolerance {eps_phys:g} in this run.")
    
    # ------------------------------------------------------------------
    # 3) Plot residual vs epoch
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.semilogy(epochs, residuals, marker="o", label="residual_rms_batch_mean")
    plt.axhline(eps_phys, linestyle="--", color="gray", label=f"tolerance = {eps_phys:g}")

    if k_star is not None:
        plt.axvline(k_star, linestyle="--", color="red", label=f"k* = {k_star}")
        plt.scatter([k_star], [res_at_k], color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Residual RMS")
    plt.title("Training residual vs epoch")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(case_dir, out_png)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved residual plot to: {out_path}")

    return {
        "epochs": epochs,
        "residuals": residuals,
        "k_star": k_star,
        "eps_phys": eps_phys,
    }

if __name__ == "__main__":
    # Example usage for your TL circular case:
    case_dir = "./1_Param_Transfer_reference_circular"
    log_name = "1_Param_Transfer_reference_circular.log"

    result = analyze_residuals(
        case_dir=case_dir,
        log_name=log_name,
        eps_phys=1e-4,          # choose your physics tolerance here
        out_png="residual_rms_analysis.png",
    )

    print("Summary:", result)
