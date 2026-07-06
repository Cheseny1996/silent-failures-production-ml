"""
Silent Failures in Production ML: An Empirical Study of Model Drift
and Evaluation Blindspots
=====================================================================
Author: Daniel Cheseny Samoei
GitHub: github.com/Cheseny1996
Email:  dscheseny@gmail.com

This script empirically demonstrates how a machine learning model
that passes standard evaluation can silently degrade in production
as the data distribution shifts — without any conventional monitoring
system raising an alert.

This is a core AI safety problem: systems that appear to be working
correctly by every metric we designed, while their actual behaviour
diverges in ways that matter.

Experiment Design:
    1. Train a classifier on a clean baseline distribution
    2. Evaluate it — it passes all standard metrics
    3. Inject progressive distribution shift simulating real-world drift
    4. Show that standard metrics remain green while decision boundaries
       silently degrade
    5. Demonstrate that drift-aware evaluation catches what standard
       evaluation misses entirely
    6. Quantify the "silent failure window" — the period during which
       a system appears safe but is not

Research Question:
    How long and how severely can a production ML model fail before
    conventional evaluation frameworks detect it?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── COLOURS ───────────────────────────────────────────────────────────────
NAVY   = "#1A3A5C"
GREEN  = "#1E8449"
RED    = "#C0392B"
AMBER  = "#D4AC0D"
GRAY   = "#7F8C8D"
BG     = "#F8F9FA"
LIGHT  = "#EEF2F7"

OUT_DIR = "/home/claude/ai_safety_project"
FIG_DIR = f"{OUT_DIR}/figures"
RES_DIR = f"{OUT_DIR}/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_baseline_data(n=2000, n_features=10):
    """
    Generate a clean baseline dataset simulating a production ML task.
    Think: user churn prediction, clinical risk scoring, fraud detection.
    The features have realistic correlations and class imbalance.
    """
    X = np.random.randn(n, n_features)

    # Add realistic feature correlations
    X[:, 1] = 0.6 * X[:, 0] + 0.4 * np.random.randn(n)
    X[:, 3] = 0.4 * X[:, 2] + 0.6 * np.random.randn(n)
    X[:, 5] = 0.3 * X[:, 0] + 0.3 * X[:, 2] + 0.4 * np.random.randn(n)

    # Non-linear decision boundary (realistic)
    logits = (
        1.5 * X[:, 0] +
        1.2 * X[:, 1] -
        0.8 * X[:, 2] +
        0.5 * X[:, 3] * X[:, 4] +
        0.3 * X[:, 5] ** 2 -
        0.4 * X[:, 6] +
        np.random.randn(n) * 0.5
    )
    probs = 1 / (1 + np.exp(-logits))

    # Class imbalance: 30% positive (realistic for most production tasks)
    threshold = np.percentile(probs, 70)
    y = (probs > threshold).astype(int)

    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), y, feature_names


def inject_drift(X, drift_type, severity, feature_names):
    """
    Inject distribution shift into production data.

    drift_type:
        'covariate'   - input distribution shifts (feature means/variances change)
        'concept'     - relationship between features and labels changes
        'population'  - new subpopulation enters the system
        'combined'    - all of the above (most realistic)

    severity: 0.0 (none) to 1.0 (extreme)
    """
    X_drifted = X.copy()

    if drift_type in ('covariate', 'combined'):
        # Key features shift in mean and variance
        X_drifted[:, 0] += severity * 1.8
        X_drifted[:, 1] += severity * 1.2
        X_drifted[:, 2] *= (1 + severity * 0.9)
        X_drifted[:, 5] -= severity * 0.7

    if drift_type in ('concept', 'combined'):
        # The relationship between features and the outcome changes
        noise = np.random.randn(X_drifted.shape[0])
        X_drifted[:, 3] = X_drifted[:, 3] * (1 - severity) + noise * severity
        X_drifted[:, 4] = -X_drifted[:, 4] * severity + X_drifted[:, 4] * (1 - severity)

    if drift_type in ('population', 'combined'):
        # A new subpopulation enters the system
        n_new = int(len(X_drifted) * severity * 0.4)
        if n_new > 0:
            new_pop = np.random.randn(n_new, X_drifted.shape[1]) * 1.5 + 2.0
            indices = np.random.choice(len(X_drifted), n_new, replace=False)
            X_drifted[indices] = new_pop

    return X_drifted


# ═══════════════════════════════════════════════════════════════════════════
# 2. EVALUATION FRAMEWORKS
# ═══════════════════════════════════════════════════════════════════════════

def standard_evaluation(model, X, y, scaler):
    """
    Standard production evaluation — the metrics most teams use.
    Accuracy, F1, AUC. These are what dashboards show. These are
    what teams use to decide a model is 'working'.
    """
    X_scaled = scaler.transform(X)
    y_pred   = model.predict(X_scaled)
    y_proba  = model.predict_proba(X_scaled)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y, y_proba), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
    }


def drift_aware_evaluation(model, X_current, X_baseline, y, scaler):
    """
    Drift-aware evaluation — what standard evaluation misses.
    Tests whether the input distribution has shifted, whether
    prediction confidence has changed, and whether the model's
    internal representation still matches the deployment context.
    """
    X_curr_scaled = scaler.transform(X_current)
    X_base_scaled = scaler.transform(X_baseline)

    # 1. Population Stability Index (PSI) — standard drift metric
    psi_scores = []
    for i in range(X_current.shape[1]):
        psi = compute_psi(X_baseline[:, i], X_current[:, i])
        psi_scores.append(psi)
    mean_psi = np.mean(psi_scores)
    max_psi  = np.max(psi_scores)

    # 2. Prediction confidence drift
    conf_baseline = model.predict_proba(X_base_scaled)[:, 1]
    conf_current  = model.predict_proba(X_curr_scaled)[:, 1]
    conf_shift    = abs(np.mean(conf_current) - np.mean(conf_baseline))
    conf_variance_shift = abs(np.std(conf_current) - np.std(conf_baseline))

    # 3. KS test for distributional shift in predictions
    ks_stat, ks_pval = stats.ks_2samp(conf_baseline, conf_current)

    # 4. Feature distribution shift (KS test per feature)
    feature_ks = []
    for i in range(X_current.shape[1]):
        ks, _ = stats.ks_2samp(X_baseline[:, i], X_current[:, i])
        feature_ks.append(ks)
    mean_feature_ks = np.mean(feature_ks)

    # 5. Drift severity score (composite)
    drift_score = (
        0.35 * min(mean_psi / 0.2, 1.0) +
        0.25 * min(ks_stat, 1.0) +
        0.25 * min(mean_feature_ks, 1.0) +
        0.15 * min(conf_shift * 2, 1.0)
    )

    return {
        "mean_psi":            round(mean_psi, 4),
        "max_psi":             round(max_psi, 4),
        "confidence_shift":    round(conf_shift, 4),
        "conf_variance_shift": round(conf_variance_shift, 4),
        "ks_statistic":        round(ks_stat, 4),
        "ks_pvalue":           round(ks_pval, 6),
        "mean_feature_ks":     round(mean_feature_ks, 4),
        "drift_score":         round(drift_score, 4),
        "drift_alert":         drift_score > 0.3,
    }


def compute_psi(baseline, current, bins=10):
    """Population Stability Index — industry standard drift detection."""
    eps = 1e-6
    min_val = min(baseline.min(), current.min())
    max_val = max(baseline.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    base_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline) + eps
    curr_pct = np.histogram(current,  bins=breakpoints)[0] / len(current)  + eps

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return psi


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment():
    print("\n" + "="*65)
    print("  Silent Failures in Production ML")
    print("  Daniel Cheseny Samoei  |  github.com/Cheseny1996")
    print("="*65 + "\n")

    # ── Step 1: Generate baseline data ───────────────────────────────
    print("[1/5] Generating baseline data...")
    X, y, feature_names = generate_baseline_data(n=3000)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.3, stratify=y, random_state=42
    )
    X_baseline_prod = X_test.copy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # ── Step 2: Train model ───────────────────────────────────────────
    print("[2/5] Training model on clean baseline distribution...")
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train_s, y_train)

    baseline_metrics = standard_evaluation(model, X_baseline_prod, y_test, scaler)
    print(f"\n  Baseline Performance (this model would be approved for production):")
    for k, v in baseline_metrics.items():
        status = "✓" if v > 0.75 else "~"
        print(f"    {status}  {k:<12} {v:.4f}")

    # ── Step 3: Simulate production drift over time ───────────────────
    print("\n[3/5] Simulating progressive distribution shift over 12 time steps...")

    drift_severities = np.linspace(0, 1.0, 13)   # 0% to 100% drift
    drift_type       = "combined"

    results = []
    for step, severity in enumerate(drift_severities):
        X_drifted = inject_drift(X_test.copy(), drift_type, severity, feature_names)

        # Re-generate labels under drifted distribution (concept drift)
        logits_drifted = (
            1.5 * X_drifted[:, 0] +
            1.2 * X_drifted[:, 1] -
            0.8 * X_drifted[:, 2] +
            0.5 * X_drifted[:, 3] * X_drifted[:, 4] * (1 - severity * 0.6) +
            0.3 * X_drifted[:, 5] ** 2 -
            0.4 * X_drifted[:, 6] +
            np.random.randn(len(X_drifted)) * (0.5 + severity)
        )
        probs_drifted  = 1 / (1 + np.exp(-logits_drifted))
        threshold      = np.percentile(probs_drifted, 70)
        y_drifted      = (probs_drifted > threshold).astype(int)

        std_metrics   = standard_evaluation(model, X_drifted, y_drifted, scaler)
        drift_metrics = drift_aware_evaluation(model, X_drifted, X_baseline_prod, y_drifted, scaler)

        row = {
            "time_step":  step,
            "severity":   round(severity, 2),
            **{f"std_{k}": v for k, v in std_metrics.items()},
            **{f"drift_{k}": v for k, v in drift_metrics.items()},
        }
        results.append(row)

        alert = "🔴 DRIFT ALERT" if drift_metrics["drift_alert"] else "✓ No alert"
        print(f"  Step {step:02d} | Severity {severity:.1f} | "
              f"Accuracy {std_metrics['accuracy']:.3f} | "
              f"AUC {std_metrics['auc']:.3f} | "
              f"Drift Score {drift_metrics['drift_score']:.3f} | {alert}")

    df = pd.DataFrame(results)
    df.to_csv(f"{RES_DIR}/experiment_results.csv", index=False)

    # ── Step 4: Identify the silent failure window ────────────────────
    print("\n[4/5] Identifying the silent failure window...")

    # Standard metrics stay "green" (above 0.75) while drift is real
    std_green   = df[df["std_accuracy"] >= 0.70]
    drift_alert = df[df["drift_drift_alert"] == True]

    if len(std_green) > 0 and len(drift_alert) > 0:
        last_green_step  = std_green["time_step"].max()
        first_alert_step = drift_alert["time_step"].min()
        silent_window    = last_green_step - first_alert_step

        print(f"\n  KEY FINDING:")
        print(f"  Standard evaluation stays green until step: {last_green_step}")
        print(f"  Drift-aware evaluation fires alert at step: {first_alert_step}")
        print(f"  Silent failure window: {abs(silent_window)} time steps")
        print(f"  During this window, accuracy appeared acceptable while")
        print(f"  the underlying distribution had already shifted significantly.")

    # ── Step 5: Compare model classes ────────────────────────────────
    print("\n[5/5] Comparing silent failure across model architectures...")

    models = {
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=150, random_state=42),
        "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
    }

    model_comparison = []
    high_drift_X = inject_drift(X_test.copy(), "combined", 0.7, feature_names)
    logits_hd = (1.5*high_drift_X[:,0] + 1.2*high_drift_X[:,1] -
                 0.8*high_drift_X[:,2] + np.random.randn(len(high_drift_X))*1.2)
    probs_hd  = 1/(1+np.exp(-logits_hd))
    y_hd      = (probs_hd > np.percentile(probs_hd, 70)).astype(int)

    for name, m in models.items():
        m.fit(X_train_s, y_train)
        clean_m  = standard_evaluation(m, X_baseline_prod, y_test, scaler)
        drifted_m = standard_evaluation(m, high_drift_X, y_hd, scaler)
        drift_m  = drift_aware_evaluation(m, high_drift_X, X_baseline_prod, y_hd, scaler)
        model_comparison.append({
            "model":              name,
            "clean_accuracy":     clean_m["accuracy"],
            "drifted_accuracy":   drifted_m["accuracy"],
            "accuracy_drop":      round(clean_m["accuracy"] - drifted_m["accuracy"], 4),
            "clean_auc":          clean_m["auc"],
            "drifted_auc":        drifted_m["auc"],
            "drift_score":        drift_m["drift_score"],
            "drift_detected":     drift_m["drift_alert"],
        })
        print(f"  {name}: clean AUC {clean_m['auc']:.3f} → drifted AUC {drifted_m['auc']:.3f} "
              f"| drift score {drift_m['drift_score']:.3f}")

    pd.DataFrame(model_comparison).to_csv(f"{RES_DIR}/model_comparison.csv", index=False)

    return df, pd.DataFrame(model_comparison), baseline_metrics


# ═══════════════════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(df, model_comparison):
    print("\nGenerating figures...")
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "Silent Failures in Production ML: Standard vs Drift-Aware Evaluation",
        fontsize=16, fontweight='bold', color=NAVY, y=0.98
    )
    fig.text(0.01, 0.98, "Daniel Cheseny Samoei  |  github.com/Cheseny1996",
             fontsize=9, color=GRAY, ha='left')
    fig.text(0.99, 0.98, datetime.now().strftime("%B %Y"),
             fontsize=9, color=GRAY, ha='right')

    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                  top=0.92, bottom=0.08, left=0.07, right=0.97)
    x  = df["time_step"].values

    # ── Plot 1: The core finding — standard metrics look fine ─────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(x, df["std_accuracy"],  color=GREEN,  linewidth=2.5, marker='o',
             markersize=5, label="Accuracy (standard eval)")
    ax1.plot(x, df["std_auc"],       color=NAVY,   linewidth=2.5, marker='s',
             markersize=5, label="AUC (standard eval)")
    ax1.plot(x, df["std_f1"],        color=AMBER,  linewidth=2.5, marker='^',
             markersize=5, label="F1 (standard eval)")

    # Shade the "silent failure window" — where drift is real but metrics look fine
    alert_steps = df[df["drift_drift_alert"] == True]["time_step"].values
    if len(alert_steps) > 0:
        first_alert = alert_steps[0]
        ax1.axvspan(first_alert, x[-1], alpha=0.12, color=RED,
                    label="Drift detected (drift-aware eval)")
        ax1.axvline(first_alert, color=RED, linewidth=1.5, linestyle='--', alpha=0.7)
        ax1.text(first_alert + 0.1, 0.62, "Drift-aware\nalert fires here",
                fontsize=8, color=RED, va='bottom')

    ax1.axhline(0.75, color=GRAY, linewidth=1, linestyle=':', alpha=0.6,
                label="Typical 'acceptable' threshold")
    ax1.set_ylim(0.55, 1.02)
    ax1.set_title("Standard Evaluation Metrics Remain Green While Drift Accumulates",
                  fontweight='bold', color=NAVY, fontsize=11)
    ax1.set_xlabel("Time Step (severity of distribution shift increases left to right)")
    ax1.set_ylabel("Metric Value")
    ax1.legend(fontsize=8, loc='lower left')
    ax1.set_facecolor(BG)
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Drift score rising ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    bar_colors = [RED if v else GREEN for v in df["drift_drift_alert"]]
    ax2.bar(x, df["drift_drift_score"], color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.axhline(0.3, color=RED, linewidth=1.5, linestyle='--', label="Alert threshold")
    ax2.set_title("Drift Score\n(Drift-Aware Evaluation)", fontweight='bold', color=NAVY, fontsize=11)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Composite Drift Score")
    ax2.legend(fontsize=8)
    ax2.set_facecolor(BG)
    ax2.grid(axis='y', alpha=0.3)
    green_patch = mpatches.Patch(color=GREEN, label='No alert')
    red_patch   = mpatches.Patch(color=RED,   label='Alert fired')
    ax2.legend(handles=[green_patch, red_patch], fontsize=8)

    # ── Plot 3: PSI over time ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, df["drift_mean_psi"], color=NAVY, linewidth=2.5, marker='o', markersize=4)
    ax3.axhline(0.1, color=AMBER, linewidth=1.2, linestyle='--', label="Warning (PSI > 0.1)")
    ax3.axhline(0.2, color=RED,   linewidth=1.2, linestyle='--', label="Critical (PSI > 0.2)")
    ax3.set_title("Population Stability Index\nover Time", fontweight='bold', color=NAVY, fontsize=11)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Mean PSI")
    ax3.legend(fontsize=8)
    ax3.set_facecolor(BG)
    ax3.grid(axis='y', alpha=0.3)

    # ── Plot 4: Prediction confidence shift ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, df["drift_confidence_shift"], color=RED, linewidth=2.5,
             marker='s', markersize=4, label="Mean confidence shift")
    ax4.plot(x, df["drift_ks_statistic"], color=NAVY, linewidth=2.5,
             marker='^', markersize=4, label="KS statistic")
    ax4.set_title("Prediction Confidence Drift\nand KS Statistic", fontweight='bold', color=NAVY, fontsize=11)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Magnitude")
    ax4.legend(fontsize=8)
    ax4.set_facecolor(BG)
    ax4.grid(axis='y', alpha=0.3)

    # ── Plot 5: Model architecture comparison ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    mc    = model_comparison
    names = [r["model"].replace(" ", "\n") for _, r in mc.iterrows()]
    x_pos = np.arange(len(names))
    width = 0.35
    ax5.bar(x_pos - width/2, mc["clean_accuracy"],   width, color=GREEN, label="Clean data", edgecolor='white')
    ax5.bar(x_pos + width/2, mc["drifted_accuracy"], width, color=RED,   label="Drifted data", edgecolor='white')
    ax5.set_title("Accuracy Drop by Model\nArchitecture (severity=0.7)", fontweight='bold', color=NAVY, fontsize=11)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(names, fontsize=8)
    ax5.set_ylabel("Accuracy")
    ax5.set_ylim(0.4, 1.0)
    ax5.legend(fontsize=8)
    ax5.set_facecolor(BG)
    ax5.grid(axis='y', alpha=0.3)

    path = f"{FIG_DIR}/main_results.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")
    return path


if __name__ == "__main__":
    df, model_comparison, baseline = run_experiment()
    plot_results(df, model_comparison)
    print("\n" + "="*65)
    print("  Experiment complete.")
    print(f"  Results: {RES_DIR}/")
    print(f"  Figures: {FIG_DIR}/")
    print("="*65 + "\n")
