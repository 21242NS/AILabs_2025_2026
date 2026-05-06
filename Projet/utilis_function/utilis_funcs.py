
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor

graph_dir = Path("../Graph")
graph_dir.mkdir(parents=True, exist_ok=True)

def _ensure_save_dir(save_dir):
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_preds_vs_true(y_true, preds, name="model", save_dir="../Graph", annotate_topk=5, figsize=(8,8)):
    """Scatter réel vs prédit, ligne y=x, coloré par erreur absolue, top-k outliers annotés."""
    y = np.asarray(y_true).ravel()
    p = np.asarray(preds).ravel()
    assert y.shape[0] == p.shape[0], "y_true et preds doivent avoir même longueur"
    res = np.abs(y - p)
    mae = mean_absolute_error(y, p)

    plt.figure(figsize=figsize)
    sc = plt.scatter(y, p, c=res, cmap="viridis_r", alpha=0.75, edgecolor="k", linewidth=0.2)
    mn, mx = min(y.min(), p.min()), max(y.max(), p.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.axhline(60, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    cb = plt.colorbar(sc)
    cb.set_label("|résidu|")
    plt.xlabel("Valeur réelle")
    plt.ylabel("Prédiction")
    plt.title(f"{name} — Réel vs Prédit  (MAE={mae:.3f})")
    # annoter outliers
    k = min(annotate_topk, len(res))
    idx_top = np.argsort(-res)[:k]
    for i in idx_top:
        plt.annotate(f"{i}: {res[i]:.2f}", (y[i], p[i]), xytext=(6,6), textcoords="offset points", fontsize=8, color="red")
    plt.grid(alpha=0.25)
    Path(_ensure_save_dir(save_dir) / f"test_preds_{name}.png").with_suffix(".png")
    out = Path(save_dir) / f"test_preds_{name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.show()
    return {"mae": float(mae), "top_outliers_idx": idx_top, "residuals": res}

def plot_error_distribution(y_true, preds, name="model", save_dir="../Graph", bins=50, figsize=(8,4)):
    """Histogramme + KDE des résidus (préd - réel) et statistiques."""
    y = np.asarray(y_true).ravel()
    p = np.asarray(preds).ravel()
    res = p - y
    mae = mean_absolute_error(y, p)
    mean = res.mean()
    std = res.std()

    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.histplot(res, bins=bins, kde=True, ax=ax, color="steelblue")
    ax.axvline(mean, color="black", linestyle="--", label=f"mean={mean:.3f}")
    ax.axvline(0, color="red", linestyle=":", label="0")
    ax.set_xlabel("Résidu (prédit - réel)")
    ax.set_title(f"{name} — Distribution des erreurs (MAE={mae:.3f}, std={std:.3f})")
    ax.legend()
    plt.tight_layout()
    out = Path(_ensure_save_dir(save_dir) / f"error_dist_{name}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    return {"mae": float(mae), "mean": float(mean), "std": float(std)}

def plot_error_by_true_band(y_true, preds, name="model", save_dir="../Graph", nb_bands=5, figsize=(8,4)):
    """MAE par tranche de valeur réelle (utile pour détecter zones problématiques)."""
    y = pd.Series(y_true).reset_index(drop=True)
    p = pd.Series(preds).reset_index(drop=True)
    df = pd.DataFrame({"y": y, "pred": p})
    df["band"] = pd.qcut(df["y"], q=nb_bands, duplicates="drop")
    mae_by = df.groupby("band").apply(lambda d: mean_absolute_error(d["y"], d["pred"]))
    counts = df.groupby("band").size()
    fig, ax = plt.subplots(1,1,figsize=figsize)
    mae_by.plot(kind="bar", ax=ax, color="orange")
    ax.set_ylabel("MAE")
    ax.set_title(f"{name} — MAE par tranche réelle (n_bands={nb_bands})")
    for i, v in enumerate(mae_by.values):
        ax.text(i, v + 0.01 * v if v!=0 else 0.01, f"{v:.3f}\n(n={counts.iloc[i]})", ha="center", fontsize=8)
    plt.tight_layout()
    out = Path(_ensure_save_dir(save_dir) / f"mae_by_band_{name}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    return pd.DataFrame({"mae": mae_by, "n": counts})

def plot_residuals_vs_feature(feature, y_true, preds, feature_name="feature", name="model", save_dir="../Graph", figsize=(8,4)):
    """Scatter feature vs residual (prédit - réel) + linéaire/regression faible pour détecter patterns."""
    feat = np.asarray(feature).ravel()
    y = np.asarray(y_true).ravel()
    p = np.asarray(preds).ravel()
    res = p - y
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.scatterplot(x=feat, y=res, alpha=0.6, ax=ax, edgecolor=None)
    sns.regplot(x=feat, y=res, scatter=False, lowess=True, ax=ax, color="red")
    ax.axhline(0, color="k", linestyle="--", alpha=0.6)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Résidu (prédit - réel)")
    ax.set_title(f"{name} — Résidus vs {feature_name}")
    plt.tight_layout()
    out = Path(_ensure_save_dir(save_dir) / f"residuals_vs_{feature_name}_{name}.png")
    plt.savefig(out, dpi=150)
    plt.show()
    return {"pearson_corr": float(np.corrcoef(feat, res)[0,1]) if len(feat)>1 else np.nan}

def plot_mae_overlay(y_true, y_pred, model_name, filename, top_n=5):
    plot_df = pd.DataFrame({
        "reel": np.asarray(y_true),
        "pred": np.asarray(y_pred),
    }).reset_index(names="sample_id")
    plot_df["erreur_abs"] = (plot_df["reel"] - plot_df["pred"]).abs()
    plot_df = plot_df.sort_values("reel").reset_index(drop=True)
    mae = plot_df["erreur_abs"].mean()
    x = np.arange(len(plot_df))

    plt.figure(figsize=(14, 6))
    plt.plot(x, plot_df["reel"], label="Réalité", linewidth=2, color="#1f77b4")
    plt.plot(x, plot_df["pred"], label="Prédiction", linewidth=1.8, color="#ff7f0e", alpha=0.9)
    plt.fill_between(
        x,
        plot_df["reel"],
        plot_df["pred"],
        color="#ff7f0e",
        alpha=0.15,
        label=f"Erreur absolue (MAE = {mae:.2f})",
    )

    top_errors = plot_df.nlargest(top_n, "erreur_abs")
    plt.vlines(
        top_errors.index,
        top_errors["reel"],
        top_errors["pred"],
        color="crimson",
        alpha=0.6,
        linewidth=1.2,
    )
    plt.scatter(top_errors.index, top_errors["pred"], color="crimson", s=35, zorder=3, label=f"Top {top_n} erreurs")

    for idx, row in top_errors.iterrows():
        plt.annotate(
            f"{row['erreur_abs']:.1f}",
            (idx, row["pred"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="crimson",
        )

    plt.xlabel("Échantillons triés par score réel")
    plt.ylabel("Score examen")
    plt.title(f"{model_name} - réalité vs prédiction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()

    display(top_errors[["sample_id", "reel", "pred", "erreur_abs"]].round(2))

def mae_par_tranches(y_true, y_pred, tranches=None):
    if tranches is None:
        tranches = [
            (0, 25, "0-25"),
            (25, 90, "25-90"),
            (90, 100.0001, "90-100"),
        ]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []

    for borne_min, borne_max, label in tranches:
        mask = (y_true >= borne_min) & (y_true < borne_max)
        n = int(mask.sum())
        mae = mean_absolute_error(y_true[mask], y_pred[mask]) if n > 0 else np.nan
        rows.append({"tranche": label, "n": n, "mae": mae})

    return pd.DataFrame(rows)

def plot_numeric_score_density(x, y, feature_label, filename, gridsize=40, quantiles=8):
    plot_df = pd.DataFrame({"feature": x, "score": y}).dropna()
    if plot_df.empty:
        print(f"Aucune donnée exploitable pour {feature_label}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    hb = axes[0].hexbin(
        plot_df["feature"],
        plot_df["score"],
        gridsize=gridsize,
        cmap="viridis",
        bins="log",
        mincnt=1,
    )
    fig.colorbar(hb, ax=axes[0], label="Densité d'observations")
    axes[0].set_title(f"{feature_label} vs score - densité")
    axes[0].set_xlabel(feature_label)
    axes[0].set_ylabel("Score examen")
    axes[0].grid(alpha=0.2)

    n_bins = min(quantiles, plot_df["feature"].nunique())
    if n_bins >= 2:
        plot_df["feature_bin"] = pd.qcut(plot_df["feature"], q=n_bins, duplicates="drop")
        sns.boxplot(data=plot_df, x="feature_bin", y="score", color="#9ecae1", showfliers=False, ax=axes[1])
        axes[1].set_title(f"Score par tranche de {feature_label}")
        axes[1].set_xlabel(feature_label)
        axes[1].set_ylabel("Score examen")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(alpha=0.2)
    else:
        sns.histplot(plot_df["score"], bins=30, kde=True, color="#9ecae1", ax=axes[1])
        axes[1].set_title(f"Distribution du score pour {feature_label}")
        axes[1].set_xlabel("Score examen")

    plt.tight_layout()
    plt.savefig(graph_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()

def plot_categorical_score_density(x, y, feature_label, filename, order=None):
    plot_df = pd.DataFrame({"feature": x, "score": y}).dropna()
    if plot_df.empty:
        print(f"Aucune donnée exploitable pour {feature_label}")
        return

    if order is None:
        order = (
            plot_df.groupby("feature", observed=False)["score"]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

    plot_df["score_band"] = pd.cut(
        plot_df["score"],
        bins=[-0.001, 25, 50, 75, 100.0001],
        labels=["0-25", "25-50", "50-75", "75-100"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    sns.violinplot(
        data=plot_df,
        x="feature",
        y="score",
        order=order,
        inner="quartile",
        cut=0,
        color="#9ecae1",
        ax=axes[0],
    )
    sampled_df = plot_df.sample(min(len(plot_df), 4000), random_state=0)
    sns.stripplot(
        data=sampled_df,
        x="feature",
        y="score",
        order=order,
        color="black",
        alpha=0.12,
        size=2,
        jitter=0.22,
        ax=axes[0],
    )
    axes[0].set_title(f"{feature_label} vs score - distribution")
    axes[0].set_xlabel(feature_label)
    axes[0].set_ylabel("Score examen")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.2)

    heatmap_df = pd.crosstab(plot_df["feature"], plot_df["score_band"], normalize="index") * 100
    heatmap_df = heatmap_df.reindex(order)
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlOrBr", cbar_kws={"label": "% dans la catégorie"}, ax=axes[1])
    axes[1].set_title(f"Densité des réponses par tranche de score")
    axes[1].set_xlabel("Tranche de score")
    axes[1].set_ylabel(feature_label)

    plt.tight_layout()
    plt.savefig(graph_dir / filename, dpi=150, bbox_inches="tight")
    plt.show()
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=0
    )
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)
def report_split_metrics(model_name, split_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mae_bands_df = mae_par_tranches(y_true, y_pred)
    print(f"MAE {split_name} du modèle {model_name} : {mae:.4f}")
    print(mae_bands_df)
    return mae, mae_bands_df
def classification_report(y_true, y_pred, threshold_real=50, threshold_pred=50):
    y_true_class = (y_true >= threshold_real).astype(int)
    y_pred_class = (y_pred >= threshold_pred).astype(int)

    tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
    tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
    fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
    fn = np.sum((y_true_class == 1) & (y_pred_class == 0))

    total = len(y_true)
    print(f"Total : {total}")
    print(f"Vrais positifs (réussite prédite et réelle) : {tp} ({tp/total:.2%})")
    print(f"Vrais négatifs (échec prédit et réel) : {tn} ({tn/total:.2%})")
    print(f"Faux positifs (réussite prédite mais échec réel) : {fp} ({fp/total:.2%})")
    print(f"Faux négatifs (échec prédit mais réussite réelle) : {fn} ({fn/total:.2%})")
    print(f"Précision (réussite prédite parmi les réussites réelles) : {tp/(tp+fp):.2%}" if (tp+fp) > 0 else "Précision : N/A")
    print(f"Rappel (echec prédite parmi les echecs réelles) : {tn/(tn+fn):.2%}" if (tn+fn) > 0 else "Rappel : N/A")
    print(f"precision global (prédictions correctes parmi toutes les prédictions) : {(tp+tn)/total:.2%}")
def score_band_distribution(y_true, y_pred, threshold=60):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bins = [-0.001, 25, 40, 45, 50, 55, 60, 75, 100.0001]
    labels = ["0-25", "25-40", "40-45", "45-50", "50-55", "55-60", "60-75", "75-100"]

    df = pd.DataFrame({
        "score_reel": y_true,
        "score_pred": y_pred,
    })

    df["classe_reelle"] = np.where(df["score_reel"] >= threshold, "réussite", "échec")
    df["classe_predite"] = np.where(df["score_pred"] >= threshold, "réussite", "échec")

    df["type_prediction"] = np.select(
        [
            (df["classe_reelle"] == "réussite") & (df["classe_predite"] == "réussite"),
            (df["classe_reelle"] == "échec") & (df["classe_predite"] == "échec"),
            (df["classe_reelle"] == "échec") & (df["classe_predite"] == "réussite"),
            (df["classe_reelle"] == "réussite") & (df["classe_predite"] == "échec"),
        ],
        ["TP", "TN", "FP", "FN"],
        default="Autre",
    )

    df["tranche_score_reel"] = pd.cut(
        df["score_reel"],
        bins=bins,
        labels=labels,
        right=False
    )

    summary = (
        df.groupby(["tranche_score_reel", "type_prediction"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["TP", "TN", "FP", "FN"], fill_value=0)
    )

    summary["total"] = summary.sum(axis=1)
    summary["erreurs"] = summary["FP"] + summary["FN"]

    summary["pct_erreur_tranche"] = np.where(
        summary["total"] > 0,
        100 * summary["erreurs"] / summary["total"],
        0.0
    )

    for col in ["TP", "TN", "FP", "FN"]:
        summary[f"pct_{col.lower()}_tranche"] = np.where(
            summary["total"] > 0,
            100 * summary[col] / summary["total"],
            0.0
        )

    display_cols = [
        "TP", "TN", "FP", "FN",
        "total", "erreurs",
        "pct_tp_tranche", "pct_tn_tranche",
        "pct_fp_tranche", "pct_fn_tranche",
        "pct_erreur_tranche",
    ]

    display(summary[display_cols].round(2))

    error_focus = summary[summary["total"] > 0].sort_values(
        ["pct_erreur_tranche", "erreurs"],
        ascending=False
    )

    if not error_focus.empty:
        worst_band = error_focus.index[0]
        worst_row = error_focus.iloc[0]

        print(
            f"Tranche avec le plus d'erreurs de classification : {worst_band} | "
            f"taux d'erreur = {worst_row['pct_erreur_tranche']:.2f}% | "
            f"FP = {int(worst_row['FP'])} | "
            f"FN = {int(worst_row['FN'])}"
        )

    # Pourcentages à afficher
    plot_df = pd.DataFrame({
        "TP": summary["pct_tp_tranche"],
        "TN": summary["pct_tn_tranche"],
        "FP": summary["pct_fp_tranche"],
        "FN": summary["pct_fn_tranche"],
    })

    # Couleurs cohérentes avec la réalité
    colors = {
        "TP": "#2ca02c",  # vert : réussite réelle prédite réussite
        "TN": "#1f77b4",  # bleu : échec réel prédit échec
        "FP": "#ff0e0e",  # orange : échec réel prédit réussite
        "FN": "#d68127",  # rouge : réussite réelle prédite échec
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(plot_df.index))
    bottom = np.zeros(len(plot_df.index))

    for col in ["TP", "TN", "FP", "FN"]:
        ax.bar(
            x,
            plot_df[col].values,
            bottom=bottom,
            color=colors[col],
            label=col
        )
        bottom += plot_df[col].values

    ax.axhline(
        100,
        color="#444",
        linewidth=0.8,
        label="_nolegend_"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=0)

    ax.set_title("Répartition des TP / TN / FP / FN par tranche de score réel")
    ax.set_xlabel("Tranche de score réel")
    ax.set_ylabel("Pourcentage dans la tranche (%)")
    ax.grid(axis="y", alpha=0.25)

    # Légende construite manuellement, donc impossible qu'elle soit décalée
    legend_handles = [
        Patch(facecolor=colors["TP"], label="TP"),
        Patch(facecolor=colors["TN"], label="TN"),
        Patch(facecolor=colors["FP"], label="FP"),
        Patch(facecolor=colors["FN"], label="FN"),
    ]

    ax.legend(handles=legend_handles, title="Type")

    plt.tight_layout()
    plt.show()

    return summary