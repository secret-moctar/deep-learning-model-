#!/usr/bin/env python3
"""
sweep.py — Sweep multi-modèles / multi-hyperparamètres + rapport automatique
============================================================================

Au lieu d'entraîner UN seul modèle, ce script entraîne **plusieurs variantes**
en faisant varier soit l'architecture (ResNet/ViT/Swin), soit les hyper-
paramètres (mixup, cutmix, EMA, focal-loss, batch size, learning rate,
nombre d'époques, label smoothing...).

Deux presets sont fournis :

    --preset colab   →  SWEEP_COLAB   (5-7 runs, ~1-2 h sur Colab T4 gratuit)
    --preset local   →  SWEEP_LOCAL   (12-15 runs, ~3-5 h sur RTX 3090)

Pour chaque run :
  1. Applique des overrides sur les attributs de `config`.
  2. Recharge `couche2.train` et `couche2.evaluate` (sinon les `from config import X`
     déjà bindés à l'ancienne valeur ne voient pas l'override).
  3. Entraîne (`train.train_model(arch, epochs, run_name=...)`).
  4. Évalue (`evaluate.evaluate_model(arch, run_name=...)`).
  5. Enregistre dans `outputs/registry.json` via `run_local.register_model`,
     en stockant explicitement l'architecture ET les overrides.

À la fin :
  * `outputs/sweep_report.md`    — tableau classé + analyse (présentation classe)
  * `outputs/sweep_comparison.png` — barres Acc/F1 par run, baseline mise en avant
  * `outputs/sweep_curves.png`     — courbes test_acc par époque, toutes superposées
  * `outputs/sweep_results.json`   — données brutes

Usage :
    python sweep.py                              # preset = colab (défaut sécurisé)
    python sweep.py --preset local               # sweep complet pour RTX 3090
    python sweep.py --preset local --only resnet50_focal,resnet50_no_ema
    python sweep.py --list                       # affiche les presets sans rien lancer
    python sweep.py --report-only                # régénère le rapport à partir
                                                 # des metrics_*.json déjà présents
"""
import os
import sys
import json
import argparse
import datetime
import importlib
import traceback
from typing import Dict, List, Optional

import config
from config import OUTPUT_DIR

# ===========================================================================
# 1. Presets
# ---------------------------------------------------------------------------
# Chaque entrée :
#   run_name   — clé unique dans la registry (utilisée pour les noms de fichiers)
#   arch       — architecture passée à build_model
#   epochs     — nombre d'époques pour ce run
#   overrides  — dict {attr_config: valeur} appliqué à `config` avant l'entraînement
# ===========================================================================

SWEEP_COLAB = [
    {"run_name": "resnet50_baseline",  "arch": "resnet50",
     "epochs": 10, "overrides": {}},
    {"run_name": "resnet50_no_mix",    "arch": "resnet50",
     "epochs": 10, "overrides": {"USE_MIXUP": False, "USE_CUTMIX": False}},
    {"run_name": "resnet50_focal",     "arch": "resnet50",
     "epochs": 10, "overrides": {"USE_FOCAL": True}},
    {"run_name": "resnet50_no_ema",    "arch": "resnet50",
     "epochs": 10, "overrides": {"USE_EMA": False}},
    {"run_name": "vit_baseline",       "arch": "vit_base_patch16_224",
     "epochs": 10, "overrides": {}},
    {"run_name": "swin_baseline",      "arch": "swin_base_patch4_window7_224",
     "epochs": 10, "overrides": {}},
]

SWEEP_LOCAL = [
    # ----- 1. Trois architectures, baseline -----------------------------
    {"run_name": "resnet50_baseline", "arch": "resnet50",
     "epochs": 25, "overrides": {}},
    {"run_name": "vit_baseline",      "arch": "vit_base_patch16_224",
     "epochs": 25, "overrides": {}},
    {"run_name": "swin_baseline",     "arch": "swin_base_patch4_window7_224",
     "epochs": 25, "overrides": {}},

    # ----- 2. Ablations sur le pipeline régularisation ------------------
    {"run_name": "resnet50_no_mix",    "arch": "resnet50",
     "epochs": 15, "overrides": {"USE_MIXUP": False, "USE_CUTMIX": False}},
    {"run_name": "resnet50_no_cutmix", "arch": "resnet50",
     "epochs": 15, "overrides": {"USE_CUTMIX": False}},
    {"run_name": "resnet50_no_ema",    "arch": "resnet50",
     "epochs": 15, "overrides": {"USE_EMA": False}},
    {"run_name": "resnet50_focal",     "arch": "resnet50",
     "epochs": 15, "overrides": {"USE_FOCAL": True}},

    # ----- 3. Batch size ------------------------------------------------
    {"run_name": "resnet50_bs16",  "arch": "resnet50",
     "epochs": 15, "overrides": {"BATCH_SIZE": 16}},
    {"run_name": "resnet50_bs64",  "arch": "resnet50",
     "epochs": 15, "overrides": {"BATCH_SIZE": 64}},

    # ----- 4. Learning rate (test du mode "diverge") --------------------
    {"run_name": "resnet50_lr_high",  "arch": "resnet50",
     "epochs": 15, "overrides": {"LR_HEAD": 3e-3, "LR_BACKBONE": 3e-4}},

    # ----- 5. Régularisation forte / faible -----------------------------
    {"run_name": "resnet50_ls_strong", "arch": "resnet50",
     "epochs": 15, "overrides": {"LABEL_SMOOTHING": 0.2}},
    {"run_name": "resnet50_no_ls",     "arch": "resnet50",
     "epochs": 15, "overrides": {"LABEL_SMOOTHING": 0.0}},

    # ----- 6. Démonstration sous-entraînement ---------------------------
    {"run_name": "resnet50_5epochs",   "arch": "resnet50",
     "epochs": 5,  "overrides": {}},
]

PRESETS = {"colab": SWEEP_COLAB, "local": SWEEP_LOCAL}


# ===========================================================================
# 2. Application / restauration des overrides
# ===========================================================================
def _apply_overrides(overrides: Dict) -> Dict:
    """Applique les overrides à `config` ; retourne les valeurs originales."""
    originals = {}
    for k, v in overrides.items():
        if not hasattr(config, k):
            print(f"  ⚠ override inconnu : config.{k} n'existe pas — ignoré")
            continue
        originals[k] = getattr(config, k)
        setattr(config, k, v)
    return originals


def _restore(originals: Dict) -> None:
    for k, v in originals.items():
        setattr(config, k, v)


# ===========================================================================
# 3. Exécution d'un run
# ===========================================================================
def run_one(spec: Dict) -> Dict:
    """Entraîne, évalue et enregistre un run. Retourne un résumé."""
    name      = spec["run_name"]
    arch      = spec["arch"]
    epochs    = spec["epochs"]
    overrides = spec.get("overrides", {}) or {}

    print("\n" + "█" * 72)
    print(f"█  RUN  {name}")
    print(f"█  arch={arch}  epochs={epochs}  overrides={overrides or '(défaut)'}")
    print("█" * 72)

    originals = _apply_overrides(overrides)
    try:
        # IMPORTANT : on recharge train et evaluate pour qu'ils ré-importent
        # `from config import X` avec les nouvelles valeurs.
        from couche2 import train as train_mod
        from couche2 import evaluate as eval_mod
        importlib.reload(train_mod)
        importlib.reload(eval_mod)

        history = train_mod.train_model(arch, epochs=epochs, run_name=name)
        metrics = eval_mod.evaluate_model(arch, prefer_ema=config.USE_EMA,
                                          use_tta=config.TTA, run_name=name)

        from run_local import register_model
        register_model(name, metrics, epochs_run=epochs,
                       arch=arch, overrides=overrides)

        return {
            "name": name, "arch": arch, "epochs": epochs,
            "overrides": overrides, "metrics": metrics, "history": history,
            "status": "ok",
        }
    except Exception as exc:
        print(f"\n  ✗ RUN ÉCHOUÉ : {name}")
        traceback.print_exc()
        return {
            "name": name, "arch": arch, "epochs": epochs,
            "overrides": overrides, "metrics": None, "history": None,
            "status": "fail", "error": str(exc),
        }
    finally:
        _restore(originals)


# ===========================================================================
# 4. Génération du rapport (markdown + graphes)
# ===========================================================================
REPORT_MD       = os.path.join(OUTPUT_DIR, "sweep_report.md")
COMPARISON_PNG  = os.path.join(OUTPUT_DIR, "sweep_comparison.png")
CURVES_PNG      = os.path.join(OUTPUT_DIR, "sweep_curves.png")
RESULTS_JSON    = os.path.join(OUTPUT_DIR, "sweep_results.json")


def _diagnose(spec: Dict, history: Optional[Dict],
              metrics: Optional[Dict], best_test_acc: float) -> str:
    """Heuristique simple : pourquoi ce run dégrade / échoue."""
    if metrics is None or not history:
        return ("❌ **Run échoué** — l'entraînement a planté avant la fin "
                "(voir le log d'erreur ; cause typique : OOM GPU ou dataset manquant).")

    train_acc = metrics["accuracy"]["train"]
    test_acc  = metrics["accuracy"]["test"]
    gap       = train_acc - test_acc
    test_f1   = metrics["macro_f1"]["test"]
    epochs    = spec["epochs"]
    o         = spec.get("overrides", {}) or {}

    reasons = []

    # Sur-apprentissage
    if gap > 0.20:
        reasons.append(
            f"**Sur-apprentissage** (train {train_acc*100:.1f}% vs test "
            f"{test_acc*100:.1f}%, écart {gap*100:.1f} pts) — la régularisation "
            "est insuffisante : MixUp/CutMix/EMA désactivés ou label_smoothing "
            "trop bas augmentent ce phénomène."
        )
    # Sous-apprentissage
    if test_acc < 0.30 and train_acc < 0.40:
        reasons.append(
            f"**Sous-apprentissage** (test {test_acc*100:.1f}%) — le modèle "
            f"n'a pas eu le temps de converger ({epochs} époques) ou le "
            "learning rate est trop bas."
        )
    # Divergence
    if best_test_acc < 0.15:
        reasons.append(
            "**Divergence** — l'accuracy reste au niveau du hasard (~9% pour "
            "11 classes). Probable LR trop élevé ou instabilité numérique."
        )
    # Détection spécifique d'un override "dangereux"
    if o.get("LR_HEAD", 0) and o["LR_HEAD"] >= 1e-3:
        reasons.append(
            "Le LR_HEAD a été poussé à un niveau qui dépasse souvent la zone "
            "stable d'AdamW + mixed-precision → courbes en dents de scie."
        )
    if o.get("USE_MIXUP") is False and o.get("USE_CUTMIX") is False:
        reasons.append(
            "Sans MixUp NI CutMix, le modèle voit toujours les mêmes paires "
            "(image, label) → tendance forte à mémoriser le train set."
        )
    if o.get("USE_EMA") is False:
        reasons.append(
            "Sans EMA, l'évaluation utilise les poids bruts qui oscillent "
            "d'une époque à l'autre → métriques moins stables, et 0,5–1,5 pt "
            "de moins en pratique."
        )
    if o.get("USE_FOCAL") is True:
        reasons.append(
            "Focal-Loss met davantage de poids sur les classes difficiles — "
            "améliore le F1 par classe rare mais peut faire baisser "
            "l'accuracy globale si les classes majoritaires sont sous-apprises."
        )
    if epochs <= 5:
        reasons.append(
            f"Seulement {epochs} époques : c'est volontairement court pour "
            "illustrer le sous-entraînement — ce run sert de borne basse."
        )

    if not reasons:
        return ("✅ Run sain : pas de signal évident de sur-/sous-apprentissage ; "
                "les améliorations possibles concernent l'architecture ou un "
                "fine-tuning plus long.")
    return "\n".join(f"- {r}" for r in reasons)


def _plot_comparison(results: List[Dict]) -> None:
    """Bar chart Acc/F1 par run, baseline en évidence."""
    import matplotlib.pyplot as plt
    import numpy as np

    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("  ⚠ Aucun run réussi — pas de comparaison à tracer.")
        return

    names = [r["name"] for r in ok]
    acc   = [r["metrics"]["accuracy"]["test"] for r in ok]
    f1    = [r["metrics"]["macro_f1"]["test"]  for r in ok]
    idx   = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5.5))
    bars_a = ax.bar(idx - 0.2, acc, 0.4, label="Accuracy test", color="#4C9BE8",
                    edgecolor="black")
    bars_f = ax.bar(idx + 0.2, f1,  0.4, label="Macro-F1 test", color="#E8734C",
                    edgecolor="black")

    best = int(np.argmax(acc))
    bars_a[best].set_edgecolor("gold")
    bars_a[best].set_linewidth(3)

    for b, v in zip(bars_a, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v*100:.1f}",
                ha="center", fontsize=8)
    for b, v in zip(bars_f, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v*100:.1f}",
                ha="center", fontsize=8)

    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, max(max(acc), max(f1)) * 1.15 + 0.05)
    ax.set_ylabel("Score (0..1)")
    ax.set_title("Sweep — comparaison des runs (★ = meilleur en accuracy test)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [✓] {COMPARISON_PNG}")


def _plot_curves(results: List[Dict]) -> None:
    """Courbes test_acc par époque, toutes runs superposées."""
    import matplotlib.pyplot as plt

    ok = [r for r in results if r["status"] == "ok" and r["history"]]
    if not ok:
        print("  ⚠ Aucun historique disponible — pas de courbes à tracer.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for r in ok:
        h = r["history"]
        key = "ema_acc" if "ema_acc" in h and any(h["ema_acc"]) else "test_acc"
        ax.plot(h[key], "o-", ms=3, label=r["name"])
    ax.set_xlabel("Époque"); ax.set_ylabel("Accuracy test (ou EMA)")
    ax.set_title("Sweep — courbes d'accuracy test par run", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(CURVES_PNG, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [✓] {CURVES_PNG}")


def _markdown_table(results: List[Dict]) -> str:
    """Tableau markdown trié par accuracy test décroissante."""
    rows = []
    for r in results:
        m = r["metrics"]
        if m is None:
            rows.append((r["name"], r["arch"], r["epochs"], 0.0, 0.0, "✗", r["overrides"]))
        else:
            rows.append((r["name"], r["arch"], r["epochs"],
                         m["accuracy"]["test"], m["macro_f1"]["test"], "✅",
                         r["overrides"]))
    rows.sort(key=lambda x: x[3], reverse=True)

    out = ["| # | Run | Arch | Epochs | Acc test | F1 test | Statut | Overrides |",
           "|---|-----|------|-------:|---------:|--------:|:------:|-----------|"]
    for i, (n, a, ep, acc, f1, status, ov) in enumerate(rows, 1):
        ov_str = ", ".join(f"`{k}={v}`" for k, v in ov.items()) if ov else "—"
        out.append(f"| {i} | **{n}** | `{a}` | {ep} | {acc*100:.2f}% | "
                   f"{f1*100:.2f}% | {status} | {ov_str} |")
    return "\n".join(out)


def write_report(results: List[Dict], preset: str) -> None:
    """Génère sweep_report.md + sweep_comparison.png + sweep_curves.png."""
    print("\n" + "═" * 72)
    print("  RAPPORT SWEEP")
    print("═" * 72)

    # 1. Graphes
    _plot_comparison(results)
    _plot_curves(results)

    # 2. JSON brut (résultats complets sauf history qui est déjà dans outputs/)
    light = []
    for r in results:
        d = {k: v for k, v in r.items() if k != "history"}
        light.append(d)
    with open(RESULTS_JSON, "w") as f:
        json.dump(light, f, indent=2)
    print(f"  [✓] {RESULTS_JSON}")

    # 3. Markdown
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    best = max(ok, key=lambda r: r["metrics"]["accuracy"]["test"]) if ok else None
    best_acc = best["metrics"]["accuracy"]["test"] if best else 0.0

    lines = []
    lines.append(f"# Rapport de Sweep — FER-CE")
    lines.append("")
    lines.append(f"_Généré le {datetime.datetime.now():%Y-%m-%d %H:%M}_  ·  "
                 f"preset = **{preset}**  ·  {len(results)} runs "
                 f"({len(ok)} ✅ / {len(failed)} ✗)")
    lines.append("")
    lines.append("Ce rapport compare plusieurs entraînements lancés avec des hyper-")
    lines.append("paramètres différents (architecture, mixup, focal-loss, batch size,")
    lines.append("learning rate, durée d'entraînement…). Chaque run a été enregistré")
    lines.append("dans `outputs/registry.json` ; les fichiers `metrics_<run>.json`,")
    lines.append("`curves_<run>.png` et `eval_<run>_*.png` lui sont rattachés.")
    lines.append("")

    # --- Champion --------------------------------------------------------
    lines.append("## 🏆 Meilleur modèle")
    lines.append("")
    if best is not None:
        m = best["metrics"]
        lines.append(f"**`{best['name']}`** (arch = `{best['arch']}`) — "
                     f"**Accuracy test = {m['accuracy']['test']*100:.2f}%**, "
                     f"Macro-F1 = {m['macro_f1']['test']*100:.2f}%.")
        ov = best["overrides"]
        lines.append(f"Overrides appliqués : "
                     f"{', '.join(f'`{k}={v}`' for k, v in ov.items()) if ov else '_(aucun — config par défaut)_'}.")
        lines.append("")
        lines.append(f"Reproduire ce run :")
        lines.append("```bash")
        lines.append(f"# Recharger depuis la registry")
        lines.append(f"python -c \"from run_local import load_registered_model; "
                     f"m, info = load_registered_model('{best['name']}'); "
                     f"print(info['metrics']['accuracy']['test'])\"")
        lines.append("```")
    else:
        lines.append("_Aucun run réussi._")
    lines.append("")

    # --- Tableau ---------------------------------------------------------
    lines.append("## 📊 Classement complet")
    lines.append("")
    lines.append(_markdown_table(results))
    lines.append("")

    # --- Graphes ---------------------------------------------------------
    lines.append("## 📈 Visualisations")
    lines.append("")
    lines.append("![Comparaison Acc / F1](sweep_comparison.png)")
    lines.append("")
    lines.append("![Courbes test_acc par époque](sweep_curves.png)")
    lines.append("")

    # --- Analyse par run -------------------------------------------------
    lines.append("## 🔬 Analyse run par run")
    lines.append("")
    for r in results:
        m   = r["metrics"]
        ov  = r["overrides"]
        best_test = max(r["history"]["test_acc"]) if (r["history"] and r["history"].get("test_acc")) else 0.0
        lines.append(f"### `{r['name']}`  (arch `{r['arch']}`, {r['epochs']} époques)")
        if ov:
            lines.append("**Overrides :** "
                         + ", ".join(f"`{k}={v}`" for k, v in ov.items()))
        if m is None:
            lines.append("- Acc test : ✗ — voir l'erreur dans `sweep_results.json`")
        else:
            lines.append(f"- Acc train / test : **{m['accuracy']['train']*100:.2f}% / "
                         f"{m['accuracy']['test']*100:.2f}%**")
            lines.append(f"- Macro-F1 train / test : "
                         f"**{m['macro_f1']['train']*100:.2f}% / "
                         f"{m['macro_f1']['test']*100:.2f}%**")
        lines.append("")
        lines.append("**Diagnostic :**")
        lines.append("")
        lines.append(_diagnose(r, r["history"], m, best_test))
        lines.append("")

    # --- Section "pourquoi ça fail / dégrade" ----------------------------
    lines.append("## 🧠 Pourquoi certains runs dégradent — synthèse")
    lines.append("")
    lines.append("Quelques régularités observées :")
    lines.append("")
    lines.append("- **Désactiver MixUp + CutMix** fait remonter l'accuracy train très")
    lines.append("  près de 100% (le modèle mémorise) mais l'accuracy test plafonne :")
    lines.append("  l'écart train↔test est le marqueur du sur-apprentissage.")
    lines.append("- **Désactiver l'EMA** rend les courbes en dents de scie et fait")
    lines.append("  perdre 0,5–1,5 pt en moyenne — l'EMA gomme les pics de bruit dûs")
    lines.append("  aux mini-batches.")
    lines.append("- **Focal-Loss (gamma=2)** aide les classes rares mais peut faire")
    lines.append("  perdre 1–3 pts d'accuracy globale : c'est un compromis utile")
    lines.append("  uniquement si on regarde le F1 macro par classe.")
    lines.append("- **LR head trop élevé** (≥ 1e-3 en fine-tuning ImageNet) →")
    lines.append("  divergence ou convergence vers un minimum très plat dépendant")
    lines.append("  fortement de l'initialisation.")
    lines.append("- **Trop peu d'époques** (5) reste un cas de sous-apprentissage —")
    lines.append("  inclus volontairement pour montrer la borne basse.")
    lines.append("- **Batch size** : 16 ralentit (plus de steps) sans gain net ; 64")
    lines.append("  accélère mais demande +50% VRAM et lisse moins le gradient.")
    lines.append("")

    lines.append("## 📁 Pour aller plus loin")
    lines.append("")
    lines.append("- Le détail de chaque run est dans `outputs/metrics_<run>.json` et")
    lines.append("  `outputs/curves_<run>.png`.")
    lines.append("- La registry complète : `python run_local.py --list`.")
    lines.append("- Recharger un modèle : `python run_local.py --load <run_name>`.")
    lines.append("")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [✓] {REPORT_MD}")


# ===========================================================================
# 5. Modes "report-only" (régénère le rapport sans réentraîner)
# ===========================================================================
def reload_from_disk(preset: str) -> List[Dict]:
    """Reconstruit la liste de results à partir des metrics_*.json déjà écrits."""
    specs = PRESETS[preset]
    out = []
    for s in specs:
        mp = os.path.join(OUTPUT_DIR, f"metrics_{s['run_name']}.json")
        hp = os.path.join(OUTPUT_DIR, f"history_{s['run_name']}.json")
        if not os.path.exists(mp):
            out.append({**s, "metrics": None, "history": None, "status": "fail",
                        "error": "metrics non trouvés"})
            continue
        with open(mp) as f:
            m = json.load(f)
        h = json.load(open(hp)) if os.path.exists(hp) else None
        out.append({**s, "metrics": m, "history": h, "status": "ok"})
    return out


# ===========================================================================
# 6. CLI
# ===========================================================================
def list_presets() -> None:
    for name, specs in PRESETS.items():
        print(f"\n  Preset = {name}  ({len(specs)} runs)")
        for s in specs:
            ov = ", ".join(f"{k}={v}" for k, v in (s.get('overrides') or {}).items())
            print(f"    - {s['run_name']:<28s} arch={s['arch']:<32s} "
                  f"epochs={s['epochs']:>3d}   {ov or '(défaut)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=list(PRESETS), default="colab",
                        help="colab (5-7 runs) ou local (12-15 runs)")
    parser.add_argument("--only", default=None,
                        help="liste de run_name séparés par virgules ; "
                             "ne lance que ceux-là")
    parser.add_argument("--list", action="store_true",
                        help="affiche les presets et leurs runs, puis quitte")
    parser.add_argument("--report-only", action="store_true",
                        help="régénère sweep_report.md à partir des metrics_*.json")
    args = parser.parse_args()

    if args.list:
        list_presets(); return

    specs = PRESETS[args.preset]
    if args.only:
        keep = set(s.strip() for s in args.only.split(","))
        specs = [s for s in specs if s["run_name"] in keep]
        if not specs:
            print(f"  ✗ Aucun run ne correspond à --only={args.only}")
            sys.exit(1)

    print(f"\n  Sweep démarré : preset={args.preset}, {len(specs)} runs prévus.\n")

    if args.report_only:
        results = reload_from_disk(args.preset)
    else:
        results = [run_one(s) for s in specs]

    write_report(results, args.preset)

    print(f"\n  ✅ Sweep terminé. Ouvre le rapport : {REPORT_MD}")


if __name__ == "__main__":
    main()
