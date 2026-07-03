import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Unbegrenzte Ausgabe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None


# --------------------------------------------------
# EINSTELLUNGEN
# --------------------------------------------------
MODEL_NAME = "SVR"

# Horizont (in Handelstagen)
HORIZON_TARGETS = {
    1: "Target_Adj_Close",
    5: "Target_Adj_5d",
    30: "Target_Adj_30d",
}
HORIZONS = list(HORIZON_TARGETS.keys())

PARAM_GRID = {
    'model__regressor__C': [1.0, 10.0, 50.0],
    'model__regressor__epsilon': [0.01, 0.05, 0.1],
    'model__regressor__gamma': ['scale', 'auto'],
}

# --------------------------------------------------
# DATEIPFADE
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA" / "pp"
PARAMS_DIR = BASE_DIR / "DATA" / "gridsearch_params"

TRAIN_PATH = "DATA/gold_train_pp.csv"
TEST_PATH = "DATA/gold_test_pp.csv"

# --------------------------------------------------
# DATEN LADEN
# --------------------------------------------------
print("\n" + "=" * 40)
print("DATEN LADEN")
print("=" * 40)

df_train_raw = pd.read_csv(TRAIN_PATH)
df_test_raw = pd.read_csv(TEST_PATH)

print("Train Shape:", df_train_raw.shape)
print("Test Shape:", df_test_raw.shape)

# Datum separat vorhalten (nur für die Visualisierung / Overlap-Check)
# NICHT als Feature verwenden
if "Date" in df_train_raw.columns:
    dates_train_full = pd.to_datetime(df_train_raw["Date"])
    dates_test_full = pd.to_datetime(df_test_raw["Date"])

    # Train/Test-Overlap-Check (Leak-Quelle)
    overlap = set(dates_train_full).intersection(set(dates_test_full))
    print(f"\nÜberlappende Daten Train/Test: {len(overlap)}")
    print(f"Train: {dates_train_full.min()} bis {dates_train_full.max()}")
    print(f"Test:  {dates_test_full.min()} bis {dates_test_full.max()}")
    if len(overlap) > 0:
        print("⚠️  WARNUNG: Train- und Testdaten überlappen sich zeitlich – möglicher Data Leak!")
else:
    print("Keine 'Date'-Spalte gefunden – nutze fortlaufenden Index für die Visualisierung.")
    dates_train_full = pd.Series(range(len(df_train_raw)))
    dates_test_full = pd.Series(range(len(df_test_raw)))

# --------------------------------------------------
# ERGEBNISSE
# --------------------------------------------------
plot_data = {}
cv_results = {}

for h in HORIZONS:
    print(f"\n--- Trainiere Modell für {h}-Tage-Horizont ---")
    target_col = HORIZON_TARGETS[h]

    if target_col not in df_train_raw.columns or target_col not in df_test_raw.columns:
        raise KeyError(f"Zielspalte '{target_col}' nicht in den Daten gefunden.")

    # Alle Target_*-Spalten (müssen komplett aus den Features raus, unabhängig vom Horizont)
    target_columns_all_train = [c for c in df_train_raw.columns if c.startswith("Target_")]
    target_columns_all_test = [c for c in df_test_raw.columns if c.startswith("Target_")]

    # Zeilen ohne gültigen Zielwert für DIESEN Horizont rauswerfen (Rand der Zeitreihe)
    train_mask = df_train_raw[target_col].notna()
    test_mask = df_test_raw[target_col].notna()

    df_train_h = df_train_raw.loc[train_mask].reset_index(drop=True)
    df_test_h = df_test_raw.loc[test_mask].reset_index(drop=True)
    dates_train_h = dates_train_full.loc[train_mask].reset_index(drop=True)
    dates_test_h = dates_test_full.loc[test_mask].reset_index(drop=True)

    y_train_abs = df_train_h[target_col].copy()
    y_test_abs = df_test_h[target_col].copy()

    drop_columns_train = ["Date", *target_columns_all_train]
    drop_columns_test = ["Date", *target_columns_all_test]

    X_train = df_train_h.drop(columns=drop_columns_train, errors="ignore")
    X_test = df_test_h.drop(columns=drop_columns_test, errors="ignore")

    # --- Data-Leak-Checks ---
    remaining_targets_train = [c for c in X_train.columns if c.startswith("Target_")]
    if remaining_targets_train:
        raise ValueError(f"Data Leakage: Target-Spalten in X_train gefunden: {remaining_targets_train}")

    remaining_targets_test = [c for c in X_test.columns if c.startswith("Target_")]
    if remaining_targets_test:
        raise ValueError(f"Data Leakage: Target-Spalten in X_test gefunden: {remaining_targets_test}")

    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Trainings- und Testdaten besitzen nicht dieselben Feature-Spalten.")

    if "Adj Close" not in X_train.columns:
        raise KeyError(
            "Spalte 'Adj Close' wird für die Baseline/Preisrückrechnung benötigt, "
            "ist aber nicht in den Features enthalten."
        )

    # --- Return-basiertes Ziel (Änderung statt absoluter Preis) ---
    current_price_train = X_train["Adj Close"].to_numpy()
    current_price_test = X_test["Adj Close"].to_numpy()

    y_train_return = y_train_abs.to_numpy() - current_price_train
    y_test_abs_np = y_test_abs.to_numpy()

    X_train_h = X_train.to_numpy()
    X_test_h = X_test.to_numpy()

    # ============================================================
    # MODELL
    # ============================================================
    feature_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
    ])

    base_model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    pipeline = Pipeline(steps=[
        ('features', feature_pipeline),
        ('model', TransformedTargetRegressor(regressor=base_model, transformer=StandardScaler())),
    ])

    # --- Cross-Val Score VOR dem GridSearch ---
    tscv = TimeSeriesSplit(n_splits=3, test_size=250, gap=h)

    pre_scores = cross_val_score(pipeline, X_train_h, y_train_return, cv=tscv,
                                  scoring='neg_mean_squared_error', n_jobs=-1)
    pre_cv_rmse = np.sqrt(-pre_scores.mean())
    print(f"[{h}d] Cross-Val RMSE VOR GridSearch (Standard-Parameter, {tscv.get_n_splits()} Folds): "
          f"{pre_cv_rmse:.4f} (Folds einzeln: {np.round(np.sqrt(-pre_scores), 4).tolist()})")

    # ============================================================
    # GRID-SEARCH
    # ============================================================
    grid_search = GridSearchCV(pipeline, PARAM_GRID, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_h, y_train_return)
    best_pipeline = grid_search.best_estimator_

    # --- Vorhersage und Preisrekonstruktion ---
    return_pred = best_pipeline.predict(X_test_h)
    y_pred_h = current_price_test + return_pred

    # --- Metriken ---
    mae = mean_absolute_error(y_test_abs_np, y_pred_h)
    rmse = np.sqrt(mean_squared_error(y_test_abs_np, y_pred_h))
    r2 = r2_score(y_test_abs_np, y_pred_h)
    p_mae = mean_absolute_error(y_test_abs_np, current_price_test)
    p_rmse = np.sqrt(mean_squared_error(y_test_abs_np, current_price_test))

    post_cv_rmse = np.sqrt(-grid_search.best_score_)

    print(f"[{h}d] Best Params: {grid_search.best_params_}")
    print(f"[{h}d] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f} "
          f"| Baseline MAE: {p_mae:.2f} | Baseline RMSE: {p_rmse:.2f}")
    print(f"[{h}d] Cross-Val RMSE NACH GridSearch (bestes Modell, {tscv.get_n_splits()} Folds): "
          f"{post_cv_rmse:.4f} (± {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f} auf MSE-Skala)")
    print(f"[{h}d] Verbesserung durch GridSearch: {pre_cv_rmse - post_cv_rmse:+.4f} RMSE "
          f"({(pre_cv_rmse - post_cv_rmse) / pre_cv_rmse * 100:+.1f} %)")

    cv_results[h] = {
        'pre_cv_rmse': pre_cv_rmse,
        'post_cv_rmse': post_cv_rmse,
        'best_params': grid_search.best_params_,
    }

    plot_data[h] = {
        'dates': dates_test_h,
        'y_true': y_test_abs_np,
        'y_pred': y_pred_h,
        'baseline': current_price_test,
        'residuals': y_test_abs_np - y_pred_h,
    }

    # --- Beste Parameter speichern ---
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    random_number = random.randint(1000, 9999)
    params_path = PARAMS_DIR / f"SVR_{h}.txt"
    model_text = (
        f"model = SVR(\n"
        f"    kernel='rbf',\n"
        f"    C={grid_search.best_params_['model__regressor__C']},\n"
        f"    epsilon={grid_search.best_params_['model__regressor__epsilon']},\n"
        f"    gamma={grid_search.best_params_['model__regressor__gamma']!r},\n"
        f")\n"
    )
    with open(params_path, "w", encoding="utf-8") as file:
        file.write(model_text)
    print(f"[{h}d] Parameter gespeichert unter: {params_path}")


# ============================================================
# CROSS-VAL-SCORE VOR UND NACH GRID-SEARCH
# ============================================================
print("\n" + "=" * 40)
print("CROSS-VAL-SCORE VOR UND NACH GRID-SEARCH")
print("=" * 40)
print(f"{'Horizont':<10}{'CV RMSE vorher':<18}{'CV RMSE nachher':<18}{'Verbesserung':<15}")
for h in HORIZONS:
    cv = cv_results[h]
    verbesserung = cv['pre_cv_rmse'] - cv['post_cv_rmse']
    print(f"{h}d{'':<8}{cv['pre_cv_rmse']:<18.4f}{cv['post_cv_rmse']:<18.4f}{verbesserung:+.4f}")


# ============================================================
# VISUALISIERUNG
# ============================================================
print("\n" + "=" * 40)
print("VISUALISIERUNG")
print("=" * 40)

fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex='col')
fig.suptitle(f'{MODEL_NAME} – Multi-Horizon Gold Price Forecast', fontsize=16, fontweight='bold', y=0.98)

for idx, h in enumerate(HORIZONS):
    data = plot_data[h]
    dates = data['dates']
    res = data['residuals']

    # --- Preisverlauf ---
    axes[0, idx].plot(dates, data['y_true'], label='Echter Wert', color='#1f77b4', linewidth=1.5)
    axes[0, idx].plot(dates, data['y_pred'], label=f'Vorhersage {MODEL_NAME} {h}d', color='#FF7F0E',
                       linewidth=1.5, linestyle='--')
    axes[0, idx].plot(dates, data['baseline'], label='Naive Baseline', color='gray', linestyle=':', alpha=0.6)

    axes[0, idx].set_title(f'{h}-Tage-Horizont (Forecast)', fontweight='bold')
    axes[0, idx].set_ylabel('Preis (USD)')
    axes[0, idx].grid(True, alpha=0.3)
    axes[0, idx].legend(fontsize=9, loc='upper left')

    # --- Residuen ---
    bar_colors = ['#d62728' if r < 0 else '#2ca02c' for r in res]
    axes[1, idx].bar(dates, res, color=bar_colors, width=1.5)
    axes[1, idx].axhline(0, color='black', linewidth=0.8)

    axes[1, idx].set_title(f'Residuals {h}d', fontweight='bold')
    axes[1, idx].set_ylabel('Fehler (USD)')
    axes[1, idx].set_xlabel('Datum')
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svr_ergebnisse_main.png', dpi=150, bbox_inches='tight')
print("svr_ergebnisse_main.png gespeichert")
plt.show()
