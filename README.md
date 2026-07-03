
# Goldpreis-Prognose (Multi-Horizon Forecast)

Dieses Repository enthält ein Machine-Learning-Projekt zur Vorhersage von Goldpreisänderungen über verschiedene Zeithorizonte (1, 5 und 30 Handelstage). Das primäre Ziel der Untersuchung ist es, mithilfe verschiedener Regressionsalgorithmen zukünftige Werte der Zielspalte `Adj Close` zu prognostizieren und diese kritisch mit einer naiven Baseline zu vergleichen.

Ein besonderer Schwerpunkt dieses Teilprojekts liegt auf einer robusten **Support Vector Regression (SVR)**-Pipeline, die gezielt Preisänderungen (Returns) statt absoluter Preisniveaus lernt, um klassische Datenleckage und das bloße "Hinterherhinken" hinter dem letzten Kurs zu vermeiden.


---

## 📂 Projektstruktur

Das Repository ist wie folgt strukturiert:

```text
.
├── DATA/
│   ├── gold_train_pp.csv               # Vorverarbeitete Trainingsdaten (2011–2018)
│   └── gold_test_pp.csv                # Vorverarbeitete Testdaten / Holdout-Set (2019)
├── README/
│   ├── Bericht_Goldpreis.pdf           # Gesamter Projektbericht des Teams
│   └── Goldpreis-Prognose_Projektbericht.pdf # Detaillierter Einzelbericht (SVR Fokus)
├── main.py                             # Python-Skript für Pipeline, GridSearch & Visualisierung
├── SVR_ergebnisse_main.jpg             # Visualisierung der Vorhersage-Horizonte & Residuen
└── README.md                           # Diese Dokumentationsdatei
```

---

## 📊 Datenbeschreibung & Features

Die Datenbasis kombiniert historische Goldpreise mit verschiedenen globalen Marktindikatoren und Asset-Klassen, um komplexe Interdependenzen zu erfassen.

* **Zeitraum:** Trainingsdaten von 2011 bis 2018, Testdaten (Holdout) aus dem Jahr 2019.
* **Zielgröße (`Target`):** `Adj Close` (angepasster Schlusskurs) für die Horizonte:
* `Target_Adj_Close` (1 Tag in der Zukunft)
* `Target_Adj_5d` (5 Tage in der Zukunft)
* `Target_Adj_30d` (30 Tage in der Zukunft)


* **Feature-Sätze:**
* **Preistrends:** `Adj Close SMA5` (Gleitender Durchschnitt)
* **Aktienmärkte:** S&P 500 (`SP_Ajclose`), Dow Jones (`DJ_Ajclose`)
* **Rohstoffe & Edelmetalle:** Platin (`PLT_Price`), Öl (`USO_Adj Close`), GDX (Gold Miners ETF)
* **Währungen & Zinsen:** US-Dollar-Index (`USDI_Price`), EUR/USD Kurs (`EU_Price`), 10-jährige US-Anleihen (`USB_Price`)

---

## ⚙️ Methodik & Modell-Architektur (SVR)

Um eine statistisch valide und ökonomisch sinnvolle Prognose zu gewährleisten, wurde eine fortgeschrittene Architektur gewählt:

1. **Return-basierte Zielgröße:** Anstatt den absoluten Preis direkt zu schätzen (was bei Zeitreihen oft dazu führt, dass das Modell einfach den Wert von $t$ für $t+1$ vorhersagt), lernt das Modell die relative Preisänderung (`target_return = Preis_Zukunft - Preis_Aktuell`). Dies wird elegant über einen `TransformedTargetRegressor` gekapselt, der die Transformation vor dem Fit durchführt und für die Evaluation automatisch re-transformiert.
2. **Skalierung & Dimensionsreduktion:**
Innerhalb einer `sklearn.pipeline.Pipeline` werden die Daten mittels `SimpleImputer` bereinigt, mit `StandardScaler` standardisiert und über eine Hauptkomponentenanalyse (`PCA(n_components=0.95)`) dimensionsreduziert, um Multikolinearität zwischen den Marktindizes zu minimieren.
3. **Strikte Validierung (TimeSeriesSplit mit Gap):**
Für das Tuning via `GridSearchCV` wird ein `TimeSeriesSplit` verwendet. Um Datenleckage bei mehrteiligen Horizonten zu vermeiden, ist ein exakter mathematischer *Gap* (`gap = h`) eingebaut. Dadurch wird verhindert, dass Trainings-Samples Werte beinhalten, die zeitlich in das Validierungsfenster hineinreichen.

---

## 📈 Zentrale Erkenntnisse & Fazit

* **Die Naive Baseline als harter Gegner:** Die SVR-Ergebnisse und die Analysen im Team zeigen eindrucksvoll, dass im hocheffizienten Finanzmarkt die *Naive Baseline* (die Annahme, der Preis von morgen entspricht dem Preis von heute) extrem schwer zu schlagen ist.
* **Zeithorizonte:** Beim 1-Tage-Horizont entspricht der optimale Score nahezu exakt der Baseline – ein realistisches Verhalten, das die Abwesenheit von Arbitrage-Möglichkeiten widerspiegelt. Bei längeren Horizonten (30 Tage) neigen Modelle ohne strikte Validierung zu Optimismus, weshalb die SVR-Pipeline hier bewusst konservative und robustere Vorhersagen liefert.
* **Inkrementelle Learnings:** Das Projekt trennt sauber zwischen Cross-Validation-Scores (Trainingsphase) und echten Test-Scores (Holdout-Phase 2019), um Overfitting frühzeitig aufzudecken.

---

## 🚀 Installation & Ausführung

### Voraussetzungen

Stelle sicher, dass Python 3.8+ installiert ist. Die benötigten Bibliotheken können über pip installiert werden:

```bash
pip install numpy pandas scikit-learn matplotlib

```

### Skript ausführen

Das Skript lädt die Daten automatisch aus dem `DATA/` Ordner, trainiert die SVR-Modelle inklusive Hyperparametertuning für alle drei Horizonte und erzeugt ein detailliertes Diagramm der Vorhersehungen sowie der Residuen:

```bash
python main.py

```

---

## 👥 Projektbeteiligte & Team

Dieses Projekt wurde im Rahmen einer Gruppenarbeit realisiert. Hauptverantwortlich für die Konzeption, Implementierung und Dokumentation der hier beschriebenen SVR-Pipeline:

* **Noel Ciborro Fernandez** (Fokus: Support Vector Regression, Return-Transformation, TimeSeriesSplit-Validierung)

**Weitere Teammitglieder (Gesamtprojekt & alternative Algorithmen wie Random Forest, Lasso, etc.):**

* Mayada Esmail
* Deniz Alakus
* Tobias Behlick
* Nico Bleh

