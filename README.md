# Vitamin Health AI — Vitamin D Prediction + Food Intelligence (Vitamin B12)

A Streamlit application with two modules:

1. **Vitamin D Prediction Module** (original, unchanged): a reproducible Vitamin D level regression project.
2. **Food Intelligence — Vitamin B12 Intake Estimation** (new): photo-assisted food logging, user-confirmed portions, traceable nutrition data, B12 intake calculation and transparent dietary-pattern analysis.

The app has three pages (top navigation): **Vitamin D Analysis**, **Food Intelligence — Vitamin B12**, **Nutrition History**.

> **Medical disclaimer.** Educational project, not a medical device. It has not been clinically validated. It does not diagnose any condition, does not measure B12 in the body, and never recommends supplement doses.

## Module 1: Vitamin D prediction

## Data and integrity

`data/VitaminD_Dataset.csv` is the 10,000-row custom CSV copied from the supplied legacy repository. That repository does not document whether it is synthetic, so generated reports label its provenance as **custom dataset; synthetic status not documented**. Do not present its metrics as clinical performance.

The regression target is `VitaminD_Level_ng_ml`. `VitaminD_Category`, `Deficiency_Status`, and `Risk_Score` are excluded before any split, preprocessing, cross-validation, or prediction.

## Install and run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\streamlit.exe run app.py
```

## Streamlit experience and optional AI assistant

The Streamlit app provides a quick estimate and a detailed analysis view with
transparent lifestyle explanations, recommendations, and prediction-based
what-if scenarios. All predictions, including the scenarios, use the saved
pipeline through `src.predict`; they do not modify training or evaluation.

The detailed view also includes an optional OpenAI chat assistant. Set its key
in the process environment before starting Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-key"
.\.venv\Scripts\streamlit.exe run app.py
```

The key is never requested in the interface, stored in source code, or read
from an `.env` file. If `OPENAI_API_KEY` is unavailable, the chat continues in
offline, rule-based guidance mode. Responses are educational only and are not
medical diagnoses or supplement prescriptions.

Training makes an 80/20 held-out split, compares Linear Regression, Random Forest, XGBoost, and CatBoost with five-fold shuffled cross-validation on the training data, then saves:

- `models/vitamin_d_pipeline.joblib` — best fully fitted preprocessing and model pipeline
- `models/metadata.json` — input schema and data provenance
- `reports/model_metrics.csv` — CV and holdout R², MAE, and RMSE
- `reports/holdout_predictions.csv` — predictions for the untouched holdout set

The winning model is selected by mean cross-validation RMSE. The holdout set is used once for reporting.

## Module 2: Food Intelligence — Vitamin B12 intake estimation

### What it does (implemented)

1. Upload a meal photo (JPG/PNG/WEBP, ≤ 5 MB; content is decoded and verified, not trusted by extension) **or** pick a food manually.
2. A local Food-101 classifier *suggests* labels with confidence scores (optional, see below). Suggestions are never auto-accepted.
3. You confirm/edit the foods and enter the portion: standard serving (only where documented), custom grams, or millilitres (only where a documented density exists).
4. The preview shows food, portion, B12 per 100 g, source and estimated B12 **before** you add it to Breakfast / Lunch / Dinner / Snacks.
5. Entries can be edited, re-labelled, re-portioned, removed, or a whole day reset (with confirmation).
6. Dashboard: today's estimate, foods/meals logged, completeness, average over *recorded* days, contribution-by-food, meal comparison, daily trend, natural-vs-fortified split. Empty log → empty state, never fake charts.
7. Adequacy comparison against a selectable dietary reference, rule-based dietary-pattern findings (each with its evidence), a *dietary screening indicator*, and educational recommendations.
8. Nutrition History page: recorded days, window averages, CSV export/import.

### Calculation

```
B12 (µg) = portion mass (g) × B12 content (µg per 100 g) / 100
```

Example (arithmetic illustration only): 200 g × 1.3 µg/100 g = 2.6 µg. Values are rounded **only for display**.

**Missing data is never zero.** A food with no verified B12 value is *unknown*: it is excluded from totals, listed explicitly, and totals are labelled "lower bounds". Unlogged days are excluded from averages (not counted as 0). Intake is always recomputed from the log + database, so editing an entry cannot double count; adds are idempotent on `entry_id`, so Streamlit reruns cannot duplicate entries.

### Architecture and data flow

```
photo / manual choice → food_recognition (optional model, replaceable via FoodRecognizer protocol)
        → user confirms food + portion → food_log (FoodLog, session state, storage-agnostic)
        → b12_calculator (+ nutrition_database from data/food_nutrition.csv)
        → dietary_analysis (reference comparison, patterns, screening indicator)
        → recommendations → dashboard (Plotly) → src/ui/food_page.py, history_page.py
```

| Module | Responsibility |
|---|---|
| `src/food_intelligence/nutrition_database.py` | Load + validate the CSV (`python -m src.food_intelligence.nutrition_database`) |
| `src/food_intelligence/food_recognition.py` | Image validation, replaceable recognizer, label→food mapping, low-confidence handling |
| `src/food_intelligence/b12_calculator.py` | Formula, unit conversion, aggregation (no Streamlit) |
| `src/food_intelligence/food_log.py` | Entries, idempotent add, in-place edit, serialisation hooks |
| `src/food_intelligence/dietary_analysis.py` | References, completeness, findings, screening indicator |
| `src/food_intelligence/recommendations.py` | Non-prescriptive educational guidance |
| `src/food_intelligence/dashboard.py` | Plotly figure builders (return `None` when no data) |
| `src/ui/food_page.py`, `src/ui/history_page.py` | Streamlit pages |

### Food database (`data/food_nutrition.csv`)

Columns: `food_id, food_name, category, aliases, diet_suitability, serving_description, serving_weight_g, density_g_per_ml, b12_ug_per_100g, b12_value_status, source_name, source_reference, data_version, is_fortified, food_preparation, data_quality_note`.

* Animal-source values (egg, milk, curd, cottage cheese, salmon, tilapia, sardines, goat/mutton, chicken breast, beef liver, clams) are transcribed from **USDA FoodData Central (SR Legacy)**. Source and description are stored per row. They were entered without live access to FDC, so **verify them against FDC before formal reporting**.
* Plant foods are stored as 0 µg with status `reported_zero_plant` (USDA-reported; not a B12 source).
* **No value is invented.** Paneer, freshwater fish curry and chicken curry have no verified record here (consult IFCT 2017 or labels), so they are *missing*, not zero. Fortified cereal, nutritional yeast and fortified plant milk are brand-dependent: you enter the product-label value in the app, otherwise B12 stays unknown. Fortified (added) B12 is kept separate from natural B12.
* The app uses only this local file and makes no network requests while logging.

Validate the file: `.\.venv\Scripts\python.exe -m src.food_intelligence.nutrition_database`

### Image recognition model and limitations

* Default model: `nateraw/food` (ViT fine-tuned on Food-101) through Hugging Face Transformers, run **locally**, no paid API. Check the model card's licence and the Food-101 dataset terms before any use beyond coursework.
* Food-101 has 101 mostly Western single-dish classes and covers few Indian dishes. It returns top-k single-dish labels, not per-item detection on a thick, mixed plate. It cannot see portion size, ingredients or fortification.
* `data/food101_label_map.csv` maps a few labels to database foods (marked *approximate*). Unmapped labels just ask you to choose manually.
* **Recognition accuracy has not been measured for this project and no accuracy figure is claimed.**
* Weights missing, packages missing, or offline → the app says so and manual logging works fully.
* To swap the model, implement the `FoodRecognizer` protocol (`is_available`, `predict`) and pass it to `recognize_food`.

Optional install (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
# one-off, user-initiated download of weights (the only network use):
.\.venv\Scripts\python.exe -m src.food_intelligence.food_recognition
```

### Dietary-reference interpretation

Default is the **US adult** 2.4 µg/day (NIH ODS), clearly labelled as a US reference; pregnancy, lactation, child values and an India ICMR-NIN adult value (2.2 µg/day, verify before citing) are selectable. Labels: *Below / Near (≥75 %) / At or above the selected dietary reference*, never "deficient". One day of logging is not long-term intake; the app flags incomplete logs. The *dietary screening indicator* is a transparent rule-based summary of the log, **not** a clinical probability. No B12 blood levels are shown and no ML risk model is trained: that would need real labelled clinical data, proper splits and evaluation.

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

97 tests: the original Vitamin D tests (unmodified) plus nutrition database, calculator, food log, dietary analysis, recognition (valid/invalid image, missing weights, low confidence, manual correction) and Streamlit `AppTest` integration (navigation, manual logging without a vision model or OpenAI key, Vitamin D page still predicts).

### Known limitations

* Session-state storage only (per browser session, not secure multi-user storage). Use CSV export/import on the History page to keep data. A database repository can be added behind `FoodLog.to_records/from_records`.
* Small curated database; mixed Indian dishes mostly lack verified values. Portions come from you, never from the photo.
* Dietary intake ≠ blood B12 status; absorption, medicines and conditions are not modelled.
* Photo recognition is unevaluated and Western-dish biased.

### Future work (not implemented)

Indian food-recognition model fine-tuned and evaluated on a documented dataset; IFCT 2017 values; SQLite persistence with per-user isolation; validated portion estimation; a B12 risk model trained on real, documented clinical labels with proper evaluation.
