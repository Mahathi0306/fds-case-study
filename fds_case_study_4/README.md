# Case Study 4: Predicting Student Dropout and Academic Success
Fundamentals of Data Science - Marru Mahathi Rao (Roll No 160124733150)

## What this project does
Analyses 4,424 students (UCI "Predict Students' Dropout and Academic Success"): data understanding, EDA, a Kruskal-Wallis hypothesis test on first-semester grades, an early-warning multinomial logistic regression (plus comparisons), interpretation, limitations, recommendations and responsible-use notes.

## Files
| File | Description |
|---|---|
| `case_study_4.ipynb` | Executed Jupyter notebook (Parts A-E with explanations) |
| `case_study_4.py` | Same analysis as a plain Python script |
| `Case_Study_4_Report.docx` | Report (title page, problem statement, methodology, statistical findings, model evaluation, conclusion) |
| `Studentdata.csv` | Dataset (semicolon-separated) |
| `requirements.txt` | Python packages used |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook case_study_4.ipynb      # Kernel -> Restart & Run All
# or
python case_study_4.py
```

## Key results
- Outcomes: 49.9% Graduate, 32.1% Dropout, 17.9% Enrolled (imbalanced).
- Kruskal-Wallis: first-semester grades differ across outcomes (p < 0.001, epsilon-squared about 0.25, a large effect).
- Early-warning logistic regression (no second-semester data): accuracy 74.6%, macro F1 0.66; weak on the Enrolled class.
- Associations only: nothing here shows causation.
