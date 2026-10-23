# Case Study 1: Analysis and Clustering of Online Retail Transactions
Fundamentals of Data Science - Marru Mahathi Rao (Roll No 160124733150)

## What this project does
Analyses 7,501 online-retail shopping baskets: cleaning, descriptive statistics, one-hot encoding, PCA, and a comparison of K-means, hierarchical, DBSCAN and spectral clustering (silhouette coefficient and Davies-Bouldin index), followed by cluster profiling, limitations and business implications.

## Files
| File | Description |
|---|---|
| `Case_Study_1_Online_Retail.ipynb` | Executed Jupyter notebook (code, outputs, plots, written interpretations) |
| `case_study_1.py` | Same analysis as a plain Python script |
| `Case_Study_1_Report.docx` | Report (title page, problem statement, methodology, statistical findings, model evaluation, conclusion) |
| `Online retail_new.csv` | Dataset (no header row) |
| `requirements.txt` | Python packages used |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Case_Study_1_Online_Retail.ipynb    # Kernel -> Restart & Run All
# or
python case_study_1.py
```

## Key results
- 7,501 baskets, 119 unique products, mean basket size 3.91 (right-skewed).
- PCA: the first two components explain only 11.2% of the variance.
- K-means (K=3) gives the most usable segmentation (silhouette about 0.09-0.10); no algorithm finds strong cluster structure. DBSCAN's good-looking scores come from discarding most points as noise.
