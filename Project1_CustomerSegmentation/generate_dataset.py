"""
generate_dataset.py
--------------------
Generates a synthetic customer dataset for Project 1 (Customer Segmentation),
in the same style as the well-known "Mall Customers" dataset used for this
kind of assignment: CustomerID, Gender, Age, Annual Income (k$), Spending
Score (1-100).

No real dataset was supplied with the lab sheet, so this script builds one
with genuine, well-separated customer groups (young high spenders, older
conservative savers, average middle-income shoppers, etc.) so that K-Means
has real structure to discover - not just random noise.
"""

import numpy as np
import pandas as pd

np.random.seed(42)


def make_cluster(n, age_range, income_range, spending_range, gender_bias=0.5):
    age = np.random.randint(age_range[0], age_range[1], n)
    income = np.random.randint(income_range[0], income_range[1], n)
    spending = np.random.randint(spending_range[0], spending_range[1], n)
    gender = np.random.choice(
        ["Male", "Female"], n, p=[gender_bias, 1 - gender_bias]
    )
    return age, income, spending, gender


# Five realistic customer segments, each with a distinct income/spending pattern
segments = [
    # (count, age_range, income_range($k), spending_score_range, gender_bias)
    (40, (18, 30), (15, 40), (60, 100), 0.45),   # young, low income, high spenders
    (40, (25, 45), (70, 140), (60, 100), 0.50),  # mid-age, high income, high spenders
    (40, (35, 60), (15, 40), (0, 40), 0.55),     # older, low income, low spenders
    (40, (35, 65), (70, 140), (0, 40), 0.50),    # older, high income, conservative
    (40, (28, 45), (40, 70), (40, 60), 0.50),    # average income, average spenders
]

ages, incomes, spendings, genders = [], [], [], []
for n, age_r, income_r, spend_r, bias in segments:
    a, i, s, g = make_cluster(n, age_r, income_r, spend_r, bias)
    ages.append(a)
    incomes.append(i)
    spendings.append(s)
    genders.append(g)

age = np.concatenate(ages)
income = np.concatenate(incomes)
spending = np.concatenate(spendings)
gender = np.concatenate(genders)

n_total = len(age)
customer_id = np.arange(1, n_total + 1)

# Shuffle rows so the segments aren't sitting in visible blocks in the raw file
idx = np.random.permutation(n_total)

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Gender": gender[idx],
    "Age": age[idx],
    "Annual_Income_k$": income[idx],
    "Spending_Score": spending[idx],
})

# A little realistic messiness for the "clean and preprocess" step to matter:
# a few missing values and one duplicate row.
missing_rows = np.random.choice(df.index, size=5, replace=False)
for r in missing_rows:
    col = np.random.choice(["Age", "Annual_Income_k$", "Spending_Score"])
    df.loc[r, col] = np.nan

df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one duplicate row

df.to_csv("dataset/Mall_Customers.csv", index=False)
print(f"Saved dataset/Mall_Customers.csv with {len(df)} rows (including 1 duplicate, 5 missing values).")
print(df.head())
