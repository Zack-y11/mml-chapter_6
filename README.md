
# Chapter 6 — Probability & Statistics (ML Concepts)

This folder is a small, runnable collection of Python scripts and mini-examples that cover probability space basics, discrete vs. continuous probability intuition, simple sampling experiments (dice/cards), Bayes-style sum/product reasoning, and basic statistics (mean/covariance).

The goal is to keep each concept in a standalone file so you can run it and observe outputs/plots without needing a larger framework.

## What’s inside

- `probability_space.py` — core probability space ideas (events, outcomes, probabilities).
- `dice_roll_prob.py` — dice roll probability experiments/examples.
- `card_drawing_ps.py` — card drawing probability space examples.
- `house-price.py` — a simple “toy” example related to house-price variables (often used to illustrate distributions / likelihoods).

Folders with focused examples:

- `bayes_sum_product/` — example code illustrating sum/product style reasoning (Bayes-rule style manipulations).
- `discretevscontinous_prob/` — example code comparing discrete vs. continuous probability.
- `statistics_mean_cov/` — example code for mean, variance/covariance.

## Requirements

- Python 3.10+ recommended
- Packages are listed in `requirements.txt`

This workspace already includes a local virtual environment in `ml-venv/`.

## Setup

### Option A: Use the included venv (recommended)

PowerShell (Windows):

```powershell
\ml-venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Option B: Create a fresh venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run examples

Run any script from the repo root:

```powershell
python probability_space.py
python dice_roll_prob.py
python card_drawing_ps.py
python house-price.py
```

Run folder examples by pointing Python at the file:

```powershell
python .\bayes_sum_product\example.py
python .\discretevscontinous_prob\example.py
python .\statistics_mean_cov\example.py
```

## Notes

- If a script opens a plot window, make sure you’re running in an environment where GUI windows are allowed (typical local Python on Windows is fine).
- If you see import errors, re-run `python -m pip install -r requirements.txt` inside the activated environment.

## Folder structure (high level)

```
chapter_6/
	README.md
	requirements.txt
	probability_space.py
	dice_roll_prob.py
	card_drawing_ps.py
	house-price.py
	bayes_sum_product/
		example.py
	discretevscontinous_prob/
		example.py
	statistics_mean_cov/
		example.py
	ml-venv/              # local virtual environment (optional to use)
```

