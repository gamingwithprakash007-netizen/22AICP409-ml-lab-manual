# 22AICP409 — Machine Learning Laboratory Portal

Interactive single-page web application and complete Python codebase for **22AICP409: Machine Learning Laboratory** (Annamalai University, Faculty of Engineering & Technology, Department of Computer Science & Engineering — AI & ML, Semester IV).

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

## Deployment on Vercel
This repository is configured for **zero-config static hosting** on Vercel:
- `index.html` is served directly via Vercel Edge CDN.
- Python packages for running the experiments locally are in `experiments/requirements.txt`.
- `.vercelignore` and `vercel.json` ensure Vercel deploys immediately as a fast static site without triggering Python function build errors.

## Repository Structure
```text
├── index.html                                # Main interactive web portal
├── vercel.json                               # Vercel static routing config
├── .vercelignore                             # Prevents serverless build conflicts
├── main.py                                   # Fallback entrypoint for Vercel
├── 22AICP409_Complete_ML_Lab_Manual.ipynb    # Master Jupyter Notebook for Google Colab
├── experiments/                              # All 18 standalone Python scripts (.py)
│   └── requirements.txt                      # Python dependencies for local execution
└── MLdatasets/                               # Sample CSV datasets for Experiments 1–4
```

## Running Experiments Locally
```bash
cd experiments
pip install -r requirements.txt
python ex01a_linear_regression.py
```
