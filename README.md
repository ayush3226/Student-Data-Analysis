# DataLens — Visual EDA
A local Python/Streamlit application. No API key needed. Data is analyzed locally when you run it on your own computer.

## Run in VS Code
1. Extract the ZIP and open the DataLens folder in VS Code.
2. Install Python 3.11 or 3.12 and the VS Code Python extension.
3. Open Terminal > New Terminal.
4. Create a virtual environment: `python -m venv .venv` (macOS may use python3).
5. Activate: Windows PowerShell `.venv\Scripts\Activate.ps1`; macOS/Linux `source .venv/bin/activate`.
6. Run `python -m pip install -r requirements.txt`.
7. Run `python -m streamlit run app.py`.
8. Open http://localhost:8501 if the browser does not open automatically.

Alternatively select the .venv interpreter in VS Code and launch Run DataLens via Run and Debug.

## Features
- Demo sales data to explore immediately
- CSV upload, separator and encoding controls
- Numeric and categorical filtering
- Dataset KPIs, types, descriptive statistics and observations
- Missingness heatmap, data quality and duplicate counts
- Histograms, outlier box plots, categorical frequency charts
- Pearson/Spearman correlations and scatter plots
- Date conversion and time-series aggregation
- Custom chart builder and filtered CSV / quality report downloads

Plotly's toolbar lets you zoom, pan, reset and download charts as PNG. Large scatter plots are sampled to 10,000 rows; previews and some charts have stated limits. The app analyzes filtered rows, including duplicate and missing counts. Date conversions are optional and invalid date values become missing. CSV schema is inferred by pandas. For very large files, available RAM is the practical limit; upload size is capped at 200 MB. Correlation does not establish causation.
