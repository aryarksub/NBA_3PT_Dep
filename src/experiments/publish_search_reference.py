"""Publish the selected search cache, then rebuild alternative summaries and plots.

The historical corrected 15ft/1ft snapshot is retained under search_sensitivity.
"""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from candidate_search_sweep import publish_reference
from cf_alternatives import CELL_SIZE, MAX_RADIUS

publish_reference(pd.read_csv('data/alt_exp_pts.csv'), CELL_SIZE, MAX_RADIUS)
for script in ['cf_alternatives.py','cf_alternatives_figures.py']:
    runpy.run_path('src/experiments/'+script,run_name='__main__')
