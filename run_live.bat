@echo off
cd /d "C:\Users\rochv\OneDrive\projetcs\algo sw"
set PYTHONPATH=.
python -u scripts/_run_live.py config/live.yaml >> runner_persistent.log 2>&1
