# Suggested Commands
- Install deps: `pip install -r requirements.txt`
- Run FastAPI app locally: `python3 run_local.py`
- Alternate dev server: `uvicorn app.main:app --host 0.0.0.0 --port 25252`
- Run interview pipeline smoke test: `python3 -m unittest tests.test_interview_sheet_mapping`
- Bootstrap project on macOS/Linux: `./setup.sh` (after `chmod +x setup.sh`).