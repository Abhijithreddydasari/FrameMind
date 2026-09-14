@echo off
setlocal
set FM_RUN_MODEL_TESTS=1
set HF_HOME=%~dp0..\.cache\huggingface
if "%~1"=="--offline" set HF_HUB_OFFLINE=1
"%~dp0..\.venv\Scripts\python.exe" -m pytest "%~dp0..\tests\integration\test_model_smoke.py" -q -s
exit /b %errorlevel%
