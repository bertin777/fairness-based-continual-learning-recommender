@echo off
echo =====================================================================
echo  Fairness-Aware Continual Learning Recommendation System Dashboard
echo =====================================================================
echo.
echo Launching Streamlit Dashboard...
echo Local prototype: http://localhost:8501
echo.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
pause
