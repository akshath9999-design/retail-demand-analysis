@echo off
echo ===================================================
echo Pushing Retail Demand Analysis to GitHub...
echo Repository: https://github.com/akshath9999-design/retail-demand-analysis
echo ===================================================

cd /d "C:\Users\hi\.gemini\antigravity\scratch\retail-demand-analysis"
"C:\Users\hi\AppData\Local\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd\git.exe" push -u origin main

echo ===================================================
echo Done! Check your repo at:
echo https://github.com/akshath9999-design/retail-demand-analysis
echo ===================================================
pause
