@echo off
rem Open the latest raw data file in Microsoft Office Excel
rem Note: close Excel before running the 40 prediction script (file lock)
start "" "C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE" "%~dp0data_ori\2026_ori_data_new.xlsx"
