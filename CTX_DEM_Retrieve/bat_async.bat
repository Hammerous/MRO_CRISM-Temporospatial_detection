@echo off
setlocal enabledelayedexpansion

REM Set the maximum number of simultaneous jobs
set max_jobs=4

REM Initialize the job counter
set /a job_count=0

REM Input file and destination path
set input_file=%1
set destination=%2

REM Function to wait for running jobs to finish when the limit is reached
:wait_for_jobs
set /a job_count=0
for /f "tokens=*" %%j in ('tasklist ^| findstr /i "rclone.exe"') do (
    set /a job_count+=1
)
if !job_count! geq %max_jobs% (
    timeout /t 1 >nul
    goto wait_for_jobs
)

REM Main loop to process input file
for /f "usebackq delims=" %%f in ("%~1") do (
    echo Copying: %%f
    start "" /b cmd /c rclone copy "%%f" "%~2" & echo Job completed for %%f & pause
    call :wait_for_jobs
)
exit /b
