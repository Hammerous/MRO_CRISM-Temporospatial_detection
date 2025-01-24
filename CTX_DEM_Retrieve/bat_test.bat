@echo off
set input_file=%1
set destination=%2

for /f "usebackq delims=" %%f in ("%~1") do (
    echo %%f
    rclone copy "%%f" "%~2"
)