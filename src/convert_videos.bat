@echo off
setlocal enabledelayedexpansion

set "SOURCE_FOLDER=D:\Mestrado\redes_neurais\dados_filtrados\videos"
set "DEST_FOLDER=D:\Mestrado\redes_neurais\dados_filtrados\videos_720p"

if not exist "%DEST_FOLDER%" mkdir "%DEST_FOLDER%"

for %%f in ("%SOURCE_FOLDER%\*") do (
    echo Converting: %%~nxf
    ffmpeg -i "%%f" -vf scale=-2:720 -c:v libx264 -crf 20 -preset medium -c:a copy "%DEST_FOLDER%\%%~nxf"
)

echo All videos converted!
pause