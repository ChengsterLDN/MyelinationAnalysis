@echo off
echo ============================================================
echo ImageJ Macro Runner
echo ============================================================
echo.

REM UPDATE THESE PATHS TO MATCH YOUR INSTALLATION
set IMAGEJ_PATH="C:\Users\jonat\Desktop\Fiji\fiji-windows-x64.exe"
set MACRO_PATH="C:\Users\jonat\Myelination\Export-as-individual-images.ijm"
set LIF_PATH="C:\Users\jonat\Documents\My Documents\MecBioMed\MyelinationProject\Benz\Benz.lif"
set OUTPUT_DIR="extracted_images"

echo Using ImageJ: %IMAGEJ_PATH%
echo Macro: %MACRO_PATH%
echo LIF file: %LIF_PATH%
echo Output: %OUTPUT_DIR%
echo.
echo Starting execution...
echo.

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Run ImageJ macro
%IMAGEJ_PATH% --headless --run %MACRO_PATH% "lif_path=%LIF_PATH%,output_dir=%OUTPUT_DIR%"

echo.
if %errorlevel% equ 0 (
    echo ✅ Macro executed successfully!
    echo Results saved to: %OUTPUT_DIR%
) else (
    echo ❌ Macro failed with error code: %errorlevel%
)

echo.
pause