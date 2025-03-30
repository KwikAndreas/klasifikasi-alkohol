@echo off
echo Setting up Python virtual environment for Wine Quality Classifier...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.6 or later.
    exit /b 1
)

:: Check if venv module is available
python -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo The venv module is not available. Please install Python 3.6 or later.
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo.
echo To activate the virtual environment, run:
echo venv\Scripts\activate
echo.
echo To deactivate the virtual environment, run:
echo deactivate
echo.
