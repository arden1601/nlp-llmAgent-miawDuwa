@echo off
echo 🤖 Discord LLM Summarization Bot - Setup Script
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv .venv

if %errorlevel% equ 0 (
    echo ✅ Virtual environment created
) else (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo ✅ Dependencies installed
) else (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file...
    copy .env.example .env
    echo ✅ .env file created
    echo.
    echo ⚠️  IMPORTANT: Please edit .env and add your:
    echo    - DISCORD_TOKEN
    echo    - OPENAI_API_KEY
) else (
    echo ℹ️  .env file already exists
)

REM Create database directory
if not exist database mkdir database
echo ✅ Database directory created

echo.
echo ================================================
echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit .env and add your Discord token and OpenAI API key
echo 2. Activate the virtual environment: venv\Scripts\activate.bat
echo 3. Run the bot: python bot.py
echo.
echo For more information, see README.md
echo.
pause
