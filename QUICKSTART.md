# Quick Start Guide

## 🐳 Using Docker (Easiest Method)

### Step 1: Create .env file
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your-actual-api-key-here
```

### Step 2: Run with Docker Compose
```bash
docker-compose up --build
```

### Step 3: Open Browser
Navigate to: `http://localhost:5508`

That's it! 🎉

---

## 💻 Local Development

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set API Key
**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Step 3: Run Server
```bash
# Option 1: Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 5508 --reload

# Option 2: Using run script
python run.py
```

### Step 4: Open Browser
Navigate to: `http://localhost:5508`

---

## 📝 Usage

1. **Upload Video** - Click to upload your video file
2. **Upload Excel** - Upload Excel/CSV with names in first column
3. **Select Names** - Check which names from transcript to replace
4. **Generate** - Click "Generate Videos" button
5. **Download** - Download your personalized videos!

---

## 🛠️ Troubleshooting

### Docker Issues
- Make sure Docker is running
- Check if port 5508 is available
- View logs: `docker-compose logs -f`

### API Key Issues
- Verify your OpenAI API key is correct
- Check if you have credits in your OpenAI account
- Ensure the key has access to Whisper and TTS APIs

### Video Processing Issues
- Ensure FFmpeg is installed (Docker includes it automatically)
- Check available disk space
- Try with a smaller video first

---

## 📦 Docker Commands Cheat Sheet

```bash
# Start
docker-compose up

# Start in background
docker-compose up -d

# Stop
docker-compose down

# Rebuild
docker-compose up --build

# View logs
docker-compose logs -f

# Remove everything
docker-compose down -v
```

