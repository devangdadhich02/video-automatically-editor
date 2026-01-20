# AI Video Editor - Name Replacement Tool

A single-page AI-based video editor that allows users to upload a video and an Excel file, then automatically replace names in the video with names from the Excel file.

## Features

- 🎬 Upload video files (MP4, AVI, MOV, MKV, WebM)
- 📊 Upload Excel/CSV files with names
- 🎤 Automatic transcription using OpenAI Whisper API
- 🔄 Name replacement in video audio
- 📥 Download multiple generated videos

## Setup Instructions

### Option 1: Using Docker (Recommended)

#### 1. Build and Run with Docker Compose

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
# OR
set OPENAI_API_KEY=your-api-key-here  # Windows CMD
# OR
$env:OPENAI_API_KEY="your-api-key-here"  # Windows PowerShell

# Build and run
docker-compose up --build
```

#### 2. Or Build and Run Docker Manually

```bash
# Build the image
docker build -t ai-video-editor .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-api-key-here" \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  --name ai-video-editor \
  ai-video-editor
```

### Option 2: Local Development

#### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Set OpenAI API Key

Set your OpenAI API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### 3. Run the Application

**Using uvicorn directly:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Or using the run script:**
```bash
python run.py
```

The server will start on `http://localhost:8000`

### 4. Open in Browser

Navigate to `http://localhost:8000` in your web browser.

## How to Use

1. **Upload Video**: Click on the video upload area and select your video file. The system will automatically extract the audio and transcribe it.

2. **Upload Excel**: Upload an Excel or CSV file containing names in the first column. These names will be used to replace names in the video.

3. **Select Names**: Review the transcript and select which names from the video should be replaced. For each selected name, choose which Excel name should replace it.

4. **Generate Videos**: Click "Generate Videos" to create new videos with replaced names. One video will be generated for each name in your Excel file.

5. **Download**: Download the generated videos from the download section.

## File Structure

```
.
├── app.py              # FastAPI backend server
├── run.py              # Uvicorn runner script
├── index.html          # Frontend HTML
├── style.css           # Styling
├── script.js           # Frontend JavaScript
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore       # Docker ignore file
├── uploads/            # Uploaded files (created automatically)
└── outputs/            # Generated videos (created automatically)
```

## Docker Commands

```bash
# Start the container
docker-compose up

# Start in background
docker-compose up -d

# Stop the container
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up --build
```

## Notes

- The application uses OpenAI's Whisper API for transcription
- Video processing may take time depending on video length
- Make sure you have sufficient disk space for generated videos
- The first column of your Excel file should contain the replacement names
- Docker containers include FFmpeg automatically
- For production, use environment variables or secrets management for API keys

## Requirements

- Python 3.8+
- OpenAI API key
- FFmpeg (required by MoviePy)

### Installing FFmpeg

**Windows:**
- Download from https://ffmpeg.org/download.html
- Extract and add to PATH, or install via chocolatey: `choco install ffmpeg`

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

## Troubleshooting

- If video processing fails, ensure FFmpeg is installed
- Check that your OpenAI API key is set correctly
- Ensure uploaded files are in supported formats
- Check console/terminal for error messages

