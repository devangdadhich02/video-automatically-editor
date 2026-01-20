from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from openai import OpenAI
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from gtts import gTTS
import re
import shutil
from pydub import AudioSegment
import numpy as np

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_EXCEL_EXTENSIONS = {'xlsx', 'xls', 'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set your OpenAI API key here or use environment variable
# Initialize client lazily to avoid errors if API key is not set
def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def secure_filename(filename):
    """Simple secure filename function"""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename

class GenerateVideosRequest(BaseModel):
    video_filename: str
    transcript: str
    names_to_replace: List[str]
    excel_names: List[str]
    word_timestamps: List[dict] = []

@app.post("/api/upload-video")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    filename = secure_filename(video.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save uploaded file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Extract audio and transcribe
    try:
        video_clip = VideoFileClip(filepath)
        
        # Check if video has audio
        if video_clip.audio is None:
            video_clip.close()
            raise HTTPException(status_code=400, detail="Video file does not contain audio track")
        
        audio_path = os.path.join(UPLOAD_FOLDER, f"{filename}_audio.wav")
        video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video_clip.close()
        
        # Transcribe using OpenAI Whisper with word-level timestamps
        try:
            client = get_openai_client()
            with open(audio_path, 'rb') as audio_file:
                # Try to get word-level timestamps
                try:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )
                    transcript_text = transcript.text
                    # Extract word timestamps - convert to dict format
                    word_timestamps = []
                    if hasattr(transcript, 'words') and transcript.words:
                        for word_obj in transcript.words:
                            word_timestamps.append({
                                'word': getattr(word_obj, 'word', ''),
                                'start': getattr(word_obj, 'start', 0),
                                'end': getattr(word_obj, 'end', 0)
                            })
                    elif hasattr(transcript, 'segments') and transcript.segments:
                        for segment in transcript.segments:
                            if hasattr(segment, 'words') and segment.words:
                                for word_obj in segment.words:
                                    word_timestamps.append({
                                        'word': getattr(word_obj, 'word', ''),
                                        'start': getattr(word_obj, 'start', 0),
                                        'end': getattr(word_obj, 'end', 0)
                                    })
                    print(f"Extracted {len(word_timestamps)} word timestamps for precise replacement")
                except Exception as e:
                    # Fallback to simple transcription if word timestamps not supported
                    print(f"Word timestamps not available: {e}, using simple transcription")
                    with open(audio_path, 'rb') as audio_file_fallback:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file_fallback
                        )
                    transcript_text = transcript.text if hasattr(transcript, 'text') else str(transcript)
                    word_timestamps = []
        except ValueError as ve:
            # Clean up audio file before raising error
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(ve)}")
        except Exception as e:
            # Clean up audio file before raising error
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
        
        # Extract potential names - improved for Hindi and English
        # Method 1: English capitalized words
        words = transcript_text.split()
        potential_names = []
        
        # Add English capitalized words
        for word in words:
            if len(word) > 2:
                # Check if it's an English capitalized word
                if word[0].isupper() and word.isalpha():
                    potential_names.append(word)
                # Check for Hindi words (Devnagari script) - names are usually 3+ characters
                elif any('\u0900' <= char <= '\u097F' for char in word) and len(word) >= 3:
                    # It's a Hindi word, add it as potential name
                    potential_names.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in potential_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        potential_names = unique_names
        
        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            'success': True,
            'filename': filename,
            'transcript': transcript_text,
            'potential_names': list(set(potential_names)),
            'word_timestamps': word_timestamps  # Include timestamps for precise replacement
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Video processing error: {error_trace}")  # Log to console for debugging
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

@app.post("/api/upload-excel")
async def upload_excel(excel: UploadFile = File(...)):
    if not excel.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(excel.filename, ALLOWED_EXCEL_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    filename = secure_filename(excel.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save uploaded file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(excel.file, buffer)
    
    try:
        # Read Excel file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Get names from first column
        names = df.iloc[:, 0].dropna().astype(str).tolist()
        
        return {
            'success': True,
            'names': names,
            'count': len(names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def replace_audio_segments(original_audio_path, word_timestamps, names_to_replace, replacement_name):
    """
    Replace ONLY name segments in audio - simple and precise approach
    Keeps original audio intact, only replaces exact name matches
    """
    if not word_timestamps:
        print("No word timestamps available, cannot do precise replacement")
        return None
    
    # Load original audio
    original_audio = AudioSegment.from_wav(original_audio_path)
    audio_segments = []
    
    # Sort timestamps by start time
    sorted_timestamps = sorted(word_timestamps, key=lambda x: x.get('start', 0))
    
    current_pos = 0  # Current position in audio (in milliseconds)
    replacement_count = 0
    
    for word_info in sorted_timestamps:
        word = word_info.get('word', '').strip()
        start_time_ms = int(word_info.get('start', 0) * 1000)  # Convert to milliseconds
        end_time_ms = int(word_info.get('end', 0) * 1000)
        
        # Check if this word matches any name to replace (exact match preferred)
        should_replace = False
        matched_name = None
        
        for name in names_to_replace:
            # Exact match (case-insensitive)
            if word.lower().strip() == name.lower().strip():
                should_replace = True
                matched_name = name
                break
            # For Hindi/Unicode: check if word contains name or vice versa
            if word.strip() == name.strip() or word.strip() in name.strip() or name.strip() in word.strip():
                if len(word.strip()) >= 2:  # At least 2 characters
                    should_replace = True
                    matched_name = name
                    break
        
        # Add audio BEFORE this word (preserve original)
        if start_time_ms > current_pos:
            audio_segments.append(original_audio[current_pos:start_time_ms])
        
        if should_replace:
            replacement_count += 1
            print(f"Replacing '{word}' with '{replacement_name}' at {start_time_ms}ms-{end_time_ms}ms")
            
            # Generate TTS for replacement name ONLY
            try:
                client = get_openai_client()
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=replacement_name
                )
                replacement_audio_path = os.path.join(UPLOAD_FOLDER, f"temp_replacement_{replacement_name}_{replacement_count}.mp3")
                response.stream_to_file(replacement_audio_path)
                
                # Load replacement audio
                replacement_audio = AudioSegment.from_mp3(replacement_audio_path)
                original_duration = end_time_ms - start_time_ms
                replacement_duration = len(replacement_audio)
                
                # Match duration EXACTLY to preserve lip sync
                if replacement_duration > original_duration:
                    # Too long - speed up to match
                    speed_factor = replacement_duration / original_duration
                    replacement_audio = replacement_audio.speedup(playback_speed=speed_factor, chunk_size=150)
                elif replacement_duration < original_duration:
                    # Too short - add silence at end to match
                    silence_needed = original_duration - replacement_duration
                    silence = AudioSegment.silent(duration=silence_needed)
                    replacement_audio = replacement_audio + silence
                
                # Ensure exact duration match
                replacement_audio = replacement_audio[:original_duration]
                audio_segments.append(replacement_audio)
                
                # Cleanup
                if os.path.exists(replacement_audio_path):
                    os.remove(replacement_audio_path)
            except Exception as e:
                print(f"TTS failed for '{replacement_name}', keeping original: {e}")
                # Keep original if TTS fails
                audio_segments.append(original_audio[start_time_ms:end_time_ms])
        else:
            # Keep original audio for this word
            audio_segments.append(original_audio[start_time_ms:end_time_ms])
        
        current_pos = end_time_ms
    
    # Add remaining audio after last word
    if current_pos < len(original_audio):
        audio_segments.append(original_audio[current_pos:])
    
    # Concatenate all segments
    if audio_segments:
        final_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            final_audio = final_audio + segment
        print(f"Audio replacement complete. Replaced {replacement_count} instances.")
        return final_audio
    else:
        return original_audio

@app.post("/api/generate-videos")
async def generate_videos(request: GenerateVideosRequest):
    video_filename = request.video_filename
    original_transcript = request.transcript
    names_to_replace = request.names_to_replace
    excel_names = request.excel_names
    word_timestamps = request.word_timestamps or []
    
    if not all([video_filename, original_transcript, names_to_replace, excel_names]):
        raise HTTPException(status_code=400, detail="Missing required data")
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    generated_videos = []
    
    try:
        # Load original video
        original_video = VideoFileClip(video_path)
        original_audio_path = os.path.join(UPLOAD_FOLDER, f"{video_filename}_original_audio.wav")
        
        # Extract and save original audio
        original_video.audio.write_audiofile(original_audio_path, verbose=False, logger=None)
        
        for excel_name in excel_names:
            modified_audio_path = None  # Initialize variable
            audio_path = None  # Initialize variable
            
            # Replace only name segments in audio
            if word_timestamps:
                # Use precise segment replacement
                modified_audio = replace_audio_segments(
                    original_audio_path, 
                    word_timestamps, 
                    names_to_replace, 
                    excel_name
                )
                
                # Save modified audio
                modified_audio_path = os.path.join(UPLOAD_FOLDER, f"temp_audio_{excel_name}.wav")
                modified_audio.export(modified_audio_path, format="wav")
                
                # Load as MoviePy audio clip
                new_audio = AudioFileClip(modified_audio_path)
            else:
                # Fallback: Generate full audio if timestamps not available
                print("Warning: Word timestamps not available, using full audio replacement")
                audio_path = os.path.join(UPLOAD_FOLDER, f"temp_audio_{excel_name}.mp3")
                try:
                    client = get_openai_client()
                    response = client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=original_transcript.replace(names_to_replace[0], excel_name) if names_to_replace else original_transcript
                    )
                    response.stream_to_file(audio_path)
                    new_audio = AudioFileClip(audio_path)
                except Exception as e:
                    print(f"TTS failed: {e}")
                    # Keep original audio
                    new_audio = original_video.audio
            
            # Set audio to video with same duration
            final_video = original_video.set_audio(new_audio)
            
            # Save output video with high quality
            output_filename = f"{excel_name}_{video_filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            final_video.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                bitrate='5000k',  # High quality video
                audio_bitrate='192k',  # High quality audio
                preset='medium',  # Balance between speed and quality
                verbose=False, 
                logger=None
            )
            
            generated_videos.append(output_filename)
            
            # Cleanup
            new_audio.close()
            final_video.close()
            
            # Clean up temporary audio files
            if modified_audio_path and os.path.exists(modified_audio_path):
                os.remove(modified_audio_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        
        original_video.close()
        if os.path.exists(original_audio_path):
            os.remove(original_audio_path)
        
        return {
            'success': True,
            'videos': generated_videos,
            'count': len(generated_videos)
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Video generation error: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Video generation error: {str(e)}")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type='application/octet-stream')
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/style.css")
async def get_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def get_js():
    return FileResponse("script.js", media_type="application/javascript")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
