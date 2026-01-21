from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from openai import OpenAI
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
from moviepy.video.fx.all import speedx
import numpy as np
import librosa
import soundfile as sf
import io
import tempfile
from gtts import gTTS
import re
import shutil
from pydub import AudioSegment
from pydub.effects import normalize

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
    language: str = "hi"  # Default to Hindi

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
        
        # Initialize variables for language detection
        detected_language = "hi"  # Default to Hindi (mostly Hindi videos)
        word_timestamps = []  # Initialize word timestamps
        
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
                    
                    # Detect language from transcript
                    if hasattr(transcript, 'language'):
                        detected_language = transcript.language
                    else:
                        # Detect language from text content
                        hindi_chars = sum(1 for char in transcript_text if '\u0900' <= char <= '\u097F')
                        total_chars = len([c for c in transcript_text if c.isalnum()])
                        if total_chars > 0 and (hindi_chars / total_chars) > 0.3:  # More than 30% Hindi
                            detected_language = "hi"
                    
                    print(f"Detected language: {detected_language}")
                    # Extract word timestamps - convert to dict format
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
                    # Detect language from fallback transcript
                    hindi_chars = sum(1 for char in transcript_text if '\u0900' <= char <= '\u097F')
                    total_chars = len([c for c in transcript_text if c.isalnum()])
                    if total_chars > 0 and (hindi_chars / total_chars) > 0.3:
                        detected_language = "hi"
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
            'word_timestamps': word_timestamps,  # Include timestamps for precise replacement
            'language': detected_language  # Include detected language for proper TTS
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

def analyze_audio_characteristics(audio_segment):
    """ADVANCED audio analysis - detects pitch, tone, language characteristics"""
    try:
        # Convert to numpy array for analysis
        samples = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Calculate basic characteristics
        rms = audio_segment.rms
        db = audio_segment.dBFS
        max_possible_amplitude = audio_segment.max_possible_amplitude
        max_amplitude = max(abs(samples)) if len(samples) > 0 else 0
        
        # Calculate dynamic range
        if len(samples) > 0:
            # Remove silence for better analysis
            non_silent = samples[np.abs(samples) > max_amplitude * 0.1]
            if len(non_silent) > 0:
                dynamic_range = np.percentile(np.abs(non_silent), 95) - np.percentile(np.abs(non_silent), 5)
            else:
                dynamic_range = max_amplitude
        else:
            dynamic_range = 0
        
        # ADVANCED: Estimate pitch/tone (fundamental frequency) - CRITICAL for matching
        pitch_estimate = None
        if len(samples) > 500:
            try:
                frame_rate = audio_segment.frame_rate
                window_size = min(8192, len(samples))
                window = samples[:window_size].astype(np.float32)
                
                if max(abs(window)) > 0:
                    window = window / max(abs(window))
                    hann = np.hanning(len(window))
                    window = window * hann
                
                autocorr = np.correlate(window, window, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                threshold = max(autocorr) * 0.15
                min_period = int(frame_rate / 400)
                max_period = int(frame_rate / 80)
                
                best_pitch = None
                best_peak = 0
                
                for i in range(max(10, min_period), min(len(autocorr) - 1, max_period)):
                    if autocorr[i] > threshold:
                        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                            fundamental_period = i
                            if fundamental_period > 0:
                                pitch_candidate = frame_rate / fundamental_period
                                if 80 <= pitch_candidate <= 400:
                                    if autocorr[i] > best_peak:
                                        best_peak = autocorr[i]
                                        best_pitch = pitch_candidate
                
                if best_pitch:
                    pitch_estimate = best_pitch
            except Exception as e:
                print(f"Pitch detection error: {e}")
                pass
        
        return {
            'rms': rms,
            'db': db,
            'max_amplitude': max_amplitude,
            'dynamic_range': dynamic_range,
            'samples': samples,
            'pitch': pitch_estimate,
            'frame_rate': audio_segment.frame_rate
        }
    except Exception as e:
        print(f"Audio analysis error: {e}")
        return None

def match_voice_characteristics(replacement_audio, original_segment):
    """Advanced voice matching - preserves natural sound while matching characteristics"""
    try:
        orig_chars = analyze_audio_characteristics(original_segment)
        repl_chars = analyze_audio_characteristics(replacement_audio)
        
        if not orig_chars or not repl_chars:
            # Still normalize volume
            return normalize(replacement_audio)
        
        # 1. Match volume (more conservative to avoid distortion)
        if orig_chars['db'] != float('-inf') and repl_chars['db'] != float('-inf'):
            volume_diff = orig_chars['db'] - repl_chars['db']
            # Limit to ±6dB to avoid making voice too heavy or thin
            volume_diff = max(-6, min(6, volume_diff))
            replacement_audio = replacement_audio.apply_gain(volume_diff)
            print(f"   🔊 Volume adjusted: {volume_diff:.1f}dB")
        
        # 2. Match pitch using librosa (preserves natural voice quality)
        if orig_chars.get('pitch') and repl_chars.get('pitch'):
            pitch_ratio = orig_chars['pitch'] / repl_chars['pitch']
            # Only adjust if difference is significant (more than 5%)
            if abs(pitch_ratio - 1.0) > 0.05 and 0.85 < pitch_ratio < 1.15:
                try:
                    # Convert to numpy for librosa
                    samples = np.array(replacement_audio.get_array_of_samples())
                    if replacement_audio.channels == 2:
                        samples = samples.reshape((-1, 2)).mean(axis=1)
                    
                    # Normalize
                    if len(samples) > 0 and max(abs(samples)) > 0:
                        samples = samples.astype(np.float32) / max(abs(samples))
                    
                    # Use librosa for pitch-preserving time stretch (but we want pitch shift)
                    # Instead, use a subtle frame rate adjustment with proper resampling
                    frame_rate = replacement_audio.frame_rate
                    new_frame_rate = int(frame_rate * pitch_ratio)
                    
                    # Keep within reasonable bounds
                    if 16000 <= new_frame_rate <= 48000:
                        # Use librosa for high-quality resampling
                        import librosa
                        import soundfile as sf
                        import tempfile
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            replacement_audio.export(tmp.name, format="wav")
                            temp_path = tmp.name
                        
                        # Load with librosa and resample
                        y, sr = librosa.load(temp_path, sr=frame_rate)
                        y_resampled = librosa.resample(y, orig_sr=frame_rate, target_sr=new_frame_rate)
                        
                        # Save resampled audio
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                            sf.write(tmp2.name, y_resampled, new_frame_rate)
                            temp_path2 = tmp2.name
                        
                        # Load back and convert to original frame rate
                        replacement_audio = AudioSegment.from_wav(temp_path2)
                        replacement_audio = replacement_audio.set_frame_rate(frame_rate)
                        
                        # Cleanup
                        os.unlink(temp_path)
                        os.unlink(temp_path2)
                        
                        print(f"   🎵 Pitch adjusted: {pitch_ratio:.2f}x (preserving natural voice)")
                except Exception as e:
                    print(f"   ⚠️ Pitch adjustment failed (using original): {e}")
        
        # 3. Normalize to prevent clipping
        return normalize(replacement_audio)
    except Exception as e:
        print(f"Voice matching error: {e}")
        return normalize(replacement_audio)

def detect_language_from_text(text):
    if not text:
        return "en"
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([c for c in text if c.isalnum()])
    if total_chars > 0 and (hindi_chars / total_chars) > 0.2:
        return "hi"
    return "en"

def convert_to_hindi_transliteration(name):
    if not name:
        return name
    if detect_language_from_text(name) == "hi":
        return name
    
    transliteration_map = {
        'Sunil': 'सुनील', 'Suresh': 'सुरेश', 'Raja': 'राजा', 'Raj': 'राज',
        'Amit': 'अमित', 'Anil': 'अनिल', 'Ravi': 'रवि', 'Kumar': 'कुमार',
        'Sharma': 'शर्मा', 'Singh': 'सिंह', 'Patel': 'पटेल', 'Gupta': 'गुप्ता',
        'Mehta': 'मेहता', 'Joshi': 'जोशी', 'Reddy': 'रेड्डी', 'Rao': 'राव',
        'Iyer': 'अय्यर', 'Nair': 'नायर', 'Pillai': 'पिल्लई', 'Menon': 'मेनन',
        'Vikram': 'विक्रम', 'Rahul': 'राहुल', 'Priya': 'प्रिया', 'Deepak': 'दीपक'
    }
    
    if name in transliteration_map:
        return transliteration_map[name]
    
    for eng_name, hindi_name in transliteration_map.items():
        if name.lower() == eng_name.lower():
            return hindi_name
    
    return name

def process_personalized_video(video_clip, original_audio_path, word_timestamps, names_to_replace, replacement_name, language="hi"):
    """
    SIMPLE & DIRECT: Replace audio segments and stretch video to match
    """
    print(f"\n🔍 DEBUG: Starting video processing")
    print(f"   - Replacement name: {replacement_name}")
    print(f"   - Names to replace: {names_to_replace}")
    print(f"   - Word timestamps: {len(word_timestamps) if word_timestamps else 0}")
    
    if not word_timestamps or len(word_timestamps) == 0:
        print("⚠️ No timestamps - using full audio replacement")
        return None  # Signal to use fallback
    
    # Load original audio for reference
    original_audio = AudioSegment.from_wav(original_audio_path)
    
    # Find ALL matching words
    matches = []
    for word_info in word_timestamps:
        word = word_info.get('word', '').strip()
        for name in names_to_replace:
            if word.lower() == name.lower() or word.lower() in name.lower() or name.lower() in word.lower():
                matches.append({
                    'word': word,
                    'start': word_info.get('start', 0),
                    'end': word_info.get('end', 0)
                })
                print(f"✅ FOUND MATCH: '{word}' at {word_info.get('start', 0):.2f}s-{word_info.get('end', 0):.2f}s")
                break
        
    if not matches:
        print("❌ NO MATCHES FOUND - returning None for fallback")
        return None
    
    print(f"📊 Total matches found: {len(matches)}")
    
    # Sort matches by time
    matches.sort(key=lambda x: x['start'])
    
    # Build video segments
    final_clips = []
    last_end = 0
    
    # Generate TTS for replacement name once
    # Use "shimmer" for Hindi - more natural Hindi accent, "alloy" for English
    tts_voice = "shimmer" if language == "hi" else "alloy"
    tts_input = convert_to_hindi_transliteration(replacement_name) if language == "hi" else replacement_name
    
    print(f"🔊 Generating TTS for '{replacement_name}' (input: '{tts_input}') with voice: {tts_voice}")
    try:
        client = get_openai_client()
        # Use tts-1-hd for better quality and more natural sound
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=tts_voice,
            input=tts_input
        )
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            response.stream_to_file(temp_audio.name)
            tts_audio_path = temp_audio.name
        
        replacement_audio_seg = AudioSegment.from_mp3(tts_audio_path)
        replacement_duration_ms = len(replacement_audio_seg)
        print(f"✅ TTS generated: {replacement_duration_ms}ms")
        
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return None
    
    # Process each match
    for match in matches:
        start_time = match['start']
        end_time = match['end']
        original_duration = end_time - start_time
        
        # Add segment before this match
        if start_time > last_end:
            before_clip = video_clip.subclip(last_end, start_time)
            final_clips.append(before_clip)
        
        # Process the replacement
        print(f"🎬 Processing replacement at {start_time:.2f}s-{end_time:.2f}s")
        
        # Get original audio segment for voice matching
        original_seg_audio = original_audio[int(start_time*1000):int(end_time*1000)]
        matched_audio = match_voice_characteristics(replacement_audio_seg, original_seg_audio)
        
        # Save matched audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            matched_audio.export(temp_wav.name, format="wav")
            matched_audio_path = temp_wav.name
        
        new_audio_clip = AudioFileClip(matched_audio_path)
        tts_duration = new_audio_clip.duration
        
        # Get video segment
        name_video_seg = video_clip.subclip(start_time, end_time)
        
        # CRITICAL: Adjust video duration to match audio EXACTLY
        if tts_duration > original_duration:
            # Slow down video to match longer audio
            speed_factor = original_duration / tts_duration  # Less than 1 = slower
            print(f"   🐌 Slowing down video: {original_duration:.2f}s -> {tts_duration:.2f}s (speed: {speed_factor:.2f}x)")
            try:
                name_video_seg = name_video_seg.fx(speedx, speed_factor)
                # Ensure duration matches exactly
                if abs(name_video_seg.duration - tts_duration) > 0.1:
                    # If still not matching, adjust by cutting or extending
                    if name_video_seg.duration > tts_duration:
                        name_video_seg = name_video_seg.subclip(0, tts_duration)
                    print(f"   ✅ Video duration adjusted to: {name_video_seg.duration:.2f}s")
            except Exception as e:
                print(f"   ⚠️ Speed adjustment failed: {e}, using original segment")
                name_video_seg = name_video_seg.subclip(0, min(tts_duration, name_video_seg.duration))
        elif tts_duration < original_duration:
            # Cut video to match shorter audio
            print(f"   ✂️ Cutting video: {original_duration:.2f}s -> {tts_duration:.2f}s")
            name_video_seg = name_video_seg.subclip(0, tts_duration)
        else:
            # Perfect match - no adjustment needed
            print(f"   ✅ Perfect duration match: {tts_duration:.2f}s")
        
        # CRITICAL: Set new audio - this replaces the original audio completely
        print(f"   🔊 Setting new audio (duration: {new_audio_clip.duration:.2f}s)")
        name_video_seg = name_video_seg.set_audio(new_audio_clip)
        
        # Verify audio is set
        if name_video_seg.audio is None:
            print(f"   ⚠️ WARNING: Audio not set properly!")
        else:
            print(f"   ✅ Audio set successfully (audio duration: {name_video_seg.audio.duration:.2f}s)")
        
        final_clips.append(name_video_seg)
        
        last_end = end_time
    
    # Add final segment
    if last_end < video_clip.duration:
        final_clips.append(video_clip.subclip(last_end, video_clip.duration))
        print(f"   📹 Added final segment: {last_end:.2f}s -> {video_clip.duration:.2f}s")
    
    # Cleanup
    if os.path.exists(tts_audio_path):
        os.unlink(tts_audio_path)
    
    # Concatenate all segments
    if final_clips:
        print(f"✨ Stitching {len(final_clips)} segments...")
        try:
            # Verify all clips have audio
            for i, clip in enumerate(final_clips):
                if clip.audio is None:
                    print(f"   ⚠️ Clip {i} has no audio!")
                else:
                    print(f"   ✅ Clip {i}: video={clip.duration:.2f}s, audio={clip.audio.duration:.2f}s")
            
            final_video = concatenate_videoclips(final_clips, method="compose")
            
            # Verify final video has audio
            if final_video.audio is None:
                print(f"❌ WARNING: Final video has no audio!")
            else:
                print(f"✅ Final video: duration={final_video.duration:.2f}s, audio={final_video.audio.duration:.2f}s")
            
            print(f"✅ Video reconstruction complete!")
            return final_video
        except Exception as e:
            print(f"❌ Concatenation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print("❌ No clips to concatenate")
    return None

@app.post("/api/generate-videos")
async def generate_videos(request: GenerateVideosRequest):
    video_filename = request.video_filename
    original_transcript = request.transcript
    names_to_replace = request.names_to_replace
    excel_names = request.excel_names
    word_timestamps = request.word_timestamps or []
    language = request.language or "hi"
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    generated_videos = []
    
    try:
        original_video = VideoFileClip(video_path)
        original_audio_path = os.path.join(UPLOAD_FOLDER, f"{video_filename}_original_audio.wav")
        original_video.audio.write_audiofile(original_audio_path, verbose=False, logger=None)
        
        for excel_name in excel_names:
            print(f"🚀 Processing: {excel_name}")
            print(f"🔍 DEBUG: word_timestamps count: {len(word_timestamps) if word_timestamps else 0}")
            print(f"🔍 DEBUG: names_to_replace: {names_to_replace}")
            print(f"🔍 DEBUG: language: {language}")
            
            # Try MAST approach first
            final_video = None
            if word_timestamps and len(word_timestamps) > 0:
                print(f"✅ Attempting MAST approach with {len(word_timestamps)} timestamps")
                final_video = process_personalized_video(
                    original_video,
                    original_audio_path, 
                    word_timestamps, 
                    names_to_replace, 
                    excel_name,
                    language=language
                )
            
            # Fallback to full audio replacement if MAST failed
            if final_video is None:
                print("⚠️ MAST approach failed/not available, using FULL AUDIO REPLACEMENT")
                try:
                    client = get_openai_client()
                    modified_transcript = original_transcript
                    if names_to_replace and len(names_to_replace) > 0:
                        # Replace all occurrences of the first name
                        for name in names_to_replace:
                            modified_transcript = modified_transcript.replace(name, excel_name)
                    
                    print(f"🔊 Generating full TTS audio...")
                    audio_path = os.path.join(UPLOAD_FOLDER, f"temp_full_audio_{excel_name}.mp3")
                    response = client.audio.speech.create(
                        model="tts-1-hd",
                        voice="shimmer" if language == "hi" else "alloy",
                        input=modified_transcript
                    )
                    response.stream_to_file(audio_path)
                    new_audio = AudioFileClip(audio_path)
                    
                    # Match audio duration to video - simple approach
                    if abs(new_audio.duration - original_video.duration) > 0.1:
                        print(f"   ⏱️ Audio duration mismatch: {new_audio.duration:.2f}s vs video {original_video.duration:.2f}s")
                        if new_audio.duration > original_video.duration:
                            new_audio = new_audio.subclip(0, original_video.duration)
                        # If shorter, video will just be shorter - that's okay
                    
                    final_video = original_video.set_audio(new_audio)
                    print(f"✅ Full audio replacement complete")
                except Exception as e:
                    print(f"❌ Full audio replacement failed: {e}")
                    import traceback
                    traceback.print_exc()
                    final_video = original_video
            
            # Ensure we have a video to save
            if final_video is None:
                print("❌ ERROR: No video to save, using original")
                final_video = original_video
            
            output_filename = f"{excel_name}_{video_filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Verify final video before saving
            print(f"📹 Final video info:")
            print(f"   - Duration: {final_video.duration:.2f}s (original: {original_video.duration:.2f}s)")
            if final_video.audio:
                print(f"   - Audio duration: {final_video.audio.duration:.2f}s")
            else:
                print(f"   - ⚠️ WARNING: No audio in final video!")
            
            print(f"💾 Saving video to: {output_path}")
            final_video.write_videofile(
                output_path, 
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(UPLOAD_FOLDER, f"temp_final_audio_{excel_name}.m4a"),
                remove_temp=True,
                fps=original_video.fps or 24,
                verbose=False, 
                logger=None
            )
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"✅ Video saved successfully! Size: {file_size:.2f} MB")
            else:
                print(f"❌ ERROR: Video file not created!")
            
            generated_videos.append(output_filename)
            
            # Cleanup video clips to free memory
            final_video.close()
            
        return {"success": True, "videos": generated_videos, "count": len(generated_videos)}
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

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
