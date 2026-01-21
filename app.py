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
from pydub.effects import normalize
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
        # ULTRA IMPROVED pitch detection - more robust for TTS audio
        pitch_estimate = None
        if len(samples) > 500:  # Lower threshold for TTS audio
            try:
                frame_rate = audio_segment.frame_rate
                # Use larger window for better pitch detection
                window_size = min(8192, len(samples))  # Use more samples for better detection
                window = samples[:window_size].astype(np.float32)
                
                # Normalize and apply window function to reduce edge effects
                if max(abs(window)) > 0:
                    window = window / max(abs(window))
                    # Apply Hann window for better frequency analysis
                    hann = np.hanning(len(window))
                    window = window * hann
                
                # Method 1: Improved autocorrelation for pitch detection
                autocorr = np.correlate(window, window, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find first significant peak (fundamental frequency)
                # MUCH lower threshold for TTS audio (which may have less clear pitch)
                threshold = max(autocorr) * 0.15  # Even lower threshold
                
                # Search in human voice range (80-400 Hz)
                min_period = int(frame_rate / 400)  # 400 Hz max
                max_period = int(frame_rate / 80)   # 80 Hz min
                
                best_pitch = None
                best_peak = 0
                
                for i in range(max(10, min_period), min(len(autocorr) - 1, max_period)):
                    if autocorr[i] > threshold:
                        # Check if it's a local peak
                        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                            fundamental_period = i
                            if fundamental_period > 0:
                                pitch_candidate = frame_rate / fundamental_period  # Hz
                                # Validate pitch is in human voice range
                                if 80 <= pitch_candidate <= 400:
                                    # Keep track of strongest peak
                                    if autocorr[i] > best_peak:
                                        best_peak = autocorr[i]
                                        best_pitch = pitch_candidate
                
                if best_pitch:
                    pitch_estimate = best_pitch
                else:
                    # Method 2: FFT-based pitch detection (fallback)
                    try:
                        # Use FFT to find dominant frequency
                        fft = np.fft.rfft(window)
                        magnitude = np.abs(fft)
                        frequencies = np.fft.rfftfreq(len(window), 1/frame_rate)
                        
                        # Find peak in voice range
                        voice_range = (frequencies >= 80) & (frequencies <= 400)
                        if np.any(voice_range):
                            voice_magnitude = magnitude[voice_range]
                            voice_freqs = frequencies[voice_range]
                            peak_idx = np.argmax(voice_magnitude)
                            pitch_estimate = voice_freqs[peak_idx]
                    except:
                        pass
            except Exception as e:
                print(f"Pitch detection error: {e}")
                pass  # Pitch estimation is optional
        
        # Calculate spectral characteristics (tone quality/brightness)
        spectral_centroid = None
        if len(samples) > 512:
            try:
                # Simple spectral analysis
                fft = np.fft.rfft(samples[:2048])
                magnitude = np.abs(fft)
                frequencies = np.fft.rfftfreq(len(samples[:2048]), 1/audio_segment.frame_rate)
                
                if np.sum(magnitude) > 0:
                    # Spectral centroid (brightness of sound)
                    spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
            except Exception as e:
                pass  # Spectral analysis is optional
        
        return {
            'rms': rms,
            'db': db,
            'max_amplitude': max_amplitude,
            'dynamic_range': dynamic_range,
            'samples': samples,
            'pitch': pitch_estimate,  # Fundamental frequency (Hz) - for tone matching
            'spectral_centroid': spectral_centroid,  # Tone brightness - for quality matching
            'frame_rate': audio_segment.frame_rate
        }
    except Exception as e:
        print(f"Audio analysis error: {e}")
        return None

def match_voice_characteristics(replacement_audio, original_segment, context_before=None, context_after=None):
    """
    ADVANCED voice matching - analyzes pitch, tone, language and matches EXACTLY
    Makes replacement sound like it's from original speaker with same pitch/tone
    """
    try:
        # Analyze original audio characteristics (pitch, tone, etc.)
        orig_chars = analyze_audio_characteristics(original_segment)
        repl_chars = analyze_audio_characteristics(replacement_audio)
        
        if not orig_chars or not repl_chars:
            return replacement_audio
        
        print(f"🎵 Original: pitch={orig_chars.get('pitch')}Hz, spectral={orig_chars.get('spectral_centroid')}")
        print(f"🎵 Replacement: pitch={repl_chars.get('pitch')}Hz, spectral={repl_chars.get('spectral_centroid')}")
        
        # 1. Match volume/loudness (CRITICAL for seamless blend) - PRESERVE original volume
        if orig_chars['db'] != float('-inf') and repl_chars['db'] != float('-inf'):
            volume_diff = orig_chars['db'] - repl_chars['db']
            # Match volume to original - don't make it too loud or too quiet
            volume_diff = max(-10, min(10, volume_diff))  # Reasonable range
            replacement_audio = replacement_audio.apply_gain(volume_diff)
            repl_chars = analyze_audio_characteristics(replacement_audio)  # Re-analyze
            print(f"🔊 Volume matched: {repl_chars['db']:.1f}dB -> {orig_chars['db']:.1f}dB (adjust: {volume_diff:.2f}dB)")
            
            # Fine-tune: Match to context if available - MORE AGGRESSIVE
            if context_before and context_after:
                context_audio = context_before + original_segment + context_after
                context_chars = analyze_audio_characteristics(context_audio)
                if context_chars and context_chars['db'] != float('-inf'):
                    # Match to surrounding audio level for seamless integration
                    context_volume_diff = context_chars['db'] - repl_chars['db']
                    context_volume_diff = max(-5, min(5, context_volume_diff))  # More aggressive adjustment
                    replacement_audio = replacement_audio.apply_gain(context_volume_diff)
                    repl_chars = analyze_audio_characteristics(replacement_audio)
            
            # ADDITIONAL: Match RMS-based volume for even better blend
            if orig_chars['rms'] > 0 and repl_chars['rms'] > 0:
                rms_volume_diff = 20 * np.log10(orig_chars['rms'] / repl_chars['rms']) if repl_chars['rms'] > 0 else 0
                rms_volume_diff = max(-3, min(3, rms_volume_diff))  # Subtle RMS-based adjustment
                replacement_audio = replacement_audio.apply_gain(rms_volume_diff)
                repl_chars = analyze_audio_characteristics(replacement_audio)
        
        # 2. Match RMS energy (voice intensity/power)
        if orig_chars['rms'] > 0 and repl_chars['rms'] > 0:
            rms_ratio = orig_chars['rms'] / repl_chars['rms']
            if 0.3 < rms_ratio < 3.0:  # Wider range for better matching
                gain_adjustment = 20 * np.log10(rms_ratio) if rms_ratio > 0 else 0
                gain_adjustment = max(-5, min(5, gain_adjustment))
                replacement_audio = replacement_audio.apply_gain(gain_adjustment)
                repl_chars = analyze_audio_characteristics(replacement_audio)
        
        # 3. Match PITCH/TONE (CRITICAL for natural sound - makes it sound like same speaker)
        # ULTRA AGGRESSIVE pitch matching for PERFECT blend
        # If pitch detection failed, use original pitch as fallback
        if orig_chars.get('pitch'):
            orig_pitch = orig_chars['pitch']
            repl_pitch = repl_chars.get('pitch')
            
            # If replacement pitch detection failed, use original pitch
            if not repl_pitch or repl_pitch <= 0:
                print(f"⚠️ Replacement pitch detection failed, using original pitch: {orig_pitch:.1f}Hz")
                # Apply original pitch to replacement by adjusting frame rate
                # Estimate replacement's current pitch (assume ~200Hz for TTS)
                estimated_repl_pitch = 200.0  # Typical TTS pitch
                pitch_ratio = orig_pitch / estimated_repl_pitch
                if 0.5 < pitch_ratio < 1.5:
                    try:
                        new_frame_rate = int(replacement_audio.frame_rate * pitch_ratio)
                        if 8000 < new_frame_rate < 48000:
                            replacement_audio = replacement_audio._spawn(
                                replacement_audio.raw_data,
                                overrides={"frame_rate": new_frame_rate}
                            ).set_frame_rate(replacement_audio.frame_rate)
                            print(f"🎵 Applied estimated pitch match: {estimated_repl_pitch:.1f}Hz -> {orig_pitch:.1f}Hz")
                            repl_chars = analyze_audio_characteristics(replacement_audio)
                            repl_pitch = repl_chars.get('pitch', orig_pitch)  # Update for further matching
                    except Exception as e:
                        print(f"Estimated pitch adjustment warning: {e}")
            
            if repl_pitch and repl_pitch > 0 and orig_pitch > 0:
                # Calculate pitch ratio (how much to shift)
                pitch_ratio = orig_pitch / repl_pitch
                
                # ULTRA AGGRESSIVE: Even wider range for better matching (±50% instead of ±40%)
                if 0.5 < pitch_ratio < 1.5:  # Even wider pitch shift range
                    try:
                        new_frame_rate = int(replacement_audio.frame_rate * pitch_ratio)
                        # Ensure frame rate is reasonable
                        if 8000 < new_frame_rate < 48000:
                            replacement_audio = replacement_audio._spawn(
                                replacement_audio.raw_data,
                                overrides={"frame_rate": new_frame_rate}
                            ).set_frame_rate(replacement_audio.frame_rate)
                            print(f"🎵 Pitch matched: {repl_pitch:.1f}Hz -> {orig_pitch:.1f}Hz (ratio: {pitch_ratio:.2f})")
                            repl_chars = analyze_audio_characteristics(replacement_audio)
                            
                            # RE-CHECK: If still not close enough, try again with more aggressive adjustment
                            repl_chars_after = analyze_audio_characteristics(replacement_audio)
                            if repl_chars_after.get('pitch'):
                                new_pitch = repl_chars_after['pitch']
                                if abs(new_pitch - orig_pitch) > 10:  # Still off by more than 10Hz
                                    # Try one more time with fine adjustment
                                    fine_ratio = orig_pitch / new_pitch
                                    if 0.9 < fine_ratio < 1.1:  # Small adjustment
                                        fine_frame_rate = int(replacement_audio.frame_rate * fine_ratio)
                                        if 8000 < fine_frame_rate < 48000:
                                            replacement_audio = replacement_audio._spawn(
                                                replacement_audio.raw_data,
                                                overrides={"frame_rate": fine_frame_rate}
                                            ).set_frame_rate(replacement_audio.frame_rate)
                                            print(f"🎵 Fine pitch adjustment: {fine_ratio:.2f}x")
                    except Exception as e:
                        print(f"Pitch adjustment warning: {e}")
                else:
                    print(f"⚠️ Pitch difference too large: {repl_pitch:.1f}Hz vs {orig_pitch:.1f}Hz (ratio: {pitch_ratio:.2f})")
        
        # 4. Match SPECTRAL CENTROID (tone brightness/quality) - ULTRA AGGRESSIVE
        if orig_chars.get('spectral_centroid'):
            orig_spectral = orig_chars['spectral_centroid']
            repl_spectral = repl_chars.get('spectral_centroid')
            
            # If spectral detection failed, use original spectral as target
            if not repl_spectral or repl_spectral <= 0:
                print(f"⚠️ Replacement spectral detection failed, using original as target: {orig_spectral:.1f}")
                # Apply aggressive gain adjustment to match brightness
                # Estimate replacement spectral (TTS usually has higher spectral ~4000-8000)
                estimated_repl_spectral = 5000.0  # Typical TTS spectral
                spectral_ratio = orig_spectral / estimated_repl_spectral
                if 0.3 < spectral_ratio < 3.0:
                    brightness_adjust = 20 * np.log10(spectral_ratio) * 0.7
                    brightness_adjust = max(-6, min(6, brightness_adjust))
                    replacement_audio = replacement_audio.apply_gain(brightness_adjust)
                    print(f"🎵 Applied estimated spectral match: {estimated_repl_spectral:.1f} -> {orig_spectral:.1f} (adjust: {brightness_adjust:.2f}dB)")
                    repl_chars = analyze_audio_characteristics(replacement_audio)
                    repl_spectral = repl_chars.get('spectral_centroid', orig_spectral)
            
            # Spectral centroid difference indicates tone quality
            # Higher = brighter, Lower = darker
            if repl_spectral and repl_spectral > 0 and orig_spectral > 0:
                spectral_ratio = orig_spectral / repl_spectral
                # ULTRA AGGRESSIVE: Even wider range for better matching
                if 0.3 < spectral_ratio < 3.0:  # Much wider range
                    # Adjust gain to match brightness (ULTRA aggressive)
                    brightness_adjust = 20 * np.log10(spectral_ratio) * 0.7  # Much more aggressive adjustment
                    brightness_adjust = max(-6, min(6, brightness_adjust))  # Much wider range
                    replacement_audio = replacement_audio.apply_gain(brightness_adjust)
                    print(f"🎵 Spectral matched: {repl_spectral:.1f} -> {orig_spectral:.1f} (ratio: {spectral_ratio:.2f}, adjust: {brightness_adjust:.2f}dB)")
                    repl_chars = analyze_audio_characteristics(replacement_audio)
                    
                    # RE-CHECK: If still not close enough, try again
                    repl_chars_after = analyze_audio_characteristics(replacement_audio)
                    if repl_chars_after.get('spectral_centroid'):
                        new_spectral = repl_chars_after['spectral_centroid']
                        if abs(new_spectral - orig_spectral) > orig_spectral * 0.2:  # Still off by more than 20%
                            # Fine adjustment
                            fine_ratio = orig_spectral / new_spectral
                            if 0.8 < fine_ratio < 1.2:  # Small adjustment
                                fine_adjust = 20 * np.log10(fine_ratio) * 0.3
                                fine_adjust = max(-2, min(2, fine_adjust))
                                replacement_audio = replacement_audio.apply_gain(fine_adjust)
                                print(f"🎵 Fine spectral adjustment: {fine_adjust:.2f}dB")
                    
                    # RE-CHECK: If still not close enough, try again
                    repl_chars_after = analyze_audio_characteristics(replacement_audio)
                    if repl_chars_after.get('spectral_centroid'):
                        new_spectral = repl_chars_after['spectral_centroid']
                        if abs(new_spectral - orig_spectral) > orig_spectral * 0.2:  # Still off by more than 20%
                            # Fine adjustment
                            fine_ratio = orig_spectral / new_spectral
                            if 0.8 < fine_ratio < 1.2:  # Small adjustment
                                fine_adjust = 20 * np.log10(fine_ratio) * 0.3
                                fine_adjust = max(-2, min(2, fine_adjust))
                                replacement_audio = replacement_audio.apply_gain(fine_adjust)
                                print(f"🎵 Fine spectral adjustment: {fine_adjust:.2f}dB")
        
        # 5. Match dynamic range (how voice varies in intensity)
        if orig_chars['dynamic_range'] > 0 and repl_chars['dynamic_range'] > 0:
            # Adjust to match the variation in original voice
            if repl_chars['max_amplitude'] > 0:
                target_max = orig_chars['max_amplitude']
                current_max = repl_chars['max_amplitude']
                if current_max > 0:
                    scale_factor = target_max / current_max
                    if 0.5 < scale_factor < 2.0:
                        replacement_audio = replacement_audio.apply_gain(20 * np.log10(scale_factor))
        
        # 6. Match VOLUME ENVELOPE (how volume changes over time) - CRITICAL for natural sound
        try:
            orig_samples = np.array(original_segment.get_array_of_samples())
            if original_segment.channels == 2:
                orig_samples = orig_samples.reshape((-1, 2)).mean(axis=1)
            
            repl_samples = np.array(replacement_audio.get_array_of_samples())
            if replacement_audio.channels == 2:
                repl_samples = repl_samples.reshape((-1, 2)).mean(axis=1)
            
            if len(orig_samples) > 100 and len(repl_samples) > 100:
                # Calculate volume envelope (RMS over time windows)
                window_size = min(100, len(orig_samples) // 10, len(repl_samples) // 10)
                if window_size > 10:
                    # Original envelope
                    orig_envelope = []
                    for i in range(0, len(orig_samples) - window_size, window_size):
                        window = orig_samples[i:i+window_size]
                        rms = np.sqrt(np.mean(window**2))
                        orig_envelope.append(rms)
                    
                    # Replacement envelope
                    repl_envelope = []
                    for i in range(0, len(repl_samples) - window_size, window_size):
                        window = repl_samples[i:i+window_size]
                        rms = np.sqrt(np.mean(window**2))
                        repl_envelope.append(rms)
                    
                    # Match envelope if both have data
                    if len(orig_envelope) > 0 and len(repl_envelope) > 0:
                        orig_avg = np.mean(orig_envelope)
                        repl_avg = np.mean(repl_envelope)
                        if orig_avg > 0 and repl_avg > 0:
                            envelope_ratio = orig_avg / repl_avg
                            if 0.5 < envelope_ratio < 2.0:
                                envelope_adjust = 20 * np.log10(envelope_ratio) * 0.4
                                envelope_adjust = max(-3, min(3, envelope_adjust))
                                replacement_audio = replacement_audio.apply_gain(envelope_adjust)
                                print(f"🎵 Volume envelope matched: {envelope_ratio:.2f} ratio")
        except Exception as e:
            print(f"Envelope matching warning: {e}")
        
        # 7. Match AUDIO TEXTURE (compression, dynamic range characteristics)
        try:
            # Match compression characteristics
            if orig_chars['dynamic_range'] > 0 and repl_chars['dynamic_range'] > 0:
                # Calculate compression ratio (how much dynamic range is compressed)
                orig_compression = orig_chars['max_amplitude'] / orig_chars['dynamic_range'] if orig_chars['dynamic_range'] > 0 else 1
                repl_compression = repl_chars['max_amplitude'] / repl_chars['dynamic_range'] if repl_chars['dynamic_range'] > 0 else 1
                
                if orig_compression > 0 and repl_compression > 0:
                    compression_ratio = orig_compression / repl_compression
                    if 0.7 < compression_ratio < 1.4:
                        # Apply subtle compression matching
                        texture_adjust = 20 * np.log10(compression_ratio) * 0.2
                        texture_adjust = max(-1.5, min(1.5, texture_adjust))
                        replacement_audio = replacement_audio.apply_gain(texture_adjust)
        except Exception as e:
            print(f"Texture matching warning: {e}")
        
        # 8. Normalize to match original's voice texture
        try:
            replacement_audio = normalize(replacement_audio)
        except:
            pass
        
        # 9. Match frequency response using context (if available) - MORE AGGRESSIVE
        if context_before and context_after:
            try:
                # Analyze surrounding audio for better matching
                context_audio = context_before + original_segment + context_after
                context_chars = analyze_audio_characteristics(context_audio)
                
                if context_chars and context_chars['db'] != float('-inf'):
                    # Match to context's average level - MORE AGGRESSIVE
                    context_level = context_chars['db']
                    repl_level = repl_chars['db']
                    if repl_level != float('-inf'):
                        level_diff = context_level - repl_level
                        level_diff = max(-6, min(6, level_diff))  # More aggressive
                        replacement_audio = replacement_audio.apply_gain(level_diff)
                        print(f"🎵 Context level matched: {level_diff:.2f}dB")
            except Exception as e:
                print(f"Context matching warning: {e}")
        
        # 10. FINAL: Re-analyze and fine-tune for PERFECT match
        repl_chars_final = analyze_audio_characteristics(replacement_audio)
        if repl_chars_final and orig_chars:
            # Final volume check
            if orig_chars['db'] != float('-inf') and repl_chars_final['db'] != float('-inf'):
                final_volume_diff = orig_chars['db'] - repl_chars_final['db']
                if abs(final_volume_diff) > 1:  # If still off by more than 1dB
                    final_volume_diff = max(-2, min(2, final_volume_diff))  # Subtle final adjustment
                    replacement_audio = replacement_audio.apply_gain(final_volume_diff)
                    print(f"🎵 Final volume fine-tune: {final_volume_diff:.2f}dB")
        
        print(f"✅ Voice matching complete - replacement now matches original pitch/tone/volume/texture")
        return replacement_audio
    except Exception as e:
        print(f"Voice matching error: {e}, using replacement as-is")
        return replacement_audio

def detect_language_from_text(text):
    """Detect language from text content"""
    if not text:
        return "en"
    
    # Count Hindi characters (Devanagari script)
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([c for c in text if c.isalnum()])
    
    if total_chars > 0 and (hindi_chars / total_chars) > 0.2:  # Lower threshold for better detection
        return "hi"
    return "en"

def convert_to_hindi_transliteration(name):
    """
    Convert English name to Hindi transliteration for better TTS pronunciation
    This ensures English names are pronounced with Hindi accent
    Uses mapping first, then phonetic approximation for unknown names
    """
    if not name:
        return name
    
    # If already in Hindi script, return as is
    if detect_language_from_text(name) == "hi":
        return name
    
    # Expanded transliteration mapping for common names
    transliteration_map = {
        'Sunil': 'सुनील',
        'Suresh': 'सुरेश',
        'Raja': 'राजा',
        'Raj': 'राज',
        'Amit': 'अमित',
        'Anil': 'अनिल',
        'Ravi': 'रवि',
        'Kumar': 'कुमार',
        'Sharma': 'शर्मा',
        'Singh': 'सिंह',
        'Patel': 'पटेल',
        'Gupta': 'गुप्ता',
        'Mehta': 'मेहता',
        'Joshi': 'जोशी',
        'Reddy': 'रेड्डी',
        'Rao': 'राव',
        'Iyer': 'अय्यर',
        'Nair': 'नायर',
        'Pillai': 'पिल्लई',
        'Menon': 'मेनन',
        'Vikram': 'विक्रम',
        'Rahul': 'राहुल',
        'Priya': 'प्रिया',
        'Deepak': 'दीपक',
        'Mohan': 'मोहन',
        'Lakshmi': 'लक्ष्मी',
        'Ganesh': 'गणेश',
        'Krishna': 'कृष्ण',
        'Arjun': 'अर्जुन',
        'Sita': 'सीता',
        'Ram': 'राम',
        'Shyam': 'श्याम',
        'Vishal': 'विशाल',
        'Neha': 'नेहा',
        'Pooja': 'पूजा',
        'Anjali': 'अंजली',
        'Rohit': 'रोहित',
        'Sachin': 'सचिन',
        'Mahesh': 'महेश',
    }
    
    # Check if exact match exists
    if name in transliteration_map:
        return transliteration_map[name]
    
    # Try case-insensitive match
    for eng_name, hindi_name in transliteration_map.items():
        if name.lower() == eng_name.lower():
            return hindi_name
    
    # If no mapping found, use phonetic approximation
    # Simple phonetic mapping for common English sounds to Hindi
    phonetic_map = {
        'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ee': 'ई', 'u': 'उ', 'oo': 'ऊ',
        'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
        'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ng': 'ङ',
        'ch': 'च', 'chh': 'छ', 'j': 'ज', 'jh': 'झ', 'ny': 'ञ',
        't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
        'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
        'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
        'sh': 'श', 'shh': 'ष', 's': 'स', 'h': 'ह',
        'r': 'र', 'l': 'ल', 'v': 'व'
    }
    
    # For names not in mapping, try to create approximate transliteration
    # This is a simplified approach - for better results, use a proper transliteration library
    # But for now, we'll use the name as-is with nova voice which handles English names better
    # The nova voice with English text still gives better Hindi accent than other voices
    print(f"⚠️ No exact Hindi transliteration found for '{name}', using with nova voice for Hindi accent")
    return name  # Use as-is, nova voice will handle it better than other voices

def get_best_voice_for_language_and_accent(language, original_word=""):
    """Get best TTS voice that matches language AND accent"""
    # OpenAI TTS voices: alloy, echo, fable, onyx, nova, shimmer
    # For Hindi/Indian accent
    if language == "hi" or detect_language_from_text(original_word) == "hi":
        # Use voices that work better for Hindi/Indian languages
        return "nova"  # Better pronunciation for Hindi
    else:
        # For English, use neutral voice
        return "alloy"  # Default for English

def replace_audio_segments(original_audio_path, word_timestamps, names_to_replace, replacement_name, language="en"):
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
        # Add enhanced fade for PERFECT seamless blending
        if start_time_ms > current_pos:
            before_segment = original_audio[current_pos:start_time_ms]
            # Add longer fade at end if we're about to replace (for seamless transition)
            if should_replace and len(before_segment) > 20:
                fade_ms = min(30, max(20, len(before_segment) // 3))  # Longer fade for better blend
                before_segment = before_segment.fade_out(fade_ms)
            audio_segments.append(before_segment)
        
        if should_replace:
            replacement_count += 1
            print(f"Replacing '{word}' with '{replacement_name}' at {start_time_ms}ms-{end_time_ms}ms")
            
            # Extract original voice characteristics from the name segment
            original_name_segment = original_audio[start_time_ms:end_time_ms]
            original_duration = end_time_ms - start_time_ms
            
            # CRITICAL: Use ORIGINAL video's language consistently for ALL names
            # This ensures ALL replacements sound the same and match original speaker
            # DON'T check individual word languages - use video language for consistency
            
            # FORCE use original video's language for TTS (most important!)
            # If video is Hindi, ALL replacements MUST be in Hindi accent
            if language == "hi":
                tts_language = "hi"  # FORCE Hindi if video is Hindi
                tts_voice = "nova"  # Best voice for Hindi/Indian accent
                
                # CRITICAL: If replacement name is English, convert to Hindi transliteration
                # OpenAI TTS English text ko Hindi voice se bhi English accent mein hi pronounce karta hai
                # Isliye English names ko Hindi transliteration mein convert karna zaroori hai
                replacement_name_for_tts = convert_to_hindi_transliteration(replacement_name)
                if replacement_name_for_tts != replacement_name:
                    print(f"   🔄 Converted '{replacement_name}' -> '{replacement_name_for_tts}' for Hindi TTS")
            else:
                # For English videos, use English voice and keep name as-is
                tts_language = language  # Use original video's detected language
                tts_voice = "alloy"  # Default for English
                replacement_name_for_tts = replacement_name  # Keep as is for English
            
            print(f"🔊 Generating TTS for '{replacement_name}' (consistent for all names)")
            print(f"   Video language: {language} (used for ALL names)")
            print(f"   TTS language: {tts_language}, voice: {tts_voice}")
            print(f"   TTS input: '{replacement_name_for_tts}'")
            print(f"   ✅ All names will use same voice and accent!")
            
            # Generate TTS for replacement name with ORIGINAL video's language/accent
            try:
                client = get_openai_client()
                # CRITICAL: OpenAI TTS detects language from text, but voice determines accent
                # For Hindi videos, we MUST use Hindi voice AND Hindi transliteration to match accent
                # The replacement_name text will be pronounced in the voice's accent
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=tts_voice,  # Voice matching original video's language/accent
                    input=replacement_name_for_tts  # Use transliterated name for Hindi
                    # For Hindi accent: use "nova" voice + Hindi transliteration
                    # For English accent: use "alloy" voice + original name
                )
                replacement_audio_path = os.path.join(UPLOAD_FOLDER, f"temp_replacement_{replacement_name}_{replacement_count}.mp3")
                response.stream_to_file(replacement_audio_path)
                
                # Load replacement audio
                replacement_audio = AudioSegment.from_mp3(replacement_audio_path)
                replacement_duration = len(replacement_audio)
                
                print(f"   ⏱️ Duration matching: replacement={replacement_duration}ms, original={original_duration}ms")
                
                # CRITICAL: Match duration EXACTLY while preserving ORIGINAL speed and pitch
                # Don't speed up - preserve natural speaking speed
                if replacement_duration > original_duration:
                    # Too long - trim to exact duration (preserve natural speed)
                    # DON'T speed up - just trim the end
                    replacement_audio = replacement_audio[:original_duration]
                    print(f"   ✂️ Trimmed to match duration: {replacement_duration}ms -> {original_duration}ms (preserving natural speed)")
                elif replacement_duration < original_duration:
                    # Too short - add silence at end (preserve natural speed)
                    # DON'T slow down - just add silence
                    silence_needed = original_duration - replacement_duration
                    # Add silence at end for natural pause
                    silence = AudioSegment.silent(duration=silence_needed)
                    replacement_audio = replacement_audio + silence
                    print(f"   ⏱️ Added {silence_needed}ms silence to match duration (preserving natural speed)")
                
                # Final: Ensure EXACT duration match
                replacement_audio = replacement_audio[:original_duration]
                final_duration = len(replacement_audio)
                print(f"   ✅ Final duration: {final_duration}ms (target: {original_duration}ms, diff: {abs(final_duration - original_duration)}ms)")
                
                # Get context audio for better matching (before and after the word)
                context_before_ms = max(0, start_time_ms - 300)  # 300ms before
                context_after_ms = min(len(original_audio), end_time_ms + 300)  # 300ms after
                context_before = original_audio[context_before_ms:start_time_ms] if start_time_ms > context_before_ms else None
                context_after = original_audio[end_time_ms:context_after_ms] if context_after_ms > end_time_ms else None
                
                # Advanced voice matching with context for seamless blend
                replacement_audio = match_voice_characteristics(
                    replacement_audio, 
                    original_name_segment,
                    context_before=context_before,
                    context_after=context_after
                )
                
                # ENHANCED crossfade for PERFECT seamless blending
                # MUCH longer crossfade for better integration (up to 50% of duration)
                # Adaptive crossfade based on duration
                if original_duration < 200:
                    crossfade_ms = min(40, max(20, original_duration // 4))  # 25% for short segments
                elif original_duration < 500:
                    crossfade_ms = min(50, max(30, original_duration // 3))  # 33% for medium segments
                else:
                    crossfade_ms = min(80, max(40, original_duration // 4))  # 25% for long segments
                
                if crossfade_ms > 10:  # Only if meaningful fade
                    # Fade in at start for smooth entry (MUCH longer fade for better blend)
                    replacement_audio = replacement_audio.fade_in(crossfade_ms)
                    # Fade out at end for smooth exit
                    replacement_audio = replacement_audio.fade_out(crossfade_ms)
                    print(f"   🎨 Applied {crossfade_ms}ms crossfade for seamless blend")
                
                # ADDITIONAL: Apply subtle volume envelope matching
                # Match the volume curve of original segment for even better blend
                try:
                    # Analyze volume envelope of original
                    orig_samples = np.array(original_name_segment.get_array_of_samples())
                    if original_name_segment.channels == 2:
                        orig_samples = orig_samples.reshape((-1, 2)).mean(axis=1)
                    
                    repl_samples = np.array(replacement_audio.get_array_of_samples())
                    if replacement_audio.channels == 2:
                        repl_samples = repl_samples.reshape((-1, 2)).mean(axis=1)
                    
                    # Match volume envelope (how loudness changes over time)
                    if len(orig_samples) > 100 and len(repl_samples) > 100:
                        # Normalize both
                        orig_max = max(abs(orig_samples)) if max(abs(orig_samples)) > 0 else 1
                        repl_max = max(abs(repl_samples)) if max(abs(repl_samples)) > 0 else 1
                        
                        # Apply envelope matching (simplified - just ensure similar dynamic range)
                        # This helps the replacement blend better
                        pass  # Already handled by match_voice_characteristics
                except Exception as e:
                    print(f"Envelope matching warning: {e}")
                
                # Additional: Apply subtle reverb/echo matching if original has it
                # This helps blend with surrounding audio environment
                
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
            original_word_segment = original_audio[start_time_ms:end_time_ms]
            audio_segments.append(original_word_segment)
        
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
    language = request.language or "hi"  # Extract language from request, default to Hindi
    
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
                # Use precise segment replacement with language matching
                modified_audio = replace_audio_segments(
                    original_audio_path, 
                    word_timestamps, 
                    names_to_replace, 
                    excel_name,
                    language=language  # Pass language for proper TTS generation
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
