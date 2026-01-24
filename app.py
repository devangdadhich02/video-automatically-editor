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
            # Try reading Excel file - may need openpyxl or xlrd
            try:
                df = pd.read_excel(filepath, engine='openpyxl')
            except Exception as e1:
                try:
                    df = pd.read_excel(filepath, engine='xlrd')
                except Exception as e2:
                    print(f"Excel read error (openpyxl): {e1}")
                    print(f"Excel read error (xlrd): {e2}")
                    raise Exception(f"Could not read Excel file. Please ensure file is valid. Error: {str(e1)}")
        
        # Check if dataframe is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="Excel file is empty")
        
        # Get names from first column
        names = df.iloc[:, 0].dropna().astype(str).tolist()
        
        if not names:
            raise HTTPException(status_code=400, detail="No names found in the first column of Excel file")
        
        # Return names as-is (English) - they will be converted to Hindi during TTS generation
        # This ensures matching works with transcript (which has English names)
        return {
            'success': True,
            'names': names,
            'count': len(names)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Excel upload error: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error reading Excel file: {str(e)}")

def analyze_audio_characteristics(audio_segment):
    """ADVANCED audio analysis - detects pitch, tone, base, spectral characteristics"""
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
        
        # ADVANCED: Use librosa for spectral analysis (base/tone)
        spectral_centroid = None
        spectral_rolloff = None
        zero_crossing_rate = None
        pitch_estimate = None
        
        if len(samples) > 500:
            try:
                frame_rate = audio_segment.frame_rate
                
                # Normalize samples for librosa
                samples_float = samples.astype(np.float32)
                if max(abs(samples_float)) > 0:
                    samples_float = samples_float / max(abs(samples_float))
                
                # Use librosa for advanced spectral analysis
                # Spectral centroid = "brightness" or "tone" of the sound
                spectral_centroid = librosa.feature.spectral_centroid(y=samples_float, sr=frame_rate)[0]
                spectral_centroid = np.mean(spectral_centroid) if len(spectral_centroid) > 0 else None
                
                # Spectral rolloff = frequency below which 85% of energy is contained
                spectral_rolloff = librosa.feature.spectral_rolloff(y=samples_float, sr=frame_rate)[0]
                spectral_rolloff = np.mean(spectral_rolloff) if len(spectral_rolloff) > 0 else None
                
                # Zero crossing rate = measure of noisiness
                zero_crossing_rate = librosa.feature.zero_crossing_rate(samples_float)[0]
                zero_crossing_rate = np.mean(zero_crossing_rate) if len(zero_crossing_rate) > 0 else None
                
                # Pitch detection using librosa (more accurate)
                pitches, magnitudes = librosa.piptrack(y=samples_float, sr=frame_rate)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if len(pitch_values) > 0:
                    # Use median for more stable pitch estimate
                    pitch_estimate = np.median(pitch_values)
                    # Filter out unrealistic values
                    if not (80 <= pitch_estimate <= 400):
                        pitch_estimate = None
                
            except Exception as e:
                print(f"Spectral analysis error: {e}")
                # Fallback to autocorrelation method
                try:
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
                except:
                    pass
        
        # Detect gender based on pitch
        # Male: ~85-180 Hz, Female: ~165-255 Hz
        gender = None
        if pitch_estimate:
            if 85 <= pitch_estimate <= 180:
                gender = "male"
            elif 165 <= pitch_estimate <= 255:
                gender = "female"
            # Overlap zone: use spectral centroid as tiebreaker
            elif 165 <= pitch_estimate <= 180:
                if spectral_centroid and spectral_centroid < 2000:
                    gender = "male"  # Lower centroid = male
                else:
                    gender = "female"
        
        return {
            'rms': rms,
            'db': db,
            'max_amplitude': max_amplitude,
            'dynamic_range': dynamic_range,
            'samples': samples,
            'pitch': pitch_estimate,
            'gender': gender,  # Added gender detection
            'spectral_centroid': spectral_centroid,  # Base/tone
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'frame_rate': audio_segment.frame_rate
        }
    except Exception as e:
        print(f"Audio analysis error: {e}")
        return None

def create_master_voice_profile(original_audio_path):
    """
    Create MASTER voice profile from entire original audio.
    This profile is used for ALL name replacements to ensure consistency.
    
    Returns a fixed profile with:
    - avg_pitch
    - avg_spectral_envelope
    - avg_low_mid_energy (150-400Hz)
    - avg_formant_curve
    - gender
    - avg_speech_rate
    """
    try:
        print(f"🎯 Creating MASTER VOICE PROFILE from original audio...")
        original_audio = AudioSegment.from_wav(original_audio_path)
        
        # Analyze entire audio (not just segments)
        master_chars = analyze_audio_characteristics(original_audio)
        
        if not master_chars:
            print(f"   ⚠️ Failed to create master profile, using defaults")
            return None
        
        # Extract spectral envelope from entire audio
        frame_rate = original_audio.frame_rate
        samples = np.array(original_audio.get_array_of_samples())
        if original_audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples = samples.astype(np.float32)
        if max(abs(samples)) > 0:
            samples = samples / max(abs(samples))
        
        # Get average spectral envelope using STFT
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
        avg_spectral_envelope = np.mean(np.abs(stft), axis=1)  # Average across time
        
        # Calculate average low-mid energy (150-400 Hz)
        freqs = librosa.fft_frequencies(sr=frame_rate, n_fft=n_fft)
        low_mid_mask = (freqs >= 150) & (freqs <= 400)
        avg_low_mid_energy = np.mean(avg_spectral_envelope[low_mid_mask])
        
        # Calculate average formant curve (500-1200 Hz)
        formant_mask = (freqs >= 500) & (freqs <= 1200)
        avg_formant_curve = np.mean(avg_spectral_envelope[formant_mask])
        
        # Estimate speech rate (syllables per second approximation)
        # Use zero crossing rate as proxy
        zcr = master_chars.get('zero_crossing_rate', 0.05)
        # Rough estimate: higher ZCR = faster speech
        avg_speech_rate = zcr * 10  # Normalized estimate
        
        master_profile = {
            'avg_pitch': master_chars.get('pitch'),
            'avg_spectral_envelope': avg_spectral_envelope,
            'avg_low_mid_energy': avg_low_mid_energy,
            'avg_formant_curve': avg_formant_curve,
            'gender': master_chars.get('gender'),
            'avg_speech_rate': avg_speech_rate,
            'avg_db': master_chars.get('db'),
            'avg_rms': master_chars.get('rms'),
            'avg_spectral_centroid': master_chars.get('spectral_centroid'),
            'avg_spectral_rolloff': master_chars.get('spectral_rolloff'),
            'frame_rate': frame_rate,
            'freqs': freqs
        }
        
        print(f"   ✅ MASTER PROFILE created:")
        print(f"      Pitch: {master_profile['avg_pitch']:.1f}Hz")
        print(f"      Gender: {master_profile['gender']}")
        print(f"      Low-Mid Energy: {master_profile['avg_low_mid_energy']:.4f}")
        print(f"      Formant Curve: {master_profile['avg_formant_curve']:.4f}")
        print(f"      Speech Rate: {master_profile['avg_speech_rate']:.2f}")
        
        return master_profile
    except Exception as e:
        print(f"   ❌ Master profile creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def match_voice_characteristics(replacement_audio, original_segment=None, master_profile=None):
    """
    EXACT voice matching using MASTER PROFILE for consistency.
    
    If master_profile is provided, uses FIXED profile for ALL replacements.
    If not, falls back to per-segment analysis (legacy mode).
    """
    try:
        # CRITICAL: Use master profile if available (ensures consistency)
        if master_profile:
            print(f"   🎯 Using MASTER VOICE PROFILE (consistent across all names)")
            orig_chars = {
                'pitch': master_profile['avg_pitch'],
                'gender': master_profile['gender'],
                'db': master_profile['avg_db'],
                'rms': master_profile['avg_rms'],
                'spectral_centroid': master_profile['avg_spectral_centroid'],
                'spectral_rolloff': master_profile['avg_spectral_rolloff'],
                'frame_rate': master_profile['frame_rate']
            }
        else:
            # Legacy mode: analyze per-segment (can cause variation)
            if original_segment:
                orig_chars = analyze_audio_characteristics(original_segment)
            else:
                print(f"   ⚠️ No master profile or original segment, using replacement audio as-is")
                return replacement_audio
        
        repl_chars = analyze_audio_characteristics(replacement_audio)
        
        if not orig_chars or not repl_chars:
            return replacement_audio
        
        # Detect gender from original audio
        orig_gender = orig_chars.get('gender')
        orig_pitch = orig_chars.get('pitch')
        
        if master_profile:
            print(f"   📊 Using MASTER PROFILE (FIXED): Pitch={orig_pitch:.1f}Hz, Gender={orig_gender or 'unknown'}, dB={orig_chars.get('db', 'N/A'):.1f}")
        else:
            print(f"   📊 Matching: Original dB={orig_chars.get('db', 'N/A'):.1f}, Pitch={orig_pitch:.1f}Hz, Gender={orig_gender or 'unknown'}, Centroid={orig_chars.get('spectral_centroid', 'N/A'):.0f}")
        print(f"   📊 Before:   Replacement dB={repl_chars['db']:.1f}, Pitch={repl_chars.get('pitch', 'N/A'):.1f}Hz, Centroid={repl_chars.get('spectral_centroid', 'N/A'):.0f}")
        
        # 1. EXACT decibel matching (no limits - match exactly)
        if orig_chars['db'] != float('-inf') and repl_chars['db'] != float('-inf'):
            volume_diff = orig_chars['db'] - repl_chars['db']
            replacement_audio = replacement_audio.apply_gain(volume_diff)
            print(f"   🔊 Decibel matched: {volume_diff:.2f}dB")
        
        # 2. Match RMS energy (overall volume level)
        if orig_chars['rms'] > 0 and repl_chars['rms'] > 0:
            rms_ratio = orig_chars['rms'] / repl_chars['rms']
            if 0.5 < rms_ratio < 2.0:  # Reasonable range
                rms_gain_db = 20 * np.log10(rms_ratio)
                replacement_audio = replacement_audio.apply_gain(rms_gain_db)
                print(f"   📈 RMS energy matched: {rms_gain_db:.2f}dB")
        
        # 3. GENDER-AWARE pitch matching with formant processing
        # CRITICAL: If original is MALE, force lower pitch + formant shift for male vocal tract
        if orig_chars.get('pitch') and repl_chars.get('pitch'):
            pitch_ratio = orig_chars['pitch'] / repl_chars['pitch']
            
            # If original is MALE, apply aggressive pitch lowering + formant shift
            if orig_gender == "male":
                # Force pitch down by -4 to -7 semitones for male voice
                # Calculate base semitones from ratio
                base_semitones = 12 * np.log2(pitch_ratio)
                
                # If TTS is too high (female/neutral), force it down more
                if repl_chars.get('pitch', 0) > 180:  # TTS is likely female
                    # Force -5 to -7 semitones down for male conversion
                    target_semitones = -6.0  # Force male range
                    print(f"   🎭 GENDER CONVERSION: Original is MALE ({orig_pitch:.1f}Hz), TTS is high ({repl_chars.get('pitch', 0):.1f}Hz)")
                    print(f"   🎭 Forcing pitch down by {abs(target_semitones):.1f} semitones for male voice")
                else:
                    # Use calculated ratio but ensure it's in male range
                    target_semitones = max(-7.0, min(-4.0, base_semitones))
                    print(f"   🎭 MALE VOICE: Adjusting pitch by {target_semitones:.1f} semitones")
            else:
                # Female or unknown - use normal pitch matching
                target_semitones = 12 * np.log2(pitch_ratio)
            
            # Match pitch even for small differences (>1%) - CRITICAL for natural sound
            if abs(target_semitones) > 0.1 and -10 <= target_semitones <= 10:
                try:
                    frame_rate = replacement_audio.frame_rate
                    
                    # Save to temp file for librosa processing
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        replacement_audio.export(tmp.name, format="wav")
                        temp_path = tmp.name
                    
                    # Load with librosa
                    y, sr = librosa.load(temp_path, sr=frame_rate)
                    
                    # Pitch shift using librosa (preserves quality)
                    # librosa.effects.pitch_shift shifts by semitones
                    y_shifted = librosa.effects.pitch_shift(
                        y=y, 
                        sr=frame_rate, 
                        n_steps=target_semitones,
                        bins_per_octave=12
                    )
                    
                    # CRITICAL: If original is MALE, apply FORMANT SHIFT (male vocal tract simulation)
                    if orig_gender == "male":
                        # Formant shift: emphasize low frequencies (150-400 Hz) for male vocal tract
                        # Use STFT to manipulate spectral envelope in low-mid range
                        n_fft = 2048
                        hop_length = 512
                        stft = librosa.stft(y_shifted, n_fft=n_fft, hop_length=hop_length)
                        magnitude = np.abs(stft)
                        phase = np.angle(stft)
                        
                        # Get frequency bins
                        freqs = librosa.fft_frequencies(sr=frame_rate, n_fft=n_fft)
                        
                        # Boost low-mid frequencies (150-400 Hz) for male formants
                        # Reduce high frequencies (>2800 Hz) to remove female brightness
                        for i, freq in enumerate(freqs):
                            if 150 <= freq <= 400:
                                # Boost male formant range by 1.5x
                                magnitude[i, :] *= 1.5
                            elif freq > 2800:
                                # Reduce high frequencies for male timbre
                                magnitude[i, :] *= 0.6
                        
                        # Reconstruct with modified magnitude
                        stft_modified = magnitude * np.exp(1j * phase)
                        y_shifted = librosa.istft(stft_modified, hop_length=hop_length)
                        
                        print(f"   🎭 Applied FORMANT SHIFT: Boosted 150-400Hz (male formants), Reduced >2800Hz")
                        
                        # Save shifted audio
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                            sf.write(tmp2.name, y_shifted, frame_rate)
                            temp_path2 = tmp2.name
                        
                        # Load back
                        replacement_audio = AudioSegment.from_wav(temp_path2)
                        
                        # Cleanup
                        os.unlink(temp_path)
                        os.unlink(temp_path2)
                        
                        print(f"   🎵 Pitch matched: {pitch_ratio:.3f}x ({target_semitones:.1f} semitones)")
                        
                        # CRITICAL: If original is MALE, apply additional male voice processing
                        if orig_gender == "male":
                            # Stronger low-pass filtering for male timbre (≤2800 Hz)
                            replacement_audio = replacement_audio.low_pass_filter(2800)
                            replacement_audio = replacement_audio.low_pass_filter(3000)
                            print(f"   🎭 Applied STRONG low-pass filters (≤2800Hz) for male timbre")
                            
                            # Low-mid boost (150-400 Hz) using EQ-like filtering
                            # This simulates male vocal tract resonance
                            try:
                                # Convert to numpy for frequency-domain processing
                                samples = np.array(replacement_audio.get_array_of_samples())
                                if replacement_audio.channels == 2:
                                    samples = samples.reshape((-1, 2)).mean(axis=1)
                                samples = samples.astype(np.float32)
                                if max(abs(samples)) > 0:
                                    samples = samples / max(abs(samples))
                                
                                # Apply low-mid boost using STFT
                                n_fft = 2048
                                hop_length = 512
                                stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
                                magnitude = np.abs(stft)
                                phase = np.angle(stft)
                                freqs = librosa.fft_frequencies(sr=frame_rate, n_fft=n_fft)
                                
                                # Boost 150-400 Hz range (male formants)
                                # LOWER FORMANTS: Reduce 500-1200 Hz for adult male (not teenager)
                                for i, freq in enumerate(freqs):
                                    if 150 <= freq <= 400:
                                        magnitude[i, :] *= 1.8  # Strong boost for male resonance
                                    elif 500 <= freq <= 1200:
                                        # Further lower formants for adult male
                                        magnitude[i, :] *= 0.85  # 15% reduction
                                
                                # Reconstruct
                                stft_boosted = magnitude * np.exp(1j * phase)
                                y_boosted = librosa.istft(stft_boosted, hop_length=hop_length)
                                
                                # Normalize
                                if max(abs(y_boosted)) > 0:
                                    y_boosted = y_boosted / max(abs(y_boosted)) * 0.95
                                
                                # Save and load back
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp3:
                                    sf.write(tmp3.name, y_boosted, frame_rate)
                                    temp_path3 = tmp3.name
                                replacement_audio = AudioSegment.from_wav(temp_path3)
                                os.unlink(temp_path3)
                                
                                print(f"   🎭 Applied LOW-MID BOOST (150-400Hz, 1.8x) for male vocal tract")
                            except Exception as e:
                                print(f"   ⚠️ Low-mid boost failed: {e}, using filtered audio")
                            
                            # Apply gentle compression/saturation for thickness
                            # This adds warmth and body to male voice
                            try:
                                # Slight compression by limiting dynamic range
                                replacement_audio = replacement_audio.normalize()
                                # Apply gentle gain reduction for warmth
                                replacement_audio = replacement_audio.apply_gain(-1.0)
                                replacement_audio = replacement_audio.normalize()
                                print(f"   🎭 Applied compression/saturation for male voice thickness")
                            except:
                                pass
                    else:
                        os.unlink(temp_path)
                        print(f"   ⚠️ Pitch difference too large ({target_semitones:.1f} semitones), using frame rate adjustment")
                        # Fallback: use frame rate adjustment for extreme cases
                        new_frame_rate = int(frame_rate * pitch_ratio)
                        if 16000 <= new_frame_rate <= 48000:
                            replacement_audio = replacement_audio._spawn(
                                replacement_audio.raw_data,
                                overrides={"frame_rate": new_frame_rate}
                            ).set_frame_rate(frame_rate)
                except Exception as e:
                    print(f"   ⚠️ Pitch matching failed: {e}, trying frame rate adjustment")
                    # Fallback to frame rate adjustment
                    try:
                        frame_rate = replacement_audio.frame_rate
                        new_frame_rate = int(frame_rate * pitch_ratio)
                        if 16000 <= new_frame_rate <= 48000:
                            replacement_audio = replacement_audio._spawn(
                                replacement_audio.raw_data,
                                overrides={"frame_rate": new_frame_rate}
                            ).set_frame_rate(frame_rate)
                    except:
                        pass
            else:
                print(f"   ✅ Pitch already very close: {pitch_ratio:.3f}x")
        
        # 4. Match spectral centroid (base/tone) using EQ-like filtering - MAKE SOUND MUCH THICKER
        if orig_chars.get('spectral_centroid') and repl_chars.get('spectral_centroid'):
            centroid_ratio = orig_chars['spectral_centroid'] / repl_chars['spectral_centroid']
            # Match even very small differences (>1%) to get closer to original tone
            if abs(centroid_ratio - 1.0) > 0.01 and 0.4 < centroid_ratio < 1.6:
                try:
                    # Apply more aggressive filtering to match spectral characteristics
                    # Higher centroid = brighter/thinner sound, lower = darker/thicker sound
                    # If replacement is thinner (higher centroid), make it MUCH thicker (lower centroid)
                    if centroid_ratio > 1.2:
                        # Replacement is much thinner - apply VERY strong low-pass (4x filters for maximum thickness)
                        replacement_audio = replacement_audio.low_pass_filter(1800)  # Very aggressive
                        replacement_audio = replacement_audio.low_pass_filter(2200)  # Strong filter
                        replacement_audio = replacement_audio.low_pass_filter(2600)  # Additional thickness
                        replacement_audio = replacement_audio.low_pass_filter(3000)  # Smooth it out
                    elif centroid_ratio > 1.1:
                        # Replacement is thinner - apply strong low-pass (3x filters)
                        replacement_audio = replacement_audio.low_pass_filter(2000)  # Strong filter
                        replacement_audio = replacement_audio.low_pass_filter(2500)  # Additional thickness
                        replacement_audio = replacement_audio.low_pass_filter(3000)  # Smooth it out
                    elif centroid_ratio > 1.05:
                        # Replacement is slightly thinner - apply moderate low-pass (2x filters)
                        replacement_audio = replacement_audio.low_pass_filter(2500)  # Moderate filter
                        replacement_audio = replacement_audio.low_pass_filter(3500)  # Additional thickness
                    elif centroid_ratio < 0.8:
                        # Replacement is thicker - apply high-pass to match
                        replacement_audio = replacement_audio.high_pass_filter(200)
                    elif centroid_ratio < 0.9:
                        # Replacement is slightly thicker
                        replacement_audio = replacement_audio.high_pass_filter(150)
                    print(f"   🎚️ Spectral centroid (base/tone) matched: {centroid_ratio:.2f}x (making MUCH thicker)")
                except Exception as e:
                    print(f"   ⚠️ Spectral matching failed: {e}")
        
        # 4.5. ADVANCED: Apply MASTER PROFILE spectral envelope for exact tone matching
        # CRITICAL: Use master profile envelope for consistency across all names
        try:
            frame_rate = replacement_audio.frame_rate
            
            # CRITICAL: Use master profile spectral envelope for consistency
            if master_profile and master_profile.get('avg_spectral_envelope') is not None:
                # Use FIXED master profile envelope (same for all names)
                master_envelope = master_profile['avg_spectral_envelope']
                master_freqs = master_profile['freqs']
                print(f"   🎯 Using MASTER PROFILE spectral envelope (FIXED - consistent across all names)")
                
                # Get replacement audio STFT
                repl_samples = np.array(replacement_audio.get_array_of_samples())
                if replacement_audio.channels == 2:
                    repl_samples = repl_samples.reshape((-1, 2)).mean(axis=1)
                repl_samples = repl_samples.astype(np.float32)
                if max(abs(repl_samples)) > 0:
                    repl_samples = repl_samples / max(abs(repl_samples))
                
                n_fft = 2048
                hop_length = 512
                repl_stft = librosa.stft(repl_samples, n_fft=n_fft, hop_length=hop_length)
                repl_mag = np.abs(repl_stft)
                
                # Use master envelope directly (expand to match replacement's time dimension)
                # Master envelope is 1D (frequency bins), expand to match replacement's time frames
                if len(master_envelope) == repl_mag.shape[0]:
                    # Expand master envelope to match replacement's time dimension
                    orig_mag = np.tile(master_envelope[:, np.newaxis], (1, repl_mag.shape[1]))
                else:
                    # Interpolate if sizes don't match (simple linear interpolation)
                    new_freqs = librosa.fft_frequencies(sr=frame_rate, n_fft=n_fft)
                    # Simple linear interpolation using numpy
                    expanded_envelope = np.interp(new_freqs[:repl_mag.shape[0]], 
                                                 master_freqs[:len(master_envelope)], 
                                                 master_envelope)
                    orig_mag = np.tile(expanded_envelope[:, np.newaxis], (1, repl_mag.shape[1]))
            else:
                # Fallback: use original segment (legacy mode - may cause variation)
                if original_segment:
                    orig_samples = np.array(original_segment.get_array_of_samples())
                    if original_segment.channels == 2:
                        orig_samples = orig_samples.reshape((-1, 2)).mean(axis=1)
                    orig_samples = orig_samples.astype(np.float32)
                    if max(abs(orig_samples)) > 0:
                        orig_samples = orig_samples / max(abs(orig_samples))
                    
                    repl_samples = np.array(replacement_audio.get_array_of_samples())
                    if replacement_audio.channels == 2:
                        repl_samples = repl_samples.reshape((-1, 2)).mean(axis=1)
                    repl_samples = repl_samples.astype(np.float32)
                    if max(abs(repl_samples)) > 0:
                        repl_samples = repl_samples / max(abs(repl_samples))
                    
                    # Resample if needed
                    if original_segment.frame_rate != frame_rate:
                        orig_samples = librosa.resample(orig_samples, orig_sr=original_segment.frame_rate, target_sr=frame_rate)
                    
                    n_fft = 2048
                    hop_length = 512
                    orig_stft = librosa.stft(orig_samples, n_fft=n_fft, hop_length=hop_length)
                    repl_stft = librosa.stft(repl_samples, n_fft=n_fft, hop_length=hop_length)
                    orig_mag = np.abs(orig_stft)
                    repl_mag = np.abs(repl_stft)
                else:
                    # Skip if no master profile and no segment
                    print(f"   ⚠️ No master profile or segment for envelope matching, skipping")
                    raise Exception("No master profile or segment available")
            
            # Match the spectral envelope shape
            # Use original's spectral shape but keep replacement's phase
            if orig_mag.shape == repl_mag.shape:
                # Normalize and apply original's spectral envelope
                orig_mag_norm = orig_mag / (np.max(orig_mag) + 1e-10)
                repl_mag_norm = repl_mag / (np.max(repl_mag) + 1e-10)
                
                # Blend: use 98% original envelope, 2% replacement (MAXIMUM THICKNESS - ALMOST IDENTICAL TO ORIGINAL)
                blended_mag = 0.98 * orig_mag_norm + 0.02 * repl_mag_norm
                
                # Apply to replacement's phase
                repl_phase = np.angle(repl_stft)
                new_stft = blended_mag * np.exp(1j * repl_phase)
                
                # Convert back to time domain
                y_matched = librosa.istft(new_stft, hop_length=hop_length)
                
                # Normalize
                if max(abs(y_matched)) > 0:
                    y_matched = y_matched / max(abs(y_matched)) * 0.95
                
                # Save and load back
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, y_matched, frame_rate)
                    temp_path = tmp.name
                
                replacement_audio = AudioSegment.from_wav(temp_path)
                os.unlink(temp_path)
                
                # Apply additional low-pass filter to make even thicker after envelope transfer
                try:
                    # VERY aggressive low-pass filters for MAXIMUM thickness (4x filters)
                    replacement_audio = replacement_audio.low_pass_filter(2000)  # Very strong filter
                    replacement_audio = replacement_audio.low_pass_filter(2500)  # Strong filter for thickness
                    replacement_audio = replacement_audio.low_pass_filter(3000)  # Additional thickness
                    replacement_audio = replacement_audio.low_pass_filter(3500)  # Smooth it out
                    
                    # Add bass boost by applying high-pass at very low frequency (keeps bass, removes only sub-bass)
                    # This enhances the low-mid frequencies that make voice sound thick
                    replacement_audio = replacement_audio.high_pass_filter(50)  # Keep bass frequencies
                    print(f"   🎚️ Applied original spectral envelope (98% blend - MAXIMUM THICKNESS) + 4x aggressive filters")
                except:
                    pass
        except Exception as e:
            print(f"   ⚠️ Spectral envelope transfer failed: {e}, using filter fallback")
            # Fallback: VERY aggressive low-pass to make MUCH thicker
            try:
                replacement_audio = replacement_audio.low_pass_filter(2000)  # Very strong filter
                replacement_audio = replacement_audio.low_pass_filter(2500)  # Strong filter
                replacement_audio = replacement_audio.low_pass_filter(3000)  # Additional thickness
                replacement_audio = replacement_audio.low_pass_filter(3500)  # Smooth it out
                # Keep bass frequencies for thickness
                replacement_audio = replacement_audio.high_pass_filter(50)  # Preserve bass
                print(f"   🔊 Applied VERY aggressive low-pass filters (4x) + bass boost to make sound MUCH thicker")
            except:
                pass
        
        # 5. Match formant characteristics (vowel quality) - helps with accent matching
        if orig_chars.get('spectral_rolloff') and repl_chars.get('spectral_rolloff'):
            rolloff_ratio = orig_chars['spectral_rolloff'] / repl_chars['spectral_rolloff']
            # Adjust to match frequency distribution - more aggressive
            if abs(rolloff_ratio - 1.0) > 0.05 and 0.6 < rolloff_ratio < 1.4:
                try:
                    # Apply more targeted filtering to match formant structure
                    if rolloff_ratio > 1.1:
                        # Original has much more high-frequency - apply stronger low-pass
                        replacement_audio = replacement_audio.low_pass_filter(4000)
                    elif rolloff_ratio > 1.0:
                        # Original has more high-frequency
                        replacement_audio = replacement_audio.low_pass_filter(5000)
                    elif rolloff_ratio < 0.9:
                        # Original has less high-frequency - apply high-pass
                        replacement_audio = replacement_audio.high_pass_filter(150)
                    else:
                        # Original has slightly less high-frequency
                        replacement_audio = replacement_audio.high_pass_filter(100)
                    print(f"   🎛️ Spectral rolloff matched: {rolloff_ratio:.2f}x")
                except Exception as e:
                    print(f"   ⚠️ Rolloff matching failed: {e}")
        
        # 5. Match dynamic range
        if orig_chars['dynamic_range'] > 0 and repl_chars['dynamic_range'] > 0:
            dr_ratio = orig_chars['dynamic_range'] / repl_chars['dynamic_range']
            if 0.5 < dr_ratio < 2.0:
                # Apply compression/expansion to match dynamic range
                # This is subtle - just normalize to similar range
                pass  # Normalize handles this
        
        # 6. GENDER-AWARE bass boost and thickness enhancement
        try:
            if orig_gender == "male":
                # For MALE voices: stronger low-pass + aggressive low-mid boost
                matched_audio_before_norm = replacement_audio.low_pass_filter(2500)  # Very strong for male
                matched_audio_before_norm = matched_audio_before_norm.low_pass_filter(2800)
                matched_audio_before_norm = matched_audio_before_norm.low_pass_filter(3000)
                # Stronger boost for male thickness
                matched_audio_before_norm = matched_audio_before_norm.apply_gain(4.0)
                print(f"   🎭 MALE VOICE: Applied aggressive bass boost (4.0dB) + 3x low-pass (≤2800Hz) for thick male voice")
            else:
                # For female/unknown: standard processing
                matched_audio_before_norm = replacement_audio.low_pass_filter(3500)
                matched_audio_before_norm = matched_audio_before_norm.low_pass_filter(4000)
                matched_audio_before_norm = matched_audio_before_norm.apply_gain(3.5)
                print(f"   🔊 Applied bass boost (3.5dB) + 2x low-pass filters for thickness")
        except:
            matched_audio_before_norm = replacement_audio
        
        # 6. Final normalization to prevent clipping while preserving matched characteristics
        matched_audio = normalize(matched_audio_before_norm)
        
        # Verify the match
        final_chars = analyze_audio_characteristics(matched_audio)
        if final_chars:
            print(f"   ✅ After:    Matched dB={final_chars['db']:.1f}, Pitch={final_chars.get('pitch', 'N/A'):.1f}Hz, Centroid={final_chars.get('spectral_centroid', 'N/A'):.0f}")
        
        return matched_audio
    except Exception as e:
        print(f"   ❌ Voice matching error: {e}")
        return replacement_audio

def detect_language_from_text(text):
    if not text:
        return "en"
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([c for c in text if c.isalnum()])
    if total_chars > 0 and (hindi_chars / total_chars) > 0.2:
        return "hi"
    return "en"

def convert_to_hindi_transliteration(name):
    """Convert English name to Hindi Devanagari for better TTS pronunciation
    
    CRITICAL: TTS will ONLY sound Hindi if input is Devanagari Unicode (U+0900-U+097F),
    NOT Roman letters. This function MUST return Devanagari for Hindi accent.
    """
    if not name:
        return name
    
    # Clean the name: strip whitespace and handle case
    name = name.strip()
    
    # If already Devanagari, return as-is
    if detect_language_from_text(name) == "hi":
        return name
    
    # Expanded transliteration map with common Indian names
    transliteration_map = {
        # First names
        'Sunil': 'सुनील', 'Suresh': 'सुरेश', 'Raja': 'राजा', 'Raj': 'राज',
        'Amit': 'अमित', 'Anil': 'अनिल', 'Ravi': 'रवि', 'Kumar': 'कुमार',
        'Vikram': 'विक्रम', 'Rahul': 'राहुल', 'Priya': 'प्रिया', 'Deepak': 'दीपक',
        'Arjun': 'अर्जुन', 'Karan': 'करण', 'Rohan': 'रोहन', 'Sohan': 'सोहन',
        'Mohan': 'मोहन', 'Gopal': 'गोपाल', 'Naresh': 'नरेश', 'Mahesh': 'महेश',
        'Rajesh': 'राजेश', 'Suresh': 'सुरेश', 'Dinesh': 'दिनेश', 'Ramesh': 'रमेश',
        'Nagendra': 'नगेंद्र', 'Venkatraman': 'वेंकटरमन', 'Venkat': 'वेंकट',
        'Sunil': 'सुनील', 'Surya': 'सूर्य', 'Shiva': 'शिव', 'Vishnu': 'विष्णु',
        'Ray': 'रे', 'Rey': 'रे', 'Rai': 'राय',
        
        # Last names
        'Sharma': 'शर्मा', 'Singh': 'सिंह', 'Patel': 'पटेल', 'Gupta': 'गुप्ता',
        'Mehta': 'मेहता', 'Joshi': 'जोशी', 'Reddy': 'रेड्डी', 'Rao': 'राव',
        'Iyer': 'अय्यर', 'Nair': 'नायर', 'Pillai': 'पिल्लई', 'Menon': 'मेनन',
        'Kumar': 'कुमार', 'Verma': 'वर्मा', 'Yadav': 'यादव', 'Jain': 'जैन',
        'Malhotra': 'मल्होत्रा', 'Kapoor': 'कपूर', 'Khanna': 'खन्ना', 'Bansal': 'बंसल'
    }
    
    # Direct match (exact)
    if name in transliteration_map:
        return transliteration_map[name]
    
    # Case-insensitive match (handles "suresh" -> "सुरेश")
    name_lower = name.lower().strip()
    for eng_name, hindi_name in transliteration_map.items():
        if name_lower == eng_name.lower():
            print(f"      ✅ Found transliteration: '{name}' -> '{hindi_name}'")
            return hindi_name
    
    # If name contains multiple words, transliterate each
    name_parts = name.split()
    if len(name_parts) > 1:
        transliterated_parts = []
        for part in name_parts:
            transliterated = convert_to_hindi_transliteration(part)
            transliterated_parts.append(transliterated)
        return ' '.join(transliterated_parts)
    
    # If no match found, try phonetic transliteration
    # Basic phonetic mapping for common sounds
    phonetic_map = {
        'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ee': 'ई', 'u': 'उ', 'oo': 'ऊ',
        'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
        'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ng': 'ङ',
        'ch': 'च', 'chh': 'छ', 'j': 'ज', 'jh': 'झ', 'ny': 'ञ',
        't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
        'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
        'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
        'sh': 'श', 'shh': 'ष', 's': 'स', 'h': 'ह'
    }
    
    # If no match found in map, log warning and return as-is
    # NOTE: This means TTS will receive English/Roman text and will sound English
    print(f"      ⚠️ TRANSLITERATION WARNING: '{name}' not found in transliteration map!")
    print(f"         TTS will receive English text and will sound English accent.")
    print(f"         Please add '{name}' to transliteration_map for Hindi pronunciation.")
    return name  # Return as-is - TTS will sound English if not Devanagari

def process_personalized_video(video_clip, original_audio_path, word_timestamps, names_to_replace, replacement_name, language="hi", master_profile=None):
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
    # CRITICAL: For Hindi, MUST use Devanagari for proper Hindi pronunciation (not English accent)
    # TTS will ONLY sound Hindi if input is Devanagari Unicode, NOT Roman letters
    if language == "hi":
        tts_voice = "onyx"  # Deeper, thicker voice for Hindi - MUST use onyx for Hindi
        # ALWAYS convert to Devanagari BEFORE TTS call
        tts_input = convert_to_hindi_transliteration(replacement_name)
        
        # VERIFY: Check if result is actually Devanagari (Unicode range U+0900-U+097F)
        is_devanagari = any('\u0900' <= char <= '\u097F' for char in tts_input) if tts_input else False
        
        print(f"   🔤 TRANSLITERATION CHECK:")
        print(f"      Original name (Roman): '{replacement_name}'")
        print(f"      TTS input (after transliteration): '{tts_input}'")
        print(f"      Is Devanagari Unicode: {is_devanagari}")
        if not is_devanagari:
            print(f"      ⚠️ WARNING: TTS input is NOT Devanagari! Will sound English!")
    else:
        tts_voice = "alloy"
        tts_input = replacement_name
        print(f"   🔤 TTS Input (English): '{tts_input}'")
    
    print(f"🔊 Generating TTS - Voice: '{tts_voice}', Input: '{tts_input}'")
    try:
        client = get_openai_client()
        # Use tts-1-hd for better quality and more natural sound
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=tts_voice,  # MUST be "onyx" for Hindi
            input=tts_input   # MUST be Devanagari Unicode for Hindi accent
        )
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            response.stream_to_file(temp_audio.name)
            tts_audio_path = temp_audio.name
        
        replacement_audio_seg = AudioSegment.from_mp3(tts_audio_path)
        replacement_duration_ms = len(replacement_audio_seg)
        
        # VERIFY: Check if TTS audio is actually valid (not empty/silent)
        if replacement_audio_seg.rms == 0 or replacement_duration_ms == 0:
            print(f"   ❌ ERROR: TTS audio is EMPTY or SILENT! RMS={replacement_audio_seg.rms}, Duration={replacement_duration_ms}ms")
            print(f"   ❌ TTS Input was: '{tts_input}'")
            return None
        
        print(f"✅ TTS generated: {replacement_duration_ms}ms, RMS={replacement_audio_seg.rms:.1f}")
        
        # CRITICAL: Apply ADULT MALE voice processing BEFORE matching
        # This ensures TTS sounds like adult (30-45 yrs), not child/teenager
        try:
            frame_rate = replacement_audio_seg.frame_rate
            
            # 1. SLOW DOWN SPEECH (Adult male pacing: 1.08x - 1.15x slower)
            # Adult males speak slower than children - this is CRITICAL for adult sound
            stretch_factor = 1.12  # 12% slower = adult male pacing
            print(f"   🎤 Slowing down TTS speech by {stretch_factor:.2f}x for adult male pacing")
            
            # Convert to numpy for time stretching
            samples = np.array(replacement_audio_seg.get_array_of_samples())
            if replacement_audio_seg.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            samples = samples.astype(np.float32)
            if max(abs(samples)) > 0:
                samples = samples / max(abs(samples))
            
            # Time stretch using librosa (preserves pitch)
            y_stretched = librosa.effects.time_stretch(samples, rate=1.0/stretch_factor)
            
            # 2. ADD SUBHARMONIC CHEST RESONANCE (80-120 Hz)
            # This simulates adult male chest vibration - CRITICAL for adult sound
            print(f"   🎤 Adding subharmonic chest resonance (80-120 Hz)")
            n_fft = 2048
            hop_length = 512
            stft = librosa.stft(y_stretched, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            freqs = librosa.fft_frequencies(sr=frame_rate, n_fft=n_fft)
            
            # Synthesize subharmonic energy at 80-120 Hz (chest resonance)
            for i, freq in enumerate(freqs):
                if 80 <= freq <= 120:
                    # Add 15% subharmonic energy (chest vibration)
                    magnitude[i, :] *= 1.15
                elif 500 <= freq <= 1200:
                    # LOWER FORMANTS: Reduce 500-1200 Hz by 12% for adult male
                    magnitude[i, :] *= 0.88
                elif 150 <= freq <= 400:
                    # Keep 150-400 Hz strong (already boosted)
                    pass
            
            # Reconstruct with subharmonic and lowered formants
            stft_processed = magnitude * np.exp(1j * phase)
            y_processed = librosa.istft(stft_processed, hop_length=hop_length)
            
            # 3. ADD MICRO GRIT / WARMTH (subtle saturation)
            # Remove "clean TTS" feel - add adult roughness
            print(f"   🎤 Adding subtle saturation/warmth for adult voice character")
            # Very light harmonic distortion (tanh saturation)
            saturation_amount = 0.08  # Very subtle
            y_processed = np.tanh(y_processed * (1 + saturation_amount)) / (1 + saturation_amount)
            
            # Normalize
            max_val = max(abs(y_processed)) if len(y_processed) > 0 else 0
            if max_val > 0:
                y_processed = y_processed / max_val * 0.92  # Slight headroom
            else:
                print(f"   ⚠️ WARNING: Processed audio is silent after processing! Using original TTS")
                raise Exception("Processed audio is silent")
            
            # VERIFY: Check if processed audio has any signal
            if max(abs(y_processed)) == 0:
                print(f"   ⚠️ WARNING: Processed audio is completely silent! Using original TTS")
                raise Exception("Processed audio is silent")
            
            # Save processed audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_adult:
                sf.write(tmp_adult.name, y_processed, frame_rate)
                replacement_audio_seg = AudioSegment.from_wav(tmp_adult.name)
                
                # VERIFY: Check if saved audio is valid
                if replacement_audio_seg.rms == 0 or len(replacement_audio_seg) == 0:
                    print(f"   ⚠️ WARNING: Saved processed audio is silent! Using original TTS")
                    os.unlink(tmp_adult.name)
                    raise Exception("Saved processed audio is silent")
                
                os.unlink(tmp_adult.name)
            
            print(f"   ✅ Adult male processing complete: Slowed {stretch_factor:.2f}x + Chest resonance + Lowered formants + Saturation")
            print(f"   ✅ Processed audio valid: RMS={replacement_audio_seg.rms:.1f}, Duration={len(replacement_audio_seg)}ms")
        except Exception as e:
            print(f"   ⚠️ Adult male processing failed: {e}, using original TTS")
        
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
        
        # CRITICAL: Use MASTER PROFILE for consistent voice matching
        # Do NOT analyze per-segment - use fixed master profile for all names
        matched_audio = match_voice_characteristics(replacement_audio_seg, original_segment=None, master_profile=master_profile)
        
        # VERIFY: Check if matched audio is valid (not empty/silent)
        if matched_audio is None or matched_audio.rms == 0 or len(matched_audio) == 0:
            print(f"   ❌ ERROR: Matched audio is EMPTY or SILENT! RMS={matched_audio.rms if matched_audio else 'None'}, Duration={len(matched_audio) if matched_audio else 0}ms")
            print(f"   ⚠️ Using original audio segment instead")
            # Use original segment instead
            name_video_seg = video_clip.subclip(start_time, end_time)
            final_clips.append(name_video_seg)
            last_end = end_time
            continue
        
        print(f"   ✅ Matched audio valid: RMS={matched_audio.rms:.1f}, Duration={len(matched_audio)}ms")
        
        # Save matched audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            matched_audio.export(temp_wav.name, format="wav")
            matched_audio_path = temp_wav.name
        
        # VERIFY: Check if audio file was saved and is readable
        if not os.path.exists(matched_audio_path) or os.path.getsize(matched_audio_path) == 0:
            print(f"   ❌ ERROR: Audio file not saved or is empty! Path: {matched_audio_path}")
            name_video_seg = video_clip.subclip(start_time, end_time)
            final_clips.append(name_video_seg)
            last_end = end_time
            continue
        
        try:
            new_audio_clip = AudioFileClip(matched_audio_path)
            tts_duration = new_audio_clip.duration
            
            # VERIFY: Check if audio clip is valid
            if tts_duration == 0 or new_audio_clip is None:
                print(f"   ❌ ERROR: AudioFileClip is invalid! Duration={tts_duration}s")
                name_video_seg = video_clip.subclip(start_time, end_time)
                final_clips.append(name_video_seg)
                last_end = end_time
                continue
            
            print(f"   ✅ AudioFileClip valid: Duration={tts_duration:.2f}s")
        except Exception as e:
            print(f"   ❌ ERROR: Failed to load AudioFileClip: {e}")
            name_video_seg = video_clip.subclip(start_time, end_time)
            final_clips.append(name_video_seg)
            last_end = end_time
            continue
        
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
        
        # CRITICAL: Create MASTER VOICE PROFILE once from entire original audio
        # This ensures ALL name replacements sound like the SAME person (consistent voice)
        master_voice_profile = create_master_voice_profile(original_audio_path)
        if not master_voice_profile:
            print(f"⚠️ WARNING: Master profile creation failed, using per-segment analysis (may cause variation)")
        
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
                    language=language,
                    master_profile=master_voice_profile  # CRITICAL: Pass master profile for consistency
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
                            # For Hindi: Convert to Devanagari for proper pronunciation
                            if language == "hi":
                                hindi_name = convert_to_hindi_transliteration(excel_name)
                                modified_transcript = modified_transcript.replace(name, hindi_name)
                                print(f"   🔤 Hindi transliteration in transcript: '{excel_name}' -> '{hindi_name}'")
                            else:
                                modified_transcript = modified_transcript.replace(name, excel_name)
                    
                    # CRITICAL: For Hindi, TTS input MUST be Devanagari Unicode
                    tts_voice_fallback = "onyx" if language == "hi" else "alloy"
                    
                    # VERIFY: Check if transcript contains Devanagari (for Hindi)
                    if language == "hi":
                        has_devanagari = any('\u0900' <= char <= '\u097F' for char in modified_transcript)
                        print(f"   🔤 FULL AUDIO TTS CHECK:")
                        print(f"      Original Excel name: '{excel_name}'")
                        print(f"      Modified transcript (first 100 chars): '{modified_transcript[:100]}...'")
                        print(f"      Contains Devanagari Unicode: {has_devanagari}")
                        if not has_devanagari:
                            print(f"      ⚠️ WARNING: Transcript does NOT contain Devanagari! Will sound English!")
                    
                    print(f"🔊 Generating full TTS audio - Voice: '{tts_voice_fallback}'")
                    audio_path = os.path.join(UPLOAD_FOLDER, f"temp_full_audio_{excel_name}.mp3")
                    # Use "onyx" for Hindi - deeper, thicker voice - MUST be onyx for Hindi
                    response = client.audio.speech.create(
                        model="tts-1-hd",
                        voice=tts_voice_fallback,  # MUST be "onyx" for Hindi
                        input=modified_transcript   # MUST contain Devanagari for Hindi accent
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
