import librosa
import numpy as np
import json
from pydub import AudioSegment
import speech_recognition as sr

def preprocess_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    if audio_path.endswith('.mp3'):
        audio.export("temp_audio.wav", format="wav")
        return "temp_audio.wav"
    else:
        return audio_path

def detect_heated_moments(audio_path, threshold=-20, duration=2):
    y, sr = librosa.load(audio_path)
    hop_length = 512
    frame_length = 2048
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    heated_moments = []
    
    for i in range(0, len(energy[0]), int(sr * duration)):
        segment_energy = np.mean(energy[0][i:i + int(sr * duration)])
        if segment_energy > threshold:
            timestamp = librosa.frames_to_time(i, sr=sr)
            heated_moments.append(timestamp)
    
    return heated_moments

def recognize_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def detect_audio_arguments(audio_path):
    processed_audio = preprocess_audio(audio_path)
    heated_moments = detect_heated_moments(processed_audio)
    transcript = recognize_speech(processed_audio)
    
    result = {
        "heated_moments": heated_moments,
        "transcript": transcript
    }
    
    return json.dumps(result, indent=4)

if __name__ == "__main__":
    audio_file = "path_to_audio_file.wav"  # Replace with the actual file path
    output = detect_audio_arguments(audio_file)
    print(output)
