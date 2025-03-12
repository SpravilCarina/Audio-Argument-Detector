import librosa
import numpy as np
import json
import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import csv

def preprocess_audio(audio_path):
    """
    Preprocesarea fișierului audio: convertește MP3 în WAV, dacă este necesar.
    """
    audio = AudioSegment.from_file(audio_path)
    if audio_path.endswith('.mp3'):
        audio.export("temp_audio.wav", format="wav")
        return "temp_audio.wav"
    else:
        return audio_path

def detect_heated_moments(audio_path, threshold=0.02, duration=2):
    """
    Detectează momentele de intensitate ridicată în fișierul audio,
    presupunând că țipetele sau argumentele aprinse au o intensitate mare.
    """
    # Încărcăm audio
    y, sr = librosa.load(audio_path)
    
    # Calculăm energia semnalului audio
    energy = librosa.feature.rms(y=y)

    heated_moments = []

    # Verificăm secvențele cu energie mare (posibile țipete)
    frame_length = 2048
    hop_length = 512
    for i in range(0, len(energy[0]), int(sr * duration)):
        segment_energy = np.mean(energy[0][i:i + int(sr * duration)])
        
        if segment_energy > threshold:
            timestamp = librosa.frames_to_time(i, sr=sr)
            heated_moments.append({
                'timestamp': timestamp,
                'energy': segment_energy
            })
    
    return heated_moments

def recognize_speech(audio_path):
    """
    Folosește Google Speech Recognition pentru a transcrie audio într-un text.
    """
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
    """
    Detectează argumentele/țipetele din fișierul audio și returnează informațiile relevante.
    """
    processed_audio = preprocess_audio(audio_path)
    heated_moments = detect_heated_moments(processed_audio)
    transcript = recognize_speech(processed_audio)
    
    result = {
        "heated_moments": heated_moments,
        "transcript": transcript
    }
    
    return result

def save_to_json(data, filename="heated_moments.json"):
    """
    Salvează rezultatele într-un fișier JSON.
    """
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)

def save_to_csv(data, filename="heated_moments.csv"):
    """
    Salvează rezultatele într-un fișier CSV.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Energy"])
        for moment in data["heated_moments"]:
            writer.writerow([moment['timestamp'], moment['energy']])

if __name__ == "__main__":
    # Path-ul către fișierul audio pe care vrei să-l analizezi
    audio_file = "path_to_audio_file.wav"  # Înlocuiește cu calea către fișierul tău audio
    
    # Detectăm argumentele/țipetele
    output = detect_audio_arguments(audio_file)
    
    # Afișăm rezultatele în consolă
    print(json.dumps(output, indent=4))
    
    # Salvăm rezultatele în fișiere JSON și CSV
    save_to_json(output)
    save_to_csv(output)

    # Opțional: Generăm un grafic al energiei semnalului
    y, sr = librosa.load(audio_file)
    energy = librosa.feature.rms(y=y)
    plt.figure(figsize=(10, 6))
    plt.plot(librosa.times_like(energy), energy[0], label="Energy")
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.title('Audio Energy Over Time')
    plt.legend()
    plt.show()
