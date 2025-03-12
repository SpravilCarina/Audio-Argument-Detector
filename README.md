# Audio Argument Detector

Această aplicație analizează fișiere audio (de exemplu, WAV, MP3) pentru a detecta momentele când o conversație devine intensă (de exemplu, țipete, argumente aprinse sau vorbire suprapusă).

## Funcționalități:
- Analizează un fișier audio.
- Detectează momentele cu intensitate mare a sunetului (știri, țipete).
- Îți returnează timpii (timestamps) în care se întâmplă aceste momente.
- Poate utiliza recunoașterea vorbirii pentru a extrage text din audio.

## Instalare:
1. Clonează acest depozit:
    ```bash
    git clone https://github.com/username/audio-argument-detector.git
    ```

2. Instalează dependențele necesare:
    ```bash
    pip install -r requirements.txt
    ```

3. Rulează aplicația:
    ```bash
    python detect_argument.py path_to_audio_file.wav
    ```

## Dependențe:
- pydub
- librosa
- numpy
- speechrecognition
- matplotlib
