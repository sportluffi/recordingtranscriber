# Recording Transcriber

This project uses Whisper and pyannote to convert speech to text. The transcript includes speakers.

The project will automatically convert .m4a files into .wav files.

## How to use the project

- Install all python requirements with `pip3 install -r requirements.txt`
- Install ffmpeg using brew. On MacOS it can be done by executing `brew install ffmpeg`
- Place recording into `data/recordings`
- Call script with `python3 transcribe.py -f <recording filename.wav>` 
- Collect the transcription in the `data/transcript` folder

# Acknowledgements

This script is heavily inspired by
[this colab](https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing#scrollTo=O0_tup8RAyBy).
I however wanted to execute the transcription on local hardware, thus converting the colab workbook into a small python 
script.