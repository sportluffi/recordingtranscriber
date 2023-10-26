
import os
import torch
import whisper
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import contextlib
import wave
from pyannote.core import Segment
from pyannote.audio import Audio
import datetime
import subprocess
from pyannote.audio import Pipeline
from pathlib import Path
import argparse
import logging

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb")

# Configure args
parser = argparse.ArgumentParser("recordingtranscriber")
parser.add_argument("-r", "--recording", help="The filename that is accessed in the data directory and processed.", type=str, default="recording.m4a")
parser.add_argument("-d", "--data_folder", help="The folder holding recording and transcript subfolders", type=str, default="data/")
args = parser.parse_args()


recording_name = args.recording
recording_file = os.getcwd() + '/' + args.data_folder + 'recording/' + recording_name


# Configure model
num_speakers = 2  # @param {type:"integer"}
language = 'any'  # @param ['any', 'English']
model_size = 'large'  # @param ['tiny', 'base', 'small', 'medium', 'large']

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


# Convert file to correct format
if Path(recording_file).suffix != 'wav':
    logging.info('Converting input file to .wav')
    target_file = os.getcwd() + '/data/recording/' + Path(recording_file).stem + '.wav'
    subprocess.call(['ffmpeg', '-i', recording_file, target_file, '-y'])
    logging.info('File converted to ' + target_file)
    recording_file = target_file


# Found at https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing#scrollTo=O0_tup8RAyBy
model = whisper.load_model(model_size)

logging.debug('Loaded model ' + model_size)
logging.info('Transcribing...')

result = model.transcribe(recording_file)

logging.info('Transcribing done.')
logging.debug('Applying segments...')
segments = result["segments"]

with contextlib.closing(wave.open(recording_file, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

audio = Audio()


def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(recording_file, clip)
    return embedding_model(waveform[None])


embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)


def time(secs):
    return datetime.timedelta(seconds=round(secs))


transcript_path = os.getcwd() + '/' + args.data_folder + 'transcript/' + Path(args.filename).stem + '.txt'


f = open(transcript_path, "w")

for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
        f.write("\n" + segment["speaker"] + ' ' +
                str(time(segment["start"])) + '\n')
    f.write(segment["text"][1:] + ' ')
f.close()
