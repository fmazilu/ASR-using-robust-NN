from pydub import AudioSegment
import numpy as np
from pydub.utils import db_to_float
from pydub.silence import split_on_silence


PATH = 'dataset\\rodigits'

audio = AudioSegment.from_mp3('florin.mp3')

audio_chunks = split_on_silence(audio,
    # must be silent for at least half a second
    min_silence_len=300,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-60
)

for i, chunk in enumerate(audio_chunks):

    out_file = ".//splitAudioEn//chunk{0}.wav".format(i)
    print("exporting", out_file)
    chunk.export(out_file, format="wav")
