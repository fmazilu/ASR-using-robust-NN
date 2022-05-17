RoDigits folder contains the MFCCs for train, val and test for each audio, but only the first 3 seconds or so of audio. Input size = 10000
RoDigits_split folder contains the MFCCs for train, val and test for each second of audio, so each second has a label, the parameters for mfcc are the defaults for librosa. Input size = 880
RoDigits_split folder contains the MFCCs for train, val and test for each second of audio, so each second has a label, the parameters for mfcc are n_fft=441 (approx. 20 ms of audio at 22kHz),
win_length=441, hop_length=220, so the overlapping between windows is 50%. Input size = 2020.
