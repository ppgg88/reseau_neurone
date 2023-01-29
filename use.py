import pyaudio
import wave
from data import *
from index import DeepNeuralNetwork
import struct
import speech_recognition as sr
import time


def get_max_level(data_s):
    # Décompactez les données audio en entiers signés de 16 bits
    values = struct.unpack("<{}h".format(len(data_s) // 2), data_s)
    # Renvoyez le maximum des valeurs
    return max(values)
    

# set up audio
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

p = pyaudio.PyAudio()

def set_ambiance_level():
    stream = p.open(            
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )
    level = []
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * 5)):
        data_s = stream.read(FRAMES_PER_BUFFER)
        level.append(get_max_level(data_s))
    stream.stop_stream()
    return (sum(level)/len(level))+700

TREASHOLD = set_ambiance_level()
print('TREASHOLD : ', TREASHOLD)

network = DeepNeuralNetwork.self_load('model.hgo')
network.set_threshold(0.7)

layer = ['hugo', 'autre']

print("vous pouvez parler :")
while True:
    stream = p.open(            
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )
    frame = []
    level = 0
    while level < TREASHOLD:
        data_s = stream.read(FRAMES_PER_BUFFER)
        level = get_max_level(data_s)
    seconds = 0.5
    frame.append(data_s)
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frame.append(data)
    
    x = frame.copy()
    last_time = time.time()
    while level > TREASHOLD or (time.time() - last_time) < 1 :
        data_s = stream.read(FRAMES_PER_BUFFER)
        level = get_max_level(data_s)
        print(level)
        if level > TREASHOLD:
            last_time = time.time()
        x.append(data_s)
    
    stream.stop_stream()
    
    with wave.open('curent.wav', 'wb') as wav_file:
        # Définissez les paramètres du fichier wave
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(RATE)
        # Écrivez les données audio dans le fichier wave
        wav_file.writeframes(b''.join(frame))
    
    with wave.open('x.wav', 'wb') as wav_file:
        # Définissez les paramètres du fichier wave
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(RATE)
        # Écrivez les données audio dans le fichier wave
        wav_file.writeframes(b''.join(x))
    
    data = wav_to_mfcc('curent.wav')[2]
    
    x = np.array([data])
    x = x.T
    x_use = x.reshape(-1, x.shape[-1])/x.max()
    y = network.predict(x_use)
    
    if y[0] == True:
        r = sr.Recognizer()
        audio_wav = sr.AudioFile('x.wav')
        with audio_wav as source:
            audio = r.listen(source)
        try:
            say = r.recognize_google(audio, language='fr-FR')
            print("you said : ",say)
        except:
            print("i can't understand you")
    else:
        print('no hugo call')