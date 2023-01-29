import pyaudio
import wave
from data import *
from index import DeepNeuralNetwork
import struct


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

TREASHOLD = 5000

p = pyaudio.PyAudio()
stream = p.open(            
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

network = DeepNeuralNetwork.self_load('model.hgo')

layer = ['hugo', 'autre']

print("vous pouvez parler :")
while True:
    frame = []
    level = 0
    while level < TREASHOLD:
        data_s = stream.read(FRAMES_PER_BUFFER)
        level = get_max_level(data_s)
    seconds = 1
    frame.append(data_s)
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frame.append(data)
    
    with wave.open('curent.wav', 'wb') as wav_file:
        # Définissez les paramètres du fichier wave
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(RATE)
        # Écrivez les données audio dans le fichier wave
        wav_file.writeframes(b''.join(frame))
    
    data = wav_to_mfcc('curent.wav')[2]
    
    x = np.array([data])
    x = x.T
    x_use = x.reshape(-1, x.shape[-1])/x.max()
    y = network.predict(x_use)
    
    if y[0] == True:
        print("hugo")
    else:
        print("autre")