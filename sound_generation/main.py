"""demande à l'utilisateur de pronconcer des mot pour les enregistrer dans un fichier wav"""
import pyaudio
from keyboard import *
import wave
import numpy as np
import csv
import random
import struct
from os import listdir


def get_max_level(data_s):
    # Décompactez les données audio en entiers signés de 16 bits
    values = struct.unpack("<{}h".format(len(data_s) // 2), data_s)
    # Renvoyez le maximum des valeurs
    return max(values)

liste_fr = list(csv.reader(open('sound_generation/liste_francais.txt', 'r')))

# set up audio
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
p = pyaudio.PyAudio()
stream = p.open(            
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

try :
    data_id = np.load('sound_generation/data_id.npy')
except :
    np.save('sound_generation/data_id.npy', 0)
    data_id = 0


j = 1
train_hugo = True

if train_hugo:
    while True:
        #j = 1 ### train only hugo call
        #j=15 ### train only other call
        frames = []
        if j<= 10:
            print("say : hugo")
        elif j<= 20:
            print("say : "+liste_fr[random.randint(0, len(liste_fr)-1)][0])
        else:
            j = 1
            print("say : hugo") 
        level = 0
        THRESHOLD = 10000                  
        while level < THRESHOLD:
            data_s = stream.read(FRAMES_PER_BUFFER)
            level = get_max_level(data_s)

        print("start recording...")
        seconds = 1
        frames.append(data_s)
        for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
            data = stream.read(FRAMES_PER_BUFFER)
            frames.append(data)

        #save audio file wav
        path = ''
        if j <= 10:
            while ('data_h_'+str(data_id)+'.wav') in listdir('datasets/hugo'):
                data_id += 1
                np.save('data_id.npy', data_id)
            path = 'datasets/hugo/data_c_h_'+str(data_id)+'.wav'
        else :
            while ('data_a_'+str(data_id)+'.wav') in listdir('datasets/other'):
                data_id += 1
                np.save('data_id.npy', data_id)
            path = 'datasets/other/data_c_a_'+str(data_id)+'.wav'
        
        with wave.open(path, 'wb') as wav_file:
            # Définissez les paramètres du fichier wave
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(RATE)
            # Écrivez les données audio dans le fichier wave
            wav_file.writeframes(b''.join(frames))
            data_id += 1
            np.save('data_id.npy', data_id)
        print("stop recording...")
        j+=1

else : 
    while True:
        frames = []
        print("munissez vous d'un texte et lisez le")
        while not is_pressed(' '):
            pass
        while is_pressed(' '):
            seconds = 1
            frames = []
            for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
                data = stream.read(FRAMES_PER_BUFFER)
                frames.append(data)
                
            path = 'other/data_o_'+str(data_id)+'.wav'  
            with wave.open(path, 'wb') as wav_file:
                # Définissez les paramètres du fichier wave
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(RATE)
                # Écrivez les données audio dans le fichier wave
                wav_file.writeframes(b''.join(frames))
                data_id += 1
                np.save('data_id.npy', data_id)
        print("stop recording...")

