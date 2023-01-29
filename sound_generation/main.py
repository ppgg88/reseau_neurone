"""demande à l'utilisateur de pronconcer des mot pour les enregistrer dans un fichier wav"""
import pyaudio
from keyboard import *
import wave
import numpy as np
import csv
import random

liste_fr = list(csv.reader(open('liste_francais.txt', 'r')))

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
    data_id = np.load('data_id.npy')
except :
    np.save('data_id.npy', 0)
    data_id = 0


j = 1
train_hugo = False

if train_hugo:
    while True:
        j = 1 ### train only hugo call
        frames = []
        if j<= 10:
            print("say : hugo")
        elif j<= 20:
            print("say : "+liste_fr[random.randint(0, len(liste_fr)-1)][0])
        else:
            j = 1
            print("say : hugo")                   
        while not is_pressed(' '):
            pass
        if is_pressed(' '):
            print("start recording...")
            seconds = 1
            for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
                data = stream.read(FRAMES_PER_BUFFER)
                frames.append(data)
            while is_pressed(' '):
                data = stream.read(FRAMES_PER_BUFFER)
                frames.pop(0) #on supprime le premier frame
                frames.append(data) #on ajoute le dernier frame
            
            #save audio file wav
            path = ''
            if j <= 10:
                path = 'hugo/data_h_'+str(data_id)+'.wav'
            else :
                path = 'other/data_a_'+str(data_id)+'.wav'
                
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
            print("start recording...")
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

