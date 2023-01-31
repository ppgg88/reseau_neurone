from index import DeepNeuralNetwork
from nltk.stem import WordNetLemmatizer 
import nltk
import json
import random
import numpy as np
import string

lemmatizer = WordNetLemmatizer()

nltk.download("punkt")
nltk.download("wordnet")

da = open("var/data.json", "r", encoding="utf8")
data = json.load(da)
da.close()

def data_train_preprocessing():
    # création des listes
    words = []
    classes = []
    doc_X = []
    doc_y = []

    # parcourir avec une boucle For toutes les intentions
    # tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
    # le tag associé à l'intention sont ajoutés aux listes correspondantes
    for intent in data["intents"]:
        try:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)
                doc_X.append(pattern)
                doc_y.append(intent["tag"])
        except:
            print(intent)
            break
        
        # ajouter le tag aux classes s'il n'est pas déjà là 
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    
    print("doc_X: ", doc_X)
    print("doc_y: ", doc_y)
    # lemmatiser tous les mots du vocabulaire et les convertir en minuscule
    # si les mots n'apparaissent pas dans la ponctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    # trier le vocabulaire et les classes par ordre alphabétique et prendre le
    # set pour s'assurer qu'il n'y a pas de doublons
    words = sorted(set(words))
    classes = sorted(set(classes))
    # liste pour les données d'entraînement
    training = []
    out_empty = [0] * len(classes)
    # création du modèle d'ensemble de mots
    train_X = []
    train_y = []
    for idx, doc in enumerate(doc_X):
        train_X.append(bag_of_words(doc, words))
        # marque l'index de la classe à laquelle le pattern atguel est associé à
        output_row = list(out_empty)
        output_row[classes.index(doc_y[idx])] = 1
        # ajoute le one hot encoded BoW et les classes associées à la liste training
        train_y.append(output_row)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    return train_X, train_y, words, classes


def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens: 
        for idx, word in enumerate(vocab):
            if word == w: 
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    x = []
    for idx, doc in enumerate(text):
        x.append(bag_of_words(doc, vocab))
    x = np.array(x).T
    x = x.reshape(-1, x.shape[-1])
    result = model.predict(x)
    print(labels[np.where(result == np.amax(result))[0][0]])
    #print("proba : " + str(float(max(probabilities))))


    thresh = 0.5
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list):
    tag = intents_list[0]
    for i in data["intents"]: 
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


if __name__ == "__main__":
    train_x_p, train_y_p, words, classes = data_train_preprocessing()
    train_x = train_x_p.T
    train_x = train_x.reshape(-1, train_x.shape[-1])
    train_y = train_y_p.T
    model = DeepNeuralNetwork(train_x, train_y, hidden_layers=[64, 32, 16])
    model.training(train_x, train_y,nb_iter=5000, learning_rate=0.001)
    model.save("var/chat.hgo")
    #model = DeepNeuralNetwork.self_load("var/chat.hgo")
    print("model is ready to use")
    print(classes)
    pred = pred_class(["qu'est-ce que tu es en train de faire"], words, classes)
    print("----------------------------------------")
    pred = pred_class(["comment te sens tu"], words, classes)
    print("----------------------------------------")
    pred = pred_class(["Quel est ton acteur préféré"], words, classes)
    print("----------------------------------------")
    
    