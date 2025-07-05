import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

from util import modifica_imagem

model = None
probability_model = None
lista_imagens = None
label_idxs = None

# Caminho para salvar/carregar o modelo
CAMINHO_MODELO = "modelo_cifar10.h5"

# Rótulos das classes
class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'veado',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

def carregar_modelo(caminho=CAMINHO_MODELO):
    """Carrega um modelo salvo"""
    global model, probability_model
    if os.path.exists(caminho):
        print(f"Carregando modelo salvo de: {caminho}")
        model = tf.keras.models.load_model(caminho)
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    else:
        raise FileNotFoundError("Modelo salvo não encontrado.")



def carrega_dataset():
    global lista_imagens, label_idxs
    (_,_),(lista_imagens, label_idxs) = tf.keras.datasets.cifar10.load_data()
    lista_imagens = lista_imagens / 255.0

def fitness(individuo, img_idx):
    global lista_imagens, label_idxs

    if lista_imagens is None or label_idxs is None:
        carrega_dataset()

    # img = lista_imagens[img_idx]
    # true_label = label_idxs[img_idx][0]

    # img_modificada = modifica_imagem(individuo, img)

    # pred_probs = probability_model.predict(np.expand_dims(img_modificada, axis=0), verbose=0)
    # pred_label = np.argmax(pred_probs)

    pred_probs, pred_label, true_label, _ = analisa_img_individuo(individuo,img_idx)

    conf_pred = pred_probs[0][pred_label]
    conf_true = pred_probs[0][true_label]

    if pred_label != true_label:
        # ataque deu certo
        return float(conf_pred - conf_true)
        
    else:
        # ataque deu errado
        return float(-conf_true)
    
def analisa_img_individuo(individuo,img_idx):
    img = lista_imagens[img_idx]
    true_label = label_idxs[img_idx][0]

    img_modificada_normalizada, img_modificada = modifica_imagem(individuo, img)

    pred_probs = probability_model.predict(np.expand_dims(img_modificada_normalizada, axis=0), verbose=0)
    pred_label = np.argmax(pred_probs)

    return pred_probs, pred_label, true_label, img_modificada

    
def mostra_imagem_alterada(individuo, img_idx):

    pred_probs, pred_label, true_label, img_modificada = analisa_img_individuo(individuo,img_idx)

    plt.xlabel(
                f"Real: {class_names[true_label]}\n" +
                f"Predito: {class_names[pred_label]}\n" + 
                f"Com certeza: {pred_probs[0][pred_label]:.4f}",
                color='green'
            )

    plt.imshow(img_modificada)
    plt.tight_layout()
    plt.show()

