import numpy as np
import random
import matplotlib.pyplot as plt

individuo_valido = {
        "width": (0, 31),
        "height":(0, 31),
        "r": (0, 255),
        "g": (0, 255),
        "b": (0, 255)
    }

def cria_individuo():
    return np.array(
        [np.random.randint(low, higth+1) for low, higth in individuo_valido.values() ]
    )
   
def seleciona_3_individuos(individuo_ignorado, populacao):
    lista = list(populacao)

    for i, _ in enumerate(lista):
        if np.array_equal(lista[i], individuo_ignorado):
            del lista[i]
            break
        
    return random.sample(lista, 3)

def verificar_limites(individuo):
    return np.clip(individuo, [low for low,_ in individuo_valido.values() ], 
                   [higth for _,higth in individuo_valido.values() ])
     
def combinar(ind, mutante, CR ):
    trial = np.copy(ind)

    mutacao_garantida = np.random.randint(0, len(ind))

    for j in range( len(ind) ):
        if np.random.rand() <= CR or j == mutacao_garantida:
            trial[j] = mutante[j]

    return trial

def modifica_imagem(individuo, imagem):
    x, y, r, g, b = individuo
    img_modificada_normal = imagem.copy()
    img_modificada_normal[y,x] = [r/255.0,g/255.0,b/255.0]
    img_modificada = imagem.copy()
    img_modificada[y,x] = [r,g,b]
    return img_modificada_normal, img_modificada

