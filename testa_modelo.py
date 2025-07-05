import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Caminho para salvar/carregar o modelo
CAMINHO_MODELO = "modelo_cifar10.h5"

# Rótulos das classes
class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'veado',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']


def construir_modelo():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model


def carregar_modelo(caminho=CAMINHO_MODELO):
    """Carrega um modelo salvo"""
    if os.path.exists(caminho):
        print(f"Carregando modelo salvo de: {caminho}")
        model = tf.keras.models.load_model(caminho)

    else:
        raise FileNotFoundError("Modelo salvo não encontrado.")
    return model


# Carregar o dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Se o modelo já estiver salvo, carregue. Senão, treine e salve.
if os.path.exists(CAMINHO_MODELO):
    model = carregar_modelo()
else:
    model = construir_modelo()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save(CAMINHO_MODELO)
    print(f"Modelo salvo em: {CAMINHO_MODELO}")

# Avaliação
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nAcurácia no teste: {test_acc:.4f}")

# Converter para probabilidade
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Mostrar 5 imagens com predição
num_images = 6
#indices = np.random.choice(len(x_test), num_images, replace=False)
indices = [43,222,980,987,985,321]
plt.figure(figsize=(10, 5))

for i, idx in enumerate(indices):
    img = x_test[idx]
    true_label = y_test[idx][0]
    pred_probs = probability_model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_label = np.argmax(pred_probs)
    print("\n")
    print(pred_probs) # debug
    print("\n")


    plt.subplot(1, num_images, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(f"Pred: {class_names[pred_label]}\nReal: {class_names[true_label]}",
               color='green' if pred_label == true_label else 'red')

plt.tight_layout()
plt.show()

