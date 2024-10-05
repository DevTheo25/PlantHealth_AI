import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import os


def load_images_from_directory(directory, target_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))
    class_indices = {class_name: index for index, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_indices[class_name])
            except Exception as e:
                print(f"Erro ao carregar a imagem {img_path}: {e}")
    
    images = np.array(images, dtype='float32') / 255.0  # Normalizar as imagens
    labels = np.array(labels)
    return images, labels, class_indices




def train_model():
    train_dir = '../data/processed/plantvillage dataset/train'
    val_dir = '../data/processed/plantvillage dataset/val'

    # Carregando os dados de treinamento
    X_train, y_train, class_indices = load_images_from_directory(train_dir, target_size=(224, 224))

    # Carregando os dados de validação
    X_val, y_val, _ = load_images_from_directory(val_dir, target_size=(224, 224))

    num_classes = len(class_indices)
    print(f"Número de classes: {num_classes}")
    print(f"Classes: {class_indices}")

    # Convertendo os rótulos para one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    # Construindo o modelo
    model = build_cnn_model(input_shape=(224, 224, 3), num_classes=num_classes)

    # Compilando o modelo
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Definindo callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        filepath='../models/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Treinando o modelo
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Salvando a história do treinamento
    np.save('../models/history.npy', history.history)

    print("Treinamento concluído.")

if __name__ == "__main__":
    train_model()
