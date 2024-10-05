from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential()


    # Arquitetura:
    
        # Camadas Convolucionais: Extraem características das imagens.
        
        # BatchNormalization: Normaliza as ativações para acelerar o treinamento.

        # MaxPooling2D: Reduz a dimensionalidade espacial.

        # Dropout: Regularização para evitar overfitting.

        # Flatten: Converte a saída das camadas convolucionais em um vetor unidimensional.

        # Camadas Densas: Realizam a classificação baseada nas características extraídas.
        
        # Camada de Saída com Softmax: Produz uma distribuição de probabilidade sobre as classes.


    # Definindo a camada de entrada
    model.add(Input(shape=input_shape))

    # Primeira camada convolucional
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Segunda camada convolucional
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Terceira camada convolucional
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Camada de flattening
    model.add(Flatten())

    # Camada totalmente conectada
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Camada de saída
    model.add(Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
