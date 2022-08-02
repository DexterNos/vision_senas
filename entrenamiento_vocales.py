from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator

datos_entrenamiento = r"C:\Users\jonba\Documents\Maestria\Vocales y audio\Entrenamiento vocales\Entrenamiento Colab"
datos_validacion = r"C:\Users\jonba\Documents\Maestria\Vocales y audio\Validacion vocales\Validacion Colab"

prepros_entre = ImageDataGenerator(rescale=1./255,  rotation_range=20, zoom_range=0.3, horizontal_flip=True)
prepros_val = ImageDataGenerator(rescale=1./255 )

data_train = prepros_entre.flow_from_directory(datos_entrenamiento, 
                                               target_size=(200,200), 
                                               class_mode='categorical',
                                               batch_size=32)

data_val = prepros_val.flow_from_directory(datos_validacion,
                                           target_size=(200,200), 
                                               class_mode='categorical',
                                               batch_size=32)

inputs = keras.Input(shape = (200,200,3))
x = layers.Conv2D(filters=32, kernel_size=4, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
#model.summary()

model.compile(optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'])

model.fit(data_train, validation_data=data_val, epochs=10, batch_size=32)

model.save('Modelo_vocales_colab_30_07_2022.h5')
model.save_weights('Pesos_vocales_colab_30_07_2022.h5')