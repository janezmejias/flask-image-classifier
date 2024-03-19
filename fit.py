import os
import numpy as np
from keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Rutas al conjunto de datos
train_dir = '/home/janez/Desktop/screen/train'
validation_dir = '/home/janez/Desktop/screen/validation'

# Preprocesamiento y Aumento de Datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # Ajuste para clasificación multiclase
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  # Ajuste para clasificación multiclase
)

# Implementar Transferencia de Aprendizaje con MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Congelar el modelo base

# Añadir nuevas capas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
predictions = layers.Dense(3, activation='softmax')(x)  # 3 clases

model = Model(inputs=base_model.input, outputs=predictions)

# Compilación del modelo con ajustes
model.compile(optimizer=Adam(learning_rate=1e-3),  # Tasa de aprendizaje inicial más alta
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback para ajustar la tasa de aprendizaje
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Entrenamiento del modelo con callback
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Ajustar según el tamaño de tu conjunto de datos
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,  # Ajustar según el tamaño de tu conjunto de datos
    callbacks=[reduce_lr]  # Incluir el callback aquí
)

# Guardar el modelo entrenado
model_path = '/home/janez/Desktop/screen/fit_categorical_model.h5'
model.save(model_path)
print(f"Model saved in {model_path}")