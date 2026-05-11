import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Caminhos
treino_dir = "dataset/treino"
validacao_dir = "dataset/validacao"

# Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Dataset treino
train_generator = train_datagen.flow_from_directory(
    treino_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical'
)

# Dataset validação
validation_generator = val_datagen.flow_from_directory(
    validacao_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical'
)

# Base pré-treinada
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Congelar pesos
base_model.trainable = False

# Modelo final
model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(4, activation='softmax')
])

# Compilar
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'modelo_lesao_lpp.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# Treinamento
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint]
)

print("Modelo treinado com sucesso!")