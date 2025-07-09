from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2, augment=False):
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=val_split,
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42,
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42,
    )
    return train_gen, val_gen
