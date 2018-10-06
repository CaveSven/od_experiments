from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import SGD, Adam

from src.models.vgg.vgg19 import build_vgg
from datetime import datetime

IMG_ROWS = 64
IMG_COLS = 64
NUM_CLASSES = 200

BATCH_SIZE=256

tb_callback = TensorBoard(log_dir=f"../../logs/{datetime.now().isoformat()}", histogram_freq=0, write_graph=True, write_images=False)
mc_callback = ModelCheckpoint("../../models/vgg11-{epoch:02d}-{val_acc:.2f}.hdf5", monitor="val_loss")
lr_callback = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

data_generator_with_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input, 
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        )
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = data_generator_with_aug.flow_from_directory(
        "../../data/raw/tiny-imagenet-200/train",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        "../../data/raw/tiny-imagenet-200/val/images",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

vgg = build_vgg(img_rows=IMG_ROWS, img_cols=IMG_COLS, num_classes=NUM_CLASSES)

optimizer_sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
optimizer_adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

vgg.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer_sgd, metrics=['accuracy'])

vgg.fit_generator(
        train_generator,
        steps_per_epoch=390,
        validation_data=validation_generator,
        epochs=150,
        validation_steps=39,
        callbacks=[tb_callback, mc_callback, lr_callback])
