from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.regularizers import l2
def build_vgg(img_rows:int=224, img_cols:int=224, num_classes:int =1000):
    vgg = Sequential()
    vgg.add(Conv2D(64, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005), input_shape=(img_rows, img_cols, 3)))
    vgg.add(Conv2D(64, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005), input_shape=(img_rows, img_cols, 3)))
    vgg.add(MaxPooling2D()) # initial size /2
    vgg.add(Conv2D(128, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(Conv2D(128, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(MaxPooling2D()) # initial size /4
    vgg.add(Conv2D(256, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(Conv2D(256, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(Conv2D(256, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(MaxPooling2D()) # initial size /8
    vgg.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(MaxPooling2D()) # initial size /16
    vgg.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same', kernel_regularizer=l2(0.0005)))
    vgg.add(MaxPooling2D()) # initial size /32
    vgg.add(Flatten())
    vgg.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)))
    vgg.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)))
    vgg.add(Dropout(0.5))
    vgg.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0005)))

    return vgg
