from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input, Activation, BatchNormalization
from keras import Model
from BatchNorm import CustomBatchNorm


def build_lenet(batch_norm=None):
    x_in = Input((28, 28, 1))
    x = Conv2D(filters=6, kernel_size=(3, 3))(x_in)
    if batch_norm: x = batch_norm()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    x = Conv2D(filters=16, kernel_size=(3, 3))(x)
    if batch_norm: x = batch_norm()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=120)(x)
    if batch_norm: x = batch_norm()(x)
    x = Activation('relu')(x)
    x = Dense(units=84)(x)
    if batch_norm: x = batch_norm()(x)
    x = Activation('relu')(x)
    x = Dense(units=10, activation='softmax')(x)
    return Model(inputs=x_in, outputs=x, name='LeNet')

def get_models():
    models = [build_lenet(CustomBatchNorm), build_lenet(BatchNormalization), build_lenet()]
    for model in models: model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return models
