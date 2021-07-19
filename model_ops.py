from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_ops import get_test_gen, get_train_gen


def train_model(model, datagen, x_train, y_train):
    return model.fit(datagen.flow(x_train, y_train, batch_size=256,
                                  subset='training'),
                     validation_data=datagen.flow(x_train, y_train,
                                                  batch_size=256, subset='validation'),
                     epochs=100, verbose=2,
                     callbacks=[EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=3)])


def train_models(models, data):
    x_train, y_train = data
    datagen = get_train_gen()
    datagen.fit(x_train)
    histories = []
    for m in models:
        histories.append(train_model(m, datagen, x_train, y_train))
    return histories


def test_model(model, datagen, x_test, y_test):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model.evaluate(datagen.flow(x_test, y_test, batch_size=256), verbose=1)


def test_models(models, data):
    x_test, y_test = data
    datagen = get_test_gen()
    datagen.fit(x_test)
    histories = []
    for m in models:
        histories.append(test_model(m, datagen, x_test, y_test))
    return histories
