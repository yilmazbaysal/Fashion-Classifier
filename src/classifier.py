from keras import Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.optimizers import Adam


class Classifier:
    def __init__(self, number_of_labels, data_size=4096):
        self.model = Sequential()

        self.model.add(BatchNormalization(input_shape=(4096,)))
        self.model.add(Dense(128, activation='relu', input_dim=4096))

        self.model.add(BatchNormalization())
        self.model.add(Dense(64, activation='relu'))

        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='relu'))

        self.model.add(BatchNormalization())
        self.model.add(Dense(number_of_labels, activation='sigmoid'))

        self.model.compile(optimizer=Adam(), loss='binary_crossentropy')

    def train(self, train_data, labels):
        self.model.fit(train_data, labels, shuffle=True, batch_size=64, epochs=1)

    def test(self, test_data):
        return self.model.predict(test_data)
