import keras
from keras import Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.optimizers import Adam


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))


class Classifier:
    def __init__(self, number_of_labels, data_size=4096):
        self.model = Sequential()

        self.model.add(BatchNormalization(input_shape=(data_size, )))
        self.model.add(Dense(128, activation='relu', input_dim=data_size))

        self.model.add(BatchNormalization())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(BatchNormalization())
        self.model.add(Dense(32, activation='relu'))

        self.model.add(BatchNormalization())
        self.model.add(Dense(number_of_labels, activation='sigmoid'))

        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_data, labels):
        loss_history = LossHistory()

        self.model.fit(
            x=train_data,
            y=labels,
            validation_split=0.1,
            shuffle=True,
            batch_size=64,
            epochs=3,
            callbacks=[loss_history]
        )

        return loss_history

    def test(self, test_data):
        return self.model.predict(test_data)
