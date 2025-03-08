from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Bidirectional, LSTM

def CNN_BISLTM():
    num_classes = 4
    input_shape_data = 3000
    model = Sequential()
    

    model.add(Conv1D(8, 10, activation='relu', input_shape=(input_shape_data, 1)))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(16, 10, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(32, 10, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(128,return_sequences = True)))
    model.add(Bidirectional(LSTM(128,return_sequences = False)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='softmax'))

    
    #model.fit(training_features, training_labels, batch_size=32, epochs=3, validation_split=0.1)

    return model