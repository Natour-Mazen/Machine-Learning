import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Path to the data file
WANG_PATH = "./ressources/WangSignatures.xlsx"

# Classification categories
CATEGORIES = [
    'Jungle', 'Plage', 'Monuments', 'Bus', 'Dinosaures', 'Éléphants',
    'Fleurs', 'Chevaux', 'Montagne', 'Plats'
]

# Load data from the Excel file
WANG_DATA = pd.read_excel(WANG_PATH, sheet_name=0, index_col=0, header=None)


def load_descriptor(sheet=0, concatenate=True):
    """
    Load descriptors from the data file.
    """
    measures = []
    for i in range(5):
        if concatenate:
            if i == 0:
                measures = WANG_DATA.values
            else:
                data = WANG_DATA.values
                measures = np.concatenate((measures, data), axis=1)
        elif i == sheet:
            measures = WANG_DATA.values
    return measures


def prepare_data():
    """
    Prepare the data for training and testing.
    """
    measures_descriptor = load_descriptor()
    Labels = np.array([int(nom[0:-4]) // 100 for nom in WANG_DATA.index])
    Target = np.array([np.eye(10)[Labels[i]] for i in range(len(Labels))])
    X_train, X_test, Y_train, Y_test = train_test_split(
        measures_descriptor, Target, test_size=0.2, random_state=1, stratify=Labels
    )
    return X_train, X_test, Y_train, Y_test, Labels, Target


def build_model():
    """
    Build and compile the sequential model.
    """
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train):
    """
    Train the model.
    """
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)
    return history


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model on the test set.
    """
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    Y_predict = model.predict(X_test, verbose=0)
    return test_loss, test_acc, Y_predict


def display_results(history, test_loss, test_acc, Y_test, Y_predict):
    """
    Display the training loss, accuracy curves, and confusion matrix.
    """
    display_loss_accuracy(history=history, test_loss=test_loss, test_acc=test_acc)
    display_confusion_matrix(Y_test=Y_test, Y_predict=Y_predict)


def display_loss_accuracy(name='', history=None, test_loss=None, test_acc=None):
    """
    Display training loss and accuracy curves.
    """
    loss = round(test_loss, 2)
    acc = round(test_acc * 100, 2)
    title_01 = f"Loss: {loss}"
    title_02 = f"Accuracy: {acc}"
    if name:
        title_01 = f"{name} - " + title_01
        title_02 = f"{name} - " + title_02
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(title_01, fontsize=10)
    ax1.plot(history.history['loss'], 'r--', label='Loss of training data')
    ax1.plot(history.history['val_loss'], 'r', label='Loss of validation data')
    ax1.legend()
    ax2.set_title(title_02, fontsize=10)
    ax2.plot(history.history['accuracy'], 'g--', label='Accuracy of training data')
    ax2.plot(history.history['val_accuracy'], 'g', label='Accuracy of validation data')
    ax2.legend()
    plt.show()


def display_confusion_matrix(name='', Y_test=None, Y_predict=None):
    """
    Display the confusion matrix of the model.
    """
    title = 'Confusion matrix of the model'
    if name:
        title = f"{name} - " + title
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            np.argmax(Y_test, axis=1),
            np.argmax(Y_predict, axis=1)
        ),
        display_labels=[c[0] for c in CATEGORIES]
    ).plot()
    plt.title(title)
    plt.show()


def main():
    """
    Main function to run the entire process.
    """
    X_train, X_test, Y_train, Y_test, Labels, Target = prepare_data()
    model = build_model()
    history = train_model(model, X_train, Y_train)
    test_loss, test_acc, Y_predict = evaluate_model(model, X_test, Y_test)
    display_results(history, test_loss, test_acc, Y_test, Y_predict)

    DESCRIPTORS = [
        ('PHOG', load_descriptor(concatenate=False)),
        ('JCD', load_descriptor(sheet=1, concatenate=False)),
        ('CEDD', load_descriptor(sheet=2, concatenate=False)),
        ('FCTH', load_descriptor(sheet=3, concatenate=False)),
        ('FuzzyColorHistogram', load_descriptor(sheet=4, concatenate=False)),
        ('Concatenated', load_descriptor(concatenate=True))
    ]

    combination_accuracies = []

    for descriptor in DESCRIPTORS:
        X_train, X_test, Y_train, Y_test = train_test_split(
            descriptor[1], Target, test_size=0.2, random_state=42, stratify=Labels
        )
        Model = build_model()
        history = train_model(Model, X_train, Y_train)
        test_loss, test_acc, Y_predict = evaluate_model(Model, X_test, Y_test)
        combination_accuracies.append(test_acc)
        print(f"Descriptor {descriptor[0]} : Loss = {test_loss} | Accuracy = {test_acc}")
        display_loss_accuracy(name=descriptor[0], history=history, test_loss=test_loss, test_acc=test_acc)
        display_confusion_matrix(name=descriptor[0], Y_test=Y_test, Y_predict=Y_predict)
        k = 5
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, np.argmax(Y_train, axis=1))

        # Prédictions et évaluation
        y_knn_pred = knn.predict(X_test)
        knn_accuracy = accuracy_score(np.argmax(Y_test, axis=1), y_knn_pred)
        print(f"KPPV accuracy (k={k}): {knn_accuracy}")

    print(f"Best descriptor : {DESCRIPTORS[combination_accuracies.index(max(combination_accuracies))][0]}")


if __name__ == "__main__":
    main()
