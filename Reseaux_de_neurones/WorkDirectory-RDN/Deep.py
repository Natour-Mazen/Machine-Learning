import cv2
import numpy as np
from keras import Sequential, Model
from keras.src.applications.xception import Xception
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation
from sklearn.model_selection import train_test_split
from Full_connected import run_general_model, CATEGORIES
from Plot_Display import display_loss_accuracy, display_confusion_matrix

IMAGES_PATH = 'ressources/Wang/'
IMAGE_EXT = '.jpg'
IMAGE_SIZE = 256


def load_images():
    labels, _, _, _ = run_general_model()
    image_paths = [IMAGES_PATH + str(i) + IMAGE_EXT for i in range(len(labels))]
    images = []
    targets = np.zeros([len(labels), 10], 'int')

    for i, path in enumerate(image_paths):
        category = int(path[len(IMAGES_PATH):-len(IMAGE_EXT)]) // 100
        targets[i, category] = 1
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
        else:
            print(f"Warning: Unable to read image {path}. Skipping this file.")

    return np.array(images), targets


def build_simple_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    return model


def build_deep_model(input_shape):
    model = Sequential([
        Conv2D(8, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    return model


def build_transfer_learning_model(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, output)
    return model


def build_augmented_model(input_shape):
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1)
    ])

    model = Sequential([
        data_augmentation,
        Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    return model


def compile_and_train(model, X_train, Y_train, epochs=10):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, batch_size=32, verbose=0)


def evaluate_and_display(model, X_test, Y_test, history, name):
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    Y_predict = model.predict(X_test, verbose=0)
    display_loss_accuracy(history=history, test_loss=test_loss, test_acc=test_acc, name=name)
    display_confusion_matrix(Y_predict=Y_predict, Y_test=Y_test, categories=CATEGORIES, name=name)



if __name__ == "__main__":
    X, Y = load_images()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    input_shape = X[0].shape

    # Simple Model
    simple_model = build_simple_model(input_shape)
    history = compile_and_train(simple_model, X_train, Y_train, epochs=10)
    evaluate_and_display(simple_model, X_test, Y_test, history, name="Simple Model")

    # Deep Model
    deep_model = build_deep_model(input_shape)
    history = compile_and_train(deep_model, X_train, Y_train, epochs=15)
    evaluate_and_display(deep_model, X_test, Y_test, history, name="Deep Model")

    # Transfer Learning Model
    # transfer_model = build_transfer_learning_model(input_shape)
    # history = compile_and_train(transfer_model, X_train, Y_train, epochs=15)
    # evaluate_and_display(transfer_model, X_test, Y_test, history, name="Transfer Learning Model")

    # Augmented Model
    # augmented_model = build_augmented_model(input_shape)
    # history = compile_and_train(augmented_model, X_train, Y_train, epochs=15)
    # evaluate_and_display(augmented_model, X_test, Y_test, history, name="Augmented Model")
