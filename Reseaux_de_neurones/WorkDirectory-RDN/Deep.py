import cv2
import numpy as np
from keras import Sequential, Model
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, \
    Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

from Full_connected import run_general_model, CATEGORIES
from Plot_Display import display_loss_accuracy, display_confusion_matrix

IMAGES_PATH = 'ressources/Wang/'
IMAGE_EXT = '.jpg'
IMAGE_SIZE = 128  # Taille de redimensionnement des images


def load_images():
    labels, _, _, X_test, _, Y_test, Y_predict = run_general_model(displayResults=False) # Récupération des labels
    classify_random_images(nb_images=5, X_test=X_test, Y_test=Y_test, Y_predict=Y_predict) # Classification des images aléatoires
    image_paths = np.array([ # Récupération des images
            (IMAGES_PATH + str(i) + IMAGE_EXT) for i in range(len(labels))
        ])
    images = [] # Création d'une liste d'images
    targets = np.zeros([len(labels), 10], 'int') # Création d'une liste de cibles

    it = 0

    for img_name in image_paths: # Parcours des images
        category = int(img_name[len(IMAGES_PATH):-len(IMAGE_EXT)]) // 100 # Récupération de la catégorie
        targets[it, category] = 1 # Ajout de la catégorie à la liste de cibles
        img = cv2.resize(
            cv2.imread(img_name),
            (IMAGE_SIZE, IMAGE_SIZE) 
        )
        if img is None:
            print(f"Warning: Unable to read image {img_name}. Skipping this file.")
        else:
            images.append(img)
            it += 1

    return np.array(images), targets, images[0].shape


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
        Dropout(0.2),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    return model


def build_transfer_learning_model(input_shape):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(64, activation='relu')(x)
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


def compile_and_train(model, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=64):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=epochs // 5, verbose=1)
    return model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        validation_split=0.2,
        callbacks=[early_stopping],
        epochs=1000,
        batch_size=batch_size,
        verbose=2
    )


def evaluate_and_display(model, X_test, Y_test, history, name):
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    Y_predict = model.predict(X_test, verbose=0)
    display_loss_accuracy(history=history, test_loss=test_loss, test_acc=test_acc, name=name)
    display_confusion_matrix(Y_predict=Y_predict, Y_test=Y_test, categories=CATEGORIES, name=name)


def classify_random_images(nb_images=5, X_test=None, Y_test=None, Y_predict=None):
    images = [np.random.randint(0, len(X_test)) for i in range(nb_images)]

    for img in images:
        true_class = np.argmax(Y_test[img])
        pred_class = np.argmax(Y_predict[img])
        accuracy = round(Y_predict[img][pred_class] * 100, 2)
        print()
        print(f"Image n°{img} of class {true_class} => Category [{CATEGORIES[true_class]}]")
        print(f"Model prediction : Class = {pred_class} => Category [{CATEGORIES[pred_class]}] | Accuracy = {accuracy}%")


if __name__ == "__main__":
    X, Y, input_shape = load_images()
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=1)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=1)

    # Simple Model
    simple_model = build_simple_model(input_shape)
    history = compile_and_train(simple_model, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=32)
    evaluate_and_display(simple_model, X_test, Y_test, history, name="Simple Model")

    # Deep Model
    deep_model = build_deep_model(input_shape)
    history = compile_and_train(deep_model, X_train, Y_train, X_val, Y_val, epochs=40, batch_size=32)
    evaluate_and_display(deep_model, X_test, Y_test, history, name="Deep Model")

    # Transfer Learning Model
    transfer_model = build_transfer_learning_model(input_shape)
    history = compile_and_train(transfer_model, X_train, Y_train, X_val, Y_val, epochs=32, batch_size=32)
    evaluate_and_display(transfer_model, X_test, Y_test, history, name="Transfer Learning Model")

    # Augmented Model
    augmented_model = build_augmented_model(input_shape)
    history = compile_and_train(augmented_model, X_train, Y_train, X_val, Y_val, epochs=30, batch_size=16)
    evaluate_and_display(augmented_model, X_test, Y_test, history, name="Augmented Model")
