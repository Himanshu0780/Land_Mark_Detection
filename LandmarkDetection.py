"""Landmark Detection - corrected and simplified.

This script fixes imports, image path helpers, the batch generator,
and builds a Keras model using VGG19 (without the top). It provides a
basic training and evaluation loop suitable for small runs.
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

from sklearn.preprocessing import LabelEncoder


# Configuration
CSV_PATH = 'train.csv'
BASE_PATH = './images/'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1


# Load CSV
df = pd.read_csv(CSV_PATH)
if 'id' not in df.columns or 'landmark_id' not in df.columns:
    raise ValueError('CSV must contain "id" and "landmark_id" columns')

# Label encoder
lencoder = LabelEncoder()
lencoder.fit(df['landmark_id'])
num_classes = len(lencoder.classes_)


def image_path_from_id(image_id: str) -> str:
    """Build the expected nested path from the image id.

    The dataset arranges images under folders named by the first
    three characters of the id, e.g. id '0001abcd' -> '0/0/0/0001abcd.jpg'
    """
    fname = f"{image_id}.jpg"
    f1, f2, f3 = fname[0], fname[1], fname[2]
    return os.path.join(BASE_PATH, f1, f2, f3, fname)


def load_image_by_path(path: str):
    im = cv2.imread(path)
    if im is None:
        # Try PIL as a fallback (some images may be unusual)
        try:
            pil = Image.open(path).convert('RGB')
            im = np.array(pil)[:, :, ::-1]  # RGB->BGR for consistency with cv2
        except Exception:
            raise FileNotFoundError(f'Unable to load image: {path}')
    return im


def get_image_from_index(idx: int, dataframe: pd.DataFrame):
    """Return (image_array, label) for row index `idx` in dataframe."""
    row = dataframe.iloc[idx]
    image_id = str(row['id'])
    label = row['landmark_id']
    path = image_path_from_id(image_id)
    im = load_image_by_path(path)
    return im, label


def preprocess_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, IMAGE_SIZE)
    return im.astype('float32') / 255.0


def get_batch(dataframe: pd.DataFrame, start: int, batch_size: int):
    image_array = []
    label_array = []
    end = min(start + batch_size, len(dataframe))
    for i in range(start, end):
        im, lbl = get_image_from_index(i, dataframe)
        im = preprocess_image(im)
        image_array.append(im)
        label_array.append(lbl)
    label_array = lencoder.transform(label_array)
    return np.array(image_array), np.array(label_array)


def build_model(num_classes: int):
    base = VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=preds)
    opt = RMSprop(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def show_random_samples(n=4):
    fig = plt.figure(figsize=(12, 4))
    sampled = random.sample(range(len(df)), min(n, len(df)))
    for i, idx in enumerate(sampled, 1):
        im, lbl = get_image_from_index(idx, df)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        fig.add_subplot(1, n, i)
        plt.imshow(cv2.resize(im, (200, 200)))
        plt.title(str(lbl))
        plt.axis('off')
    plt.show()


def main():
    # Quick visual check (optional)
    try:
        show_random_samples(4)
    except Exception as e:
        print('Warning: could not show sample images:', e)

    # Split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    model = build_model(num_classes)
    model.summary()

    # Training (simple train_on_batch loop)
    for e in range(EPOCHS):
        print(f'Epoch: {e+1}/{EPOCHS}')
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        steps = int(np.ceil(len(train_df) / BATCH_SIZE))
        for it in range(steps):
            x_batch, y_batch = get_batch(train_df, it * BATCH_SIZE, BATCH_SIZE)
            if len(x_batch) == 0:
                continue
            loss, acc = model.train_on_batch(x_batch, y_batch)
        print(f'  Last batch loss={loss:.4f} acc={acc:.4f}')

    model.save('Model')

    # Evaluation
    errors = 0
    good_preds = []
    bad_preds = []
    steps = int(np.ceil(len(val_df) / BATCH_SIZE))
    for it in range(steps):
        x_val, y_val = get_batch(val_df, it * BATCH_SIZE, BATCH_SIZE)
        if len(x_val) == 0:
            continue
        preds = model.predict(x_val)
        cla = np.argmax(preds, axis=1)
        for j in range(len(cla)):
            if cla[j] != y_val[j]:
                errors += 1
                bad_preds.append((it * BATCH_SIZE + j, cla[j], preds[j, cla[j]]))
            else:
                good_preds.append((it * BATCH_SIZE + j, cla[j], preds[j, cla[j]]))

    print(f'Evaluation errors: {errors} / {len(val_df)}')


if __name__ == '__main__':
    main()