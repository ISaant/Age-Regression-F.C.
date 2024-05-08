#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:13:36 2024

@author: isaac
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# current_path = os.getcwd()
# parentPath = os.path.abspath(os.path.join(current_path, '../'))
# setting path
# sys.path.append(parentPath)
from my_image_tools import *

def create_df(data_path):
    images = []
    labels = []

    for p in data_path.glob("*.jpg"):
        image_name = p.parts[-1]
        images.append(image_name)
        labels.append("_".join(image_name.split("_")[0:-1]))

    df = pd.DataFrame(data={"image": images, "label": labels})
    df["is_valid"] = False
    df.loc[df.sample(frac=0.2, random_state=42).index, "is_valid"] = True

    return df

def gray_to_rgb(img):
    gray_img = np.expand_dims(rgb2gray(img),axis=-1)
    return np.repeat(gray_img, 3, 2)

def main(data_dir, epochs, lr, batch_size, dataset_name, num_channels, greyscale, pretrained, freeze):

    # Load dataset
    train_df = create_df(Path(data_dir + '/' + dataset_name + "/train"))
    valid_df = create_df(Path(data_dir + '/' + dataset_name + "/validation"))

    train_path = data_dir + "/train"
    val_path = data_dir + "/validation"

    # ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=gray_to_rgb if greyscale else None
    )
    valid_datagen = ImageDataGenerator(rescale=1./255,
                                       preprocessing_function=gray_to_rgb if greyscale else None)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=str(data_dir + '/' + dataset_name + "/train"),
        x_col="image",
        y_col="label",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        color_mode='rgb',
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        valid_df,
        directory=str(data_dir + '/' + dataset_name + "/validation"),
        x_col="image",
        y_col="label",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        color_mode= 'rgb',
    )

    num_classes = len(train_generator.class_indices)

    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet' if pretrained else None, include_top=False, input_shape=(224, 224, 3))

    # Freeze layers if specified
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom classification head
    x = GlobalAveragePooling2D()(base_model.output) 
    # x = Flatten()(base_model.output) 
    x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(lr=lr, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    parser.add_argument("--epochs", required=True, help="The number of epochs to train for", type=int)
    parser.add_argument("--lr", required=False, help="learning rate", type=float, default=0.001)
    parser.add_argument("--batch_size", required=False, help="batch_size", type=int, default=32)
    parser.add_argument("--dataset", required=False, help="dataset name", type=str, default="Beans")
    parser.add_argument("--num_channels", required=False, help="the number of image channels", type=int, default=3)
    parser.add_argument("--greyscale", action='store_true', help="convert images to greyscale")
    parser.add_argument("--pretrained", action='store_true', help="use pretrained weights")
    parser.add_argument("--freeze", action='store_true', help="freeze base model layers")
    args = parser.parse_args()

    main(
        args.data_dir,
        args.epochs,
        args.lr,
        args.batch_size,
        args.dataset,
        args.num_channels,
        args.greyscale,
        args.pretrained,
        args.freeze,
    )