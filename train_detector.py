import os

os.environ["KERAS_BACKEND"] = "torch"  # or "tensorflow" or "torch"

import keras_nlp
import numpy as np
import keras_core as keras
import time
import argparse
import json
import tensorflow as tf
from tensorflow.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, choices=[32, 64, 128, 256, 512])
parser.add_argument("--dataset", type=str, choices=["wiki-spgc", "wikimia2", "wikimia2-spgc", "bookmia"])
args = parser.parse_args()


def load_dataset(dataset):
    train_data, test_data = [], []
    with open(f"./dataset/{dataset}/length_{args.length}.json", "r") as fr:
        data = json.load(fr)
        train_data = [d[0] for d in data["train"]]
        test_data = [d[0] for d in data["test"]]
    train_dataset = Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.batch(4).cache().prefetch(tf.data.AUTOTUNE)
    test_dataset = Dataset.from_tensor_slices(test_data)
    test_dataset = test_dataset.batch(4).cache().prefetch(tf.data.AUTOTUNE)
    return {"train": train_dataset, "test": test_dataset} 

dataset = load_dataset(args.dataset)

for k, v in dataset.items():
    train_ds = v
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=args.length,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )

    num_epochs = 15

    # Linearly decaying learning rate.
    learning_rate = keras.optimizers.schedules.PolynomialDecay(
        5e-5,
        decay_steps=train_ds.cardinality() * num_epochs,
        end_learning_rate=0.0,
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    checkpoint_filepath = f"./dataset/{args.dataset}/length_{args.length}_{k}.weights.h5"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='perplexity',
        mode='min',
        save_best_only=True)
    gpt2_lm.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=[keras_nlp.metrics.Perplexity(from_logits=True)],
    )

    gpt2_lm.fit(train_ds, epochs=num_epochs, callbacks=[model_checkpoint_callback])
