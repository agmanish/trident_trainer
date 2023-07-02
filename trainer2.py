from tensorflow import keras
from tensorflow.keras import callbacks
from kerastuner import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
import json
import os

import pandas as pd
import time
import numpy as np


def transformer_encoder(
    inputs,
    head_size,
    num_heads,
    ff_dim,
    dropout=0.0,
):
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs

    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res


def get_indices(feature_list):
    feature_dict = {
        "timestamp": 0,
        "lat": 1,
        "lon": 2,
        "sog": 3,
        "cog": 4,
        "oa": 5,
        "head": 6,
        "time_interval": 7,
        "distance": 8,
        "speed": 9,
        "acceleration": 10,
        "lat_speed": 11,
        "lon_speed": 12,
    }

    return [feature_dict[i] for i in feature_list]


def create_model(
    input_shape,
    n_classes,
    num_transformer_blocks,
    head_size,
    num_heads,
    ff_dim,
    dropout,
    kernel_size,
    mlp_units,
    mlp_dropout,
):
    inputs = keras.layers.Input(shape=input_shape)

    x = inputs
    x = keras.layers.Conv1D(
        filters=ff_dim,
        kernel_size=kernel_size,
        activation="relu",
    )(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            head_size,
            num_heads,
            ff_dim,
            dropout,
        )
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    mlp_units = sorted(mlp_units)
    for unit in mlp_units:
        x = keras.layers.Dense(unit, activation="relu")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def build_model(
    hp,
    input_shape,
    n_classes,
):
    num_transformer_blocks = hp.Int(
        "num_transformer_blocks", min_value=1, max_value=8, step=1
    )
    head_size = hp.Choice("head_size", [16, 32, 64, 128, 256, 512])
    kernel_size = hp.Int("kernel_size", min_value=1, max_value=9, step=1)
    num_heads = hp.Int("num_heads", min_value=1, max_value=8, step=1)
    ff_dim = hp.Int("ff_dim", min_value=1, max_value=8, step=1)
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    mlp_unit_1 = hp.Choice("mlp_unit_1", values=[32, 64, 128, 256, 512])
    mlp_unit_2 = hp.Choice("mlp_unit_2", values=[32, 64, 128, 256, 512, 0])
    mlp_unit_3 = hp.Choice("mlp_unit_3", values=[32, 64, 128, 256, 512, 0])

    mlp_units = [value for value in [mlp_unit_1, mlp_unit_2, mlp_unit_3] if value != 0]
    mlp_dropout = hp.Float("mlp_dropout", min_value=0.0, max_value=0.5, step=0.1)

    return create_model(
        input_shape,
        n_classes,
        num_transformer_blocks,
        head_size,
        num_heads,
        ff_dim,
        dropout,
        kernel_size,
        mlp_units,
        mlp_dropout,
    )


def store_metrics(tuner, max_trials, X_test, y_test):
    df = pd.DataFrame(
        columns=[
            "trial_id",
            "name",
            "num_transformer_blocks",
            "head_size",
            "num_heads",
            "ff_dim",
            "dropout",
            "mlp_unit_1",
            "mlp_unit_2",
            "mlp_unit_3",
            "mlp_dropout",
            "val_loss",
            "val_sc_accuracy",
            "paramsize",
            "inferece_time",
        ]
    )

    for trial in tuner.oracle.get_best_trials(num_trials=max_trials):
        trial_id = trial.trial_id
        num_transformer_blocks = trial.hyperparameters.values["num_transformer_blocks"]
        head_size = trial.hyperparameters.values["head_size"]
        num_heads = trial.hyperparameters.values["num_heads"]
        ff_dim = trial.hyperparameters.values["ff_dim"]
        dropout = trial.hyperparameters.values["dropout"]
        mlp_unit_1 = trial.hyperparameters.values["mlp_unit_1"]
        mlp_unit_2 = trial.hyperparameters.values["mlp_unit_2"]
        mlp_unit_3 = trial.hyperparameters.values["mlp_unit_3"]
        mlp_dropout = trial.hyperparameters.values["mlp_dropout"]

        name = f"trial_{trial_id}_ntb_{num_transformer_blocks}_hs_{head_size}_nheads_{num_heads}_ffdim_{ff_dim}_do_{dropout}_mlp1_{mlp_unit_1}_mlp2_{mlp_unit_2}_mlp3_{mlp_unit_3}_mlpdo_{mlp_dropout}"

        test_model = tuner.hypermodel.build(trial.hyperparameters)
        [
            val_loss,
            val_sc_accuracy,
        ] = test_model.evaluate(X_test, y_test)

        st = time.time()
        y_pred = test_model.predict(X_test)
        inferece_time = time.time() - st

        paramsize = test_model.count_params()

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "trial_id": trial_id,
                        "name": name,
                        "num_transformer_blocks": num_transformer_blocks,
                        "head_size": head_size,
                        "num_heads": num_heads,
                        "ff_dim": ff_dim,
                        "dropout": dropout,
                        "mlp_unit_1": mlp_unit_1,
                        "mlp_unit_2": mlp_unit_2,
                        "mlp_unit_3": mlp_unit_3,
                        "mlp_dropout": mlp_dropout,
                        "val_loss": val_loss,
                        "val_sc_accuracy": val_sc_accuracy,
                        "paramsize": paramsize,
                        "inferece_time": inferece_time,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    return df


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    X = np.load(
        open(
            f"{file_dir}/data/1hr_inp_shifted.npy",
            "rb",
        )
    )

    y = np.load(
        open(
            f"{file_dir}/data/1hr_op_shifted.npy",
            "rb",
        )
    )

    feature_indices = get_indices(["lat", "lon", "oa", "head"])

    X = X[:, :, feature_indices]
    input_shape = X.shape[1:]
    n_classes = len(np.unique(y))

    max_trials = 300

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=333)

    for count, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        tuner = BayesianOptimization(
            hypermodel=lambda hp: build_model(
                hp=hp, input_shape=input_shape, n_classes=n_classes
            ),
            objective="sparse_categorical_accuracy",
            max_trials=max_trials,
            seed=42,
            directory="bayesian_optimization",
            project_name="trajectory_classification",
        )

        tuner.search(
            x=X_train,
            y=y_train,
            epochs=30,
            validation_data=(X_test, y_test),
        )

        df = store_metrics(tuner, max_trials, X_test, y_test)
        df.to_csv(f"{file_dir}/results/split_{count}.csv", index=False)
