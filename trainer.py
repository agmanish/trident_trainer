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


def build_model(hp, input_shape, n_classes):
    inputs = keras.layers.Input(shape=input_shape)

    x = inputs
    x = keras.layers.Conv1D(
        filters=hp.Int("ff_dim", min_value=1, max_value=8, step=1),
        kernel_size=1,
        activation="relu",
    )(x)
    for _ in range(hp.Int("num_transformer_blocks", min_value=1, max_value=8, step=1)):
        x = transformer_encoder(
            x,
            hp.Choice("head_size", [16, 32, 64, 128, 256, 512]),
            hp.Int("num_heads", min_value=1, max_value=8, step=1),
            hp.Int("ff_dim", min_value=1, max_value=8, step=1),
            hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1),
        )
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    num_units = hp.Choice("num_units", values=[1, 2, 3])
    mlp_units = []
    for _ in range(num_units):
        unit = hp.Choice("mlp_units", values=[32, 64, 128, 256, 512])
        mlp_units.append(unit)

    mlp_units = sorted(mlp_units)

    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(
            hp.Float("mlp_dropout", min_value=0.0, max_value=0.5, step=0.1)
        )(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr = hp.Choice("lr", values=[1e-4, 1e-3, 1e-2])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    np.random.shuffle(X)
    np.random.seed(random_state)
    np.random.shuffle(y)

    print(X.shape)

    split_index = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    split_index = int(X_train.shape[0] * (1 - val_size))
    X_train, X_val = X_train[:split_index], X_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]

    return X_train, X_val, X_test, y_train, y_val, y_test


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


def custom_objective(metrics_dict):
    # sourcery skip: inline-immediately-returned-variable
    # define weight factors
    acc_weight = (
        1  # we care about reaching 90% accuracy, but not about further improvements
    )
    time_weight = 5  # we care the most about inference time
    size_weight = 3  # we care about model size as well

    # calculate score
    score = (
        acc_weight * metrics_dict["sparse_categorical_accuracy"]
        - time_weight * metrics_dict["inference_time"]
        - size_weight * np.log10(metrics_dict["model_param_size"])
    )

    return score


def hyperparam_tuner(X, y, n_classes, max_trials=10):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize empty DataFrame to store results
    results_df = pd.DataFrame()
    input_shape = X.shape[1:]

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        tuner = BayesianOptimization(
            hypermodel=lambda hp: build_model(
                hp=hp, input_shape=input_shape, n_classes=n_classes
            ),
            objective="sparse_categorical_accuracy",
            max_trials=max_trials,
            seed=42,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            directory="bayesian_optimization",
            project_name="trajectory_classification",
        )

        stop_early = callbacks.EarlyStopping(monitor="val_loss", patience=9)

        tuner.search(
            x=X_train,
            y=y_train,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[stop_early],
            verbose=1,
        )

        # Evaluate all models and store the results
        for trial in tuner.oracle.trials.values():
            model = tuner.hypermodel.build(trial.hyperparameters)
            metrics = model.evaluate(X_val, y_val)
            metrics_names = model.metrics_names
            metrics_dict = dict(zip(metrics_names, metrics))

            model_complexity = len(model.layers)
            model_param_size = model.count_params()
            start_time = time.time()
            model.predict(X_val)
            inference_time = time.time() - start_time

            metrics_dict["model_complexity"] = model_complexity
            metrics_dict["model_param_size"] = model_param_size
            metrics_dict["inference_time"] = inference_time

            score = custom_objective(metrics_dict)
            model_info = {
                "score": score,
                **metrics_dict,
                **trial.hyperparameters.values,
            }
            results_df = results_df.append(model_info, ignore_index=True)
            # store the model if it meets the accuracy criterion and is better than the current best
            if metrics_dict["sparse_categorical_accuracy"] > 0.9 and (
                not best_models or score > best_models[0][0]
            ):
                best_models = [
                    (score, model, metrics_dict, trial.hyperparameters.values)
                ]

            # Append the dictionary to the results DataFrame

        keras.backend.clear_session()

    # Write the results DataFrame to a CSV file
    results_df.to_csv(
        f"{os.getcwd()}/results/hyperparam_tuning_results.csv", index=False
    )

    # sort the models by score and return the best one
    results_df.sort_values(by="score", ascending=False, inplace=True)
    best_model_info = None if results_df.empty else results_df.iloc[0]

    if best_model_info is not None:
        with open(f"{os.getcwd()}/results/tuner_results.txt", "w") as f:
            f.write(json.dumps(best_model_info["metrics"], indent=4))  # metrics_dict
            f.write("\nBest Hyperparameters: ")
            f.write(
                json.dumps(best_model_info["hyperparameters"], indent=4)
            )  # hyperparameters
            f.write("\n\n")
    # return the best model
    else:
        print("No model found that meets the accuracy criterion.")
        return None


if __name__ == "__main__":
    X = np.load(
        open(
            f"{os.getcwd()}/data/1hr_inp_shifted.npy",
            "rb",
        )
    )

    y = np.load(
        open(
            f"{os.getcwd()}/data/1hr_op_shifted.npy",
            "rb",
        )
    )

    feature_indices = get_indices(["lat", "lon", "oa", "head"])

    X = X[:, :, feature_indices]
    n_classes = len(np.unique(y))

    hyperparam_tuner(X, y, n_classes, max_trials=100)
