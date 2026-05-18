from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class PredictiveModelling:
    """Prepare Weymouth (WEY) to London Waterloo (WAT) train data for modelling."""

    DROP_COLUMNS = ("rid", "late_canc_reason")
    TIME_COLUMNS = (
        "planned_arrival_time",
        "planned_departure_time",
        "actual_arrival_time",
        "actual_departure_time",
    )
    DELAY_BUCKET_BINS = [-np.inf, 0, 5, 15, 30, 60, np.inf]
    DELAY_BUCKET_LABELS = [
        "early_or_on_time",
        "minor_delay",
        "moderate_delay",
        "major_delay",
        "severe_delay",
        "very_severe_delay",
    ]

    def __init__(
        self,
        data_path: str | Path = "./train_historical_data/2024_WEY2WAT.xlsx",
        destination: str = "WAT",
    ):
        self.data_path = Path(data_path)
        self.destination = destination.upper()
        self.raw_data: pd.DataFrame | None = None
        self.preprocessed_data: pd.DataFrame | None = None
        self.feature_data: pd.DataFrame | None = None
        self.feature_columns: list[str] = []
        self.target_column = "target_arrival_delay_minutes"
        self.arrival_time_target_column = (
            "target_arrival_minutes_from_service_day_start"
        )

    def preprocess_data(self) -> pd.DataFrame:
        """Load, normalise, and return stop-level data without rid/cancellation fields."""
        dataset = pd.read_excel(self.data_path)
        dataset.columns = dataset.columns.str.strip()

        required_columns = {
            "rid",
            "date_of_service",
            "toc_code",
            "location",
            "planned_arrival_time",
            "planned_departure_time",
            "actual_arrival_time",
            "actual_departure_time",
            "late_canc_reason",
        }
        missing_columns = required_columns.difference(dataset.columns)
        if missing_columns:
            raise ValueError(
                "Missing expected train data columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        dataset["_row_order"] = np.arange(len(dataset))
        dataset["date_of_service"] = pd.to_datetime(
            dataset["date_of_service"], errors="coerce"
        ).dt.normalize()
        dataset["toc_code"] = self._clean_station_code(dataset["toc_code"])
        dataset["location"] = self._clean_station_code(dataset["location"])
        dataset = dataset.drop(columns=["late_canc_reason"], errors="ignore")

        dataset = dataset.sort_values(["rid", "_row_order"], kind="stable").reset_index(
            drop=True
        )

        # Keep rid internally for grouping journeys, but remove it from the public
        # preprocessed table requested by the coursework brief.
        self.raw_data = dataset
        self.preprocessed_data = dataset.drop(
            columns=[*self.DROP_COLUMNS, "_row_order"], errors="ignore"
        ).copy()
        return self.preprocessed_data

    def feature_engineering(self, destination: str | None = None) -> pd.DataFrame:
        """Return model-ready stop observations without internal journey IDs."""
        return self._engineer_features(destination=destination, keep_rid=False)

    def _engineer_features(
        self, destination: str | None = None, keep_rid: bool = False
    ) -> pd.DataFrame:
        """Create supervised features for predicting arrival time at destination.

        Each output row represents a historical train observed at one station before
        the destination. The target is the actual arrival delay at the destination;
        predicted arrival time can be calculated as planned destination arrival time
        plus the predicted delay.
        """
        if self.raw_data is None:
            self.preprocess_data()

        destination = (destination or self.destination).upper()
        data = self.raw_data.copy()
        data = self._add_journey_datetimes(data)
        data = self._add_stop_level_features(data)

        destination_rows = (
            data[data["location"] == destination]
            .sort_values(["rid", "stop_number"], kind="stable")
            .drop_duplicates("rid", keep="last")
        )
        if destination_rows.empty:
            raise ValueError(f"No destination rows found for station code {destination}.")

        destination_rows = destination_rows[
            [
                "rid",
                "stop_number",
                "planned_arrival_datetime",
                "actual_arrival_datetime",
            ]
        ].rename(
            columns={
                "stop_number": "destination_stop_number",
                "planned_arrival_datetime": "planned_destination_arrival_datetime",
                "actual_arrival_datetime": "target_arrival_datetime",
            }
        )

        origin_rows = (
            data.sort_values(["rid", "stop_number"], kind="stable")
            .drop_duplicates("rid", keep="first")[
                [
                    "rid",
                    "location",
                    "planned_departure_datetime",
                    "actual_departure_datetime",
                ]
            ]
            .rename(
                columns={
                    "location": "origin",
                    "planned_departure_datetime": "planned_origin_departure_datetime",
                    "actual_departure_datetime": "actual_origin_departure_datetime",
                }
            )
        )

        features = data.merge(destination_rows, on="rid", how="inner").merge(
            origin_rows, on="rid", how="left"
        )
        features["current_location"] = features["location"]
        features["destination"] = destination

        features = features[
            features["stop_number"] < features["destination_stop_number"]
        ].copy()

        features["target_arrival_delay_minutes"] = self._minutes_between(
            features["target_arrival_datetime"],
            features["planned_destination_arrival_datetime"],
        )
        features["target_arrival_minutes_from_service_day_start"] = self._minutes_between(
            features["target_arrival_datetime"],
            features["date_of_service"],
        )
        features["scheduled_minutes_to_destination"] = self._minutes_between(
            features["planned_destination_arrival_datetime"],
            features["planned_current_datetime"],
        )
        features["minutes_since_origin_departure"] = self._minutes_between(
            features["planned_current_datetime"],
            features["planned_origin_departure_datetime"],
        )
        features["scheduled_total_journey_minutes"] = self._minutes_between(
            features["planned_destination_arrival_datetime"],
            features["planned_origin_departure_datetime"],
        )
        features["origin_departure_delay_minutes"] = self._minutes_between(
            features["actual_origin_departure_datetime"],
            features["planned_origin_departure_datetime"],
        )
        features["stops_to_destination"] = (
            features["destination_stop_number"] - features["stop_number"]
        )
        features["route_progress_pct"] = (
            features["stop_number"] / features["destination_stop_number"]
        ).replace([np.inf, -np.inf], np.nan)
        features["remaining_journey_pct"] = 1 - features["route_progress_pct"]
        features["current_delay_to_remaining_ratio"] = (
            features["current_delay_minutes"]
            / features["scheduled_minutes_to_destination"].replace(0, np.nan)
        )

        features = self._add_calendar_features(features)
        features = self._add_time_of_day_features(
            features,
            source_col="planned_current_datetime",
            prefix="current",
        )
        features = self._add_time_of_day_features(
            features,
            source_col="planned_destination_arrival_datetime",
            prefix="destination",
        )
        features["current_delay_bucket"] = pd.cut(
            features["current_delay_minutes"],
            bins=self.DELAY_BUCKET_BINS,
            labels=self.DELAY_BUCKET_LABELS,
        ).astype("string")

        features = features.dropna(
            subset=[
                "planned_current_datetime",
                "actual_current_datetime",
                "planned_destination_arrival_datetime",
                "target_arrival_datetime",
                "current_delay_minutes",
                "scheduled_minutes_to_destination",
                "target_arrival_delay_minutes",
            ]
        )
        features = features[features["scheduled_minutes_to_destination"] > 0].copy()
        features = self._fill_model_missing_values(features)

        model_columns = [
            "date_of_service",
            "toc_code",
            "origin",
            "current_location",
            "destination",
            "planned_current_datetime",
            "actual_current_datetime",
            "planned_destination_arrival_datetime",
            "target_arrival_datetime",
            "planned_current_minutes_of_day",
            "actual_current_minutes_of_day",
            "planned_destination_arrival_minutes_of_day",
            "current_delay_minutes",
            "arrival_delay_minutes",
            "departure_delay_minutes",
            "origin_departure_delay_minutes",
            "planned_dwell_minutes",
            "actual_dwell_minutes",
            "arrival_delay_minutes_missing",
            "departure_delay_minutes_missing",
            "origin_departure_delay_minutes_missing",
            "planned_dwell_minutes_missing",
            "actual_dwell_minutes_missing",
            "scheduled_minutes_to_destination",
            "minutes_since_origin_departure",
            "scheduled_total_journey_minutes",
            "stop_number",
            "total_stops_in_service",
            "destination_stop_number",
            "stops_to_destination",
            "route_progress_pct",
            "remaining_journey_pct",
            "current_delay_to_remaining_ratio",
            "service_month",
            "service_day_of_month",
            "service_day_of_week",
            "service_day_of_year",
            "is_weekend",
            "current_is_peak",
            "destination_is_peak",
            "current_delay_bucket",
            "service_day_of_week_sin",
            "service_day_of_week_cos",
            "service_day_of_year_sin",
            "service_day_of_year_cos",
            "planned_current_time_sin",
            "planned_current_time_cos",
            "planned_destination_arrival_time_sin",
            "planned_destination_arrival_time_cos",
            "target_arrival_delay_minutes",
            "target_arrival_minutes_from_service_day_start",
        ]
        if keep_rid:
            model_columns = ["rid", *model_columns]

        features = features[model_columns].reset_index(drop=True)
        if not keep_rid:
            self.feature_data = features
        self.feature_columns = [
            column
            for column in features.columns
            if column
            not in {
                "rid",
                "target_arrival_delay_minutes",
                "target_arrival_datetime",
                "target_arrival_minutes_from_service_day_start",
            }
        ]
        return features

    def train_linear_model(
        self,
        destination: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        alpha: float = 1.0,
        max_train_rows: int | None = None,
        max_test_rows: int | None = None,
    ) -> dict:
        """Train a regularised linear regression model for destination delay."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        training_data = self._prepare_tabular_training_data(
            destination=destination,
            test_size=test_size,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )
        model = Pipeline(
            steps=[
                ("preprocessor", training_data["preprocessor"]),
                ("regressor", Ridge(alpha=alpha, random_state=random_state)),
            ]
        )
        model.fit(training_data["X_train"], training_data["y_train"])
        train_predictions = model.predict(training_data["X_train"])
        predictions = model.predict(training_data["X_test"])
        return self._build_training_result(
            model_name="linear_ridge",
            model=model,
            training_data=training_data,
            train_predictions=train_predictions,
            predictions=predictions,
        )

    def train_knn_model(
        self,
        destination: str | None = None,
        test_size: float = 0.2,
        n_neighbors: int = 7,
        weights: str = "distance",
        max_train_rows: int | None = 30_000,
        max_test_rows: int | None = 10_000,
    ) -> dict:
        """Train a K-nearest-neighbours regressor for destination delay."""
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline

        training_data = self._prepare_tabular_training_data(
            destination=destination,
            test_size=test_size,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )
        model = Pipeline(
            steps=[
                ("preprocessor", training_data["preprocessor"]),
                (
                    "regressor",
                    KNeighborsRegressor(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        metric="minkowski",
                    ),
                ),
            ]
        )
        model.fit(training_data["X_train"], training_data["y_train"])
        train_predictions = model.predict(training_data["X_train"])
        predictions = model.predict(training_data["X_test"])
        return self._build_training_result(
            model_name="knn",
            model=model,
            training_data=training_data,
            train_predictions=train_predictions,
            predictions=predictions,
        )

    def train_lstm_model(
        self,
        destination: str | None = None,
        test_size: float = 0.2,
        sequence_length: int = 6,
        units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
        epochs: int = 20,
        batch_size: int = 256,
        patience: int = 4,
        random_state: int = 42,
        max_train_sequences: int | None = 50_000,
        max_test_sequences: int | None = 10_000,
        verbose: int = 0,
    ) -> dict:
        """Train an LSTM on each train's ordered stop observations."""
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler

        tf.keras.utils.set_random_seed(random_state)
        sequence_data = self._prepare_sequence_training_data(
            destination=destination,
            test_size=test_size,
            sequence_length=sequence_length,
            max_train_sequences=max_train_sequences,
            max_test_sequences=max_test_sequences,
        )

        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(
            sequence_data["y_train"].reshape(-1, 1)
        )
        y_test_scaled = target_scaler.transform(sequence_data["y_test"].reshape(-1, 1))

        model = tf.keras.Sequential(
            [
                tf.keras.Input(
                    shape=(
                        sequence_data["X_train"].shape[1],
                        sequence_data["X_train"].shape[2],
                    )
                ),
                tf.keras.layers.Masking(mask_value=0.0),
                tf.keras.layers.LSTM(units),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dense_units, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="mse",
            metrics=["mae"],
        )
        history = model.fit(
            sequence_data["X_train"],
            y_train_scaled,
            validation_data=(sequence_data["X_test"], y_test_scaled),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )
            ],
        )
        scaled_predictions = model.predict(
            sequence_data["X_test"], batch_size=batch_size, verbose=0
        )
        scaled_train_predictions = model.predict(
            sequence_data["X_train"], batch_size=batch_size, verbose=0
        )
        predictions = target_scaler.inverse_transform(scaled_predictions).ravel()
        train_predictions = target_scaler.inverse_transform(
            scaled_train_predictions
        ).ravel()
        return self._build_training_result(
            model_name="lstm",
            model=model,
            training_data=sequence_data,
            train_predictions=train_predictions,
            predictions=predictions,
            extra={
                "history": history.history,
                "loss_log": self._build_neural_loss_log(history.history),
                "preprocessor": sequence_data["preprocessor"],
                "target_scaler": target_scaler,
                "sequence_length": sequence_length,
            },
        )

    def train_mlp_model(
        self,
        destination: str | None = None,
        test_size: float = 0.2,
        hidden_layers: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        epochs: int = 20,
        batch_size: int = 256,
        patience: int = 4,
        random_state: int = 42,
        max_train_rows: int | None = 75_000,
        max_test_rows: int | None = 10_000,
        verbose: int = 0,
    ) -> dict:
        """Train a feed-forward neural network on the engineered tabular features."""
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler

        tf.keras.utils.set_random_seed(random_state)
        training_data = self._prepare_tabular_training_data(
            destination=destination,
            test_size=test_size,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )

        X_train = training_data["preprocessor"].fit_transform(training_data["X_train"])
        X_test = training_data["preprocessor"].transform(training_data["X_test"])
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(
            training_data["y_train"].to_numpy().reshape(-1, 1)
        )
        y_test_scaled = target_scaler.transform(
            training_data["y_test"].to_numpy().reshape(-1, 1)
        )

        layers: list[tf.keras.layers.Layer] = [
            tf.keras.Input(shape=(X_train.shape[1],))
        ]
        for layer_size in hidden_layers:
            layers.extend(
                [
                    tf.keras.layers.Dense(layer_size, activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(dropout),
                ]
            )
        layers.append(tf.keras.layers.Dense(1))

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="mse",
            metrics=["mae"],
        )
        history = model.fit(
            X_train,
            y_train_scaled,
            validation_data=(X_test, y_test_scaled),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )
            ],
        )
        scaled_predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
        scaled_train_predictions = model.predict(
            X_train, batch_size=batch_size, verbose=0
        )
        predictions = target_scaler.inverse_transform(scaled_predictions).ravel()
        train_predictions = target_scaler.inverse_transform(
            scaled_train_predictions
        ).ravel()
        return self._build_training_result(
            model_name="mlp",
            model=model,
            training_data=training_data,
            train_predictions=train_predictions,
            predictions=predictions,
            extra={
                "history": history.history,
                "loss_log": self._build_neural_loss_log(history.history),
                "preprocessor": training_data["preprocessor"],
                "target_scaler": target_scaler,
            },
        )

    def run_training_cycles(
        self,
        destination: str | None = None,
        output_dir: str | Path = "./model_performance",
        quick_run: bool = True,
        include_neural_models: bool = True,
    ) -> dict[str, dict]:
        """Train all configured models and save performance charts to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if quick_run:
            training_plan = {
                "linear": (
                    self.train_linear_model,
                    {"max_train_rows": 10_000, "max_test_rows": 2_000},
                ),
                "knn": (
                    self.train_knn_model,
                    {
                        "max_train_rows": 5_000,
                        "max_test_rows": 1_000,
                        "n_neighbors": 5,
                    },
                ),
                "mlp": (
                    self.train_mlp_model,
                    {
                        "epochs": 5,
                        "max_train_rows": 10_000,
                        "max_test_rows": 2_000,
                        "batch_size": 256,
                    },
                ),
                "lstm": (
                    self.train_lstm_model,
                    {
                        "epochs": 5,
                        "max_train_sequences": 10_000,
                        "max_test_sequences": 2_000,
                        "sequence_length": 6,
                        "batch_size": 256,
                    },
                ),
            }
        else:
            training_plan = {
                "linear": (self.train_linear_model, {}),
                "knn": (self.train_knn_model, {}),
                "mlp": (self.train_mlp_model, {"epochs": 20}),
                "lstm": (self.train_lstm_model, {"epochs": 20}),
            }

        if not include_neural_models:
            training_plan = {
                name: plan
                for name, plan in training_plan.items()
                if name not in {"mlp", "lstm"}
            }

        results: dict[str, dict] = {}
        summary_rows: list[dict] = []

        for model_key, (trainer, params) in training_plan.items():
            print(f"\nTraining {model_key}...")
            result = trainer(destination=destination, **params)
            results[model_key] = result

            self._save_model_performance_outputs(
                model_key=model_key,
                result=result,
                output_dir=output_dir,
            )

            summary_row = {
                "model": model_key,
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                **result["metrics"],
                **result["accuracy_metrics"],
            }
            summary_rows.append(summary_row)

            print(
                f"{model_key}: MAE={result['metrics']['mae_minutes']:.2f} min, "
                f"RMSE={result['metrics']['rmse_minutes']:.2f} min, "
                f"bucket accuracy={result['accuracy_metrics']['bucket_accuracy']:.1%}, "
                f"within 5 min={result['accuracy_metrics']['within_5_min_accuracy']:.1%}"
            )

        summary = pd.DataFrame(summary_rows)
        self._save_summary_charts(summary=summary, output_dir=output_dir)
        print(f"\nSaved performance charts to: {output_dir.resolve()}")
        return results

    def _prepare_tabular_training_data(
        self,
        destination: str | None,
        test_size: float,
        max_train_rows: int | None = None,
        max_test_rows: int | None = None,
    ) -> dict:
        features = self.feature_engineering(destination=destination)
        features = features.sort_values(
            ["date_of_service", "planned_current_datetime", "current_location"],
            kind="stable",
        ).reset_index(drop=True)

        train_frame, test_frame = self._chronological_train_test_split(
            features, test_size=test_size
        )
        train_frame = self._limit_frame(train_frame, max_train_rows, keep="tail")
        test_frame = self._limit_frame(test_frame, max_test_rows, keep="head")

        feature_columns = self._select_model_feature_columns(features)
        preprocessor = self._build_tabular_preprocessor(train_frame, feature_columns)

        return {
            "X_train": train_frame[feature_columns],
            "X_test": test_frame[feature_columns],
            "y_train": train_frame[self.target_column],
            "y_test": test_frame[self.target_column],
            "train_frame": train_frame,
            "test_frame": test_frame,
            "feature_columns": feature_columns,
            "preprocessor": preprocessor,
        }

    def _prepare_sequence_training_data(
        self,
        destination: str | None,
        test_size: float,
        sequence_length: int,
        max_train_sequences: int | None = None,
        max_test_sequences: int | None = None,
    ) -> dict:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")

        features = self._engineer_features(destination=destination, keep_rid=True)
        features = features.sort_values(
            ["date_of_service", "rid", "stop_number"], kind="stable"
        ).reset_index(drop=True)

        journey_dates = (
            features[["rid", "date_of_service"]]
            .drop_duplicates("rid")
            .sort_values(["date_of_service", "rid"], kind="stable")
            .reset_index(drop=True)
        )
        train_journeys, test_journeys = self._chronological_train_test_split(
            journey_dates, test_size=test_size
        )
        train_rids = set(train_journeys["rid"])
        test_rids = set(test_journeys["rid"])

        train_frame = (
            features[features["rid"].isin(train_rids)]
            .sort_values(["date_of_service", "rid", "stop_number"], kind="stable")
            .reset_index(drop=True)
        )
        test_frame = (
            features[features["rid"].isin(test_rids)]
            .sort_values(["date_of_service", "rid", "stop_number"], kind="stable")
            .reset_index(drop=True)
        )
        feature_columns = self._select_model_feature_columns(features)
        preprocessor = self._build_tabular_preprocessor(train_frame, feature_columns)

        X_train_flat = preprocessor.fit_transform(train_frame[feature_columns]).astype(
            "float32"
        )
        X_test_flat = preprocessor.transform(test_frame[feature_columns]).astype(
            "float32"
        )

        X_train, y_train, train_metadata = self._build_lstm_sequences(
            frame=train_frame,
            transformed_features=X_train_flat,
            sequence_length=sequence_length,
        )
        X_test, y_test, test_metadata = self._build_lstm_sequences(
            frame=test_frame,
            transformed_features=X_test_flat,
            sequence_length=sequence_length,
        )

        X_train, y_train, train_metadata = self._limit_arrays_and_frame(
            X_train,
            y_train,
            train_metadata,
            max_rows=max_train_sequences,
            keep="tail",
        )
        X_test, y_test, test_metadata = self._limit_arrays_and_frame(
            X_test,
            y_test,
            test_metadata,
            max_rows=max_test_sequences,
            keep="head",
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_frame": train_metadata,
            "test_frame": test_metadata,
            "feature_columns": feature_columns,
            "preprocessor": preprocessor,
        }

    def _build_lstm_sequences(
        self,
        frame: pd.DataFrame,
        transformed_features: np.ndarray,
        sequence_length: int,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        metadata_rows: list[pd.Series] = []
        feature_count = transformed_features.shape[1]

        for _, journey in frame.groupby("rid", sort=False):
            row_positions = journey.index.to_numpy()
            for journey_position in range(len(row_positions)):
                window_positions = row_positions[
                    max(0, journey_position - sequence_length + 1) : journey_position + 1
                ]
                sequence = np.zeros(
                    (sequence_length, feature_count),
                    dtype="float32",
                )
                window = transformed_features[window_positions]
                sequence[-len(window) :] = window
                current_row = frame.iloc[row_positions[journey_position]]
                sequences.append(sequence)
                targets.append(float(current_row[self.target_column]))
                metadata_rows.append(current_row)

        return (
            np.asarray(sequences, dtype="float32"),
            np.asarray(targets, dtype="float32"),
            pd.DataFrame(metadata_rows).reset_index(drop=True),
        )

    @staticmethod
    def _chronological_train_test_split(
        frame: pd.DataFrame, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        split_index = int(len(frame) * (1 - test_size))
        split_index = min(max(split_index, 1), len(frame) - 1)
        return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()

    @staticmethod
    def _limit_frame(
        frame: pd.DataFrame, max_rows: int | None, keep: str
    ) -> pd.DataFrame:
        if max_rows is None or len(frame) <= max_rows:
            return frame
        if keep == "tail":
            return frame.tail(max_rows).copy()
        if keep == "head":
            return frame.head(max_rows).copy()
        raise ValueError("keep must be either 'head' or 'tail'.")

    @staticmethod
    def _limit_arrays_and_frame(
        X: np.ndarray,
        y: np.ndarray,
        frame: pd.DataFrame,
        max_rows: int | None,
        keep: str,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if max_rows is None or len(frame) <= max_rows:
            return X, y, frame
        if keep == "tail":
            return X[-max_rows:], y[-max_rows:], frame.tail(max_rows).reset_index(
                drop=True
            )
        if keep == "head":
            return X[:max_rows], y[:max_rows], frame.head(max_rows).reset_index(
                drop=True
            )
        raise ValueError("keep must be either 'head' or 'tail'.")

    def _select_model_feature_columns(self, features: pd.DataFrame) -> list[str]:
        excluded_columns = {
            "rid",
            "date_of_service",
            "planned_current_datetime",
            "actual_current_datetime",
            "planned_destination_arrival_datetime",
            "target_arrival_datetime",
            "target_arrival_delay_minutes",
            "target_arrival_minutes_from_service_day_start",
        }
        return [
            column
            for column in features.columns
            if column not in excluded_columns
            and not pd.api.types.is_datetime64_any_dtype(features[column])
        ]

    @staticmethod
    def _build_tabular_preprocessor(
        frame: pd.DataFrame, feature_columns: list[str]
    ):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        categorical_columns = [
            column
            for column in feature_columns
            if pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        ]
        numeric_columns = [
            column for column in feature_columns if column not in categorical_columns
        ]

        transformers = []
        if numeric_columns:
            transformers.append(("numeric", StandardScaler(), numeric_columns))
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                )
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _build_training_result(
        self,
        model_name: str,
        model,
        training_data: dict,
        train_predictions: np.ndarray,
        predictions: np.ndarray,
        extra: dict | None = None,
    ) -> dict:
        train_predictions = np.asarray(train_predictions).ravel()
        predictions = np.asarray(predictions).ravel()
        y_train = np.asarray(training_data["y_train"]).ravel()
        y_test = np.asarray(training_data["y_test"]).ravel()
        prediction_frame = self._build_prediction_frame(
            training_data["test_frame"], predictions
        )
        final_loss_log = self._build_final_loss_log(
            y_train=y_train,
            train_predictions=train_predictions,
            y_test=y_test,
            test_predictions=predictions,
        )
        confusion_matrix = self._build_delay_confusion_matrix(y_test, predictions)
        accuracy_metrics = self._accuracy_metrics(y_test, predictions)

        result = {
            "model_name": model_name,
            "model": model,
            "metrics": self._regression_metrics(y_test, predictions),
            "accuracy_metrics": accuracy_metrics,
            "loss_log": final_loss_log.copy(),
            "final_loss_log": final_loss_log,
            "confusion_matrix": confusion_matrix,
            "confusion_matrix_normalized": self._normalize_confusion_matrix(
                confusion_matrix
            ),
            "feature_columns": training_data["feature_columns"],
            "target_column": self.target_column,
            "train_rows": len(training_data["train_frame"]),
            "test_rows": len(training_data["test_frame"]),
            "predictions": prediction_frame,
        }
        if extra:
            result.update(extra)
        return result

    def _save_model_performance_outputs(
        self,
        model_key: str,
        result: dict,
        output_dir: Path,
    ) -> None:
        self._plot_confusion_matrix(
            confusion_matrix=result["confusion_matrix"],
            output_path=output_dir / f"{model_key}_confusion_matrix.png",
            title=f"{model_key.upper()} Delay-Band Confusion Matrix",
        )
        self._plot_loss_chart(
            loss_log=result["loss_log"],
            final_loss_log=result["final_loss_log"],
            output_path=output_dir / f"{model_key}_loss.png",
            title=f"{model_key.upper()} Train vs Test Loss",
        )
        self._plot_accuracy_chart(
            accuracy_metrics=result["accuracy_metrics"],
            output_path=output_dir / f"{model_key}_accuracy.png",
            title=f"{model_key.upper()} Accuracy Summary",
        )

    def _save_summary_charts(self, summary: pd.DataFrame, output_dir: Path) -> None:
        self._plot_model_error_comparison(
            summary=summary,
            output_path=output_dir / "model_error_comparison.png",
        )
        self._plot_model_accuracy_comparison(
            summary=summary,
            output_path=output_dir / "model_accuracy_comparison.png",
        )

    @staticmethod
    def _plot_confusion_matrix(
        confusion_matrix: pd.DataFrame,
        output_path: Path,
        title: str,
    ) -> None:
        plt = PredictiveModelling._matplotlib_pyplot()
        counts = confusion_matrix.to_numpy()
        row_totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=float),
            where=row_totals != 0,
        )

        labels = [
            label.replace("actual_", "").replace("predicted_", "").replace("_", "\n")
            for label in confusion_matrix.index
        ]
        fig, ax = plt.subplots(figsize=(11, 8))
        image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, pad=18)
        ax.set_xlabel("Predicted delay band")
        ax.set_ylabel("Actual delay band")
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels=labels)

        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                value = counts[row, col]
                percent = percentages[row, col]
                text_color = "white" if percent >= 0.45 else "black"
                ax.text(
                    col,
                    row,
                    f"{value}\n{percent:.0%}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Share of actual band")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_loss_chart(
        loss_log: pd.DataFrame,
        final_loss_log: pd.DataFrame,
        output_path: Path,
        title: str,
    ) -> None:
        plt = PredictiveModelling._matplotlib_pyplot()
        final_row = final_loss_log.iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

        if {"train_loss_mse_scaled", "test_loss_mse_scaled"}.issubset(
            loss_log.columns
        ):
            axes[0].plot(
                loss_log["epoch"],
                loss_log["train_loss_mse_scaled"],
                marker="o",
                label="Train MSE",
            )
            axes[0].plot(
                loss_log["epoch"],
                loss_log["test_loss_mse_scaled"],
                marker="o",
                label="Test MSE",
            )
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Scaled MSE")
            axes[0].set_title("Learning curve")
            axes[0].legend()
        else:
            axes[0].bar(
                ["Train", "Test"],
                [final_row["train_loss_mse"], final_row["test_loss_mse"]],
                color=["#4C78A8", "#F58518"],
            )
            axes[0].set_ylabel("MSE")
            axes[0].set_title("Final loss")

        metric_labels = ["Train RMSE", "Test RMSE", "Train MAE", "Test MAE"]
        metric_values = [
            final_row["train_rmse_minutes"],
            final_row["test_rmse_minutes"],
            final_row["train_mae_minutes"],
            final_row["test_mae_minutes"],
        ]
        bars = axes[1].bar(
            metric_labels,
            metric_values,
            color=["#4C78A8", "#F58518", "#72B7B2", "#E45756"],
        )
        axes[1].set_ylabel("Minutes")
        axes[1].set_title("Final error")
        axes[1].tick_params(axis="x", rotation=25)
        for bar in bars:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_accuracy_chart(
        accuracy_metrics: dict,
        output_path: Path,
        title: str,
    ) -> None:
        plt = PredictiveModelling._matplotlib_pyplot()
        labels = ["Delay band", "Within 5 min", "Within 10 min", "Within 15 min"]
        values = [
            accuracy_metrics["bucket_accuracy"],
            accuracy_metrics["within_5_min_accuracy"],
            accuracy_metrics["within_10_min_accuracy"],
            accuracy_metrics["within_15_min_accuracy"],
        ]

        fig, ax = plt.subplots(figsize=(8, 4.8))
        bars = ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#54A24B", "#B279A2"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.yaxis.set_major_formatter(PredictiveModelling._percent_formatter())
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_model_error_comparison(summary: pd.DataFrame, output_path: Path) -> None:
        plt = PredictiveModelling._matplotlib_pyplot()
        x = np.arange(len(summary))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, summary["mae_minutes"], width, label="MAE")
        ax.bar(x + width / 2, summary["rmse_minutes"], width, label="RMSE")
        ax.set_xticks(x, summary["model"])
        ax.set_ylabel("Minutes")
        ax.set_title("Model Error Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_model_accuracy_comparison(
        summary: pd.DataFrame, output_path: Path
    ) -> None:
        plt = PredictiveModelling._matplotlib_pyplot()
        x = np.arange(len(summary))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            x - width,
            summary["bucket_accuracy"],
            width,
            label="Delay-band accuracy",
        )
        ax.bar(x, summary["within_5_min_accuracy"], width, label="Within 5 min")
        ax.bar(
            x + width,
            summary["within_10_min_accuracy"],
            width,
            label="Within 10 min",
        )
        ax.set_xticks(x, summary["model"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.yaxis.set_major_formatter(PredictiveModelling._percent_formatter())
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _matplotlib_pyplot():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    @staticmethod
    def _percent_formatter():
        from matplotlib.ticker import PercentFormatter

        return PercentFormatter(xmax=1.0)

    @staticmethod
    def _build_final_loss_log(
        y_train: np.ndarray,
        train_predictions: np.ndarray,
        y_test: np.ndarray,
        test_predictions: np.ndarray,
    ) -> pd.DataFrame:
        train_errors = train_predictions - y_train
        test_errors = test_predictions - y_test
        return pd.DataFrame(
            [
                {
                    "epoch": "final",
                    "train_loss_mse": float(np.mean(train_errors**2)),
                    "train_rmse_minutes": float(np.sqrt(np.mean(train_errors**2))),
                    "train_mae_minutes": float(np.mean(np.abs(train_errors))),
                    "test_loss_mse": float(np.mean(test_errors**2)),
                    "test_rmse_minutes": float(np.sqrt(np.mean(test_errors**2))),
                    "test_mae_minutes": float(np.mean(np.abs(test_errors))),
                }
            ]
        )

    @staticmethod
    def _build_neural_loss_log(history: dict) -> pd.DataFrame:
        loss_log = pd.DataFrame(history).reset_index()
        loss_log["epoch"] = loss_log["index"] + 1
        loss_log = loss_log.drop(columns=["index"])
        return loss_log.rename(
            columns={
                "loss": "train_loss_mse_scaled",
                "mae": "train_mae_scaled",
                "val_loss": "test_loss_mse_scaled",
                "val_mae": "test_mae_scaled",
            }
        )

    def _build_delay_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        from sklearn.metrics import confusion_matrix

        actual_classes = self._delay_classes(y_true)
        predicted_classes = self._delay_classes(y_pred)
        matrix = confusion_matrix(
            actual_classes,
            predicted_classes,
            labels=self.DELAY_BUCKET_LABELS,
        )
        return pd.DataFrame(
            matrix,
            index=[f"actual_{label}" for label in self.DELAY_BUCKET_LABELS],
            columns=[f"predicted_{label}" for label in self.DELAY_BUCKET_LABELS],
        )

    @staticmethod
    def _normalize_confusion_matrix(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
        row_totals = confusion_matrix.sum(axis=1).replace(0, np.nan)
        return confusion_matrix.div(row_totals, axis=0).fillna(0)

    def _delay_classes(self, delay_minutes: np.ndarray) -> pd.Series:
        return pd.cut(
            pd.Series(delay_minutes),
            bins=self.DELAY_BUCKET_BINS,
            labels=self.DELAY_BUCKET_LABELS,
            include_lowest=True,
        ).astype("string")

    def _accuracy_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        absolute_errors = np.abs(y_pred - y_true)
        actual_classes = self._delay_classes(y_true)
        predicted_classes = self._delay_classes(y_pred)
        return {
            "bucket_accuracy": float((actual_classes == predicted_classes).mean()),
            "within_5_min_accuracy": float((absolute_errors <= 5).mean()),
            "within_10_min_accuracy": float((absolute_errors <= 10).mean()),
            "within_15_min_accuracy": float((absolute_errors <= 15).mean()),
        }

    def _build_prediction_frame(
        self, test_frame: pd.DataFrame, predicted_delay_minutes: np.ndarray
    ) -> pd.DataFrame:
        display_columns = [
            column
            for column in [
                "date_of_service",
                "current_location",
                "destination",
                "planned_destination_arrival_datetime",
                "target_arrival_datetime",
                self.target_column,
            ]
            if column in test_frame.columns
        ]
        prediction_frame = test_frame[display_columns].copy().reset_index(drop=True)
        prediction_frame["predicted_arrival_delay_minutes"] = predicted_delay_minutes

        if "planned_destination_arrival_datetime" in prediction_frame.columns:
            prediction_frame["predicted_arrival_datetime"] = (
                prediction_frame["planned_destination_arrival_datetime"]
                + pd.to_timedelta(predicted_delay_minutes, unit="m")
            )

        prediction_frame["arrival_delay_error_minutes"] = (
            prediction_frame["predicted_arrival_delay_minutes"]
            - prediction_frame[self.target_column]
        )
        return prediction_frame

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        errors = y_pred - y_true
        return {
            "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
            "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mean_error_minutes": float(np.mean(errors)),
            "median_absolute_error_minutes": float(np.median(np.abs(errors))),
        }

    @staticmethod
    def _clean_station_code(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip().str.upper()

    @staticmethod
    def _minutes_between(end: pd.Series, start: pd.Series) -> pd.Series:
        return (end - start).dt.total_seconds() / 60

    def _add_journey_datetimes(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values(["rid", "_row_order"], kind="stable").reset_index(
            drop=True
        )

        for column in self.TIME_COLUMNS:
            data[column.replace("_time", "_datetime")] = (
                self._combine_service_date_and_time_by_train(data, column)
            )

        data["planned_current_datetime"] = data[
            "planned_departure_datetime"
        ].combine_first(data["planned_arrival_datetime"])
        data["actual_current_datetime"] = data[
            "actual_departure_datetime"
        ].combine_first(data["actual_arrival_datetime"])
        data["planned_current_minutes_of_day"] = self._minutes_since_midnight(
            data["planned_current_datetime"], data["date_of_service"]
        )
        data["actual_current_minutes_of_day"] = self._minutes_since_midnight(
            data["actual_current_datetime"], data["date_of_service"]
        )
        data["planned_destination_arrival_minutes_of_day"] = np.nan
        return data

    def _combine_service_date_and_time_by_train(
        self, data: pd.DataFrame, time_column: str
    ) -> pd.Series:
        datetimes = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")

        for _, journey in data.groupby("rid", sort=False):
            rollover_days = 0
            previous_minutes: float | None = None

            for index, value in journey[time_column].items():
                if pd.isna(value):
                    continue

                minutes = self._time_to_minutes(value)
                if (
                    previous_minutes is not None
                    and minutes < previous_minutes - (12 * 60)
                ):
                    rollover_days += 1

                previous_minutes = minutes
                service_date = data.at[index, "date_of_service"]
                datetimes.at[index] = service_date + pd.to_timedelta(
                    minutes + (rollover_days * 24 * 60), unit="m"
                )

        return datetimes

    @staticmethod
    def _time_to_minutes(value) -> float:
        if hasattr(value, "hour") and hasattr(value, "minute"):
            return value.hour * 60 + value.minute + (value.second / 60)

        parsed = pd.to_datetime(str(value), errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Could not parse time value: {value!r}")
        return parsed.hour * 60 + parsed.minute + (parsed.second / 60)

    def _add_stop_level_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data["stop_number"] = data.groupby("rid").cumcount()
        data["total_stops_in_service"] = data.groupby("rid")["location"].transform(
            "size"
        )
        data["arrival_delay_minutes"] = self._minutes_between(
            data["actual_arrival_datetime"], data["planned_arrival_datetime"]
        )
        data["departure_delay_minutes"] = self._minutes_between(
            data["actual_departure_datetime"], data["planned_departure_datetime"]
        )
        data["current_delay_minutes"] = self._minutes_between(
            data["actual_current_datetime"], data["planned_current_datetime"]
        )
        data["planned_dwell_minutes"] = self._minutes_between(
            data["planned_departure_datetime"], data["planned_arrival_datetime"]
        )
        data["actual_dwell_minutes"] = self._minutes_between(
            data["actual_departure_datetime"], data["actual_arrival_datetime"]
        )
        return data

    @staticmethod
    def _fill_model_missing_values(data: pd.DataFrame) -> pd.DataFrame:
        missing_indicator_columns = [
            "arrival_delay_minutes",
            "departure_delay_minutes",
            "origin_departure_delay_minutes",
            "planned_dwell_minutes",
            "actual_dwell_minutes",
        ]
        for column in missing_indicator_columns:
            data[f"{column}_missing"] = data[column].isna().astype(int)

        data["arrival_delay_minutes"] = data["arrival_delay_minutes"].fillna(
            data["current_delay_minutes"]
        )
        data["departure_delay_minutes"] = data["departure_delay_minutes"].fillna(
            data["current_delay_minutes"]
        )
        data["origin_departure_delay_minutes"] = data[
            "origin_departure_delay_minutes"
        ].fillna(0)
        data["planned_dwell_minutes"] = data["planned_dwell_minutes"].fillna(0)
        data["actual_dwell_minutes"] = data["actual_dwell_minutes"].fillna(0)
        return data

    def _add_calendar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        service_date = data["date_of_service"]
        data["service_month"] = service_date.dt.month
        data["service_day_of_month"] = service_date.dt.day
        data["service_day_of_week"] = service_date.dt.dayofweek
        data["service_day_of_year"] = service_date.dt.dayofyear
        data["is_weekend"] = data["service_day_of_week"].isin([5, 6]).astype(int)
        data["service_day_of_week_sin"] = self._cyclical_sin(
            data["service_day_of_week"], 7
        )
        data["service_day_of_week_cos"] = self._cyclical_cos(
            data["service_day_of_week"], 7
        )
        data["service_day_of_year_sin"] = self._cyclical_sin(
            data["service_day_of_year"], 366
        )
        data["service_day_of_year_cos"] = self._cyclical_cos(
            data["service_day_of_year"], 366
        )
        return data

    def _add_time_of_day_features(
        self, data: pd.DataFrame, source_col: str, prefix: str
    ) -> pd.DataFrame:
        minutes = self._minutes_since_midnight(
            data[source_col], data["date_of_service"]
        )
        minutes_column = f"planned_{prefix}_minutes_of_day"
        if prefix == "destination":
            minutes_column = "planned_destination_arrival_minutes_of_day"

        data[minutes_column] = minutes
        data[f"{prefix}_is_peak"] = self._is_peak_time(minutes).astype(int)

        sin_prefix = (
            "planned_destination_arrival_time"
            if prefix == "destination"
            else f"planned_{prefix}_time"
        )
        data[f"{sin_prefix}_sin"] = self._cyclical_sin(minutes, 24 * 60)
        data[f"{sin_prefix}_cos"] = self._cyclical_cos(minutes, 24 * 60)
        return data

    @staticmethod
    def _minutes_since_midnight(
        datetimes: pd.Series, service_dates: pd.Series
    ) -> pd.Series:
        return (datetimes - service_dates).dt.total_seconds() / 60

    @staticmethod
    def _is_peak_time(minutes: pd.Series) -> pd.Series:
        minutes_in_day = minutes % (24 * 60)
        morning_peak = minutes_in_day.between(7 * 60, (9 * 60) + 30)
        evening_peak = minutes_in_day.between(16 * 60, 19 * 60)
        return morning_peak | evening_peak

    @staticmethod
    def _cyclical_sin(values: Iterable[float], period: int) -> pd.Series:
        return np.sin(2 * np.pi * pd.Series(values) / period)

    @staticmethod
    def _cyclical_cos(values: Iterable[float], period: int) -> pd.Series:
        return np.cos(2 * np.pi * pd.Series(values) / period)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare WEY-to-WAT data and train arrival-delay models."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training cycles and save performance outputs.",
    )
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="Use larger default training settings instead of quick smoke settings.",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Train only linear and KNN models.",
    )
    parser.add_argument(
        "--destination",
        default="WAT",
        help="Passenger destination station code to predict.",
    )
    parser.add_argument(
        "--output-dir",
        default="./model_performance",
        help="Directory for performance chart images.",
    )
    args = parser.parse_args()

    predictive_model = PredictiveModelling(destination=args.destination)
    stops = predictive_model.preprocess_data()
    features = predictive_model.feature_engineering()
    print(f"Preprocessed stop rows: {stops.shape}")
    print(f"Engineered modelling rows: {features.shape}")
    preview_columns = [
        "current_location",
        "destination",
        "current_delay_minutes",
        "scheduled_minutes_to_destination",
        "target_arrival_delay_minutes",
    ]
    print(features[preview_columns].head().to_string(index=False))

    if args.train:
        predictive_model.run_training_cycles(
            destination=args.destination,
            output_dir=args.output_dir,
            quick_run=not args.full_training,
            include_neural_models=not args.skip_neural,
        )
    else:
        print(
            "\nRun training with: "
            "uv run python task_2_predictive_modelling.py --train"
        )
