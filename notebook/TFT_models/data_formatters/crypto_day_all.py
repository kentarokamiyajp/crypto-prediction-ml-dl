# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Volatility dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class CryptoDayFormatter(GenericDataFormatter):
    """Defines and formats data for the crypto_day dataset.

    Attributes:
        column_definition: Defines input and data type of column used in the
        experiment.
        identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        ("btc_row_id", DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ("btc_id", DataTypes.CATEGORICAL, InputTypes.ID),
        ("btc_dt", DataTypes.DATE, InputTypes.TIME),
        ("btc_low", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_high", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_open", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ("btc_close", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_volume", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_macd", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_macd_single", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_rsi", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_bollinger_bands_sma", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_bollinger_bands_lower_band", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_bollinger_bands_upper_band", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_obv", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ichimoku_chikou_span", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ichimoku_kijun_sen", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ichimoku_tenkan_sen", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ichimoku_senkou_span_a", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ichimoku_senkou_span_b", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_stoch_oscillator", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_stoch_signal", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_stoch_percent_j", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_aroon_up", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_aroon_down", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_aroon_oscillator", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_sma5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_sma10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_sma30", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ema5", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ema10", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("btc_ema30", DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ("target", DataTypes.REAL_VALUED, InputTypes.TARGET),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, valid_boundary=900, test_boundary=1030):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
        df: Source data frame to split.
        valid_boundary: Starting index(e.g., year) for validation data
        test_boundary: Starting index(e.g., year) for test data

        Returns:
        Tuple of transformed (train, valid, test) data.
        """

        print("Formatting train-valid-test splits.")

        train = df.iloc[:valid_boundary]
        valid = df.iloc[valid_boundary - 7 : test_boundary]
        test = df.iloc[test_boundary - 7 :]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
        df: Data to use to calibrate scalers.
        """
        print("Setting scalers with training data...")

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
        df: Data frame to transform.

        Returns:
        Transformed data frame.

        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError("Scalers have not been set!")

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
        predictions: Dataframe of model predictions.

        Returns:
        Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {"forecast_time", "identifier"}:
                output[col] = self._target_scaler.inverse_transform(predictions[col].to_frame())

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            "total_time_steps": 30 + 7,
            "num_encoder_steps": 30,
            "num_epochs": 100,
            "early_stopping_patience": 5,
            "multiprocessing_workers": 5,
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            "dropout_rate": 0.3,
            "hidden_layer_size": 160,
            "learning_rate": 0.01,
            "minibatch_size": 32,
            "max_gradient_norm": 0.01,
            "num_heads": 1,
            "stack_size": 1,
        }

        return model_params
