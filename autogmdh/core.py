# autogmdh/core.py

import os
import json
import shutil
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import gc
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# SETTINGS (tweak for compute/performance)
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Two-stage NAS controls
POLY_SCREEN_TOP_PCT = 0.30    # fraction of pairs to keep for NAS
NAS_MAX_TRIALS = 8            # increase to 12-24 if you have compute
NAS_EPOCHS = 80               # per trial training epochs budget
NAS_PATIENCE = 10

# NAS search expansion (wider search space)
NAS_ALLOW_DEEP = True         # allow up to 4 layers if True

# Weighted hybrid threshold
WEIGHTED_THRESHOLD_RATIO = 0.10  # if NN MSE within 10% of poly MSE, use weighted ensemble

# GMDH/hybrid controls
MAX_LAYERS = 4
SELECTION_PERCENTAGE = 0.5
MAX_FEATURES = 10
LAYER_PATIENCE = 2
MIN_IMPROVEMENT = 0.005  # smaller threshold to allow deeper search
ALPHA_RIDGE = 1.0

# Artifact folder
ARTIFACT_ROOT = 'pair_nas_artifacts_v2'
os.makedirs(ARTIFACT_ROOT, exist_ok=True)

# Suppress noisy warnings and TF info logs
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer 'adam'")


# ----------------------------
# Helper functions
# ----------------------------
def build_poly_features(X_pair: np.ndarray) -> np.ndarray:
    """Return polynomial features for a 2-feature input: x1, x2, x1^2, x2^2, x1*x2."""
    x1, x2 = X_pair[:, 0], X_pair[:, 1]
    return np.column_stack([x1, x2, x1 ** 2, x2 ** 2, x1 * x2])


def make_expanded_pair_hypermodel(input_shape=(2,)):
    """
    Expanded HyperModel for per-pair NAS with a richer search space.
    Keeps hyperparameter ranges reasonable to balance capacity and compute.
    """

    class PairHyperModel(kt.HyperModel):
        def build(self, hp):
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=input_shape))

            # variable depth: 1 to 4 layers if NAS_ALLOW_DEEP else 1-2
            max_layers = 4 if NAS_ALLOW_DEEP else 2
            n_layers = hp.Int(
                "n_layers",
                1,
                max_layers if NAS_ALLOW_DEEP else min(2, max_layers),
                step=1,
            )

            for i in range(n_layers):
                units = hp.Choice(f'units_{i}', [16, 32, 64, 128])
                act = hp.Choice(f'act_{i}', ['relu', 'tanh', 'swish'])
                reg = hp.Float(f'l2_{i}', 0.0, 1e-2, step=1e-4)
                model.add(
                    keras.layers.Dense(
                        units,
                        activation=act,
                        kernel_regularizer=keras.regularizers.l2(reg),
                        kernel_initializer=keras.initializers.GlorotUniform(SEED + i),
                    )
                )
                if hp.Boolean(f'dropout_{i}', default=False):
                    rate = hp.Float(f'dropout_rate_{i}', 0.0, 0.4, step=0.1)
                    model.add(keras.layers.Dropout(rate))

            model.add(keras.layers.Dense(1))
            lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
            opt_choice = hp.Choice('opt', ['adam', 'nadam'])
            optimizer = (
                keras.optimizers.Adam(learning_rate=lr)
                if opt_choice == 'adam'
                else keras.optimizers.Nadam(learning_rate=lr)
            )
            model.compile(optimizer=optimizer, loss='mse')
            return model

    return PairHyperModel()


def search_pair_nas_artifacts(
    X_sub_pair,
    y_sub,
    X_val_pair,
    y_val,
    layer_idx,
    pair_tuple,
    max_trials=NAS_MAX_TRIALS,
    max_epochs=NAS_EPOCHS,
    patience=NAS_PATIENCE,
):
    """
    Expanded NAS (RandomSearch) with artifact saving and safe weight copying.
    Returns: (clean_model_or_None, validation_mse)
    """
    project_name = f'l{layer_idx}_p{pair_tuple[0]}_{pair_tuple[1]}'
    artifact_dir = os.path.join(ARTIFACT_ROOT, project_name)
    if os.path.exists(artifact_dir):
        shutil.rmtree(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    hyper = make_expanded_pair_hypermodel(input_shape=(2,))
    tuner = kt.RandomSearch(
        hyper,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory=artifact_dir,
        project_name='tuner',
        overwrite=True,
    )

    es = EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0
    )
    try:
        tuner.search(
            X_sub_pair,
            y_sub,
            epochs=max_epochs,
            validation_data=(X_val_pair, y_val),
            callbacks=[es],
            verbose=0,
        )
    except Exception as e:
        print(f"[NAS ERROR] tuner.search failed for pair {pair_tuple} layer {layer_idx}: {e}")
        try:
            del tuner
        except Exception:
            pass
        K.clear_session()
        gc.collect()
        return None, float('inf')

    try:
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_trained = tuner.get_best_models(num_models=1)[0]
        best_weights = best_trained.get_weights()
    except Exception as e:
        print(f"[NAS WARN] extraction failed for {pair_tuple}: {e}")
        try:
            del tuner
        except Exception:
            pass
        K.clear_session()
        gc.collect()
        return None, float('inf')

    # Save best_hyperparameters
    try:
        with open(os.path.join(artifact_dir, 'best_hp.json'), 'w') as f:
            json.dump(best_hp.values, f)
    except Exception:
        pass

    # Build a fresh model and set its weights
    clean_model = hyper.build(best_hp)
    try:
        clean_model.set_weights(best_weights)
        try:
            clean_model.save_weights(os.path.join(artifact_dir, 'weights.h5'))
        except Exception:
            pass
    except Exception as e:
        print(f"[NAS WARN] set_weights failed for {pair_tuple}: {e}")

    # Evaluate validation MSE
    try:
        y_pred_val = clean_model.predict(X_val_pair, verbose=0).flatten()
        val_mse = mean_squared_error(y_val, y_pred_val)
    except Exception as e:
        print(f"[NAS WARN] predict failed for {pair_tuple}: {e}")
        val_mse = float('inf')

    # Cleanup
    try:
        del best_trained
    except Exception:
        pass
    try:
        del tuner
    except Exception:
        pass
    K.clear_session()
    gc.collect()

    return clean_model, val_mse


# ----------------------------
# Two-stage GMDH + weighted hybrid core
# ----------------------------
class SelfOrganizingHybridAdvanced:
    """
    Core model that expects:
    - X: scaled features
    - y: scaled target

    It returns scaled predictions; scaling to original space is handled outside.
    """

    def __init__(
        self,
        max_layers=MAX_LAYERS,
        selection_percentage=SELECTION_PERCENTAGE,
        max_features=MAX_FEATURES,
        patience=LAYER_PATIENCE,
        min_improvement=MIN_IMPROVEMENT,
        alpha=ALPHA_RIDGE,
        verbose=True,
        poly_top_pct=POLY_SCREEN_TOP_PCT,
    ):
        self.max_layers = max_layers
        self.selection_percentage = selection_percentage
        self.max_features = max_features
        self.patience = patience
        self.min_improvement = min_improvement
        self.alpha = alpha
        self.verbose = verbose
        self.poly_top_pct = poly_top_pct

        self.layers = []
        self.norm_params = []
        self.layer_stats = []
        self.best_mse = float('inf')
        self.no_improvement_count = 0
        self.final_aggregator = None

    def fit(self, X, y):
        # X and y are already scaled
        X_sub, X_val, y_sub, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        current_X_sub, current_X_val = X_sub, X_val

        for layer_idx in range(self.max_layers):
            n_features = current_X_sub.shape[1]
            if n_features < 2:
                if self.verbose:
                    print(f"[STOP] Layer {layer_idx + 1}: only {n_features} features remain")
                break

            pairs = list(combinations(range(n_features), 2))
            if self.verbose:
                print(f"[L{layer_idx + 1}] {len(pairs)} pairs from {n_features} features")

            # Stage 1: Polynomial screening
            poly_results = []
            for (i, j) in pairs:
                X_sub_pair = current_X_sub[:, [i, j]]
                X_val_pair = current_X_val[:, [i, j]]
                try:
                    poly_sub = build_poly_features(X_sub_pair)
                    reg = Ridge(alpha=self.alpha, fit_intercept=True).fit(
                        poly_sub, y_sub
                    )
                    y_val_pred = reg.predict(build_poly_features(X_val_pair))
                    mse = mean_squared_error(y_val, y_val_pred)
                except Exception:
                    reg = None
                    mse = float('inf')
                poly_results.append(
                    {'pair': (i, j), 'poly_model': reg, 'poly_mse': mse}
                )

            poly_sorted = sorted(poly_results, key=lambda x: x['poly_mse'])
            keep_k = max(1, int(len(poly_sorted) * self.poly_top_pct))
            top_poly = poly_sorted[:keep_k]
            if self.verbose:
                print(
                    f"[L{layer_idx + 1}] Stage1: kept top {keep_k}/{len(poly_sorted)} pairs for NAS"
                )

            # Stage 2: NAS on top pairs
            models = []
            layer_stats = {
                'layer': layer_idx + 1,
                'input_features': n_features,
                'pairs_total': len(pairs),
                'poly_count': len(pairs),
                'nas_count': 0,
                'poly_selected': 0,
                'nn_selected': 0,
                'poly_mse_total': 0.0,
                'nn_mse_total': 0.0,
                'best_model_type': '',
                'best_mse': float('inf'),
            }

            for entry in top_poly:
                i, j = entry['pair']
                poly_model = entry['poly_model']
                poly_mse = entry['poly_mse']
                X_sub_pair = current_X_sub[:, [i, j]]
                X_val_pair = current_X_val[:, [i, j]]

                # Run NAS for this pair
                nn_model, nn_mse = search_pair_nas_artifacts(
                    X_sub_pair,
                    y_sub,
                    X_val_pair,
                    y_val,
                    layer_idx=layer_idx,
                    pair_tuple=(i, j),
                    max_trials=NAS_MAX_TRIALS,
                )

                layer_stats['poly_mse_total'] += poly_mse
                if np.isfinite(nn_mse):
                    layer_stats['nn_mse_total'] += nn_mse
                    layer_stats['nas_count'] += 1

                chosen_type = None
                chosen_model = None
                chosen_mse = None

                # If no NN found, choose polynomial
                if not np.isfinite(nn_mse):
                    chosen_type = 'poly'
                    chosen_model = poly_model
                    chosen_mse = poly_mse
                    layer_stats['poly_selected'] += 1
                    if self.verbose:
                        print(
                            f"[L{layer_idx + 1}] Pair ({i},{j}) - NN fail: selected poly (MSE: {chosen_mse:.4f})"
                        )
                else:
                    # NN clearly better
                    if nn_mse < poly_mse * (1 - WEIGHTED_THRESHOLD_RATIO):
                        chosen_type = 'nn'
                        chosen_model = nn_model
                        chosen_mse = nn_mse
                        layer_stats['nn_selected'] += 1
                        if self.verbose:
                            print(
                                f"[L{layer_idx + 1}] Pair ({i},{j}): selected NN (MSE: {chosen_mse:.4f})"
                            )
                    # Close: hybrid
                    elif abs(nn_mse - poly_mse) <= poly_mse * WEIGHTED_THRESHOLD_RATIO:
                        w_poly = nn_mse / (poly_mse + nn_mse + 1e-12)
                        w_nn = poly_mse / (poly_mse + nn_mse + 1e-12)
                        chosen_type = 'hybrid'
                        chosen_model = {
                            'poly': poly_model,
                            'nn': nn_model,
                            'w_poly': w_poly,
                            'w_nn': w_nn,
                        }
                        try:
                            val_poly = poly_model.predict(
                                build_poly_features(X_val_pair)
                            )
                        except Exception:
                            val_poly = np.zeros(X_val_pair.shape[0])
                        try:
                            val_nn = nn_model.predict(
                                X_val_pair, verbose=0
                            ).flatten()
                        except Exception:
                            val_nn = np.zeros(X_val_pair.shape[0])
                        ensemble_pred = w_poly * val_poly + w_nn * val_nn
                        chosen_mse = mean_squared_error(y_val, ensemble_pred)
                        layer_stats['poly_selected'] += 0.5
                        layer_stats['nn_selected'] += 0.5
                        if self.verbose:
                            print(
                                f"[L{layer_idx + 1}] Pair ({i},{j}): selected HYBRID "
                                f"(poly_mse={poly_mse:.4f}, nn_mse={nn_mse:.4f}, hybrid_mse={chosen_mse:.4f})"
                            )
                    else:
                        chosen_type = 'poly'
                        chosen_model = poly_model
                        chosen_mse = poly_mse
                        layer_stats['poly_selected'] += 1
                        if self.verbose:
                            print(
                                f"[L{layer_idx + 1}] Pair ({i},{j}): selected poly (MSE: {chosen_mse:.4f})"
                            )

                # compute normalization params on training-sub predictions
                try:
                    if chosen_type == 'poly':
                        y_pred_sub = chosen_model.predict(
                            build_poly_features(X_sub_pair)
                        )
                    elif chosen_type == 'nn':
                        y_pred_sub = chosen_model.predict(
                            X_sub_pair, verbose=0
                        ).flatten()
                    else:
                        p_sub = chosen_model['poly'].predict(
                            build_poly_features(X_sub_pair)
                        )
                        n_sub = chosen_model['nn'].predict(
                            X_sub_pair, verbose=0
                        ).flatten()
                        y_pred_sub = (
                            chosen_model['w_poly'] * p_sub
                            + chosen_model['w_nn'] * n_sub
                        )
                    mean = float(np.mean(y_pred_sub))
                    std = float(np.std(y_pred_sub)) + 1e-8
                except Exception:
                    mean, std = 0.0, 1.0

                models.append(
                    {
                        'type': chosen_type,
                        'model': chosen_model,
                        'pair': (i, j),
                        'mse': chosen_mse,
                        'norm_params': (mean, std),
                    }
                )

            layer_stats['avg_poly_mse'] = (
                layer_stats['poly_mse_total'] / layer_stats['poly_count']
                if layer_stats['poly_count']
                else float('inf')
            )
            layer_stats['avg_nn_mse'] = (
                layer_stats['nn_mse_total'] / layer_stats['nas_count']
                if layer_stats['nas_count']
                else float('inf')
            )
            self.layer_stats.append(layer_stats)

            print("\n" + "=" * 50)
            print(f"LAYER {layer_idx + 1} SUMMARY")
            print("=" * 50)
            print(f"Input features: {layer_stats['input_features']}")
            print(f"Pairs processed: {layer_stats['pairs_total']}")
            print(
                f"Polynomial models: {layer_stats['poly_count']} created, "
                f"approx selected: {layer_stats['poly_selected']:.1f}"
            )
            print(
                f"Neural networks (NAS): {layer_stats['nas_count']} created, "
                f"approx selected: {layer_stats['nn_selected']:.1f}"
            )
            print(
                f"Avg poly mse: {layer_stats['avg_poly_mse']:.4f}, "
                f"avg nn mse: {layer_stats['avg_nn_mse']:.4f}"
            )
            print("=" * 50 + "\n")

            # select top-k
            models = [m for m in models if np.isfinite(m['mse'])]
            if not models:
                break
            models.sort(key=lambda x: x['mse'])
            k = max(
                1, min(self.max_features, int(len(models) * self.selection_percentage))
            )
            selected_models = models[:k]
            best_layer_mse = selected_models[0]['mse']

            # create new layer features
            new_X_sub = np.zeros((X_sub.shape[0], k))
            new_X_val = np.zeros((X_val.shape[0], k))
            layer_norms = []

            for idx, info in enumerate(selected_models):
                i, j = info['pair']
                X_pair_sub = current_X_sub[:, [i, j]]
                X_pair_val = current_X_val[:, [i, j]]
                try:
                    if info['type'] == 'poly':
                        pred_sub = info['model'].predict(
                            build_poly_features(X_pair_sub)
                        )
                        pred_val = info['model'].predict(
                            build_poly_features(X_pair_val)
                        )
                    elif info['type'] == 'nn':
                        pred_sub = info['model'].predict(
                            X_pair_sub, verbose=0
                        ).flatten()
                        pred_val = info['model'].predict(
                            X_pair_val, verbose=0
                        ).flatten()
                    else:
                        p_sub = info['model']['poly'].predict(
                            build_poly_features(X_pair_sub)
                        )
                        n_sub = info['model']['nn'].predict(
                            X_pair_sub, verbose=0
                        ).flatten()
                        pred_sub = (
                            info['model']['w_poly'] * p_sub
                            + info['model']['w_nn'] * n_sub
                        )
                        p_val = info['model']['poly'].predict(
                            build_poly_features(X_pair_val)
                        )
                        n_val = info['model']['nn'].predict(
                            X_pair_val, verbose=0
                        ).flatten()
                        pred_val = (
                            info['model']['w_poly'] * p_val
                            + info['model']['w_nn'] * n_val
                        )
                except Exception:
                    pred_sub = np.zeros(X_pair_sub.shape[0])
                    pred_val = np.zeros(X_pair_val.shape[0])

                mean, std = info['norm_params']
                new_X_sub[:, idx] = (pred_sub - mean) / std
                new_X_val[:, idx] = (pred_val - mean) / std
                layer_norms.append((mean, std))

            self.layers.append(selected_models)
            self.norm_params.append(layer_norms)

            # early stopping by improvement
            if self.best_mse == float('inf'):
                improvement = float('inf')
            else:
                improvement = (self.best_mse - best_layer_mse) / self.best_mse

            if improvement < self.min_improvement:
                self.no_improvement_count += 1
                if self.verbose:
                    print(
                        f"[L{layer_idx + 1}] No sufficient improvement ({improvement:.2%})"
                    )
                if self.no_improvement_count >= self.patience:
                    print(
                        f"[L{layer_idx + 1}] Early stopping across layers triggered."
                    )
                    break
            else:
                self.no_improvement_count = 0
                self.best_mse = best_layer_mse

            current_X_sub, current_X_val = new_X_sub, new_X_val

        # Train final aggregator: stacked ensemble (NN + ElasticNet + GBoost)
        self.print_model_selection_summary()
        if self.layers:
            final_features = self._transform_features(X)
            try:
                agg_nn = keras.Sequential(
                    [
                        keras.layers.Input(shape=(final_features.shape[1],)),
                        keras.layers.Dense(128, activation='relu'),
                        keras.layers.Dropout(0.1),
                        keras.layers.Dense(32, activation='relu'),
                        keras.layers.Dense(1),
                    ]
                )
                agg_nn.compile(
                    optimizer=keras.optimizers.Adam(1e-3), loss='mse'
                )
                es = EarlyStopping(
                    monitor='loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=0,
                )
                agg_nn.fit(
                    final_features,
                    y,
                    epochs=400,
                    batch_size=16,
                    callbacks=[es],
                    verbose=0,
                )

                enet = ElasticNet()
                params = {
                    'alpha': [1e-3, 1e-2, 1e-1, 1.0],
                    'l1_ratio': [0.2, 0.5, 0.8],
                }
                gs = GridSearchCV(
                    enet,
                    params,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,
                )
                gs.fit(final_features, y)
                best_enet = gs.best_estimator_

                gbr = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=SEED,
                )
                gbr.fit(final_features, y)

                self.final_aggregator = ('stacked', (agg_nn, best_enet, gbr))
            except Exception as e:
                print(
                    f"[AGGREGATOR WARN] stacked aggregator failed: {e}. Falling back to Ridge."
                )
                ridge = Ridge(alpha=self.alpha).fit(final_features, y)
                self.final_aggregator = ('ridge', ridge)

    def _transform_features(self, X):
        current = X.copy()
        for layer_idx, layer_models in enumerate(self.layers):
            new = np.zeros((current.shape[0], len(layer_models)))
            for model_idx, model_info in enumerate(layer_models):
                i, j = model_info['pair']
                X_pair = current[:, [i, j]]
                try:
                    if model_info['type'] == 'poly':
                        y_pred = model_info['model'].predict(
                            build_poly_features(X_pair)
                        )
                    elif model_info['type'] == 'nn':
                        y_pred = model_info['model'].predict(
                            X_pair, verbose=0
                        ).flatten()
                    else:
                        p = model_info['model']['poly'].predict(
                            build_poly_features(X_pair)
                        )
                        n = model_info['model']['nn'].predict(
                            X_pair, verbose=0
                        ).flatten()
                        y_pred = (
                            model_info['model']['w_poly'] * p
                            + model_info['model']['w_nn'] * n
                        )
                except Exception:
                    y_pred = np.zeros(X_pair.shape[0])

                mean, std = self.norm_params[layer_idx][model_idx]
                new[:, model_idx] = (y_pred - mean) / std
            current = new
        return current

    def predict(self, X):
        """
        Returns scaled predictions (same scale as y given to fit()).
        """
        if not self.layers:
            raise RuntimeError("Model has not been trained")
        features = self._transform_features(X)
        typ, models = self.final_aggregator
        if typ == 'stacked':
            nn, enet, gbr = models
            y_nn = nn.predict(features).flatten()
            y_enet = enet.predict(features).flatten()
            y_gbr = gbr.predict(features).flatten()
            y_avg = (y_nn + y_enet + y_gbr) / 3.0
            return y_avg
        elif typ == 'ridge':
            y_ridge = models.predict(features).flatten()
            return y_ridge
        else:
            raise RuntimeError("Unknown aggregator type")

    def print_model_selection_summary(self):
        print("\n" + "=" * 60)
        print("FINAL MODEL SELECTION SUMMARY (ADVANCED)")
        print("=" * 60)
        total_poly = sum(s['poly_selected'] for s in self.layer_stats)
        total_nn = sum(s['nn_selected'] for s in self.layer_stats)
        total_pairs = sum(s['pairs_total'] for s in self.layer_stats)
        for s in self.layer_stats:
            print(
                f"Layer {s['layer']}: pairs {s['pairs_total']}, "
                f"poly sel {s['poly_selected']:.1f}, nn sel {s['nn_selected']:.1f}"
            )
        print(
            f"\nTotals: poly selected ~{total_poly:.1f}, "
            f"nn selected ~{total_nn:.1f}, pairs sum {total_pairs}"
        )
        print("=" * 60 + "\n")


# ----------------------------
# High-level wrapper (public API)
# ----------------------------
class AutoGMDHRegressor:
    """
    High-level regressor with a scikit-learn-like API.

    Usage:
        model = AutoGMDHRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
    """

    def __init__(self, **kwargs):
        self.core_params = kwargs
        self._x_scaler = None
        self._y_scaler = None
        self._core_model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self._x_scaler = StandardScaler().fit(X)
        X_scaled = self._x_scaler.transform(X)

        self._y_scaler = StandardScaler().fit(y)
        y_scaled = self._y_scaler.transform(y).flatten()

        self._core_model = SelfOrganizingHybridAdvanced(**self.core_params)
        self._core_model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        if self._core_model is None:
            raise RuntimeError("You must call fit() before predict().")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._x_scaler.transform(X)
        y_scaled_pred = self._core_model.predict(X_scaled)
        y_pred = self._y_scaler.inverse_transform(
            y_scaled_pred.reshape(-1, 1)
        ).flatten()
        return y_pred

    def score(self, X, y):
        from sklearn.metrics import r2_score

        y = np.asarray(y, dtype=np.float32)
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    @property
    def x_scaler(self):
        return self._x_scaler

    @property
    def y_scaler(self):
        return self._y_scaler

    @property
    def core_model(self):
        return self._core_model
