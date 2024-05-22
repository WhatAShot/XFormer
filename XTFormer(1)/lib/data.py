import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union, cast, Dict, List, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from . import env, util
from .metrics import calculate_metrics as calculate_metrics_
from .util import TaskType

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = "__nan__"
CAT_RARE_VALUE = "__rare__"
Normalization = Literal["standard", "quantile"]
NumNanPolicy = Literal["drop-rows", "mean"]
CatNanPolicy = Literal["most_frequent"]
CatEncoding = Literal["one-hot", "counter"]
YPolicy = Literal["default"]


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


@dataclass(frozen=True)
class Dataset2:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    X_str: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    feature_names: Dict[str, List[str]]
    task_type: TaskType
    n_classes: Optional[int]
    name: Optional[str]
    prompt_infos: Optional[Dict[str, str]] = None
    prompt_ids: Optional[torch.Tensor] = None

    @classmethod
    def from_csv(
        cls,
        data_dir: Union[Path, str],
        data_name: str,
        max_cat_num: Optional[int] = None,
        min_y_frequency: Union[int, float] = 2,
        tt: Optional[str] = None,  # given task type if filename is not annotated
    ) -> "Dataset2":

        csv_file = Path(data_dir) / f"{data_name}.csv"

        df = pd.read_csv(csv_file)
        max_cat_num = max_cat_num or int(len(df) / 100)
        feat_types = [str(df.iloc[:, i].dtype) for i in range(df.shape[1] - 1)]
        feat_names = np.array(df.columns[:-1])

        X_num_idx, X_cat_idx, X_str_idx = [], [], []
        for i, ft in enumerate(feat_types):
            if "id" in feat_names[i].lower():
                if len(df.iloc[:, i].unique()) == len(df):
                    pass
                elif len(df.iloc[:, i].unique()) <= 100 or ft == "object":
                    X_cat_idx.append(i)
                else:
                    X_num_idx.append(i)
                continue
            if ft == "object":
                X_cat_idx.append(i)
            elif len(df.iloc[:, i].unique()) <= max_cat_num:
                X_cat_idx.append(i)
            else:
                if ft == "object":
                    continue
                X_num_idx.append(i)

        X_num, X_cat, X_str = None, None, None
        feature_names = {"num": [], "cat": [], "str": []}
        if len(X_num_idx) > 0:
            feature_names["num"] = feat_names[X_num_idx].tolist()
            X_num = {"train": df.iloc[:, X_num_idx].values.astype(np.float32)}
        if len(X_cat_idx) > 0:
            feature_names["cat"] = feat_names[X_cat_idx].tolist()
            X_cat = {"train": df.iloc[:, X_cat_idx].values.astype(object)}
        if len(X_str_idx) > 0:
            feature_names["str"] = feat_names[X_str_idx].tolist()
            X_str = {
                "train": df.iloc[:, X_str_idx].values.astype(np.str).astype(object)
            }

        def strlabeltransform(labels):
            label_mapping = {}
            encoded_labels = [
                label_mapping.setdefault(label, len(label_mapping)) for label in labels
            ]
            encoded_labels_array = np.array(encoded_labels, dtype=np.int64)
            return encoded_labels_array

        def remove_data(rm_mask):
            if any(rm_mask):
                ys["train"] = ys["train"][~rm_mask]
                if X_num is not None:
                    X_num["train"] = X_num["train"][~rm_mask]
                if X_cat is not None:
                    X_cat["train"] = X_cat["train"][~rm_mask]
                if X_str is not None:
                    X_str["train"] = X_str["train"][~rm_mask]

        def pure_task(name):
            if any(name.startswith(x) for x in ["bin", "mul", "reg"]):
                return name[:3]
            if any(name.startswith(x) for x in ["(bin)", "(mul)", "(reg)"]):
                return name[1:4]
            return None

        # y info
        task_type = (
            tt
            or {"bin": "binclass", "mul": "multiclass", "reg": "regression"}[
                pure_task(data_name)
            ]
        )

        # process y
        ys = {"train": df.iloc[:, -1].values}  # use iloc.values not values
        # check nan in y
        if ys["train"].dtype == object:
            remove_mask = ys["train"].astype(str) == "nan"

        else:
            remove_mask = np.isnan(ys["train"])
        remove_data(remove_mask)
        remove_ys = {}
        if task_type == "multiclass":
            y_counter = Counter(ys["train"])
            if min_y_frequency < 1:
                min_y_frequency = int(len(ys["train"]) * min_y_frequency)
            remove_ys = {k for k, v in y_counter.items() if v <= min_y_frequency}
            remove_mask = np.array([False] * len(ys["train"]))
            for rare_y in remove_ys:
                remove_mask |= ys["train"] == rare_y
            remove_data(remove_mask)

        if ys["train"].dtype == object:
            if task_type == "regression":
                ys["train"] = ys["train"].astype(np.float32)
            else:
                assert task_type in ["multiclass", "binclass"]
                pass
        if task_type == "multiclass" or task_type == "binclass":
            ys["train"] = strlabeltransform(ys["train"])
            if ys["train"].dtype != np.int64:
                ys["train"] = ys["train"].astype(np.int64)

        n_classes = len(np.unique(ys["train"])) if task_type == "multiclass" else 1
        if task_type == "regression":
            ys = {k: v.astype(np.float32) for k, v in ys.items()}

        def split():
            stratify = None if tt == "regression" else ys["train"]
            datas = [x["train"] for x in [X_num, X_cat] if x is not None] + [
                ys["train"]
            ]
            _datas = train_test_split(
                *datas, test_size=0.2, random_state=42, stratify=stratify
            )
            tr_datas = [_datas[2 * i] for i in range(len(datas))]
            te_datas = [_datas[2 * i + 1] for i in range(len(datas))]
            stratify = None if tt == "regression" else tr_datas[-1]
            _datas = train_test_split(
                *tr_datas, test_size=0.2, random_state=42, stratify=stratify
            )
            tr_datas = [_datas[2 * i] for i in range(len(datas))]
            ev_datas = [_datas[2 * i + 1] for i in range(len(datas))]
            i = 0
            datas = {}
            for x in ["X_num", "X_cat", "ys"]:
                if eval(x) is not None:
                    datas[x] = {
                        "train": tr_datas[i],
                        "val": ev_datas[i],
                        "test": te_datas[i],
                    }
                    i += 1
                else:
                    datas[x] = None
            return datas

        datas = split()

        return Dataset2(
            datas["X_num"],
            datas["X_cat"],
            None,
            datas["ys"],
            {},
            feature_names,
            TaskType(task_type),
            n_classes,
            data_name,
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_str_features(self) -> int:
        return 0 if self.X_str is None else self.X_str["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features + self.n_str_features

    def size(self, part: Optional[str]) -> int:
        if part == "test" and "test" not in self.y:
            for x in ["X_num", "X_cat", "X_str"]:
                data = getattr(self, x)
                if data is not None:
                    return len(data["test"])
            return 0
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = "rmse"
            score_sign = -1
        else:
            score_key = "accuracy"
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics["score"] = score_sign * part_metrics[score_key]
        return metrics


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> "Dataset":
        dir_ = Path(dir_)

        def load(item) -> ArrayDict:
            return {
                x: cast(np.ndarray, np.load(dir_ / f"{item}_{x}.npy"))
                for x in ["train", "val", "test"]
            }

        info = util.load_json(dir_ / "info.json")

        return Dataset(
            load("X_num") if dir_.joinpath("X_num_train.npy").exists() else None,
            load("X_cat") if dir_.joinpath("X_cat_train.npy").exists() else None,
            load("y"),
            {},
            TaskType(info["task_type"]),
            info.get("n_classes"),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = "rmse"
            score_sign = -1
        else:
            score_key = "accuracy"
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics["score"] = score_sign * part_metrics[score_key]
        return metrics


def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):
        assert policy is None
        return dataset

    assert policy is not None
    if policy == "drop-rows":
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            "test"
        ].all(), "Cannot drop test rows, since this will affect the final metrics."
        new_data = {}
        for data_name in ["X_num", "X_cat", "y"]:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == "mean":
        new_values = np.nanmean(dataset.X_num["train"], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown("policy", policy)
    return dataset


def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int]
) -> ArrayDict:
    X_train = X["train"]
    if normalization == "standard":
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == "quantile":
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        noise = 1e-3
        if noise > 0:
            assert seed is not None
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)
            X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
                X_train.shape
            )
    else:
        util.raise_unknown("normalization", normalization)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):
        if policy is None:
            X_new = X
        elif policy == "most_frequent":
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)
            imputer.fit(X["train"])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown("categorical NaN policy", policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X["train"]) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X["train"].shape[1]):
        counter = Counter(X["train"][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
) -> Tuple[ArrayDict, bool]:
    if encoding != "counter":
        y_train = None

    unknown_value = np.iinfo("int64").max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        dtype="int64",
    ).fit(X["train"])
    X = {k: encoder.transform(v) for k, v in X.items()}
    max_values = X["train"].max(axis=0)
    for part in ["val", "test"]:
        for column_idx in range(X[part].shape[1]):
            X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    # Step 2. Encode.
    if encoding is None:
        return (X, False)
    elif encoding == "one-hot":
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown="ignore", sparse=False, dtype=np.float32
        )
        encoder.fit(X["train"])
        return ({k: encoder.transform(v) for k, v in X.items()}, True)
    elif encoding == "counter":
        assert y_train is not None
        assert seed is not None
        encoder = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.fit(X["train"], y_train)
        X = {k: encoder.transform(v).astype("float32") for k, v in X.items()}
        if not isinstance(X["train"], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}
        return (X, True)
    else:
        util.raise_unknown("encoding", encoding)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == "default":
        if task_type == TaskType.REGRESSION:
            mean, std = float(y["train"].mean()), float(y["train"].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        util.raise_unknown("policy", policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = "default"


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
) -> Dataset:
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode("utf-8")
        ).hexdigest()
        transformations_str = "__".join(map(str, astuple(transformations)))
        data_name = getattr(dataset, "name", "")
        cache_path = (
            cache_dir
            / f"cache__{data_name}__{transformations_str}__{transformations_md5}.pickle"
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f"Hash collision for {cache_path}")
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    X_num = dataset.X_num
    if dataset.X_cat is None:
        replace(
            transformations,
            cat_nan_policy=None,
            cat_min_frequency=None,
            cat_encoding=None,
        )
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y["train"],
            transformations.seed,
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    if X_num is not None and transformations.normalization is not None:
        X_num = normalize(X_num, transformations.normalization, transformations.seed)

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    return dataset


def build_dataset(
    path: Union[str, Path], transformations: Transformations, cache: bool
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(path)
    return transform_dataset(dataset, transformations, path if cache else None)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    if isinstance(device, str):
        device = torch.device(device)
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != "cpu":
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    # assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y


def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    path = env.DATA / dataset_dir_name
    info = util.load_json(path / "info.json")
    info["size"] = info["train_size"] + info["val_size"] + info["test_size"]
    info["n_features"] = info["n_num_features"] + info["n_cat_features"]
    info["path"] = path
    return info


def check_class(Y, n_classes):
    if len(torch.unique(Y).shape[0]) == n_classes:
        return True
    else:
        return False


import json


def write_id(filename, dataset_name, ids):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data[dataset_name] = ids
    with open(filename, "w") as file:
        json.dump(data, file)


def read_id(filename, dataset_name):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
            if dataset_name in data.keys():
                return data[dataset_name]
            else:
                return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


