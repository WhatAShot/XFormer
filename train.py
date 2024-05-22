import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from category_encoders import CatBoostEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from bin import XTFormer
from lib import (
    CAT_MISSING_VALUE,
    DATA,
    Dataset2,
    Transformations,
    get_training_args,
    make_optimizer,
    prepare_tensors,
    read_id,
    seed_everything,
    transform_dataset,
    write_id,
)

pre_train = input("Pre-train or Use existing parameters? (p/u): ")


def record_exp(args, final_score, best_score, **kwargs):
    results = {"config": args, "final": final_score, "best": best_score, **kwargs}
    with open(f"{args['output']}/results.json", "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


"""args"""
device = torch.device("cuda")
args, cfg = get_training_args()
seed_everything(args.seed)

if args.tensorboard:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(log_dir=f"{args.output}/logs")

DATASETS = []


def get_datasets():
    DatasetDict = {}
    Bin_Data = os.path.join(DATA, "bin_data")
    Reg_Data = os.path.join(DATA, "reg_data")
    for folder in [Bin_Data, Reg_Data]:
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                name_without_extension = os.path.splitext(filename)[0]
                if folder == Bin_Data:
                    task_type = "binclass"
                elif folder == Reg_Data:
                    task_type = "regression"
                else:
                    task_type = "multiclass"

                DatasetDict[name_without_extension] = {"task_type": task_type}

    return DatasetDict


if os.path.exists("./data/FineTuneData.json"):
    with open("./data/FineTuneData.json", "r") as f:
        FineTuneDataDict = json.load(f)
        f.close()
if os.path.exists("./data/PreTrainData.json"):
    with open("./data/PreTrainData.json", "r") as f:
        PreTrainDataDict = json.load(f)
        f.close()

if not os.path.exists("./data/FineTuneData.json") or not os.path.exists(
    "./data/PreTrainData.json"
):
    DatasetDict = get_datasets()

    keys = list(DatasetDict.keys())
    random.shuffle(keys)
    percent = 0.90
    split_point = int(len(keys) * percent)
    PreTrainDataDict = {key: DatasetDict[key] for key in keys[:split_point]}
    FineTuneDataDict = {key: DatasetDict[key] for key in keys[split_point:]}


if args.save:
    with open(f"{args.output}/PreTrainData.json", "w") as json_file:
        json.dump(PreTrainDataDict, json_file)
    with open(f"{args.output}/FineTuneData.json", "w") as json_file:
        json.dump(FineTuneDataDict, json_file)


class DataSpecific(object):

    def __init__(self, dataset_name, task_type, all_data=True, small_dataset=False):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.T_cache = True
        self.all_data = all_data
        self.small_dataset = small_dataset
        self.data_preparation()
        self.epoch = 0

    def epoch_add(self):
        self.epoch += 1

    def data_preparation(self):
        normalization = args.normalization if args.normalization != "__none__" else None
        if self.task_type == "binclass":
            from lib import BIN_DATA as DATA

            print(DATA)
        elif self.task_type == "multiclass":
            from lib import MUL_DATA as DATA
        else:
            from lib import REG_DATA as DATA

        dataset = Dataset2.from_csv(DATA, dataset_name, 16, tt=self.task_type)

        num_nan_policy = (
            "mean"
            if dataset.X_num is not None
            and any(np.isnan(dataset.X_num[spl]).any() for spl in dataset.X_num)
            else None
        )
        cat_nan_policy = (
            "most_frequent"
            if dataset.X_cat is not None
            and any(
                (dataset.X_cat[spl] == CAT_MISSING_VALUE).any() for spl in dataset.X_cat
            )
            else None
        )
        cat_min_frequency = 0.05
        transformation = Transformations(
            normalization=normalization,
            num_nan_policy=num_nan_policy,
            cat_nan_policy=cat_nan_policy,
            cat_min_frequency=cat_min_frequency,
        )
        dataset = transform_dataset(dataset, transformation, DATA)

        if (dataset.X_num is not None) and dataset.X_num["train"].dtype == np.float64:
            dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}

        if args.catenc and dataset.X_cat is not None:
            cardinalities = dataset.get_category_sizes("train")

            enc = CatBoostEncoder(
                cols=list(range(len(cardinalities))), return_df=False
            ).fit(dataset.X_cat["train"], dataset.y["train"])
            if dataset.X_num is not None:
                for k in ["train", "val", "test"]:
                    dataset.X_num[k] = np.concatenate(
                        [
                            enc.transform(dataset.X_cat[k]).astype(np.float32),
                            dataset.X_num[k],
                        ],
                        axis=1,
                    )

        d_out = dataset.n_classes or 1
        X_num, X_cat, ys = prepare_tensors(dataset, device=device)
        if args.catenc:
            X_cat = None

        if dataset.X_num is not None:

            mi_cache_dir = "./data/cache/mi"
            if not os.path.isdir(mi_cache_dir):
                os.makedirs(mi_cache_dir)
            mi_cache_file = f"{mi_cache_dir}/{self.dataset_name}.npy"
            if os.path.exists(mi_cache_file):
                mi_scores = np.load(mi_cache_file)
            else:
                mi_func = (
                    mutual_info_regression
                    if dataset.is_regression
                    else mutual_info_classif
                )
                mi_scores = mi_func(dataset.X_num["train"], dataset.y["train"])
                np.save(mi_cache_file, mi_scores)
            mi_ranks = np.argsort(-mi_scores)
            X_num = {k: v[:, mi_ranks] for k, v in X_num.items()}
            sorted_mi_scores = (
                torch.from_numpy(mi_scores[mi_ranks] / mi_scores.sum())
                .float()
                .to(device)
            )
            """ END FEATURE REORDER """

        batch_size = args.batch_size
        val_batch_size = args.batch_size

        cfg["training"].update(
            {
                "batch_size": batch_size,
                "eval_batch_size": val_batch_size,
                "patience": args.early_stop,
            }
        )
        if X_num is None and X_cat is None:
            raise ValueError("No features in dataset")
        else:
            if X_cat is None:
                data_list = [X_num, ys]
            elif X_num is None:
                data_list = [X_cat, ys]
            else:
                data_list = [X_num, X_cat, ys]
            if self.all_data:

                train_data = [*(d["train"] for d in data_list)]
                valid_data = [*(d["val"] for d in data_list)]
                test_data = [*(d["test"] for d in data_list)]

                x = torch.cat(train_data[0:1] + valid_data[0:1] + test_data[0:1], dim=0)
                y = torch.cat(train_data[1:2] + valid_data[1:2] + test_data[1:2], dim=0)

                train_dataset = TensorDataset(x, y)
                val_dataset = TensorDataset(*(d["val"] for d in data_list))

            else:

                if self.small_dataset:

                    train_dataset = [*(d["train"] for d in data_list)]
                    x_ = train_dataset[0]
                    y_0 = train_dataset[1]
                    if y_0.shape[0] > args.n_small_train:
                        if (
                            read_id(
                                "./data/sample_id.json",
                                self.dataset_name + "_train_" + str(args.n_small_train),
                            )
                            is None
                        ):
                            random_id = np.random.choice(
                                x_.shape[0], args.n_small_train, replace=False
                            ).tolist()

                            y_ = y_0[random_id]
                            while self.task_type == "binclass" and not (
                                len(torch.unique(y_)) == len(torch.unique(y_0))
                            ):
                                random_id = np.random.choice(
                                    x_.shape[0], args.n_small_train, replace=False
                                ).tolist()
                                y_ = y_0[random_id]

                            write_id(
                                "./data/sample_id.json",
                                self.dataset_name + "_train_" + str(args.n_small_train),
                                random_id,
                            )
                        else:
                            random_id = read_id(
                                "./data/sample_id.json",
                                self.dataset_name + "_train_" + str(args.n_small_train),
                            )
                        y_ = y_0[random_id]
                        x_ = x_[random_id, :]
                        train_dataset = TensorDataset(x_, y_)
                    else:
                        train_dataset = TensorDataset(x_, y_0)

                    val_dataset = [*(d["val"] for d in data_list)]

                    x_ = val_dataset[0]
                    y_0 = val_dataset[1]
                    if y_0.shape[0] > args.n_small_train // 4:
                        if (
                            read_id(
                                "./data/sample_id.json",
                                self.dataset_name + "_val_" + str(args.n_small_train),
                            )
                            is None
                        ):
                            random_id = np.random.choice(
                                x_.shape[0], args.n_small_train // 4, replace=False
                            ).tolist()
                            y_ = y_0[random_id]
                            while self.task_type == "binclass" and not (
                                len(torch.unique(y_)) == len(torch.unique(y_0))
                            ):
                                random_id = np.random.choice(
                                    x_.shape[0], args.n_small_train // 4, replace=False
                                ).tolist()
                                y_ = y_0[random_id]
                            write_id(
                                "./data/sample_id.json",
                                self.dataset_name + "_val_" + str(args.n_small_train),
                                random_id,
                            )
                        else:
                            random_id = read_id(
                                "./data/sample_id.json",
                                self.dataset_name + "_val_" + str(args.n_small_train),
                            )
                        y_ = y_0[random_id]
                        x_ = x_[random_id, :]
                        val_dataset = TensorDataset(x_, y_)
                    else:
                        val_dataset = TensorDataset(x_, y_0)
                else:
                    train_dataset = TensorDataset(*(d["train"] for d in data_list))
                    val_dataset = TensorDataset(*(d["val"] for d in data_list))

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                shuffle=True,
            )

            test_dataset = TensorDataset(*(d["test"] for d in data_list))
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=val_batch_size,
                shuffle=False,
            )
            dataloaders = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }

            n_num_features = dataset.n_num_features
            cardinalities = dataset.get_category_sizes("train")
            n_categories = len(cardinalities)
            if args.catenc:
                n_categories = 0
            cardinalities = None if n_categories == 0 else cardinalities

            self.dataloaders = dataloaders
            self.dataset = dataset
            self.data_list = data_list
            self.d_out = d_out
            self.sorted_mi_scores = sorted_mi_scores
            self.n_num_features = n_num_features
            self.cardinalities = cardinalities
            self.batch_size = batch_size
            self.n_feat = self.n_num_features + n_categories

    @property
    def n_tokens(self):
        return self.n_feat

    @property
    def categories(self):
        return self.cardinalities

    @property
    def n_numerical(self):
        return self.n_num_features

    @property
    def n_samples(self):
        return (
            len(self.dataset.y["train"])
            + len(self.dataset.y["val"])
            + len(self.dataset.y["test"])
        )


pretrain_dataset_list = []
finetune_dataset_list = []

count = 0
keys_to_remove = []
sample_num = []

for dataset_name, dataset_info in PreTrainDataDict.items():

    dataset = DataSpecific(
        dataset_name, dataset_info["task_type"], all_data=True, small_dataset=False
    )

    if dataset.n_feat <= 50:

        PreTrainDataDict[dataset_name]["d_out"] = dataset.d_out
        PreTrainDataDict[dataset_name]["n_tokens"] = dataset.n_tokens
        PreTrainDataDict[dataset_name]["categories"] = dataset.categories
        PreTrainDataDict[dataset_name]["d_numerical"] = dataset.n_numerical

        pretrain_dataset_list.append(dataset)
        sample_num.append(dataset.n_samples)
        count += 1
    else:
        PreTrainDataDict[dataset_name]["d_numerical"] = dataset.n_numerical
        keys_to_remove.append(dataset_name)

        continue
for key in keys_to_remove:
    del PreTrainDataDict[key]
del keys_to_remove

print("--------------------")
print("{} datasets for pre-training".format(count))
print(sample_num)


count = 0
keys_to_remove = []
for dataset_name, dataset_info in FineTuneDataDict.items():

    dataset = DataSpecific(
        dataset_name,
        dataset_info["task_type"],
        all_data=False,
        small_dataset=args.small_dataset,
    )

    if dataset.n_feat <= 50:
        FineTuneDataDict[dataset_name]["d_out"] = dataset.d_out
        FineTuneDataDict[dataset_name]["n_tokens"] = dataset.n_tokens
        FineTuneDataDict[dataset_name]["categories"] = dataset.categories
        FineTuneDataDict[dataset_name]["d_numerical"] = dataset.n_numerical

        finetune_dataset_list.append(dataset)
        count += 1
    else:
        FineTuneDataDict[dataset_name]["d_numerical"] = dataset.n_numerical
        keys_to_remove.append(dataset_name)
        continue

for key in keys_to_remove:
    del FineTuneDataDict[key]
del keys_to_remove

print("{} datasets for fine-tuning".format(count))
print("--------------------")
time.sleep(10)

DatasetDict = {**PreTrainDataDict, **FineTuneDataDict}

""" Prepare Model """
kwargs = {"DataSet_Info": DatasetDict, "token_bias": True, "n_experts": 4}

default_model_configs = {
    "ffn_dropout": 0.0,
    "attention_dropout": 0.3,
    "residual_dropout": 0.0,
    "n_layers": args.n_layers,
    "n_heads": 32,
    "d_token": args.d_token,
    "if_MoE": args.if_MoE,
    "init_scale": 0.01,
}

default_training_configs = {
    "lr": 1e-4,
    "weight_decay": 0.0,
}

kwargs.update(default_model_configs)
cfg["training"].update(default_training_configs)


model = XTFormer(**kwargs)
model.to(device)


def needs_wd(name):
    return all(x not in name for x in ["tokenizer", ".norm", ".bias"])


def fix_parameters(name):
    return all(
        x not in name for x in ["noise", "tokenizer", "head_fcs", "last_fcs", ".norm"]
    )


"""Utils Function"""


def apply_model(dataset_name, x_num, x_cat=None, mixup=False):
    if mixup:
        x, feat_masks, shuffled_ids = model(
            dataset_name=dataset_name,
            x_num=x_num,
            x_cat=x_cat,
            mixup=True,
            mtype=args.mix_type,
        )
        return x, feat_masks, shuffled_ids
    else:
        return model(
            dataset_name=dataset_name,
            x_num=x_num,
            x_cat=x_cat,
            mixup=False,
            mtype=args.mix_type,
        )


@torch.inference_mode()
def evaluate(parts, model, dataloaders, dataset, dataset_name):

    model.eval()
    predictions = {}
    true_labels = {}
    for part in parts:
        assert part in ["train", "val", "test"]
        infer_time = 0.0
        predictions[part] = []
        true_labels[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y_ = (batch[0], None, batch[1]) if len(batch) == 2 else batch
            start = time.time()

            predictions[part].append(
                apply_model(
                    dataset_name=dataset_name, x_num=x_num, x_cat=x_cat, mixup=False
                )
            )
            true_labels[part].append(y_)

            infer_time += time.time() - start

        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
        true_labels[part] = torch.cat(true_labels[part]).cpu().numpy()
        if part == "test":
            print("test time: ", infer_time)

    prediction_type = None if dataset.is_regression else "logits"
    return dataset.calculate_metrics(predictions, true_labels, prediction_type)


def fine_tune(ft_model, pre_train_epoch, ft_max_epoch=args.ft_max_epoch):

    for specific_dataset in finetune_dataset_list:

        dataloaders, dataset, data_list = (
            specific_dataset.dataloaders,
            specific_dataset.dataset,
            specific_dataset.data_list,
        )
        d_out, sorted_mi_scores, batch_size = (
            specific_dataset.d_out,
            specific_dataset.sorted_mi_scores,
            specific_dataset.batch_size,
        )
        dataset_name = specific_dataset.dataset_name
        print(
            f"======== Fine-tuning on dataset {dataset_name} | task type: {specific_dataset.task_type} ========"
        )
        metric = "roc_auc" if dataset.is_binclass else "score"

        init_score = evaluate(
            parts=["test"],
            model=ft_model,
            dataloaders=dataloaders,
            dataset=dataset,
            dataset_name=dataset_name,
        )["test"][
            metric
        ]  # test before training

        print(f"Test score before training: {init_score: .4f}")

        """Loss Function"""
        loss_fn = (
            F.binary_cross_entropy_with_logits
            if specific_dataset.task_type == "binclass"
            else (
                F.cross_entropy
                if specific_dataset.task_type == "multiclass"
                else F.mse_loss
            )
        )

        fine_tune_parameters_wd = [
            v
            for k, v in ft_model.named_parameters()
            if not fix_parameters(k) and needs_wd(k)
        ]
        fine_tune_parameters_without_wd = [
            v
            for k, v in ft_model.named_parameters()
            if not fix_parameters(k) and not needs_wd(k)
        ]

        fine_tune_parameters_wd_makeup = [
            v
            for k, v in ft_model.named_parameters()
            if fix_parameters(k) and needs_wd(k)
        ]
        fine_tune_parameters_without_wd_makeup = [
            v
            for k, v in ft_model.named_parameters()
            if fix_parameters(k) and not needs_wd(k)
        ]

        ft_optimizer = make_optimizer(
            cfg["training"]["optimizer"],
            (
                [
                    {"params": fine_tune_parameters_wd},
                    {"params": fine_tune_parameters_without_wd, "weight_decay": 0.0},
                ]
            ),
            cfg["training"]["lr"],
            cfg["training"]["weight_decay"],
        )

        ft_optimizer_makeup = make_optimizer(
            cfg["training"]["optimizer"],
            (
                [
                    {"params": fine_tune_parameters_wd},
                    {"params": fine_tune_parameters_without_wd, "weight_decay": 0.0},
                    {
                        "params": fine_tune_parameters_wd_makeup,
                        "lr": cfg["training"]["lr"] / 10,
                    },
                    {
                        "params": fine_tune_parameters_without_wd_makeup,
                        "lr": cfg["training"]["lr"] / 10,
                        "weight_decay": 0.0,
                    },
                ]
            ),
            cfg["training"]["lr"],
            cfg["training"]["weight_decay"],
        )

        """Training for Finetuning"""
        metric = "roc_auc" if dataset.is_binclass else "score"
        ft_scheduler = CosineAnnealingLR(optimizer=ft_optimizer, T_max=ft_max_epoch)
        ft_scheduler_makeup = CosineAnnealingLR(
            optimizer=ft_optimizer_makeup, T_max=ft_max_epoch
        )
        report_frequency = max(len(data_list[-1]["train"]) // batch_size // 3, 1)
        loss_holder = AverageMeter()
        best_score = -np.inf
        running_time = 0.0
        no_improvement = 0
        EARLY_STOP = args.early_stop

        for epoch in range(1, ft_max_epoch + 1):

            if epoch < ft_max_epoch * 0.8:
                mode = "partially"
            else:
                mode = "fully"

            ft_model.train()
            for iteration, batch in enumerate(dataloaders["train"]):
                x_num, x_cat, y = (
                    (batch[0], None, batch[1]) if len(batch) == 2 else batch
                )

                start = time.time()

                (
                    ft_optimizer.zero_grad()
                    if mode == "partially"
                    else ft_optimizer_makeup.zero_grad()
                )

                if args.mix_type == "none":  # no mixup
                    loss = loss_fn(
                        apply_model(
                            dataset_name=dataset_name,
                            x_num=x_num,
                            x_cat=x_cat,
                            mixup=False,
                        ),
                        y,
                    )
                else:
                    preds, feat_masks, shuffled_ids = apply_model(
                        dataset_name=dataset_name, x_num=x_num, x_cat=x_cat, mixup=True
                    )
                    if args.mix_type == "hidden_mix":
                        lambdas = feat_masks
                        lambdas2 = 1 - lambdas
                    elif args.mix_type == "naive_mix":
                        lambdas = feat_masks
                        lambdas2 = 1 - lambdas
                    if dataset.is_regression:
                        mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                        loss = loss_fn(preds, mix_y)
                    else:
                        loss = lambdas * loss_fn(
                            preds, y, reduction="none"
                        ) + lambdas2 * loss_fn(preds, y[shuffled_ids], reduction="none")
                        loss = loss.mean()

                loss.backward()
                (
                    ft_optimizer.step()
                    if mode == "partially"
                    else ft_optimizer_makeup.step()
                )
                (
                    ft_scheduler.step()
                    if mode == "partially"
                    else ft_scheduler_makeup.step()
                )

                running_time += time.time() - start
                loss_holder.update(loss.item(), len(data_list[-1]))

                if iteration % report_frequency == 0:
                    print(
                        f"dataset: {specific_dataset.dataset_name} | task type: {specific_dataset.task_type} | epoch: {epoch} | batch: {iteration} | loss: {loss_holder.val:.4f} | avg_loss: {loss_holder.avg:.4f}"
                    )

            if args.tensorboard:
                writer.add_scalar(
                    "{}/FineTune/{}/{}/TrainLoss".format(
                        specific_dataset.task_type,
                        specific_dataset.dataset_name,
                        pre_train_epoch,
                    ),
                    loss_holder.avg,
                    epoch,
                )
            loss_holder.reset()

            scores = evaluate(
                parts=["val", "test"],
                model=ft_model,
                dataloaders=dataloaders,
                dataset=dataset,
                dataset_name=dataset_name,
            )
            val_score, test_score = scores["val"][metric], scores["test"][metric]

            if args.tensorboard:
                writer.add_scalar(
                    "{}/FineTune/{}/{}/Val_Score".format(
                        specific_dataset.task_type,
                        specific_dataset.dataset_name,
                        pre_train_epoch,
                    ),
                    val_score,
                    epoch,
                )
                writer.add_scalar(
                    "{}/FineTune/{}/{}/Test_Score".format(
                        specific_dataset.task_type,
                        specific_dataset.dataset_name,
                        pre_train_epoch,
                    ),
                    test_score,
                    epoch,
                )

            print(
                f"Dataset {specific_dataset.dataset_name} | task type: {specific_dataset.task_type} | Epoch {epoch} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}",
                end="",
            )

            if val_score > best_score:
                best_score = val_score
                final_test_score = test_score
                print(" <<< BEST VALIDATION EPOCH")

                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement == EARLY_STOP:
                break

        print(
            f"DataSet: {specific_dataset.dataset_name}; Running time: {running_time:.2f}s; Best validation score: {best_score:.4f}; Final test score: {final_test_score:.4f}"
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pretrain(model):
    for epoch in range(1, args.pt_max_epoch + 1):

        specific_dataset = random.choice(pretrain_dataset_list)
        specific_dataset.epoch_add()
        dataloaders, dataset, data_list = (
            specific_dataset.dataloaders,
            specific_dataset.dataset,
            specific_dataset.data_list,
        )
        dataset_name = specific_dataset.dataset_name

        """Loss Function"""
        loss_fn = (
            F.binary_cross_entropy_with_logits
            if specific_dataset.task_type == "binclass"
            else (
                F.cross_entropy
                if specific_dataset.task_type == "multiclass"
                else F.mse_loss
            )
        )

        model_parameters_with_wd = [
            v for k, v in model.named_parameters() if needs_wd(k)
        ]
        model_parameters_without_wd = [
            v for k, v in model.named_parameters() if not needs_wd(k)
        ]

        optimizer = make_optimizer(
            cfg["training"]["optimizer"],
            (
                [
                    {"params": model_parameters_with_wd},
                    {"params": model_parameters_without_wd, "weight_decay": 0.0},
                ]
            ),
            cfg["training"]["lr"],
            cfg["training"]["weight_decay"],
        )

        warmup_parameters_with_wd = [
            v
            for k, v in model.named_parameters()
            if needs_wd(k) and not fix_parameters(k)
        ]
        warmup_parameters_without_wd = [
            v
            for k, v in model.named_parameters()
            if not needs_wd(k) and not fix_parameters(k)
        ]

        warmup_optimizer = make_optimizer(
            cfg["training"]["optimizer"],
            (
                [
                    {"params": warmup_parameters_with_wd},
                    {"params": warmup_parameters_without_wd, "weight_decay": 0.0},
                ]
            ),
            cfg["training"]["lr"],
            cfg["training"]["weight_decay"],
        )

        """Training"""
        metric = "roc_auc" if dataset.is_binclass else "score"
        loss_holder = AverageMeter()

        running_time = 0.0

        model.train()
        start = time.time()

        for iteration, batch in enumerate(dataloaders["train"]):
            if iteration < 5:
                optimi = warmup_optimizer
            else:
                optimi = optimizer

            if iteration > 25:
                break

            optimi.zero_grad()

            x_num, x_cat, y = (batch[0], None, batch[1]) if len(batch) == 2 else batch

            if args.mix_type == "none":
                loss = loss_fn(
                    apply_model(
                        dataset_name=dataset_name, x_num=x_num, x_cat=x_cat, mixup=False
                    ),
                    y,
                )
            else:
                preds, feat_masks, shuffled_ids = apply_model(
                    dataset_name=dataset_name, x_num=x_num, x_cat=x_cat, mixup=True
                )

                if args.mix_type == "hidden_mix":
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                elif args.mix_type == "naive_mix":
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                if dataset.is_regression:
                    mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                    loss = loss_fn(preds, mix_y)
                else:
                    loss = lambdas * loss_fn(
                        preds, y, reduction="none"
                    ) + lambdas2 * loss_fn(preds, y[shuffled_ids], reduction="none")
                    loss = loss.mean()

            loss.backward()
            optimi.step()

            running_time += time.time() - start
            loss_holder.update(loss.item(), len(data_list[-1]))
            if iteration == 10:
                print(
                    f"dataset: {specific_dataset.dataset_name} | task type: {specific_dataset.task_type} | epoch: {specific_dataset.epoch} | batch: {iteration} | loss: {loss_holder.val:.4f} | avg_loss: {loss_holder.avg:.4f}"
                )

        if args.tensorboard:
            writer.add_scalar(
                "{}/pt/{}/TrainLoss".format(
                    specific_dataset.task_type, specific_dataset.dataset_name
                ),
                loss_holder.avg,
                specific_dataset.epoch,
            )

        loss_holder.reset()
        scores = evaluate(
            parts=["val", "test"],
            model=model,
            dataloaders=dataloaders,
            dataset=dataset,
            dataset_name=dataset_name,
        )
        val_score, test_score = scores["val"][metric], scores["test"][metric]

        if args.tensorboard:
            writer.add_scalar(
                "{}/pt/{}/Val_Score".format(
                    specific_dataset.task_type, specific_dataset.dataset_name
                ),
                val_score,
                specific_dataset.epoch,
            )
            writer.add_scalar(
                "{}/pt/{}/Test_Score".format(
                    specific_dataset.task_type, specific_dataset.dataset_name
                ),
                test_score,
                specific_dataset.epoch,
            )

        print(
            f"Dataset {specific_dataset.dataset_name} | task type: {specific_dataset.task_type} | Epoch {specific_dataset.epoch} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}",
            end="",
        )

        if epoch % 100 == 0 and args.save:
            torch.save(
                model.state_dict(),
                f"{args.output}/thousand_{args.mix_type}_newest_pretrain.pt",
            )

        if (epoch > 200000 and epoch % 5000 == 0) or (epoch == 1):  # debug
            print("======================= FINE TUNING =====================")
            fine_tune(
                ft_model=model, pre_train_epoch=epoch, ft_max_epoch=args.ft_max_epoch
            )
            print("======================= FINE TUNING DONE =====================")
            if args.save:
                torch.save(
                    model.state_dict(),
                    f"{args.output}/{model.name}_{args.mix_type}_EPOCH_{epoch}_pretrain.pt",
                )


if pre_train == "u":
    # NOTE: SET THE PATH TO THE PRE-TRAINED MODEL
    params_path = ""
    print("Loading parameters from: ", params_path)
    time.sleep(10)
    params = torch.load(params_path)
    model.load_state_dict(params)
    print("Parameters loaded from {}!!!".format(params_path))

    fine_tune(
        ft_model=model,
        pre_train_epoch=args.pt_max_epoch,
        ft_max_epoch=args.ft_max_epoch,
    )

elif pre_train == "p":
    pretrain(model)
