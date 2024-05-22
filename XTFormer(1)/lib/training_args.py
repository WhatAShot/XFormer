import argparse
import os


def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="XTFormer Pretrain")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--normalization", type=str, default="quantile")
    parser.add_argument("--n_sample_mapping", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pt_max_epoch", type=int, default=500000)
    parser.add_argument("--ft_max_epoch", type=int, default=50)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="hyper-parameter of Beta Distribution in mixup, we choose 0.5 for all datasets in default config",
    )
    parser.add_argument(
        "--mix_type",
        type=str,
        default="none",
        choices=["naive_mix", "feat_mix", "hidden_mix", "none"],
        help='mixup type, set to "naive_mix" for naive mixup, set to "none" if no mixup',
    )
    parser.add_argument(
        "--save", action="store_true", help="whether to save model", default=True
    )
    parser.add_argument("--n_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--d_token", type=int, default=256, help="dimension of token")
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="keywords for remembering the experiment",
    )
    parser.add_argument(
        "--n_small_train",
        type=int,
        default=100,
        help="number of samples for small training set",
    )
    parser.add_argument(
        "--catenc",
        action="store_false",
        help="whether to use catboost encoder for categorical features",
    )
    parser.add_argument(
        "--n_experts", type=int, default=4, help="number of experts in MoE"
    )
    parser.add_argument("--if_MoE", type=bool, default=True, help="if use MoE")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="whether to use tensorboard",
        default=True,
    )
    parser.add_argument(
        "--small_dataset",
        action="store_true",
        help="whether to use small training set",
        default=True,
    )
    args = parser.parse_args()

    args.keywords = input("keywords for the experiment: ")

    if args.small_dataset == False:
        ask_small_data = input("use small-scaled dataset in fine-tuning? (y/n): ")
        if ask_small_data == "y":
            args.small_dataset = True
            # debug
            args.output = f"{args.output}/pre_train/ThouPath/{args.keywords}"
        else:
            args.output = f"{args.output}/pre_train/ThouPath"
    else:
        # debug
        args.output = f"{args.output}/pre_train/ThouPath/{args.keywords}"

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # some basic model configuration
    cfg = {
        "model": {
            "kv_compression": None,
            "kv_compression_sharing": None,
            "token_bias": True,
        },
        "training": {
            "optimizer": "adamw",
        },
    }

    return args, cfg
