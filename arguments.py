import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser(description="Overide config.yaml parameters.")

    with open("config.yaml", "r") as f:
        defaults = yaml.safe_load(f)
    # train params
    parser.add_argument("--seed", type=int, default=defaults["train"]["seed"])
    parser.add_argument("--lr", type=float, default=defaults["train"]["lr"])
    parser.add_argument("--weight_decay", type=float, default=defaults["train"]["weight_decay"])
    parser.add_argument(
        "--batch_size", type=int, default=defaults["train"]["batch_size"]
    )
    parser.add_argument("--epochs", type=int, default=defaults["train"]["epochs"])
    parser.add_argument("--gamma", type=float, default=defaults["train"]["gamma"])
    parser.add_argument("--n_classes", type=int, default=defaults["train"]["n_classes"])
    parser.add_argument("--workers", type=int, default=defaults["misc"]["workers"])
    parser.add_argument(
        "--normalize", type=int, choices=[0, 1], default=defaults["train"]["normalize"]
    )

    parser.add_argument("--dataset", type=str, choices=["mnist","fashion","kmnist"], default=defaults["train"]["dataset"])


    # Model stuff
    parser.add_argument(
        "--layer_dims", type=int, nargs="+", default=defaults["model"]["layer_dims"]
    )
    parser.add_argument("--emb_dim", type=int, default=defaults["model"]["emb_dim"])
    parser.add_argument(
        "--use_cnn_cond",
        type=int,
        choices=[0, 1],
        default=defaults["model"]["cnn_cond"],
    )

    parser.add_argument("--data_dir", type=str, default=defaults["model"]["data_dir"])
    parser.add_argument(
        "--clip_grad", type=float, default=defaults["model"]["clip_grad"]
    )
    # Log
    parser.add_argument("--log_interval", type=int, default=defaults["log"]["interval"])
    parser.add_argument("--run_dir", type=str, default=defaults["log"]["run_dir"])
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1], default=defaults["log"]["verbose"]
    )

    # Test params
    parser.add_argument(
        "--testing_batch_size", type=float, default=defaults["testing"]["batch_size"]
    )

    return parser.parse_args()
