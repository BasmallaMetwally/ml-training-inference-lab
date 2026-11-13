import argparse
import json


def main() -> None:
    """Dispatch a reproducible experiment stage or development grader."""
    p = argparse.ArgumentParser(
        description="Train, evaluate and benchmark a small PyTorch classifier"
    )
    sub = p.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--data", default="data/fashion-mnist")
    tr = sub.add_parser("train")
    tr.add_argument("--data", default="data/fashion-mnist")
    tr.add_argument("--run", default="runs/seed42")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--lr", type=float, default=0.001)
    tr.add_argument("--threads", type=int, default=1)
    tr.add_argument("--resume", action="store_true")
    tr.add_argument(
        "--model-kind",
        choices=["mlp", "cnn", "mlp_small", "mlp_large", "cnn_small", "cnn_large"],
        default="mlp",
    )
    tr.add_argument("--split-seed", type=int, default=None)
    for cmd in ["evaluate", "benchmark"]:
        q = sub.add_parser(cmd)
        q.add_argument("--data", default="data/fashion-mnist")
        q.add_argument("--run", default="runs/seed42")
        if cmd == "benchmark":
            q.add_argument("--repeats", type=int, default=5)
            q.add_argument("--min-run-time", type=float, default=0.1)
            q.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 128, 512])
    rp = sub.add_parser("report")
    rp.add_argument("--run", default="runs/seed42")
    rp.add_argument("--out", default="results/seed42")
    gr = sub.add_parser("grade")
    gr.add_argument("--task", choices=["train_step", "stable_loss", "predict"], required=True)
    gr.add_argument("--candidate", required=True)
    gr.add_argument("--timeout", type=float, default=30)
    gr.add_argument("--out", default=None)
    a = vars(p.parse_args())
    command = a.pop("command")
    if command == "download":
        from .data import download

        result = download(a["data"])
    elif command == "train":
        from .engine import train

        result = train(data_dir=a.pop("data"), run_dir=a.pop("run"), **a)[-1]
    elif command == "evaluate":
        from .engine import final_evaluation

        result = final_evaluation(a["data"], a["run"])
    elif command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(data_dir=a.pop("data"), run_dir=a.pop("run"), **a)
        result = {k: v for k, v in result.items() if k != "measurements"}
    elif command == "report":
        from .report import report

        result = str(report(a["run"], a["out"]))
    else:
        from .grader import grade

        out = a.pop("out")
        result = grade(**a)
        if out:
            from .engine import write_json

            write_json(out, result)
        print(json.dumps(result, indent=2))
        if result["status"] != "pass":
            raise SystemExit(1)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
