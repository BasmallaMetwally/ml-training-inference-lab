import html
import json
from pathlib import Path


def report(run_dir: str | Path, out_dir: str | Path) -> Path:
    """Generate standalone Markdown and HTML from recorded experiment artifacts."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def read(name: str) -> dict | list:
        """Load a recorded stage artifact."""
        return json.loads((run_dir / name).read_text())

    history, metrics, bench, config = (
        read("history.json"),
        read("test_metrics.json"),
        read("benchmark.json"),
        read("config.json"),
    )
    env = bench["environment"]
    lines = [
        "# ML Training and Inference Lab",
        "",
        "## Measured experiment",
        "",
        "Fashion-MNIST classification with a small PyTorch MLP, trained from random initialization on CPU.",
        "",
        "- Train / validation / test: 50,000 / 10,000 / 10,000 images.",
        f"- Seed: {config['seed']}; epochs: {len(history)}; selected epoch: {metrics['selected_epoch']} (minimum validation loss).",
        f"- Official test accuracy: **{metrics['accuracy']:.2%}**; test cross entropy: **{metrics['loss']:.4f}**.",
        f"- Majority-class baseline: {metrics['majority_class_baseline_accuracy']:.2%}.",
        f"- Training wall time (epoch loops): {sum(h['seconds'] for h in history):.1f} seconds.",
        f"- Environment: Python {env['python']}, PyTorch {env['torch']}, {env['machine']}, {env['threads']} CPU thread(s).",
        f"- Checkpoint SHA256: `{metrics['checkpoint_sha256']}`.",
        "",
        "## Training history",
        "",
        "| Epoch | Train loss | Validation loss | Validation accuracy | Seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | {row['validation_loss']:.4f} | {row['validation_accuracy']:.2%} | {row['seconds']:.2f} |"
        )
    lines += [
        "",
        "## CPU inference benchmark",
        "",
        "Same selected weights, eval mode, preloaded normalized validation images, warm-up, randomized interleaved repeats and fixed thread count. Times include input validation. Batch latency is not request latency.",
        "",
        "| Variant | Batch | Median batch ms | Repeat range ms | Images/s | vs autograd | vs inference mode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baselines = {
        r["batch_size"]: r["median_batch_ms"]
        for r in bench["measurements"]
        if r["variant"] == "eval_autograd"
    }
    inference_baselines = {
        r["batch_size"]: r["median_batch_ms"]
        for r in bench["measurements"]
        if r["variant"] == "eval_inference_mode"
    }
    for row in bench["measurements"]:
        lines.append(
            f"| {row['variant']} | {row['batch_size']} | {row['median_batch_ms']:.4f} | {row['min_repeat_ms']:.4f}–{row['max_repeat_ms']:.4f} | {row['images_per_second']:,.0f} | {baselines[row['batch_size']] / row['median_batch_ms']:.2f}× | {inference_baselines[row['batch_size']] / row['median_batch_ms']:.2f}× |"
        )
    eq = bench["equivalence"]
    lines += [
        "",
        "## Correctness of fusion",
        "",
        f"Folded two Linear/BatchNorm pairs using running statistics. Maximum absolute logit difference: {eq['max_abs_logit_error']:.8f}; prediction disagreements: {eq['prediction_disagreements']}/{eq['validation_count']}.",
        f"Validation accuracy: {eq['original_accuracy']:.2%} before and {eq['fused_accuracy']:.2%} after fusion.",
        "",
        "## Limits",
        "",
        "One seed and one CPU machine; no GPU, production-load, or cross-machine performance claims. The test split is used after checkpoint selection; optimization equivalence is checked on validation data. This is a small educational dataset, not evidence of production model quality. Timing excludes preprocessing and I/O. Parameter storage is reported separately from peak memory, which is not measured. Debugging-task validation is not an observed external coding-agent pass rate.",
        "",
        "Full configuration, epoch metrics, confusion matrix, environment and raw timing blocks accompany this report.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines))
    # Standalone HTML report; no JavaScript, CDN or external assets.
    width, height = 860, 240

    def poly(key, ymax, color):
        points = " ".join(
            f"{40 + i * (width - 60) / max(1, len(history) - 1):.1f},{height - 25 - r[key] / ymax * (height - 45):.1f}"
            for i, r in enumerate(history)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>'

    ymax = max(r["train_loss"] for r in history) * 1.1
    labels = f'<text x="5" y="20" font-size="11">{ymax:.2f}</text><text x="20" y="215" font-size="11">0</text><text x="40" y="235" font-size="11">Epoch 1</text><text x="785" y="235" font-size="11">Epoch {len(history)}</text>'
    chart = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Mean cross entropy by epoch"><path d="M40 15 V215 H840" fill="none" stroke="#aaa"/>{labels}{poly("train_loss", ymax, "#235cca")}{poly("validation_loss", ymax, "#c04c31")}</svg>'
    blocks, table = [], False
    for line in lines:
        if line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if all(c.replace(":", "").replace("-", "").strip() == "" for c in cells):
                continue
            tag = "td" if table else "th"
            if not table:
                blocks.append('<div class="table"><table>')
                table = True
            blocks.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        else:
            if table:
                blocks.append("</table></div>")
                table = False
            if line.startswith("## "):
                blocks.append("<h2>" + html.escape(line[3:]) + "</h2>")
            elif line.startswith("# "):
                continue
            elif line:
                blocks.append("<p>" + html.escape(line).replace("**", "").replace("`", "") + "</p>")
    if table:
        blocks.append("</table></div>")
    content = "".join(blocks)
    (out_dir / "report.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ML Training and Inference Lab</title><style>body{{font:16px/1.6 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#14213b}}h1{{font-size:32px}}h2{{margin-top:34px}}svg{{width:100%;background:#f4f7fc;border-radius:12px}}p{{overflow-wrap:anywhere}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #dce3ed;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf0fa}}tr:nth-child(even){{background:#f8faff}}</style><h1>ML Training and Inference Lab</h1><p>Mean cross entropy by epoch — blue: training · orange: validation</p>{chart}{content}</html>'
    )
    return out_dir / "REPORT.md"
