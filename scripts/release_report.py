"""Build release documentation and figures directly from measured evidence."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path: str) -> dict:
    """Load a canonical recorded result."""
    return json.loads(Path(path).read_text())


def main() -> None:
    """Write the findings, methods and charts without inventing missing measurements."""
    study = read("results/study/summary.json")
    mac = read("results/performance/summary.json")
    linux = read("results/performance-linux/summary.json")

    def latency(result: dict, name: str, checked: bool, batch: int) -> float:
        """Select one measured timing without replacing missing values."""
        return next(
            x["median_ms"]
            for x in result["variants"][name]["measurements"]
            if x["checked"] == checked and x["batch_size"] == batch
        )

    mlp, cnn = study["accuracy"]["mlp"], study["accuracy"]["cnn"]
    difference = study["paired_difference_cnn_minus_mlp"]
    a, b = [x * 100 for x in difference["ci95"]]
    result_dir = Path("results/figures")
    result_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, (platform, data) in zip(
        axes, [("macOS arm64", mac), ("Linux x86_64", linux)], strict=True
    ):
        names = []
        values = []
        colors = []
        for name, value in data["variants"].items():
            if value["status"] != "measured":
                continue
            names.append(name)
            values.append(latency(data, name, True, 1))
            colors.append("#2861a4" if value["accepted_exact_predictions"] else "#bb4b29")
        ax.bar(names, values, color=colors)
        ax.set(
            title=platform,
            xlabel="Runtime, checked inputs",
            ylabel="Median batch-1 forward latency (ms)",
        )
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Blue: identical validation predictions · orange: rejected by parity policy\nDo not compare hardware speeds across panels as a controlled platform effect"
    )
    fig.savefig(result_dir / "runtimes.png", dpi=180)
    plt.close(fig)
    batches = [1, 32, 128, 512]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for checked, label in [(True, "Input checks enabled"), (False, "Input checks disabled")]:
        ratios = [
            latency(mac, "eager", checked, b) / latency(mac, "fused", checked, b) for b in batches
        ]
        ax.plot(range(4), ratios, "o-", label=label)
    ax.set(
        xticks=range(4),
        xticklabels=batches,
        xlabel="Batch size",
        ylabel="Fusion speedup vs same-mode unfused inference",
        title="Fusion ablation · macOS · seven randomized timing repetitions",
    )
    ax.axhline(1, color="gray", linestyle="--")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(result_dir / "ablation.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Runtime and profiler study",
        "",
        "All speedups compare the same selected checkpoint on the same platform. Zero validation prediction disagreements are required for accepted parity-preserving speed claims. Compilation and session construction are excluded from forward timing and recorded separately.",
        "",
        "## macOS ablation",
        "",
        "| Batch | Eager checked ms | Fused checked ms | Checked speedup | Unchecked speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for batch in batches:
        eager = latency(mac, "eager", True, batch)
        fused = latency(mac, "fused", True, batch)
        unchecked = latency(mac, "eager", False, batch) / latency(mac, "fused", False, batch)
        lines.append(
            f"| {batch} | {eager:.4f} | {fused:.4f} | {eager / fused:.2f}× | {unchecked:.2f}× |"
        )
    lines += [
        "",
        "Fusion helps with validation both enabled and disabled. The absolute input-check cost grows with batch size and can dominate the checked forward. This reduces the relative visible fusion gain; the speedup is not explained by silently removing input checks.",
        "",
        "![Fusion ablation](../figures/ablation.png)",
        "",
        "## Alternative runtimes",
        "",
        "| Platform | Runtime | Batch-1 checked ms | Prediction changes / 10,000 | Exact-parity accepted |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for platform, data in [("macOS", mac), ("Linux", linux)]:
        for name, row in data["variants"].items():
            if row["status"] == "measured":
                lines.append(
                    f"| {platform} | {name} | {latency(data, name, True, 1):.4f} | {row['prediction_disagreements']} | {row['accepted_exact_predictions']} |"
                )
            else:
                lines.append(f"| {platform} | {name} | unavailable | — | not accepted |")
    lines += [
        "",
        "![Runtime comparison](../figures/runtimes.png)",
        "",
        "Dynamic int8 is a measured negative result when it changes classifications, even if a particular platform makes it faster. ONNX is accepted only after checkpoint and graph fingerprints match. The full batch-size table, raw timing blocks and setup times are in each platform summary.",
        "",
        "## Operator attribution",
        "",
        "| macOS path | ATen calls / 100 forwards | Validation total µs | Network core total µs |",
        "| --- | ---: | ---: | ---: |",
    ]
    profiles = read("results/performance/profiles.json")
    for name, row in profiles.items():
        ops = {x["name"]: x for x in row["operators"]}
        lines.append(
            f"| {name} | {row['aten_calls']} | {ops.get('input_validation', {}).get('total_cpu_us', 0):.1f} | {ops['network_core']['total_cpu_us']:.1f} |"
        )
    lines += [
        "",
        "Fusion removes the BatchNorm operator paths and reduces the number of ATen invocations. These instrumented totals include profiling overhead; operator self time does not independently identify pure Python or C++ dispatch overhead. Uninstrumented latency ablations supply the performance comparison. Do not subtract nested profiler totals as if they were independent costs.",
        "",
        "## Actual peak process memory",
        "Linux v1.0 memory measurements are inconclusive. Use the [corrected three-repeat Linux study](../performance-linux/corrected/REPORT.md). The following table contains macOS observations only.",
        "",
        "| Platform | Runtime | Checks | Peak process RSS MiB | Baseline after imports MiB |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for platform, path in [
        ("macOS", "results/performance/memory.json"),
    ]:
        for row in read(path)["measurements"]:
            if "peak_process_rss_bytes" not in row:
                continue
            lines.append(
                f"| {platform} | {row['variant']} | {row['checked']} | {row['peak_process_rss_bytes'] / 1048576:.1f} | {row['baseline_rss_after_imports_bytes'] / 1048576:.1f} |"
            )
    lines += [
        "",
        "Measured with OS high-water RSS in separate fresh processes, including imports, model/runtime construction and 50 batch-128 forwards. macOS ru_maxrss is in bytes. Corrected Linux measurements use exec-local VmHWM; original Linux values must not be compared. These are process peaks, not peak activations or parameter storage. One process measurement per variant is not evidence of a small memory improvement. Compiled construction may reuse an on-disk compiler cache.",
        "",
        "## Attempt audit",
        "",
        "The macOS Inductor attempt failed while loading the C++ runtime and is recorded as unavailable; Linux compilation is separately measured. A first Linux ONNX attempt reused an exported graph from a different checkpoint and was invalidated. The cache now verifies both checkpoint SHA256 and graph SHA256 before reuse, with regression tests. Only the corrected run is included in the comparison.",
        "",
        "Corrected Linux runtime run: https://github.com/BasmallaMetwally/ml-training-inference-lab/actions/runs/43453954524",
    ]
    Path("results/performance/REPORT.md").write_text("\n".join(lines) + "\n")
    findings = f"""# ما الذي وجدناه

**ML Training and Inference Lab — v1.0 — 13 نوفمبر 2026**

1. **الدقة والحجم والسرعة لا يتحركون معًا.** عبر 10 seeds، حقق الـMLP دقة {mlp["mean"] * 100:.3f}% ± {mlp["std"] * 100:.3f} نقطة مئوية، والـCNN دقة {cnn["mean"] * 100:.3f}% ± {cnn["std"] * 100:.3f}. فرق الـCNN المزدوج {difference["mean"] * 100:.3f} نقطة، بفاصل ثقة 95% من {a:.3f} إلى {b:.3f} وp={difference["exact_sign_flip_p_two_sided"]:.6f}. رغم امتلاكه 9,098 parameters مقابل 109,770، كان الـCNN أبطأ عند batch=1 على الجهاز المقاس. فواصل الثقة تصف اختلاف التدريب بين الـseeds على تقسيم ثابت، وليس اختلاف جميع البيانات الممكنة.

2. **الـfusion تحسن حقيقي لكنه يعتمد على حجم الدفعة وحدود القياس.** استمر التحسن مع تشغيل فحص المدخلات وإيقافه. يوضح profiler انخفاض عدد استدعاءات ATen بعد إزالة مسارات BatchNorm، بينما تكبر تكلفة فحص كل عناصر المدخلات مع حجم الدفعة. لم تُقاس تكلفة dispatch الخالصة بشكل مستقل. جدول ذاكرة Linux الأصلي غير حاسم؛ القياس المصحح يستخدم VmHWM وثلاثة تكرارات لكل حالة. int8 الذي يغيّر التصنيفات يُرفض تحت شرط التطابق التام؛ نجاح compile على Linux لا يعني نجاحه على macOS.

3. **نتائج الـagents كشفت سقفًا للاختبار.** نجح GPT-6-Luna وGPT-6.1-Sol في 36/36 محاولة لكل نموذج: 12 مهمة × 3 تكرارات، داخل Docker بدون شبكة، مع المراجع خارج الحاويات. لم نرَ timeout أو crash أو reward hacking. فاصل Wilson الوصفي على المهام الـ12 التي نجحت في جميع التكرارات يقارب 75.8%–100%؛ التكرارات مترابطة والمهام مختارة، فلا يوجد دليل كافٍ لترتيب النموذجين أو تعميم نسبة 100%.

4. **قابلية الاستكمال محلية بالبيئة.** تطابقت الأوزان حرفيًا بين التدريب المتصل والمتوقف داخل Linux وداخل macOS، لكن بصمات Linux اختلفت عن macOS. وفي النموذج الأصلي، بلغ Shirt recall نحو 72.7%؛ أمثلة Shirt→T-shirt تشير إلى تشابه بصري بدقة منخفضة، وهو تفسير محتمل وليس سببًا مثبتًا.

الأدلة: [Pareto](results/study/pareto.png)، [السرعة والذاكرة](results/performance/REPORT.md)، [تجارب النماذج](results/agents/REPORT.md)، [أمثلة الأخطاء](results/demo/shirt_errors.png). جميع النتائج تعليمية ومحدودة بالنماذج والبيانات والأجهزة المختبرة.
"""
    Path("FINDINGS.ar.md").write_text(findings)
    Path("POST.md").write_text(f"""# What a small ML engineering study actually showed

I built an ML training and inference lab with coding-agent assistance, then tested its claims rather than stopping at a demo.

Across ten paired seeds, a small CNN improved Fashion-MNIST accuracy by {difference["mean"] * 100:.2f} percentage points over an MLP, with a 95% seed-level interval of {a:.2f}–{b:.2f}. It used about twelve times fewer parameters but had higher batch-1 CPU latency on the measured machine.

I also compared checked and unchecked inference, profiled operator calls, measured peak process RSS, and tested fusion, compilation, dynamic int8 and ONNX. Optimizations that changed predictions were reported as rejected under an exact-parity policy.

For evaluation practice, two models each attempted twelve ML repair tasks three times. Both passed every task under withheld Docker grading. That is a ceiling effect, not a credible model ranking. Transcripts, uncertainty caveats and the negative findings are included.

Finally, checkpoint resumption was bitwise reproducible within Linux and macOS, while the cross-platform tensors differed.

Code and evidence: https://github.com/BasmallaMetwally/ml-training-inference-lab
""")


if __name__ == "__main__":
    main()
