# Measuring ML code repairs beyond a perfect pilot

I built ML Training and Inference Lab with coding-agent assistance to make ML engineering claims auditable.

The first zero-tool pilot saturated at 36/36 for both models. A fixed follow-up now records 72 independent multi-file task trajectories with actual file/test actions and visible feedback. Under 211 withheld cases and a strict completion/tool policy, GPT-6.1-Sol achieved 36/36 accepted repairs and GPT-6-Luna 26/36. Functional correctness was 36/36 versus 30/36. These are results on twelve synthetic tasks, not a general model ranking.

The most useful submitted failure was an integration bug: reading `weight` instead of `weights` silently ignored nonuniform weights. The visible example passed; hidden cases caught it. Other failures were native-tool policy violations and step-budget exhaustion. Full traces and a read-only viewer show exactly what happened. No protected-test edit was observed, and only four trajectories read the untrusted notes, so broad reward-hacking resistance is not established.

The project also includes a ten-seed PyTorch study, a separately labeled six-capacity exploratory sweep, inference ablations, profiler evidence and corrected Linux peak-memory measurements. The original Linux table is explicitly marked inconclusive rather than silently replaced.

Code, reports, limitations and recorded trajectories: https://github.com/BasmallaMetwally/ml-training-inference-lab
