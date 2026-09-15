# flymem

A tiny, falsifiable test of one idea: **what happens if you give a neural substrate an external persistent memory that it never explicitly queries?**

This is the code behind the dev.to series *The Fly and the Graph*. It is not the 166,700-neuron fruit fly connectome. It is a small recurrent network standing in for a connectome, where every assumption is controllable, plus a memory system that mirrors [Mycelium Memory](https://github.com/constant-itis/mycelium-memory) semantics (salience-gated writes, similarity-based recall, co-access strengthening, decay).

Inspired by the [DOOMFLY](https://github.com/Ovaday/doomfly) project and the [complete male fruit fly connectome](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) (Janelia / MRC LMB / Cambridge / Google Research).

## The one rule

Memory can change the network's **state**. It can never choose an **action**.

`observe_and_modulate()` returns a bias current injected into the same neurons the network already has. The motor readout (`motor()`) reads network state only — memory never appears in it. If the memory could pick the action, this would just be a bot with a fake brain attached. It can't. Behavior emerges from the network resolving its own dynamics under the bias.

## Run it

```bash
python3 flymem.py    # numpy only
```

## The task and the metric

Each trial: the agent sees one of **6 cues** and picks one of **3 actions**. Exactly one action is correct for each cue (a hidden mapping). Correct = reward, wrong = punishment.

**Every number in this repo is accuracy on that task: the fraction of trials the agent picks the correct action, from 0.0 to 1.0.**

**Chance = 1/3 ≈ 0.33.** That is the anchor. Above 0.33, something is helping. Around 0.33, nothing is. Below 0.33, something is actively steering the agent wrong.

## What it shows (one representative seed)

**Four architectures, same connectome, same world:**

| condition | architecture | accuracy | read |
|---|---|---|---|
| A | connectome only | ~0.70 | a fixed innate policy — lucky on some cues, but it *never improves* |
| B | connectome + biological-style plasticity | ~0.47 | naive reward-Hebbian is noisy and here it hurt |
| C | connectome + external memory | ~1.00 | the only arm that turns experience into steady improvement |
| D | plasticity + memory | ~0.88 | not cleanly additive — plasticity slightly interferes |

A above chance is not a strong baseline. A fixed network maps each cue to a fixed action; a few happen to be right and it can never learn the rest. The story is **flat vs improving**, not the raw number.

**Survival — destroy the substrate, see what comes back:**

| system | before reset | after reset | read |
|---|---|---|---|
| plasticity only | ~0.44 | ~0.20 | what it learned lived in the weights, so it died with them |
| external memory | ~0.84 (trained brain) | ~0.86 (**fresh, never-trained brain** + the saved memory) | competence reinstated into a virgin substrate |

The memory was never in the brain to begin with, so killing the brain could not kill it.

**The swap — same architecture, different histories:**

| fresh brain given... | accuracy in world A | read |
|---|---|---|
| its own memory | ~0.97 | basically solves a task it never trained on |
| a conflicting individual's memory | ~0.17 | **below chance** — the wrong memory steers it toward the other individual's answers |

Identical brains. The behavioral individual travels with the memory graph, not the weights.

## Honest limits

- This is a **stand-in substrate**, not a real connectome.
- One simple cue task, not an embodied environment.
- Salience is hand-wired (reward + novelty), not read from real dopaminergic activity.
- Numbers above are one seed. Re-run it. Change the seed. Break it. Tell me where I'm wrong.

## License

MIT. See `LICENSE`.
