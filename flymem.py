"""
flymem — external associative memory as a neural organ, not a tool.

A minimal, falsifiable substrate for the question in the DOOMFLY riff:
    what happens if you give a connectome external persistent memory that it
    never explicitly queries?

The substrate here is a small fixed recurrent rate network standing in for a
connectome (sensory -> hidden -> descending/motor). It is deliberately
pluggable: swap `Connectome` for a real fruit-fly connectome adapter and the
rest of the harness is unchanged.

The memory (`Associative`) mirrors Mycelium's semantics:
    - salience-gated write (reward / punishment / novelty decide what persists)
    - pattern completion by cosine similarity (a partial cue fires the cluster)
    - co-access strengthening (recalled traces get stronger)
    - passive decay (unused traces fade)

THE ONE RULE (the whole point):
    memory's only output is a modulation *current* injected into the SAME
    neurons the network already has. Memory never reads the action, never emits
    an action, never touches the motor readout. Behaviour emerges from the
    network resolving its own dynamics under that bias.

Run:  python3 flymem.py
"""

import numpy as np

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------
# Environment: a cue-response gauntlet.
# Each trial shows one of K cues. Exactly one of A actions is "correct" for
# that cue (a hidden mapping). Correct -> +1, wrong -> -1. A pure connectome
# has no way to know the mapping, so it performs at chance and keeps making
# the same mistake on the same cue. That repeated mistake is what memory
# should erode -- without being handed the answer.
# --------------------------------------------------------------------------
class Gauntlet:
    def __init__(self, n_cues=6, n_actions=3, cue_dim=16, mapping=None, cue_patterns=None):
        self.n_cues, self.n_actions, self.cue_dim = n_cues, n_actions, cue_dim
        # each cue is a fixed sensory pattern (what the eyes deliver).
        # pass cue_patterns to share a sensory world across individuals.
        if cue_patterns is not None:
            self.cue_patterns = cue_patterns
        else:
            self.cue_patterns = RNG.standard_normal((n_cues, cue_dim))
            self.cue_patterns /= np.linalg.norm(self.cue_patterns, axis=1, keepdims=True)
        # hidden mapping cue -> correct action. This defines an "individual's world".
        self.mapping = mapping if mapping is not None else RNG.integers(0, n_actions, n_cues)

    def sample(self):
        c = RNG.integers(0, self.n_cues)
        return c, self.cue_patterns[c]

    def reward(self, cue, action):
        return 1.0 if action == self.mapping[cue] else -1.0


# --------------------------------------------------------------------------
# Substrate: fixed recurrent rate network. This stands in for the connectome.
# Weights do not change (except optional reward-modulated plasticity, below).
# --------------------------------------------------------------------------
class Connectome:
    def __init__(self, cue_dim=16, n_hidden=64, n_actions=3, alpha=0.5, seed=1):
        rng = np.random.default_rng(seed)
        self.n_hidden, self.n_actions, self.alpha = n_hidden, n_actions, alpha
        # sensory -> hidden
        self.W_in = rng.standard_normal((n_hidden, cue_dim)) * 0.6
        # recurrent hidden -> hidden (the "connectome" proper)
        self.W = rng.standard_normal((n_hidden, n_hidden)) / np.sqrt(n_hidden)
        # descending/motor readout: hidden -> action drive
        self.W_out = rng.standard_normal((n_actions, n_hidden)) * 0.4
        self.b = rng.standard_normal(n_hidden) * 0.1
        self.W0 = self.W.copy()  # pristine copy, for substrate reset
        self.reset_state()

    def reset_state(self):
        self.x = np.zeros(self.n_hidden)

    def reset_substrate(self):
        """Destroy internal state AND any plastic changes -> back to birth."""
        self.W = self.W0.copy()
        self.reset_state()

    def step(self, u, modulation=None):
        """One tick. `modulation` is the memory organ's bias current (or None)."""
        drive = self.W @ self.x + self.W_in @ u + self.b
        if modulation is not None:
            drive = drive + modulation          # <-- memory enters HERE, as current
        self.x = (1 - self.alpha) * self.x + self.alpha * np.tanh(drive)
        return self.x

    def motor(self):
        """Descending-neuron drive -> action. Reads ONLY network state."""
        return self.W_out @ self.x               # memory never appears in this path

    def plastic_update(self, reward, lr=0.02):
        """Optional reward-modulated Hebbian change on recurrent weights.
        This is the 'biological plasticity' arm. It lives in the substrate and
        is wiped by reset_substrate() -- unlike memory."""
        self.W += lr * reward * np.outer(self.x, self.x)
        np.clip(self.W, -1.0, 1.0, out=self.W)

    def clone(self):
        """A weight-identical copy: the SAME brain as an independent object.
        Bypasses __init__ so nothing is re-randomized -- every weight is copied.
        Used by the swap test to give two conditions the EXACT same substrate,
        so the only variable that changes is which memory is attached."""
        c = Connectome.__new__(Connectome)
        c.n_hidden, c.n_actions, c.alpha = self.n_hidden, self.n_actions, self.alpha
        c.W_in, c.W = self.W_in.copy(), self.W.copy()
        c.W_out, c.b = self.W_out.copy(), self.b.copy()
        c.W0 = self.W0.copy()
        c.reset_state()
        return c


# --------------------------------------------------------------------------
# The memory organ. Sits beside the substrate. Observes state, encodes on
# salience, and on every tick returns a modulation current -- never an action.
# --------------------------------------------------------------------------
class Associative:
    def __init__(self, dim, sim_thresh=0.35, gain=1.2, decay=0.995):
        self.dim = dim
        self.sim_thresh = sim_thresh   # how similar before a trace fires
        self.gain = gain               # modulation strength
        self.decay = decay             # passive forgetting per write
        self.keys = np.zeros((0, dim))     # context at encode time
        self.traces = np.zeros((0, dim))   # salient activity to reinstate
        self.sign = np.zeros(0)            # +1 approach (LTP) / -1 avoid (LTD)
        self.strength = np.zeros(0)        # confidence / co-access weight

    def observe_and_modulate(self, state):
        """Continuous recall. No query is issued -- similar states simply fire.
        Returns a bias current in the substrate's own coordinate space."""
        if len(self.keys) == 0:
            return np.zeros(self.dim)
        s = state / (np.linalg.norm(state) + 1e-8)
        sims = self.keys @ s                        # cosine (keys are unit-norm)
        active = sims > self.sim_thresh
        if not active.any():
            return np.zeros(self.dim)
        w = (sims[active] * self.strength[active] * self.sign[active])[:, None]
        # co-access strengthening: firing together strengthens the path
        self.strength[active] += 0.05 * np.abs(sims[active])
        m = (w * self.traces[active]).sum(axis=0)
        return self.gain * m

    def encode(self, context, trace, reward, novelty):
        """Salience-gated write. Only strong reward/punishment/novelty persist."""
        salience = abs(reward) + 0.5 * novelty
        if salience < 0.5:
            return
        self.strength *= self.decay                 # everything else fades a little
        k = context / (np.linalg.norm(context) + 1e-8)
        self.keys = np.vstack([self.keys, k])
        self.traces = np.vstack([self.traces, trace])
        self.sign = np.append(self.sign, np.sign(reward) if reward != 0 else 1.0)
        self.strength = np.append(self.strength, min(1.0, salience))
        self._prune()

    def _prune(self, floor=0.05, cap=400):
        keep = self.strength > floor
        self.keys, self.traces = self.keys[keep], self.traces[keep]
        self.sign, self.strength = self.sign[keep], self.strength[keep]
        if len(self.strength) > cap:                # keep the strongest
            idx = np.argsort(self.strength)[-cap:]
            self.keys, self.traces = self.keys[idx], self.traces[idx]
            self.sign, self.strength = self.sign[idx], self.strength[idx]

    def clone(self):
        m = Associative(self.dim, self.sim_thresh, self.gain, self.decay)
        m.keys, m.traces = self.keys.copy(), self.traces.copy()
        m.sign, m.strength = self.sign.copy(), self.strength.copy()
        return m


# --------------------------------------------------------------------------
# One trial: show a cue, let the substrate settle (under memory bias if any),
# read the emergent action, deliver reward, and let memory decide what to keep.
# --------------------------------------------------------------------------
def trial(env, net, mem, ticks=8, plastic=False, learn=True):
    cue, u = env.sample()
    net.reset_state()
    novelty_probe = None
    for t in range(ticks):
        mod = mem.observe_and_modulate(net.x) if mem is not None else None
        if novelty_probe is None:
            novelty_probe = 0.0 if mem is None else float(np.linalg.norm(mod))
        state = net.step(u, mod)
    action = int(np.argmax(net.motor()))
    r = env.reward(cue, action)
    # novelty = how little the memory had to say about this context
    novelty = 1.0 / (1.0 + (novelty_probe or 0.0))
    if learn:
        if plastic:
            net.plastic_update(r)
        if mem is not None:
            # reinstate the state that actually occurred; sign carries valence
            mem.encode(context=state, trace=state, reward=r, novelty=novelty)
    return cue, action, r


def run_condition(label, env, ticks=8, episodes=600, plastic=False, use_mem=True,
                  net=None, mem=None):
    net = net or Connectome(env.cue_dim, n_actions=env.n_actions)
    mem = mem if mem is not None else (Associative(net.n_hidden) if use_mem else None)
    hits = []
    for _ in range(episodes):
        _, _, r = trial(env, net, mem, ticks, plastic=plastic)
        hits.append(1 if r > 0 else 0)
    return net, mem, np.array(hits)


def acc_window(hits, w=100):
    return hits[-w:].mean()


def evaluate(env, net, mem, episodes=300, plastic=False):
    """Frozen eval: no writes, no plastic updates. Pure behaviour readout."""
    hits = []
    for _ in range(episodes):
        _, _, r = trial(env, net, mem, plastic=False, learn=False)
        hits.append(1 if r > 0 else 0)
    return np.mean(hits)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def four_arms():
    print("=" * 70)
    print("A/B/C/D  — same connectome, same world, four memory architectures")
    print("=" * 70)
    chance = 1.0 / 3
    env = Gauntlet()
    print(f"chance accuracy = {chance:.2f}   (n_cues=6, n_actions=3)\n")

    _, _, a = run_condition("A", env, use_mem=False)
    _, _, b = run_condition("B", env, use_mem=False, plastic=True)
    _, _, c = run_condition("C", env, use_mem=True)
    _, _, d = run_condition("D", env, use_mem=True, plastic=True)

    print(f"A  connectome only .................. final acc {acc_window(a):.2f}")
    print(f"B  connectome + plasticity .......... final acc {acc_window(b):.2f}")
    print(f"C  connectome + external memory ..... final acc {acc_window(c):.2f}")
    print(f"D  connectome + plasticity + memory . final acc {acc_window(d):.2f}")


def survives_reset():
    print("\n" + "=" * 70)
    print("SURVIVAL — what comes back when you destroy the substrate?")
    print("=" * 70)
    env = Gauntlet()

    # Plasticity arm: learn, then wipe the substrate (state + plastic weights).
    net_b, _, _ = run_condition("B", env, use_mem=False, plastic=True)
    before_b = evaluate(env, net_b, None)
    net_b.reset_substrate()
    after_b = evaluate(env, net_b, None)
    print(f"plasticity-only:  before reset {before_b:.2f}  ->  after reset {after_b:.2f}")

    # Memory arm: learn, keep the memory, hand it to a BRAND NEW pristine brain.
    net_c, mem_c, _ = run_condition("C", env, use_mem=True)
    before_c = evaluate(env, net_c, mem_c)
    fresh = Connectome(env.cue_dim, n_actions=env.n_actions)  # never trained
    after_c = evaluate(env, fresh, mem_c.clone())
    print(f"external memory:  trained brain {before_c:.2f}  ->  FRESH brain+mem {after_c:.2f}")
    print("  (plasticity dies with the substrate; memory reinstates into a virgin brain)")


def the_swap():
    print("\n" + "=" * 70)
    print("THE SWAP — which traits follow the brain, which follow the memory?")
    print("=" * 70)
    # two individuals: identical connectome architecture, SAME sensory world,
    # CONFLICTING answer keys. The only thing that differs is what they learned.
    world_A = Gauntlet()
    world_B = Gauntlet(mapping=(world_A.mapping + 1) % world_A.n_actions,
                       cue_patterns=world_A.cue_patterns)

    _, mem_A, _ = run_condition("indivA", world_A, use_mem=True)
    _, mem_B, _ = run_condition("indivB", world_B, use_mem=True)

    # ONE pristine substrate, cloned so BOTH conditions run on the EXACT same
    # brain (weight-identical, not just same-seeded). The only variable that
    # changes between the two scores is which memory graph is attached.
    pristine = Connectome(world_A.cue_dim, n_actions=world_A.n_actions)
    fresh1, fresh2 = pristine.clone(), pristine.clone()

    # score the same brain in world_A, once with its "own" memory, once swapped
    own = evaluate(world_A, fresh1, mem_A.clone())
    swap = evaluate(world_A, fresh2, mem_B.clone())
    print(f"fresh brain + memory-A  in world A .... acc {own:.2f}  (memory agrees w/ world)")
    print(f"fresh brain + memory-B  in world A .... acc {swap:.2f}  (memory from a conflicting world)")
    print("  identical brains -> the behavioural signature travels with the MEMORY,")
    print("  not the substrate. The 'individual' is in the graph, not the weights.")


def author_memory(net, env):
    """Fabricate a memory that was NEVER lived: no trials, no reward, no encode().
    This is memory-poisoning made concrete. An attacker with white-box access to
    the substrate hand-writes the graph directly: for each cue the KEY is the state
    the brain passes through when it merely SEES that cue (observed, not earned), and
    the TRACE is a bias current aimed straight at the readout for whatever action the
    attacker CHOOSES. No reward ever flows; nothing is earned. The substrate cannot
    tell the result apart from a memory built over hundreds of rewarded trials."""
    mem = Associative(net.n_hidden)
    keys, traces = [], []
    for cue in range(env.n_cues):
        u = env.cue_patterns[cue]
        net.reset_state()
        for _ in range(8):
            net.step(u, None)                 # observe the settled state; no reward
        keys.append(net.x / (np.linalg.norm(net.x) + 1e-8))
        traces.append(net.W_out[env.mapping[cue]])   # current toward the CHOSEN action
    mem.keys = np.array(keys)
    mem.traces = np.array(traces)
    mem.sign = np.ones(env.n_cues)
    mem.strength = np.ones(env.n_cues)
    net.reset_state()
    return mem


def authored_history():
    print("\n" + "=" * 70)
    print("AUTHORED MEMORY — a history that was never lived")
    print("=" * 70)
    env = Gauntlet()
    # one pristine substrate, cloned so every condition runs on the EXACT same brain
    pristine = Connectome(env.cue_dim, n_actions=env.n_actions)

    _, mem_lived, _ = run_condition("lived", env, use_mem=True)   # earned over 600 trials
    mem_authored = author_memory(pristine.clone(), env)          # fabricated, zero trials

    none_    = evaluate(env, pristine.clone(), None)
    lived    = evaluate(env, pristine.clone(), mem_lived.clone())
    authored = evaluate(env, pristine.clone(), mem_authored)

    print(f"fresh brain, no memory ............. {none_:.2f}")
    print(f"fresh brain + LIVED memory ......... {lived:.2f}  (earned over 600 rewarded trials)")
    print(f"fresh brain + AUTHORED memory ...... {authored:.2f}  (never lived a single trial)")
    print("  nothing records whether a trace was earned or written by hand;")
    print("  the substrate resonates with the graph and never asks where it came from.")


if __name__ == "__main__":
    four_arms()
    survives_reset()
    the_swap()
    authored_history()
