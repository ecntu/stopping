import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange, reduce, repeat
import optax
from optax import softmax_cross_entropy_with_integer_labels as softmax_ce
from functools import partial
from dataclasses import dataclass
import simple_parsing


def gen_parity_batch(key, bs, seq_len=96, min_nonzero=1, max_nonzero=48):
    """Parity examples a la PonderNet.

    A random number of positions are set to +/-1 uniformly, with the rest at 0.
    The label is the parity of the count of +1s.

    Training: max_nonzero = seq_len // 2 (e.g. 48 for seq_len=96).
    Extrapolation eval: set min_nonzero/max_nonzero higher, up to seq_len.
    """

    k_count, k_pos, k_sign = jax.random.split(key, 3)

    n_nonzero = jax.random.randint(k_count, (bs,), min_nonzero, max_nonzero + 1)

    # Pick nonzero positions by ranking random scores
    scores = jax.random.uniform(k_pos, (bs, seq_len))
    ranks = jnp.argsort(scores, axis=-1).argsort(axis=-1)
    mask = ranks < n_nonzero[:, None]

    # Random signs for the nonzero positions
    signs = jax.random.rademacher(k_sign, (bs, seq_len)).astype(jnp.float32)
    x = jnp.where(mask, signs, 0.0)
    y = ((x == 1.0).sum(axis=-1) % 2).astype(jnp.int32)

    return x, y


class Model(nnx.Module):
    """Simple PonderNet-style GRUCell and two heads"""

    def __init__(self, seq_len, h_dim, max_steps, rngs):

        self.max_steps = max_steps

        self.cell = nnx.GRUCell(in_features=seq_len, hidden_features=h_dim, rngs=rngs)
        self.out_head = nnx.Linear(in_features=h_dim, out_features=2, rngs=rngs)
        self.halt_head = nnx.Linear(in_features=h_dim, out_features=1, rngs=rngs)

    def __call__(self, x):

        bs, seq_len = x.shape

        def step(carry, _):
            carry, _ = self.cell(carry, x)
            return carry, (self.out_head(carry), self.halt_head(carry))

        _, (ys, hs) = jax.lax.scan(
            step,
            init=jnp.zeros((bs, self.cell.hidden_features)),
            xs=None,
            length=self.max_steps,
        )

        ys = rearrange(ys, "t b c -> t b c")
        hs = rearrange(hs, "t b c -> t b c")
        return ys, hs


def loss_fn(model, x, y):
    """Continious Thought Machine style loss"""

    ys, _ = model(x)

    t, bs, c = ys.shape

    per_tick_loss = softmax_ce(ys, repeat(y, "b -> t b", t=t))

    per_tick_entropy = reduce(
        (-jnp.exp(ys) * jax.nn.log_softmax(ys)), "t b c -> t b", reduction="sum"
    )
    per_tick_certainty = 1 - (per_tick_entropy / jnp.log(c))

    b_ix = jnp.arange(bs)
    min_loss_ix = per_tick_loss.argmin(axis=0)
    max_cert_ix = per_tick_certainty.argmax(axis=0)

    loss = 0.5 * (per_tick_loss[min_loss_ix, b_ix] + per_tick_loss[max_cert_ix, b_ix])
    return reduce(loss, "b ->", "mean")


grad_fn = nnx.value_and_grad(loss_fn)


@nnx.jit
def eval_batch(model, x, y):
    ys, _ = model(x)
    return (jnp.argmax(ys[-1], axis=-1) == y).sum()


def test_acc(key, model, gen_batch_fn, steps):
    correct, total = 0, 0
    for k in jax.random.split(key, steps):
        x, y = gen_batch_fn(k)
        correct += eval_batch(model, x, y)
        total += len(y)
    return correct / total


@dataclass
class Config:
    seq_len: int = 96
    h_dim: int = 128
    max_steps: int = 20

    batch_size: int = 128
    lr: float = 3e-4
    steps: int = 20_000
    test_steps: int = 32
    warmup_steps: int = 500
    grad_clip: float = 1.0

    seed: int = 0


if __name__ == "__main__":
    cfg = simple_parsing.parse(Config)
    rngs = nnx.Rngs(params=cfg.seed, train=cfg.seed + 1)
    test_key = jax.random.key(cfg.seed)

    gen_batch = partial(gen_parity_batch, bs=cfg.batch_size, seq_len=cfg.seq_len)
    gen_train_batch = partial(gen_batch, max_nonzero=cfg.seq_len // 2)
    gen_eval_batch = partial(gen_batch, max_nonzero=cfg.seq_len)
    test_acc = partial(
        test_acc, key=test_key, gen_batch_fn=gen_eval_batch, steps=cfg.test_steps
    )

    model = Model(
        seq_len=cfg.seq_len, h_dim=cfg.h_dim, max_steps=cfg.max_steps, rngs=rngs
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.steps,
    )
    tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(schedule))
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def train_step(model, opt, x, y):
        loss, grads = grad_fn(model, x, y)
        opt.update(model, grads)
        return loss

    for step in range(cfg.steps):
        x, y = gen_train_batch(rngs.train())
        loss = train_step(model, opt, x, y)

        if step % 25 == 0:
            acc = test_acc(model=model)
            print(f"Step {step} - Loss: {loss:.4f} - Eval acc: {acc:.4f}")
