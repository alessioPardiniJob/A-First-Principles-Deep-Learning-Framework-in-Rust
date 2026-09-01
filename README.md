# A First-Principles Deep Learning Framework in Rust

A small deep learning framework written from scratch in Rust, with **no external
dependencies**, no ML, tensor or autodiff crates, and nothing at all beyond the
Rust standard library.

The design is specified in [`docs/architecture.pdf`](docs/architecture.pdf), which
derives four abstractions, `Tensor`, `Module`, `Loss` and `Optimizer`, from the
structural, differentiation and optimization requirements of training a network.
The code implements that specification.

---

## How to run

**You only need Rust** (1.70 or newer). If you don't have it: <https://rustup.rs>.

From the project folder:

```sh
cargo run --release
```

That's it, it builds and runs the whole experiment suite (about 35 seconds).
On Windows you can also just double-click **`run.bat`**.

To run the tests:

```sh
cargo test
```

### Datasets

The command above works straight away with no setup: if the real datasets are not
present, the program falls back to synthetic data and says so on screen.

To use the real MNIST and California Housing data instead, download them once
(needs internet):

- **Windows:** `powershell -File scripts/download_data.ps1`
- **Linux / macOS:** `./scripts/download_data.sh`

Then run `cargo run --release` again. The `run.bat` / `run.sh` scripts do both
steps for you. Datasets land in `data/`, which is not committed to the repository.

### Why there is no Docker image

A container would only add prerequisites, Docker Desktop, WSL2 on Windows, a
~1 GB base image to download, to solve a problem this project does not have: it
has zero external dependencies and compiles offline in a few seconds. Installing
Rust through rustup is lighter and faster than installing Docker.

---

## What the experiments show

`cargo run --release` executes six experiments that together exercise every part
of the framework:

| # | Experiment | What it validates |
|---|---|---|
| 1 | Convolutional vs fully-connected network on MNIST | `Conv2d`, `Flatten`, `Linear`, `ReLU`; training time and accuracy trade-offs |
| 2 | Regression on California Housing | `MSELoss`, continuous targets, generalization vs a mean-predicting baseline |
| 3 | SGD vs Momentum | `Optimizer` trait, stateless vs stateful optimization, lazily allocated velocity |
| 4 | Batch size 10 / 50 / 200 | mini-batch training, gradient averaging, time vs convergence |
| 5 | Gradient check | analytical VJPs vs central finite differences, through both stacks |
| 6 | O(1) tensor views | `t()` / `reshape()` as pure metadata edits vs a full buffer copy |

Experiment 5 is the strongest correctness evidence: it validates the entire
backward chain — every layer's VJP, the loss gradient seed, and gradient
accumulation — against numerically computed derivatives of the loss itself.

Runs are **reproducible**: dataset order is fixed and model parameters are
re-initialized from a seeded PRNG, so every reported metric is identical across
runs (only wall-clock timings vary).

Beyond the experiments, `cargo test` runs 30 tests: unit tests for every tensor
operation, layer, loss and optimizer, including finite-difference gradient
checks, plus end-to-end training integration tests.

---

## Project layout

```
src/
  lib.rs                    the library crate: the framework itself
  tensor.rs                 Tensor: flat Rc<Vec<f32>> buffer + shape/strides
  module.rs                 Module trait (forward / backward / params / grads / zero_grad)
  module/
    sequential.rs           Sequential: Vec<Box<dyn Module>>, itself a Module
    linear.rs               fully connected layer
    relu.rs                 rectified linear activation
    conv2d.rs               2D convolution over [B, C, H, W], via im2col
    flatten.rs              [B, ...] -> [B, F], bridges Conv2d to Linear
  loss.rs                   Loss trait, with an associated Target type
  loss/
    mse.rs                  MSELoss              (Target = Tensor)
    softmax_cross_entropy.rs SoftmaxCrossEntropy (Target = [usize])
  optimizer.rs              Optimizer trait (step over &mut dyn Module)
  optimizer/
    sgd.rs                  stateless gradient descent
    momentum.rs             gradient descent with velocity

  main.rs                   the experiment suite (binary only)
  datasets.rs, datasets/    dataset loading for the experiments (binary only,
                            not part of the library API)

tests/end_to_end.rs         integration tests: full training runs
docs/architecture.pdf       the design specification this code implements
```

The library crate (`lib.rs` and everything it declares) contains only the
framework. `main.rs` and `datasets/` belong to the binary and are never exposed as
framework API, so the experiments consume the library exactly as any external user
would.

---

