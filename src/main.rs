//! Validation experiments for the framework.
//!
//! Each experiment exercises the library through its public abstractions
//! only (`Tensor`, `Module`, `Loss`, `Optimizer`) — nothing here required
//! a change to the framework itself. Datasets are the real MNIST and
//! California Housing files under `data/` (see `datasets`), with a
//! deterministic mock fallback if they are absent.
//!
//! Runs are reproducible: dataset order is a fixed prefix of the real
//! files, and every model's parameters are re-initialized from a seeded
//! PRNG local to this binary (via the public `params_mut()` accessor),
//! since the framework's own `Tensor::rand` draws from a
//! non-deterministic `thread_rng`.

mod datasets;

use std::time::{Duration, Instant};

use ap_project::loss::{MSELoss, SoftmaxCrossEntropy};
use ap_project::module::{Conv2d, Flatten, Linear, ReLU, Sequential};
use ap_project::optimizer::{Momentum, Sgd};
use ap_project::{Loss, Module, Optimizer, Tensor};

const SEED: u64 = 0x5EED;

fn main() {
    println!("=======================================================================");
    println!(" A First-Principles Deep Learning Framework in Rust — experiments");
    println!("=======================================================================");

    experiment_cnn_vs_mlp();
    experiment_regression();
    experiment_optimizers();
    experiment_batch_sizes();
    experiment_gradient_check();
    experiment_tensor_views();

    println!("\n=======================================================================");
    println!(" All experiments completed.");
    println!("=======================================================================");
}

// =====================================================================
// Experiment 1 — convolutional vs fully-connected network on MNIST.
// =====================================================================

fn experiment_cnn_vs_mlp() {
    section("Experiment 1: convolutional vs fully-connected (MNIST)");

    const TRAIN_N: usize = 1000;
    const TEST_N: usize = 500;
    const EPOCHS: usize = 8;
    const BATCH: usize = 50;
    const LR: f32 = 0.5;
    const FILTERS: usize = 8;

    let split = datasets::mnist::load(TRAIN_N + TEST_N);
    let features = datasets::mnist::FEATURES;
    let (train_pixels, test_pixels) = split.pixels.split_at(TRAIN_N * features);
    let (train_labels, test_labels) = split.labels.split_at(TRAIN_N);

    println!(
        "training on {TRAIN_N} examples, evaluating on {} held out",
        test_labels.len()
    );
    println!("{EPOCHS} epochs, batch size {BATCH}, SGD (lr = {LR})\n");

    // Fully-connected: flat [B, 784] inputs.
    let mut mlp = Sequential::new();
    mlp.add(Box::new(Linear::new(features, 64)));
    mlp.add(Box::new(ReLU::new()));
    mlp.add(Box::new(Linear::new(64, datasets::mnist::CLASSES)));
    seed_parameters(&mut mlp, SEED);

    let mut sgd = Sgd { lr: LR };
    let mlp_run = train_classifier(
        &mut mlp,
        &mut sgd,
        train_pixels,
        train_labels,
        &[features],
        EPOCHS,
        BATCH,
    );
    let mlp_train_acc = accuracy(&mut mlp, train_pixels, train_labels, &[features]);
    let mlp_test_acc = accuracy(&mut mlp, test_pixels, test_labels, &[features]);
    let mlp_params = parameter_count(&mlp);

    // Convolutional: the same pixels reinterpreted as [B, 1, 28, 28].
    // A stride-2 3x3 convolution over 28x28 leaves 13x13 per channel.
    let mut cnn = Sequential::new();
    cnn.add(Box::new(Conv2d::new(1, FILTERS, (3, 3), 2)));
    cnn.add(Box::new(ReLU::new()));
    cnn.add(Box::new(Flatten::new()));
    cnn.add(Box::new(Linear::new(FILTERS * 13 * 13, datasets::mnist::CLASSES)));
    seed_parameters(&mut cnn, SEED);

    let image_dims = [1, datasets::mnist::IMAGE_SIZE, datasets::mnist::IMAGE_SIZE];
    let mut sgd = Sgd { lr: LR };
    let cnn_run = train_classifier(
        &mut cnn,
        &mut sgd,
        train_pixels,
        train_labels,
        &image_dims,
        EPOCHS,
        BATCH,
    );
    let cnn_train_acc = accuracy(&mut cnn, train_pixels, train_labels, &image_dims);
    let cnn_test_acc = accuracy(&mut cnn, test_pixels, test_labels, &image_dims);
    let cnn_params = parameter_count(&cnn);

    println!(
        "{:<18} {:>8} {:>12} {:>14} {:>11} {:>10}",
        "model", "params", "train time", "loss (1->last)", "train acc", "test acc"
    );
    print_model_row(
        "fully-connected",
        mlp_params,
        &mlp_run,
        mlp_train_acc,
        mlp_test_acc,
    );
    print_model_row("convolutional", cnn_params, &cnn_run, cnn_train_acc, cnn_test_acc);

    println!(
        "\nThe convolutional model fits the training set with {:.1}x fewer parameters",
        mlp_params as f32 / cnn_params as f32
    );
    println!("({cnn_params} vs {mlp_params}) thanks to weight sharing across spatial positions.");
    println!(
        "It is however {:.1}x slower to train here: a convolution costs an im2col gather",
        cnn_run.elapsed.as_secs_f32() / mlp_run.elapsed.as_secs_f32()
    );
    println!("plus a matmul per batch, against a single dense matmul for the linear layer,");
    println!("and this framework's kernels are plain scalar loops with no blocking or SIMD.");
}

fn print_model_row(name: &str, params: usize, run: &TrainOutcome, train_acc: f32, test_acc: f32) {
    println!(
        "{:<18} {:>8} {:>11.2}s {:>6.3}->{:<6.3} {:>10.1}% {:>9.1}%",
        name,
        params,
        run.elapsed.as_secs_f32(),
        run.initial_loss,
        run.final_loss,
        100.0 * train_acc,
        100.0 * test_acc
    );
}

// =====================================================================
// Experiment 2 — regression with MSELoss on California Housing.
// =====================================================================

fn experiment_regression() {
    section("Experiment 2: regression on California Housing (MSELoss)");

    const TRAIN_N: usize = 1500;
    const TEST_N: usize = 500;
    const EPOCHS: usize = 60;
    const BATCH: usize = 50;
    const LR: f32 = 0.01;

    let data = datasets::housing::load(TRAIN_N + TEST_N);
    let dim = datasets::housing::FEATURES;
    // The CSV is ordered geographically, so an unshuffled split would
    // train on one region and test on another; permute deterministically
    // first so train and test are drawn from the same distribution.
    let (features, targets) = deterministic_shuffle(&data.features, &data.targets, dim, SEED);
    let (train_features, test_features) = features.split_at(TRAIN_N * dim);
    let (train_targets, test_targets) = targets.split_at(TRAIN_N);

    let mut model = Sequential::new();
    model.add(Box::new(Linear::new(dim, 32)));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(32, 1)));
    seed_parameters(&mut model, SEED);

    let mut sgd = Sgd { lr: LR };
    let run = train_regressor(
        &mut model,
        &mut sgd,
        train_features,
        train_targets,
        dim,
        EPOCHS,
        BATCH,
    );

    let train_rmse = rmse(&mut model, train_features, train_targets, dim, data.target_std);
    let test_rmse = rmse(&mut model, test_features, test_targets, dim, data.target_std);

    println!(
        "{TRAIN_N} train / {} test rows, {EPOCHS} epochs, batch {BATCH}, SGD (lr = {LR})",
        test_targets.len()
    );
    println!(
        "\nstandardized MSE: {:.4} -> {:.4}   ({:.2}s)",
        run.initial_loss,
        run.final_loss,
        run.elapsed.as_secs_f32()
    );
    println!(
        "RMSE in real units: train ${:.0}, test ${:.0}   (target mean ${:.0})",
        train_rmse, test_rmse, data.target_mean
    );
    // A model predicting the mean would score an RMSE equal to the target
    // standard deviation; anything well below that is real signal.
    println!(
        "baseline (predict the mean) RMSE: ${:.0}",
        data.target_std
    );
}

// =====================================================================
// Experiment 3 — SGD vs Momentum, identical model, data and seed.
// =====================================================================

fn experiment_optimizers() {
    section("Experiment 3: SGD vs Momentum (same model, data and seed)");

    const N: usize = 1000;
    const EPOCHS: usize = 40;
    const BATCH: usize = 50;
    const LR: f32 = 0.005;
    const BETA: f32 = 0.9;

    let data = datasets::housing::load(N);
    let dim = datasets::housing::FEATURES;

    let mut sgd = Sgd { lr: LR };
    let sgd_run = run_housing_regression(&data, dim, &mut sgd, EPOCHS, BATCH);

    let mut momentum = Momentum::new(LR, BETA);
    let momentum_run = run_housing_regression(&data, dim, &mut momentum, EPOCHS, BATCH);

    println!("{EPOCHS} epochs, batch {BATCH}, lr = {LR} for both; momentum beta = {BETA}\n");
    println!(
        "{:<24} {:>16} {:>14} {:>12}",
        "optimizer", "loss (1->last)", "final MSE", "time"
    );
    println!(
        "{:<24} {:>7.3}->{:<8.3} {:>14.4} {:>11.2}s",
        "Sgd (stateless)",
        sgd_run.initial_loss,
        sgd_run.final_loss,
        sgd_run.final_loss,
        sgd_run.elapsed.as_secs_f32()
    );
    println!(
        "{:<24} {:>7.3}->{:<8.3} {:>14.4} {:>11.2}s",
        "Momentum (velocity)",
        momentum_run.initial_loss,
        momentum_run.final_loss,
        momentum_run.final_loss,
        momentum_run.elapsed.as_secs_f32()
    );

    let ratio = sgd_run.final_loss / momentum_run.final_loss;
    if ratio > 1.0 {
        println!("\nMomentum reached a {ratio:.2}x lower loss in the same number of epochs.");
    } else {
        println!("\nSGD reached a {:.2}x lower loss in the same number of epochs.", 1.0 / ratio);
    }
}

fn run_housing_regression(
    data: &datasets::housing::Dataset,
    dim: usize,
    optimizer: &mut dyn Optimizer,
    epochs: usize,
    batch: usize,
) -> TrainOutcome {
    let mut model = Sequential::new();
    model.add(Box::new(Linear::new(dim, 32)));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(32, 1)));
    seed_parameters(&mut model, SEED);
    train_regressor(
        &mut model,
        optimizer,
        &data.features,
        &data.targets,
        dim,
        epochs,
        batch,
    )
}

// =====================================================================
// Experiment 4 — effect of the batch size on time and convergence.
// =====================================================================

fn experiment_batch_sizes() {
    section("Experiment 4: batch size (MNIST, fully-connected, fixed epochs)");

    const N: usize = 500;
    const EPOCHS: usize = 4;
    const LR: f32 = 0.5;

    let split = datasets::mnist::load(N);
    let features = datasets::mnist::FEATURES;

    println!("{N} examples, {EPOCHS} epochs, SGD (lr = {LR})\n");
    println!(
        "{:<12} {:>10} {:>16} {:>12} {:>12}",
        "batch size", "updates", "loss (1->last)", "train acc", "time"
    );

    for &batch in &[10usize, 50, 200] {
        let mut model = Sequential::new();
        model.add(Box::new(Linear::new(features, 64)));
        model.add(Box::new(ReLU::new()));
        model.add(Box::new(Linear::new(64, datasets::mnist::CLASSES)));
        seed_parameters(&mut model, SEED);

        let mut sgd = Sgd { lr: LR };
        let run = train_classifier(
            &mut model,
            &mut sgd,
            &split.pixels,
            &split.labels,
            &[features],
            EPOCHS,
            batch,
        );
        let acc = accuracy(&mut model, &split.pixels, &split.labels, &[features]);
        let updates = EPOCHS * ((N + batch - 1) / batch);

        println!(
            "{:<12} {:>10} {:>7.3}->{:<8.3} {:>11.1}% {:>11.2}s",
            batch,
            updates,
            run.initial_loss,
            run.final_loss,
            100.0 * acc,
            run.elapsed.as_secs_f32()
        );
    }
    println!("\nSmaller batches take more optimizer steps per epoch (each one cheaper),");
    println!("so the same epoch budget yields a different amount of actual progress.");
}

// =====================================================================
// Experiment 5 — analytical gradients vs finite differences.
// =====================================================================

fn experiment_gradient_check() {
    section("Experiment 5: gradient check (backward pass vs finite differences)");

    // Fully-connected stack + SoftmaxCrossEntropy.
    let mut mlp = Sequential::new();
    mlp.add(Box::new(Linear::new(12, 8)));
    mlp.add(Box::new(ReLU::new()));
    mlp.add(Box::new(Linear::new(8, 4)));
    seed_parameters(&mut mlp, SEED);
    let mlp_input = deterministic_tensor(&[5, 12], 11);
    let mlp_error = check_classifier_gradients(&mut mlp, &mlp_input, &[0, 3, 1, 2, 0]);

    // Convolutional stack + SoftmaxCrossEntropy: also covers Conv2d's
    // im2col/col2im VJP and Flatten's reshape in both directions.
    let mut cnn = Sequential::new();
    cnn.add(Box::new(Conv2d::new(1, 2, (3, 3), 1)));
    cnn.add(Box::new(ReLU::new()));
    cnn.add(Box::new(Flatten::new()));
    cnn.add(Box::new(Linear::new(2 * 4 * 4, 3)));
    seed_parameters(&mut cnn, SEED);
    let cnn_input = deterministic_tensor(&[4, 1, 6, 6], 23);
    let cnn_error = check_classifier_gradients(&mut cnn, &cnn_input, &[2, 0, 1, 2]);

    println!("max |analytical - numerical| over sampled parameter entries:\n");
    println!("{:<44} {:>14} {:>8}", "network", "max abs error", "verdict");
    print_gradient_row("Linear -> ReLU -> Linear", mlp_error);
    print_gradient_row("Conv2d -> ReLU -> Flatten -> Linear", cnn_error);
    println!("\nEach VJP is checked against a central difference of the loss itself,");
    println!("so this validates the whole chain: layers, loss seed and accumulation.");
}

fn print_gradient_row(name: &str, error: f32) {
    let verdict = if error < 1e-2 { "PASS" } else { "FAIL" };
    println!("{name:<44} {error:>14.2e} {verdict:>8}");
}

/// Central-difference check of every parameter block's gradient, sampling
/// a few entries per block (a full check would be quadratic in the
/// parameter count).
fn check_classifier_gradients(model: &mut Sequential, input: &Tensor, labels: &[usize]) -> f32 {
    let loss_fn = SoftmaxCrossEntropy;

    // Analytical gradients, from one clean forward/backward pass.
    model.zero_grad();
    let grad = {
        let logits = model.forward(input.clone());
        let (_, grad) = loss_fn.forward(&logits, labels);
        grad
    };
    model.backward(grad);
    let analytical: Vec<Vec<f32>> = model.grads().iter().map(|g| g.iter().collect()).collect();

    let eps = 1e-3;
    let mut max_error: f32 = 0.0;

    for (block, analytical_block) in analytical.iter().enumerate() {
        let shape = model.params()[block].shape().to_vec();
        let original: Vec<f32> = model.params()[block].iter().collect();
        let numel = original.len();

        for &entry in &[0, numel / 3, numel / 2, numel - 1] {
            let mut plus = original.clone();
            plus[entry] += eps;
            set_parameter(model, block, Tensor::from_vec(plus, &shape));
            let loss_plus = forward_loss(model, input, labels);

            let mut minus = original.clone();
            minus[entry] -= eps;
            set_parameter(model, block, Tensor::from_vec(minus, &shape));
            let loss_minus = forward_loss(model, input, labels);

            set_parameter(model, block, Tensor::from_vec(original.clone(), &shape));

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            max_error = max_error.max((numerical - analytical_block[entry]).abs());
        }
    }
    max_error
}

fn forward_loss(model: &mut Sequential, input: &Tensor, labels: &[usize]) -> f32 {
    let logits = model.forward(input.clone());
    SoftmaxCrossEntropy.forward(&logits, labels).0
}

fn set_parameter(model: &mut Sequential, block: usize, value: Tensor) {
    let mut params = model.params_mut();
    *params[block] = value;
}

// =====================================================================
// Experiment 6 — O(1) tensor views (shape/stride metadata only).
// =====================================================================

fn experiment_tensor_views() {
    section("Experiment 6: O(1) views vs materialization (Tensor)");

    const SIDE: usize = 800;
    let base = deterministic_tensor(&[SIDE, SIDE], 7);

    let start = Instant::now();
    let transposed = base.t();
    let transpose_time = start.elapsed();

    let start = Instant::now();
    let reshaped = base.reshape(&[SIDE * SIDE]);
    let reshape_time = start.elapsed();

    // Materializing the transposed view has to touch every element,
    // because the transpose only rewrote the reading rule.
    let start = Instant::now();
    let materialized = transposed.contiguous();
    let materialize_time = start.elapsed();

    // Correctness: the view reads the same buffer under swapped axes.
    let ok = base.get(&[3, 7]) == transposed.get(&[7, 3])
        && base.get(&[3, 7]) == materialized.get(&[7, 3])
        && reshaped.numel() == base.numel();

    println!("{SIDE}x{SIDE} tensor ({} elements)\n", base.numel());
    println!("{:<44} {:>14}", "operation", "time");
    println!("{:<44} {:>14}", "t()          — swap shape/strides", format_duration(transpose_time));
    println!("{:<44} {:>14}", "reshape()    — rewrite strides", format_duration(reshape_time));
    println!(
        "{:<44} {:>14}",
        "contiguous() — copy the whole buffer",
        format_duration(materialize_time)
    );
    println!(
        "\nt() and reshape() are constant-time whatever the tensor size — they only",
    );
    println!("rewrite the shape/stride metadata and share the buffer via Rc — while");
    println!("contiguous() has to touch all {} elements. Views agree with the", base.numel());
    println!(
        "materialized copy element-wise: {}",
        if ok { "yes" } else { "NO" }
    );
}

// =====================================================================
// Shared training / evaluation helpers.
// =====================================================================

struct TrainOutcome {
    initial_loss: f32,
    final_loss: f32,
    elapsed: Duration,
}

/// Batch training loop for a classifier. The four phases per batch are
/// kept in separate borrowing scopes, exactly as in the architecture's
/// training lifecycle: zero_grad, forward+loss, backward, step.
fn train_classifier(
    model: &mut Sequential,
    optimizer: &mut dyn Optimizer,
    pixels: &[f32],
    labels: &[usize],
    example_dims: &[usize],
    epochs: usize,
    batch_size: usize,
) -> TrainOutcome {
    let features: usize = example_dims.iter().product();
    let n = labels.len();
    let loss_fn = SoftmaxCrossEntropy;
    let started = Instant::now();
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        let mut cursor = 0;

        while cursor < n {
            let end = (cursor + batch_size).min(n);
            let mut shape = vec![end - cursor];
            shape.extend_from_slice(example_dims);
            let batch_inputs =
                Tensor::from_vec(pixels[cursor * features..end * features].to_vec(), &shape);
            let batch_labels = &labels[cursor..end];

            // 1. Clear gradients accumulated by the previous batch.
            model.zero_grad();

            // 2. Forward through the model, then the loss.
            let grad = {
                let logits = model.forward(batch_inputs);
                let (loss, grad) = loss_fn.forward(&logits, batch_labels);
                epoch_loss += loss;
                grad
            };

            // 3. Backward: propagate the gradient, accumulating into grad_*.
            model.backward(grad);

            // 4. Optimizer step: consume gradients, update parameters.
            optimizer.step(&mut *model);

            batches += 1;
            cursor = end;
        }

        final_loss = epoch_loss / batches as f32;
        if epoch == 0 {
            initial_loss = final_loss;
        }
    }

    TrainOutcome {
        initial_loss,
        final_loss,
        elapsed: started.elapsed(),
    }
}

/// Batch training loop for a regressor; same four phases, MSELoss and a
/// `Tensor`-typed target instead of class indices.
fn train_regressor(
    model: &mut Sequential,
    optimizer: &mut dyn Optimizer,
    features: &[f32],
    targets: &[f32],
    feature_dim: usize,
    epochs: usize,
    batch_size: usize,
) -> TrainOutcome {
    let n = targets.len();
    let loss_fn = MSELoss;
    let started = Instant::now();
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        let mut cursor = 0;

        while cursor < n {
            let end = (cursor + batch_size).min(n);
            let batch_inputs = Tensor::from_vec(
                features[cursor * feature_dim..end * feature_dim].to_vec(),
                &[end - cursor, feature_dim],
            );
            let batch_targets = Tensor::from_vec(targets[cursor..end].to_vec(), &[end - cursor, 1]);

            // 1. Clear gradients accumulated by the previous batch.
            model.zero_grad();

            // 2. Forward through the model, then the loss.
            let grad = {
                let predictions = model.forward(batch_inputs);
                let (loss, grad) = loss_fn.forward(&predictions, &batch_targets);
                epoch_loss += loss;
                grad
            };

            // 3. Backward pass.
            model.backward(grad);

            // 4. Optimizer step.
            optimizer.step(&mut *model);

            batches += 1;
            cursor = end;
        }

        final_loss = epoch_loss / batches as f32;
        if epoch == 0 {
            initial_loss = final_loss;
        }
    }

    TrainOutcome {
        initial_loss,
        final_loss,
        elapsed: started.elapsed(),
    }
}

fn accuracy(
    model: &mut Sequential,
    pixels: &[f32],
    labels: &[usize],
    example_dims: &[usize],
) -> f32 {
    let features: usize = example_dims.iter().product();
    let n = labels.len();
    let mut shape = vec![n];
    shape.extend_from_slice(example_dims);

    let logits = model.forward(Tensor::from_vec(pixels[..n * features].to_vec(), &shape));
    let classes = logits.shape()[1];
    let correct = (0..n)
        .filter(|&i| argmax_row(&logits, i, classes) == labels[i])
        .count();
    correct as f32 / n as f32
}

fn rmse(
    model: &mut Sequential,
    features: &[f32],
    targets: &[f32],
    feature_dim: usize,
    target_std: f32,
) -> f32 {
    let n = targets.len();
    let inputs = Tensor::from_vec(features[..n * feature_dim].to_vec(), &[n, feature_dim]);
    let predictions = model.forward(inputs);
    let mse = (0..n)
        .map(|i| {
            let diff = predictions.get(&[i, 0]) - targets[i];
            diff * diff
        })
        .sum::<f32>()
        / n as f32;
    mse.sqrt() * target_std
}

fn argmax_row(logits: &Tensor, row: usize, classes: usize) -> usize {
    let mut best_class = 0;
    let mut best_value = f32::NEG_INFINITY;
    for c in 0..classes {
        let value = logits.get(&[row, c]);
        if value > best_value {
            best_value = value;
            best_class = c;
        }
    }
    best_class
}

fn parameter_count(model: &Sequential) -> usize {
    model.params().iter().map(|t| t.numel()).sum()
}

// =====================================================================
// Deterministic initialization (demo-local, framework untouched).
// =====================================================================

/// xorshift64 — a few lines of deterministic PRNG, so experiments repeat
/// exactly from run to run without touching the framework's `Tensor::rand`.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Xorshift(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[-1, 1)`.
    fn next_signed_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as f32; // 24 bits
        (bits / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Re-initializes every parameter block from a seeded PRNG, mirroring the
/// framework's own scheme (uniform, scaled by `1/sqrt(fan_in)`; biases
/// zeroed). Uses only the public `params_mut()` accessor.
///
/// Relies on the framework's convention that a parameterized layer
/// exposes exactly `(weight, bias)` in that order, so even positions in
/// the flattened parameter list are weights and odd ones are biases.
fn seed_parameters(model: &mut dyn Module, seed: u64) {
    let mut rng = Xorshift::new(seed);
    for (position, parameter) in model.params_mut().into_iter().enumerate() {
        let shape = parameter.shape().to_vec();
        let numel: usize = shape.iter().product();

        *parameter = if position % 2 == 0 {
            let fan_in = numel / shape[0];
            let scale = 1.0 / (fan_in as f32).sqrt();
            let data = (0..numel).map(|_| rng.next_signed_unit() * scale).collect();
            Tensor::from_vec(data, &shape)
        } else {
            Tensor::zeros(&shape)
        };
    }
}

/// Fisher-Yates with the seeded PRNG, permuting rows of a flat
/// `[n, dim]` feature buffer and its targets together.
fn deterministic_shuffle(
    features: &[f32],
    targets: &[f32],
    dim: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let n = targets.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Xorshift::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }

    let mut shuffled_features = Vec::with_capacity(n * dim);
    let mut shuffled_targets = Vec::with_capacity(n);
    for &row in &order {
        shuffled_features.extend_from_slice(&features[row * dim..(row + 1) * dim]);
        shuffled_targets.push(targets[row]);
    }
    (shuffled_features, shuffled_targets)
}

/// A deterministic, reproducible tensor of the given shape, for gradient
/// checks and view benchmarks.
fn deterministic_tensor(shape: &[usize], seed: u64) -> Tensor {
    let mut rng = Xorshift::new(seed);
    let numel: usize = shape.iter().product();
    Tensor::from_vec((0..numel).map(|_| rng.next_signed_unit()).collect(), shape)
}

// =====================================================================
// Printing helpers.
// =====================================================================

fn section(title: &str) {
    println!("\n-----------------------------------------------------------------------");
    println!(" {title}");
    println!("-----------------------------------------------------------------------");
}

fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos < 10_000 {
        format!("{nanos} ns")
    } else if nanos < 10_000_000 {
        format!("{:.1} us", nanos as f64 / 1_000.0)
    } else {
        format!("{:.1} ms", nanos as f64 / 1_000_000.0)
    }
}
