use ap_project::loss::{MSELoss, SoftmaxCrossEntropy};
use ap_project::module::{Linear, ReLU, Sequential};
use ap_project::optimizer::Sgd;
use ap_project::{Loss, Module, Optimizer, Tensor};

/// Two linearly separable clusters in 2D. A small MLP trained with
/// `SoftmaxCrossEntropy` + `Sgd` should drive the loss down sharply and
/// classify (almost) every training example correctly.
#[test]
fn classification_task_converges() {
    #[rustfmt::skip]
    let inputs = Tensor::from_vec(
        vec![
            -1.0, -1.0, -1.2, -0.8, -0.9, -1.1, -1.1, -0.9,
             1.0,  1.0,  1.2,  0.8,  0.9,  1.1,  1.1,  0.9,
        ],
        &[8, 2],
    );
    let labels = [0usize, 0, 0, 0, 1, 1, 1, 1];

    let mut model = Sequential::new();
    model.add(Box::new(Linear::new(2, 16)));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(16, 2)));

    let loss_fn = SoftmaxCrossEntropy;
    let mut optimizer = Sgd { lr: 0.3 };

    let mut first_loss = None;
    let mut last_loss = 0.0;
    for _ in 0..400 {
        model.zero_grad();
        let logits = model.forward(inputs.clone());
        let (loss, grad) = loss_fn.forward(&logits, &labels);
        model.backward(grad);
        optimizer.step(&mut model);
        first_loss.get_or_insert(loss);
        last_loss = loss;
    }
    let first_loss = first_loss.unwrap();

    assert!(
        last_loss < first_loss * 0.3,
        "loss should drop sharply: {first_loss} -> {last_loss}"
    );

    let logits = model.forward(inputs);
    let correct = labels
        .iter()
        .enumerate()
        .filter(|&(i, &label)| {
            let predicted = if logits.get(&[i, 1]) > logits.get(&[i, 0]) { 1 } else { 0 };
            predicted == label
        })
        .count();
    assert!(correct >= 7, "expected at least 7/8 correct, got {correct}/8");
}

/// Fits `y = 2x + 1` with an MLP trained via `MSELoss` + `Sgd`; the loss
/// should drop sharply from its initial (near-random) value.
#[test]
fn regression_task_converges() {
    let xs: Vec<f32> = (0..10).map(|i| i as f32 * 0.3 - 1.5).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
    let n = xs.len();
    let inputs = Tensor::from_vec(xs, &[n, 1]);
    let targets = Tensor::from_vec(ys, &[n, 1]);

    let mut model = Sequential::new();
    model.add(Box::new(Linear::new(1, 16)));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(16, 1)));

    let loss_fn = MSELoss;
    let mut optimizer = Sgd { lr: 0.05 };

    let mut first_loss = None;
    let mut last_loss = 0.0;
    for _ in 0..500 {
        model.zero_grad();
        let pred = model.forward(inputs.clone());
        let (loss, grad) = loss_fn.forward(&pred, &targets);
        model.backward(grad);
        optimizer.step(&mut model);
        first_loss.get_or_insert(loss);
        last_loss = loss;
    }
    let first_loss = first_loss.unwrap();

    assert!(
        last_loss < first_loss * 0.1,
        "loss should drop sharply: {first_loss} -> {last_loss}"
    );
}

/// A tiny CNN classifier: Conv2d -> ReLU -> Flatten -> Linear, exercising
/// the layer that bridges [B,C,H,W] to [B,F].
#[test]
fn conv_classifier_shapes_compose_end_to_end() {
    let mut model = Sequential::new();
    model.add(Box::new(ap_project::module::Conv2d::new(1, 2, (3, 3), 1)));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(ap_project::module::Flatten::new()));
    model.add(Box::new(Linear::new(2 * 6 * 6, 2)));

    let input = Tensor::zeros(&[4, 1, 8, 8]);
    let logits = model.forward(input);
    assert_eq!(logits.shape(), &[4, 2]);

    let labels = [0usize, 1, 0, 1];
    let (_, grad) = SoftmaxCrossEntropy.forward(&logits, &labels);
    let grad_input = model.backward(grad);
    assert_eq!(grad_input.shape(), &[4, 1, 8, 8]);
}
