use std::fs;
use std::path::Path;

pub const IMAGE_SIZE: usize = 28;
pub const FEATURES: usize = IMAGE_SIZE * IMAGE_SIZE;
pub const CLASSES: usize = 10;

pub struct Split {
    /// Flat, row-major `[n, FEATURES]`, pixels normalized to `[0, 1]`.
    pub pixels: Vec<f32>,
    pub labels: Vec<usize>,
}

fn read_u32_be(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

/// Parses the IDX image format: a 16-byte header (magic, count, rows,
/// cols — all big-endian u32) followed by `count * rows * cols` raw
/// pixel bytes.
fn read_images(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    assert_eq!(read_u32_be(&bytes, 0), 0x0803, "unexpected IDX image magic in {path:?}");
    let count = read_u32_be(&bytes, 4) as usize;
    let rows = read_u32_be(&bytes, 8) as usize;
    let cols = read_u32_be(&bytes, 12) as usize;
    assert_eq!((rows, cols), (IMAGE_SIZE, IMAGE_SIZE), "unexpected image size in {path:?}");
    bytes[16..16 + count * rows * cols]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect()
}

/// Parses the IDX label format: an 8-byte header (magic, count) followed
/// by one label byte (0-9) per example.
fn read_labels(path: &Path) -> Vec<usize> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    assert_eq!(read_u32_be(&bytes, 0), 0x0801, "unexpected IDX label magic in {path:?}");
    let count = read_u32_be(&bytes, 4) as usize;
    bytes[8..8 + count].iter().map(|&b| b as usize).collect()
}

/// Loads the real, downloaded MNIST training split from `data/mnist/`
/// (see the download step), capped to `limit` examples to bound this
/// demo's runtime. `None` if the files are not present (e.g. a fresh
/// checkout — `data/` is gitignored, it holds a real downloaded dataset).
fn load_real(limit: usize) -> Option<Split> {
    let dir = Path::new("data/mnist");
    let image_path = dir.join("train-images-idx3-ubyte");
    let label_path = dir.join("train-labels-idx1-ubyte");
    if !image_path.exists() || !label_path.exists() {
        return None;
    }

    let pixels_all = read_images(&image_path);
    let labels_all = read_labels(&label_path);
    assert_eq!(pixels_all.len() / FEATURES, labels_all.len());

    let n = limit.min(labels_all.len());
    Some(Split {
        pixels: pixels_all[..n * FEATURES].to_vec(),
        labels: labels_all[..n].to_vec(),
    })
}

/// Ten class centers spread evenly around a 28x28 image; a Gaussian blob
/// near each, so classes stay separable by blob position alone.
fn mock_example(label: usize, idx: usize) -> Vec<f32> {
    let angle = label as f32 * std::f32::consts::TAU / CLASSES as f32;
    let (cx0, cy0) = (14.0 + 8.0 * angle.cos(), 14.0 + 8.0 * angle.sin());
    let cx = cx0 + (idx as f32 * 0.37).sin() * 2.0;
    let cy = cy0 + (idx as f32 * 0.53).cos() * 2.0;

    let mut pixels = vec![0.0f32; FEATURES];
    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            let dx = col as f32 - cx;
            let dy = row as f32 - cy;
            pixels[row * IMAGE_SIZE + col] = (-(dx * dx + dy * dy) / 30.0).exp();
        }
    }
    pixels
}

/// Deterministic, formula-generated fallback with the exact shape and
/// target representation the real split has (no network/files needed).
/// Generated in round-robin class order so unshuffled mini-batches still
/// see a balanced mix of classes.
fn mock(examples_per_class: usize) -> Split {
    let mut pixels = Vec::with_capacity(examples_per_class * CLASSES * FEATURES);
    let mut labels = Vec::with_capacity(examples_per_class * CLASSES);
    for idx in 0..examples_per_class {
        for label in 0..CLASSES {
            pixels.extend(mock_example(label, idx));
            labels.push(label);
        }
    }
    Split { pixels, labels }
}

/// Loads real MNIST if `data/mnist/` holds the downloaded IDX files,
/// otherwise falls back to the deterministic mock.
pub fn load(limit: usize) -> Split {
    if let Some(split) = load_real(limit) {
        println!("MNIST: loaded {} real examples from data/mnist/", split.labels.len());
        split
    } else {
        println!("MNIST: data/mnist/ not found, falling back to deterministic mock data");
        mock((limit / CLASSES).max(1))
    }
}
