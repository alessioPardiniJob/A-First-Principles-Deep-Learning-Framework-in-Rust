use std::fs;
use std::path::Path;

pub const FEATURES: usize = 8;

pub struct Dataset {
    /// Flat, row-major `[n, FEATURES]`, standardized (zero mean, unit
    /// variance) per feature.
    pub features: Vec<f32>,
    /// Standardized target values.
    pub targets: Vec<f32>,
    /// Mean/std of the *un*-standardized target, to convert a prediction
    /// or loss back to real units for reporting.
    pub target_mean: f32,
    pub target_std: f32,
}

/// Standardizes each feature column and the target independently: plain
/// SGD on raw scales (rooms in the thousands, income in single digits,
/// prices in the hundreds of thousands) is badly conditioned — this was
/// observed directly with the earlier synthetic dataset, where an
/// unnormalized learning rate blew up the weights in a single step.
fn standardize(raw_features: &[[f32; FEATURES]], raw_targets: &[f32]) -> Dataset {
    let n = raw_targets.len() as f32;

    let mut feature_mean = [0f32; FEATURES];
    let mut feature_std = [0f32; FEATURES];
    for j in 0..FEATURES {
        let mean = raw_features.iter().map(|r| r[j]).sum::<f32>() / n;
        let var = raw_features.iter().map(|r| (r[j] - mean).powi(2)).sum::<f32>() / n;
        feature_mean[j] = mean;
        feature_std[j] = var.sqrt().max(1e-6);
    }
    let target_mean = raw_targets.iter().sum::<f32>() / n;
    let target_var = raw_targets.iter().map(|&y| (y - target_mean).powi(2)).sum::<f32>() / n;
    let target_std = target_var.sqrt().max(1e-6);

    let mut features = Vec::with_capacity(raw_features.len() * FEATURES);
    for row in raw_features {
        for j in 0..FEATURES {
            features.push((row[j] - feature_mean[j]) / feature_std[j]);
        }
    }
    let targets = raw_targets.iter().map(|&y| (y - target_mean) / target_std).collect();

    Dataset {
        features,
        targets,
        target_mean,
        target_std,
    }
}

/// Parses the California Housing CSV (columns: longitude, latitude,
/// housing_median_age, total_rooms, total_bedrooms, population,
/// households, median_income, median_house_value, ocean_proximity).
/// `ocean_proximity` (categorical) is dropped, keeping the same 8 numeric
/// input features as the mock fallback; rows with a missing
/// `total_bedrooms` (this dataset's known source of missing values) are
/// skipped. `None` if the file is not present.
fn load_real(limit: usize) -> Option<Dataset> {
    let path = Path::new("data/housing/housing.csv");
    if !path.exists() {
        return None;
    }
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    let mut lines = text.lines();
    lines.next(); // header

    let mut raw_features = Vec::new();
    let mut raw_targets = Vec::new();
    for line in lines {
        if raw_targets.len() >= limit {
            break;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 9 {
            continue;
        }
        let Some(values) = cols[..9]
            .iter()
            .map(|c| c.parse::<f32>().ok())
            .collect::<Option<Vec<f32>>>()
        else {
            continue; // missing/malformed numeric field (e.g. total_bedrooms)
        };
        let mut row = [0f32; FEATURES];
        row.copy_from_slice(&values[..FEATURES]);
        raw_features.push(row);
        raw_targets.push(values[8]); // median_house_value
    }
    assert!(!raw_targets.is_empty(), "no valid rows parsed from {path:?}");

    Some(standardize(&raw_features, &raw_targets))
}

fn mock_example(idx: usize) -> ([f32; FEATURES], f32) {
    let t = idx as f32;
    let med_inc = 3.0 + 1.5 * (0.13 * t).sin();
    let house_age = 2.0 + 1.0 * (0.07 * t).cos();
    let ave_rooms = 5.0 + 1.0 * (0.05 * t).sin();
    let ave_bedrms = 1.0 + 0.2 * (0.11 * t).cos();
    let population = 3.0 + 1.0 * (0.03 * t).sin();
    let ave_occup = 3.0 + 0.5 * (0.09 * t).cos();
    let latitude = 0.5 * (0.02 * t).sin();
    let longitude = 0.5 * (0.02 * t).cos();

    let features = [
        med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude,
    ];
    let target =
        2.0 * med_inc - 0.3 * house_age + 0.4 * ave_rooms - 0.5 * ave_occup + latitude - 0.5 * longitude + 1.0;
    (features, target)
}

/// Deterministic, formula-generated fallback with the same shape and
/// target representation as the real dataset (no network/files needed).
fn mock(n: usize) -> Dataset {
    let mut raw_features = Vec::with_capacity(n);
    let mut raw_targets = Vec::with_capacity(n);
    for idx in 0..n {
        let (f, y) = mock_example(idx);
        raw_features.push(f);
        raw_targets.push(y);
    }
    standardize(&raw_features, &raw_targets)
}

/// Loads the real California Housing CSV if `data/housing/housing.csv`
/// is present, otherwise falls back to the deterministic mock.
pub fn load(limit: usize) -> Dataset {
    if let Some(dataset) = load_real(limit) {
        println!(
            "Housing: loaded {} real rows from data/housing/housing.csv",
            dataset.targets.len()
        );
        dataset
    } else {
        println!("Housing: data/housing/housing.csv not found, falling back to deterministic mock data");
        mock(limit)
    }
}
