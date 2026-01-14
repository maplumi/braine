pub(super) fn softmax_temp(items: &[(String, f32)], temp: f32) -> Vec<f32> {
    if items.is_empty() {
        return Vec::new();
    }

    let t = temp.max(1.0e-6);

    let mut max_score = f32::NEG_INFINITY;
    for (_name, score) in items {
        if score.is_finite() {
            max_score = max_score.max(*score);
        }
    }
    if !max_score.is_finite() {
        max_score = 0.0;
    }

    let mut exps: Vec<f32> = Vec::with_capacity(items.len());
    let mut sum = 0.0f32;
    for (_name, score) in items {
        let s = if score.is_finite() { *score } else { 0.0 };
        let z = ((s - max_score) / t).clamp(-80.0, 80.0);
        let e = z.exp();
        sum += e;
        exps.push(e);
    }

    if !sum.is_finite() || sum <= 0.0 {
        let p = 1.0 / (items.len() as f32);
        return vec![p; items.len()];
    }

    exps.into_iter().map(|e| e / sum).collect()
}
