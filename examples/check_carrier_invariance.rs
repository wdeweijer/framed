use std::sync::Arc;

use ofposets::{
    CubularityMode, FramedPoset, FramedPosetSubset, RandomFramedPosetGenerator, Sign, boundary,
    is_cubular,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn main() {
    check_generator(
        &RandomFramedPosetGenerator::new_without_full_frame(3, 9),
        20_000,
        0x0a11ce55,
    );
    check_generator(&RandomFramedPosetGenerator::new(2, 9), 20_000, 0xca441e42);
    check_generator(&RandomFramedPosetGenerator::new(3, 13), 5_000, 0x3dca441e);
}

fn check_generator(generator: &RandomFramedPosetGenerator, samples: u64, seed: u64) {
    let mut strong = 0;
    let mut connected_strong = 0;
    for sample in 1..=samples {
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(sample));
        let shape = Arc::new(generator.generate(&mut rng));
        if !is_cubular(CubularityMode::Strong, &shape) {
            continue;
        }
        strong += 1;
        if !shape.is_connected() {
            continue;
        }
        connected_strong += 1;
        if let Some((left, right)) = incomparable_carriers(&shape) {
            println!("INCOMPARABLE CARRIERS sample={sample}: {left:?} and {right:?}");
            println!("{}", serde_json::to_string_pretty(&*shape).unwrap());
            return;
        }
    }
    println!(
        "maximal frames were a chain in all {connected_strong}/{strong} connected/total strongly cubular samples from generator {:?}",
        generator
    );
}

fn incomparable_carriers(shape: &Arc<FramedPoset>) -> Option<(Vec<usize>, Vec<usize>)> {
    let carriers = carriers(shape);
    let carriers = carriers.iter().flatten().collect::<Vec<_>>();
    for (left_index, left) in carriers.iter().enumerate() {
        for right in &carriers[left_index + 1..] {
            if !is_subset(left, right) && !is_subset(right, left) {
                return Some(((*left).clone(), (*right).clone()));
            }
        }
    }
    None
}

fn is_subset(left: &[usize], right: &[usize]) -> bool {
    left.iter()
        .all(|element| right.binary_search(element).is_ok())
}

fn carriers(shape: &Arc<FramedPoset>) -> Vec<Vec<Vec<usize>>> {
    let directions = shape.total_frame();
    let boundaries = directions
        .iter()
        .map(|&direction| {
            [Sign::Input, Sign::Output].map(|sign| {
                let (_, embedding) = boundary(sign, direction, shape);
                FramedPosetSubset::from_embedding(&embedding)
            })
        })
        .collect::<Vec<_>>();
    shape
        .sizes()
        .into_iter()
        .enumerate()
        .map(|(dim, size)| {
            (0..size)
                .map(|pos| {
                    directions
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &direction)| {
                            (!boundaries[index][0].contains(dim, pos)
                                || !boundaries[index][1].contains(dim, pos))
                            .then_some(direction)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
