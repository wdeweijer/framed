use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, Read};
use std::time::{Duration, Instant};

use ofposets::{FramedPoset, normalize};

const DATASET_DIR: &str = "visualizations/random_9_cells_normal_forms_cubular";
const TARGET_JSON: &str = "visualizations/standard_2_cube_normalized.ofp.json";
const REPORT_INTERVAL: Duration = Duration::from_secs(5);

fn main() -> io::Result<()> {
    let square = normalize(&standard_2_cube());
    let target_json = serde_json::to_string_pretty(&square).map_err(io::Error::other)? + "\n";
    fs::write(TARGET_JSON, &target_json)?;

    let target = target_json.as_bytes();
    let mut contents = Vec::with_capacity(target.len());
    let mut checked = 0usize;
    let mut matching_size = 0usize;
    let mut last_report = Instant::now();

    println!(
        "searching {DATASET_DIR} for an exact match with {TARGET_JSON} ({} bytes)",
        target.len()
    );

    for entry in fs::read_dir(DATASET_DIR)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("json")) {
            continue;
        }

        checked += 1;
        if entry.metadata()?.len() == target.len() as u64 {
            matching_size += 1;
            contents.clear();
            File::open(&path)?.read_to_end(&mut contents)?;

            if contents == target {
                println!(
                    "found the standard 2-cube at {} after checking {checked} JSON files",
                    path.display()
                );
                return Ok(());
            }
        }

        if last_report.elapsed() >= REPORT_INTERVAL {
            println!("checked {checked} JSON files ({matching_size} had the target byte length)");
            last_report = Instant::now();
        }
    }

    println!(
        "no exact match found after checking {checked} JSON files ({matching_size} had the target byte length)"
    );
    Ok(())
}

fn standard_2_cube() -> FramedPoset {
    FramedPoset::from_faces(
        vec![
            vec![vec![], vec![], vec![], vec![]],
            vec![vec![0], vec![0], vec![1], vec![1]],
            vec![vec![0, 1]],
        ],
        vec![
            vec![vec![], vec![], vec![], vec![]],
            vec![vec![0], vec![2], vec![0], vec![1]],
            vec![vec![0, 2]],
        ],
        vec![
            vec![vec![], vec![], vec![], vec![]],
            vec![vec![1], vec![3], vec![2], vec![3]],
            vec![vec![1, 3]],
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_square_has_nine_cells_and_normalizes() {
        let square = standard_2_cube();
        let normalized = normalize(&square);

        assert_eq!(square.sizes(), vec![4, 4, 1]);
        assert_eq!(square.sizes().into_iter().sum::<usize>(), 9);
        assert!(normalized.is_normal());
    }
}
