use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use ofposets::enumeration::{
    PolyvoxelEnumerationPhase, PolyvoxelFactorization, enumerate_polyvoxels_with_progress,
};
use ofposets::{FramedPoset, Renderer, to_dot};
use serde::Serialize;

const MAX_CELLS: usize = 27;
const ALLOWED_DIRECTIONS: &[usize] = &[0, 1, 2];
const OUTPUT_DIRECTORY: &str = "visualizations/polyvoxels_up_to_27_cells_directions_0_1_2";

#[derive(Serialize)]
struct CatalogRecord<'a> {
    version: usize,
    id: usize,
    cells: usize,
    active_directions: Vec<usize>,
    is_voxel: bool,
    factorizations: &'a [PolyvoxelFactorization],
    ofp: &'a FramedPoset,
}

fn main() -> io::Result<()> {
    let output_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join(OUTPUT_DIRECTORY);
    fs::create_dir_all(&output_directory)?;

    println!(
        "enumerating polyvoxels with at most {MAX_CELLS} cells and active directions contained in {ALLOWED_DIRECTIONS:?}"
    );
    let started = Instant::now();
    let catalog = enumerate_polyvoxels_with_progress(MAX_CELLS, ALLOWED_DIRECTIONS, |progress| {
        if progress.phase == PolyvoxelEnumerationPhase::Complete {
            println!(
                "fixed point complete after {} rounds: {} representatives, {} factorizations ({:.1?})",
                progress.round,
                progress.representatives,
                progress.factorizations,
                started.elapsed(),
            );
        } else {
            println!(
                "round {} {:?}: {}/{} jobs; {} representatives, {} factorizations ({:.1?})",
                progress.round,
                progress.phase,
                progress.completed_jobs,
                progress.total_jobs,
                progress.representatives,
                progress.factorizations,
                started.elapsed(),
            );
        }
    });
    let mut jsonl = BufWriter::new(File::create(output_directory.join("catalog.jsonl"))?);
    let mut summary = BufWriter::new(File::create(output_directory.join("summary.tsv"))?);
    writeln!(
        summary,
        "id\tcells\tactive_directions\tis_voxel\tfactorizations"
    )?;

    let mut by_cell_count = BTreeMap::<usize, usize>::new();
    let mut total_factorizations = 0usize;
    for (id, entry) in catalog.entries().iter().enumerate() {
        let cells = entry.shape.sizes().iter().sum();
        let active_directions = entry.shape.active_directions();
        *by_cell_count.entry(cells).or_default() += 1;
        total_factorizations += entry.factorizations.len();

        serde_json::to_writer(
            &mut jsonl,
            &CatalogRecord {
                version: 1,
                id,
                cells,
                active_directions: active_directions.clone(),
                is_voxel: entry.is_voxel,
                factorizations: &entry.factorizations,
                ofp: &entry.shape,
            },
        )
        .map_err(io::Error::other)?;
        writeln!(jsonl)?;

        writeln!(
            summary,
            "{id}\t{cells}\t{}\t{}\t{}",
            serde_json::to_string(&active_directions).map_err(io::Error::other)?,
            entry.is_voxel,
            entry.factorizations.len(),
        )?;

        fs::write(
            output_directory.join(format!("polyvoxel_{id:03}.dot")),
            to_dot(&entry.shape, Renderer::CompassSpring),
        )?;
    }

    jsonl.flush()?;
    summary.flush()?;
    println!(
        "wrote {} polyvoxels with {total_factorizations} immediate factorizations to {}",
        catalog.len(),
        output_directory.display(),
    );
    for (cells, count) in by_cell_count {
        println!("  {cells} cells: {count}");
    }
    println!(
        "render the spring DOT files with scripts/render_visualizations.sh {}",
        output_directory.display(),
    );

    Ok(())
}
