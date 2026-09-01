use ofposets::{Renderer, to_dot};
use ofposets::polyvoxel::*;
use std::{fs, io};

fn main() -> io::Result<()> {
    let p = point();
    let e = cylinder(&p, &p);
    let t = cylinder(&e, &p);
    let prism = cylinder(&t, &t);

    let double_prism = paste(&prism, &prism, 2).1;
    let u = paste(&double_prism, &t, 1).1;
    let v = paste(&u, &double_prism,0).1;

    println!("{}", serde_json::to_string(v.as_ref())?);
    fs::write("visualizations/non_uniform_lydir_example.dot", to_dot(&v, Renderer::CompassSpring))?;

    Ok(())
}
