use ofposets::polyvoxel::*;
use ofposets::{Renderer, to_dot};
use std::{fs, io};

fn main() -> io::Result<()> {
    let p = point();
    let e = cylinder(&p, &p);
    let t = cylinder(&e, &p);
    let prism = cylinder(&t, &t);
    let t2 = shift(&t);
    let u1 = cylinder(&prism, &t2);

    let square = cylinder(&e, &e);
    let e2 = shift(&e);
    let u2 = cylinder(&square, &e2);

    let (_, u) = paste(&u1, &u2, 2);

    let v1 = cylinder(&t2, &t2);
    let v2 = cylinder(&e2, &e2);

    let (v_pasting, v) = paste(&v1, &v2, 2);
    let v2_into_v = v_pasting.inr;

    let (w_pasting, w) = paste(&u, &v, 0);
    let v_into_w = w_pasting.inr;

    let witness = v2.greatest_element().expect("v2 must be a voxel");
    let witness = v2_into_v.apply(witness);
    let witness = v_into_w.apply(witness);
    let witness_frame = w.frame_of_element(witness);
    assert!(w.maximal(witness.dim).contains(&witness.pos));
    assert!(witness_frame.len() + 1 < w.total_frame().len());

    println!("{}", serde_json::to_string(w.as_ref())?);
    println!("{:?}", witness_frame);
    println!("{:?}", w.total_frame());
    // fs::write("visualizations/double_degen_example.dot", to_dot(&w, Renderer::CompassSpring))?;

    Ok(())
}
