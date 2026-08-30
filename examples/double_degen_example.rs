use std::{fs, io};
use std::sync::Arc;
use ofposets::pushout::paste_along_boundary;
use ofposets::*;

#[allow(unused_variables)]
fn main() -> io::Result<()> {
    let p = Arc::new(FramedPoset::point());
    let e = elementary_cylinder(&p,&p);
    let t = elementary_cylinder(&e, &p);
    let prism = elementary_cylinder(&t, &t);
    let t2 = Arc::new(shift(&t));
    let u1 = elementary_cylinder(&prism, &t2);
    
    let square = elementary_cylinder(&e, &e);
    let e2 = Arc::new(shift(&e));
    let u2 = elementary_cylinder(&square, &e2);

    let u = paste_along_boundary(&u1, &u2, 2).tip;

    let v1 = elementary_cylinder(&t2, &t2);
    let v2 = elementary_cylinder(&e2, &e2);

    let v = paste_along_boundary(&v1, &v2, 2).tip;

    let w = paste_along_boundary(&u, &v, 0).tip;

    // println!("{}",serde_json::to_string(w.as_ref())?);
    // println!("{:?}", v1.active_directions());
    // println!("{:?}", w.active_directions());
    // fs::write("visualizations/double_degen_example.dot", to_dot(&w, Renderer::CompassSpring))?;


    let ee = paste_along_boundary(&e, &e, 0).tip;
    let huh = elementary_cylinder(&ee, &ee);

    Ok(())
}
