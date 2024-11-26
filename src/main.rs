use std::io::{BufReader, Cursor};

use closest_string_assignment::*;

fn main() {
    let mut instance = parse_input(BufReader::new(
        std::fs::File::open("string1.in").unwrap(),
        // Cursor::new("4\nada\nabb\nacc\nbad".to_string()),
    ));
    instance.shorten();
    instance.dedup();
    // dbg!(&instance);
    // dbg!(instance.strings.len());
    // dbg!(instance.find_max_pairwise_d());
    // dbg!(instance.distance_to(&[1, 2, 1, 3, 2, 1, 2, 2, 0, 0, 0, 0, 2, 2, 0, 1, 0, 3, 1, 1,]));
    // dbg!(&instance);
    // let (instance, bools) = instance.restrict_instance_to_d_core(instance.find_max_pairwise_d());
    // dbg!(instance.strings.len());
    // // dbg!(instance.restrict_instance_to_d_core(50).0.strings.len());
    // dbg!(instance.find_max_pairwise_d());
    instance.populate_tensor();
    instance.solve();
    // dbg!(new_instance.restrict_instance_to_d_core(4).0.strings.len());
    // println!("Hello, world!\n{:?}", instance);
}
