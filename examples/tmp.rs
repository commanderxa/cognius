use cognius::{randn, Tensor};

fn main() {
    let a = randn!(2, 4, 5);
    let b = randn!(5);
    let c = Tensor::mm(a.clone(), b.clone());
    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
}
