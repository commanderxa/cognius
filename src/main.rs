use minigrad::tensor::Tensor;

fn main() {
    // let tensor = Tensor::from_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    // let tensor2 = Tensor::from_f64(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);
    // let tensor2 = Tensor::ones(vec![3, 2]);
    // let res = Tensor::mm(tensor, tensor2);
    // println!("{}\n", tensor2);
    // let res = tensor.mm(tensor2.t());
    let res = Tensor::arange(1., 6., 1.);
    println!("{}", res);
}
