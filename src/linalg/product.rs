use crate::{op::Op, tensor_data::TensorData, Tensor};

/// Matrix multiplication
///
/// Accepts:
/// * a: `Tensor`
/// * b: `Tensor`
///
/// The inner dimensions of the matrices must be the same.
pub fn matmul(a: Tensor, b: Tensor) -> Tensor {
    // shapes of the tensors
    let mut a_shape: Vec<usize> = a.shape();
    let b_shape: Vec<usize> = b.shape();
    // check wether the operation is able to be proceeded
    assert!(
        a_shape.len() > 1,
        "The shape of the tensor must not be scalar."
    );
    assert!(
        b_shape.len() > 1,
        "The shape of the tensor must not be scalar."
    );
    assert_eq!(
        a_shape.last().unwrap(),
        b_shape.first().unwrap(),
        "The shapes of the tensors must have the same inner dimension -> (M x N) @ (N x M), but you have tensors A: {:?} and B: {:?}", 
        format!("({a_shape:?})").replace('[', "").replace(']', ""), 
        format!("({b_shape:?})").replace('[', "").replace(']', ""), 
    );
    // get batch dimensions if they exist
    let mut batches: Vec<usize> = vec![];
    for i in 2..a_shape.len() {
        batches.push(a_shape[i - 2]);
    }
    // remove batch dimensions from the A tensor shape
    a_shape.drain(0..batches.len());
    let batch_prod = batches.iter().product::<usize>();
    // new tensor result
    let mut result = vec![0.0; batch_prod * a_shape[0] * (b_shape[b_shape.len() - 1])];
    // data of the tensors, the tensor b is transposed
    let a_data = a.item();
    let b_data = b.t().item();
    // literal notations for matrix dimensions for the conviniency and comprehension
    let m = a_shape[0];
    let n = a_shape[1];
    let p = b_shape[1];
    // iterate over the batch dimensions
    // `k` is a batch dimension
    for k in 0..batch_prod {
        // iterate over the result tensor, it zips the slices of the left and
        // right tensors then it multiplies the two zipped values and returns
        // the slice back, after it sums the vector to obtain the value
        for i in 0..m {
            for j in 0..p {
                let b_data = &b_data[(j * n)..(j * n + n)];
                result[k * m * p + i * p + j] = a_data
                    [(k * m * n + i * n)..(k * m * n + i * n + n)]
                    .iter()
                    .zip(b_data)
                    .map(|(&a, &b)| a * b)
                    .collect::<Vec<f64>>()
                    .iter()
                    .sum();
            }
        }
    }
    // add batch dimensions to the new shape
    let new_shape = batches
        .into_iter()
        .chain(vec![a_shape[0], b_shape[b_shape.len() - 1]])
        .collect();
    let inner = TensorData::from_op(result, new_shape, vec![a, b], Op::MatMul);
    Tensor::new(inner)
}

/// Cross Product
///
/// Accepts:
/// * a: `Tensor`
/// * b: `Tensor`
///
/// Performs the cross product of 3-dimensional vectors.
pub fn cross(a: Tensor, b: Tensor) -> Tensor {
    // check the dimensions
    assert_eq!(
        a.shape().len(),
        b.shape().len(),
        "Shape length of Tensor `b` does not much shape length of Tensor `a`"
    );
    assert_eq!(
        a.shape()[a.shape().len() - 1],
        3,
        "Last dimension of Tensor `a` does not equal 3."
    );
    assert_eq!(
        b.shape()[b.shape().len() - 1],
        3,
        "Last dimension of Tensor `b` does not equal 3."
    );
    // get data of the Tensors
    let _a = a.item();
    let _b = b.item();
    // get batch dimensions
    let mut a_batch_dims = a.shape();
    let _ = a_batch_dims.pop();
    let mut b_batch_dims = b.shape();
    let _ = b_batch_dims.pop();
    for i in 0..a_batch_dims.len() {
        assert!(
            a_batch_dims[i] == b_batch_dims[i] || (b_batch_dims[i] == 1 || a_batch_dims[i] == 1),
            "The size of tensor a ({:?}) must match the size of tensor b ({:?}) at a dimension {:?}", 
            format!("{}", a_batch_dims[i]), 
            format!("{}", b_batch_dims[i]), 
            format!("{}", i)
        );
    }
    // batch dimensions to expand
    let mut batch_expand = vec![0; a_batch_dims.len()];
    // get batches to expand
    for i in 0..a_batch_dims.len() {
        let diff = a_batch_dims[i] - b_batch_dims[i];
        if diff > 0 {
            batch_expand[i] = diff;
        }
    }
    println!("{:?}", batch_expand);
    for i in 0..batch_expand.len() {
        if batch_expand[i] > 0 {}
    }
    // check dimensional correspondence
    // result init
    let mut result = vec![0.0; a.length()];
    let mut i: usize = 0;
    let mut j: usize = 0;
    while i < (a.length() - 2) {
        result[i] = (_a[i + 1] * _b[j + 2]) - (_a[i + 2] * _b[j + 1]);
        result[i + 1] = (_a[i + 2] * _b[j]) - (_a[i] * _b[j + 2]);
        result[i + 2] = (_a[i] * _b[j + 1]) - (_a[i + 1] * _b[j]);
        i += 2;
        j += 2;
        if j >= (b.length() - 2) {
            j = 0;
        }
    }
    // computation
    // construct a Tensor
    let inner = TensorData::from_op(result, a.shape(), vec![a, b], Op::Cross);
    Tensor::new(inner)
}
