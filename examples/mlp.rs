use minigrad::{Value, MLP, nn::Module};

fn main() {
    let mlp = MLP::new(4, vec![10, 20, 10, 1]);
    let data = Value::from_vec(vec![0.5, 1.0, 1.5, 2.0]);
    let target = Value::from(2.0);

    let epochs = 1000;
    let model_size = mlp.parameters().len();
    println!("Model is of size {model_size} parameters");
    println!("Training on {epochs} epochs...");

    // training loop
    for _epoch in 1..epochs + 1 {
        // inference
        let out = mlp.call(data.clone());
        let out = out.first();
        let out = out.unwrap();

        // compute the loss (MSE Loss)
        let loss = (out.clone() - target.clone()).pow(2);

        // compute the grads
        mlp.zero_grad();
        loss.backward();

        // optimization step
        for p in mlp.parameters() {
            p.add_data(-0.2 * p.get_grad());
        }

        // println!("{epoch}/{epochs} | Loss: {0}", loss.get_data());
    }

    let out = mlp.call(data);
    let out = out.first();
    let out = out.unwrap();
    let loss = (out.clone() - target.clone()).pow(2);
    out.backward();

    println!(
        "\n### Results ###\n\nOut: {}\nTarget: {}\nLoss: {}",
        out.get_data(),
        target.get_data(),
        loss.get_data()
    );
}
