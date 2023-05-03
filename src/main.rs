use minigrad::{Value, MLP, nn::Module};

fn main() {
    // let t1 = Value::from(1.0);
    // let t2 = Value::from(0.2355);
    // let pr1 = t1.clone() * t2.clone();

    // let t4 = Value::from(0.0);
    // let pr2 = t4.clone() + pr1.clone();

    // let t6 = Value::from(0.06655);
    // let t7 = Value::from(-2.0);
    // let pr3 = t6.clone() * t7.clone();

    // let pr4 = pr2.clone() + pr3.clone();
    // let pr5 = pr4.clone().relu();
    // pr5.backward();
    // println!("\n{}\n", pr4);
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", t1.get_data(), t1.get_grad());
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", t2.get_data(), t2.get_grad());
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", pr1.get_data(), pr1.get_grad());
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", t4.get_data(), t4.get_grad());
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", pr2.get_data(), pr2.get_grad());
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", t6.get_data(), t6.get_grad());
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", t7.get_data(), t7.get_grad());
    // println!("Data: {} \t\t\t\t\t\t| Grad: {}", pr3.get_data(), pr3.get_grad());
    // println!("Data: {:.4} \t\t\t\t\t\t| Grad: {}", pr4.get_data(), pr4.get_grad());
    // println!("Data: {:.4} \t\t\t\t\t\t| Grad: {}", pr5.get_data(), pr5.get_grad());

    let mlp = MLP::new(5, vec![10, 20, 10, 1]);
    let data = Value::from_vec(vec![0.5, 1.0, 1.5, 2.0]);
    let target = Value::from(2.0);

    let epochs = 10000;
    for epoch in 1..epochs + 1 {
        let out = mlp.call(data.clone());
        let out = out.first();
        let out = out.unwrap();
        let loss = (out.clone() - target.clone()).pow(2);

        mlp.zero_grad();
        loss.backward();

        for p in mlp.parameters() {
            p.add_data(-0.2 * p.get_grad());
        }

        println!("{epoch}/{epochs} | Loss: {0}", loss.get_data());
    }

    let out = mlp.call(data);
    let out = out.first();
    let out = out.unwrap();
    let loss = (out.clone() - target.clone()).pow(2);
    out.backward();

    println!(
        "Out: {} | Target: {} | Loss: {}",
        out.get_data(),
        target.get_data(),
        loss.get_data()
    );
}
