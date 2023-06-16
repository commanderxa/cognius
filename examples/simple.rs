use minigrad::Value;

fn main() {
    let t1 = Value::from(1.0);
    let t2 = Value::from(0.2355);
    let pr1 = t1.clone() * t2.clone();

    let t4 = Value::from(0.0);
    let pr2 = t4.clone() + pr1.clone();

    let t6 = Value::from(0.06655);
    let t7 = Value::from(-2.0);
    let pr3 = t6.clone() * t7.clone();

    let pr4 = pr2.clone() + pr3.clone();
    let pr5 = pr4.clone().relu();
    pr5.backward();

    println!("\n=== A demo of `Value` interactions and gradient computation ===");
    println!("Notes:\n\t* T ot t stands for tensor.\n\t*PR or pr stands for product of tensors.\n");

    println!("\n### The Value represents ###\n{}\n", pr4);

    println!("=== Value decomposition ===");
    println!(
        "T1\t= Data: {} \t| Grad: {}",
        t1.item(),
        t1.get_grad()
    );
    println!(
        "T2\t= Data: {} \t| Grad: {}",
        t2.item(),
        t2.get_grad()
    );
    println!(
        "PR1\t= Data: {} \t| Grad: {}",
        pr1.item(),
        pr1.get_grad()
    );
    println!(
        "T4\t= Data: {} \t| Grad: {}",
        t4.item(),
        t4.get_grad()
    );
    println!(
        "PR2\t= Data: {} \t| Grad: {}",
        pr2.item(),
        pr2.get_grad()
    );
    println!(
        "T6\t= Data: {:.4} \t| Grad: {}",
        t6.item(),
        t6.get_grad()
    );
    println!(
        "T7\t= Data: {} \t| Grad: {}",
        t7.item(),
        t7.get_grad()
    );
    println!(
        "PR3\t= Data: {:.3} \t| Grad: {}",
        pr3.item(),
        pr3.get_grad()
    );
    println!(
        "PR4\t= Data: {:.4} \t| Grad: {}",
        pr4.item(),
        pr4.get_grad()
    );
    println!(
        "PR5\t= Data: {:.4} \t| Grad: {}",
        pr5.item(),
        pr5.get_grad()
    );
}
