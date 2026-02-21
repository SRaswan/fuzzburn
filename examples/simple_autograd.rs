/// Minimal Burn autograd example: y = x^2, compute dy/dx at x=2.0
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;
type Device = <NdArray as burn::tensor::backend::Backend>::Device;

fn main() {
    let device = Device::default();

    let x: Tensor<Backend, 1> = Tensor::from_floats([2.0_f32], &device).require_grad();
    let y = x.clone() * x.clone();

    let grads = y.backward();
    let x_grad = x.grad(&grads).unwrap();

    println!("y = x^2,  x = 2.0");
    println!("dy/dx = {}", x_grad.into_data()); // expected: 4.0
}
