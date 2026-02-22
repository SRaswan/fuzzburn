/// Burn autograd example
/// cargo run --example simple_autograd

use burn::backend::{Autodiff, LibTorch, NdArray};
use burn::backend::libtorch::LibTorchDevice;
use burn::tensor::{backend::AutodiffBackend, Tensor};

fn run<B: AutodiffBackend<FloatElem = f32>>(device: &B::Device, label: &str) {
    let x_0: Tensor<B, 2> = Tensor::full([4140, 1], 0.1234_f32, device).require_grad();

    let t0 = x_0.clone();
    let t1 = t0.log();
    let grads = t1.backward();

    let x_grad = x_0.grad(&grads).unwrap();
    // d/dx log(x) = 1/x, so we expect x_grad to be full of 2.0
    println!("[{label}] x_0.grad = {}", x_grad.into_data());
}

fn main() {
    run::<Autodiff<NdArray>>(
        &<NdArray as burn::tensor::backend::Backend>::Device::default(),
        "NdArray",
    );

    run::<Autodiff<LibTorch>>(
        &LibTorchDevice::Cpu,
        "LibTorch",
    );
}

