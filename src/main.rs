fn main() {
    println!("Run example:                      cargo run --example simple_autograd");
    println!("Tensor Single Operation Fuzzing:         cargo +nightly fuzz run fuzz_single_op");
    println!("Tensor Operation Fuzzing:         cargo +nightly fuzz run fuzz_tensor_ops");
    println!("Autograd Fuzzing:         cargo +nightly fuzz run fuzz_autograd");

}
