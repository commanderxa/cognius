# Contributing

## Project

This is a deep learning library that is created for `rust` and `python`.

## Setup

Requires `Rust` and `cargo`.

## Project Structure

The following project strucutre is described to introduce the project to new contributors:

- `/` - _*(project's root)*_
  - `examples` - _*(examples of usage of the library)*_
  - `src` - _*(source code)*_
    - `data` - _*(functions and operations that work with data)*_
    - `nn` - _*(neural network module)*_
  - `tests` - _*(tests of the library)*_
  - `Cargo.toml` - _*(cargo file)*_

## Test

If you wish to contribute to this project then make sure that all the tests are passed using:

```sh
cargo test
```

Don't forget to write unit tests for new features, is something breakes in the future the tests will show.

## Examples

See `examples` directory.

This directory is describing the functionality of library in practice. If you created a new feature, then you should also create an example for this new feature.

To run an example:

```sh
cargo run --example <example name>
```

For example:

```sh
cargo run --example simple
```

## Secutiry

If it happened that you've found any security vulnerability, then please, refer to the `SECURITY.md` to share that information.
