# ParallelVec

[![crates.io](https://img.shields.io/crates/v/parallel-vec.svg)](https://crates.io/crates/parallel-vec)
[![Documentation](https://docs.rs/parallel-vec/badge.svg)](https://docs.rs/parallel-vec)
![License](https://img.shields.io/crates/l/parallel-vec.svg)
[![codecov](https://codecov.io/gh/HouraiTeahouse/parallel_vec/branch/main/graph/badge.svg?token=BEKDFDO5HW)](https://codecov.io/gh/HouraiTeahouse/parallel_vec)

`ParallelVec` is a generic collection of contiguously stored heterogenous values with
an API similar to that of a `Vec<(T1, T2, ...)>` but stores the data laid out as a 
separate slice per field, using a [structures of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays)
layout. The advantage of this layout is that cache utilization may be signifgantly improved 
when iterating over the data.

This approach is common to game engines, and Entity-Component-Systems in particular but is
applicable anywhere that cache coherency and memory bandwidth are important for performance.

Unlike a struct of `Vec`s, only one length and capacity field is stored, and only one contiguous
allocation is made for the entire data structs. Upon reallocation, a struct of `Vec` may apply
additional allocation pressure. `ParallelVec` only allocates once per resize.

## Example
```rust
use parallel_vec::ParallelVec;

// #Some 'entity' data.
struct Position { x: f64, y: f64 }
struct Velocity { dx: f64, dy: f64 }
struct ColdData { /* Potentially many fields omitted here */ }

// Create a vec of entities
let mut entities: ParallelVec<(Position, Velocity, ColdData)> = ParallelVec::new();
entities.push((Position {x: 1.0, y: 2.0}, Velocity { dx: 0.0, dy: 0.5 }, ColdData {}));
entities.push((Position {x: 0.0, y: 2.0}, Velocity { dx: 0.5, dy: 0.5 }, ColdData {}));

// Update entities. This loop only loads position and velocity data, while skipping over
// the ColdData which is not necessary for the physics simulation.
for (position, velocity, _) in entities.iter_mut() {
    *position = *position + *velocity;
}

// Remove an entity
entities.swap_remove(0);
```

## Nightly
This crate requires use of GATs and therefore requires the following nightly features:
 * `generic_associated_types`

## `no_std` Support 
By default, this crate requires the standard library. Disabling the default features 
enables this crate to compile in `#![no_std]` environments. There must be a set global
allocator and heap support for this crate to work.

## `serde` Support 
`ParallelVec` can be serialized if it's parameters can be serialized. This is disabled by
default. Use the `serde` feature to enable support for serialization and deserialization.