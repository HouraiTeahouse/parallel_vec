#![allow(non_snake_case)]
#![deny(missing_docs)]
#![feature(generic_associated_types)]
#![no_std]

//! [`ParallelVec`] is a generic collection of contiguously stored heterogenous values with
//!  an API similar to that of a `Vec<(T1, T2, ...)>` but stores the data laid out as a
//! separate slice per field, using a [structures of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays)
//! layout. The advantage of this layout is that cache utilization may be signifgantly improved
//! when iterating over the data.
//!
//! This approach is common to game engines, and Entity-Component-Systems in particular but is
//! applicable anywhere that cache coherency and memory bandwidth are important for performance.
//!
//! Unlike a struct of `Vec`s, only one length and capacity field is stored, and only one contiguous
//! allocation is made for the entire data structs. Upon reallocation, a struct of `Vec` may apply
//! additional allocation pressure. `ParallelVec` only allocates once per resize.
//!
//! ## Example
//! ```rust
//! use parallel_vec::ParallelVec;
//!
//! /// Some 'entity' data.
//! # #[derive(Copy, Clone)]
//! struct Position { x: f64, y: f64 }
//! # #[derive(Copy, Clone)]
//! struct Velocity { dx: f64, dy: f64 }
//! struct ColdData { /* Potentially many fields omitted here */ }
//!
//! # use std::ops::Add;
//! # impl Add<Velocity> for Position { type Output=Self; fn add(self, other: Velocity) -> Self { Self { x: self.x + other.dx, y: self.y + other.dy } } }
//! // Create a vec of entities
//! let mut entities: ParallelVec<(Position, Velocity, ColdData)> = ParallelVec::new();
//! entities.push((Position {x: 1.0, y: 2.0}, Velocity { dx: 0.0, dy: 0.5 }, ColdData {}));
//! entities.push((Position {x: 0.0, y: 2.0}, Velocity { dx: 0.5, dy: 0.5 }, ColdData {}));
//!
//! // Update entities. This loop only loads position and velocity data, while skipping over
//! // the ColdData which is not necessary for the physics simulation.
//! for (position, velocity, _) in entities.iter_mut() {
//!     *position = *position + *velocity;
//! }
//!
//! // Remove an entity
//! entities.swap_remove(0);
//! ```
//!
//! ## Nightly
//! This crate requires use of GATs and therefore requires the following nightly features:
//! * `generic_associated_types`
//!
//! ## `no_std` Support
//! By default, this crate requires the standard library. Disabling the default features
//! enables this crate to compile in `#![no_std]` environments. There must be a set global
//! allocator and heap support for this crate to work.

extern crate alloc;

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;

/// A collection of iterators types for [`ParallelVec`].
pub mod iter;
/// Implementations for [`ParallelVecParam`].
pub mod param;
mod slice;
mod vec;

pub use param::ParallelVecParam;
pub use slice::{ParallelSlice, ParallelSliceMut};
pub use vec::ParallelVec;

/// Error when attempting to convert types to [`ParallelVec`].
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ParallelVecConversionError {
    /// The provided inputs were not the same length.
    UnevenLengths,
}
