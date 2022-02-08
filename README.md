# ParallelVec

`ParallelVec` is a generic contiguously stored collection of heterogenous values.
It utilizes a [structure of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays)
memory layout, which may increase the cache utilization for uses cases like games. It exposes an 
API similar to a `Vec` of tuples.

Unlike a struct of `Vec`s, only one length and capacity field is stored, and only one contiguous
allocation is made for the entire data structs. Upon reallocation, a struct of `Vec` may apply
additional allocation strain. `ParallelVec` only allocates once per resize.

This repo is a general attempt at reimplementing https://github.com/That3Percent/soa-vec in a
generic manner. Unlike `SoaN` in that crate, this implementation attempts to make a singular 
generic `ParallelVec` implementation. This crate also use stablized allocator APIs instead of 
the unstable ones.

Currently this project will not compile without [GAT](https://github.com/rust-lang/rfcs/blob/master/text/1598-generic_associated_types.md)
(Generic Associated Types) enabled on Nightly.