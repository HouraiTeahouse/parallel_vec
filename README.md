# ParallelVec

`ParallelVec` is a generic collection of multiple types of contiguously stored elements.
It utilizes a [structure of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays)
memory layout, which may increase the cache utilization for uses cases like games.

It exposes an API similar to a `Vec` of tuples.

This repo is a general attempt at reimplementing https://github.com/That3Percent/soa-vec in a
generic manner. Unlike `SoaN` in that crate, this implementation attempts to make a generic 
`ParallelVec`. This crate also use stablized allocator APIs instead of the unstable ones.

Currently this project will not compile without [GAT](https://github.com/rust-lang/rfcs/blob/master/text/1598-generic_associated_types.md)
(Generic Associated Types) enabled on Nightly.