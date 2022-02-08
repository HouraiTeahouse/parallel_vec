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
//! 	*position = *position + *velocity;
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

pub use param::ParallelVecParam;

use core::marker::PhantomData;
use iter::*;

/// A contiguously growable heterogenous array type.
///
/// This type stores the values [structure of arrays] layout. This layout
/// may improve cache utilizatoin in specific use cases, which may have.
///
/// Unlike a struct of `Vec`s, this type allocates memory for the all individual
/// fields simultaneously. This may minimize memory fragmentation and decrease
/// allocation pressure. It also only stores one length and capacity instead
/// of duplicating the values across multiple `Vec` fields.
///
/// [structures of arrays]: https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays
pub struct ParallelVec<Param: ParallelVecParam> {
    len: usize,
    capacity: usize,
    storage: Param::Storage,
}

impl<Param: ParallelVecParam> ParallelVec<Param> {
    /// Constructs a new, empty `ParallelVec`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    pub fn new() -> Self {
        Self {
            len: 0,
            capacity: 0,
            storage: Param::dangling(),
        }
    }

    /// Constructs a new, empty [`ParallelVec`] with the specified capacity.  
    ///
    /// The vector will be able to hold exactly capacity elements without reallocating.
    /// If capacity is 0, the vector will not allocate.
    ///
    /// It is important to note that although the returned vector has the capacity specified,
    /// the vector will have a zero length.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            Self::new()
        } else {
            unsafe {
                Self {
                    len: 0,
                    capacity,
                    storage: Param::alloc(capacity),
                }
            }
        }
    }

    /// Returns the number of elements in the vector, also referred to as its ‘length’.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Returns a immutable reference to the element at `index`, if available, or
    /// [`None`] if it is out of bounds.
    ///
    /// [`None`]: Option::None
    #[inline]
    pub fn get<'a>(&'a self, index: usize) -> Option<Param::Ref<'a>> {
        if self.len <= index {
            None
        } else {
            unsafe { Some(self.get_unchecked(index)) }
        }
    }

    /// Returns a mutable reference to the element at `index`, if available, or
    /// [`None`] if it is out of bounds.
    ///
    /// [`None`]: Option::None
    #[inline]
    pub fn get_mut<'a>(&'a mut self, index: usize) -> Option<Param::RefMut<'a>> {
        if self.len <= index {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index)) }
        }
    }

    /// Returns the first element of the `ParallelVec`, or `None` if it is empty.
    #[inline(always)]
    pub fn first<'a>(&'a self) -> Option<Param::Ref<'a>> {
        self.get(0)
    }

    /// Returns the mutable pointer first element of the `ParallelVec`, or `None` if it is empty.
    #[inline(always)]
    pub fn first_mut<'a>(&'a mut self) -> Option<Param::RefMut<'a>> {
        self.get_mut(0)
    }

    /// Returns the last element of the `ParallelVec`, or `None` if it is empty.
    #[inline]
    pub fn last<'a>(&'a self) -> Option<Param::Ref<'a>> {
        if self.len == 0 {
            None
        } else {
            unsafe { Some(self.get_unchecked(self.len - 1)) }
        }
    }

    /// Returns the mutable pointer last element of the `ParallelVec`, or `None` if it is empty.
    #[inline]
    pub fn last_mut<'a>(&'a mut self) -> Option<Param::RefMut<'a>> {
        if self.len == 0 {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(self.len - 1)) }
        }
    }

    /// Gets a immutable reference to the elements at `index`.
    ///
    /// # Panics
    /// This function will panic if `index` is >= `self.len`.
    #[inline]
    pub fn index(&self, index: usize) -> Param::Ref<'_> {
        if self.len <= index {
            panic!("ParallelVec: Index out of bounds: {}", index);
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }

    /// Gets a mutable reference to the elements at `index`.
    ///
    /// # Panics
    /// This function will panic if `index` is >= `self.len`.
    #[inline]
    pub fn index_mut<'a>(&'a mut self, index: usize) -> Param::RefMut<'a> {
        if self.len <= index {
            panic!("ParallelVec: Index out of bounds: {}", index);
        } else {
            unsafe { self.get_unchecked_mut(index) }
        }
    }

    /// Returns references to elements, without doing bounds checking.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    ///
    /// [`get`]: Self::get
    #[inline]
    pub unsafe fn get_unchecked<'a>(&'a self, index: usize) -> Param::Ref<'a> {
        let ptr = Param::as_ptr(self.storage);
        Param::as_ref(Param::add(ptr, index))
    }

    /// Returns mutable references to elements, without doing bounds checking.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    ///
    /// [`get_mut`]: Self::get_mut
    #[inline]
    pub unsafe fn get_unchecked_mut<'a>(&'a mut self, index: usize) -> Param::RefMut<'a> {
        let ptr = self.as_mut_ptrs();
        Param::as_mut(Param::add(ptr, index))
    }

    /// Returns a raw pointer to the slice’s buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns, or else it will end up pointing
    /// to garbage.
    ///
    /// Modifying the container referenced by this slice may cause its buffer to be reallocated, which would also make any
    /// pointers to it invalid.
    #[inline]
    pub fn as_mut_ptrs(&mut self) -> Param::Ptr {
        Param::as_ptr(self.storage)
    }

    /// Gets the individual slices for very sub-`Vec`.
    #[inline]
    pub fn as_slices(&self) -> Param::Slices<'_> {
        unsafe { Param::as_slices(Param::as_ptr(self.storage), self.len) }
    }

    /// Gets mutable individual slices for very sub-`Vec`.
    #[inline]
    pub fn as_slices_mut(&mut self) -> Param::SlicesMut<'_> {
        unsafe { Param::as_slices_mut(self.as_mut_ptrs(), self.len) }
    }

    /// Swaps two elements.
    ///
    /// # Arguments
    ///  - `a` - The index of the first element
    ///  - `b` - The index of the second element
    ///
    /// # Panics
    /// Panics if a or b are out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a >= self.len {
            panic!("ParallelVec: Index out of bounds: {}", a);
        }
        if b >= self.len {
            panic!("ParallelVec: Index out of bounds: {}", b);
        }
        unsafe {
            self.swap_unchecked(a, b);
        }
    }

    /// Swaps two elements in the slice, without doing bounds checking.
    ///
    /// For a safe alternative see [`swap`].
    ///
    /// # Arguments
    ///  - `a` - The index of the first element
    ///  - `b` - The index of the second element
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller has to ensure that `a < self.len()` and `b < self.len()`.
    ///
    /// [`swap`]: Self::swap
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, a: usize, b: usize) {
        let base = Param::as_ptr(self.storage);
        let a_ptr = Param::add(base, a);
        let b_ptr = Param::add(base, b);
        Param::swap(a_ptr, b_ptr);
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than the vector’s current length, this has no effect.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    pub fn truncate(&mut self, len: usize) {
        if self.len <= len {
            return;
        }
        let start = len;
        let end = self.len;
        self.len = len;
        unsafe {
            let base = Param::as_ptr(self.storage);
            for idx in start..end {
                Param::drop(Param::add(base, idx));
            }
        }
    }

    /// Reverses the order of elements in the [`ParallelVec`], in place.
    ///
    /// This is a `O(n)` operation.
    pub fn reverse(&mut self) {
        if self.len == 0 {
            return;
        }
        Param::reverse(self.as_slices_mut())
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and
    /// the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        unsafe {
            if min_capacity > self.capacity {
                return;
            }
            let capacity = core::cmp::max(self.len, min_capacity);
            let src = Param::as_ptr(self.storage);
            let dst = Param::alloc(capacity);
            Param::copy_to_nonoverlapping(src, Param::as_ptr(dst), self.len);
            Param::dealloc(&mut self.storage, self.capacity);
            self.storage = dst;
            self.capacity = capacity;
        }
    }

    /// Swaps all elements in `self` with those in `other`.
    ///
    /// The length of other must be the same as `self`.  
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    pub fn swap_with(&mut self, other: &mut Self) {
        if self.len != other.len {
            panic!(
                "ParallelVec: attempted to use swap_with with ParallelVecs of different lenghths: {} vs {}",
                self.len,
                other.len
            )
        }
        unsafe {
            let mut a = self.as_mut_ptrs();
            let mut b = other.as_mut_ptrs();
            for _ in 0..self.len {
                Param::swap(a, b);
                a = Param::add(a, 1);
                b = Param::add(b, 1);
            }
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator may
    /// still inform the vector that there is space for a few more elements.
    pub fn shrink_to_fit(&mut self) {
        self.shrink_to(self.len);
    }

    /// Moves all the elements of `other` into `Self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut ParallelVec<Param>) {
        self.reserve(other.len);
        unsafe {
            let src = other.as_mut_ptrs();
            let dst = Param::add(self.as_mut_ptrs(), self.len);
            Param::copy_to_nonoverlapping(src, dst, other.len);
            // No need to drop from the other vec, data has been moved to
            // the current one. Just set the length here.
            other.len = 0;
        }
    }

    /// Appends an element to the back of a collection.
    pub fn push(&mut self, value: Param) {
        unsafe {
            self.reserve(1);
            let ptr = Param::add(self.as_mut_ptrs(), self.len);
            Param::write(ptr, value);
            self.len += 1;
        }
    }

    /// Removes the last element from the vector and returns it,
    /// or [`None`] if it is empty.
    ///
    /// [`None`]: Option::None
    pub fn pop(&mut self) -> Option<Param> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                let ptr = Param::add(self.as_mut_ptrs(), self.len);
                let value = Param::read(ptr);
                self.len -= 1;
                Some(value)
            }
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.  
    ///
    /// This does not preserve ordering, but is `O(1)`. If you need to
    /// preserve the element order, use [`remove`] instead.
    ///
    /// [`remove`]: Self::remove
    pub fn swap_remove(&mut self, index: usize) -> Param {
        if index >= self.len {
            panic!("ParallelVec: Index out of bounds {}", index);
        }

        unsafe {
            let target_ptr = Param::add(self.as_mut_ptrs(), index);
            let value = Param::read(target_ptr);
            self.len -= 1;

            if self.len != index {
                let end = Param::add(self.as_mut_ptrs(), self.len);
                Param::copy_to_nonoverlapping(end, target_ptr, 1);
            }

            value
        }
    }

    /// Inserts a value at `index`. Moves all of the elements above
    /// `index` up one index. This is a `O(N)` operation.
    ///
    /// # Panics
    /// This function will panic if `index` is greater than or equal to
    /// `len()`.
    pub fn insert(&mut self, index: usize, value: Param) {
        if index > self.len {
            panic!("ParallelVec: Index out of bounds {}", index);
        }
        unsafe {
            // TODO: (Performance) In the case where we do grow, this can result in redundant copying.
            self.reserve(1);
            let ptr = Param::add(self.as_mut_ptrs(), index);
            Param::copy_to(ptr, Param::add(ptr, 1), self.len - index);
            Param::write(ptr, value);
            self.len += 1;
        }
    }

    /// Removes a value at `index`. Moves all of the elements above
    /// `index` down one index. This is a `O(N)` operation.
    ///
    /// Returns `None` if `index` is is greater than or equal to `len()`.
    pub fn remove(&mut self, index: usize) -> Option<Param> {
        if index >= self.len {
            return None;
        }
        unsafe {
            let ptr = Param::add(self.as_mut_ptrs(), index);
            let value = Param::read(ptr);
            Param::copy_to(Param::add(ptr, 1), ptr, self.len - index - 1);
            self.len -= 1;
            Some(value)
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the
    /// given [`ParallelVec`]. The collection may reserve more space to avoid frequent
    /// reallocations. After calling reserve, capacity will be greater than or
    /// equal to `self.len() + additional`. Does nothing if capacity is already
    /// sufficient.
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            let new_len = self.len + additional;
            if new_len > self.capacity {
                let capacity = new_len.next_power_of_two().max(4);
                let dst = Param::alloc(capacity);
                let src = self.as_mut_ptrs();
                Param::copy_to_nonoverlapping(src, Param::as_ptr(dst), self.len);
                Param::dealloc(&mut self.storage, self.capacity);
                self.storage = dst;
                self.capacity = capacity;
            }
        }
    }

    /// Returns an iterator over the [`ParallelVec`].
    pub fn iter(&self) -> Iter<'_, Param> {
        Iter {
            base: Param::as_ptr(self.storage),
            idx: 0,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator that allows modifying each value.
    pub fn iter_mut(&mut self) -> IterMut<'_, Param> {
        IterMut {
            base: self.as_mut_ptrs(),
            idx: 0,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over the [`ParallelVec`].
    pub fn iters<'a>(&'a self) -> Param::Iters<'a> {
        unsafe {
            let ptr = Param::as_ptr(self.storage);
            let slices = Param::as_slices(ptr, self.len);
            Param::iters(slices)
        }
    }

    /// Gets individual iterators.
    pub fn iters_mut<'a>(&'a mut self) -> Param::ItersMut<'a> {
        unsafe {
            let ptr = Param::as_ptr(self.storage);
            let slices = Param::as_slices_mut(ptr, self.len);
            Param::iters_mut(slices)
        }
    }
}

impl<Param: ParallelVecParam + Copy> ParallelVec<Param> {
    /// Creates a [`ParallelVec`] by repeating `self` `n` times.
    pub fn repeat(&self, n: usize) -> ParallelVec<Param> {
        let mut new = ParallelVec::with_capacity(n * self.len);
        let mut dst = Param::as_ptr(new.storage);
        new.len = n * self.len;
        unsafe {
            let base = Param::as_ptr(self.storage);
            for _ in 0..n {
                for idx in 0..self.len {
                    let value = Param::read(Param::add(base, idx));
                    Param::write(dst, value.clone());
                    dst = Param::add(dst, 1);
                }
            }
        }
        new
    }
}

impl<Param: ParallelVecParam + Clone> ParallelVec<Param> {
    /// Fills self with elements by cloning value.
    #[inline(always)]
    pub fn fill(&mut self, value: Param) {
        self.fill_with(|| value.clone());
    }
}

impl<Param: ParallelVecParam> ParallelVec<Param> {
    /// Fills self with elements returned by calling a closure repeatedly.
    ///
    /// This method uses a closure to create new values. If you’d rather [`Clone`]
    /// a given value, use fill. If you want to use the [`Default`] trait to generate
    /// values, you can pass `Default::default` as the argument.
    pub fn fill_with<F: FnMut() -> Param>(&mut self, mut f: F) {
        unsafe {
            let base = self.as_mut_ptrs();
            for idx in 0..self.len {
                Param::write(Param::add(base, idx), f());
            }
        }
    }
}

impl<Param: ParallelVecParam> Drop for ParallelVec<Param> {
    fn drop(&mut self) {
        self.len = 0;
        unsafe {
            let base = Param::as_ptr(self.storage);
            for idx in 0..self.len {
                Param::drop(Param::add(base, idx));
            }
            Param::dealloc(&mut self.storage, self.capacity);
        }
    }
}

impl<Param: ParallelVecParam> IntoIterator for ParallelVec<Param> {
    type Item = Param;
    type IntoIter = IntoIter<Param>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { vec: self, idx: 0 }
    }
}

impl<Param: ParallelVecParam> Extend<Param> for ParallelVec<Param> {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Param>,
    {
        let iterator = iter.into_iter();
        let (min, _) = iterator.size_hint();
        self.reserve(min);
        for param in iterator {
            self.push(param);
        }
    }
}

impl<Param: ParallelVecParam + Clone> Clone for ParallelVec<Param> {
    fn clone(&self) -> Self {
        let mut clone = Self::with_capacity(self.len);
        unsafe {
            let base = Param::as_ptr(self.storage);
            for idx in 0..self.len {
                let value = Param::read(Param::add(base, idx));
                clone.push(value.clone());
            }
        }
        clone
    }
}

impl<Param: ParallelVecParam> Default for ParallelVec<Param> {
    fn default() -> Self {
        Self::new()
    }
}

/// Error when attempting to convert types to [`ParallelVec`].
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ParallelVecConversionError {
    /// The provided inputs were not the same length.
    UnevenLengths,
}

#[cfg(test)]
mod tests {
    use super::ParallelVec;

    #[test]
    fn layouts_do_not_overlap() {
        // Trying with both (small, large) and (large, small) to ensure nothing bleeds into anything else.
        // This verifies we correctly chunk the slices from the larger allocations.
        let mut vec_ab = ParallelVec::new();
        let mut vec_ba = ParallelVec::new();

        fn ab(v: usize) -> (u8, f64) {
            (v as u8, 200.0 + ((v as f64) / 200.0))
        }

        fn ba(v: usize) -> (f64, u8) {
            (15.0 + ((v as f64) / 16.0), (200 - v) as u8)
        }

        // Combined with the tests inside, also verifies that we are copying the data on grow correctly.
        for i in 0..100 {
            vec_ab.push(ab(i));
            let (a, b) = vec_ab.as_slices();
            assert_eq!(i + 1, a.len());
            assert_eq!(i + 1, b.len());
            assert_eq!(ab(0).0, a[0]);
            assert_eq!(ab(0).1, b[0]);
            assert_eq!(ab(i).0, a[i]);
            assert_eq!(ab(i).1, b[i]);

            vec_ba.push(ba(i));
            let (b, a) = vec_ba.as_slices();
            assert_eq!(i + 1, a.len());
            assert_eq!(i + 1, b.len());
            assert_eq!(ba(0).0, b[0]);
            assert_eq!(ba(0).1, a[0]);
            assert_eq!(ba(i).0, b[i]);
            assert_eq!(ba(i).1, a[i]);
        }
    }

    #[test]
    fn clones() {
        let mut src = ParallelVec::new();
        src.push((1.0, 2.0));
        src.push((3.0, 4.0));

        let dst = src.clone();
        assert_eq!(dst.len(), 2);
        assert_eq!(dst.index(0), (&1.0, &2.0));
        assert_eq!(dst.index(1), (&3.0, &4.0));
    }

    #[test]
    fn insert() {
        let mut src = ParallelVec::new();
        src.insert(0, (1, 2));
        src.insert(0, (3, 4));
        src.insert(1, (4, 5));
        assert_eq!(src.index(0), (&3, &4));
        assert_eq!(src.index(1), (&4, &5));
        assert_eq!(src.index(2), (&1, &2));
    }

    #[test]
    fn remove() {
        let mut src = ParallelVec::new();
        src.push((1, 2));
        src.push((3, 4));
        assert_eq!(src.remove(0), Some((1, 2)));
        assert_eq!(src.remove(0), Some((3, 4)));
        assert_eq!(src.len(), 0);
    }
}
