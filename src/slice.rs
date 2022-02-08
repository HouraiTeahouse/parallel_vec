use crate::ParallelVecParam;
use crate::iter::{Iter, IterMut};
use core::marker::PhantomData;

/// A dynamically-sized view into a contiguous heterogeneous sequence. Contiguous 
/// here means that elements are laid out so that every element is the same 
/// distance from its neighbors.
///
/// Unlike a struct of slices, this type only stores one length instead
/// of duplicating the values across multiple slice fields.
pub struct ParallelSlice<'a, Param: ParallelVecParam> {
    len: usize,
    storage: Param::Storage,
    _marker: PhantomData<&'a usize>,
}

impl<'a, Param: ParallelVecParam> ParallelSlice<'a, Param> {
    /// Forms a slice from a pointer and a length.
    /// 
    /// The `len` argument is the number of elements, not the number of bytes.
    /// 
    /// # Safety
    /// 
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// * `data` must be valid for both reads and writes for `len * mem::size_of::<T>()` many bytes,
    ///   and it must be properly aligned. This means in particular:
    ///
    ///     * The entire memory range of this slice must be contained within a single allocated object!
    ///       Slices can never span across multiple allocated objects.
    ///     * `data` must be non-null and aligned even for zero-length slices. One
    ///       reason for this is that enum layout optimizations may rely on references
    ///       (including slices of any length) being aligned and non-null to distinguish
    ///       them from other data. You can obtain a pointer that is usable as `data`
    ///       for zero-length slices using [`ParallelVecParam::dangling()`].
    /// * `data` must point to `len` consecutive properly initialized values of type `Param`.
    /// * The memory referenced by the returned slice must not be accessed through any other pointer
    ///   (not derived from the return value) for the duration of lifetime `'a`.
    ///   Both read and write accesses are forbidden.
    /// * The total size `len * mem::size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///   See the safety documentation of [`pointer::offset`].
    ///
    /// [`ParallelVecParam::dangling()`]: ParallelVecParam::dangling
    pub unsafe fn from_raw_parts(data: Param::Storage, len: usize) -> Self {
        Self {
            len,
            storage: data,
            _marker: PhantomData,
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

    /// Returns a immutable reference to the element at `index`, if available, or
    /// [`None`] if it is out of bounds.
    ///
    /// [`None`]: Option::None
    #[inline]
    pub fn get<'b: 'a>(&'b self, index: usize) -> Option<Param::Ref<'b>> {
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
    pub fn get_mut(&mut self, index: usize) -> Option<Param::RefMut<'_>> {
        if self.len <= index {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index)) }
        }
    }

    /// Returns the first element of the slice, or `None` if it is empty.
    #[inline(always)]
    pub fn first(&self) -> Option<Param::Ref<'_>> {
        self.get(0)
    }

    /// Returns the mutable pointer first element of the slice, or `None` if it is empty.
    #[inline(always)]
    pub fn first_mut(&mut self) -> Option<Param::RefMut<'_>> {
        self.get_mut(0)
    }

    /// Returns the last element of the slice, or `None` if it is empty.
    #[inline]
    pub fn last(&self) -> Option<Param::Ref<'_>> {
        if self.len == 0 {
            None
        } else {
            unsafe { Some(self.get_unchecked(self.len - 1)) }
        }
    }

    /// Returns the mutable pointer last element of the slice, or `None` if it is empty.
    #[inline]
    pub fn last_mut(&mut self) -> Option<Param::RefMut<'_>> {
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
            panic!("ParallelSlice: Index out of bounds: {}", index);
        } else {
            unsafe { self.get_unchecked(index) }
        }
    }

    /// Gets a mutable reference to the elements at `index`.
    ///
    /// # Panics
    /// This function will panic if `index` is >= `self.len`.
    #[inline]
    pub fn index_mut(&mut self, index: usize) -> Param::RefMut<'_> {
        if self.len <= index {
            panic!("ParallelSlice: Index out of bounds: {}", index);
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
    pub unsafe fn get_unchecked(&self, index: usize) -> Param::Ref<'_> {
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
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> Param::RefMut<'_> {
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

    /// Gets the individual slices for every sub-slice.
    #[inline]
    pub fn as_slices(&self) -> Param::Slices<'_> {
        unsafe { Param::as_slices(Param::as_ptr(self.storage), self.len) }
    }

    /// Gets mutable individual slices for every sub-slice.
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

    /// Reverses the order of elements in the [`ParallelVec`], in place.
    ///
    /// This is a `O(n)` operation.
    pub fn reverse(&mut self) {
        if self.len == 0 {
            return;
        }
        Param::reverse(self.as_slices_mut())
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
    pub fn iters(&self) -> Param::Iters<'_> {
        unsafe {
            let ptr = Param::as_ptr(self.storage);
            let slices = Param::as_slices(ptr, self.len);
            Param::iters(slices)
        }
    }

    /// Gets individual iterators.
    pub fn iters_mut(&mut self) -> Param::ItersMut<'_> {
        unsafe {
            let ptr = Param::as_ptr(self.storage);
            let slices = Param::as_slices_mut(ptr, self.len);
            Param::iters_mut(slices)
        }
    }
}

impl<'a, Param: ParallelVecParam + Clone> ParallelSlice<'a, Param> {
    /// Fills self with elements by cloning value.
    #[inline(always)]
    pub fn fill(&mut self, value: Param) {
        self.fill_with(|| value.clone());
    }
}

impl<'a, Param: ParallelVecParam> ParallelSlice<'a, Param> {
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

// impl<'a, Param: ParallelVecParam> IntoIterator for ParallelSlice<'a, Param> {
//     type Item = Param::;
//     type IntoIter = IntoIter<Param>;
//     fn into_iter(self) -> Self::IntoIter {
//         IntoIter { vec: self, idx: 0 }
//     }
// }