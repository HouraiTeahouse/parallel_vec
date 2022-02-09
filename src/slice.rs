use crate::assert_in_bounds;
use crate::iter::{Iter, IterMut};
use crate::ParallelParam;
use core::marker::PhantomData;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

/// A immutable dynamically-sized view into a contiguous heterogeneous sequence.
/// Contiguous here means that elements are laid out so that every element is
/// the same distance from its neighbors.
///
/// Unlike a struct of slices, this type only stores one length instead
/// of duplicating the values across multiple slice fields.
#[repr(C)]
pub struct ParallelSlice<'a, Param: ParallelParam> {
    // Do not reorder these fields. These must be in the same order as
    // ParallelVec for Deref and DerefMut to work properly.
    len: usize,
    storage: Param::Storage,
    _marker: PhantomData<&'a usize>,
}

impl<'a, Param: ParallelParam> ParallelSlice<'a, Param> {
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
    ///       for zero-length slices using [`ParallelParam::dangling()`].
    /// * `data` must point to `len` consecutive properly initialized values of type `Param`.
    /// * The memory referenced by the returned slice must not be accessed through any other pointer
    ///   (not derived from the return value) for the duration of lifetime `'a`.
    ///   Both read and write accesses are forbidden.
    /// * The total size `len * mem::size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///
    /// [`ParallelParam::dangling()`]: ParallelParam::dangling
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
    pub fn get<'b: 'a, I: ParallelSliceIndex<Self>>(&'b self, index: I) -> Option<I::Output> {
        index.get(self)
    }

    /// Returns the first element of the slice, or `None` if it is empty.
    #[inline(always)]
    pub fn first(&self) -> Option<Param::Ref<'_>> {
        self.get(0)
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

    /// Gets a immutable reference to the elements at `index`.
    ///
    /// # Panics
    /// This function will panic if `index` is >= `self.len`.
    #[inline]
    pub fn index<I>(&self, index: I) -> I::Output
    where
        I: ParallelSliceIndex<Self>,
    {
        index.index(self)
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
        Param::as_ref(Param::ptr_at(self.storage, index))
    }

    /// Gets the individual slices for every sub-slice.
    #[inline]
    pub fn as_slices(&self) -> Param::Slices<'_> {
        unsafe { Param::as_slices(Param::as_ptr(self.storage), self.len) }
    }

    /// Returns an iterator over the [`ParallelSlice`].
    pub fn iter(&self) -> Iter<'_, Param> {
        Iter {
            base: Param::as_ptr(self.storage),
            idx: 0,
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over the [`ParallelSlice`].
    pub fn iters(&self) -> Param::Iters<'_> {
        unsafe {
            let ptr = Param::as_ptr(self.storage);
            let slices = Param::as_slices(ptr, self.len);
            Param::iters(slices)
        }
    }
}

/// A mutable dynamically-sized view into a contiguous heterogeneous sequence.
/// Contiguous here means that elements are laid out so that every element is
/// the same distance from its neighbors.
///
/// Unlike a struct of slices, this type only stores one length instead
/// of duplicating the values across multiple slice fields.
#[repr(C)]
pub struct ParallelSliceMut<'a, Param: ParallelParam> {
    // Do not reorder these fields. These must be in the same order as
    // ParallelVec for Deref and DerefMut to work properly.
    len: usize,
    storage: Param::Storage,
    _marker: PhantomData<&'a usize>,
}

impl<'a, Param: ParallelParam> ParallelSliceMut<'a, Param> {
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
    ///       for zero-length slices using [`ParallelParam::dangling()`].
    /// * `data` must point to `len` consecutive properly initialized values of type `Param`.
    /// * The memory referenced by the returned slice must not be accessed through any other pointer
    ///   (not derived from the return value) for the duration of lifetime `'a`.
    ///   Both read and write accesses are forbidden.
    /// * The total size `len * mem::size_of::<T>()` of the slice must be no larger than `isize::MAX`.
    ///
    /// [`ParallelParam::dangling()`]: ParallelParam::dangling
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
    pub fn get<'b: 'a, I>(&'b self, index: I) -> Option<I::Output>
    where
        I: ParallelSliceIndex<Self>,
    {
        index.get(self)
    }

    /// Returns a mutable reference to the element at `index`, if available, or
    /// [`None`] if it is out of bounds.
    ///
    /// [`None`]: Option::None
    #[inline]
    pub fn get_mut<I>(&mut self, index: I) -> Option<I::Output>
    where
        I: ParallelSliceIndexMut<Self>,
    {
        index.get_mut(self)
    }

    /// Returns the first element of the slice, or `None` if it is empty.
    #[inline(always)]
    pub fn first(&self) -> Option<Param::Ref<'_>> {
        self.get(0)
    }

    /// Returns the mutable pointer first element of the slice, or `None` if it is empty.
    #[inline(always)]
    pub fn first_mut<'b: 'a>(&'b mut self) -> Option<Param::RefMut<'b>> {
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
    pub fn index<I>(&self, index: I) -> I::Output
    where
        I: ParallelSliceIndex<Self>,
    {
        index.index(self)
    }

    /// Gets a mutable reference to the elements at `index`.
    ///
    /// # Panics
    /// This function will panic if `index` is >= `self.len`.
    #[inline]
    pub fn index_mut<I>(&mut self, index: I) -> I::Output
    where
        I: ParallelSliceIndexMut<Self>,
    {
        index.index_mut(self)
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
        Param::as_ref(Param::ptr_at(self.storage, index))
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
        Param::as_mut(Param::ptr_at(self.storage, index))
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
        assert_in_bounds(a, self.len);
        assert_in_bounds(b, self.len);
        unsafe { self.swap_unchecked(a, b) }
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

    /// Reverses the order of elements in the [`ParallelSliceMut`], in place.
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
                "Attempted to use swap_with with slices of different lenghths: {} vs {}",
                self.len, other.len
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

    /// Returns an iterator over the [`ParallelSliceMut`].
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

    /// Returns an iterator over the [`ParallelSliceMut`].
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

impl<'a, Param: ParallelParam + Clone> ParallelSliceMut<'a, Param> {
    /// Fills self with elements by cloning value.
    #[inline(always)]
    pub fn fill(&mut self, value: Param) {
        self.fill_with(|| value.clone());
    }
}

impl<'a, Param: ParallelParam> ParallelSliceMut<'a, Param> {
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

pub trait ParallelSliceIndex<T> {
    type Output;
    fn get(self, slice: &T) -> Option<Self::Output>;
    fn index(self, slice: &T) -> Self::Output;
}

pub trait ParallelSliceIndexMut<T> {
    type Output;
    fn get_mut(self, slice: &mut T) -> Option<Self::Output>;
    fn index_mut(self, slice: &mut T) -> Self::Output;
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSlice<'s, Param>> for usize {
    type Output = Param::Ref<'s>;
    fn get(self, slice: &ParallelSlice<'s, Param>) -> Option<Self::Output> {
        if self > slice.len {
            return None;
        }

        unsafe { Some(Param::as_ref(Param::ptr_at(slice.storage, self))) }
    }

    fn index(self, slice: &ParallelSlice<'s, Param>) -> Self::Output {
        assert_in_bounds(self, slice.len);
        unsafe { Param::as_ref(Param::ptr_at(slice.storage, self)) }
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSliceMut<'s, Param>> for usize {
    type Output = Param::Ref<'s>;
    fn get(self, slice: &ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        if self > slice.len {
            return None;
        }

        unsafe { Some(Param::as_ref(Param::ptr_at(slice.storage, self))) }
    }

    fn index(self, slice: &ParallelSliceMut<'s, Param>) -> Self::Output {
        assert_in_bounds(self, slice.len);
        unsafe { Param::as_ref(Param::ptr_at(slice.storage, self)) }
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndexMut<ParallelSliceMut<'s, Param>> for usize {
    type Output = Param::RefMut<'s>;
    fn get_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        if self > slice.len {
            return None;
        }

        unsafe { Some(Param::as_mut(Param::ptr_at(slice.storage, self))) }
    }

    fn index_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Self::Output {
        assert_in_bounds(self, slice.len);
        unsafe { Param::as_mut(Param::ptr_at(slice.storage, self)) }
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSlice<'s, Param>> for Range<usize> {
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSlice<'s, Param>) -> Option<Self::Output> {
        if self.start > slice.len || self.end > slice.len {
            return None;
        }

        unsafe {
            let ptr = Param::ptr_at(slice.storage, self.start);
            Some(ParallelSlice::from_raw_parts(
                Param::as_storage(ptr),
                self.end - self.start,
            ))
        }
    }

    fn index(self, slice: &ParallelSlice<'s, Param>) -> Self::Output {
        assert_in_bounds(self.start, slice.len);
        assert_in_bounds(self.end, slice.len);
        unsafe {
            let ptr = Param::ptr_at(slice.storage, self.start);
            ParallelSlice::from_raw_parts(Param::as_storage(ptr), self.end - self.start)
        }
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSliceMut<'s, Param>> for Range<usize> {
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        if self.start > slice.len || self.end > slice.len {
            return None;
        }

        unsafe {
            let ptr = Param::ptr_at(slice.storage, self.start);
            Some(ParallelSlice::from_raw_parts(
                Param::as_storage(ptr),
                self.end - self.start,
            ))
        }
    }

    fn index(self, slice: &ParallelSliceMut<'s, Param>) -> Self::Output {
        assert_in_bounds(self.start, slice.len);
        assert_in_bounds(self.end, slice.len);
        unsafe {
            let ptr = Param::ptr_at(slice.storage, self.start);
            ParallelSlice::from_raw_parts(Param::as_storage(ptr), self.end - self.start)
        }
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndexMut<ParallelSliceMut<'s, Param>>
    for Range<usize>
{
    type Output = ParallelSliceMut<'s, Param>;
    fn get_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        if self.start > slice.len || self.end > slice.len {
            return None;
        }

        unsafe {
            let ptr = Param::ptr_at(slice.storage, self.start);
            Some(ParallelSliceMut::from_raw_parts(
                Param::as_storage(ptr),
                self.end - self.start,
            ))
        }
    }

    fn index_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Self::Output {
        assert_in_bounds(self.start, slice.len);
        assert_in_bounds(self.end, slice.len);
        unsafe {
            let ptr = Param::ptr_at(slice.storage, self.start);
            ParallelSliceMut::from_raw_parts(Param::as_storage(ptr), self.end - self.start)
        }
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSlice<'s, Param>>
    for RangeInclusive<usize>
{
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSlice<'s, Param>) -> Option<Self::Output> {
        let range = Range {
            start: *self.start(),
            end: *self.end() + 1,
        };
        range.get(slice)
    }

    fn index(self, slice: &ParallelSlice<'s, Param>) -> Self::Output {
        let range = Range {
            start: *self.start(),
            end: *self.end() + 1,
        };
        range.index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSliceMut<'s, Param>>
    for RangeInclusive<usize>
{
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        let range = Range {
            start: *self.start(),
            end: *self.end() + 1,
        };
        range.get(slice)
    }

    fn index(self, slice: &ParallelSliceMut<'s, Param>) -> Self::Output {
        let range = Range {
            start: *self.start(),
            end: *self.end() + 1,
        };
        range.index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndexMut<ParallelSliceMut<'s, Param>>
    for RangeInclusive<usize>
{
    type Output = ParallelSliceMut<'s, Param>;
    fn get_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        let range = Range {
            start: *self.start(),
            end: *self.end() + 1,
        };

        range.get_mut(slice)
    }

    fn index_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Self::Output {
        let range = Range {
            start: *self.start(),
            end: *self.end() + 1,
        };
        range.index_mut(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSlice<'s, Param>> for RangeTo<usize> {
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSlice<'s, Param>) -> Option<Self::Output> {
        Range {
            start: 0,
            end: self.end,
        }
        .get(slice)
    }

    fn index(self, slice: &ParallelSlice<'s, Param>) -> Self::Output {
        Range {
            start: 0,
            end: self.end,
        }
        .index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSliceMut<'s, Param>>
    for RangeTo<usize>
{
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        Range {
            start: 0,
            end: self.end,
        }
        .get(slice)
    }

    fn index(self, slice: &ParallelSliceMut<'s, Param>) -> Self::Output {
        Range {
            start: 0,
            end: self.end,
        }
        .index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndexMut<ParallelSliceMut<'s, Param>>
    for RangeTo<usize>
{
    type Output = ParallelSliceMut<'s, Param>;
    fn get_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        Range {
            start: 0,
            end: self.end,
        }
        .get_mut(slice)
    }

    fn index_mut<'a>(self, slice: &mut ParallelSliceMut<'s, Param>) -> Self::Output {
        Range {
            start: 0,
            end: self.end,
        }
        .index_mut(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSlice<'s, Param>>
    for RangeFrom<usize>
{
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSlice<'s, Param>) -> Option<Self::Output> {
        Range {
            start: self.start,
            end: slice.len,
        }
        .get(slice)
    }

    fn index(self, slice: &ParallelSlice<'s, Param>) -> Self::Output {
        Range {
            start: self.start,
            end: slice.len,
        }
        .index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSliceMut<'s, Param>>
    for RangeFrom<usize>
{
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        Range {
            start: self.start,
            end: slice.len,
        }
        .get(slice)
    }

    fn index(self, slice: &ParallelSliceMut<'s, Param>) -> Self::Output {
        Range {
            start: self.start,
            end: slice.len,
        }
        .index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndexMut<ParallelSliceMut<'s, Param>>
    for RangeFrom<usize>
{
    type Output = ParallelSliceMut<'s, Param>;
    fn get_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        Range {
            start: self.start,
            end: slice.len,
        }
        .get_mut(slice)
    }

    fn index_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Self::Output {
        Range {
            start: self.start,
            end: slice.len,
        }
        .index_mut(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSlice<'s, Param>> for RangeFull {
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSlice<'s, Param>) -> Option<Self::Output> {
        Range {
            start: 0,
            end: slice.len,
        }
        .get(slice)
    }

    fn index(self, slice: &ParallelSlice<'s, Param>) -> Self::Output {
        Range {
            start: 0,
            end: slice.len,
        }
        .index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndex<ParallelSliceMut<'s, Param>> for RangeFull {
    type Output = ParallelSlice<'s, Param>;
    fn get(self, slice: &ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        Range {
            start: 0,
            end: slice.len,
        }
        .get(slice)
    }

    fn index(self, slice: &ParallelSliceMut<'s, Param>) -> Self::Output {
        Range {
            start: 0,
            end: slice.len,
        }
        .index(slice)
    }
}

impl<'s, Param: ParallelParam> ParallelSliceIndexMut<ParallelSliceMut<'s, Param>> for RangeFull {
    type Output = ParallelSliceMut<'s, Param>;
    fn get_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Option<Self::Output> {
        Range {
            start: 0,
            end: slice.len,
        }
        .get_mut(slice)
    }

    fn index_mut(self, slice: &mut ParallelSliceMut<'s, Param>) -> Self::Output {
        Range {
            start: 0,
            end: slice.len,
        }
        .index_mut(slice)
    }
}
