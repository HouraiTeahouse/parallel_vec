#![allow(non_snake_case)]
#![feature(generic_associated_types)]
use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

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

    #[inline]
    pub fn get<'a>(&'a self, index: usize) -> Option<Param::Ref<'a>> {
        if self.len <= index {
            None
        } else {
            unsafe { Some(self.get_unchecked(index)) }
        }
    }

    #[inline]
    pub fn get_mut<'a>(&'a mut self, index: usize) -> Option<Param::RefMut<'a>> {
        if self.len <= index {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index)) }
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
            let capacity = std::cmp::max(self.len, min_capacity);
            let src = Param::as_ptr(self.storage);
            let dst = Param::alloc(capacity);
            Param::copy_to_nonoverlapping(src, Param::as_ptr(dst), self.len);
            Param::dealloc(&mut self.storage, self.capacity);
            self.storage = dst;
            self.capacity = capacity;
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

    pub fn iter(&self) -> Iter<'_, Param> {
        Iter {
            base: Param::as_ptr(self.storage),
            idx: 0,
            len: self.len,
            _marker: PhantomData,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, Param> {
        IterMut {
            base: self.as_mut_ptrs(),
            idx: 0,
            len: self.len,
            _marker: PhantomData,
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

pub struct Iter<'a, Param: ParallelVecParam> {
    base: Param::Ptr,
    idx: usize,
    len: usize,
    _marker: PhantomData<&'a Param>,
}

impl<'a, Param: ParallelVecParam> Iterator for Iter<'a, Param> {
    type Item = Param::Ref<'a>;
    fn next(&mut self) -> Option<Param::Ref<'a>> {
        unsafe {
            if self.idx >= self.len {
                return None;
            }
            let ptr = Param::add(self.base, self.idx);
            let value = Param::as_ref(ptr);
            self.idx += 1;
            Some(value)
        }
    }
}

pub struct IterMut<'a, Param: ParallelVecParam> {
    base: Param::Ptr,
    idx: usize,
    len: usize,
    _marker: PhantomData<&'a Param>,
}

impl<'a, Param: ParallelVecParam> Iterator for IterMut<'a, Param> {
    type Item = Param::RefMut<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.idx >= self.len {
                return None;
            }
            let ptr = Param::add(self.base, self.idx);
            let value = Param::as_mut(ptr);
            self.idx += 1;
            Some(value)
        }
    }
}

/// This trait contains the basic operations for creating variadic
/// parallel vector implementations.
///
/// This trait is sealed and cannot be implemented outside of `parallel_vec`.
pub unsafe trait ParallelVecParam: Sized + private::Sealed {
    type Storage: Copy;
    type Ptr: Copy;
    type Offsets;
    type Ref<'a>;
    type RefMut<'a>;
    type Vecs;
    type Slices<'a>;
    type SlicesMut<'a>;

    /// Creates a set of dangling pointers for the given types.
    fn dangling() -> Self::Storage;

    /// Converts a set of [`NonNull`]s into their associated
    /// pointer types.
    fn as_ptr(storage: Self::Storage) -> Self::Ptr;

    /// Allocates a buffer for a given capacity.
    ///
    /// # Safety
    /// Capacity should be non-zero.
    unsafe fn alloc(capacity: usize) -> Self::Storage;

    /// Deallocates a buffer allocated from [`alloc`].
    ///
    /// # Safety
    /// `storage` must have been allocated from [`alloc`] alongside
    /// the provided `capacity`.
    ///
    /// [`alloc`]: Self::alloc
    unsafe fn dealloc(storage: &mut Self::Storage, capacity: usize);

    /// Creates a layout for a [`ParallelVec`] for a given `capacity`
    fn layout_for_capacity(capacity: usize) -> MemoryLayout<Self>;

    /// Gets the legnth for the associated `Vec`s.
    ///
    /// Returns `None` if not all of the `Vec`s share the same
    /// length.
    fn get_vec_len(vecs: &Self::Vecs) -> Option<usize>;

    /// Gets the underlying pointers for the associated `Vec`s.
    ///
    /// # Safety
    /// The provided `Vec`s must be correctly allocated.
    unsafe fn get_vec_ptrs(vecs: &mut Self::Vecs) -> Self::Ptr;

    /// Adds `offset` to all of the pointers in `base`.
    ///
    /// # Safety
    /// `base` and `base + offset` must be valid non-null pointers for
    /// the associated types.
    unsafe fn add(base: Self::Ptr, offset: usize) -> Self::Ptr;

    /// Copies `size` elements from the continguous memory pointed to by `src` into
    /// `dst`.
    ///
    /// # Safety
    ///  - `src` and `dst` must be a valid, non-null pointer for the associated types.
    ///  - `size` must be approriately set for the allocation that both `src` and `dst`
    ///    point to.
    unsafe fn copy_to(src: Self::Ptr, dst: Self::Ptr, size: usize);

    /// Copies `size` elements from the continguous memory pointed to by `src` into
    /// `dst`.
    ///
    /// # Safety
    ///  - `src` and `dst` must be a valid, non-null pointer for the associated types.
    ///  - `size` must be approriately set for the allocation that both `src` and `dst`
    ///    point to.
    ///  - `src..src + size` must not overlap with the memory range of `dst..dst + size`.
    unsafe fn copy_to_nonoverlapping(src: Self::Ptr, dst: Self::Ptr, size: usize);

    /// Creates a set of immutable slices from `ptr` and a provided length.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer. `len` must be approriately set
    /// for the allocation that `ptr` points to.
    unsafe fn as_slices<'a>(ptr: Self::Ptr, len: usize) -> Self::Slices<'a>;

    /// Creates a set of mutable slices from `ptr` and a provided length.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer. `len` must be approriately set
    /// for the allocation that `ptr` points to.
    unsafe fn as_slices_mut<'a>(ptr: Self::Ptr, len: usize) -> Self::SlicesMut<'a>;

    /// Converts `ptr` into a set of immutable references.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer.
    unsafe fn as_ref<'a>(ptr: Self::Ptr) -> Self::Ref<'a>;

    /// Converts `ptr` into a set of mutable references.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer.
    unsafe fn as_mut<'a>(ptr: Self::Ptr) -> Self::RefMut<'a>;

    /// Reads the values to pointed to by `ptr`.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer.
    unsafe fn read(ptr: Self::Ptr) -> Self;

    /// Writes `value` to `ptr`.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer.
    unsafe fn write(ptr: Self::Ptr, value: Self);

    /// Swaps the values pointed to by the provided pointers.
    ///
    /// # Safety
    /// Both `a` and `b` must be valid for all of it's consitutent member pointers.
    unsafe fn swap(a: Self::Ptr, other: Self::Ptr);

    /// Drops the values pointed to by the pointers.
    ///
    /// # Safety
    /// The caller must ensure that the values pointed to by the pointers have
    /// not already been dropped prior.
    unsafe fn drop(ptr: Self::Ptr);
}

pub struct MemoryLayout<Param: ParallelVecParam> {
    layout: Layout,
    offsets: Param::Offsets,
}

pub enum ParallelVecConversionError {
    UnevenLengths,
}

mod private {
    pub trait Sealed {}

    macro_rules! impl_seal {
        ($($ts:ident),*) => {
            impl<$($ts,)*> Sealed for ($($ts,)*) {}
        }
    }

    impl_seal!(T1, T2);
    impl_seal!(T1, T2, T3);
    impl_seal!(T1, T2, T3, T4);
    impl_seal!(T1, T2, T3, T4, T5);
    impl_seal!(T1, T2, T3, T4, T5, T6);
    impl_seal!(T1, T2, T3, T4, T5, T6, T7);
    impl_seal!(T1, T2, T3, T4, T5, T6, T7, T8);
    impl_seal!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
    impl_seal!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
    impl_seal!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
    impl_seal!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
}

macro_rules! skip_first {
    ($first:ident, $second: ident) => {
        $second
    };
}

macro_rules! impl_parallel_vec_param {
    ($t1: ident, $v1: ident, $($ts:ident, $vs:ident),*) => {
        unsafe impl<$t1: 'static $(, $ts: 'static)*> ParallelVecParam for ($t1 $(, $ts)*) {
            type Storage = (NonNull<$t1> $(, NonNull<$ts>)*);
            type Ref<'a> = (&'a $t1, $(&'a $ts,)*);
            type RefMut<'a> = (&'a mut $t1, $(&'a mut $ts,)*);
            type Slices<'a> = (&'a [$t1] $(, &'a [$ts])*);
            type SlicesMut<'a> = (&'a mut [$t1] $(, &'a mut [$ts])*);
            type Vecs = (Vec<$t1> $(, Vec<$ts>)*);
            type Ptr = (*mut $t1 $(, *mut $ts)*);
            type Offsets = (usize $(, skip_first!($ts, usize))*);

            #[inline(always)]
            fn dangling() -> Self::Storage {
                (NonNull::dangling(), $(NonNull::<$ts>::dangling()),*)
            }

            #[inline(always)]
            fn as_ptr(storage: Self::Storage) -> Self::Ptr {
                let ($t1$(, $ts)*) = storage;
                ($t1.as_ptr() $(, $ts.as_ptr())*)
            }

            unsafe fn alloc(capacity: usize) -> Self::Storage {
                let layout = Self::layout_for_capacity(capacity);
                let bytes = std::alloc::alloc(layout.layout);
                let (_ $(, $ts)*) = layout.offsets;
                (
                    NonNull::new_unchecked(bytes.cast::<$t1>())
                    $(, NonNull::new_unchecked(bytes.add($ts).cast::<$ts>()))*
                )
            }

            unsafe fn dealloc(storage: &mut Self::Storage, capacity: usize) {
                if capacity > 0 {
                    let layout = Self::layout_for_capacity(capacity);
                    std::alloc::dealloc(storage.0.as_ptr().cast::<u8>(), layout.layout);
                }
            }

            fn layout_for_capacity(capacity: usize) -> MemoryLayout<Self> {
                let layout = Layout::array::<$t1>(capacity).unwrap();
                $(let (layout, $ts) = layout.extend(Layout::array::<$ts>(capacity).unwrap()).unwrap();)*
                MemoryLayout {
                    layout,
                    offsets: (0, $($ts),*)
                }
            }

            #[inline(always)]
            unsafe fn add(base: Self::Ptr, offset: usize) -> Self::Ptr {
                let ($t1, $($ts),*) = base;
                ($t1.add(offset), $($ts.add(offset)),*)
            }

            #[inline(always)]
            unsafe fn copy_to(src: Self::Ptr, dst: Self::Ptr, len: usize) {
                let ($t1, $($ts),*) = src;
                let ($v1, $($vs),*) = dst;
                $t1.copy_to($v1, len);
                $($ts.copy_to($vs, len);)*
            }

            #[inline(always)]
            unsafe fn copy_to_nonoverlapping(src: Self::Ptr, dst: Self::Ptr, len: usize) {
                let ($t1, $($ts),*) = src;
                let ($v1, $($vs),*) = dst;
                $t1.copy_to_nonoverlapping($v1, len);
                $(
                    $ts.copy_to_nonoverlapping($vs, len);
                )*
            }

            #[inline(always)]
            unsafe fn as_slices<'a>(ptr: Self::Ptr, len: usize) -> Self::Slices<'a> {
                let ($t1, $($ts),*) = ptr;
                (
                    std::slice::from_raw_parts($t1, len)
                    $(
                        , std::slice::from_raw_parts($ts, len)
                    )*
                )
            }

            #[inline(always)]
            unsafe fn as_slices_mut<'a>(ptr: Self::Ptr, len: usize) -> Self::SlicesMut<'a> {
                let ($t1, $($ts),*) = ptr;
                (
                    std::slice::from_raw_parts_mut($t1, len)
                    $(
                        , std::slice::from_raw_parts_mut($ts, len)
                    )*
                )
            }

            #[inline(always)]
            unsafe fn as_ref<'a>(ptr: Self::Ptr) -> Self::Ref<'a> {
                let ($t1, $($ts),*) = ptr;
                (&*$t1 $(, &*$ts)*)
            }

            #[inline(always)]
            unsafe fn as_mut<'a>(ptr: Self::Ptr) -> Self::RefMut<'a> {
                let ($t1, $($ts),*) = ptr;
                (&mut *$t1 $(, &mut *$ts)*)
            }

            #[inline(always)]
            unsafe fn read(ptr: Self::Ptr) -> Self {
                let ($t1, $($ts),*) = ptr;
                ($t1.read() $(, $ts.read())*)
            }

            #[inline(always)]
            unsafe fn write(ptr: Self::Ptr, value: Self) {
                let ($t1, $($ts),*) = ptr;
                let ($v1, $($vs),*) = value;
                $t1.write($v1);
                $($ts.write($vs);)*
            }

            #[inline(always)]
            unsafe fn swap(a: Self::Ptr, b: Self::Ptr) {
                let ($v1, $($vs),*) = a;
                let ($t1, $($ts),*) = b;
                std::ptr::swap($t1, $v1);
                $(std::ptr::swap($ts, $vs);)*
            }

            #[inline(always)]
            unsafe fn drop(ptr: Self::Ptr) {
                let ($t1, $($ts),*) = Self::read(ptr);
                std::mem::drop($t1);
                $(std::mem::drop($ts);)*
            }

            fn get_vec_len(vecs: &Self::Vecs) -> Option<usize> {
                let ($t1, $($ts),*) = vecs;
                let len = $t1.len();
                $(
                    if $ts.len() != len {
                        return None;
                    }
                )*
                Some(len)
            }

            unsafe fn get_vec_ptrs(vecs: &mut Self::Vecs) -> Self::Ptr {
                let ($t1, $($ts),*) = vecs;
                ($t1.as_mut_ptr() $(, $ts.as_mut_ptr())*)
            }
        }

        impl<$t1: 'static $(, $ts: 'static)*> TryFrom<(Vec<$t1> $(, Vec<$ts>)*)> for ParallelVec<($t1 $(, $ts)*)> {
            type Error = ParallelVecConversionError;
            fn try_from(mut vecs: (Vec<$t1> $(, Vec<$ts>)*)) -> Result<Self, Self::Error> {
                let len = <($t1 $(, $ts)*) as ParallelVecParam>::get_vec_len(&vecs);
                if let Some(len) = len {
                    let parallel_vec = Self::with_capacity(len);
                    // SAFE: This is a move. Nothing should be dropped here.
                    unsafe {
                        let src = <($t1 $(, $ts)*) as ParallelVecParam>::get_vec_ptrs(&mut vecs);
                        let dst = <($t1 $(, $ts)*) as ParallelVecParam>::as_ptr(parallel_vec.storage);
                        <($t1 $(, $ts)*) as ParallelVecParam>::copy_to_nonoverlapping(src, dst, len);
                        std::mem::forget(vecs);
                    }
                    Ok(parallel_vec)
                } else {
                    Err(ParallelVecConversionError::UnevenLengths)
                }
            }
        }
    }
}

impl_parallel_vec_param!(T1, V1, T2, V2);
impl_parallel_vec_param!(T1, V1, T2, V2, T3, V3);
impl_parallel_vec_param!(T1, V1, T2, V2, T3, V3, T4, V4);
impl_parallel_vec_param!(T1, V1, T2, V2, T3, V3, T4, V4, T5, V5);
impl_parallel_vec_param!(T1, V1, T2, V2, T3, V3, T4, V4, T5, V5, T6, V6);
impl_parallel_vec_param!(T1, V1, T2, V2, T3, V3, T4, V4, T5, V5, T6, V6, T7, V7);
impl_parallel_vec_param!(T1, V1, T2, V2, T3, V3, T4, V4, T5, V5, T6, V6, T7, V7, T8, V8);
impl_parallel_vec_param!(T1, V1, T2, V2, T3, T4, V3, V4, T5, V5, T6, V6, T7, V7, T8, V8, T9, V9);
impl_parallel_vec_param!(
    T1, V1, T2, V2, T3, T4, V3, V4, T5, V5, T6, V6, T7, V7, T8, V8, T9, V9, T10, V10
);
impl_parallel_vec_param!(
    T1, V1, T2, V2, T3, T4, V3, V4, T5, V5, T6, V6, T7, V7, T8, V8, T9, V9, T10, V10, T11, V11
);
impl_parallel_vec_param!(
    T1, V1, T2, V2, T3, T4, V3, V4, T5, V5, T6, V6, T7, V7, T8, V8, T9, V9, T10, V10, T11, V11,
    T12, V12
);

#[cfg(test)]
mod tests {
    use super::ParallelVec;
    use testdrop::TestDrop;

    fn assert_all_dropped(td: &TestDrop) {
        assert_eq!(td.num_dropped_items(), td.num_tracked_items());
    }

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

    // #[test]
    // fn sort() {
    //     let mut vec = ParallelVec::new();

    //     vec.push((3, 'a', 4.0));
    //     vec.push((1, 'b', 5.0));
    //     vec.push((2, 'c', 6.0));

    //     vec.sort_unstable_by(|(a1, _, _), (a2, _, _)| a1.cmp(a2));

    //     assert_eq!(vec.index(0), (&1, &('b'), &5.0));
    //     assert_eq!(vec.index(1), (&2, &('c'), &6.0));
    //     assert_eq!(vec.index(2), (&3, &('a'), &4.0));
    // }

    // #[test]
    // fn drops() {
    //     let td = TestDrop::new();
    //     let (id, item) = td.new_item();
    //     {
    //         let mut soa = ParallelVec::new();
    //         soa.push((1.0, item));
    //         // Did not drop when moved into the soa
    //         td.assert_no_drop(id);
    //         // Did not drop through resizing the soa.
    //         for _ in 0..50 {
    //             soa.push((2.0, td.new_item().1));
    //         }
    //         td.assert_no_drop(id);
    //     }
    //     // Dropped with the soa
    //     td.assert_drop(id);
    //     assert_all_dropped(&td);
    // }

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
