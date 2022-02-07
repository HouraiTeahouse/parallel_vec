#![allow(non_snake_case)]
#![feature(generic_associated_types)]
use std::alloc::Layout;
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
            storage: Param::Storage::dangling(),
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
                    storage: Param::Storage::alloc(capacity),
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
    pub fn get(
        &self,
        index: usize,
    ) -> Option<<<Param::Storage as Storage>::Ptr as ParamPtr>::Ref<'_>> {
        if self.len < index {
            None
        } else {
            unsafe { Some(self.get_unchecked(index)) }
        }
    }

    #[inline]
    pub fn get_mut(
        &self,
        index: usize,
    ) -> Option<<<Param::Storage as Storage>::Ptr as ParamPtr>::MutRef<'_>> {
        if self.len < index {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index)) }
        }
    }

    /// Returns references to elements, without doing bounds checking.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    #[inline]
    pub unsafe fn get_unchecked(
        &self,
        index: usize,
    ) -> <<Param::Storage as Storage>::Ptr as ParamPtr>::Ref<'_> {
        self.storage.as_ptr().add(index).as_ref()
    }

    /// Returns mutable references to elements, without doing bounds checking.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior even if the resulting reference is not used.
    #[inline]
    pub unsafe fn get_unchecked_mut(
        &self,
        index: usize,
    ) -> <<Param::Storage as Storage>::Ptr as ParamPtr>::MutRef<'_> {
        self.storage.as_ptr().add(index).as_mut()
    }

    /// Returns a raw pointer to the slice’s buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns, or else it will end up pointing
    /// to garbage.
    ///
    /// Modifying the container referenced by this slice may cause its buffer to be reallocated, which would also make any
    /// pointers to it invalid.
    #[inline]
    pub fn as_mut_ptrs(&mut self) -> <Param::Storage as Storage>::Ptr {
        self.storage.as_ptr()
    }

    /// Gets the individual slices for very sub-`Vec`.
    #[inline]
    pub fn as_slices(&self) -> <<Param::Storage as Storage>::Ptr as ParamPtr>::Slices<'_> {
        unsafe { self.storage.as_ptr().as_slices(self.len) }
    }

    /// Gets mutable individual slices for very sub-`Vec`.
    #[inline]
    pub fn as_slices_mut(&self) -> <<Param::Storage as Storage>::Ptr as ParamPtr>::SlicesMut<'_> {
        unsafe { self.storage.as_ptr().as_slices_mut(self.len) }
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
        let a_ptr = self.storage.as_ptr().add(a);
        let b_ptr = self.storage.as_ptr().add(b);
        a_ptr.swap(b_ptr);
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than the vector’s current length, this has no effect.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    pub fn truncate(&mut self, len: usize) {
        while self.len > len {
            self.pop();
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
            let ptr = Param::Storage::alloc(capacity);
            self.storage
                .as_ptr()
                .copy_to_nonoverlapping(ptr.as_ptr(), self.len);
            self.storage.dealloc(capacity);
            self.storage = ptr;
            self.capacity = capacity;
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator may
    /// still inform the vector that there is space for a few more elements.
    pub fn shirnk_to_fit(&mut self) {
        self.shrink_to(self.len);
    }

    /// Moves all the elements of `other` into `Self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut ParallelVec<Param>) {
        self.reserve(other.len);
        unsafe {
            let src = other.as_mut_ptrs();
            let dst = self.as_mut_ptrs().add(self.len);
            src.copy_to_nonoverlapping(dst, other.len);
        }
        other.clear();
    }

    /// Appends an element to the back of a collection.
    pub fn push(&mut self, value: <<Param::Storage as Storage>::Ptr as ParamPtr>::Param) {
        unsafe {
            self.reserve(1);
            self.storage.as_ptr().add(self.len).write(value);
            self.len += 1;
        }
    }

    pub fn pop(&mut self) -> Option<<<Param::Storage as Storage>::Ptr as ParamPtr>::Param> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                let value = self.storage.as_ptr().add(self.len).read();
                self.len -= 1;
                Some(value)
            }
        }
    }

    pub fn swap_remove(
        &mut self,
        index: usize,
    ) -> <<Param::Storage as Storage>::Ptr as ParamPtr>::Param {
        if index >= self.len {
            panic!("ParallelVec: Index out of bounds {}", index);
        }

        unsafe {
            let target_ptr = self.storage.as_ptr().add(index);
            let value = target_ptr.read();
            self.len -= 1;

            if self.len != index {
                let end = self.storage.as_ptr().add(self.len);
                target_ptr.copy_to_nonoverlapping(end, 1);
            }

            value
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            let new_len = self.len + additional;
            if new_len > self.capacity {
                let capacity = new_len.next_power_of_two().max(4);
                let ptr = Param::Storage::alloc(capacity);
                self.storage
                    .as_ptr()
                    .copy_to_nonoverlapping(ptr.as_ptr(), self.len);
                self.storage.dealloc(capacity);
                self.storage = ptr;
                self.capacity = capacity;
            }
        }
    }
}

impl<Param: ParallelVecParam> Drop for ParallelVec<Param> {
    fn drop(&mut self) {
        self.clear();
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

pub trait ParallelVecParam {
    type Storage: Storage;
}

pub trait Storage: Copy {
    type Ptr: ParamPtr;
    type Offsets;
    fn dangling() -> Self;
    fn as_ptr(self) -> Self::Ptr;
    unsafe fn alloc(capacity: usize) -> Self;
    unsafe fn dealloc(&mut self, capacity: usize);
    fn layout_for_capacity(capacity: usize) -> MemoryLayout<Self>;
}

pub trait ParamPtr: Copy {
    type Param: ParallelVecParam;
    type Ref<'a>;
    type MutRef<'a>;
    type Slices<'a>;
    type SlicesMut<'a>;
    unsafe fn add(self, offset: usize) -> Self;
    unsafe fn copy_to(self, dst: Self, size: usize);
    unsafe fn copy_to_nonoverlapping(self, dst: Self, size: usize);
    unsafe fn as_slices<'a>(self, len: usize) -> Self::Slices<'a>;
    unsafe fn as_slices_mut<'a>(self, len: usize) -> Self::SlicesMut<'a>;
    unsafe fn as_ref<'a>(self) -> Self::Ref<'a>;
    unsafe fn as_mut<'a>(self) -> Self::MutRef<'a>;
    unsafe fn read(self) -> Self::Param;
    unsafe fn write(self, value: Self::Param);
    unsafe fn swap(self, other: Self);
}

pub struct MemoryLayout<Param: Storage> {
    layout: Layout,
    offsets: Param::Offsets,
}

macro_rules! skip_first {
    ($first:ident, $second: ident) => {
        $second
    };
}

macro_rules! impl_parallel_vec_param {
    ($t1: ident, $($ts:ident),*) => {
        impl<$t1: 'static $(, $ts: 'static)*> ParallelVecParam for ($t1 $(, $ts)*) {
            type Storage = (NonNull<$t1> $(, NonNull<$ts>)*);
        }

        impl<$t1: 'static $(, $ts: 'static)*> Storage for (NonNull<$t1>, $(NonNull<$ts>,)*) {
            type Ptr = (*mut $t1 $(, *mut $ts)*);
            type Offsets = (usize $(, skip_first!($ts, usize))*);
            #[inline(always)]
            fn dangling() -> Self {
                (NonNull::dangling(), $(NonNull::<$ts>::dangling()),*)
            }

            #[inline(always)]
            fn as_ptr(self) -> Self::Ptr {
                let ($t1$(, $ts)*) = self;
                ($t1.as_ptr() $(, $ts.as_ptr())*)
            }

            unsafe fn alloc(capacity: usize) -> Self {
                let layout = Self::layout_for_capacity(capacity);
                let bytes = std::alloc::alloc(layout.layout);
                let (_ $(, $ts)*) = layout.offsets;
                (
                    NonNull::new_unchecked(bytes.cast::<$t1>())
                    $(, NonNull::new_unchecked(bytes.add($ts).cast::<$ts>()))*
                )
            }

            unsafe fn dealloc(&mut self, capacity: usize) {
                if capacity > 0 {
                    let layout = Self::layout_for_capacity(capacity);
                    std::alloc::dealloc(self.0.as_ptr().cast::<u8>(), layout.layout);
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
        }

        impl<$t1: 'static $(, $ts: 'static)*> ParamPtr for (*mut $t1, $(*mut $ts,)*) {
            type Param = ($t1, $($ts,)*);
            type Ref<'a> = (&'a $t1, $(&'a $ts,)*);
            type MutRef<'a> = (&'a mut $t1, $(&'a mut $ts,)*);
            type Slices<'a> = (&'a [$t1] $(, &'a [$ts])*);
            type SlicesMut<'a> = (&'a [$t1] $(, &'a [$ts])*);

            #[inline(always)]
            unsafe fn add(self, offset: usize) -> Self {
                let ($t1, $($ts),*) = self;
                ($t1.add(offset), $($ts.add(offset)),*)
            }

            #[inline(always)]
            unsafe fn copy_to(self, dst: Self, len: usize) {
                let ($t1, $($ts),*) = dst;
                let ($t1, $($ts),*) = self;
                $t1.copy_to($t1, len);
                $(
                    $ts.copy_to($ts, len);
                )*
            }

            #[inline(always)]
            unsafe fn copy_to_nonoverlapping(self, dst: Self, len: usize) {
                let ($t1, $($ts),*) = dst;
                let ($t1, $($ts),*) = self;
                $t1.copy_to_nonoverlapping($t1, len);
                $(
                    $ts.copy_to_nonoverlapping($ts, len);
                )*
            }

            #[inline(always)]
            unsafe fn as_slices<'a>(self, len: usize) -> Self::Slices<'a> {
                let ($t1, $($ts),*) = self;
                (
                    std::slice::from_raw_parts($t1, len)
                    $(
                        , std::slice::from_raw_parts($ts, len)
                    )*
                )
            }

            #[inline(always)]
            unsafe fn as_slices_mut<'a>(self, len: usize) -> Self::SlicesMut<'a> {
                let ($t1, $($ts),*) = self;
                (
                    std::slice::from_raw_parts_mut($t1, len)
                    $(
                        , std::slice::from_raw_parts_mut($ts, len)
                    )*
                )
            }

            #[inline(always)]
            unsafe fn as_ref<'a>(self) -> Self::Ref<'a> {
                let ($t1, $($ts),*) = self;
                (&*$t1 $(, &*$ts)*)
            }

            #[inline(always)]
            unsafe fn as_mut<'a>(self) -> Self::MutRef<'a> {
                let ($t1, $($ts),*) = self;
                (&mut *$t1 $(, &mut *$ts)*)
            }

            #[inline(always)]
            unsafe fn read(self) -> Self::Param {
                let ($t1, $($ts),*) = self;
                ($t1.read() $(, $ts.read())*)
            }

            #[inline(always)]
            unsafe fn write(self, _: Self::Param) {
                // let ($t1, $($ts),*) = value;
                // let ($t1, $($ts),*) = self;
                // $t1.write($t1);
                // $(
                //     $ts.write($ts);
                // )*
            }

            #[inline(always)]
            unsafe fn swap(self, other: Self) {
                let ($t1, $($ts),*) = other;
                let ($t1, $($ts),*) = self;
                std::ptr::swap($t1, $t1);
                $(std::ptr::swap($ts, $ts);)*
            }
        }
    }
}

impl_parallel_vec_param!(T1, T2);
impl_parallel_vec_param!(T1, T2, T3);
impl_parallel_vec_param!(T1, T2, T3, T4, T5);
impl_parallel_vec_param!(T1, T2, T3, T4, T5, T6);
impl_parallel_vec_param!(T1, T2, T3, T4, T5, T6, T7);
impl_parallel_vec_param!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_parallel_vec_param!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_parallel_vec_param!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_parallel_vec_param!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
impl_parallel_vec_param!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);