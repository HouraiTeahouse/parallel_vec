use super::{ParallelVec, ParallelVecConversionError};
use alloc::{
    alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout},
    vec::Vec,
};
use core::ptr::NonNull;

/// This trait contains the basic operations for creating variadic
/// parallel vector implementations.
///
/// This trait is sealed and cannot be implemented outside of
/// `parallel_vec`.
///
/// This trait has blanket implementations of all tuples of up
/// to size 12 of all types that are `'static`.
///
/// # Safety
/// None of the associated functions can panic.
pub unsafe trait ParallelParam: Sized + private::Sealed {
    /// A set of [`NonNull`] pointers of the parameter.
    /// This is the main backing storage pointers for [`ParallelVec`].
    type Storage: Copy + Eq;
    /// A set of pointers of the parameter.
    type Ptr: Copy;
    /// A set of memory offsets of the parameter.
    type Offsets;
    /// A set of immutable references of the parameter.
    type Ref<'a>;
    /// A set of mutable references of the parameter.
    type RefMut<'a>;
    /// A set of [`Vec<T>`]s of the parameter.
    type Vecs;
    /// A set of mutable slice references of the parameter.
    type Slices<'a>;
    /// A set of mutable slice references of the parameter.
    type SlicesMut<'a>;
    /// A set of iterators of immutable references of the parameter.
    type Iters<'a>;
    /// A set of iterators of mutable references of the parameter.
    type ItersMut<'a>;

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

    /// Realloc a buffer allocated from [`alloc`].
    ///
    /// # Safety
    /// `storage` must have been allocated from [`alloc`] or [`realloc`] alongside
    /// the provided `current_capacity`.
    ///
    /// [`alloc`]: Self::alloc
    unsafe fn realloc(
        storage: Self::Storage,
        current_capacity: usize,
        new_capacity: usize,
    ) -> Self::Storage;

    /// Deallocates a buffer allocated from [`alloc`].
    ///
    /// # Safety
    /// `storage` must have been allocated from [`alloc`] alongside
    /// the provided `capacity`.
    ///
    /// [`alloc`]: Self::alloc
    unsafe fn dealloc(storage: Self::Storage, capacity: usize);

    /// Gets the pointer at a given index.
    ///
    /// # Safety
    /// `storage` must be a set of valid pointers, and `idx` must be in
    /// bounds for the associated allocation.
    ///
    /// [`alloc`]: Self::alloc
    unsafe fn ptr_at(storage: Self::Storage, idx: usize) -> Self::Ptr {
        Self::add(Self::as_ptr(storage), idx)
    }

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

    /// Creates a set of iterators from slices.
    #[allow(clippy::needless_lifetimes)]
    fn iters<'a>(slices: Self::Slices<'a>) -> Self::Iters<'a>;

    /// Creates a set of iterators of mutable references from slices.
    #[allow(clippy::needless_lifetimes)]
    fn iters_mut<'a>(slices: Self::SlicesMut<'a>) -> Self::ItersMut<'a>;

    /// Reverses the order of elements in the slice, in place.
    fn reverse(ptr: Self::SlicesMut<'_>);

    /// Converts `ptr` into a set of immutable references.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer.
    unsafe fn as_ref<'a>(ptr: Self::Ptr) -> Self::Ref<'a>;

    /// Converts `ptr` into the storage type.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null pointer.
    unsafe fn as_storage(ptr: Self::Ptr) -> Self::Storage;

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
        unsafe impl<$t1: 'static $(, $ts: 'static)*> ParallelParam for ($t1 $(, $ts)*) {
            type Storage = (NonNull<$t1> $(, NonNull<$ts>)*);
            type Ref<'a> = (&'a $t1, $(&'a $ts,)*);
            type RefMut<'a> = (&'a mut $t1, $(&'a mut $ts,)*);
            type Slices<'a> = (&'a [$t1] $(, &'a [$ts])*);
            type SlicesMut<'a> = (&'a mut [$t1] $(, &'a mut [$ts])*);
            type Vecs = (Vec<$t1> $(, Vec<$ts>)*);
            type Ptr = (*mut $t1 $(, *mut $ts)*);
            type Offsets = (usize $(, skip_first!($ts, usize))*);
            type Iters<'a> = (core::slice::Iter<'a, $t1> $(, core::slice::Iter<'a, $ts>)*);
            type ItersMut<'a>= (core::slice::IterMut<'a, $t1> $(, core::slice::IterMut<'a, $ts>)*);

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
                debug_assert!(capacity != 0);
                let $t1 = if core::mem::size_of::<$t1>() != 0 {
                    let layout = Layout::array::<$t1>(capacity).unwrap();
                    let ptr = alloc(layout).cast::<$t1>();
                    NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout))
                } else {
                    NonNull::dangling()
                };
                $(
                    let $ts = if core::mem::size_of::<$ts>() != 0 {
                        let layout = Layout::array::<$ts>(capacity).unwrap();
                        let ptr = alloc(layout).cast::<$ts>();
                        NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout))
                    } else {
                        NonNull::dangling()
                    };
                )*
                ($t1 $(, $ts)*)
            }

            unsafe fn realloc(storage: Self::Storage, current_capacity: usize, new_capacity: usize) -> Self::Storage {
                if new_capacity == 0 {
                    Self::dealloc(storage, current_capacity);
                    return Self::dangling();
                }
                if current_capacity == 0 {
                    return Self::alloc(new_capacity);
                }
                let ($t1 $(, $ts)*) = storage;
                let $t1 = if core::mem::size_of::<$t1>() != 0 {
                    let layout = Layout::array::<$t1>(current_capacity).unwrap();
                    let new_size = core::mem::size_of::<$t1>().checked_mul(new_capacity).unwrap();
                    let ptr = realloc($t1.as_ptr().cast::<u8>(), layout, new_size).cast::<$t1>();
                    NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout))
                } else {
                    $t1
                };
                $(
                    let $ts = if core::mem::size_of::<$ts>() != 0 {
                        let layout = Layout::array::<$ts>(current_capacity).unwrap();
                        let new_size = core::mem::size_of::<$ts>().checked_mul(new_capacity).unwrap();
                        let ptr = realloc($ts.as_ptr().cast::<u8>(), layout, new_size).cast::<$ts>();
                        NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout))
                    } else {
                        $ts
                    };
                )*
                ($t1 $(, $ts)*)
            }

            unsafe fn dealloc(storage: Self::Storage, capacity: usize) {
                if capacity == 0 {
                    return;
                }
                let ($t1 $(, $ts)*) = storage;
                if core::mem::size_of::<$t1>() != 0 {
                    dealloc($t1.as_ptr().cast::<u8>(), Layout::array::<$t1>(capacity).unwrap_unchecked());
                }
                $(
                    if core::mem::size_of::<$ts>() != 0 {
                        dealloc($ts.as_ptr().cast::<u8>(), Layout::array::<$ts>(capacity).unwrap_unchecked());
                    }
                )*
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
                    core::slice::from_raw_parts($t1, len)
                    $(
                        , core::slice::from_raw_parts($ts, len)
                    )*
                )
            }

            #[inline(always)]
            unsafe fn as_slices_mut<'a>(ptr: Self::Ptr, len: usize) -> Self::SlicesMut<'a> {
                let ($t1, $($ts),*) = ptr;
                (
                    core::slice::from_raw_parts_mut($t1, len)
                    $(
                        , core::slice::from_raw_parts_mut($ts, len)
                    )*
                )
            }

            #[inline(always)]
            fn iters<'a>(slices: Self::Slices<'a>) -> Self::Iters<'a> {
                let ($t1, $($ts),*) = slices;
                ($t1.iter() $(, $ts.iter())*)
            }

            #[inline(always)]
            fn iters_mut<'a>(slices: Self::SlicesMut<'a>) -> Self::ItersMut<'a> {
                let ($t1, $($ts),*) = slices;
                ($t1.iter_mut() $(, $ts.iter_mut())*)
            }

            #[inline(always)]
            fn reverse<'a>(slices: Self::SlicesMut<'a>) {
                let ($t1, $($ts),*) = slices;
                $t1.reverse();
                $($ts.reverse();)*
            }

            #[inline(always)]
            unsafe fn as_storage<'a>(ptr: Self::Ptr) -> Self::Storage {
                let ($t1 $(, $ts)*) = ptr;
                (
                    NonNull::new_unchecked($t1)
                    $(, NonNull::new_unchecked($ts))*
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
                core::ptr::swap($t1, $v1);
                $(core::ptr::swap($ts, $vs);)*
            }

            #[inline(always)]
            unsafe fn drop(ptr: Self::Ptr) {
                let ($t1, $($ts),*) = ptr;
                core::ptr::drop_in_place($t1);
                $(core::ptr::drop_in_place($ts);)*
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
                let len = <($t1 $(, $ts)*) as ParallelParam>::get_vec_len(&vecs);
                if let Some(len) = len {
                    let parallel_vec = Self::with_capacity(len);
                    // SAFE: This is a move. Nothing should be dropped here.
                    unsafe {
                        let src = <($t1 $(, $ts)*) as ParallelParam>::get_vec_ptrs(&mut vecs);
                        let dst = <($t1 $(, $ts)*) as ParallelParam>::as_ptr(parallel_vec.storage);
                        <($t1 $(, $ts)*) as ParallelParam>::copy_to_nonoverlapping(src, dst, len);
                        core::mem::forget(vecs);
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
