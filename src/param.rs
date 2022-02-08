use super::{ParallelVec, ParallelVecConversionError};
use std::alloc::Layout;
use std::ptr::NonNull;

/// This trait contains the basic operations for creating variadic
/// parallel vector implementations.
///
/// This trait is sealed and cannot be implemented outside of 
/// `parallel_vec`.
///
/// # Safety
/// None of the associated functions can panic.
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