#![allow(non_snake_case)]
use std::ptr::NonNull;
use std::alloc::Layout;

pub struct ParallelVec<Param: ParallelVecParam> {
    len: usize,
    capacity: usize,
    storage: Param::Storage,
}

impl<Param: ParallelVecParam> ParallelVec<Param> {
    pub fn new() -> Self {
        Self {
            len: 0,
            capacity: 0,
            storage: Param::Storage::dangling(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        while self.len > 0 {
            self.pop();
        }
    }

    pub fn push(&mut self, value: <<Param::Storage as ParallelVecParamStorage>::Ptr as ParallelVecParamPtr>::Param) {
        unsafe {
            self.check_grow();
            self.storage.as_ptr().add(self.len).write(value);
            self.len += 1;
        }
    }

    pub fn pop(&mut self, value: Param) -> Option<<<Param::Storage as ParallelVecParamStorage>::Ptr as ParallelVecParamPtr>::Param> {
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

    pub fn swap_remove(&mut self, index: usize) -> <<Param::Storage as ParallelVecParamStorage>::Ptr as ParallelVecParamPtr>::Param {
        if index >= self.len {
            panic!("ParallelVec: Index out of bounds {}", index);
        }

        unsafe {
            let target_ptr = self.storage.as_ptr().add(index);
            let value = target_ptr.read();
            self.len -= 1;

            if self.len != index {
                let end = self.storage.as_ptr().add(self.len);
                target_ptr.copy_nonoverlapping(end, 1);
            }

            value
        }
    }

    fn check_grow(&mut self, growth: usize) {
        unsafe {
            let new_len = self.len + growth;
            if new_len > self.capacity {
                let capacity = new_len.next_power_of_two().max(4);
                let ptr = Param::Storage::alloc(capacity);
                self.storage.as_ptr().copy_nonoverlapping(ptr.as_ptr(), self.len);
                self.storage.dealloc(capacity);
                self.storage = ptr;
                self.capacity = capacity;
            }
        }
    }
}

pub trait ParallelVecParam {
    type Storage: ParallelVecParamStorage;
}

pub trait ParallelVecParamStorage : Copy {
    type Ptr: ParallelVecParamPtr;
    type Offsets;
    fn dangling() -> Self;
    fn as_ptr(self) -> Self::Ptr;
    unsafe fn alloc(capacity: usize) -> Self;
    unsafe fn dealloc(&mut self, capacity: usize);
    fn layout_for_capacity(capacity: usize) -> MemoryLayout<Self>;
}

pub trait ParallelVecParamPtr : Copy {
    type Param: ParallelVecParam;
    type Ref<'a>;
    type MutRef<'a>;
    unsafe fn add(self, offset: usize) -> Self;
    unsafe fn copy_nonoverlapping(self, dst: Self, size: usize);
    unsafe fn as_ref<'a>(self) -> Self::Ref<'a>;
    unsafe fn as_mut<'a>(self) -> Self::MutRef<'a>;
    unsafe fn read(self) -> Self::Param;
    unsafe fn write(self, value: Self::Param);
}

pub struct MemoryLayout<Param: ParallelVecParamStorage>  {
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
        impl<$t1: Sized $(, $ts: Sized)*> ParallelVecParam for ($t1 $(, $ts)*) {
            type Storage = (NonNull<$t1> $(, NonNull<$ts>)*);
        }

        impl<$t1: Sized $(, $ts: Sized)*> ParallelVecParamStorage for (NonNull<$t1>, $(NonNull<$ts>,)*) {
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

        impl<$t1: Sized $(, $ts: Sized)*> ParallelVecParamPtr for (*mut $t1, $(*mut $ts,)*) {
            type Param = ($t1, $($ts,)*);
            type Ref<'a> = (&'a $t1, $(&'a $ts,)*);
            type MutRef<'a> = (&'a mut $t1, $(&'a mut $ts,)*);

            #[inline(always)]
            unsafe fn add(self, offset: usize) -> Self {
                let ($t1, $($ts),*) = self;
                ($t1.add(offset), $($ts.add(offset)),*)
            }

            #[inline(always)]
            unsafe fn copy_nonoverlapping(self, dst: Self, len: usize) {
                let ($t1, $($ts),*) = self;
                let ($t1, $($ts),*) = dst;
                std::ptr::copy_nonoverlapping($t1, $t1, len);
                $(
                    std::ptr::copy_nonoverlapping($ts, $ts, len);
                )*
            }

            #[inline(always)]
            unsafe fn read(self) -> Self::Param {
                let ($t1, $($ts),*) = self;
                (std::ptr::read($t1) $(, std::ptr::read($ts))*)
            }

            #[inline(always)]
            unsafe fn write(self, value: Self::Param) {
                let ($t1, $($ts),*) = self;
                let ($t1, $($ts),*) = self;
                std::ptr::write($t1, $t1);
                $(
                    std::ptr::write($ts, $ts);
                )*
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