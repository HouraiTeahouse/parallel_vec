use crate::ParallelParam;
use core::{
    iter::{DoubleEndedIterator, ExactSizeIterator},
    marker::PhantomData,
};

/// An iterator over immutable references to values in a [`ParallelSlice`].
///
/// See [`ParallelSlice::iter`].
///
/// [`ParallelSlice`]: crate::ParallelSlice
/// [`ParallelSlice::iter`]: crate::ParallelSlice::iter
pub struct Iter<'a, Param: ParallelParam> {
    pub(crate) ptr: Param::Ptr,
    pub(crate) remaining: usize,
    pub(crate) _marker: PhantomData<&'a Param>,
}

impl<'a, Param: ParallelParam> Iterator for Iter<'a, Param> {
    type Item = Param::Ref<'a>;
    fn next(&mut self) -> Option<Param::Ref<'a>> {
        unsafe {
            if self.remaining == 0 {
                return None;
            }
            let output = Param::as_ref(self.ptr);
            self.ptr = Param::add(self.ptr, 1);
            self.remaining -= 1;
            Some(output)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, Param: ParallelParam> ExactSizeIterator for Iter<'a, Param> {}

impl<'a, Param: ParallelParam> DoubleEndedIterator for Iter<'a, Param> {
    fn next_back(&mut self) -> Option<Param::Ref<'a>> {
        unsafe {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            let ptr = Param::add(self.ptr, self.remaining);
            Some(Param::as_ref(ptr))
        }
    }
}

/// An iterator over mutable reference to values in a [`ParallelSliceMut`].
///
/// See [`ParallelSliceMut::iter_mut`].
///
/// [`ParallelSliceMut`]: crate::ParallelSliceMut
/// [`ParallelSliceMut::iter_mut`]: crate::ParallelSliceMut::iter_mut
pub struct IterMut<'a, Param: ParallelParam> {
    pub(crate) ptr: Param::Ptr,
    pub(crate) remaining: usize,
    pub(crate) _marker: PhantomData<&'a Param>,
}

impl<'a, Param: ParallelParam> Iterator for IterMut<'a, Param> {
    type Item = Param::RefMut<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.remaining == 0 {
                return None;
            }
            let output = Param::as_mut(self.ptr);
            self.ptr = Param::add(self.ptr, 1);
            self.remaining -= 1;
            Some(output)
        }
    }
}

impl<'a, Param: ParallelParam> ExactSizeIterator for IterMut<'a, Param> {}

impl<'a, Param: ParallelParam> DoubleEndedIterator for IterMut<'a, Param> {
    fn next_back(&mut self) -> Option<Param::RefMut<'a>> {
        unsafe {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            let ptr = Param::add(self.ptr, self.remaining);
            Some(Param::as_mut(ptr))
        }
    }
}

/// An iterator over values from a [`ParallelVec`].
///
/// See [`ParallelVec::into_iter`].
///
/// [`ParallelVec`]: crate::ParallelVec
/// [`ParallelVec::iter_mut`]: crate::ParallelVec::into_iter
#[repr(C)]
pub struct IntoIter<Param: ParallelParam> {
    pub(crate) len: usize,
    pub(crate) storage: Param::Storage,
    pub(crate) capacity: usize,
    pub(crate) idx: usize,
}

impl<Param: ParallelParam> Iterator for IntoIter<Param> {
    type Item = Param;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.idx >= self.len {
                return None;
            }
            let ptr = Param::ptr_at(self.storage, self.idx);
            let value = Param::read(ptr);
            self.idx += 1;
            Some(value)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.idx;
        (remaining, Some(remaining))
    }
}

impl<Param: ParallelParam> ExactSizeIterator for IntoIter<Param> {}

impl<Param: ParallelParam> DoubleEndedIterator for IntoIter<Param> {
    fn next_back(&mut self) -> Option<Param> {
        unsafe {
            if self.len == 0 {
                return None;
            }
            self.len -= 1;
            let ptr = Param::ptr_at(self.storage, self.len);
            Some(Param::read(ptr))
        }
    }
}

impl<Param: ParallelParam> Drop for IntoIter<Param> {
    fn drop(&mut self) {
        unsafe {
            // Drop the unconsumed items.
            for idx in self.idx..self.len {
                Param::drop(Param::ptr_at(self.storage, idx));
            }
            Param::dealloc(self.storage, self.capacity);
        }
    }
}
