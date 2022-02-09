use crate::{ParallelVec, ParallelVecParam};
use core::marker::PhantomData;

/// An iterator over immutable references to values in a [`ParallelSlice`].
///
/// See [`ParallelSlice::iter`].
///
/// [`ParallelSlice`]: crate::ParallelSlice
/// [`ParallelSlice::iter`]: crate::ParallelSlice::iter
pub struct Iter<'a, Param: ParallelVecParam> {
    pub(crate) base: Param::Ptr,
    pub(crate) idx: usize,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a Param>,
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

/// An iterator over mutable reference to values in a [`ParallelSliceMut`].
///
/// See [`ParallelSliceMut::iter_mut`].
///
/// [`ParallelSliceMut`]: crate::ParallelSliceMut
/// [`ParallelSliceMut::iter_mut`]: crate::ParallelSliceMut::iter_mut
pub struct IterMut<'a, Param: ParallelVecParam> {
    pub(crate) base: Param::Ptr,
    pub(crate) idx: usize,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a Param>,
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

/// An iterator over values from a [`ParallelVec`].
///
/// See [`ParallelVec::into_iter`].
///
/// [`ParallelVec`]: crate::ParallelVec
/// [`ParallelVec::iter_mut`]: crate::ParallelVec::into_iter
pub struct IntoIter<Param: ParallelVecParam> {
    pub(crate) vec: ParallelVec<Param>,
    pub(crate) idx: usize,
}

impl<Param: ParallelVecParam> Iterator for IntoIter<Param> {
    type Item = Param;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.idx >= self.vec.len {
                return None;
            }
            let base = self.vec.as_mut_ptrs();
            let ptr = Param::add(base, self.idx);
            let value = Param::read(ptr);
            self.idx += 1;
            Some(value)
        }
    }
}
