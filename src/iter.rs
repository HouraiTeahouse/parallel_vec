use crate::ParallelVecParam;
use std::marker::PhantomData;

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