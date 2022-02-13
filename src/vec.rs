use crate::{assert_in_bounds, iter::IntoIter, out_of_bounds, ParallelParam, ParallelSliceMut};
use alloc::vec::Vec;
use core::{
    fmt::{Debug, Formatter},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};

/// A contiguously growable heterogenous array type.
///
/// This type stores the values [structure of arrays] layout. This layout
/// may improve cache utilizatoin in specific use cases.
///
/// Unlike a struct of `Vec`s, this type allocates memory for the all individual
/// fields simultaneously. This may minimize memory fragmentation and decrease
/// allocation pressure. It also only stores one length and capacity instead
/// of duplicating the values across multiple `Vec` fields.
///
/// [structures of arrays]: https://en.wikipedia.org/wiki/AoS_and_SoA#Structure_of_arrays
#[repr(C)]
pub struct ParallelVec<Param: ParallelParam> {
    pub(crate) len: usize,
    pub(crate) storage: Param::Storage,
    pub(crate) capacity: usize,
}

impl<Param: ParallelParam> ParallelVec<Param> {
    /// Constructs a new, empty `ParallelVec`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Constructs a new, empty [`ParallelVec`] with the specified capacity.  
    ///
    /// The vector will be able to hold exactly capacity elements without reallocating.
    /// If capacity is 0, the vector will not allocate.
    ///
    /// It is important to note that although the returned vector has the capacity specified,
    /// the vector will have a zero length.
    pub fn with_capacity(capacity: usize) -> Self {
        unsafe {
            Self {
                len: 0,
                capacity,
                storage: if capacity == 0 {
                    Param::dangling()
                } else {
                    Param::alloc(capacity)
                },
            }
        }
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

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than the vector’s current length, this has no effect.
    ///
    /// Note that this method has no effect on the allocated capacity of the vector.
    pub fn truncate(&mut self, len: usize) {
        if self.len <= len {
            return;
        }
        unsafe {
            self.drop_range(len, self.len);
            self.len = len;
        }
    }

    pub(crate) unsafe fn drop_range(&mut self, start: usize, end: usize) {
        let base = Param::as_ptr(self.storage);
        for idx in start..end {
            Param::drop(Param::add(base, idx));
        }
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length and
    /// the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        if min_capacity > self.capacity {
            return;
        }
        let capacity = core::cmp::max(self.len, min_capacity);
        let src = Param::as_ptr(self.storage);
        unsafe {
            let dst = Param::alloc(capacity);
            Param::copy_to_nonoverlapping(src, Param::as_ptr(dst), self.len);
            Param::dealloc(&mut self.storage, self.capacity);
            self.storage = dst;
        }
        self.capacity = capacity;
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
            let src = Param::as_ptr(other.storage);
            let dst = Param::ptr_at(self.storage, self.len);
            Param::copy_to_nonoverlapping(src, dst, other.len);
            self.len += other.len;
            // No need to drop from the other vec, data has been moved to
            // the current one. Just set the length here.
            other.len = 0;
        }
    }

    /// Appends an element to the back of a collection.
    pub fn push(&mut self, value: Param) {
        unsafe {
            self.reserve(1);
            let ptr = Param::ptr_at(self.storage, self.len);
            Param::write(ptr, value);
            self.len += 1;
        }
    }

    /// Removes the last element from the vector and returns it,
    /// or [`None`] if it is empty.
    ///
    /// [`None`]: Option::None
    pub fn pop(&mut self) -> Option<Param> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                let ptr = Param::ptr_at(self.storage, self.len - 1);
                let value = Param::read(ptr);
                self.len -= 1;
                Some(value)
            }
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.  
    ///
    /// This does not preserve ordering, but is `O(1)`. If you need to
    /// preserve the element order, use [`remove`] instead.
    ///
    /// [`remove`]: Self::remove
    pub fn swap_remove(&mut self, index: usize) -> Param {
        assert_in_bounds(index, self.len);

        unsafe {
            let target_ptr = Param::ptr_at(self.storage, index);
            let value = Param::read(target_ptr);
            self.len -= 1;

            if self.len != index {
                let end = Param::ptr_at(self.storage, self.len);
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
            out_of_bounds(index, self.len);
        }
        unsafe {
            // TODO: (Performance) In the case where we do grow, this can result in redundant copying.
            self.reserve(1);
            let ptr = Param::ptr_at(self.storage, index);
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
            let ptr = Param::ptr_at(self.storage, index);
            let value = Param::read(ptr);
            Param::copy_to(Param::add(ptr, 1), ptr, self.len - index - 1);
            self.len -= 1;
            Some(value)
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the
    /// given [`ParallelVec`]. The collection may reserve more space to avoid frequent
    /// reallocations. After calling reserve, capacity will be greater than or
    /// equal to `self.len() + additional`. Does nothing if capacity is already
    /// sufficient.
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            let new_capacity = self.len.checked_add(additional).expect("capacity overflow");
            if new_capacity > self.capacity {
                let capacity = new_capacity.next_power_of_two().max(4);
                assert!(capacity > self.len, "capacity overflow");
                let dst = Param::alloc(capacity);
                let src = self.as_mut_ptrs();
                Param::copy_to_nonoverlapping(src, Param::as_ptr(dst), self.len);
                Param::dealloc(&mut self.storage, self.capacity);
                self.storage = dst;
                self.capacity = capacity;
            }
        }
    }
}

impl<Param: ParallelParam + Copy> ParallelVec<Param> {
    /// Creates a [`ParallelVec`] by repeating `self` `n` times.
    pub fn repeat(&self, n: usize) -> ParallelVec<Param> {
        let mut new = ParallelVec::with_capacity(n * self.len);
        let mut dst = Param::as_ptr(new.storage);
        new.len = n * self.len;
        unsafe {
            let base = Param::as_ptr(self.storage);
            for _ in 0..n {
                for idx in 0..self.len {
                    let value = Param::read(Param::add(base, idx));
                    Param::write(dst, value);
                    dst = Param::add(dst, 1);
                }
            }
        }
        new
    }
}

impl<Param: ParallelParam> Drop for ParallelVec<Param> {
    fn drop(&mut self) {
        let end = self.len;
        // Set len to 0 first in case one of the Drop impls panics
        self.len = 0;
        unsafe {
            self.drop_range(0, end);
            Param::dealloc(&mut self.storage, self.capacity);
        }
    }
}

impl<Param: ParallelParam> From<Vec<Param>> for ParallelVec<Param> {
    fn from(value: Vec<Param>) -> Self {
        Self::from_iter(value.into_iter())
    }
}

impl<'a, Param: ParallelParam> PartialEq for ParallelVec<Param>
where
    Param: 'a,
    Param::Ref<'a>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        if self.storage == other.storage {
            // Pointing to the same storage. Shortcut out.
            return true;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a.eq(&b))
    }
}

impl<'a, Param: ParallelParam> Eq for ParallelVec<Param>
where
    Param: 'a,
    Param::Ref<'a>: Eq,
{
}

impl<'a, Param: ParallelParam> Debug for ParallelVec<Param>
where
    Param: 'a,
    Param::Ref<'a>: Debug,
{
    fn fmt(&self, fmt: &mut Formatter<'_>) -> core::fmt::Result {
        fmt.write_str("ParallelVec")?;
        fmt.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, Param: ParallelParam> Hash for ParallelVec<Param>
where
    Param: 'a,
    Param::Ref<'a>: Hash,
{
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        self.deref().hash(hasher);
    }
}

impl<Param: ParallelParam> FromIterator<Param> for ParallelVec<Param> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Param>,
    {
        let iter = iter.into_iter();
        let (min, _) = iter.size_hint();
        let mut parallel_vec = Self::with_capacity(min);
        for item in iter {
            parallel_vec.push(item);
        }
        parallel_vec
    }
}

impl<Param: ParallelParam> IntoIterator for ParallelVec<Param> {
    type Item = Param;
    type IntoIter = IntoIter<Param>;
    fn into_iter(self) -> Self::IntoIter {
        let iter = IntoIter {
            storage: self.storage,
            capacity: self.capacity,
            len: self.len,
            idx: 0,
        };
        core::mem::forget(self);
        iter
    }
}

impl<Param: ParallelParam> Extend<Param> for ParallelVec<Param> {
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

impl<Param: ParallelParam + Clone> Clone for ParallelVec<Param> {
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

impl<Param: ParallelParam> Default for ParallelVec<Param> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Param: ParallelParam> Deref for ParallelVec<Param> {
    type Target = ParallelSliceMut<'static, Param>;
    fn deref(&self) -> &Self::Target {
        // SAFE: Both ParallelVec and ParallelSliceMut have the same
        // layout in memory due to #[repr(C)]
        unsafe {
            let ptr: *const Self = self;
            &*(ptr.cast::<Self::Target>())
        }
    }
}

impl<Param: ParallelParam> DerefMut for ParallelVec<Param> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFE: Both ParallelVec and ParallelSliceMut have the same
        // layout in memory due to #[repr(C)]
        unsafe {
            let ptr: *mut Self = self;
            &mut *(ptr.cast::<Self::Target>())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ParallelVec;
    use std::convert::From;
    use std::rc::Rc;
    use std::vec::Vec;

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    struct ZST;

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    struct ZST2;

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

    #[test]
    fn test_new() {
        let src: ParallelVec<(i32, i32, u64)> = ParallelVec::new();
        assert_eq!(src.len(), 0);
        assert_eq!(src.capacity(), 0);
        assert!(src.is_empty());
    }

    #[test]
    fn test_default() {
        let src: ParallelVec<(i32, i32, u64)> = Default::default();
        assert_eq!(src.len(), 0);
        assert_eq!(src.capacity(), 0);
        assert!(src.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let src: ParallelVec<(i32, i32, u64)> = ParallelVec::with_capacity(1000);
        assert_eq!(src.len(), 0);
        assert!(src.capacity() >= 1000);
        assert!(src.is_empty());
    }

    #[test]
    fn test_reserve() {
        let mut src = ParallelVec::new();
        src.push((0, 0, 0, 0));
        assert_eq!(src.len(), 1);
        assert!(src.capacity() >= 1);
        src.reserve(10);
        assert_eq!(src.len(), 1);
        assert!(src.capacity() >= 10);
        src.reserve(100);
        assert_eq!(src.len(), 1);
        assert!(src.capacity() >= 100);
        src.reserve(1000);
        assert_eq!(src.len(), 1);
        assert!(src.capacity() >= 1000);
        src.reserve(100000);
        assert_eq!(src.len(), 1);
        assert!(src.capacity() >= 10000);
    }

    #[test]
    fn test_clone() {
        let mut src = ParallelVec::new();
        src.push((1.0, 2.0));
        src.push((3.0, 4.0));

        let dst = src.clone();
        assert_eq!(dst.len(), 2);
        assert_eq!(dst.index(0), (&1.0, &2.0));
        assert_eq!(dst.index(1), (&3.0, &4.0));
    }

    #[test]
    fn test_works_with_zsts() {
        let mut src = ParallelVec::new();
        src.push((1, ZST, 20u64, ZST2));
        src.push((1, ZST, 21u64, ZST2));
        src.push((1, ZST, 22u64, ZST2));
        src.push((1, ZST, 23u64, ZST2));
        assert_eq!(src.index(0), (&1, &ZST, &20u64, &ZST2));
        assert_eq!(src.index(1), (&1, &ZST, &21u64, &ZST2));
        assert_eq!(src.index(2), (&1, &ZST, &22u64, &ZST2));
        assert_eq!(src.index(3), (&1, &ZST, &23u64, &ZST2));
        assert_eq!(src.len(), 4);
    }

    #[test]
    fn test_push() {
        let mut src = ParallelVec::new();
        src.push((1, 2));
        assert_eq!(src.index(0), (&1, &2));
        assert_eq!(src.len(), 1);
    }

    #[test]
    fn test_pop() {
        let mut src = ParallelVec::new();
        src.push((1, 2));
        src.push((3, 4));
        src.push((5, 6));
        src.push((7, 8));
        assert_eq!(src.len(), 4);
        let value = src.pop();
        assert_eq!(value, Some((7, 8)));
        assert_eq!(src.len(), 3);
        let value = src.pop();
        assert_eq!(value, Some((5, 6)));
        assert_eq!(src.len(), 2);
        let value = src.pop();
        assert_eq!(value, Some((3, 4)));
        assert_eq!(src.len(), 1);
        let value = src.pop();
        assert_eq!(value, Some((1, 2)));
        assert_eq!(src.len(), 0);
        let value = src.pop();
        assert_eq!(value, None);
        assert_eq!(src.len(), 0);
        let value = src.pop();
        assert_eq!(value, None);
        assert_eq!(src.len(), 0);
    }

    #[test]
    fn test_insert() {
        let mut src = ParallelVec::new();
        src.insert(0, (1, 2));
        src.insert(0, (3, 4));
        src.insert(1, (4, 5));
        assert_eq!(src.index(0), (&3, &4));
        assert_eq!(src.index(1), (&4, &5));
        assert_eq!(src.index(2), (&1, &2));
    }

    #[test]
    #[should_panic]
    fn test_insert_panics() {
        let mut src = ParallelVec::new();
        src.insert(0, (1, 2));
        src.insert(0, (3, 4));
        src.insert(1, (4, 5));
        src.insert(20, (4, 5));
    }

    #[test]
    fn test_remove() {
        let mut src = ParallelVec::new();
        src.push((1, 2));
        src.push((3, 4));
        assert_eq!(src.remove(1), Some((3, 4)));
        assert_eq!(src.remove(0), Some((1, 2)));
        assert_eq!(src.len(), 0);
        assert_eq!(src.remove(5), None);
    }

    #[test]
    fn test_iter() {
        let mut src = ParallelVec::new();
        src.push((1, 2));
        src.push((3, 4));
        src.push((5, 6));
        src.push((7, 8));
        let mut iter = src.iter();
        assert_eq!(iter.next(), Some((&1, &2)));
        assert_eq!(iter.next(), Some((&3, &4)));
        assert_eq!(iter.next(), Some((&5, &6)));
        assert_eq!(iter.next(), Some((&7, &8)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        // Shouldn't have removed any of them
        assert_eq!(src.len(), 4);
    }

    #[test]
    fn test_iters() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let (mut a, mut b) = src.iters();
        assert_eq!(a.next(), Some(&1));
        assert_eq!(a.next(), Some(&3));
        assert_eq!(a.next(), Some(&5));
        assert_eq!(a.next(), Some(&7));
        assert_eq!(a.next(), Some(&9));
        assert_eq!(a.next(), Some(&11));
        assert_eq!(a.next(), None);
        assert_eq!(a.next(), None);
        assert_eq!(b.next(), Some(&2));
        assert_eq!(b.next(), Some(&4));
        assert_eq!(b.next(), Some(&6));
        assert_eq!(b.next(), Some(&8));
        assert_eq!(b.next(), Some(&10));
        assert_eq!(b.next(), Some(&12));
        assert_eq!(b.next(), None);
        assert_eq!(b.next(), None);
    }

    #[test]
    fn test_iter_mut() {
        let mut src = ParallelVec::new();
        src.push((1, 2));
        src.push((3, 4));
        src.push((5, 6));
        src.push((7, 8));
        let mut iter = src.iter_mut();
        assert_eq!(iter.next(), Some((&mut 1, &mut 2)));
        assert_eq!(iter.next(), Some((&mut 3, &mut 4)));
        assert_eq!(iter.next(), Some((&mut 5, &mut 6)));
        assert_eq!(iter.next(), Some((&mut 7, &mut 8)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        // Shouldn't have removed any of them
        assert_eq!(src.len(), 4);
    }

    #[test]
    fn test_shrink_to() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.reserve(1000);
        assert_eq!(src.len(), 4);
        assert!(src.capacity() >= 1000);
        src.shrink_to(200);
        assert_eq!(src.len(), 4);
        assert!(src.capacity() <= 200);
        src.shrink_to(100);
        assert_eq!(src.len(), 4);
        assert!(src.capacity() <= 100);
        src.shrink_to(10);
        assert_eq!(src.len(), 4);
        assert!(src.capacity() <= 10);
        src.shrink_to(1);
        assert_eq!(src.len(), 4);
        assert!(src.capacity() >= 4);
        let (a, b) = src.as_slices();
        assert_eq!(a, &[1, 3, 5, 7]);
        assert_eq!(b, &[2, 4, 6, 8]);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.reserve(1000);
        assert_eq!(src.len(), 4);
        assert!(src.capacity() >= 1000);
        src.shrink_to_fit();
        assert_eq!(src.len(), 4);
        assert!(src.capacity() <= src.len());
        let (a, b) = src.as_slices();
        assert_eq!(a, &[1, 3, 5, 7]);
        assert_eq!(b, &[2, 4, 6, 8]);
    }

    #[test]
    fn test_truncate() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        {
            let (a, b) = src.as_slices();
            assert_eq!(a, &[1, 3, 5, 7]);
            assert_eq!(b, &[2, 4, 6, 8]);
        }
        src.truncate(2);
        {
            let (a, b) = src.as_slices();
            assert_eq!(a, &[1, 3]);
            assert_eq!(b, &[2, 4]);
            assert_eq!(src.len(), 2);
            assert!(src.capacity() >= 4);
        }
    }

    #[test]
    fn test_truncate_drops() {
        let rc = Rc::new(0);
        let mut src = ParallelVec::new();
        src.extend(vec![
            (rc.clone(), rc.clone()),
            (rc.clone(), rc.clone()),
            (rc.clone(), rc.clone()),
            (rc.clone(), rc.clone()),
        ]);
        assert_eq!(Rc::strong_count(&rc), 9);
        src.truncate(1);
        assert_eq!(Rc::strong_count(&rc), 3);
    }

    #[test]
    fn test_reverse() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        {
            let (a, b) = src.as_slices();
            assert_eq!(a, &[1, 3, 5, 7]);
            assert_eq!(b, &[2, 4, 6, 8]);
        }
        src.reverse();
        {
            let (a, b) = src.as_slices();
            assert_eq!(a, &[7, 5, 3, 1]);
            assert_eq!(b, &[8, 6, 4, 2]);
        }
    }

    #[test]
    fn test_clear() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.clear();
        assert_eq!(src.len(), 0);
        assert!(src.capacity() > 0);
        let (a, b) = src.as_slices();
        assert_eq!(a, &[]);
        assert_eq!(b, &[]);
    }

    #[test]
    fn test_repeat() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let repeated = src.repeat(3);
        let (a, b) = repeated.as_slices();
        assert_eq!(a, &[1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7]);
        assert_eq!(b, &[2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8]);
        let (a, b) = src.as_slices();
        assert_eq!(src.len(), 4);
        assert_eq!(a, &[1, 3, 5, 7]);
        assert_eq!(b, &[2, 4, 6, 8]);
        assert_eq!(repeated.len(), 12);
    }

    #[test]
    fn test_eq() {
        let a = ParallelVec::from(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let b = ParallelVec::from(vec![(1, 2), (3, 4), (9, 6), (7, 8)]);
        let c = ParallelVec::from(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert!(a == a);
        assert!(a != b);
        assert!(a == c);
        assert!(b != a);
        assert!(b == b);
        assert!(b != c);
        assert!(c == a);
        assert!(c != b);
        assert!(c == c);
    }

    #[test]
    fn test_extend() {
        let mut src = ParallelVec::new();
        src.push((0, 0));
        src.push((-1, -1));
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.push((10, 10));
        src.push((11, 11));
        let (a, b) = src.as_slices();
        assert_eq!(a, &[0, -1, 1, 3, 5, 7, 10, 11]);
        assert_eq!(b, &[0, -1, 2, 4, 6, 8, 10, 11]);
        assert_eq!(src.len(), 8);
    }

    #[test]
    fn test_swap_remove() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.swap_remove(1);
        let (a, b) = src.as_slices();
        assert_eq!(a, &[1, 7, 5]);
        assert_eq!(b, &[2, 8, 6]);
        assert_eq!(src.len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_swap_remove_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.swap_remove(12);
    }

    #[test]
    fn test_swap() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.swap(1, 2);
        let (a, b) = src.as_slices();
        assert_eq!(a, &[1, 5, 3, 7]);
        assert_eq!(b, &[2, 6, 4, 8]);
        assert_eq!(src.len(), 4);
        src.swap(0, 3);
        let (a, b) = src.as_slices();
        assert_eq!(a, &[7, 5, 3, 1]);
        assert_eq!(b, &[8, 6, 4, 2]);
        assert_eq!(src.len(), 4);
        src.swap(3, 0);
        let (a, b) = src.as_slices();
        assert_eq!(a, &[1, 5, 3, 7]);
        assert_eq!(b, &[2, 6, 4, 8]);
        assert_eq!(src.len(), 4);
    }

    #[test]
    #[should_panic]
    fn test_swap_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.swap(20, 2);
    }

    #[test]
    fn test_swap_with() {
        let mut src_a = ParallelVec::new();
        src_a.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let mut src_b = ParallelVec::new();
        src_b.extend(vec![(9, 9), (2, 2), (4, 4), (7, 7)]);
        let (a, b) = src_a.as_slices();
        assert_eq!(a, &[1, 3, 5, 7]);
        assert_eq!(b, &[2, 4, 6, 8]);
        assert_eq!(src_a.len(), 4);
        let (a, b) = src_b.as_slices();
        assert_eq!(a, &[9, 2, 4, 7]);
        assert_eq!(b, &[9, 2, 4, 7]);
        assert_eq!(src_b.len(), 4);
        src_a.swap_with(&mut src_b);
        let (a, b) = src_a.as_slices();
        assert_eq!(a, &[9, 2, 4, 7]);
        assert_eq!(b, &[9, 2, 4, 7]);
        assert_eq!(src_a.len(), 4);
        let (a, b) = src_b.as_slices();
        assert_eq!(a, &[1, 3, 5, 7]);
        assert_eq!(b, &[2, 4, 6, 8]);
        assert_eq!(src_b.len(), 4);
    }

    #[test]
    fn test_drop() {
        let rc = Rc::new(0);
        let mut src_a = ParallelVec::new();
        src_a.extend(vec![
            (rc.clone(), rc.clone()),
            (rc.clone(), rc.clone()),
            (rc.clone(), rc.clone()),
            (rc.clone(), rc.clone()),
        ]);
        assert_eq!(Rc::strong_count(&rc), 9);
        core::mem::drop(src_a);
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn test_append() {
        let mut src_a = ParallelVec::new();
        src_a.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let mut src_b = ParallelVec::new();
        src_b.extend(vec![(9, 9), (2, 2), (4, 4), (7, 7)]);
        src_a.append(&mut src_b);
        let (a, b) = src_a.as_slices();
        assert_eq!(a, &[1, 3, 5, 7, 9, 2, 4, 7]);
        assert_eq!(b, &[2, 4, 6, 8, 9, 2, 4, 7]);
        assert_eq!(src_a.len(), 8);
        let (a, b) = src_b.as_slices();
        assert_eq!(a, &[]);
        assert_eq!(b, &[]);
        assert_eq!(src_b.len(), 0);
    }

    #[test]
    #[should_panic]
    fn test_swap_with_panics() {
        let mut src_a = ParallelVec::new();
        src_a.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let mut src_b = ParallelVec::new();
        src_b.extend(vec![(9, 9), (2, 2), (7, 7)]);
        src_a.swap_with(&mut src_b);
    }

    #[test]
    fn test_set() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.set(2, (0, 0));
        let (a, b) = src.as_slices();
        assert_eq!(a, &[1, 3, 0, 7]);
        assert_eq!(b, &[2, 4, 0, 8]);
        assert_eq!(src.len(), 4);
    }

    #[test]
    #[should_panic]
    fn test_set_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.set(10, (0, 0));
    }

    #[test]
    fn test_get_single() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.get(0), Some((&1, &2)));
        assert_eq!(src.get(1), Some((&3, &4)));
        assert_eq!(src.get(2), Some((&5, &6)));
        assert_eq!(src.get(3), Some((&7, &8)));
        assert_eq!(src.get(4), None);
        assert_eq!(src.get(5), None);
    }

    #[test]
    fn test_get_mut_single() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.get_mut(0), Some((&mut 1, &mut 2)));
        assert_eq!(src.get_mut(1), Some((&mut 3, &mut 4)));
        assert_eq!(src.get_mut(2), Some((&mut 5, &mut 6)));
        assert_eq!(src.get_mut(3), Some((&mut 7, &mut 8)));
        assert_eq!(src.get_mut(4), None);
        assert_eq!(src.get_mut(5), None);
    }

    #[test]
    fn test_first() {
        let mut src = ParallelVec::new();
        assert_eq!(src.first(), None);
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.first(), Some((&1, &2)));
    }

    #[test]
    fn test_first_mut() {
        let mut src = ParallelVec::new();
        assert_eq!(src.first_mut(), None);
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.first_mut(), Some((&mut 1, &mut 2)));
    }

    #[test]
    fn test_last() {
        let mut src = ParallelVec::new();
        assert_eq!(src.last(), None);
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.last(), Some((&7, &8)));
    }

    #[test]
    fn test_last_mut() {
        let mut src = ParallelVec::new();
        assert_eq!(src.last_mut(), None);
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.last_mut(), Some((&mut 7, &mut 8)));
    }

    #[test]
    fn test_get_slice() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let slice = src.get(1..3);
        assert!(slice.is_some());
        let slice = slice.unwrap();
        assert_eq!(slice.len(), 2);
        let (a, b) = slice.as_slices();
        assert_eq!(a, &[3, 5]);
        assert_eq!(b, &[4, 6]);
        let slice = src.get(1..5);
        assert!(slice.is_none());
    }

    #[test]
    fn test_get_mut_slice() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let slice = src.get_mut(1..3);
        assert!(slice.is_some());
        let mut slice = slice.unwrap();
        assert_eq!(slice.len(), 2);
        let (a, b) = slice.as_slices_mut();
        assert_eq!(a, &mut [3, 5]);
        assert_eq!(b, &mut [4, 6]);
        let slice = src.get(1..5);
        assert!(slice.is_none());
    }

    #[test]
    fn test_index_single() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.index(0), (&1, &2));
        assert_eq!(src.index(1), (&3, &4));
        assert_eq!(src.index(2), (&5, &6));
        assert_eq!(src.index(3), (&7, &8));
    }

    #[test]
    #[should_panic]
    fn test_index_single_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.index(4);
    }

    #[test]
    fn test_index_mut_single() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        assert_eq!(src.index_mut(0), (&mut 1, &mut 2));
        assert_eq!(src.index_mut(1), (&mut 3, &mut 4));
        assert_eq!(src.index_mut(2), (&mut 5, &mut 6));
        assert_eq!(src.index_mut(3), (&mut 7, &mut 8));
    }

    #[test]
    #[should_panic]
    fn test_index_mut_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.index_mut(4);
    }

    #[test]
    fn test_index_slice() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let slice = src.index(1..3);
        let (a, b) = slice.as_slices();
        assert_eq!(a, &[3, 5]);
        assert_eq!(b, &[4, 6]);
    }

    #[test]
    #[should_panic]
    fn test_index_slice_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.index(1..9);
    }

    #[test]
    fn test_index_mut_slice() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let slice = src.index_mut(1..3);
        let (a, b) = slice.as_slices();
        assert_eq!(a, &mut [3, 5]);
        assert_eq!(b, &mut [4, 6]);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_slice_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        src.index_mut(1..9);
    }

    #[test]
    fn test_into_iter() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let vec: Vec<_> = src.into_iter().collect();
        assert_eq!(vec, vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
    }

    #[test]
    fn test_slice_is_empty() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8)]);
        let slice = src.index(1..1);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_slice_first() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let slice = src.index(1..4);
        assert_eq!(slice.first(), Some((&3, &4)));
    }

    #[test]
    fn test_slice_last() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let slice = src.index(1..4);
        assert_eq!(slice.last(), Some((&7, &8)));
    }

    #[test]
    fn test_slice_get() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let slice = src.index(1..4);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.get(0), Some((&3, &4)));
        assert_eq!(slice.get(1), Some((&5, &6)));
        assert_eq!(slice.get(2), Some((&7, &8)));
        assert_eq!(slice.get(3), None);
        assert_eq!(slice.get(4), None);
    }

    #[test]
    fn test_slice_index() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let slice = src.index(1..4);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.index(0), (&3, &4));
        assert_eq!(slice.index(1), (&5, &6));
        assert_eq!(slice.index(2), (&7, &8));
    }

    #[test]
    #[should_panic]
    fn test_slice_index_panics() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let slice = src.index(1..4);
        slice.index(3);
    }

    #[test]
    fn test_slice_iter() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let slice = src.index(1..4);
        let mut iter = slice.iter();
        assert_eq!(iter.next(), Some((&3, &4)));
        assert_eq!(iter.next(), Some((&5, &6)));
        assert_eq!(iter.next(), Some((&7, &8)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_slice_iters() {
        let mut src = ParallelVec::new();
        src.extend(vec![(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]);
        let slice = src.index(1..4);
        let (mut a, mut b) = slice.iters();
        assert_eq!(a.next(), Some(&3));
        assert_eq!(a.next(), Some(&5));
        assert_eq!(a.next(), Some(&7));
        assert_eq!(a.next(), None);
        assert_eq!(a.next(), None);
        assert_eq!(b.next(), Some(&4));
        assert_eq!(b.next(), Some(&6));
        assert_eq!(b.next(), Some(&8));
        assert_eq!(b.next(), None);
        assert_eq!(b.next(), None);
    }

    #[test]
    #[should_panic]
    fn reserve_overflow_negative() {
        let mut v = ParallelVec::new();
        (1..8).for_each(|i| v.push((i, i)));
        v.reserve(usize::MAX - 8);
    }

    #[test]
    #[should_panic]
    fn reserve_overflow() {
        let mut v = ParallelVec::new();
        (1..8).for_each(|i| v.push((i, i)));
        v.reserve(usize::MAX);
    }
}
