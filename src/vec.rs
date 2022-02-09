use crate::iter::IntoIter;
use crate::{assert_in_bounds, out_of_bounds};
use crate::{ParallelSliceMut, ParallelVecParam};
use core::ops::{Deref, DerefMut};

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
pub struct ParallelVec<Param: ParallelVecParam> {
    pub(crate) len: usize,
    pub(crate) storage: Param::Storage,
    pub(crate) capacity: usize,
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
            let capacity = core::cmp::max(self.len, min_capacity);
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
            let src = Param::as_ptr(other.storage);
            let dst = Param::ptr_at(self.storage, self.len);
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
                let ptr = Param::ptr_at(self.storage, self.len);
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
}

impl<Param: ParallelVecParam + Copy> ParallelVec<Param> {
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

impl<Param: ParallelVecParam> FromIterator<Param> for ParallelVec<Param> {
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

impl<Param: ParallelVecParam> IntoIterator for ParallelVec<Param> {
    type Item = Param;
    type IntoIter = IntoIter<Param>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { vec: self, idx: 0 }
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

impl<Param: ParallelVecParam> Deref for ParallelVec<Param> {
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

impl<Param: ParallelVecParam> DerefMut for ParallelVec<Param> {
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
