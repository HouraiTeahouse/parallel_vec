#![allow(non_snake_case)]
#![feature(generic_associated_types)]
use std::marker::PhantomData;

pub mod iter;
mod param;

pub use param::*;
use iter::*;

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

pub enum ParallelVecConversionError {
    UnevenLengths,
}

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
