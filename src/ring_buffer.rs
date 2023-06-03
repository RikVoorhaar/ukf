use arrayvec::ArrayVec;
use std::{
    cmp::min,
    ops::{Index, IndexMut},
};

#[derive(Clone)]
pub struct RingBuffer<T: Clone, const N: usize> {
    buffer: ArrayVec<T, N>,
    start: usize,
    end: usize,
    capacity: usize,
    size: usize,
}

impl<T: Clone, const N: usize> RingBuffer<T, N> {
    pub fn new() -> Self {
        RingBuffer {
            buffer: ArrayVec::<T, N>::new(),
            start: 0,
            end: 0,
            capacity: N,
            size: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        if self.size == self.capacity {
            self.start = (self.start + 1) % self.capacity;
        }
        if self.end < self.buffer.len() {
            self.buffer[self.end] = value;
        } else {
            self.buffer.push(value);
        }
        self.end = (self.end + 1) % self.capacity;
        self.size = min(self.capacity, self.size + 1)
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            None
        } else {
            self.size -= 1;
            self.end = modular_sub(self.end, 1, self.capacity);
            let value = self.buffer[self.end].clone();
            Some(value)
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        if self.start < self.end {
            self.buffer[self.start..self.end].iter().chain([].iter())
        } else if self.size > 0 {
            let (left_half, right_half) = self.buffer.split_at(self.start);
            let first_half = right_half;
            let second_half = &left_half[..self.end];
            first_half.iter().chain(second_half)
        } else {
            [].iter().chain(&[])
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        if self.start < self.end {
            self.buffer[self.start..self.end]
                .iter_mut()
                .chain([].iter_mut())
        } else if self.size > 0 {
            let (left_half, right_half) = self.buffer.split_at_mut(self.start);
            let first_half = right_half;
            let second_half = &mut left_half[..self.end];
            first_half.iter_mut().chain(second_half)
        } else {
            [].iter_mut().chain(&mut [])
        }
    }
}

impl<T: Clone, const N: usize, const M: usize> From<[T; M]> for RingBuffer<T, N> {
    fn from(array: [T; M]) -> Self {
        let mut buffer = RingBuffer::<T, N>::new();
        let skip = if M > N { M - N } else { 0 };

        array.into_iter().skip(skip).for_each(|v| buffer.push(v));
        buffer
    }
}

impl<T: Clone, const N: usize> From<Vec<T>> for RingBuffer<T, N> {
    fn from(vec: Vec<T>) -> Self {
        let mut buffer = RingBuffer::<T, N>::new();
        let skip = if vec.len() > N { vec.len() - N } else { 0 };

        vec.into_iter().skip(skip).for_each(|v| buffer.push(v));

        buffer
    }
}

impl<T: Clone, const N: usize, const M: usize> From<ArrayVec<T, M>> for RingBuffer<T, N> {
    fn from(arrayvec: ArrayVec<T, M>) -> Self {
        let mut buffer = RingBuffer::<T, N>::new();
        let skip = if M > N { M - N } else { 0 };

        arrayvec.into_iter().skip(skip).for_each(|v| buffer.push(v));
        buffer
    }
}

impl<T: Clone, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const N: usize> Index<usize> for RingBuffer<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.size {
            panic!("Index out of bounds")
        }
        let i = (self.start + index) % self.capacity;
        &self.buffer[i]
    }
}

impl<T: Clone, const N: usize> IndexMut<usize> for RingBuffer<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.size {
            panic!("Index out of bounds")
        }
        let i = (self.start + index) % self.capacity;
        &mut self.buffer[i]
    }
}

fn modular_sub<T>(a: T, b: T, modulus: T) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Rem<Output = T> + Copy,
{
    ((a + modulus) - b) % modulus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_pull() {
        let mut buffer = RingBuffer::<i32, 3>::new();
        assert_eq!(buffer.size, 0);
        buffer.push(1);
        assert_eq!(buffer.size, 1);
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), None);
        buffer.push(1);
        buffer.push(2);
        assert_eq!(buffer.size, 2);
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), None);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert_eq!(buffer.size, 3);
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), None);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4);
        assert_eq!(buffer.pop(), Some(4));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_ring_buffer_iter() {
        let mut buffer = RingBuffer::<f32, 4>::new();
        let empty: Vec<&f32> = buffer.iter().collect();
        assert_eq!(empty, [] as [&f32; 0]);

        buffer.push(1.0);
        buffer.push(2.0);
        let iter = buffer.iter();
        let vec: Vec<&f32> = iter.collect();
        assert_eq!(vec, vec![&1.0, &2.0]);

        buffer.push(3.0);
        buffer.push(4.0);
        buffer.push(5.0);
        let iter = buffer.iter();
        let vec: Vec<&f32> = iter.collect();
        assert_eq!(vec, vec![&2.0, &3.0, &4.0, &5.0]);

        for _ in 0..buffer.capacity {
            buffer.pop();
        }
        let empty: Vec<&f32> = buffer.iter().collect();
        assert_eq!(empty, [] as [&f32; 0]);
    }

    #[test]
    fn test_ring_buffer_iter_mut() {
        let mut buffer = RingBuffer::<f32, 4>::new();
        let empty: Vec<&mut f32> = buffer.iter_mut().collect::<Vec<&mut f32>>();
        assert_eq!(empty, [] as [&mut f32; 0]);

        buffer.push(1.0);
        buffer.push(2.0);
        let iter = buffer.iter_mut();
        for value in iter {
            *value *= 2.0;
        }
        let vec: Vec<&f32> = buffer.iter().collect();
        assert_eq!(vec, vec![&2.0, &4.0]);

        buffer.push(3.0);
        buffer.push(4.0);
        buffer.push(5.0);
        // buffer is now [5.0, 4.0, 3.0, 4.0]
        buffer.iter_mut().for_each(|x| *x *= 3.0);
        let vec: Vec<&f32> = buffer.iter().collect();
        assert_eq!(vec, vec![&12.0, &9.0, &12.0, &15.0]);

        for _ in 0..buffer.capacity {
            buffer.pop();
        }
        let empty: Vec<&mut f32> = buffer.iter_mut().collect::<Vec<&mut f32>>();
        assert_eq!(empty, [] as [&mut f32; 0]);
    }

    #[test]
    fn test_ring_buffer_index() {
        let mut buffer = RingBuffer::<f32, 4>::new();
        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);
        assert_eq!(buffer[0], 1.0);

        buffer[1] = 4.0;
        assert_eq!(buffer[1], 4.0);
        assert_eq!(buffer.iter().collect::<Vec<&f32>>(), vec![&1.0, &4.0, &3.0]);

        buffer.push(1.0);
        assert_eq!(buffer[3], 1.0);
        buffer.push(2.0);
        buffer[0] = 5.0;
        assert_eq!(
            buffer.iter().collect::<Vec<&f32>>(),
            vec![&5.0, &3.0, &1.0, &2.0]
        );
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_ring_buffer_index_mut_out_of_bounds() {
        let mut buffer = RingBuffer::<f32, 4>::new();
        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);
        buffer[3] = 1.0;
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_ring_buffer_index_out_of_bounds() {
        let mut buffer = RingBuffer::<f32, 4>::new();
        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);
        assert_eq!(buffer[3], 4.0);
    }

    #[test]
    fn test_ring_buffer_clone() {
        let mut buffer = RingBuffer::<i32, 5>::new();
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4);

        let buffer2 = buffer.clone();
        assert_eq!(buffer2.size, buffer.size);
        assert_eq!(buffer2.capacity, buffer.capacity);
        assert_eq!(
            buffer2.iter().collect::<Vec<&i32>>(),
            buffer.iter().collect::<Vec<&i32>>()
        );
    }

    #[test]
    fn test_ring_buffer_from_array() {
        let buffer = RingBuffer::<i32, 7>::from([1, 2, 3, 4, 5]);
        assert_eq!(buffer.size, 5);
        assert_eq!(buffer.capacity, 7);
        assert_eq!(
            buffer.iter().collect::<Vec<&i32>>(),
            vec![&1, &2, &3, &4, &5]
        );
    }

    #[test]
    fn test_ring_buffer_from_vec() {
        let buffer = RingBuffer::<i32, 7>::from(vec![1, 2, 3, 4, 5]);
        assert_eq!(buffer.size, 5);
        assert_eq!(buffer.capacity, 7);
        assert_eq!(
            buffer.iter().collect::<Vec<&i32>>(),
            vec![&1, &2, &3, &4, &5]
        );
    }

    #[test]
    fn test_ring_buffer_from_arrayvec() {
        let buffer = RingBuffer::<i32, 7>::from(ArrayVec::from([1, 2, 3, 4, 5]));
        assert_eq!(buffer.size, 5);
        assert_eq!(buffer.capacity, 7);
        assert_eq!(
            buffer.iter().collect::<Vec<&i32>>(),
            vec![&1, &2, &3, &4, &5]
        );
    }



}
