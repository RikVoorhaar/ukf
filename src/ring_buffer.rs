use std::cmp::min;

pub struct RingBuffer<T: Clone> {
    buffer: Vec<T>,
    start: usize,
    end: usize,
    capacity: usize,
    size: usize,
}

fn modular_sub<T>(a: T, b: T, modulus: T) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Rem<Output = T> + Copy,
{
    ((a + modulus) - b) % modulus
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        RingBuffer {
            buffer: Vec::<T>::with_capacity(capacity),
            start: 0,
            end: 0,
            capacity,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_pull() {
        let mut buffer = RingBuffer::<i32>::new(3);
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
        let mut buffer = RingBuffer::<f32>::new(4);
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

        let mut buffer = RingBuffer::<f32>::new(4);
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
}
