use super::cuda_vec::CudaVec;

extern "C" {
    fn vec_usize_cmp(a: *mut usize, b: *mut usize, len: usize) -> u32;

    fn vec_i8_cmp(a:  *mut i8,  b: *mut i8,  len: usize) -> i32;
    fn vec_i16_cmp(a: *mut i16, b: *mut i16, len: usize) -> i32;
    fn vec_i32_cmp(a: *mut i32, b: *mut i32, len: usize) -> i32;
    fn vec_i64_cmp(a: *mut i64, b: *mut i64, len: usize) -> i32;

    fn vec_u8_cmp(a:  *mut u8,  b: *mut u8,  len: usize) -> i32;
    fn vec_u16_cmp(a: *mut u16, b: *mut u16, len: usize) -> i32;
    fn vec_u32_cmp(a: *mut u32, b: *mut u32, len: usize) -> i32;
    fn vec_u64_cmp(a: *mut u64, b: *mut u64, len: usize) -> i32;

    fn vec_f32_cmp(a: *mut f32, b: *mut f32, len: usize) -> i32;
    fn vec_f64_cmp(a: *mut f64, b: *mut f64, len: usize) -> i32;
}

macro_rules! impl_cmp {
    ($call:ident, $t:ty) => {
        impl PartialEq for CudaVec<$t> {
            fn eq(&self, other: &Self) -> bool {
                if self.len != other.len {
                    return false;
                }
                unsafe { $call(self.ptr, other.ptr, self.len) != 0 }
            }
        }
    };
}

impl_cmp!(vec_usize_cmp, usize);

impl_cmp!(vec_i8_cmp,  i8);
impl_cmp!(vec_i16_cmp, i16);
impl_cmp!(vec_i32_cmp, i32);
impl_cmp!(vec_i64_cmp, i64);

impl_cmp!(vec_u8_cmp,  u8);
impl_cmp!(vec_u16_cmp, u16);
impl_cmp!(vec_u32_cmp, u32);
impl_cmp!(vec_u64_cmp, u64);

impl_cmp!(vec_f32_cmp, f32);
impl_cmp!(vec_f64_cmp, f64);

#[cfg(test)]
mod tests {
    use crate::prelude::CudaVec;

    #[test]
    fn test_eq() {
        let a = CudaVec::from([1., 2., 3., 4.]);
        let b = CudaVec::from([1., 2., 3., 4.]);
        let c = CudaVec::from([1., 3., 2., 4.]);
        assert!(a != c);
        assert!(a == b);
    }
}