extern crate libc;
use libc::size_t;
use std::ffi::{c_void, c_char, CStr};
use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Arc;
pub type ValueType = f32;

pub struct CudaVec<T: Default + Clone + Debug> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T: Default + Clone + Debug> CudaVec<T> {
    pub fn new(len: usize) -> Arc<Self> {
        let ptr = match malloc(len) {
            Ok(ptr) => ptr,
            Err(err) => panic!("array new: {}", error_string(err)),
        };
        Arc::new(Self { ptr, len })
    }

    pub fn from<I: AsRef<[T]>>(data: I) -> Arc<Self> {
        let data = data.as_ref();

        let res = Self::new(data.len());
        let err = memcpy(data.as_ptr(), res.ptr, res.len, cudaMemcpyKind::cudaMemcpyHostToDevice);
        if let Err(e) = err {
            panic!("Vec from: data copy error: {}", error_string(e));
        }
        
        res
    }

    pub fn copy(&self) -> Arc<Self> {
        let res = Self::new(self.len);
        let err = memcpy(self.ptr, res.ptr, self.len, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if let Err(e) = err {
            panic!("Vec copy: data copy error: {}", error_string(e));
        }
        res
    }

    pub fn as_vec(&self) -> Vec<T> {
        let mut host_data: Vec<T> = vec![T::default(); self.len];
        let err = memcpy(self.ptr, host_data.as_mut_ptr(), self.len, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        if let Err(e) = err {
            panic!("Vec as_vec: data copy error: {}", error_string(e));
        }
        host_data
    }

    pub fn read_from_index(&self, index: usize) -> T {
        debug_assert!(index < self.len);

        let mut host_value: T = Default::default();

        let offset = index * size_of::<T>();
        let device_ptr = (self.ptr as usize + offset) as *const T;

        let err = memcpy(device_ptr, &mut host_value as *mut T, 1, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        if let Err(e) = err {
            panic!("Vec read_from_index: element copy error: {}", error_string(e));
        }
        host_value
    }

    fn deallocate(&mut self) {
        if self.ptr.is_null() {
            println!("deallocate: null pointer, ptr: {:?}", self.ptr);
            return;
        }
        let err = free(self.ptr);
        if let Err(e) = err {
            panic!("{:?}\nVec deallocate: free error: {}", self.ptr, error_string(e));
        }
        self.ptr = std::ptr::null_mut();
    }
}

impl<T: Default + Clone + Debug> Drop for CudaVec<T> {
    fn drop(&mut self) {
        self.deallocate();
    }
}

extern "C" {
    fn cudaFree(ptr: *mut c_void) -> u32;
    fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> u32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> u32;
    fn cudaGetErrorString(error: u32) -> *const c_char;
}

pub fn free<T>(ptr: *mut T) -> Result<(), u32> {
    if !ptr.is_null() {
        //println!("Freeing pointer: {:?}", ptr); // Debug statement
        unsafe {
            let err = cudaFree(ptr as *mut c_void);
            if err != 0 {
                return Err(err);
            }
        }
    }
    Ok(())
}

pub fn malloc<T>(len: usize) -> Result<*mut T, u32> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        let err = cudaMalloc(
            &mut ptr as *mut *mut c_void,
            len * size_of::<T>()
        );
        if err != 0 {
            return Err(err);
        }
    }
    Ok(ptr as *mut T)
}

pub fn memcpy<T>(source: *const T, dist: *mut T, len: usize, kind: cudaMemcpyKind) -> Result<(), u32>{
    unsafe {
        let err = cudaMemcpy(
            dist as *mut c_void,
            source as *const c_void,
            len * size_of::<T>(),
            kind,
        );
        if err != 0 {
            return Err(err);
        };
    }
    Ok(())
}

pub fn error_string(error: u32) -> String {
    unsafe {
        CStr::from_ptr(cudaGetErrorString(error)).to_string_lossy().into_owned()
    }
}

#[allow(non_camel_case_types)] #[repr(C)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
}