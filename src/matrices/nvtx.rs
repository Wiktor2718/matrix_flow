use std::ffi::CString;

extern "C" {
    fn nvtxRangePushA(message: *const libc::c_char);
    fn nvtxRangePop();
}

pub fn range_push(message: &str) {
    let message = CString::new(message).expect("CString::new failed");
    unsafe {
        nvtxRangePushA(message.as_ptr());
    }
}

pub fn range_pop() {
    unsafe {
        nvtxRangePop();
    }
}

pub fn debug_range_push(message: &str) {
    #[cfg(debug_assertions)]
    {
        let message = CString::new(message).expect("CString::new failed");
        unsafe {
            nvtxRangePushA(message.as_ptr());
        }
    }
}

pub fn debug_range_pop() {
    #[cfg(debug_assertions)]
    unsafe {
        nvtxRangePop();
    }
}
