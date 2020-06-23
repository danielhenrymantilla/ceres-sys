#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use ::std::{
    convert::TryInto,
    mem,
    ptr::{self, null_mut as NULL},
    slice,
    os::raw::{c_double, c_int, c_void},
};

use ffi::{
    ceres_cost_function_t, ceres_create_problem, ceres_free_problem, ceres_init,
    ceres_problem_add_residual_block, ceres_problem_t, ceres_solve,
};

#[repr(C)]
struct ClosureData<F>
where
    F: FnMut(&[f64], &mut [f64], Option<&mut [f64]>)
{
    rust_closure: F,
    nparams: usize,
}
impl ClosureData {
    fn as_mut_ptr(&mut self) -> *mut Self {
        self
    }
}

fn unpack_closure<F>(
    closure: F,
    nparams: usize,
) -> (ClosureData<F>, ceres_cost_function_t)
where
    F: FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
{
    #[allow(unused_unsafe)]
    unsafe extern "C" fn trampoline<F>(
        data: *mut c_void,
        parameters: *mut *mut c_double,
        mut residuals: *mut c_double,
        mut jacobian: *mut *mut c_double,
    ) -> c_int
    where
        F: FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
    {
        use ::core::{mem, slice};
        let abort_on_drop_guard = {
            struct AbortOnDrop;
            impl Drop for AbortOnDrop { fn drop(&mut self) {
                eprintln!("\
                    Error, Rust was about to unwind across an `extern \"C\"` \
                    function, which is Undefined Behavior.\n\
                    Aborting the process for soundness.\
                ");
                ::std::process::abort();
            }}
            Guard
        };          
        unsafe {
            let mut data: *mut ClosureData = data.cast();
            let closure_data: &mut ClosureData = data.as_mut().expect("Got NULL `data`");
            assert!(closure_data.nparams < 1e4 as usize);
            let slice = |ptr: *const c_double| unsafe {
                if ptr.is_null() {
                    panic!("Got NULL slice pointer");
                }
                slice::from_raw_parts(ptr, closure_data.nparams)
            };
            let slice_mut = |ptr: *mut c_double| unsafe {
                if ptr.is_null() {
                    panic!("Got NULL slice pointer");
                }
                slice::from_raw_parts_mut(ptr, closure_data.nparams)
            };
            let &params: &(*mut c_double) = parameters.as_ref().expect("Got NULL `parameters`");
            let params: &[c_double] = slice(params);
            let &mut residuals: &mut (*mut c_double) = residuals.as_mut().expect("Got NULL `residuals`");
            let closure_residuals: &mut [c_double] = slice_mut(residuals);
            let closure_jac = jacobian.as_mut().map(|&mut ptr: &mut (*mut c_double)| {
                slice_mut(ptr)
            });
            (closure_data.rust_closure)(closure_params, closure_residuals, closure_jac);
        }
        // If we reach this point, no panic has happened, so we can defuse the abort bomb
        mem::forget(abort_on_drop_guard);
        1
    }
    let data = ClosureData {
        rust_closure: closure,
        nparams,
    };
    (data, Some(trampoline::<F>))
}

pub struct CeresSolver {
    problem: *mut ceres_problem_t,
}

impl CeresSolver {
    /// Docs yadda yadda
    ///
    /// # Safety
    ///
    ///   - Can only be called once.
    #[allow(unused_unsafe)]
    pub unsafe fn new() -> Self {
        unsafe {
            // Safety: guaranteed by the caller /* Or the Once guard */
            ffi::ceres_init();
        }
        Self {
            problem: unsafe {
                // Safety: FFI Wrapper + `ceres_init()` has been called.
                ffi::ceres_create_problem()
            },
        }
    }

    pub fn solve<R>(
        &mut self,
        residual_function: impl FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
        x0: &mut [f64],
    ) {
        let (mut data, resid_func) = unpack_closures(residual_function, x0.len());
        let mut x_ptr = x0.as_mut_ptr();
        let mut len: i32 = x0.len().try_into()
        unsafe {
            // Safety: ...
            ffi::ceres_problem_add_residual_block(
                self.problem,
                resid_func,
                data.as_mut_ptr().cast(),
                None,
                NULL(),
                len,
                1,
                &mut len,
                &mut x_ptr,
            );
        }
        unsafe {
            // Safety: ...
            ffi::ceres_solve(self.problem);
        }
    }
}

impl Drop for CeresSolver {
    fn drop(&mut self) {
        unsafe {
            // Safety: FFI Wrapper + `self.problem` originates from `ceres_crate_problem`
            ffi::ceres_free_problem(self.problem);
        }
    }
}
