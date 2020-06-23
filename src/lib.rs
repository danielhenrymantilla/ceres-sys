#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::os::raw::{c_double, c_int, c_void};

use ffi::{
    ceres_cost_function_t, ceres_create_problem, ceres_free_problem, ceres_init,
    ceres_problem_add_residual_block, ceres_problem_t, ceres_solve,
};

#[repr(C)]
struct ClosureData<'a> {
    cost_fn: &'a mut dyn FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
    nparams: usize,
}

unsafe fn unpack_closures<F>(
    mut closure: &mut F,
    nparams: usize,
) -> (*mut c_void, ceres_cost_function_t)
where
    F: FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
{
    extern "C" fn trampoline(
        data: *mut c_void,
        parameters: *mut *mut c_double,
        residuals: *mut c_double,
        jacobian: *mut *mut c_double,
    ) -> c_int {
        unsafe {
            let closure_data: &mut ClosureData = &mut *(data as *mut ClosureData);
            let params = std::slice::from_raw_parts(parameters, 1);
            let closure_params = std::slice::from_raw_parts(params[0], closure_data.nparams);
            let mut closure_residuals =
                std::slice::from_raw_parts_mut(residuals, closure_data.nparams);
            let closure_jac = if jacobian.is_null() {
                None
            } else {
                let jacobians = std::slice::from_raw_parts_mut(jacobian, 1);
                if jacobians[0].is_null() {
                    None
                } else {
                    Some(std::slice::from_raw_parts_mut(
                        jacobians[0],
                        closure_data.nparams,
                    ))
                }
            };
            (*closure_data.cost_fn)(&closure_params, &mut closure_residuals, closure_jac);
        }
        1
    }

    (
        &mut ClosureData {
            cost_fn: &mut closure,
            nparams,
        } as *mut ClosureData as *mut c_void,
        Some(trampoline),
    )
}

pub struct CeresSolver {
    problem: *mut ceres_problem_t,
}

impl CeresSolver {
    pub unsafe fn new() -> Self {
        ceres_init();
        Self {
            problem: ceres_create_problem(),
        }
    }

    pub fn solve<R>(&self, residual_function: &mut R, x0: &mut [f64])
    where
        R: FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
    {
        let (data, resid_func) = unsafe { unpack_closures(residual_function, x0.len()) };
        let mut x_ptr = x0.as_mut_ptr();
        unsafe {
            ceres_problem_add_residual_block(
                self.problem,
                resid_func,
                data,
                None,
                std::ptr::null_mut(),
                x0.len() as i32,
                1,
                &mut x0.len() as *mut usize as *mut i32,
                &mut x_ptr as *mut *mut f64,
            );
        }
        unsafe {
            ceres_solve(self.problem);
        }
    }
}

impl Drop for CeresSolver {
    fn drop(&mut self) {
        unsafe {
            ceres_free_problem(self.problem);
        }
    }
}
