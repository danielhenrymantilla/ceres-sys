extern crate bindgen;
use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    let dst = Config::new("cereswrapper").build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=cereswrapper");
    println!("cargo:rustc-link-lib=ceres");

    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_arg(format!("-I{}", dst.display()))
        .whitelist_type("ceres_cost_function_t")
        .whitelist_type("ceres_loss_function_t")
        .whitelist_type("ceres_problem_t")
        .whitelist_function("ceres_init")
        .whitelist_function("ceres_create_problem")
        .whitelist_function("ceres_problem_add_residual_block")
        .whitelist_function("ceres_solve")
        .whitelist_function("ceres_free_problem")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
