macro_rules! call_api {
    ($func:ident) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))()
        }
    };

    ($func:ident, $a1: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))($a1)
        }
    };

    ($func:ident, $a1: expr, $a2: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))($a1, $a2)
        }
    };

    ($func:ident, $a1: expr, $a2: expr, $a3: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))($a1, $a2, $a3)
        }
    };

    ($func:ident, $a1: expr, $a2: expr, $a3: expr, $a4: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))($a1, $a2, $a3, $a4)
        }
    };

    ($func:ident, $a1: expr, $a2: expr, $a3: expr, $a4: expr, $a5: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))(
                $a1, $a2, $a3, $a4, $a5
            )
        }
    };

    ($func:ident, $a1: expr, $a2: expr, $a3: expr, $a4: expr, $a5: expr, $a6: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))(
                $a1, $a2, $a3, $a4, $a5, $a6,
            )
        }
    };

    ($func:ident, $a1: expr, $a2: expr, $a3: expr, $a4: expr, $a5: expr, $a6: expr, $a7: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))(
                $a1, $a2, $a3, $a4, $a5, $a6, $a7,
            )
        }
    };

    ($func:ident, $a1: expr, $a2: expr, $a3: expr, $a4: expr, $a5: expr, $a6: expr, $a7: expr, $a8: expr) => {
        unsafe {
            crate::api::API
                .$func
                .as_ref()
                .expect(&format!("failed to get {}", stringify!($func)))(
                $a1, $a2, $a3, $a4, $a5, $a6, $a7, $a8,
            )
        }
    };
}

pub(crate) use call_api;
