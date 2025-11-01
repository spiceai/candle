<<<<<<< HEAD
pub const AFFINE: &str = include_str!(concat!(env!("OUT_DIR"), "/affine.ptx"));
pub const BINARY: &str = include_str!(concat!(env!("OUT_DIR"), "/binary.ptx"));
pub const CAST: &str = include_str!(concat!(env!("OUT_DIR"), "/cast.ptx"));
pub const CONV: &str = include_str!(concat!(env!("OUT_DIR"), "/conv.ptx"));
pub const FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/fill.ptx"));
pub const FUSED_RMS_NORM: &str = include_str!(concat!(env!("OUT_DIR"), "/fused_rms_norm.ptx"));
pub const FUSED_ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/fused_rope.ptx"));
pub const INDEXING: &str = include_str!(concat!(env!("OUT_DIR"), "/indexing.ptx"));
pub const KVCONCAT: &str = include_str!(concat!(env!("OUT_DIR"), "/kvconcat.ptx"));
pub const MUL_AND_ACT: &str = include_str!(concat!(env!("OUT_DIR"), "/mul_and_act.ptx"));
pub const QUANTIZED: &str = include_str!(concat!(env!("OUT_DIR"), "/quantized.ptx"));
pub const REDUCE: &str = include_str!(concat!(env!("OUT_DIR"), "/reduce.ptx"));
pub const SORT: &str = include_str!(concat!(env!("OUT_DIR"), "/sort.ptx"));
pub const TERNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/ternary.ptx"));
pub const UNARY: &str = include_str!(concat!(env!("OUT_DIR"), "/unary.ptx"));
=======
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);
>>>>>>> main
