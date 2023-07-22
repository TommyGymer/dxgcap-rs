/// Color represented by additive channels: Blue (b), Green (g), Red (r), and Alpha (a).
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Eq, Ord)]
pub struct BGRA8 {
    pub b: u8,
    pub g: u8,
    pub r: u8,
    pub a: u8,
}

impl From<RGBA8> for BGRA8 {
    fn from(px: RGBA8) -> Self {
        Self {
            b: px.b,
            g: px.g,
            r: px.r,
            a: px.a,
        }
    }
}

/// Color represented by additive channels: Red (r), Green (g), Blue (b), and Alpha (a).
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Eq, Ord)]
pub struct RGBA8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl From<BGRA8> for RGBA8 {
    fn from(px: BGRA8) -> Self {
        Self {
            r: px.r,
            g: px.g,
            b: px.b,
            a: px.a,
        }
    }
}