/// This struct is used for Next-Event-Estimation when tracing rays.
/// Instead of only a single random ray the renderer traces an additional ray called shadow ray directed to an emissive triangle
/// This converges the rendering equation much faster especially at the current 1 spp
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EmissiveTriangle {
    pub v0: [f32; 4],       // x, y, z = Position 0 | w = Surface Area
    pub v1: [f32; 4],       // x, y, z = Position 1 | w = Padding
    pub v2: [f32; 4],       // x, y, z = Position 2 | w = Padding
    pub emission: [f32; 4], // r, g, b = Emission Color * Strength | a = Padding
}
