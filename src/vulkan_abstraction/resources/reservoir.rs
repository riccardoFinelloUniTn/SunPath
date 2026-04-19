/// Represents a single light candidate reservoir for Spatiotemporal Reservoir Resampling (ReSTIR).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reservoir {
    /// The index of the winning light candidate in the emissive triangles array.
    pub light_idx: u32,
    pub _pad0: [u32; 3], // std430 vec3 alignment forces offset to 16

    /// The exact 3D world position on the light source that was sampled.
    pub light_pos: [f32; 3],
    pub _pad1: f32, // std430 vec3 alignment forces offset to 32

    /// The 3D world normal of the light source at the sampled position.
    pub light_normal: [f32; 3],
    /// The sum of all candidate weights evaluated so far.
    pub w_sum: f32,
    /// The number of light candidates that have been processed to get this winner.
    pub m: f32,
    /// The final unbiased probabilistic weight of this reservoir, used to scale the final shadow ray.
    pub w: f32,
    pub _pad2: [u32; 2], // Pad out to exactly 64 bytes total
}
