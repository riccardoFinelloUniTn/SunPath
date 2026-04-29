use nalgebra as na;

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct MatricesBufferContents {
    pub view_inverse: na::Matrix4<f32>,
    pub proj_inverse: na::Matrix4<f32>,
    pub view_proj: na::Matrix4<f32>,
    pub prev_view_proj: na::Matrix4<f32>,
}
