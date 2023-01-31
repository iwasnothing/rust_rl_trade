pub struct StepRow {
    pub obs: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub info: String,
}
pub trait RLEnv {
    fn reset(&mut self) -> Vec<f32>;
    fn save_asset_value(&mut self) -> f32;
    fn step(&mut self, action: usize) -> StepRow;
    fn action_space(&self) -> usize;
    fn state_space(&self) -> usize;
}
