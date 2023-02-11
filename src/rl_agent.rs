use crate::rl_env::RLEnv;
use crate::autograd_nn::{DQN};
use autograd as ag;
use ag::prelude::*;
use ag::tensor_ops as T;
use rulinalg::utils;
use math::round;
use rand::seq::SliceRandom;
use ndarray;
use ndarray::{Array,Array1,Array2};

type Tensor<'graph> = ag::Tensor<'graph, f32>;

pub struct MemRow {
    pub done: bool,
    pub action: usize,
    pub obs: Vec<f32>,
    pub prev_obs: Vec<f32>,
    pub reward: f32
}
impl Clone for MemRow {
    fn clone(&self) -> MemRow {
	MemRow {
	   done: self.done,
	   action: self.action,
	   obs: self.obs.clone(),
	   prev_obs: self.prev_obs.clone(),
	   reward: self.reward
	}
    }
}
impl MemRow {
    pub fn get_done_as_val(&self) -> f32 {
	if self.done {
	    return 0f32;
	} else {
	    return 1f32;
	}
    }
}
pub struct MemBuffer {
    data: Vec<MemRow>,
    ptr: usize,
    length: usize,
    max_size: usize,
}
impl MemBuffer {
    pub fn new(max_size: usize) -> MemBuffer {
	MemBuffer {
	    data: Vec::with_capacity(max_size),
	    ptr: 0,
	    length: 0,
	    max_size: max_size
	}
    }
    pub fn add(&mut self, entry: &MemRow) {
	if self.ptr >= self.data.len() {
	    self.data.push(entry.clone());
	}
	else {
	    self.data[self.ptr] = entry.clone();
        }
	self.ptr += 1;
	self.length += 1;
	if self.ptr >= self.max_size {
	    self.ptr = 0;
	}
	if self.length > self.max_size {
	    self.length = self.max_size;
	}
    }
    pub fn get(&self, idx: usize) -> &MemRow {
	return &self.data[idx];
    }
    pub fn len(&self) -> usize {
	return self.length;
    }
}
pub struct RLAgent<'a> {
    env_action_space: usize,
    input_size: usize,
    output_size: usize,
    epsilon: f32,
    memory: MemBuffer,
    layers: DQN<'a>,
    losses: f32,
    n_losses: i64
}
impl<'a> RLAgent<'a> {
	pub fn new(action_space: usize, input_size: usize, output_size: usize) -> RLAgent<'a> {
	    return RLAgent {
		env_action_space: action_space,
		input_size: input_size,
		output_size: output_size,
		epsilon: 1.0,
		memory: MemBuffer::new(20000),
		layers: DQN::new(input_size,128,output_size,0.9),
		losses: 0f32,
		n_losses: 0,
	    }
	}
    pub fn get_mean_loss(&mut self) -> f32 {
        let mean_loss = self.losses/ self.n_losses as f32 ;
	self.losses = 0f32;
	self.n_losses = 0;
	return mean_loss;
    }
    pub fn select_action(&mut self, observation: &Vec<f32>) -> usize {
	let r = rand::random::<f32>();
	if r >= self.epsilon {
	    let idx = self.layers.infer_action(observation);
	    return idx;
	} else {
	    let rn = rand::random::<f64>();
	    let n_actions = self.env_action_space;
	    let mut rng = &mut rand::thread_rng();
	    let sample: Vec<usize> = (0..n_actions).collect();
	    let result: Vec<usize> = sample.choose_multiple(&mut rng, 3).cloned().collect();
	    return result[0];
        }
    }
    pub fn remember(&mut self, done: bool, action: usize, observation: Vec<f32>, prev_obs: Vec<f32>, reward: f32) {
	let _row: MemRow = MemRow {
		done: done,
		action: action,
		obs: observation,
		prev_obs: prev_obs,
		reward: reward
	};
        self.memory.add(&_row);
     }
     pub fn get_memory_batch(&self, update_size: usize) -> (Array2<f32>,Array2<f32>,Array1<i64>,Array1<f32>,Array1<f32>) {
	let mem_len = self.memory.len();
	let mut rng = &mut rand::thread_rng();
	let sample: Vec<usize> = (0..mem_len).collect();
	let batch_indices: Vec<usize> = sample.choose_multiple(&mut rng, update_size).cloned().collect();
	let n = self.input_size as usize;
	let mut done_vec: Vec<f32> = Vec::with_capacity(update_size);
	let mut action_vec: Vec<i64> = Vec::with_capacity(update_size);
	let mut reward_vec: Vec<f32> = Vec::with_capacity(update_size);
	let mut obs_vec: Vec<f32> = Vec::with_capacity(update_size*n);
	let mut prev_obs_vec: Vec<f32> = Vec::with_capacity(update_size*n);
	for index in batch_indices {
	    let row: &MemRow = self.memory.get(index);
	    done_vec.push(row.get_done_as_val());
	    action_vec.push(row.action as i64);
	    reward_vec.push(row.reward);
	    obs_vec.extend(row.obs.to_owned());
	    prev_obs_vec.extend(row.prev_obs.to_owned());
	}
	let obs_array = Array2::from_shape_vec((update_size,n),obs_vec).unwrap();
	let prev_obs_array = Array2::from_shape_vec((update_size,n),prev_obs_vec).unwrap();
	let done_array = Array1::from_shape_vec(update_size,done_vec).unwrap();
	let action_array = Array1::from_shape_vec(update_size,action_vec).unwrap();
	let reward_array = Array1::from_shape_vec(update_size,reward_vec).unwrap();

	return (prev_obs_array,obs_array,action_array,done_array,reward_array);
     }
	
     pub fn experience_replay(&mut self) {
        let update_size = 64;
	let mem_len = self.memory.len();
	if mem_len < update_size {
	    return;
	} else {
	    let (prev_obs_array,obs_array,action_array,done_array,reward_array) = self.get_memory_batch(update_size);
	    self.backward(prev_obs_array,obs_array,action_array,done_array,reward_array);
	}
	if self.epsilon >= 0.01 {
	    self.epsilon = self.epsilon*0.9;
	}
    }
    pub fn backward(&mut self, prev_obs_array: Array2<f32>,obs_array: Array2<f32>,action_array: Array1<i64>,done_array: Array1<f32>, reward_array:Array1<f32> ) {
	self.losses += self.layers.backward(prev_obs_array, obs_array, action_array, done_array,reward_array);
	self.n_losses += 1;
    }
}

