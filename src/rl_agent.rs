use crate::rl_env::RLEnv;
use crate::neural_network::{DQN};
use tch::{Tensor,Reduction};
use rulinalg::utils;
use math::round;
use rand::seq::SliceRandom;

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
pub struct RLAgent {
    env_action_space: usize,
    input_size: i64,
    output_size: i64,
    epsilon: f32,
    gamma: f32,
    memory: MemBuffer,
    layers: DQN,
    losses: f64,
    n_losses: i64
}
impl RLAgent {
	pub fn new(action_space: usize, input_size: i64, output_size: i64) -> RLAgent {
	    return RLAgent {
		env_action_space: action_space,
		input_size: input_size,
		output_size: output_size,
		epsilon: 1.0,
		gamma: 0.9,
		memory: MemBuffer::new(20000),
		layers: DQN::new(input_size,128,output_size),
		losses: 0f64,
		n_losses: 0,
	    }
	}
    pub fn get_mean_loss(&mut self) -> f64 {
        let mean_loss = self.losses/ self.n_losses as f64 ;
	self.losses = 0f64;
	self.n_losses = 0;
	return mean_loss;
    }
    pub fn infer_action(&self, observation: &Vec<f32>) -> usize {
	let obs_tensor = Tensor::of_slice(observation);
	let values = self.forward(&obs_tensor);
	//println!("infer to {:?}",Vec::<f32>::from(&values));
	let idx = values.argmax(None,false);
	return idx.int64_value(&[]) as usize;
    }
    pub fn select_action(&self, observation: &Vec<f32>) -> usize {
	let obs_tensor = Tensor::of_slice(observation);
	let values = self.forward(&obs_tensor);
	let r = rand::random::<f32>();
	if r >= self.epsilon {
	    let idx = values.argmax(None,false);
	    return idx.int64_value(&[]) as usize;
	} else {
	    let rn = rand::random::<f64>();
	    let n_actions = self.env_action_space;
	    let mut rng = &mut rand::thread_rng();
	    let sample: Vec<usize> = (0..n_actions).collect();
	    let result: Vec<usize> = sample.choose_multiple(&mut rng, 3).cloned().collect();
	    return result[0];
        }
    }
    pub fn forward(&self, xs: &Tensor) -> Tensor {
	let y =  self.layers.forward(xs);
	return y;
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
     pub fn get_memory_batch(&self, update_size: usize) -> (Tensor,Tensor,Tensor,Tensor,Tensor) {
	let mem_len = self.memory.len();
	let mut rng = &mut rand::thread_rng();
	let sample: Vec<usize> = (0..mem_len).collect();
	let batch_indices: Vec<usize> = sample.choose_multiple(&mut rng, update_size).cloned().collect();

	let mut done_vec: Vec<f32> = Vec::with_capacity(update_size);
	let mut action_vec: Vec<i64> = Vec::with_capacity(update_size);
	let mut reward_vec: Vec<f32> = Vec::with_capacity(update_size);
	let mut obs_vec: Vec<Tensor> = Vec::with_capacity(update_size);
	let mut prev_obs_vec: Vec<Tensor> = Vec::with_capacity(update_size);
	for index in batch_indices {
	    let row: &MemRow = self.memory.get(index);
	    done_vec.push(row.get_done_as_val());
	    action_vec.push(row.action as i64);
	    reward_vec.push(row.reward);
	    obs_vec.push(Tensor::of_slice(&row.prev_obs));
	    prev_obs_vec.push(Tensor::of_slice(&row.prev_obs));
	}
	let obs_tensors = Tensor::vstack(&obs_vec);
	let prev_obs_tensors = Tensor::vstack(&prev_obs_vec);
	let done_tensor = Tensor::of_slice(&done_vec);
	let action_tensor = Tensor::of_slice(&action_vec);
	let reward_tensor = Tensor::of_slice(&reward_vec);

	return (done_tensor,action_tensor,reward_tensor,obs_tensors,prev_obs_tensors);
     }
	
     pub fn experience_replay(&mut self) {
        let update_size = 64;
	let mem_len = self.memory.len();
	if mem_len < update_size {
	    return;
	} else {
	    let (done_tensor,action_tensor,reward_tensor,obs_tensors,prev_obs_tensors) = self.get_memory_batch(update_size);
	    let action_values: Tensor = self.forward(&prev_obs_tensors);
       	    let next_action_values: Tensor = self.forward(&obs_tensors);
	    let next_action_max = next_action_values.amax(&[1],false);
	    let current_at = action_tensor.unsqueeze(1);
	    let current_value = action_values.gather(1,&current_at, false).flatten(0,1);
	    let target_value = reward_tensor + self.gamma*next_action_max * done_tensor;
	    self.backward(&current_value, &target_value);
	}
	if self.epsilon >= 0.01 {
	    self.epsilon = self.epsilon*0.9;
	}
    }
    pub fn backward(&mut self, calculated_values: &Tensor, experimental_values: &Tensor) {
	let loss = calculated_values.l1_loss(&experimental_values,Reduction::Mean);
	self.layers.backward(&loss);
	self.losses += f64::from(&loss);
	self.n_losses += 1;
    }
}

