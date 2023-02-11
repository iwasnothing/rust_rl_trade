use autograd as ag;
use ndarray;

use ag::optimizers;
use ag::optimizers::Adam;
use ag::prelude::*;
use ag::rand::seq::SliceRandom;
use ag::tensor_ops as T;
use ag::ndarray_ext::ArrayRng;
use ag::VariableEnvironment;
use ag::{ndarray_ext as array, Context};
use ndarray::s;
use ndarray::{Array,Array1,Array2,arr1};

type Tensor<'graph> = ag::Tensor<'graph, f32>;
const lr: f64 = 1e-4;

pub struct DQN<'a> {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    gamma: f32,
    env: VariableEnvironment<'a,f32>,
    rng: ArrayRng<f32>,
    opt: Adam<f32>
}
impl<'a> DQN<'a> {
    pub fn new(input_size: usize,hidden_size: usize,output_size:usize, gamma:f32) -> DQN<'a>{
    	let mut env = ag::VariableEnvironment::new();
	let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
	env.name("w1").set(rng.glorot_uniform(&[input_size, hidden_size]));
	env.name("w2").set(rng.glorot_uniform(&[hidden_size, hidden_size]));
	env.name("w3").set(rng.glorot_uniform(&[hidden_size, output_size]));
	env.name("b1").set(array::zeros(&[1,hidden_size]));
	env.name("b2").set(array::zeros(&[1,hidden_size]));
	env.name("b3").set(array::zeros(&[1,output_size]));
        let opt = Adam::default("adam", env.default_namespace().current_var_ids(), &mut env);
        let mut dqn = DQN{
    	     input_size: input_size,
    	     hidden_size: hidden_size,
    	     output_size: output_size,
    	     gamma: gamma,
    	     env: env,
             rng: rng,
    	     opt: opt
	}; 
	return dqn;
    }
    pub fn infer_action(&mut self, observation: &Vec<f32>) -> usize {
	let mut idx: usize = 0;
	let n = self.input_size as usize;	
        let obs_array = Array1::from_shape_vec(n,observation.to_vec()).unwrap().into_dyn();
	self.env.run( |ctx| {
	    let x = ctx.placeholder("x", &[n.try_into().unwrap()]);
	    let values = nn_forward(ctx,T::expand_dims(x,&[0]));
	    let action = T::argmax(values, -1, false);
	    let result = ctx.evaluator()
                .push(action)
                .feed("x", obs_array.view())
                .run();
	    idx = result[0].as_ref().unwrap()[0].round() as usize;
	});
        return idx;
    }
    pub fn backward<'g>(&mut self, prev_obs_array: Array2<f32>, obs_array: Array2<f32>, action_array: Array1<i64>, done_array: Array1<f32>, reward_array: Array1<f32>) -> f32 {
        let mut eval_loss = 0f32;
	let actionf32_array = action_array.map(|x| *x as f32);
	let n = self.input_size as usize;
	let update_size = action_array.len() as f32;
        let _opt = &self.opt;
	self.env.run( |ctx| {
	    let prev_obs = ctx.placeholder("prev_obs", &[-1, n.try_into().unwrap()]);
	    let obs = ctx.placeholder("obs", &[-1, n.try_into().unwrap()]);
	    let action = ctx.placeholder("action", &[-1]);
	    let done = ctx.placeholder("done", &[-1]);
	    let reward = ctx.placeholder("reward", &[-1]);
	    let diag = ctx.placeholder("diag", &[-1]);
	    let diag_array = Array::range(0.0, update_size*update_size, update_size+1.0);
            let loss = dqn_loss(ctx,prev_obs,obs,action,done,reward,diag,self.gamma);
	    let ns = ctx.default_namespace();
	    let (vars, grads) = optimizers::grad_helper(&[loss], &ns);
	    let update_op = _opt.get_update_op(&vars, &grads, ctx);
            let eval_results = ctx
                        .evaluator()
                        .push(loss)
                        .push(update_op)
                        .feed("prev_obs", prev_obs_array.view())
                        .feed("obs", obs_array.view())
                        .feed("action", actionf32_array.view())
                        .feed("done", done_array.view())
                        .feed("reward", reward_array.view())
                        .feed("diag", diag_array.view())
                        .run();

             let result_loss_value = eval_results[0].as_ref().unwrap();
	     eval_loss = result_loss_value[ndarray::IxDyn(&[])];
	});
        return eval_loss;
    }
}
pub fn nn_forward<'g>(c: &'g Context<f32>, x: Tensor<'g>) -> Tensor<'g> {
        let z1 = T::matmul(x, c.variable("w1")) + c.variable("b1");
        let z2 = T::matmul(z1, c.variable("w2")) + c.variable("b2");
        let z3 = T::matmul(z2, c.variable("w3")) + c.variable("b3");
	return z3;
}
pub fn dqn_loss<'g>(c: &'g Context<f32>, prev_obs_tensors: Tensor<'g>, obs_tensors: Tensor<'g>, action_tensor: Tensor<'g>, done_tensor: Tensor<'g>, reward_tensor: Tensor<'g>,diag: Tensor<'g>, gamma: f32) -> Tensor<'g> {
        let action_values = nn_forward(c,prev_obs_tensors);
        let next_action_values = nn_forward(c,obs_tensors);
        let next_action_max = T::reduce_max(next_action_values, &[1], false);
	let current_at = T::expand_dims(action_tensor,&[1]);
        let current_value = T::squeeze(T::gather(action_values,&current_at, 1),&[2]);
	let current_value_flatten = T::gather(T::flatten(current_value),&diag,0);
        let target_value = reward_tensor + gamma*next_action_max*done_tensor;
	let loss = T::mean_squared_error(current_value_flatten,target_value);
	return loss;
    }
