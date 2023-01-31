use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, nn::Sequential, nn::Optimizer,Tensor, Kind::Float};

const lr: f64 = 1e-4;
pub struct DQN {
    vs: nn::VarStore,
    input_size: i64,
    hidden_size: i64,
    output_size: i64,
    layers: Sequential,
    opt: Optimizer
}

impl DQN {
	pub fn new(input_size: i64,hidden_size: i64,output_size:i64) -> DQN{
	    let vs = nn::VarStore::new(Device::Cpu);
	    let p = vs.root();
	    let layers = nn::seq()
		.add(nn::linear(&p, input_size, hidden_size, Default::default()))
		.add_fn(|xs| xs.relu())
		.add(nn::linear(&p, hidden_size, hidden_size, Default::default()))
		.add_fn(|xs| xs.relu())
		.add(nn::linear(&p, hidden_size, output_size, Default::default()));
	    let opt = nn::Adam::default().build(&vs, lr).unwrap();
	    return DQN {
		vs: vs,
		input_size: input_size,
		hidden_size: hidden_size,
		output_size: output_size,
		layers: layers,
		opt: opt
	    }
	}
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        return self.layers.forward(xs);
    }
    pub fn backward(&mut self, xs: &Tensor) {
	self.opt.zero_grad();
        self.opt.backward_step(xs);
    }
}
