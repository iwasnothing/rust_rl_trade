use std::collections::HashMap;
use math::round;
use crate::rl_env::{RLEnv,StepRow};

pub struct TradeRLEnv {
    pub prices: Vec<f32>,
    pub features: HashMap<String,Vec<f32>>,
    pub max_steps: usize,
    pub feature_names: Vec<String>,
    pub done: bool,
    pub terminated: bool,
    pub t: usize,
    pub reward: f32,
    pub n_actions: usize,
    pub train_threshold: usize,
    pub cumulate_PL: f32,
    pub position_history: Vec<f32>,
    pub asset_value: Vec<f32>,
    pub PL: Vec<f32>,
    pub state_high: Vec<f32>,
    pub state_low: Vec<f32>,
    pub capital: f32,
    pub stop_loss: f32,
    pub short_sell: bool,
    pub position: f32,
    pub prev_bought: f32,
    pub waiting_intervals: u32,
    pub held_intervals: u32,
    pub isTraining: bool,
    pub isEndDay: bool,
}
impl TradeRLEnv {
	pub fn new(_prices: &Vec<f32>, _features: &HashMap<String,Vec<f32>>, _names: &Vec<String>, train_test_split: f32,shortsell: bool, stoploss: f32 ) -> TradeRLEnv {

	    let n: usize = _prices.len() as usize;
	    println!("max_size={}",n);
	
	    let _train_threshold: usize = round::floor(n as f64 * train_test_split as f64,0) as usize;
	    let _capital = 1000.0;
	    let state_size = _names.len()+3;
	    let mut _state_high: Vec<f32> = Vec::with_capacity(state_size);
	    let mut _state_low: Vec<f32> = Vec::with_capacity(state_size);
	    for f in _names {
		match _features.get(f) {
		    Some(_vec) => {
			match _vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
			    Some(m) => _state_low.push(*m),
			    None => panic!("no min for feature {}",&f),
			};
			match _vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
			    Some(m) => _state_high.push(*m),
			    None => panic!("no max for feature {}",&f),
			};
		    },
		    None => panic!("feature {} not found",&f),
		}
	    }
	    match _prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
		Some(m) => {
		    _state_low.push(*m);
		},
		None => panic!("no min for price"),
	    };
	    match _prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
		Some(m) => _state_high.push(*m),
		None => panic!("no max for price"),
	    };
	    let lowest_price = _state_low[_state_low.len()-1];
	    _state_low.push(-1.0*_capital/lowest_price);
	    _state_high.push(_capital/lowest_price);
	    _state_low.push(0.0);
	    _state_high.push(n as f32);
	    println!("state_low_high={:?},{:?}",_state_low,_state_high);
	    return TradeRLEnv{
		prices: _prices.clone(),
		features: _features.clone(),
		max_steps: n,
		feature_names: _names.clone(),
		done: false,
		terminated: false,
		t: 0,
		reward: 0.0,
		n_actions: 3,
		train_threshold: _train_threshold,
		cumulate_PL: 0f32,
		position_history: Vec::with_capacity(n),
		asset_value: Vec::with_capacity(n),
		PL: Vec::with_capacity(n),
		state_high: _state_high.clone(),
		state_low: _state_low.clone(),
		capital: _capital,
		stop_loss: stoploss,
		short_sell: shortsell,
		position: 0.0,
		prev_bought: 0.0,
		waiting_intervals: 0,
		held_intervals: 0,
		isTraining: true,
		isEndDay: false,
	    };
	}
    pub fn reward(&self) -> f32 {
        return self.reward;
    }
    pub fn is_terminated(&self) -> bool {
        return self.t >= self.max_steps-1;
    }
    pub fn state(&self) -> Vec<f32> {
        let mut current_state: Vec<f32> = Vec::with_capacity(self.state_space());
        for f in &self.feature_names {
            match self.features.get(f) {
                Some(_vec) => {
                    let val = _vec[self.t];
                    current_state.push(val);
                },
                None => panic!("feature {} not found",&f),
            }
        }
        current_state.push(self.prices[self.t]);
        current_state.push(self.position);
        current_state.push(self.waiting_intervals as f32);
        let n = current_state.len();
        for i in 0..n {
            current_state[i] = (current_state[i] - self.state_low[i]) / (self.state_high[i]-self.state_low[i])
        }
        return current_state
    }
    pub fn start_backtest(&mut self) -> Vec<f32> {
        self.t = self.train_threshold + 1;
        self.isTraining = false;
        self.PL.clear();
        self.PL.push(0.0);
	self.cumulate_PL = 0f32;
        self.position_history.clear();
        self.position_history.push(0.0);
        self.asset_value.clear();
        return self.reset();
    }
    fn _enter_pos(&mut self, buysell: i32) {
        self.held_intervals = self.waiting_intervals;
        self.waiting_intervals = 0;
        self.prev_bought = self.prices[self.t];
        self.position = (buysell as f32) * self.capital / self.prev_bought;
        self.reward = 0.0;
    }
    fn _close_pos(&mut self) {
        self.held_intervals = self.waiting_intervals;
        self.waiting_intervals = 0;
        self.done = true;
        let profit = self.position * ( self.prices[self.t] - self.prev_bought) / self.capital;
        self.reward = 1000.0 * profit;
        self.PL.push(profit);
	self.cumulate_PL += profit;
        self.position = 0.0;
    }
    fn _punish_repeat_action(&mut self) {
        self.waiting_intervals = self.waiting_intervals + 1;
        self.reward = -3.0;
    }
    fn _punish_long_wait(&mut self) {
        self.waiting_intervals = self.waiting_intervals + 1;
        if self.waiting_intervals > 30 {
            self.reward = -5.0;
        } else if self.waiting_intervals > 10 {
            self.reward = -3.0;
        } else {
            self.reward = -1.0;
        }
    }
    fn _end_day(&mut self) {
        self.isEndDay = true;
        self.position = 0.0;
        self.held_intervals = self.waiting_intervals;
        self.waiting_intervals = 0;
        self.done = true;
        self.reward = 0.0;
        self.prev_bought = 0.0;
    }
    fn _meet_stop_loss(&mut self) -> bool {
        let profit = self.position*(self.prices[self.t]-self.prev_bought);
        let rtn_pct = profit / self.capital;
        if rtn_pct < -1.0 * self.stop_loss  {
            return true;
        } else {
            return false;
        }
    }
    fn _move_next_day(&mut self) {
        self.position_history.push(self.position);
        self.t = self.t + 1;
        if self.isTraining && self.t >= self.train_threshold {
            self._end_day();
            let r = rand::random::<f64>() * self.train_threshold as f64;
            let i: usize = round::floor(r,0) as usize;
            self.t = i;
        }
        if !self.isTraining && self.t >= self.max_steps-1 {
            self._end_day();
            //self.t = self.train_threshold + 1
	    self.terminated = true;
        }
    }
}
impl RLEnv for TradeRLEnv {
    fn step(&mut self, action: usize) -> StepRow {
        if action == 0 {
            if self.position > 0.0 {
                self._close_pos();
            } else if self.position == 0.0 && self.short_sell {
                self._enter_pos(-1);
            } else {
                self._punish_repeat_action();
            }
        } else if action == 2 {
            if self.position == 0.0 {
                self._enter_pos(1);
            } else if self.position < 0.0 && self.short_sell {
                self._close_pos();
            } else {
                self._punish_repeat_action();
            }
        } else if self._meet_stop_loss() {
            self._close_pos();
        } else {
            self._punish_long_wait();
        }
        self._move_next_day();
        return StepRow{
		obs: self.state(),
		reward: self.reward,
		done: self.done, 
		info: "".to_string()
		};
    }
    fn save_asset_value(&mut self) -> f32 {
	let mut val = self.capital;
	if self.position > 0.0 {
	    val = self.position * self.prices[self.t];
        }
	if self.position < 0.0 {
	    val = val + self.position * self.prices[self.t];
        }
	self.asset_value.push(val);
	return val;
    }
	
    fn reset(&mut self) -> Vec<f32> {
        self.done = false;
        self.reward = 0.0;
        self.isEndDay = false;
        self.prev_bought = 0.0;
        self.position = 0.0;
        self.position_history.push(0.0);
        self.waiting_intervals = 0;
        self.held_intervals = 0;
	return self.state();
    }
    fn action_space(&self) -> usize {
	return self.n_actions;
    }                
    fn state_space(&self) -> usize {
	let n = self.feature_names.len();
	return n+3;
    }                

}
