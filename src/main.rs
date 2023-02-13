use ndarray::Array;
use std::collections::HashMap;
use crate::rl_env::{RLEnv,StepRow};
use crate::rl_agent::{RLAgent,MemRow};
use crate::trade_env::{TradeRLEnv};
use crate::db_access::{ReqData,claim_req,perf_report};
use substring::Substring;
use tokio::time::{sleep, Duration};

mod trade_env;
mod rl_env;
mod rl_agent;
mod neural_network;
mod market_data;
mod db_access;

#[tokio::main]
async fn main() {
   loop {
      process_req().await;
      sleep(Duration::from_millis(10000)).await;
   }
}
async fn process_req() {
    let req = match claim_req().await {
	Some(r) => r,
	None => return,
    };
    println!("{:?}",&req);
    let mut env: TradeRLEnv = match get_trade_env(&req).await {
				Some(env) => env,
				None => panic!("cannot get trade env"),
    };
    let num_episodes = req.epi;
    println!("total episodes: {:?}", num_episodes);
    let max_timesteps = 2000;
    let input_size = env.state_space() as i64;
    let output_size = env.action_space() as i64;
    let mut model = RLAgent::new(env.action_space(), input_size, output_size);
    
    for i_episode in 0..num_episodes {
	let observation = env.reset();
	let mut total_reward = 0f32;
        for t in 1..max_timesteps {
	    let action = model.select_action(&observation);
            let prev_obs = observation.clone();
            let mut steprow: StepRow = env.step(action);
	    total_reward += steprow.reward;
	    model.remember(steprow.done, action,steprow.obs, prev_obs,steprow.reward );
	    model.experience_replay();
            if steprow.done { 
		break;
	    } 
        }
	if i_episode % 100 == 0 {
	    let mean_loss = model.get_mean_loss();
	    println!("looping episode {:?}, loss={:?}, total reward = {:?}", &i_episode, &mean_loss, &total_reward);
	    if mean_loss < 1.0 && total_reward > 0.0 {
		break;
	    }
	}
    }
    let mut observation = env.start_backtest();
    let mut action_list: Vec<usize> = Vec::with_capacity(max_timesteps);
    for t in 1..max_timesteps {
            if env.is_terminated() { 
		break;
	    } 
	    //println!("{},{},{},{}",t, env.t, env.train_threshold, env.max_steps);
	    let action = model.infer_action(&observation);
	    action_list.push(action);
            let steprow: StepRow = env.step(action);
	    observation = steprow.obs.clone();
            env.save_asset_value();
    }
    println!("total PL={:?}", &env.cumulate_PL);
    let result = perf_report(&req.id, req.epi, env.cumulate_PL, &env.PL, &env.position_history, &action_list, &(env.prices[env.train_threshold+1..env.max_steps-1].to_vec()), &env.asset_value).await;
    match result {
	Some(s) => println!("{:?}",s),
        None => println!("cannot report performance")
    }
}
async fn get_trade_env(req: &ReqData) -> Option<TradeRLEnv> {

    let train_test_split = match req.split_pct.substring(0,2).parse::<i32>() {
			Ok(i) => i as f32/100.0,
			Error => 0.5,
    };
    let nyrs = match req.num_years.substring(0,1).parse::<i32>() {
			Ok(i) => i ,
			Error => 5,
    };
    let freq = req.freq.replace("1-day","1Day");
    let stoploss = match req.stop_loss.replace("%","").parse::<i32>() {
			Ok(i) => (100.0 - i as f32) / 100.0,
			Error => 0.2,
    };
    let mut shortsell = false;
    if req.short_Sell == "enable" {
	shortsell = true;
    }

    let sym = req.asset.clone();
    let bars = market_data::download_data(&sym,nyrs,&freq).await.unwrap();
    let prices = bars.to_price_vec();
    let timestamps = bars.to_datetime_vec();
    let pct = market_data::pctchg(&prices);
    let ma = market_data::ma(&prices,3);
    let mut min_len = prices.len();
    let mut hmap = HashMap::new();
    let mut features: Vec<String> = Vec::new();
    for f in req.feature_list.iter() {
        println!("{:?}",f);
        let fname = f.get("name")?.to_string();
        features.push(fname.clone());
        if fname == "RSI" {
	    let lookback = match f.get("look-back-period")?.to_string().parse::<i32>() {
				Ok(i) => i as usize,
				Err(e) => return None,
	    };
            let rsi = market_data::rsi(&prices,lookback);
	    let l = rsi.len();
	    if l < min_len {
		min_len = l;
	    }
            hmap.insert(fname.clone(),rsi.clone());
	}
        if fname == "Moving-Avg-Cross" {
	    let lookback = match f.get("look-back-period")?.to_string().parse::<i32>() {
				Ok(i) => i as usize,
				Err(e) => return None,
	    };
    	    let cross = market_data::ma_cross(&prices,lookback);
	    let l = cross.len();
	    if l < min_len {
		min_len = l;
	    }
    	    hmap.insert(fname.clone(),cross.clone());
	}
        if fname == "Lower-Band-Cross" || fname == "Upper-Band-Cross" {
	    let lookback = match f.get("look-back-period")?.to_string().parse::<i32>() {
				Ok(i) => i as usize,
				Err(e) => return None,
	    };
	    let (lower,higher) = market_data::band_cross(&prices,lookback);
	    let l = lower.len();
	    if l < min_len {
		min_len = l;
	    }
            if fname == "Lower-Band-Cross"{
    	        hmap.insert(fname.clone(),lower.clone());
	    }
            if fname == "Upper-Band-Cross" {
    	        hmap.insert(fname.clone(),higher.clone());
	    }
	}
    }
    let n = prices.len();
    let _price_ref = &prices[n-min_len..n];
    for k in &features {
	let v = hmap.get_mut(k)?;
	let l = (*v).len();
	*v = (*v)[l-min_len..l].to_vec().clone();
    }
    return Some(TradeRLEnv::new(&(&_price_ref).to_vec(), &hmap, &features,train_test_split,shortsell,stoploss ));
}
