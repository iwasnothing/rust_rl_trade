use chrono::{DateTime, Utc, Duration };
use serde::{Deserialize, Serialize};
use std::error;
use std::fmt;
use base64::{Engine as _, engine::general_purpose};
use std::collections::HashMap;
use serde_json::value::Value;

#[derive(Debug,Copy,Clone)]
pub struct DBError;
impl fmt::Display for DBError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error in access DB data")
    }
}
impl error::Error for DBError {}

#[derive(Debug,Clone)]
pub struct ReqData {
    pub asset: String, 
    pub epi: i64, 
    pub feature_list: Vec<HashMap<String,String>>,
    pub freq: String, 
    pub id: String, 
    pub name: String, 
    pub num_years: String, 
    pub short_Sell: String, 
    pub split_pct: String, 
    pub status: String,
    pub stop_loss: String, 
    pub timestamp: i64,
}
pub fn parse_req_data(val: &Value) -> Option<ReqData> {
    let mut feature_map = val.get("feature-list")?.as_object()?;
    let mut feature_list = Vec::new();
    for (k,v) in feature_map.iter() {
        let mut _feature = HashMap::<String,String>::new();
        let _name = v.get("name")?.as_str()?.to_string();
	_feature.insert("name".to_string(),_name);
	let args = v.get("arguments")?.as_object()?;
        for (arg_name,arg_val) in args.iter() {
	    _feature.insert(arg_name.to_string(), arg_val.as_str()?.to_string() );
	}
	feature_list.push(_feature.clone());
    }
    return Some( ReqData {
		    asset: val.get("asset")?.as_str()?.to_string(),
		    epi: val.get("epi")?.as_i64()?,
		    feature_list: feature_list.clone(),
		    freq: val.get("freq")?.as_str()?.to_string(),
		    id: val.get("id")?.as_str()?.to_string(),
		    name: val.get("name")?.as_str()?.to_string(),
		    num_years: val.get("num_years")?.as_str()?.to_string(),
		    short_Sell: val.get("short_Sell")?.as_str()?.to_string(),
		    split_pct: val.get("split_pct")?.as_str()?.to_string(),
		    status: val.get("status")?.as_str()?.to_string(),
		    stop_loss: val.get("stop_loss")?.as_str()?.to_string(),
		    timestamp: val.get("timestamp")?.as_i64()?,
	});
}
    
        

pub async fn run_db_query(sql: &str) -> Result<Value,Box<dyn error::Error>> {
    let db_password = "root:root".as_bytes();
    let encoded_auth: String = format!("Basic {}", general_purpose::STANDARD_NO_PAD.encode(db_password) );
    println!("auth: {:?}", &encoded_auth);
    let uri = "https://backtest.fly.dev/sql";
    let client = reqwest::Client::new();
    let response = client.post(uri)
        .header("Accept","application/json")
        .header("NS","test")
        .header("DB","test")
        .header("authorization",encoded_auth)
        .body(sql.to_string())
        .send()
        .await?;
    match response.status() {
        reqwest::StatusCode::OK => {
            match response.json::<Value>().await {
                Ok(parsed) => return Ok(parsed),
                Err(e) => return Err(Box::new(e))
            };
        }
        reqwest::StatusCode::UNAUTHORIZED => {
            println!("Need to grab a new token");
            return Err(Box::new(DBError));
        }
        other => {
            println!("Uh oh! Something unexpected happened: {:?}", other);
            return Err(Box::new(DBError));
        }
    };
}
pub fn pop_db_result(val: &Value) -> Option<Value> {
	let _array = val.as_array()?;
	if _array.len() > 0 {
		return Some((_array[0].get("result")?).clone());
	} else {
		return None;
	}
}
pub fn pop_db_result_status(val: &Value) -> Option<String> {
	let _array = val.as_array()?;
	if _array.len() > 0 {
		return Some((_array[0].get("status")?).as_str()?.to_string());
	} else {
		return None;
	}
}
pub async fn log2db(reqid: &str, i_episode: &i64, mean_loss: &f64, total_reward: &f32) { 
    let sql_text = format!("create Training_log set reqid = \'{}\', epi={}, loss={:.2}, reward = {:.2}", reqid, i_episode, mean_loss, total_reward);
    println!("log sql: {:?}", &sql_text);
    let resp = run_db_query(&sql_text).await;
    println!("log db status: {:?}", resp);
}
pub async fn reject_req() {
    let sql_text = "let $req  = ( select id from REQ where status = \'new\' order by timestamp ASC LIMIT 1 ); update $req set status = \'rejected\' RETURN after;";
    println!("update req sql: {:?}", &sql_text);
    let resp = run_db_query(&sql_text).await;
    println!("update req status: {:?}", resp);
}
pub async fn update_req_status(reqid: String, status: String) {
    let sql_text = format!("update {} set status= \'{}\'", reqid, status);
    println!("update req sql: {:?}", &sql_text);
    let resp = run_db_query(&sql_text).await;
    println!("update req status: {:?}", resp);
}
pub async fn claim_req() -> Option<ReqData> {
    let sql_text = "select * from REQ where status = 'new' or status = 'running' order by timestamp LIMIT 1";
    let resp = run_db_query(sql_text).await;
    let result = match resp {
        Ok(p) => p,
        Err(e) => return None,
    };
    let obj_array = pop_db_result(&result)?.as_array()?.clone();
    if obj_array.len() > 0 {
	let obj = &obj_array[0];
	let rq =  parse_req_data(obj)?;
        update_req_status((&rq).id.clone(), "running".to_string()).await;
	return Some(rq);
    } else {
	return None;
    }

}
pub fn sign0f32(x: f32) -> f32 {
    if x > 0f32 {
	return 1f32;
    } else if x < 0f32 {
        return -1f32;
    } else {
        return 0f32;
    }
}
pub async fn perf_report(reqid: &str,epi: i64,  position_history: &Vec<f32>,  prices: &Vec<f32>) -> Option<String>{
    let num = position_history.len();
    let num_prices = prices.len();
    println!("pos len = {}, prices len = {}", num, num_prices);
    assert!(num == num_prices);
    let mut action_list: Vec<f32> = Vec::with_capacity(num);
    let mut asset_list: Vec<f32> = Vec::with_capacity(num);
    let mut pl_list: Vec<f32> = Vec::with_capacity(num);
    let capital = 1000f32;
    let mut cash = 1000f32;
    let mut accum_pl = 0f32;
    let mut prev_buy = 0f32;
    let mut win = 0f32;
    let mut total_trade = 0f32;
    action_list.push(sign0f32(position_history[0]));
    asset_list.push(cash+position_history[0]*prices[0]-capital);
    for i in 1..num {
        let action = sign0f32(position_history[i] - position_history[i-1]);
        if action.abs() < 0.01 {
            action_list.push(0f32);
        }
        else {
            action_list.push(action);
            cash  = cash-(position_history[i]-position_history[i-1])*prices[i];
	    if position_history[i] == 0f32 {
		let _pl = (prices[i] - prev_buy)*position_history[i-1];
		pl_list.push(_pl);
		accum_pl += _pl;
                total_trade += 1.0;
        	if _pl > 0f32 {
            	    win += 1.0;
		}
	    } else {
		prev_buy = prices[i]
	    }
        }
        asset_list.push(cash+position_history[i]*prices[i]-capital);
    }
    println!("action len = {}, asset len = {}", action_list.len(), asset_list.len());
    let avg_PL =  accum_pl/total_trade/capital * 100.0;
    update_req_status(reqid.to_string(), "completed".to_string()).await;
    println!("annual PL per trade = {:.2}/{}={:.2}", &accum_pl,&total_trade,&avg_PL);
    let sql = format!("update {} set total_trade={:.2}, win={:.2} , return={:.2}, position={:?}, price={:?}, asset={:?}, profit={:?}, action={:?}, epi={}",
                reqid.replace("REQ","RESULT"),&total_trade,&win,&avg_PL, position_history, prices,&asset_list, &pl_list, &action_list, epi);
    let resp = run_db_query(&sql).await;
    let result = match resp {
        Ok(p) => return pop_db_result_status(&p),
        Err(e) => return None,
    };
}


