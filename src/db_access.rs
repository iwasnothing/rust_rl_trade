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
pub async fn claim_req() -> Option<ReqData> {
    let sql_text = "select * from REQ order by timestamp LIMIT 1";
    let resp = run_db_query(sql_text).await;
    let result = match resp {
        Ok(p) => p,
        Err(e) => return None,
    };
    let obj_array = pop_db_result(&result)?.as_array()?.clone();
    if obj_array.len() > 0 {
	let obj = &obj_array[0];
	return  parse_req_data(obj);
    } else {
	return None;
    }

}
pub async fn perf_report(reqid: &str,epi: i64, cumulate_PL: f32, PL: &Vec<f32>, position_history: &Vec<f32>, action_list: &Vec<usize>, prices: &Vec<f32>, asset_value: &Vec<f32>) -> Option<String>{
    let total_trade = PL.len();
    let mut win = 0;
    for v in PL {
        if *v > 0f32 {
            win += 1;
        }
    }
    let sql = format!("update {} set total_trade={}, win={} , return={}, position={:?}, price={:?}, asset={:?}, profit={:?}, action={:?}, epi={}",
                reqid.replace("REQ","RESULT"),&total_trade,&win,&100.0*cumulate_PL, position_history, prices,asset_value, PL, action_list, epi);
    let resp = run_db_query(&sql).await;
    let result = match resp {
        Ok(p) => return pop_db_result_status(&p),
        Err(e) => return None,
    };
}

