use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration };
use serde::{Deserialize, Serialize};
use std::error;
use std::fmt;
use statistical::{mean,population_standard_deviation};

const api_data_limit: usize = 1000;
#[derive(Debug,Copy,Clone)]
pub struct DataSrcError;
#[derive(Serialize, Deserialize, Debug,  Clone)]
pub struct Bar {
    t: String,
    o: f32,
    h: f32,
    l: f32,
    c: f32,
    v: i32,
    n: i32,
    vw: f32
}
#[derive(Serialize, Deserialize, Debug)]
pub struct BarsData {
    bars: Vec<Bar>
}
impl BarsData {
    pub fn to_price_vec(&self) -> Vec<f32> {
        let mut vec: Vec<f32> = Vec::with_capacity(api_data_limit);
        for b in &self.bars {
            vec.push(b.c);
        }
        return vec;
    }
    pub fn to_datetime_vec(&self) -> Vec<String> {
        let mut time_vec: Vec<String> = Vec::with_capacity(api_data_limit);
        for b in &self.bars {
            time_vec.push(b.t.clone());
        }
        return time_vec;
    }
}
impl fmt::Display for DataSrcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error in download data")
    }
}
impl error::Error for DataSrcError {}

pub async fn download_data(sym: &str, nyrs: i32, freq: &str) -> Result<BarsData,Box<dyn error::Error>> {
    let host="https://data.alpaca.markets";
    let end_date_str = (Utc::now()-Duration::days(1)).format("%Y-%m-%dT%H:%M:%SZ");
    let start_date_str = (Utc::now()-Duration::days((nyrs*365) as i64)).format("%Y-%m-%dT%H:%M:%SZ");
    //let freq="1Day";
    println!("{},{}",&start_date_str,&end_date_str);
    let api_key = "PKFNMK9UBQPSX57OC3JK";
    let api_secret ="VVp9vMTjNvSEUqJJuqpRLw5r4oUq2uFGbX2VipCE";

    let uri=format!("{}/v2/stocks/{}/bars?start={}&end={}&timeframe={}",&host,sym,&start_date_str,&end_date_str,&freq);
    println!("{}",uri);
    let client = reqwest::Client::new();
    let response = client.get(&uri)
        .header("Apca-Api-Key-Id",api_key) 
        .header("Apca-Api-Secret-Key",api_secret)
        .send()
        .await?;
    match response.status() {
        reqwest::StatusCode::OK => {
            match response.json::<BarsData>().await {
                Ok(parsed) => return Ok(parsed),
                Err(e) => return Err(Box::new(e))
            };
        }
        reqwest::StatusCode::UNAUTHORIZED => {
            println!("Need to grab a new token");
            return Err(Box::new(DataSrcError));
        }
        other => {
            println!("Uh oh! Something unexpected happened: {:?}", other);
            return Err(Box::new(DataSrcError));
        }
    };
    
}
pub fn pctchg(price: &Vec<f32>) -> Vec<f32>{
    let n = price.len();
    let mut pct: Vec<f32> = Vec::with_capacity(n-1);
    for i in 1..n {
        pct.push((price[i] - price[i-1])/price[i-1]);
    }
    return pct;
}
pub fn rsi(price: &Vec<f32>, lookback: usize) -> Vec<f32>{
    let n = price.len();
    let mut up: Vec<f32> = Vec::with_capacity(n-1);
    let mut down: Vec<f32> = Vec::with_capacity(n-1);
    let mut rsi: Vec<f32> = Vec::with_capacity(n-1);
    let pct = pctchg(price);
    for i in 0..n-1 {
        if pct[i] > 0.0 {
            up.push(pct[i]);
            down.push(0.0);
        } else {
            up.push(0.0);
            down.push(-1.0*pct[i]);
        }
    }
    for i in lookback-1..n-1 {
        let sum_up: f32 = up[i+1-lookback..i+1].iter().sum();
        let sum_down: f32 = down[i+1-lookback..i+1].iter().sum();
        rsi.push( 100.0 - ( 100.0 / (  1.0 +  sum_up / sum_down )));
    }

    return rsi;
}
pub fn ma(price: &Vec<f32>, lookback: usize) -> Vec<f32>{
    let n = price.len();
    let mut ma: Vec<f32> = Vec::with_capacity(n-1);
    for i in lookback-1..n {
        let sum_up: f32 = price[i+1-lookback..i+1].iter().sum();
        ma.push( sum_up/ (lookback as f32) );
    }

    return ma;
}
pub fn ma_cross(price: &Vec<f32>, lookback: usize) -> Vec<f32>{
    let n = price.len();
    let ma_vec = ma(price,lookback);
    let mut cross: Vec<f32> = Vec::with_capacity(n-1);
    for i in lookback-1..n {
        let diff = price[i] - ma_vec[i+1-lookback];
        cross.push(diff)
    }

    return cross;
}
pub fn upper_lower_band(price: &Vec<f32>,lookback: usize) -> (Vec<f32>,Vec<f32>) {
    let n = price.len();
    let ma_vec = ma(price,lookback);
    let mut std_vec: Vec<f32> = Vec::with_capacity(n);
    for i in lookback-1..n {
 	 let p = &price[i+1-lookback..i+1];
         let u: f32 = mean(p);
         let sigma: f32 = population_standard_deviation(p,Some(u));
	 std_vec.push(sigma);
    }
    let L = std_vec.len();
    let mut lower_vec: Vec<f32> = Vec::with_capacity(n);
    let mut upper_vec: Vec<f32> = Vec::with_capacity(n);
    for i in 0..L {
        lower_vec.push(ma_vec[i] - std_vec[i]);
        upper_vec.push(ma_vec[i] + std_vec[i]);
    }
    return (lower_vec,upper_vec);
}
pub fn band_cross(price: &Vec<f32>,lookback: usize) -> (Vec<f32>,Vec<f32>) {
    let ma_vec = ma(price,lookback);
    let (lower, upper) = upper_lower_band(price,lookback);
    let n = ma_vec.len();
    let mut lower_cross: Vec<f32> = Vec::with_capacity(n);
    let mut upper_cross: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        lower_cross.push(ma_vec[i] - lower[i]);
        upper_cross.push(ma_vec[i] - upper[i]);
    }
    return (lower_cross,upper_cross);
}
