//! MiMo cluster latency measurement and selection.
//!
//! Measures round-trip latency to each MiMo cluster endpoint
//! so the user can pick the fastest one for their region.

use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;

/// A MiMo cluster with its measured latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MimoCluster {
    /// Human-readable label (e.g. "China (cn)").
    pub label: String,
    /// Full base URL (e.g. "https://token-plan-cn.xiaomimimo.com").
    pub url: String,
    /// Measured latency in milliseconds, or None if unreachable.
    pub latency_ms: Option<u64>,
}

/// Known MiMo token-plan clusters.
const CLUSTERS: &[(&str, &str)] = &[
    ("China (cn)", "https://token-plan-cn.xiaomimimo.com"),
    ("Singapore (sgp)", "https://token-plan-sgp.xiaomimimo.com"),
    ("Europe (ams)", "https://token-plan-ams.xiaomimimo.com"),
];

/// Measure latency to all known MiMo clusters.
///
/// Sends an authenticated GET request to each cluster's `/v1/models`
/// endpoint with the given API key. Clusters that reject the key (401)
/// are excluded — a token-plan key only works on its assigned region.
pub async fn measure_cluster_latencies(api_key: &str) -> Vec<MimoCluster> {
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    let mut results: Vec<MimoCluster> = Vec::new();

    for (label, url) in CLUSTERS {
        let (latency_ms, is_authorized) = measure_one(&client, url, api_key).await;
        if is_authorized {
            results.push(MimoCluster {
                label: label.to_string(),
                url: url.to_string(),
                latency_ms,
            });
        }
    }

    // Sort: reachable first (by latency).
    results.sort_by_key(|c| c.latency_ms.unwrap_or(u64::MAX));

    results
}

/// Returns (latency_ms, is_authorized).
/// `is_authorized` is false if the cluster returns 401 (wrong region for this key).
async fn measure_one(client: &Client, base_url: &str, api_key: &str) -> (Option<u64>, bool) {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let start = Instant::now();

    match timeout(
        Duration::from_secs(3),
        client.get(&url).header("api-key", api_key).send(),
    )
    .await
    {
        Ok(Ok(response)) => {
            let ms = Some(start.elapsed().as_millis() as u64);
            // 401 = wrong region for this key; exclude it.
            (ms, response.status().as_u16() != 401)
        }
        Ok(Err(_)) => (None, false),
        Err(_) => (None, false), // timeout
    }
}
