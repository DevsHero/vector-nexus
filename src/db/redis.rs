use redis::aio::MultiplexedConnection;
use redis::{ Client, RedisError, cmd, FromRedisValue, Value as RedisValue };
use serde_json::Value;
use std::error::Error;
use async_trait::async_trait;
use std::collections::{ HashSet, HashMap };
use log::{ info, error, debug, warn };

use super::IndexSchema;
use super::VectorStore;

pub struct RedisVectorStore {
    client: Client,
}

impl RedisVectorStore {
    pub fn new(host: &str) -> Result<Self, RedisError> {
        info!("Connecting to Redis at {}", host);
        Ok(Self {
            client: Client::open(host)?,
        })
    }

    pub async fn get_connection(&self) -> Result<MultiplexedConnection, RedisError> {
        self.client.get_multiplexed_async_connection().await
    }

    pub async fn scan_keys(
        &self,
        pattern: &str
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let mut con = self.get_connection().await?;
        let mut cursor: u64 = 0;
        let mut all_keys = Vec::new();
        debug!("Scanning Redis keys with pattern: {}", pattern);
        loop {
            let (next_cursor, batch): (u64, Vec<String>) = cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(pattern)
                .arg("COUNT")
                .arg("1000")
                .query_async(&mut con).await?;

            all_keys.extend(batch);

            if next_cursor == 0 {
                debug!("SCAN finished, found {} keys matching '{}'", all_keys.len(), pattern);
                break;
            }
            cursor = next_cursor;
        }
        Ok(all_keys)
    }

    fn normalize_index_name(topic: &str) -> String {
        if topic.starts_with("idx:") {
            topic.to_string()
        } else if let Some(n) = topic.strip_prefix("redis:") {
            format!("idx:{}", n)
        } else if let Some(n) = topic.strip_prefix("qdrant:") {
            format!("idx:{}", n)
        } else {
            format!("idx:{}", topic)
        }
    }
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    async fn search(
        &self,
        query_vec: &[f32],
        limit: usize,
        topic: &str,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        let mut con = self.get_connection().await?;
        let index_name = Self::normalize_index_name(topic);
        debug!(
            "Redis vector search on index '{}' with limit {}. Requesting fields: {:?}",
            index_name,
            limit,
            schema_fields.unwrap_or(&Vec::new())
        );

        let query_blob: Vec<u8> = query_vec
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let query_str = format!("*=>[KNN {} @vector $vec AS vector_score]", limit);

        let mut ft_cmd = cmd("FT.SEARCH");
        ft_cmd
            .arg(&index_name)
            .arg(&query_str)
            .arg("PARAMS")
            .arg("2")
            .arg("query_blob")
            .arg(&query_blob)
            .arg("DIALECT")
            .arg("2")
            .arg("RETURN")
            .arg("3")
            .arg("vector_score")
            .arg("id")
            .arg("$");

        let redis_result: RedisValue = ft_cmd.query_async(&mut con).await?;
        debug!("Raw FT.SEARCH Result for index {}: {:?}", index_name, redis_result);

        let results = parse_ft_search_results(redis_result, schema_fields).await;
        debug!("Parsed {} results from Redis vector search.", results.len());
        Ok(results)
    }

    async fn search_hybrid(
        &self,
        topic: &str,
        text_query: &str,
        query_vec: &[f32],
        limit: usize,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, serde_json::Value)>, Box<dyn Error + Send + Sync>> {
        let mut conn = self.get_connection().await?;
        let index_name = Self::normalize_index_name(topic);
        debug!(
            "Redis hybrid search (using vector only) on index '{}' with limit {}, text query ignored: '{}'. Requesting fields: {:?}",
            index_name,
            limit,
            text_query,
            schema_fields.unwrap_or(&Vec::new())
        );

        let query_blob: Vec<u8> = query_vec
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let query_str = format!("*=>[KNN {} @vector $vec AS vector_score]", limit);

        debug!("DEBUG: Hybrid query (using vector only): {}", query_str);

        let mut ft_cmd = cmd("FT.SEARCH");
        ft_cmd
            .arg(&index_name)
            .arg(&query_str)
            .arg("PARAMS")
            .arg("2")
            .arg("vec")
            .arg(&query_blob)
            .arg("DIALECT")
            .arg("2")
            .arg("RETURN")
            .arg("3")
            .arg("vector_score")
            .arg("id")
            .arg("$");

        let redis_result: RedisValue = ft_cmd.query_async(&mut conn).await?;
        debug!(
            "Raw FT.SEARCH Hybrid (Vector Only) Result for index {}: {:?}",
            index_name,
            redis_result
        );

        let results = parse_ft_search_results(redis_result, schema_fields).await;
        debug!("Parsed {} results from Redis hybrid (vector only) search.", results.len());
        Ok(results)
    }

    async fn count_documents(&self, topic: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let mut con = self.get_connection().await?;
        let index_name = Self::normalize_index_name(topic);
        debug!("Counting documents in Redis index '{}'", index_name);

        let mut ft = cmd("FT.SEARCH");
        ft.arg(&index_name).arg("*").arg("LIMIT").arg("0").arg("0");

        let res: RedisValue = ft.query_async(&mut con).await?;
        debug!("Raw FT.SEARCH count result for {}: {:?}", index_name, res);

        if let RedisValue::Bulk(arr) = res {
            if !arr.is_empty() {
                let count = usize::from_redis_value(&arr[0])?;
                debug!("Count for index '{}': {}", index_name, count);
                return Ok(count);
            } else {
                warn!("FT.SEARCH for count returned an unexpected empty array for index '{}'", index_name);
            }
        } else {
            warn!(
                "FT.SEARCH for count returned unexpected format for index '{}': {:?}",
                index_name,
                res
            );
        }
        Ok(0)
    }

    async fn generate_schema(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        info!("Starting Redis schema generation...");
        let mut conn = self.get_connection().await?;

        let all_keys = self.scan_keys("item:*:*").await?;
        if all_keys.is_empty() {
            warn!("No keys found matching 'item:*:*'. Cannot generate schema.");
            let wrapped = serde_json::json!({ "indexes": Vec::<IndexSchema>::new() });
            if let Some(parent) = std::path::Path::new(output_path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(output_path, serde_json::to_string_pretty(&wrapped)?)?;
            info!("Wrote empty schema file to {}", output_path);
            return Ok(Vec::new());
        }

        let mut index_types = HashSet::new();
        for key in &all_keys {
            if let Some(type_name) = key.split(':').nth(1) {
                if !type_name.is_empty() {
                    index_types.insert(type_name.to_string());
                }
            }
        }

        info!("Found index types: {:?}", index_types);

        let mut schemas = Vec::new();
        for index_name in index_types {
            info!("Processing index type: {}", index_name);
            let sample_key_opt = all_keys
                .iter()
                .find(|k| k.starts_with(&format!("item:{}:", index_name)));

            let mut fields = Vec::new();
            if let Some(sample_key) = sample_key_opt {
                debug!("Using sample key '{}' for index type '{}'", sample_key, index_name);
                let json_result: Result<String, RedisError> = cmd("JSON.GET")
                    .arg(sample_key)
                    .query_async(&mut conn).await;

                match json_result {
                    Ok(json_str) => {
                        match serde_json::from_str::<Value>(&json_str) {
                            Ok(Value::Object(map)) => {
                                fields = map
                                    .keys()
                                    .cloned()
                                    .filter(|k| k != "vector")
                                    .collect();
                                fields.sort();
                                info!(
                                    "Extracted fields for '{}' from JSON: {:?}",
                                    index_name,
                                    fields
                                );
                            }
                            Ok(_) => warn!("Document {} is not a JSON object.", sample_key),
                            Err(e) => warn!("Failed to parse JSON for {}: {}", sample_key, e),
                        }
                    }
                    Err(e) => {
                        error!("Redis error getting JSON for key {}: {}", sample_key, e);
                    }
                }
            } else {
                warn!("No sample key found for index type '{}'", index_name);
            }

            if fields.is_empty() {
                warn!("No fields extracted for index type '{}'. Using default 'text'.", index_name);
                fields.push("text".to_string());
            }
            schemas.push(IndexSchema {
                name: index_name.clone(),
                prefix: format!("redis:{}", index_name),
                fields,
            });
        }

        schemas.sort_by(|a, b| a.name.cmp(&b.name));

        if !schemas.is_empty() {
            let wrapped = serde_json::json!({ "indexes": schemas.clone() });
            if let Some(parent) = std::path::Path::new(output_path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(output_path, serde_json::to_string_pretty(&wrapped)?)?;
            info!("✅ Generated {} index schemas to {}", schemas.len(), output_path);
        } else {
            warn!("No schemas generated (no valid index types found?). Schema file not written.");
        }
        info!("Redis schema generation finished.");
        Ok(schemas)
    }
}

async fn parse_ft_search_results(
    raw: RedisValue,
    schema_fields: Option<&Vec<String>>
) -> Vec<(f32, String, serde_json::Value)> {
    let mut out = Vec::new();
    if let RedisValue::Bulk(mut arr) = raw {
        if arr.is_empty() {
            warn!("parse_ft_search_results received empty array.");
            return out;
        }
        let _count = match usize::from_redis_value(&arr[0]) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to parse count from FT.SEARCH result: {}", e);
                return out;
            }
        };
        arr.drain(0..1);

        for chunk in arr.chunks(2) {
            if chunk.len() < 2 {
                warn!("Skipping incomplete chunk in FT.SEARCH results: {:?}", chunk);
                continue;
            }
            let id = match String::from_redis_value(&chunk[0]) {
                Ok(id_str) => id_str,
                Err(e) => {
                    error!("Failed to parse document ID from FT.SEARCH result: {}", e);
                    continue;
                }
            };

            let mut score = f32::MAX;
            let mut doc: Option<Value> = None;

            if let RedisValue::Bulk(pairs) = &chunk[1] {
                let data_map: HashMap<String, RedisValue> = pairs
                    .chunks(2)
                    .filter_map(|p| {
                        if p.len() == 2 {
                            String::from_redis_value(&p[0])
                                .ok()
                                .map(|key| (key, p[1].clone()))
                        } else {
                            warn!("Skipping incomplete pair in FT.SEARCH data: {:?}", p);
                            None
                        }
                    })
                    .collect();

                if
                    let Some(score_val) = data_map
                        .get("vector_score")
                        .or_else(|| data_map.get("hybrid_score"))
                {
                    if let Ok(score_str) = String::from_redis_value(score_val) {
                        if let Ok(dist) = score_str.parse::<f32>() {
                            score = 1.0 - dist;
                        } else {
                            warn!("Failed to parse score string '{}' for ID {}", score_str, id);
                        }
                    } else {
                        warn!("Failed to convert score value to string for ID {}", id);
                    }
                } else {
                    warn!("Score field ('vector_score' or 'hybrid_score') not found for ID {}", id);
                }

                if let Some(doc_val) = data_map.get("$") {
                    if let Ok(json_str) = String::from_redis_value(doc_val) {
                        match serde_json::from_str::<Value>(&json_str) {
                            Ok(parsed_doc) => {
                                if
                                    let (Some(fields_to_keep), Value::Object(mut obj)) = (
                                        schema_fields,
                                        parsed_doc.clone(),
                                    )
                                {
                                    obj.retain(|k, _| fields_to_keep.contains(k));
                                    doc = Some(Value::Object(obj));
                                } else {
                                    doc = Some(parsed_doc);
                                }
                            }
                            Err(e) => error!("Failed to parse document JSON for ID {}: {}", id, e),
                        }
                    } else {
                        error!("Failed to convert document value to string for ID {}", id);
                    }
                } else {
                    error!("Document content field '$' not found for ID {}", id);
                }
            } else {
                warn!("Expected an array of pairs for document data, got: {:?}", chunk[1]);
            }

            if let Some(doc_content) = doc {
                if score != f32::MAX {
                    out.push((score, id, doc_content));
                } else {
                    warn!("Skipping result for ID {} due to missing or invalid score.", id);
                }
            } else {
                warn!("Skipping result for ID {} due to missing document content.", id);
            }
        }
    } else {
        error!("FT.SEARCH result was not a Redis Array: {:?}", raw);
    }
    out
}
