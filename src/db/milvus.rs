use async_trait::async_trait;
use std::error::Error;
use serde_json::Value;
use reqwest::Client;
use serde_json::json;
use log::{ info, error, debug, warn };
use super::{ VectorStore, IndexSchema };

pub struct MilvusVectorStore {
    client: Client,
    host: String,
    database: String,
    api_key: Option<String>,
    dimension: usize,
    metric: String,
}

impl MilvusVectorStore {
    pub async fn new(
        host: &str,
        api_key: Option<&str>,
        database: Option<&str>,
        dimension: usize,
        metric: &str
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let host = host.trim_end_matches('/').to_string();
        let database = database.unwrap_or("default").to_string();

        let normalized_metric = match metric.to_uppercase().as_str() {
            "COSINE" | "IP" | "L2" => metric.to_uppercase(),
            "COSINE_SIMILARITY" => "COSINE".to_string(),
            "DOT_PRODUCT" => "IP".to_string(),
            "EUCLIDEAN" => "L2".to_string(),
            _ => {
                return Err(
                    format!("Invalid metric type '{}' for Milvus. Use COSINE, IP, or L2 (or equivalents).", metric).into()
                );
            }
        };

        info!(
            "Initializing Milvus client for host: {}, db: {}, metric: {}, dim: {}",
            host,
            database,
            normalized_metric,
            dimension
        );

        Ok(Self {
            client: Client::new(),
            host,
            database,
            api_key: api_key.map(String::from),
            dimension,
            metric: normalized_metric,
        })
    }

    fn build_request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self.client.request(method, url);
        if let Some(token) = &self.api_key {
            if !token.is_empty() {
                builder = builder.header("Authorization", format!("Bearer {}", token));
            } else {
                warn!("Milvus API key/token provided but is empty.");
            }
        }
        builder.header("Accept", "application/json").header("Content-Type", "application/json")
    }

    async fn get_fields_from_samples(
        &self,
        collection: &str
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let url = &self.host;
        let db_name = &self.database;
        let dimension = self.dimension;
        let metric = &self.metric;

        let describe_url = format!("{}/v2/vectordb/collections/describe", url);
        let describe_req = self
            .build_request(reqwest::Method::POST, &describe_url)
            .json(&json!({ "dbName": db_name, "collectionName": collection }));

        let response = match describe_req.send().await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Error sending describe collection request for '{}': {}", collection, e);
                return Ok(Vec::new());
            }
        };

        let response_text = match response.text().await {
            Ok(text) => {
                debug!("Raw describe response for {}: {}", collection, text);
                text
            }
            Err(e) => {
                error!("Error getting describe response text for '{}': {}", collection, e);
                return Ok(Vec::new());
            }
        };

        let response_json: Value = match serde_json::from_str(&response_text) {
            Ok(json) => json,
            Err(e) => {
                warn!(
                    "Error parsing describe collection response JSON for '{}': {}. Response: {}",
                    collection,
                    e,
                    response_text
                );
                return Ok(Vec::new());
            }
        };

        let mut fields = Vec::new();
        if let Some(data) = response_json.get("data") {
            if let Some(schema) = data.get("schema") {
                if let Some(fields_array) = schema.get("fields").and_then(|f| f.as_array()) {
                    for field in fields_array {
                        if let Some(name) = field.get("name").and_then(|n| n.as_str()) {
                            if name != "id" && name != "vector" && !name.starts_with('$') {
                                fields.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }

        if fields.is_empty() {
            debug!("No fields found via describe for '{}', trying search...", collection);
            let dummy_vec: Vec<f32> = vec![0.0; dimension];
            let search_url = format!("{}/v2/vectordb/entities/search", url);
            let search_payload =
                json!({
                "dbName": db_name,
                "collectionName": collection,
                "data": [dummy_vec],
                "outputFields": ["*"],
                "limit": 5,
                "searchParams": {
                    "metricType": metric,
                    "params": { "nprobe": 10 }
                }
            });

            let search_req = self
                .build_request(reqwest::Method::POST, &search_url)
                .json(&search_payload);

            match search_req.send().await {
                Ok(search_resp) => {
                    let status = search_resp.status();
                    match search_resp.text().await {
                        Ok(search_text) => {
                            debug!("Search response status for {}: {}", collection, status);
                            debug!("Search response text for {}: {}", collection, search_text);
                            if status.is_success() {
                                match serde_json::from_str::<Value>(&search_text) {
                                    Ok(search_json) => {
                                        if
                                            let Some(data_array) = search_json
                                                .get("data")
                                                .and_then(|d| d.as_array())
                                        {
                                            for result_group in data_array {
                                                if
                                                    let Some(entities) = result_group
                                                        .get("entities")
                                                        .and_then(|e| e.as_array())
                                                {
                                                    for entity in entities {
                                                        if let Some(obj) = entity.as_object() {
                                                            for key in obj.keys() {
                                                                if
                                                                    key != "id" &&
                                                                    key != "vector" &&
                                                                    key != "score" &&
                                                                    key != "distance" &&
                                                                    !key.starts_with('$') &&
                                                                    !fields.contains(key)
                                                                {
                                                                    fields.push(key.clone());
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Error parsing search response JSON for '{}': {}. Response: {}",
                                            collection,
                                            e,
                                            search_text
                                        );
                                    }
                                }
                            } else {
                                warn!(
                                    "Search request failed for '{}' with status: {}",
                                    collection,
                                    status
                                );
                            }
                        }
                        Err(e) => {
                            error!(
                                "Error reading search response text for '{}': {}",
                                collection,
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    error!("Error sending search request for '{}': {}", collection, e);
                }
            }
        }

        if fields.is_empty() {
            debug!("No fields found via search, trying /collections/get...");
            let fields_url = format!("{}/v2/vectordb/collections/get", url);
            let fields_req = self
                .build_request(reqwest::Method::POST, &fields_url)
                .json(&json!({ "dbName": db_name, "collectionName": collection }));

            if let Ok(fields_resp) = fields_req.send().await {
                if let Ok(fields_json) = fields_resp.json::<Value>().await {
                    debug!("Collection info response (/get): {:?}", fields_json);

                    if let Some(data) = fields_json.get("data") {
                        if let Some(schema) = data.get("schema") {
                            if
                                let Some(fields_array) = schema
                                    .get("fields")
                                    .and_then(|f| f.as_array())
                            {
                                for field in fields_array {
                                    if let Some(name) = field.get("name").and_then(|n| n.as_str()) {
                                        if
                                            name != "id" &&
                                            name != "vector" &&
                                            !name.starts_with('$') &&
                                            !fields.contains(&name.to_string())
                                        {
                                            fields.push(name.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    warn!("Failed to parse JSON from /collections/get for {}", collection);
                }
            } else {
                warn!("Failed to send request to /collections/get for {}", collection);
            }
        }

        if !fields.is_empty() {
            debug!("Discovered fields for {}: {:?}", collection, fields);
        } else {
            warn!("Could not discover fields for collection '{}'", collection);
        }

        Ok(fields)
    }
}

#[async_trait]
impl VectorStore for MilvusVectorStore {
    async fn search(
        &self,
        query_vec: &[f32],
        limit: usize,
        topic: &str,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        let url = &self.host;
        let db_name = &self.database;
        let metric = &self.metric;

        let output_fields = match schema_fields {
            Some(fields) if !fields.is_empty() => {
                let mut requested = fields.clone();
                if !requested.contains(&"id".to_string()) {
                    requested.push("id".to_string());
                }
                requested
            }
            _ => vec!["*".to_string()],
        };
        debug!("Milvus search requesting output fields: {:?}", output_fields);

        let search_url = format!("{}/v2/vectordb/entities/search", url);
        let payload =
            json!({
            "dbName": db_name,
            "collectionName": topic,
            "data": [query_vec],
            "outputFields": output_fields,
            "limit": limit,
            "searchParams": {
                "metricType": metric,
                "params": { "nprobe": 10 }
            }
        });

        let search_req = self.build_request(reqwest::Method::POST, &search_url).json(&payload);

        let response = search_req
            .send().await
            .map_err(|e| format!("Milvus search request failed: {}", e))?;

        let status = response.status();
        let response_text = response
            .text().await
            .map_err(|e| { format!("Failed to read Milvus search response text: {}", e) })?;

        if !status.is_success() {
            error!(
                "Milvus search failed for collection {} (status {}): {}",
                topic,
                status,
                response_text
            );
            return Err(
                format!("Milvus search failed with status {}: {}", status, response_text).into()
            );
        }

        let response_json: Value = serde_json
            ::from_str(&response_text)
            .map_err(|e| {
                format!(
                    "Failed to parse Milvus search response JSON: {}. Response: {}",
                    e,
                    response_text
                )
            })?;

        let mut results = Vec::new();

        if let Some(data_array) = response_json.get("data").and_then(|d| d.as_array()) {
            if let Some(result_group) = data_array.first() {
                if let Some(entities) = result_group.get("entities").and_then(|e| e.as_array()) {
                    for item in entities {
                        let id_val = item.get("id");
                        let score_val = item.get("distance").or_else(|| item.get("score"));

                        let id = match id_val {
                            Some(Value::String(s)) => s.clone(),
                            Some(Value::Number(n)) => n.to_string(),
                            _ => {
                                continue;
                            }
                        };

                        let score = score_val
                            .and_then(|s| s.as_f64())
                            .map(|f| f as f32)
                            .unwrap_or(f32::MAX);

                        let mut clean_item = serde_json::Map::new();
                        if let Some(obj) = item.as_object() {
                            for (key, value) in obj {
                                if
                                    key != "id" &&
                                    key != "vector" &&
                                    key != "score" &&
                                    key != "distance" &&
                                    !key.starts_with('$')
                                {
                                    clean_item.insert(key.clone(), value.clone());
                                }
                            }
                        }
                        results.push((score, id, Value::Object(clean_item)));
                    }
                }
            }
        }

        Ok(results)
    }

    async fn search_hybrid(
        &self,
        topic: &str,
        text_query: &str,
        query_vec: &[f32],
        limit: usize,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        info!("Performing Milvus hybrid search (vector + rerank) for query on topic: '{}'", topic);
        warn!("Milvus hybrid search text_query ('{}') is currently ignored by this implementation.", text_query);

        let url = &self.host;
        let db_name = &self.database;
        let metric = &self.metric;

        let output_fields = match schema_fields {
            Some(fields) if !fields.is_empty() => {
                let mut requested = fields.clone();
                if !requested.contains(&"id".to_string()) {
                    requested.push("id".to_string());
                }
                requested
            }
            _ => vec!["*".to_string()],
        };
        debug!("Milvus hybrid search requesting output fields: {:?}", output_fields);

        let hybrid_search_url = format!("{}/v2/vectordb/entities/hybrid_search", url);

        let payload =
            json!({
            "dbName": db_name,
            "collectionName": topic,
            "search": [
                {
                    "data": [query_vec],
                    "annsField": "vector",
                    "limit": limit * 2,
                    "params": {
                        "metricType": metric,
                        "params": { "nprobe": 10 }
                    }
                }
            ],
            "rerank": {
                "strategy": "rrf",
                "params": {
                    "k": 60
                }
            },
            "limit": limit,
            "outputFields": output_fields
        });

        debug!("Hybrid search payload: {}", serde_json::to_string_pretty(&payload)?);

        let search_req = self
            .build_request(reqwest::Method::POST, &hybrid_search_url)
            .json(&payload);

        let response = search_req
            .send().await
            .map_err(|e| format!("Hybrid search request failed: {}", e))?;

        let status = response.status();
        let response_text = response
            .text().await
            .map_err(|e| { format!("Failed to read hybrid search response text: {}", e) })?;

        debug!("Hybrid search response status: {}", status);
        debug!("Hybrid search response text: {}", response_text);

        if !status.is_success() {
            error!("Hybrid search failed with status {}: {}", status, response_text);
            return Err(
                format!("Hybrid search failed with status {}: {}", status, response_text).into()
            );
        }

        let response_json: Value = match serde_json::from_str(&response_text) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to parse hybrid search response JSON: {}", e);
                return Err(format!("Failed to parse hybrid search response JSON: {}", e).into());
            }
        };

        let mut results = Vec::new();
        if let Some(data_array) = response_json.get("data").and_then(|d| d.as_array()) {
            for item in data_array {
                let id_val = item.get("id");
                let score_val = item.get("score").or_else(|| item.get("distance"));

                let id = match id_val {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Number(n)) => n.to_string(),
                    _ => {
                        continue;
                    }
                };

                let score = score_val
                    .and_then(|s| s.as_f64())
                    .map(|f| f as f32)
                    .unwrap_or(0.0);

                let mut clean_item = serde_json::Map::new();
                if let Some(obj) = item.as_object() {
                    for (key, value) in obj {
                        if
                            key != "id" &&
                            key != "vector" &&
                            key != "score" &&
                            key != "distance" &&
                            !key.starts_with('$')
                        {
                            clean_item.insert(key.clone(), value.clone());
                        }
                    }
                }
                results.push((score, id, Value::Object(clean_item)));
            }
        }

        Ok(results)
    }

    async fn count_documents(&self, topic: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let url = &self.host;
        let db_name = &self.database;

        let stats_url = format!("{}/v2/vectordb/collections/get_stats", url);
        let payload = json!({ "dbName": db_name, "collectionName": topic });

        let stats_req = self.build_request(reqwest::Method::POST, &stats_url).json(&payload);

        let stats_resp = stats_req
            .send().await
            .map_err(|e| format!("Milvus get_stats request failed: {}", e))?;

        let status = stats_resp.status();
        let stats_text = stats_resp
            .text().await
            .map_err(|e| { format!("Failed to read Milvus get_stats response text: {}", e) })?;

        debug!("Stats response status for {}: {}", topic, status);
        debug!("Stats response text for {}: {}", topic, stats_text);

        if !status.is_success() {
            error!("Failed to get stats for {}: {}", topic, stats_text);
            return Err(format!("Failed to get stats for {}: {}", topic, stats_text).into());
        }

        let stats_json: Value = match serde_json::from_str(&stats_text) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to parse stats JSON for {}: {}", topic, e);
                return Err(format!("Failed to parse stats JSON for {}: {}", topic, e).into());
            }
        };

        let count = stats_json
            .get("data")
            .and_then(|data| data.get("rowCount"))
            .and_then(|count_val| {
                if count_val.is_string() {
                    count_val.as_str().and_then(|s| s.parse::<u64>().ok())
                } else if count_val.is_number() {
                    count_val.as_u64()
                } else {
                    None
                }
            })
            .unwrap_or(0) as usize;

        debug!("Total count for {}: {}", topic, count);
        Ok(count)
    }

    async fn generate_schema(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        info!("Starting Milvus schema generation…");

        let url = &self.host;
        let db_name = &self.database;
        let mut collections = Vec::new();

        let list_url = format!("{}/v2/vectordb/collections/list", url);
        let list_payload = json!({ "dbName": db_name });
        let list_req = self.build_request(reqwest::Method::POST, &list_url).json(&list_payload);

        let list_resp_result = list_req.send().await;

        match list_resp_result {
            Ok(list_resp) => {
                let status = list_resp.status();
                match list_resp.text().await {
                    Ok(response_text) => {
                        debug!("List collections response status (Primary): {}", status);
                        debug!("List collections response body (Primary): {}", response_text);

                        if status.is_success() {
                            match serde_json::from_str::<Value>(&response_text) {
                                Ok(list_json) => {
                                    if let Some(data) = list_json.get("data") {
                                        if
                                            let Some(colls_array) = data
                                                .get("collections")
                                                .and_then(|c| c.as_array())
                                        {
                                            for coll_name in colls_array {
                                                if let Some(name) = coll_name.as_str() {
                                                    collections.push(name.to_string());
                                                }
                                            }
                                        }
                                    }
                                    if collections.is_empty() {
                                        if
                                            let Some(data_array) = list_json
                                                .get("data")
                                                .and_then(|d| d.as_array())
                                        {
                                            for coll in data_array {
                                                if let Some(name) = coll.as_str() {
                                                    collections.push(name.to_string());
                                                }
                                            }
                                        }
                                    }
                                    if collections.is_empty() {
                                        if
                                            let Some(colls) = list_json
                                                .get("collections")
                                                .and_then(|c| c.as_array())
                                        {
                                            for coll in colls {
                                                if let Some(name) = coll.as_str() {
                                                    collections.push(name.to_string());
                                                }
                                            }
                                        }
                                    }
                                    if collections.is_empty() && list_json.is_array() {
                                        if let Some(array) = list_json.as_array() {
                                            for coll in array {
                                                if
                                                    let Some(name) = coll
                                                        .get("collection_name")
                                                        .and_then(|n| n.as_str())
                                                {
                                                    collections.push(name.to_string());
                                                } else if
                                                    let Some(name) = coll
                                                        .get("name")
                                                        .and_then(|n| n.as_str())
                                                {
                                                    collections.push(name.to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to parse primary list collections JSON: {}. Response: {}",
                                        e,
                                        response_text
                                    );
                                }
                            }
                        } else {
                            warn!("Primary list collections request failed with status: {}", status);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read primary list collections response text: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to send primary list collections request: {}", e);
            }
        }

        if collections.is_empty() {
            warn!(
                "Primary list failed or yielded no collections, trying fallback GET /collections..."
            );
            let alt_url = format!("{}/collections", url);
            let alt_req = self.build_request(reqwest::Method::GET, &alt_url);

            match alt_req.send().await {
                Ok(alt_resp) => {
                    let alt_status = alt_resp.status();
                    match alt_resp.text().await {
                        Ok(alt_text) => {
                            debug!("List collections response status (Fallback): {}", alt_status);
                            debug!("List collections response body (Fallback): {}", alt_text);
                            if alt_status.is_success() {
                                match serde_json::from_str::<Value>(&alt_text) {
                                    Ok(alt_json) => {
                                        if alt_json.is_array() {
                                            if let Some(array) = alt_json.as_array() {
                                                for coll in array {
                                                    if
                                                        let Some(name) = coll
                                                            .get("collection_name")
                                                            .and_then(|n| n.as_str())
                                                    {
                                                        collections.push(name.to_string());
                                                    } else if
                                                        let Some(name) = coll
                                                            .get("name")
                                                            .and_then(|n| n.as_str())
                                                    {
                                                        collections.push(name.to_string());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) =>
                                        warn!("Failed to parse fallback list collections JSON: {}", e),
                                }
                            } else {
                                warn!("Fallback list collections request failed with status: {}", alt_status);
                            }
                        }
                        Err(e) =>
                            warn!("Failed to read fallback list collections response text: {}", e),
                    }
                }
                Err(e) => warn!("Failed to send fallback list collections request: {}", e),
            }
        }

        if collections.is_empty() {
            warn!(
                "WARNING: Couldn't retrieve collections from API via primary or fallback methods. Check API endpoint and authentication."
            );
            let wrapped = json!({ "indexes": Vec::<IndexSchema>::new() });
            let parent = std::path::Path
                ::new(output_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            std::fs::create_dir_all(parent)?;
            std::fs::write(output_path, serde_json::to_string_pretty(&wrapped)?)?;
            info!("✅ Wrote empty schema file as no collections were found.");
            return Ok(Vec::new());
        }

        info!("Discovered collections: {:?}", collections);

        let mut schemas = Vec::new();
        for collection in collections {
            info!("Inspecting collection {}", collection);
            let fields = self.get_fields_from_samples(&collection).await?;

            let final_fields = if fields.is_empty() {
                warn!("⚠️ No fields found for collection '{}', using default fields.", collection);
                vec!["text".to_string()]
            } else {
                info!("-> Found fields for {}: {:?}", collection, fields);
                fields
            };

            schemas.push(IndexSchema {
                name: collection.clone(),
                fields: final_fields,
                prefix: format!("milvus '{}'", collection),
            });
        }

        schemas.sort_by(|a, b| a.name.cmp(&b.name));

        if !schemas.is_empty() {
            let wrapped = json!({ "indexes": schemas.clone() });
            let parent = std::path::Path
                ::new(output_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            std::fs::create_dir_all(parent)?;
            std::fs::write(output_path, serde_json::to_string_pretty(&wrapped)?)?;
            info!(
                "✅ Generated schema for {} Milvus collections to {}",
                schemas.len(),
                output_path
            );
        } else {
            warn!("⚠️ No schemas were generated (collections might have failed processing).");
        }
        info!("Milvus schema generation finished.");
        Ok(schemas)
    }
}
