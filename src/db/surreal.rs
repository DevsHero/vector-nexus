use async_trait::async_trait;
use std::error::Error;
use std::collections::HashSet;
use serde_json::Value;
use base64::{ engine::general_purpose::STANDARD, Engine as _ };
use reqwest::Client;
use log::{ info, error, debug, warn };
use super::{ VectorStore, IndexSchema };

pub struct SurrealVectorStore {
    client: Client,
    host: String,
    namespace: String,
    database: String,
    user: Option<String>,
    pass: Option<String>,
}

impl SurrealVectorStore {
    pub async fn new(
        host: &str,
        namespace: Option<&str>,
        database: Option<&str>,
        user: Option<&str>,
        pass: Option<&str>
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let client = Client::new();
        let host = host.trim_end_matches('/').to_string();
        let sql_url = format!("{}/sql", host);

        let namespace = namespace.ok_or("SurrealDB namespace not provided")?.to_string();
        let database = database.ok_or("SurrealDB database name not provided")?.to_string();

        let auth_header = if let (Some(u), Some(p)) = (user, pass) {
            if !u.is_empty() && !p.is_empty() {
                Some(format!("Basic {}", STANDARD.encode(format!("{}:{}", u, p))))
            } else {
                warn!("SurrealDB user or pass provided but empty.");
                None
            }
        } else {
            None
        };

        let define_ns_sql = format!("DEFINE NAMESPACE IF NOT EXISTS {};", namespace);
        info!("Sending DEFINE NAMESPACE: {}", define_ns_sql);
        let mut req_ns = client
            .post(&sql_url)
            .header("Content-Type", "text/plain")
            .header("Accept", "application/json")
            .body(define_ns_sql);

        if let Some(ref auth) = auth_header {
            req_ns = req_ns.header("Authorization", auth);
        }

        let resp_ns = req_ns.send().await?;
        let status_ns = resp_ns.status();
        let text_ns = resp_ns.text().await?;
        debug!("SurrealDB DEFINE NAMESPACE response: {}", text_ns);
        if !status_ns.is_success() && !text_ns.contains("already exists") {
            error!("Failed to execute DEFINE NAMESPACE (Status: {}): {}", status_ns, text_ns);
        }

        let define_db_sql = format!("DEFINE DATABASE IF NOT EXISTS {};", database);
        info!("Sending DEFINE DATABASE: {}", define_db_sql);
        let mut req_db = client
            .post(&sql_url)
            .header("Content-Type", "text/plain")
            .header("Accept", "application/json")
            .header("NS", &namespace)
            .body(define_db_sql);

        if let Some(ref auth) = auth_header {
            req_db = req_db.header("Authorization", auth);
        }

        let resp_db = req_db.send().await?;
        let status_db = resp_db.status();
        let text_db = resp_db.text().await?;
        debug!("SurrealDB DEFINE DATABASE response: {}", text_db);
        if !status_db.is_success() && !text_db.contains("already exists") {
            error!("Failed to execute DEFINE DATABASE (Status: {}): {}", status_db, text_db);
        }

        info!("SurrealDB client initialized for ns: {}, db: {}", namespace, database);
        Ok(Self {
            client,
            host,
            namespace,
            database,
            user: user.map(String::from),
            pass: pass.map(String::from),
        })
    }

    async fn execute_query(
        &self,
        query: &str
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let sql_url = format!("{}/sql", self.host);
        let ns = &self.namespace;
        let db = &self.database;

        let auth_header = if let (Some(u), Some(p)) = (&self.user, &self.pass) {
            Some(format!("Basic {}", STANDARD.encode(format!("{}:{}", u, p))))
        } else {
            None
        };

        let query_owned = query.to_string();

        let mut req = self.client
            .post(&sql_url)
            .header("Content-Type", "text/plain")
            .header("Accept", "application/json")
            .header("NS", ns)
            .header("DB", db)
            .header("Surreal-NS", ns)
            .header("Surreal-DB", db)
            .body(query_owned);

        if let Some(ref auth) = auth_header {
            req = req.header("Authorization", auth);
        }

        debug!("Executing query: {} with NS: {} DB: {}", query, ns, db);

        let resp = req.send().await?;
        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            error!("Query execution failed (Status: {}): {}", status, text);
            if let Ok(json_err) = serde_json::from_str::<Value>(&text) {
                error!("SurrealDB Error Details: {:?}", json_err);
            }
            return Err(format!("Query execution failed (Status: {}): {}", status, text).into());
        }

        debug!("Query successful. Response text: {}", text);

        match serde_json::from_str(&text) {
            Ok(json) => Ok(json),
            Err(e) => {
                error!("Failed to parse successful query response JSON: {}. Response: {}", e, text);
                Err(format!("Failed to parse query response JSON: {}", e).into())
            }
        }
    }

    async fn get_tables_from_info(&self) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let info_query = "INFO FOR DB;";
        info!("Executing INFO FOR DB query...");
        let v = self.execute_query(info_query).await?;
        debug!("Raw INFO FOR DB response: {:?}", v);

        let mut tables = Vec::new();

        if let Some(arr) = v.as_array() {
            debug!("INFO FOR DB response is an array. Length: {}", arr.len());
            if let Some(first_obj) = arr.first() {
                if let Some(res_obj) = first_obj.get("result").and_then(Value::as_object) {
                    debug!("Found 'result' object: {:?}", res_obj);
                    if let Some(tables_obj) = res_obj.get("tables").and_then(Value::as_object) {
                        debug!("Found 'tables' in result");
                        for name in tables_obj.keys() {
                            debug!("→ table: {}", name);
                            tables.push(name.clone());
                        }
                    } else {
                        debug!("'tables' key not found or not an object within 'result'.");
                    }
                } else {
                    warn!("'result' key not found in INFO FOR DB response, checking top level.");
                    if let Some(tables_obj) = first_obj.get("tables").and_then(Value::as_object) {
                        debug!("Found top-level 'tables' object");
                        for name in tables_obj.keys() {
                            debug!("→ table: {}", name);
                            tables.push(name.clone());
                        }
                    } else {
                        debug!("'tables' key not found or not an object at top level either.");
                    }
                }
            } else {
                warn!("INFO FOR DB response array is empty.");
            }
        } else {
            error!("INFO FOR DB response is not a JSON array: {:?}", v);
        }

        info!("Finished parsing INFO FOR DB. Found tables: {:?}", tables);
        Ok(tables)
    }

    async fn get_fields_from_info(
        &self,
        table: &str
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let q = format!("INFO FOR TABLE {};", table);
        debug!("Executing INFO FOR TABLE query: {}", q);
        let v = self.execute_query(&q).await?;
        debug!("Raw INFO FOR TABLE {} response: {:?}", table, v);

        let mut fields = Vec::new();
        if let Some(arr) = v.as_array() {
            if let Some(obj0) = arr.first() {
                if let Some(res) = obj0.get("result").and_then(Value::as_object) {
                    if let Some(fobj) = res.get("fields").and_then(Value::as_object) {
                        debug!("Found 'fields' in result for table '{}'", table);
                        for name in fobj.keys() {
                            if name != "id" && name != "vector" {
                                fields.push(name.clone());
                            }
                        }
                    } else {
                        debug!("'fields' key not found or not an object within 'result' for table '{}'.", table);
                    }
                } else {
                    debug!("'result' key not found in INFO FOR TABLE response for '{}'.", table);
                }
            } else {
                warn!("INFO FOR TABLE response array is empty for '{}'.", table);
            }
        } else {
            error!("INFO FOR TABLE response is not a JSON array for '{}': {:?}", table, v);
        }
        debug!("Fields found via introspection for '{}': {:?}", table, fields);
        Ok(fields)
    }

    async fn generate_schema_internal(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        info!("Starting SurrealDB schema generation…");

        let mut tables = match self.get_tables_from_info().await {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to get tables using INFO FOR DB: {}", e);
                Vec::new()
            }
        };

        if tables.is_empty() {
            warn!("INFO FOR DB failed or returned no tables, trying $system.table query...");
            match self.execute_query("RETURN SELECT VALUE tb FROM $system.table;").await {
                Ok(resp) => {
                    debug!("Raw $system.table response: {:?}", resp);
                    if let Some(arr) = resp.as_array() {
                        if let Some(first_obj) = arr.first() {
                            if
                                let Some(res_arr) = first_obj
                                    .get("result")
                                    .and_then(Value::as_array)
                            {
                                tables = res_arr
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .map(String::from)
                                    .collect();
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to query $system.table: {}", e);
                }
            }
        }

        if tables.is_empty() {
            warn!("⚠️ No tables discovered via INFO or $system.table. Writing empty schema.");
            let wrapped = serde_json::json!({ "indexes": Vec::<IndexSchema>::new() });
            if let Some(parent) = std::path::Path::new(output_path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(output_path, serde_json::to_string_pretty(&wrapped)?)?;
            info!("Wrote empty schema file to {}", output_path);
            return Ok(Vec::new());
        }
        info!("Discovered tables: {:?}", tables);

        let mut schemas = Vec::new();
        for table in tables {
            if table.starts_with('_') {
                debug!("Skipping internal table: {}", table);
                continue;
            }
            info!("Inspecting table {}", table);

            let mut fields = match self.get_fields_from_info(&table).await {
                Ok(f) => f,
                Err(e) => {
                    error!("Failed to get fields via INFO for table '{}': {}", table, e);
                    Vec::new()
                }
            };

            if fields.is_empty() {
                info!("-> Introspection failed or yielded no fields for '{}', falling back to sampling.", table);
                let sample_query = format!("SELECT * FROM {} LIMIT 5;", table);
                match self.execute_query(&sample_query).await {
                    Ok(sample_query_result) => {
                        debug!("Raw sample query result for {}: {:?}", table, sample_query_result);
                        let mut field_set = HashSet::new();
                        if let Some(result_array) = sample_query_result.as_array() {
                            if let Some(first_result) = result_array.first() {
                                if
                                    let Some(data_records) = first_result
                                        .get("result")
                                        .and_then(Value::as_array)
                                {
                                    debug!(
                                        "--> Found {} sample records in 'result' field for '{}'.",
                                        data_records.len(),
                                        table
                                    );
                                    for rec in data_records {
                                        if let Some(obj) = rec.as_object() {
                                            for k in obj
                                                .keys()
                                                .filter(|k| *k != "id" && *k != "vector") {
                                                field_set.insert(k.clone());
                                            }
                                        }
                                    }
                                } else {
                                    debug!("--> 'result' field not found or not an array in sample query response for table '{}'.", table);
                                }
                            }
                        } else {
                            warn!("--> Sample query response for table '{}' was not an array.", table);
                        }
                        fields = field_set.into_iter().collect();
                    }
                    Err(e) => {
                        error!("Failed to execute sample query for table '{}': {}", table, e);
                    }
                }
            } else {
                info!("-> Found fields via introspection for '{}': {:?}", table, fields);
            }

            if fields.is_empty() {
                warn!("⚠️ No fields found for table '{}' via introspection or sampling, using default fields.", table);
                fields = vec!["content".to_string()];
            }
            fields.sort();
            schemas.push(IndexSchema {
                name: table.clone(),
                prefix: format!("surreal:{}", table),
                fields,
            });
        }

        if !schemas.is_empty() {
            let wrapped = serde_json::json!({ "indexes": schemas.clone() });
            if let Some(parent) = std::path::Path::new(output_path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(output_path, serde_json::to_string_pretty(&wrapped)?)?;
            info!("✅ Generated schema for {} tables to {}", schemas.len(), output_path);
        } else {
            warn!("⚠️ No schemas were generated (after filtering/errors).");
        }
        info!("SurrealDB schema generation finished.");
        Ok(schemas)
    }

    async fn search_internal(
        &self,
        query_vec: &[f32],
        limit: usize,
        topic: &str,
        _schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        let query_vec_json = serde_json::to_string(query_vec)?;
        let query = format!(
            r#"
            SELECT
                id,
                vector::similarity::cosine(vector, <vector>{}) AS score,
                *
            FROM {}
            WHERE vector @@ <vector>{}
            ORDER BY score DESC
            LIMIT {};
            "#,
            query_vec_json,
            topic,
            query_vec_json,
            limit
        );
        debug!("Executing search query: {}", query);

        let result = self.execute_query(&query).await?;
        debug!("Raw search result: {:?}", result);

        let mut search_results = Vec::new();
        if let Some(result_array) = result.as_array() {
            if let Some(first_result) = result_array.first() {
                if let Some(data_records) = first_result.get("result").and_then(Value::as_array) {
                    debug!("Found {} search results in 'result' field.", data_records.len());
                    for item in data_records {
                        if let Some(item_obj) = item.as_object() {
                            let id = item_obj
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown_id")
                                .to_string();

                            let score = item_obj
                                .get("score")
                                .and_then(Value::as_f64)
                                .map(|s| s as f32)
                                .unwrap_or(0.0);

                            let mut clean_item = serde_json::Map::new();
                            for (key, value) in item_obj {
                                if key != "id" && key != "score" && key != "vector" {
                                    clean_item.insert(key.clone(), value.clone());
                                }
                            }
                            search_results.push((score, id, Value::Object(clean_item)));
                        }
                    }
                } else {
                    debug!("'result' field not found or not an array in search response.");
                }
            }
        } else {
            warn!("Search response was not an array: {:?}", result);
        }

        Ok(search_results)
    }

    async fn search_hybrid_internal(
        &self,
        topic: &str,
        text_query: &str,
        query_vec: &[f32],
        limit: usize,
        _schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        info!(
            "Performing SurrealDB hybrid search (vector + text) for query '{}' on topic '{}'",
            text_query,
            topic
        );

        let query_vec_json = serde_json::to_string(query_vec)?;
        let escaped_text_query = text_query;

        let query = format!(
            r#"
            SELECT
                id,
                vector::similarity::cosine(vector, <vector>{}) AS score,
                *
            FROM {}
            WHERE content @@ '{}' AND vector @@ <vector>{}
            ORDER BY score DESC
            LIMIT {};
            "#,
            query_vec_json,
            topic,
            escaped_text_query,
            query_vec_json,
            limit
        );
        debug!("Executing hybrid search query: {}", query);

        let result = self.execute_query(&query).await?;
        debug!("Raw hybrid search result: {:?}", result);

        let mut search_results = Vec::new();
        if let Some(result_array) = result.as_array() {
            if let Some(first_result) = result_array.first() {
                if let Some(data_records) = first_result.get("result").and_then(Value::as_array) {
                    debug!("Found {} hybrid search results in 'result' field.", data_records.len());
                    for item in data_records {
                        if let Some(item_obj) = item.as_object() {
                            let id = item_obj
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown_id")
                                .to_string();

                            let score = item_obj
                                .get("score")
                                .and_then(Value::as_f64)
                                .map(|s| s as f32)
                                .unwrap_or(0.0);

                            let mut clean_item = serde_json::Map::new();
                            for (key, value) in item_obj {
                                if key != "id" && key != "score" && key != "vector" {
                                    clean_item.insert(key.clone(), value.clone());
                                }
                            }
                            search_results.push((score, id, Value::Object(clean_item)));
                        }
                    }
                } else {
                    debug!("'result' field not found or not an array in hybrid search response.");
                }
            }
        } else {
            warn!("Hybrid search response was not an array: {:?}", result);
        }

        Ok(search_results)
    }

    async fn count_documents_internal(
        &self,
        topic: &str
    ) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let query = format!("RETURN count(SELECT id FROM {});", topic);
        debug!("Executing count query: {}", query);
        let result = self.execute_query(&query).await?;
        debug!("Raw count result: {:?}", result);

        let mut total_count = 0;
        if let Some(array) = result.as_array() {
            if let Some(first_obj) = array.first() {
                if let Some(result_value) = first_obj.get("result") {
                    if let Some(count) = result_value.as_u64() {
                        total_count = count as usize;
                    } else {
                        warn!(
                            "'result' value in count response is not a number: {:?}",
                            result_value
                        );
                    }
                } else {
                    warn!("'result' key not found in count response object: {:?}", first_obj);
                }
            } else {
                warn!("Count response array is empty.");
            }
        } else {
            error!("Count response is not a JSON array: {:?}", result);
        }

        info!("Total count for table '{}': {}", topic, total_count);
        Ok(total_count)
    }
}

#[async_trait]
impl VectorStore for SurrealVectorStore {
    async fn generate_schema(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        self.generate_schema_internal(output_path).await
    }

    async fn search(
        &self,
        query_vec: &[f32],
        limit: usize,
        topic: &str,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        self.search_internal(query_vec, limit, topic, schema_fields).await
    }

    async fn search_hybrid(
        &self,
        topic: &str,
        text_query: &str,
        query_vec: &[f32],
        limit: usize,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        self.search_hybrid_internal(topic, text_query, query_vec, limit, schema_fields).await
    }

    async fn count_documents(&self, topic: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
        self.count_documents_internal(topic).await
    }
}
