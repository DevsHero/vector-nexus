use async_trait::async_trait;
use reqwest::{ Client, RequestBuilder };
use serde_json::{ Value, json };
use std::{ error::Error, time::Duration, collections::HashSet };
use log::{ info, warn, error, debug };

use super::{ VectorStore, IndexSchema };

const PINECONE_API_VERSION: &str = "2024-07";
const PINECONE_CONTROL_PLANE_BASE_URL: &str = "https://api.pinecone.io";

pub struct PineconeVectorStore {
    client: Client,
    data_plane_url: String,
    api_key: Option<String>,
    namespace: String,
    use_auth: bool,
}

impl PineconeVectorStore {
    pub async fn new(
        host_param: &str,
        api_key_param: Option<&str>,
        namespace_param: Option<&str>,
        index_name_param: Option<&str>
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

        let is_local =
            host_param.contains("localhost") ||
            host_param.contains("127.0.0.1") ||
            host_param.contains("::1");

        let use_auth = !is_local;
        let api_key = if use_auth {
            Some(
                api_key_param
                    .filter(|k| !k.is_empty())
                    .map(String::from)
                    .ok_or_else(||
                        Box::<dyn Error + Send + Sync>::from(
                            "Pinecone cloud requires a non-empty API key."
                        )
                    )?
            )
        } else {
            None
        };

        let control_plane_url = if is_local {
            host_param.trim_end_matches('/').to_string()
        } else {
            PINECONE_CONTROL_PLANE_BASE_URL.to_string()
        };

        info!("Pinecone mode: {}", if is_local { "LOCAL" } else { "CLOUD" });
        info!("Control plane URL: {}", control_plane_url);

        let data_plane_url = if is_local {
            if host_param.is_empty() {
                return Err("Local Pinecone requires a host URL.".into());
            }
            host_param.trim_end_matches('/').to_string()
        } else {
            let mut determined_host: Option<String> = None;

            if let Some(index_name) = index_name_param.filter(|n| !n.is_empty()) {
                info!("Attempting to describe index '{}' via control plane...", index_name);
                let describe_url = format!("{}/indexes/{}", control_plane_url, index_name);

                let api_key_ref = api_key
                    .as_ref()
                    .ok_or("API Key is required for cloud control plane operations")?;

                let describe_req = client
                    .get(&describe_url)
                    .header("Accept", "application/json")
                    .header("X-Pinecone-API-Version", PINECONE_API_VERSION)
                    .header("Api-Key", api_key_ref);

                let describe_resp = describe_req.send().await?;

                if describe_resp.status().is_success() {
                    let j: Value = describe_resp.json().await?;
                    if let Some(host) = j.get("host").and_then(Value::as_str) {
                        info!("Found host for index '{}' via control plane: {}", index_name, host);
                        determined_host = Some(format!("https://{}", host));
                    } else {
                        warn!("Index '{}' described successfully, but no 'host' field found in response.", index_name);
                    }
                } else {
                    let status = describe_resp.status();
                    let txt = describe_resp.text().await.unwrap_or_default();
                    warn!(
                        "Failed to describe index '{}' (Status: {}): {}. Will try using host parameter if provided.",
                        index_name,
                        status,
                        txt
                    );
                }
            } else {
                info!("Index name not provided, will rely on host parameter for data plane URL.");
            }

            if determined_host.is_none() {
                if host_param.is_empty() {
                    return Err(
                        "Pinecone data plane URL could not be determined. Provide a valid host URL or index name.".into()
                    );
                }
                if host_param.contains(".svc.") && host_param.contains(".pinecone.io") {
                    info!("Using provided host parameter as data plane URL: {}", host_param);
                    let host_url = host_param.trim_end_matches('/');
                    determined_host = Some(
                        if host_url.starts_with("https://") {
                            host_url.to_string()
                        } else {
                            format!("https://{}", host_url)
                        }
                    );
                } else {
                    return Err(
                        format!("Provided host ('{}') does not look like a Pinecone data plane URL and index name was not used or failed.", host_param).into()
                    );
                }
            }

            determined_host.ok_or("Failed to determine data plane URL.")?
        };

        info!("Using data plane URL: {}", data_plane_url);

        let namespace = namespace_param
            .filter(|ns| !ns.is_empty())
            .map(String::from)
            .unwrap_or_else(|| {
                warn!("Pinecone namespace not provided, using empty string (default namespace).");
                "".to_string()
            });

        info!("Using namespace for schema fallback: '{}'", if namespace.is_empty() {
            "<default>"
        } else {
            &namespace
        });

        let status_url = format!("{}/describe_index_stats", data_plane_url);
        let mut req_builder = client
            .post(&status_url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("X-Pinecone-API-Version", PINECONE_API_VERSION);

        if use_auth {
            if let Some(key) = api_key.as_ref() {
                req_builder = req_builder.header("Api-Key", key);
            } else {
                return Err("API Key missing for authenticated connection check.".into());
            }
        }

        let resp = req_builder.json(&json!({})).send().await;

        match resp {
            Ok(r) => {
                if r.status().is_success() {
                    let stats: Value = r.json().await?;
                    debug!("Pinecone index stats: {:?}", stats);
                    let server_dimension = stats
                        .get("dimension")
                        .and_then(Value::as_u64)
                        .map(|d| d as usize)
                        .ok_or("Failed to parse dimension from index stats")?;

                    info!("Pinecone connection successful. Index dimension: {}", server_dimension);

                    Ok(Self {
                        client,
                        data_plane_url,
                        api_key,
                        namespace,
                        use_auth,
                    })
                } else {
                    let status = r.status();
                    let text = r
                        .text().await
                        .unwrap_or_else(|_| "Failed to read error body".to_string());
                    error!("Pinecone connection check failed (Status: {}): {}", status, text);
                    Err(format!("Pinecone connection failed ({}): {}", status, text).into())
                }
            }
            Err(e) => {
                error!("Error during Pinecone connection check: {}", e);
                if e.is_timeout() {
                    Err(format!("Pinecone connection timed out: {}", e).into())
                } else {
                    Err(format!("Pinecone connection error: {}", e).into())
                }
            }
        }
    }

    fn add_common_headers(&self, req_builder: RequestBuilder) -> RequestBuilder {
        let builder_with_version = req_builder.header(
            "X-Pinecone-API-Version",
            PINECONE_API_VERSION
        );
        if self.use_auth {
            if let Some(key) = self.api_key.as_ref() {
                builder_with_version.header("Api-Key", key)
            } else {
                warn!("Auth expected but API key is missing in add_common_headers");
                builder_with_version
            }
        } else {
            builder_with_version
        }
    }

    fn parse_metadata(metadata_value: Option<&Value>) -> Value {
        match metadata_value {
            Some(Value::Object(map)) => {
                let mut parsed_map = serde_json::Map::new();
                for (k, v) in map {
                    if let Some(s) = v.as_str() {
                        if
                            (s.starts_with('{') && s.ends_with('}')) ||
                            (s.starts_with('[') && s.ends_with(']'))
                        {
                            if let Ok(parsed_json) = serde_json::from_str(s) {
                                parsed_map.insert(k.clone(), parsed_json);
                                continue;
                            }
                        }
                    }
                    parsed_map.insert(k.clone(), v.clone());
                }
                Value::Object(parsed_map)
            }
            Some(v) => v.clone(),
            None => Value::Null,
        }
    }
}

#[async_trait]
impl VectorStore for PineconeVectorStore {
    async fn search(
        &self,
        query_vec: &[f32],
        limit: usize,
        topic: &str,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        let namespace = if topic.is_empty() { "" } else { topic };
        debug!(
            "Pinecone vector search in namespace '{}' with limit {}. Requesting fields: {:?}",
            if namespace.is_empty() {
                "<default>"
            } else {
                namespace
            },
            limit,
            schema_fields.unwrap_or(&Vec::new())
        );

        let url = format!("{}/query", self.data_plane_url);

        let payload =
            json!({
            "namespace": namespace,
            "vector": query_vec,
            "topK": limit,
            "includeValues": false,
            "includeMetadata": true 
        });

        let req_builder = self.client
            .post(&url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&payload);

        let req_builder = self.add_common_headers(req_builder);

        let resp = req_builder.send().await?;

        if resp.status().is_success() {
            let result_json: Value = resp.json().await?;
            debug!("Raw Pinecone search result: {:?}", result_json);

            let matches = result_json
                .get("matches")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| {
                            let id = m.get("id").and_then(Value::as_str)?.to_string();
                            let score = m.get("score").and_then(Value::as_f64)? as f32;
                            let mut metadata = Self::parse_metadata(m.get("metadata"));

                            if
                                let (Some(fields_to_keep), Value::Object(obj)) = (
                                    schema_fields,
                                    &mut metadata,
                                )
                            {
                                if !fields_to_keep.is_empty() {
                                    obj.retain(|k, _| fields_to_keep.contains(k));
                                }
                            } else if schema_fields.map_or(false, |f| f.is_empty()) {
                                metadata = Value::Object(Default::default());
                            }

                            Some((score, id, metadata))
                        })
                        .collect()
                })
                .unwrap_or_else(Vec::new);

            debug!("Parsed {} results from Pinecone vector search.", matches.len());
            Ok(matches)
        } else {
            let status = resp.status();
            let text = resp.text().await?;
            error!("Pinecone search failed ({}): {}", status, text);
            Err(format!("Pinecone search error: {}", text).into())
        }
    }

    async fn search_hybrid(
        &self,
        topic: &str,
        text_query: &str,
        query_vec: &[f32],
        limit: usize,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        let namespace = if topic.is_empty() { "" } else { topic };
        debug!(
            "Pinecone hybrid search (vector + filter) in namespace '{}' with limit {} and filter '{}'. Requesting fields: {:?}",
            if namespace.is_empty() {
                "<default>"
            } else {
                namespace
            },
            limit,
            text_query,
            schema_fields.unwrap_or(&Vec::new())
        );

        let url = format!("{}/query", self.data_plane_url);

        let filter_value: Value = match serde_json::from_str(text_query) {
            Ok(json_filter) => json_filter,
            Err(_) => {
                warn!(
                    "Filter '{}' is not valid JSON. Performing vector-only search in namespace '{}'.",
                    text_query,
                    namespace
                );
                json!({})
            }
        };

        let mut payload =
            json!({
            "namespace": namespace,
            "vector": query_vec,
            "topK": limit,
            "includeValues": false,
            "includeMetadata": true
        });

        if filter_value.as_object().map_or(false, |m| !m.is_empty()) {
            debug!("Applying filter: {}", filter_value.to_string());
            payload["filter"] = filter_value;
        } else if !text_query.is_empty() {
            warn!("Filter resulted in an empty object, performing vector-only search in namespace '{}'.", namespace);
        }

        let req_builder = self.client
            .post(&url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&payload);

        let req_builder = self.add_common_headers(req_builder);

        let resp = req_builder.send().await?;

        if resp.status().is_success() {
            let result_json: Value = resp.json().await?;
            debug!("Raw Pinecone hybrid search result: {:?}", result_json);

            let matches = result_json
                .get("matches")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| {
                            let id = m.get("id").and_then(Value::as_str)?.to_string();
                            let score = m.get("score").and_then(Value::as_f64)? as f32;
                            let mut metadata = Self::parse_metadata(m.get("metadata"));
                            if
                                let (Some(fields_to_keep), Value::Object(obj)) = (
                                    schema_fields,
                                    &mut metadata,
                                )
                            {
                                if !fields_to_keep.is_empty() {
                                    obj.retain(|k, _| fields_to_keep.contains(k));
                                }
                            } else if schema_fields.map_or(false, |f| f.is_empty()) {
                                metadata = Value::Object(Default::default());
                            }

                            Some((score, id, metadata))
                        })
                        .collect()
                })
                .unwrap_or_else(Vec::new);

            debug!(
                "Parsed {} results from Pinecone hybrid search in namespace '{}'.",
                matches.len(),
                namespace
            );
            Ok(matches)
        } else {
            let status = resp.status();
            let text = resp.text().await?;
            error!(
                "Pinecone hybrid search failed in namespace '{}' ({}): {}",
                namespace,
                status,
                text
            );
            Err(format!("Pinecone hybrid search error: {}", text).into())
        }
    }

    async fn count_documents(&self, topic: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let namespace = if topic.is_empty() { "" } else { topic };
        debug!("Counting documents in Pinecone namespace '{}'", if namespace.is_empty() {
            "<default>"
        } else {
            namespace
        });

        let url = format!("{}/describe_index_stats", self.data_plane_url);

        let req_builder = self.client
            .post(&url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&json!({}));

        let req_builder = self.add_common_headers(req_builder);

        let resp = req_builder.send().await?;

        if resp.status().is_success() {
            let stats: Value = resp.json().await?;
            debug!("Raw Pinecone index stats for count: {:?}", stats);

            let count = stats
                .get("namespaces")
                .and_then(|n| n.get(namespace))
                .and_then(|ns_stats| ns_stats.get("vectorCount"))
                .and_then(Value::as_u64)
                .unwrap_or(0);

            debug!(
                "Count for namespace '{}': {}",
                if namespace.is_empty() {
                    "<default>"
                } else {
                    namespace
                },
                count
            );
            Ok(count as usize)
        } else {
            let status = resp.status();
            let text = resp.text().await?;
            error!("Pinecone describe_index_stats failed ({}): {}", status, text);
            Err(format!("Pinecone count error: {}", text).into())
        }
    }

    async fn generate_schema(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        info!(
            "Generating schema for Pinecone index by listing namespaces and fetching sample vector metadata..."
        );

        let stats_url = format!("{}/describe_index_stats", self.data_plane_url);
        let stats_req_builder = self.client
            .post(&stats_url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&json!({}));

        let stats_req_builder = self.add_common_headers(stats_req_builder);
        let stats_resp = stats_req_builder.send().await?;

        if !stats_resp.status().is_success() {
            let status = stats_resp.status();
            let text = stats_resp.text().await?;
            error!("Failed to fetch Pinecone index stats for schema generation: {}", text);
            return Err(format!("Pinecone schema generation failed ({}): {}", status, text).into());
        }

        let stats: Value = stats_resp.json().await?;
        debug!("Raw Pinecone index stats for schema generation: {:?}", stats);

        let namespaces_map = stats
            .get("namespaces")
            .and_then(Value::as_object)
            .ok_or_else(||
                "Failed to extract namespaces map from Pinecone index stats".to_string()
            )?;

        let mut schemas = Vec::new();
        if namespaces_map.is_empty() {
            warn!("No namespaces found in the Pinecone index via stats endpoint.");
        } else {
            for namespace_name_str in namespaces_map.keys() {
                let namespace_name = if namespace_name_str.is_empty() {
                    ""
                } else {
                    namespace_name_str
                };
                info!("Processing namespace '{}'...", if namespace_name.is_empty() {
                    "<default>"
                } else {
                    namespace_name
                });
                let mut fields: HashSet<String> = HashSet::new();
                fields.insert("text".to_string());

                let mut sample_vector_id: Option<String> = None;

                let list_url = format!("{}/vectors/list", self.data_plane_url);
                let list_req_builder = self.client
                    .get(&list_url)
                    .query(
                        &[
                            ("namespace", namespace_name),
                            ("limit", &String::from("1")),
                        ]
                    )
                    .header("Accept", "application/json");

                let list_req_builder = self.add_common_headers(list_req_builder);

                match list_req_builder.send().await {
                    Ok(list_resp) => {
                        if list_resp.status().is_success() {
                            match list_resp.json::<Value>().await {
                                Ok(list_data) => {
                                    debug!(
                                        "Raw /vectors/list response for namespace '{}': {:?}",
                                        namespace_name,
                                        list_data
                                    );
                                    if
                                        let Some(vectors_array) = list_data
                                            .get("vectors")
                                            .and_then(Value::as_array)
                                    {
                                        if let Some(first_vector_info) = vectors_array.first() {
                                            if
                                                let Some(id) = first_vector_info
                                                    .get("id")
                                                    .and_then(Value::as_str)
                                            {
                                                sample_vector_id = Some(id.to_string());
                                                info!(
                                                    "Namespace '{}': Found sample vector ID: {}",
                                                    namespace_name,
                                                    id
                                                );
                                            } else {
                                                warn!("Namespace '{}': First vector in /vectors/list response missing 'id'.", namespace_name);
                                            }
                                        } else {
                                            warn!("Namespace '{}': /vectors/list returned empty 'vectors' array.", namespace_name);
                                        }
                                    } else {
                                        warn!("Namespace '{}': /vectors/list response missing 'vectors' array.", namespace_name);
                                    }
                                }
                                Err(e) => {
                                    warn!(
                                        "Namespace '{}': Failed to parse /vectors/list JSON response: {}",
                                        namespace_name,
                                        e
                                    );
                                }
                            }
                        } else {
                            let status = list_resp.status();
                            let text = list_resp.text().await.unwrap_or_default();
                            warn!(
                                "Namespace '{}': Failed to call /vectors/list (Status: {}): {}",
                                namespace_name,
                                status,
                                text
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Namespace '{}': Error sending /vectors/list request: {}",
                            namespace_name,
                            e
                        );
                    }
                }

                if let Some(id_to_fetch) = sample_vector_id {
                    let fetch_url = format!("{}/vectors/fetch", self.data_plane_url);
                    let fetch_req_builder = self.client
                        .get(&fetch_url)
                        .query(
                            &[
                                ("ids", &id_to_fetch),
                                ("namespace", &namespace_name.to_string()),
                            ]
                        )
                        .header("Accept", "application/json");

                    let fetch_req_builder = self.add_common_headers(fetch_req_builder);

                    match fetch_req_builder.send().await {
                        Ok(fetch_resp) => {
                            if fetch_resp.status().is_success() {
                                match fetch_resp.json::<Value>().await {
                                    Ok(fetch_data) => {
                                        debug!(
                                            "Raw /vectors/fetch response for ID '{}' in namespace '{}': {:?}",
                                            id_to_fetch,
                                            namespace_name,
                                            fetch_data
                                        );
                                        if
                                            let Some(vectors_map) = fetch_data
                                                .get("vectors")
                                                .and_then(Value::as_object)
                                        {
                                            if
                                                let Some(fetched_vector) = vectors_map.get(
                                                    &id_to_fetch
                                                )
                                            {
                                                if
                                                    let Some(metadata_obj) = fetched_vector
                                                        .get("metadata")
                                                        .and_then(Value::as_object)
                                                {
                                                    let discovered_fields: Vec<String> =
                                                        metadata_obj.keys().cloned().collect();
                                                    if !discovered_fields.is_empty() {
                                                        info!(
                                                            "Discovered metadata fields for namespace '{}': {:?}",
                                                            namespace_name,
                                                            discovered_fields
                                                        );
                                                        fields.extend(discovered_fields);
                                                    } else {
                                                        warn!(
                                                            "Namespace '{}', ID '{}': Fetched vector has no metadata fields.",
                                                            namespace_name,
                                                            id_to_fetch
                                                        );
                                                    }
                                                } else {
                                                    warn!(
                                                        "Namespace '{}', ID '{}': Fetched vector missing 'metadata' object.",
                                                        namespace_name,
                                                        id_to_fetch
                                                    );
                                                }
                                            } else {
                                                warn!(
                                                    "Namespace '{}': Fetched data missing vector object for ID '{}'.",
                                                    namespace_name,
                                                    id_to_fetch
                                                );
                                            }
                                        } else {
                                            warn!("Namespace '{}': /vectors/fetch response missing 'vectors' map.", namespace_name);
                                        }
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Namespace '{}': Failed to parse /vectors/fetch JSON response: {}",
                                            namespace_name,
                                            e
                                        );
                                    }
                                }
                            } else {
                                let status = fetch_resp.status();
                                let text = fetch_resp.text().await.unwrap_or_default();
                                warn!(
                                    "Namespace '{}': Failed to call /vectors/fetch for ID '{}' (Status: {}): {}",
                                    namespace_name,
                                    id_to_fetch,
                                    status,
                                    text
                                );
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Namespace '{}': Error sending /vectors/fetch request: {}",
                                namespace_name,
                                e
                            );
                        }
                    }
                } else {
                    warn!("Namespace '{}': Could not obtain a sample vector ID. Using default fields.", namespace_name);
                }

                let mut sorted_fields: Vec<String> = fields.into_iter().collect();
                sorted_fields.sort();

                let schema = IndexSchema {
                    name: namespace_name.to_string(),
                    prefix: format!("pinecone:{}", namespace_name),
                    fields: sorted_fields,
                };
                schemas.push(schema);
            }
        }

        if schemas.is_empty() {
            warn!(
                "No namespaces discovered via stats. Adding schema for namespace specified in args: '{}'",
                if self.namespace.is_empty() {
                    "<default>"
                } else {
                    &self.namespace
                }
            );
            let fallback_schema = IndexSchema {
                name: self.namespace.clone(),
                prefix: format!("pinecone:{}", self.namespace),
                fields: vec!["text".to_string()],
            };
            schemas.push(fallback_schema);
        }

        let wrapped = json!({ "indexes": schemas });
        if let Some(parent) = std::path::Path::new(output_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(output_path, serde_json::to_string_pretty(&wrapped)?)?;

        info!("✅ Generated schema with {} namespaces to {}", schemas.len(), output_path);

        Ok(schemas)
    }
}
