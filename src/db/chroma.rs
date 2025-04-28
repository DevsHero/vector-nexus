use async_trait::async_trait;
use reqwest::Client;
use reqwest::header::{ AUTHORIZATION, CONTENT_TYPE, ACCEPT };
use serde::{ Deserialize, Serialize };
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::sync::RwLock;
use log::{ info, error, warn, debug };
use base64::{ engine::general_purpose::STANDARD, Engine as _ };
use super::{ VectorStore, IndexSchema };

#[derive(Debug, Deserialize)]
struct ChromaCollection {
    id: String,
    name: String,
    _dimension: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct ListCollectionsResponse {
    collections: Vec<ChromaCollection>,
}

#[derive(Debug, Serialize)]
struct QueryEmbeddingsRequest<'a> {
    #[serde(rename = "queryEmbeddings")]
    query_embeddings: &'a [&'a [f32]],
    #[serde(rename = "nResults")]
    n_results: usize,
    include: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QueryEmbeddingsResponse {
    ids: Option<Vec<Vec<String>>>,
    distances: Option<Vec<Vec<f32>>>,
    metadatas: Option<Vec<Vec<Option<Value>>>>,
    _documents: Option<Vec<Vec<Option<String>>>>,
    _embeddings: Option<Value>,
    error: Option<String>,
}

pub struct ChromaVectorStore {
    client: Client,
    collection_id_cache: RwLock<HashMap<String, String>>,
    host: String,
    tenant: String,
    database: String,
    api_key: Option<String>,
    user: Option<String>,
    pass: Option<String>,
}

impl ChromaVectorStore {
    pub async fn new(
        host: &str,
        tenant: Option<&str>,
        database: Option<&str>,
        api_key: Option<&str>,
        user: Option<&str>,
        pass: Option<&str>
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tenant = tenant.ok_or("ChromaDB requires a tenant name.")?.to_string();
        let database = database.ok_or("ChromaDB requires a database name.")?.to_string();

        info!(
            "Initializing Chroma client for host: {}, tenant: {}, database: {}",
            host,
            tenant,
            database
        );

        Ok(Self {
            client: Client::new(),
            collection_id_cache: RwLock::new(HashMap::new()),
            host: host.trim_end_matches('/').to_string(),
            tenant,
            database,
            api_key: api_key.map(String::from),
            user: user.map(String::from),
            pass: pass.map(String::from),
        })
    }

    fn get_base_url(&self) -> String {
        format!("{}/api/v1", self.host)
    }

    fn build_request<T: Serialize>(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<&T>
    ) -> Result<reqwest::RequestBuilder, reqwest::Error> {
        let mut request_builder = self.client
            .request(method, url)
            .header(ACCEPT, "application/json")
            .header("X-Chroma-Tenant", &self.tenant)
            .header("X-Chroma-Database", &self.database);

        let use_auth = self.user.is_some() || self.pass.is_some() || self.api_key.is_some();

        if use_auth {
            let auth_header_value = if let (Some(user), Some(pass)) = (&self.user, &self.pass) {
                if !user.is_empty() && !pass.is_empty() {
                    let credentials = format!("{}:{}", user, pass);
                    let encoded = STANDARD.encode(credentials);
                    Some(format!("Basic {}", encoded))
                } else {
                    warn!("Chroma Basic Auth user or pass provided but empty.");
                    None
                }
            } else if let Some(secret) = &self.api_key {
                if !secret.is_empty() {
                    Some(format!("Bearer {}", secret))
                } else {
                    warn!("Chroma Bearer token (API key) provided but empty.");
                    None
                }
            } else {
                warn!("Auth details incomplete for Chroma request.");
                None
            };
            if let Some(auth) = auth_header_value {
                request_builder = request_builder.header(AUTHORIZATION, auth);
            }
        }

        if let Some(b) = body {
            request_builder = request_builder.header(CONTENT_TYPE, "application/json").json(b);
        }

        Ok(request_builder)
    }

    async fn get_collection_id(
        &self,
        collection_name: &str
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        if let Some(id) = self.collection_id_cache.read().unwrap().get(collection_name) {
            debug!("Cache hit for collection ID: {}", collection_name);
            return Ok(id.clone());
        }
        debug!("Cache miss for collection ID: {}", collection_name);

        let list_url = format!("{}/collections", self.get_base_url());
        let request = self.build_request(reqwest::Method::GET, &list_url, None::<&String>)?;

        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            error!("Failed to list Chroma collections (Status: {}): {}", status, text);
            return Err(format!("Failed to list Chroma collections: {}", text).into());
        }

        let collections_response: ListCollectionsResponse = match serde_json::from_str(&text) {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to parse list collections response: {}. Text: {}", e, text);
                return Err(format!("Failed to parse list collections response: {}", e).into());
            }
        };

        for collection in collections_response.collections {
            self.collection_id_cache
                .write()
                .unwrap()
                .insert(collection.name.clone(), collection.id.clone());
            if collection.name == collection_name {
                info!("Found collection ID for '{}': {}", collection_name, collection.id);
                return Ok(collection.id);
            }
        }

        error!("Collection '{}' not found in Chroma.", collection_name);
        Err(format!("Collection '{}' not found", collection_name).into())
    }

    async fn search_internal(
        &self,
        query_vec: &[f32],
        limit: usize,
        collection_name: &str,
        _schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        let collection_id = self.get_collection_id(collection_name).await?;
        let query_url = format!("{}/collections/{}/query", self.get_base_url(), collection_id);

        let request_body = QueryEmbeddingsRequest {
            query_embeddings: &[query_vec],
            n_results: limit,
            include: vec!["metadatas".to_string(), "distances".to_string()],
        };

        debug!("Chroma query request to {}: {:?}", query_url, request_body);

        let request = self.build_request(reqwest::Method::POST, &query_url, Some(&request_body))?;
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            error!("Chroma query failed (Status: {}): {}", status, text);
            return Err(format!("Chroma query failed: {}", text).into());
        }

        debug!("Chroma query response text: {}", text);

        let query_response: QueryEmbeddingsResponse = match serde_json::from_str(&text) {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to parse Chroma query response: {}. Text: {}", e, text);
                return Err(format!("Failed to parse Chroma query response: {}", e).into());
            }
        };

        if let Some(err_msg) = query_response.error {
            error!("Chroma query returned error: {}", err_msg);
            return Err(format!("Chroma query error: {}", err_msg).into());
        }

        let mut results = Vec::new();
        if
            let (Some(ids_batch), Some(distances_batch), Some(metadatas_batch)) = (
                query_response.ids,
                query_response.distances,
                query_response.metadatas,
            )
        {
            if
                let (Some(ids), Some(distances), Some(metadatas)) = (
                    ids_batch.get(0),
                    distances_batch.get(0),
                    metadatas_batch.get(0),
                )
            {
                for i in 0..ids.len() {
                    if
                        let (Some(id), Some(dist), Some(meta_opt)) = (
                            ids.get(i),
                            distances.get(i),
                            metadatas.get(i),
                        )
                    {
                        let score = 1.0 - dist;
                        let metadata = meta_opt
                            .clone()
                            .unwrap_or(Value::Object(Default::default()));
                        results.push((score, id.clone(), metadata));
                    }
                }
            } else {
                warn!("Chroma query response structure unexpected (missing inner batch).");
            }
        } else {
            warn!("Chroma query response missing expected fields (ids, distances, metadatas).");
        }

        debug!("Parsed {} results from Chroma search.", results.len());
        Ok(results)
    }
}

#[async_trait]
impl VectorStore for ChromaVectorStore {
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
        warn!(
            "Chroma search_hybrid currently only performs vector search for topic '{}'. Text query '{}' ignored.",
            topic,
            text_query
        );
        self.search(query_vec, limit, topic, schema_fields).await
    }

    async fn count_documents(&self, topic: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
        let collection_id = self.get_collection_id(topic).await?;
        let count_url = format!("{}/collections/{}/count", self.get_base_url(), collection_id);
        debug!("Counting documents in Chroma collection '{}' (ID: {})", topic, collection_id);

        let request = self.build_request(reqwest::Method::GET, &count_url, None::<&String>)?;
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            error!("Failed to count Chroma documents (Status: {}): {}", status, text);
            return Err(format!("Failed to count Chroma documents: {}", text).into());
        }

        match text.parse::<usize>() {
            Ok(count) => {
                debug!("Count for collection '{}': {}", topic, count);
                Ok(count)
            }
            Err(e) => {
                error!("Failed to parse Chroma count response '{}': {}", text, e);
                Err(format!("Failed to parse Chroma count response: {}", e).into())
            }
        }
    }

    async fn generate_schema(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        info!("Starting Chroma schema generation...");
        let list_url = format!("{}/collections", self.get_base_url());
        let request = self.build_request(reqwest::Method::GET, &list_url, None::<&String>)?;

        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            error!(
                "Failed to list Chroma collections for schema gen (Status: {}): {}",
                status,
                text
            );
            return Err(format!("Failed to list Chroma collections: {}", text).into());
        }

        let collections_response: ListCollectionsResponse = match serde_json::from_str(&text) {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to parse list collections response: {}. Text: {}", e, text);
                return Err(format!("Failed to parse list collections response: {}", e).into());
            }
        };

        info!("Found {} collections in Chroma.", collections_response.collections.len());
        let mut schemas = Vec::new();

        for collection in collections_response.collections {
            info!("Processing collection: {}", collection.name);
            let fields = vec!["text".to_string()];
            warn!(
                "Using default fields ['text'] for Chroma collection '{}' as schema introspection is limited.",
                collection.name
            );

            schemas.push(IndexSchema {
                name: collection.name.clone(),

                prefix: format!("chroma:{}", collection.name),
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
            info!(
                "✅ Generated schema for {} Chroma collections to {}",
                schemas.len(),
                output_path
            );
        } else {
            warn!("No collections found in Chroma. Schema file not written.");
        }
        info!("Chroma schema generation finished.");
        Ok(schemas)
    }
}
