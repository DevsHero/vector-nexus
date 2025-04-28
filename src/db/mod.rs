pub mod chroma;
pub mod milvus;
pub mod pinecone;
pub mod qdrant;
pub mod redis;
pub mod surreal;
use async_trait::async_trait;
use serde_json::Value;
use std::error::Error;
use std::sync::Arc;
use log::info;

use crate::schema::IndexSchema;

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn search(
        &self,
        query_vec: &[f32],
        limit: usize,
        topic: &str,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>>;

    async fn search_hybrid(
        &self,
        topic: &str,
        text_query: &str,
        query_vec: &[f32],
        limit: usize,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>>;

    async fn count_documents(&self, topic: &str) -> Result<usize, Box<dyn Error + Send + Sync>>;

    async fn generate_schema(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum StoreType {
    Redis,
    Chroma,
    Milvus,
    Qdrant,
    Surreal,
    Pinecone,
}

#[derive(Clone, Debug)]
pub struct VectorStoreConfig {
    pub store_type: StoreType,
    pub host: String,
    pub api_key: Option<String>,
    pub tenant: Option<String>,
    pub database: Option<String>,
    pub namespace: Option<String>,
    pub index_name: Option<String>,
    pub user: Option<String>,
    pub pass: Option<String>,
    pub dimension: Option<usize>,
    pub metric: Option<String>,
}

pub async fn create_vector_store(
    config: VectorStoreConfig
) -> Result<Arc<dyn VectorStore>, Box<dyn Error + Send + Sync>> {
    info!("Creating vector store of type: {:?}", config.store_type);
    match config.store_type {
        StoreType::Redis => {
            let store = redis::RedisVectorStore::new(&config.host)?;
            Ok(Arc::new(store))
        }
        StoreType::Qdrant => {
            let store = qdrant::QdrantVectorStore::new(
                &config.host,
                config.api_key.as_deref()
            ).await?;
            Ok(Arc::new(store))
        }
        StoreType::Pinecone => {
            let store = pinecone::PineconeVectorStore::new(
                &config.host,
                config.api_key.as_deref(),
                config.namespace.as_deref(),
                config.index_name.as_deref()
            ).await?;
            Ok(Arc::new(store))
        }
        StoreType::Chroma => {
            let store = chroma::ChromaVectorStore::new(
                &config.host,
                config.tenant.as_deref(),
                config.database.as_deref(),
                config.api_key.as_deref(),
                config.user.as_deref(),
                config.pass.as_deref()
            ).await?;
            Ok(Arc::new(store))
        }
        StoreType::Milvus => {
            let dim = config.dimension.ok_or("Milvus requires dimension")?;
            let metric = config.metric.as_deref().ok_or("Milvus requires metric")?;
            let store = milvus::MilvusVectorStore::new(
                &config.host,
                config.api_key.as_deref(),
                config.database.as_deref(),
                dim,
                metric
            ).await?;
            Ok(Arc::new(store))
        }
        StoreType::Surreal => {
            let store = surreal::SurrealVectorStore::new(
                &config.host,
                config.namespace.as_deref(),
                config.database.as_deref(),
                config.user.as_deref(),
                config.pass.as_deref()
            ).await?;
            Ok(Arc::new(store))
        }
    }
}

pub fn get_store_type(type_str: &str) -> Result<StoreType, String> {
    match type_str.to_lowercase().as_str() {
        "redis" => Ok(StoreType::Redis),
        "chroma" => Ok(StoreType::Chroma),
        "milvus" => Ok(StoreType::Milvus),
        "qdrant" => Ok(StoreType::Qdrant),
        "surreal" | "surrealdb" => Ok(StoreType::Surreal),
        "pinecone" => Ok(StoreType::Pinecone),
        _ => Err(format!("Unsupported vector store type: {}", type_str)),
    }
}
