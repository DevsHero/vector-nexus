use qdrant_client::qdrant::{ CountPoints, ScrollPoints };
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    SearchPoints,
    Value as QdrantValue,
    PointId,
    value::Kind as QdrantValueKind,
    with_payload_selector::SelectorOptions as WithPayloadOptions,
    WithPayloadSelector,
    ReadConsistency,
    ReadConsistencyType,
    PayloadIncludeSelector,
};
use async_trait::async_trait;
use std::error::Error;
use serde_json::Value;
use log::{ info, error, debug, warn };

use super::{ VectorStore, IndexSchema };

pub struct QdrantVectorStore {
    client: Qdrant,
}

impl QdrantVectorStore {
    pub async fn new(
        host: &str,
        api_key: Option<&str>
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut client_builder = Qdrant::from_url(host);

        if let Some(key) = api_key.filter(|k| !k.is_empty()) {
            client_builder.set_api_key(key);
            info!("Configuring Qdrant client with API key.");
        } else if api_key.is_some() {
            warn!("Qdrant API key provided but is empty.");
        }

        let client = client_builder.build()?;
        info!("Qdrant client connected to {}", host);

        Ok(Self {
            client,
        })
    }

    fn point_id_to_string(point_id: Option<PointId>) -> String {
        match point_id {
            Some(id) =>
                match id.point_id_options {
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid,
                    Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) =>
                        num.to_string(),
                    _ => "unknown_id_format".to_string(),
                }
            None => "missing_id".to_string(),
        }
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn search(
        &self,
        query_vec: &[f32],
        limit: usize,
        topic: &str,
        schema_fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        debug!(
            "Qdrant searching collection '{}' with limit {}. Requesting fields: {:?}",
            topic,
            limit,
            schema_fields.unwrap_or(&Vec::new())
        );

        let payload_selector = match schema_fields {
            Some(fields) if !fields.is_empty() => {
                WithPayloadSelector {
                    selector_options: Some(
                        WithPayloadOptions::Include(PayloadIncludeSelector {
                            fields: fields.clone(),
                        })
                    ),
                }
            }
            _ => {
                warn!("No specific schema fields provided for Qdrant search in topic '{}'. Requesting all fields.", topic);
                WithPayloadSelector {
                    selector_options: Some(WithPayloadOptions::Enable(true)),
                }
            }
        };

        let search_request = SearchPoints {
            collection_name: topic.to_string(),
            vector: query_vec.to_vec(),
            limit: limit as u64,
            with_payload: Some(payload_selector),
            read_consistency: Some(ReadConsistency {
                value: Some(
                    qdrant_client::qdrant::read_consistency::Value::Type(
                        ReadConsistencyType::Majority as i32
                    )
                ),
            }),
            ..Default::default()
        };

        let response = self.client
            .search_points(search_request).await
            .map_err(|e| format!("Qdrant search failed for topic '{}': {}", topic, e))?;

        debug!("Qdrant search returned {} results for topic '{}'", response.result.len(), topic);

        let results = response.result
            .into_iter()
            .map(|point| {
                let id = Self::point_id_to_string(point.id);
                let score = point.score;
                let payload = point.payload
                    .into_iter()
                    .map(|(k, v)| (k, convert_qdrant_value_to_json(v)))
                    .collect::<serde_json::Map<String, Value>>();
                (score, id, Value::Object(payload))
            })
            .collect();

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
        warn!(
            "Qdrant search_hybrid currently only performs vector search for topic '{}'. Text query '{}' ignored.",
            topic,
            text_query
        );
        self.search(query_vec, limit, topic, schema_fields).await
    }

    async fn count_documents(&self, topic: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
        debug!("Qdrant counting documents in collection '{}'", topic);
        let count_request = CountPoints {
            collection_name: topic.to_string(),
            exact: Some(true),
            ..Default::default()
        };

        let response = self.client.count(count_request).await?;
        let count = response.result.map_or(0, |r| r.count);
        debug!("Qdrant count for '{}': {}", topic, count);
        Ok(count as usize)
    }

    async fn generate_schema(
        &self,
        output_path: &str
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        info!("Starting Qdrant schema generation...");

        let collections = self.client.list_collections().await?;
        info!("Found {} collections in Qdrant.", collections.collections.len());

        let mut schemas = Vec::new();

        for collection_desc in collections.collections {
            let collection_name = &collection_desc.name;
            info!("Processing collection: {}", collection_name);

            let info_res = self.client.collection_info(collection_name).await?;

            let mut fields = if let Some(info) = info_res.result {
                if !info.payload_schema.is_empty() {
                    info!("Found payload schema info for '{}'", collection_name);
                    info.payload_schema.keys().cloned().collect::<Vec<_>>()
                } else {
                    debug!("Payload schema info for '{}' is empty.", collection_name);
                    vec![]
                }
            } else {
                warn!("Could not retrieve collection info for '{}'", collection_name);
                vec![]
            };

            if fields.is_empty() {
                info!("No schema info found for '{}', retrieving sample points...", collection_name);

                if collection_name == "conversation_history" {
                    warn!(
                        "Skipping field discovery for special collection: 'conversation_history'"
                    );
                } else {
                    let scroll_request = ScrollPoints {
                        collection_name: collection_name.clone(),
                        limit: Some(5),
                        with_payload: Some(WithPayloadSelector {
                            selector_options: Some(WithPayloadOptions::Enable(true)),
                        }),
                        read_consistency: Some(ReadConsistency {
                            value: Some(
                                qdrant_client::qdrant::read_consistency::Value::Type(
                                    ReadConsistencyType::Majority as i32
                                )
                            ),
                        }),
                        ..Default::default()
                    };

                    match self.client.scroll(scroll_request).await {
                        Ok(response) => {
                            let mut field_set = std::collections::HashSet::new();
                            for point in response.result {
                                for (key, _) in point.payload {
                                    field_set.insert(key);
                                }
                            }

                            if !field_set.is_empty() {
                                fields = field_set.into_iter().collect();
                                fields.sort();
                                info!(
                                    "Found {} fields via sample points for '{}': {:?}",
                                    fields.len(),
                                    collection_name,
                                    fields
                                );
                            } else {
                                warn!("No payload fields found in sample points for '{}'", collection_name);
                            }
                        }
                        Err(e) => {
                            error!("Failed to get sample points for '{}': {}", collection_name, e);
                        }
                    }
                }

                if fields.is_empty() && collection_name != "conversation_history" {
                    fields = vec!["text".to_string()];
                    warn!("Using default fields ['text'] for '{}'", collection_name);
                } else if fields.is_empty() && collection_name == "conversation_history" {
                    warn!("Using no specific fields for 'conversation_history'.");
                }
            } else {
                info!(
                    "Using fields from payload schema info for '{}': {:?}",
                    collection_name,
                    fields
                );
            }

            schemas.push(IndexSchema {
                name: collection_name.clone(),
                prefix: format!("qdrant:{}", collection_name),
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
                "✅ Generated schema for {} Qdrant collections to {}",
                schemas.len(),
                output_path
            );
        } else {
            warn!("No collections found or processed in Qdrant. Schema file not written.");
        }
        info!("Qdrant schema generation finished.");
        Ok(schemas)
    }
}

fn convert_qdrant_value_to_json(value: QdrantValue) -> Value {
    match value.kind {
        Some(QdrantValueKind::NullValue(_)) => Value::Null,
        Some(QdrantValueKind::BoolValue(b)) => Value::Bool(b),
        Some(QdrantValueKind::IntegerValue(i)) => Value::Number(i.into()),
        Some(QdrantValueKind::DoubleValue(d)) => {
            serde_json::Number::from_f64(d).map(Value::Number).unwrap_or(Value::Null)
        }
        Some(QdrantValueKind::StringValue(s)) => Value::String(s),
        Some(QdrantValueKind::ListValue(list)) => {
            Value::Array(list.values.into_iter().map(convert_qdrant_value_to_json).collect())
        }
        Some(QdrantValueKind::StructValue(obj)) => {
            let map = obj.fields
                .into_iter()
                .map(|(key, val)| (key, convert_qdrant_value_to_json(val)))
                .collect();
            Value::Object(map)
        }
        None => Value::Null,
    }
}
