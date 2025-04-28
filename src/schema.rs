use serde::{ Deserialize, Serialize };
/// Represents the schema for a single vector index (collection/table).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IndexSchema {
    /// The name of the index (e.g., collection name, table name).
    pub name: String,
    /// List of fields expected in the documents/points within this index.
    pub fields: Vec<String>,
    /// Optional description of the index's purpose or content.
    pub prefix: String,
}

/// Represents the structure of the schema file (e.g., schema.json).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchemaFile {
    /// A list of all defined index schemas.
    pub indexes: Vec<IndexSchema>,
}
