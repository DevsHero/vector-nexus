# Vector Nexus  
Vector Nexus provides a unified trait (`VectorStore`) and schema definition (`IndexSchema`) for interacting with various vector databases in Rust. It aims to abstract the specific implementation details of different vector stores, allowing you to switch between them more easily in your applications (like RAG engines).
## Auto-Schema Generation for Vector Databases

A key feature facilitated by the `VectorStore` trait is **Auto-Schema Generation**. This addresses the challenge of keeping the application's understanding of the database structure synchronized with the actual database.

**The Problem:**

RAG (Retrieval-Augmented Generation) applications often need to know the structure of the underlying vector database – specifically, the names of different data collections (like "profile", "experience", "documents") and the relevant fields within each collection (like "name", "company_name", "content", "tags"). Manually defining this structure in a configuration file (`IndexSchema`) can be tedious and error-prone, especially if the database schema evolves.

**The Solution: `generate_schema`**

The `VectorStore` trait includes an optional but highly recommended method:

```rust
async fn generate_schema(
    &self,
    output_path: &str, // Where to save the generated schema JSON
    force: bool        // Often used to trigger seeding test data if needed
) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>>;
```
## Core Concepts

1.  **`VectorStore` Trait:**
    This is the central piece of the library. It defines a common interface for operations you'd typically perform on a vector database, such as:
    *   Searching for similar vectors (e.g., `search`, `search_hybrid`).
    *   Counting documents within a specific index/topic.
    *   Generating an `IndexSchema` by inspecting the database.
    *   (Potentially others like adding, updating, deleting documents - *add these if your trait defines them*).
    You implement this trait for each specific vector database you want to support (e.g., SurrealDB, Qdrant, Redis, etc.).

2.  **`IndexSchema` Struct:**
    Defines the structure of your data within a specific index or table. It typically includes:
    *   `name`: The name of the index/table/collection (e.g., "profile", "documents").
    *   `prefix`: A string prefix often used for namespacing keys (e.g., `surreal:profile`).
    *   `fields`: A list of field names within the index that are relevant for searching or filtering (excluding `id` and `vector`).

## Integration with `db2vec`

Vector Nexus is designed to work seamlessly with data prepared and exported using tools like [`db2vec`](https://github.com/DevsHero/db2vec).

1.  **Export Data:** Use `db2vec` to export data from your source database (e.g., PostgreSQL, MySQL) into a suitable format (like JSON Lines), potentially including pre-computed vector embeddings for relevant text fields.
2.  **Load Data:** Load the exported data from `db2vec` into your chosen vector database (e.g., SurrealDB, Qdrant, Redis with vector search module). Ensure the table/collection names and field names match your intended structure.
3.  **Generate Schema:** *After* loading the data, use the `generate_schema` method provided by the specific `VectorStore` implementation in `vector-nexus` (e.g., `SurrealVectorStore::generate_schema`). This will inspect the loaded data in the vector database and create an `IndexSchema` JSON file that accurately reflects the structure (table names and fields) of your loaded data.
4.  **Use `VectorStore`:** Initialize your `VectorStore` implementation (e.g., `SurrealVectorStore`) and load the generated `IndexSchema`. Your application can now use the `VectorStore` trait methods (`search_hybrid`, `count_documents`, etc.) to interact with the data originally exported by `db2vec`.

Essentially, `db2vec` prepares the data, you load it into the vector DB, and `vector-nexus` provides the unified Rust interface to query that data via its `VectorStore` trait and the generated `IndexSchema`.

## How to Implement `VectorStore`

To add support for a new vector database, you create a struct representing the connection/client for that database and implement the `VectorStore` trait for it.

```rust
// Example for a hypothetical "MyVectorDB"

use async_trait::async_trait;
use serde_json::Value;
use vector_nexus::{IndexSchema, VectorStore}; // Assuming these are in the root
use std::error::Error;

// Represents connection to your specific DB
struct MyVectorDbClient {
    // connection details, client instance, etc.
}

impl MyVectorDbClient {
    // Constructor to establish connection
    pub fn new(/* connection args */) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // ... connect ...
        Ok(Self { /* ... */ })
    }
}

#[async_trait]
impl VectorStore for MyVectorDbClient {
    // --- Schema Generation (Optional but Recommended) ---
    async fn generate_schema(
        &self,
        output_path: &str,
        force: bool
    ) -> Result<Vec<IndexSchema>, Box<dyn Error + Send + Sync>> {
        // Logic to inspect your DB, discover tables/collections and their fields,
        // and generate a Vec<IndexSchema>.
        // Write the schema to output_path if provided.
        println!("Generating schema for MyVectorDB...");
        // ... implementation ...
        let schemas = vec![/* ... discovered schemas ... */];
        // ... write to file ...
        Ok(schemas)
    }

    // --- Counting ---
    async fn count_documents(&self, index_name: &str) -> Result<u64, Box<dyn Error + Send + Sync>> {
        // Logic to count documents in the specified index/table/collection
        println!("Counting documents in MyVectorDB index: {}", index_name);
        // ... implementation ...
        let count = 0; // Replace with actual count
        Ok(count)
    }

    // --- Searching ---
    async fn search_hybrid(
        &self,
        index_name: &str,
        text_query: &str, // Optional: Use for keyword part of hybrid search if supported
        vector: &[f32],
        limit: usize,
        // Optional: Use fields for filtering or specifying search scope if supported
        _fields: Option<&Vec<String>>
    ) -> Result<Vec<(f32, String, Value)>, Box<dyn Error + Send + Sync>> {
        // Logic to perform a hybrid (vector + keyword) or pure vector search
        println!(
            "Performing hybrid search in MyVectorDB index: {} for query: '{}' with limit {}",
            index_name,
            text_query,
            limit
        );
        // ... implementation using `vector`, `text_query`, `limit` ...
        let results: Vec<(f32, String, Value)> = vec![
            // (score, document_id, document_value)
        ];
        Ok(results)
    }

    // --- Implement other required VectorStore trait methods ---
    // async fn search(...) -> ... {}
    // async fn add_documents(...) -> ... {}
    // etc.
}

```

## How to Use `VectorStore`

Once you have one or more implementations of `VectorStore`, you can use them generically in your application code. This is often done by storing the specific implementation behind an `Arc<dyn VectorStore>`.

```rust
use vector_nexus::{IndexSchema, VectorStore};
use serde_json::Value;
use std::sync::Arc;
use std::error::Error;

// Assume MyVectorDbClient implements VectorStore
// use crate::my_vector_db::MyVectorDbClient;

async fn run_rag(vector_store: Arc<dyn VectorStore>, schemas: &[IndexSchema]) -> Result<(), Box<dyn Error + Send + Sync>> {
    let user_question = "What is the latest experience?";
    let query_vector: Vec<f32> = vec![0.1, 0.2, /* ... */]; // Get embedding from your model
    let topic = "experience"; // Inferred or specified topic

    // Find the schema for the topic
    let schema = schemas.iter().find(|s| s.name == topic);
    let fields = schema.map(|s| &s.fields);

    // Perform a hybrid search
    let search_results = vector_store.search_hybrid(
        topic,
        user_question,
        &query_vector,
        5, // limit
        fields
    ).await?;

    println!("Search Results:");
    if search_results.is_empty() {
        println!("No documents found.");
    } else {
        for (score, id, doc) in search_results {
            println!("- ID: {}, Score: {:.4}", id, score);
            // Process the document (Value)
            if let Some(company) = doc.get("company_name").and_then(|v| v.as_str()) {
                 println!("  Company: {}", company);
            }
             if let Some(position) = doc.get("position_name").and_then(|v| v.as_str()) {
                 println!("  Position: {}", position);
            }
            // ... access other fields ...
        }
    }

    // Example: Count documents
    let total_skills = vector_store.count_documents("skill").await?;
    println!("\nTotal skills found: {}", total_skills);

    Ok(())
}

// Example Usage (in your main or setup function)
// #[tokio::main]
// async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
//     // 1. Create your specific vector store client
//     let my_db_client = MyVectorDbClient::new(/* ... */)?;
//
//     // 2. Put it behind an Arc<dyn VectorStore>
//     let store: Arc<dyn VectorStore> = Arc::new(my_db_client);
//
//     // 3. Load or generate your schemas
//     let schemas: Vec<IndexSchema> = vec![/* ... load from file or generate ... */];
//
//     // 4. Pass the generic store and schemas to your application logic
//     run_rag(store, &schemas).await?;
//
//     Ok(())
// }

```

This setup allows your core application logic (`run_rag` in this example) to work with any vector database that has a `VectorStore` implementation, simply by changing which concrete type you instantiate and wrap in the `Arc` at the beginning.