//! Define APIs for RAG operations.

use crate::{embeddings::embeddings, error::LlamaCoreError, running_mode, RunningMode};
use endpoints::{
    embeddings::{EmbeddingObject, EmbeddingsResponse, InputText},
    rag::{RagEmbeddingRequest, RagScoredPoint, RetrieveObject},
};
use qdrant::*;
use text_splitter::{MarkdownSplitter, TextSplitter};
use tiktoken_rs::cl100k_base;

/// Convert document chunks to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
///
/// * `qdrant_url` - URL of the Qdrant server.
///
/// * `qdrant_collection_name` - Name of the Qdrant collection to be created.
///
/// # Returns
///
/// Name of the Qdrant collection if successful.
pub async fn rag_doc_chunks_to_embeddings(
    rag_embedding_request: &RagEmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Convert document chunks to embeddings.");

    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        let err_msg = format!(
            "Creating knowledge base is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let embedding_request = &rag_embedding_request.embedding_request;
    let qdrant_url = rag_embedding_request.qdrant_url.as_str();
    let qdrant_collection_name = rag_embedding_request.qdrant_collection_name.as_str();

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute embeddings for document chunks.");

    #[cfg(feature = "logging")]
    if let Ok(request_str) = serde_json::to_string(&embedding_request) {
        info!(target: "stdout", "Embedding request: {}", request_str);
    }

    // compute embeddings for the document
    let response = embeddings(embedding_request).await?;
    let embeddings = response.data.as_slice();
    let dim = embeddings[0].embedding.len();

    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.to_string());

    // create a collection
    qdrant_create_collection(&qdrant_client, qdrant_collection_name, dim).await?;

    let chunks = match &embedding_request.input {
        InputText::String(text) => vec![text.clone()],
        InputText::ArrayOfStrings(texts) => texts.clone(),
        InputText::ArrayOfTokens(tokens) => tokens.iter().map(|t| t.to_string()).collect(),
        InputText::ArrayOfTokenArrays(token_arrays) => token_arrays
            .iter()
            .map(|tokens| tokens.iter().map(|t| t.to_string()).collect())
            .collect(),
    };

    // create and upsert points
    qdrant_persist_embeddings(
        &qdrant_client,
        qdrant_collection_name,
        embeddings,
        chunks.as_slice(),
    )
    .await?;

    Ok(response)
}

/// Convert a query to embeddings.
///
/// # Arguments
///
/// * `embedding_request` - A reference to an `EmbeddingRequest` object.
pub async fn rag_query_to_embeddings(
    rag_embedding_request: &RagEmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute embeddings for the user query.");

    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        let err_msg = format!("The RAG query is not supported in the {running_mode} mode.",);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    embeddings(&rag_embedding_request.embedding_request).await
}

/// Retrieve similar points from the Qdrant server using the query embedding
///
/// # Arguments
///
/// * `query_embedding` - A reference to a query embedding.
///
/// * `qdrant_url` - URL of the Qdrant server.
///
/// * `qdrant_collection_name` - Name of the Qdrant collection to be created.
///
/// * `limit` - Max number of retrieved result.
pub async fn rag_retrieve_context(
    query_embedding: &[f32],
    qdrant_url: impl AsRef<str>,
    qdrant_collection_name: impl AsRef<str>,
    limit: usize,
    score_threshold: Option<f32>,
) -> Result<RetrieveObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    {
        info!(target: "stdout", "Retrieve context.");

        info!(target: "stdout", "qdrant_url: {}, qdrant_collection_name: {}, limit: {}, score_threshold: {}", qdrant_url.as_ref(), qdrant_collection_name.as_ref(), limit, score_threshold.unwrap_or_default());
    }

    let running_mode = running_mode()?;
    if running_mode != RunningMode::Rag {
        let err_msg = format!(
            "The context retrieval is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    // create a Qdrant client
    let qdrant_client = qdrant::Qdrant::new_with_url(qdrant_url.as_ref().to_string());

    // search for similar points
    let scored_points = match qdrant_search_similar_points(
        &qdrant_client,
        qdrant_collection_name.as_ref(),
        query_embedding,
        limit,
        score_threshold,
    )
    .await
    {
        Ok(points) => points,
        Err(e) => {
            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", e.to_string());

            return Err(e);
        }
    };

    let ro = match scored_points.is_empty() {
        true => RetrieveObject {
            points: None,
            limit,
            score_threshold: score_threshold.unwrap_or(0.0),
        },
        false => {
            let mut points: Vec<RagScoredPoint> = vec![];
            for point in scored_points.iter() {
                if let Some(payload) = &point.payload {
                    if let Some(source) = payload.get("source") {
                        points.push(RagScoredPoint {
                            source: source.to_string(),
                            score: point.score,
                        })
                    }
                }
            }

            RetrieveObject {
                points: Some(points),
                limit,
                score_threshold: score_threshold.unwrap_or(0.0),
            }
        }
    };

    Ok(ro)
}

async fn qdrant_create_collection(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    dim: usize,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Create a Qdrant collection named {} of {} dimensions.", collection_name.as_ref(), dim);

    if let Err(e) = qdrant_client
        .create_collection(collection_name.as_ref(), dim as u32)
        .await
    {
        let err_msg = e.to_string();

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    Ok(())
}

async fn qdrant_persist_embeddings(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    embeddings: &[EmbeddingObject],
    chunks: &[String],
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Persist embeddings to the Qdrant instance.");

    let mut points = Vec::<Point>::new();
    for embedding in embeddings {
        // convert the embedding to a vector
        let vector: Vec<_> = embedding.embedding.iter().map(|x| *x as f32).collect();

        // create a payload
        let payload = serde_json::json!({"source": chunks[embedding.index as usize]})
            .as_object()
            .map(|m| m.to_owned());

        // create a point
        let p = Point {
            id: PointId::Num(embedding.index),
            vector,
            payload,
        };

        points.push(p);
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Number of points to be upserted: {}", points.len());

    if let Err(e) = qdrant_client
        .upsert_points(collection_name.as_ref(), points)
        .await
    {
        let err_msg = format!("Failed to upsert points. Reason: {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    Ok(())
}

async fn qdrant_search_similar_points(
    qdrant_client: &qdrant::Qdrant,
    collection_name: impl AsRef<str>,
    query_vector: &[f32],
    limit: usize,
    score_threshold: Option<f32>,
) -> Result<Vec<ScoredPoint>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Search similar points from the qdrant instance.");

    match qdrant_client
        .search_points(
            collection_name.as_ref(),
            query_vector.to_vec(),
            limit as u64,
            score_threshold,
        )
        .await
    {
        Ok(search_result) => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Number of similar points found: {}", search_result.len());

            Ok(search_result)
        }
        Err(e) => {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "Fail to search similar points from the qdrant instance. Reason: {}", &err_msg);

            Err(LlamaCoreError::Operation(err_msg))
        }
    }
}

/// Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.
///
/// # Arguments
///
/// * `text` - A reference to a text.
///
/// * `ty` - Type of the text, `txt` for text content or `md` for markdown content.
///
/// * `chunk_capacity` - The max tokens each chunk contains.
///
/// # Returns
///
/// A vector of strings.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn chunk_text(
    text: impl AsRef<str>,
    ty: impl AsRef<str>,
    chunk_capacity: usize,
) -> Result<Vec<String>, LlamaCoreError> {
    if ty.as_ref().to_lowercase().as_str() != "txt" && ty.as_ref().to_lowercase().as_str() != "md" {
        let err_msg = "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported.";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::Operation(err_msg.into()));
    }

    match ty.as_ref().to_lowercase().as_str() {
        "txt" => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Chunk the plain text contents.");

            let tokenizer = cl100k_base().map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // create a text splitter
            let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);

            let chunks = splitter
                .chunks(text.as_ref(), chunk_capacity)
                .map(|s| s.to_string())
                .collect::<Vec<_>>();

            #[cfg(feature = "logging")]
            info!(target: "stdout", "Number of chunks: {}", chunks.len());

            Ok(chunks)
        }
        "md" => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Chunk the markdown contents.");

            let tokenizer = cl100k_base().map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // create a markdown splitter
            let splitter = MarkdownSplitter::new(tokenizer).with_trim_chunks(true);

            let chunks = splitter
                .chunks(text.as_ref(), chunk_capacity)
                .map(|s| s.to_string())
                .collect::<Vec<_>>();

            #[cfg(feature = "logging")]
            info!(target: "stdout", "Number of chunks: {}", chunks.len());

            Ok(chunks)
        }
        _ => {
            let err_msg =
                "Failed to upload the target file. Only text and markdown files are supported.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            Err(LlamaCoreError::Operation(err_msg.into()))
        }
    }
}
