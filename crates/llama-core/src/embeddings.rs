//! Define APIs for computing embeddings.

use crate::{
    error::{BackendError, LlamaCoreError},
    metadata::ggml::GgmlMetadata,
    running_mode,
    utils::{get_output_buffer, get_token_info_by_graph},
    Graph, RunningMode, CHAT_GRAPHS, EMBEDDING_GRAPHS, OUTPUT_TENSOR,
};
use endpoints::{
    common::Usage,
    embeddings::{EmbeddingObject, EmbeddingRequest, EmbeddingsResponse, InputText},
};
use serde::{Deserialize, Serialize};

/// Compute embeddings for the given input.
///
/// # Argument
///
/// * `embedding_request` - The embedding request.
///
/// # Returns
///
/// The embeddings response.
pub async fn embeddings(
    embedding_request: &EmbeddingRequest,
) -> Result<EmbeddingsResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Computing embeddings");

    let running_mode = running_mode()?;
    if running_mode == RunningMode::Chat {
        let err_msg = format!(
            "Computing embeddings is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let model_name = &embedding_request.model;

    // For general embedding scenario, the embedding model is the same as the chat model.
    // For RAG scenario, the embedding model is different from the chat model.
    let embedding_graphs = match EMBEDDING_GRAPHS.get() {
        Some(embedding_graphs) => embedding_graphs,
        None => match CHAT_GRAPHS.get() {
            Some(chat_graphs) => chat_graphs,
            None => {
                let err_msg = "No embedding model is available.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        },
    };

    let mut embedding_graphs = embedding_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let graph = match embedding_graphs.contains_key(model_name) {
        true => embedding_graphs.get_mut(model_name).unwrap(),
        false => match embedding_graphs.iter_mut().next() {
            Some((_, graph)) => graph,
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        },
    };

    // check if the `embedding` option of metadata is enabled
    if !graph.metadata.embeddings {
        graph.metadata.embeddings = true;
        graph.update_metadata()?;
    }

    // compute embeddings
    let (data, usage) = match &embedding_request.input {
        InputText::String(text) => compute_embeddings(graph, &[text.to_owned()])?,
        InputText::ArrayOfStrings(texts) => compute_embeddings(graph, texts.as_slice())?,
        InputText::ArrayOfTokens(tokens) => {
            let texts: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
            compute_embeddings(graph, texts.as_slice())?
        }
        InputText::ArrayOfTokenArrays(token_arrays) => {
            let texts: Vec<String> = token_arrays
                .iter()
                .map(|tokens| {
                    tokens
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<String>>()
                        .join(" ")
                })
                .collect();
            compute_embeddings(graph, texts.as_slice())?
        }
    };

    let embedding_reponse = EmbeddingsResponse {
        object: String::from("list"),
        data,
        model: graph.name().to_owned(),
        usage,
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Embeddings computed successfully.");

    Ok(embedding_reponse)
}

fn compute_embeddings(
    graph: &mut Graph<GgmlMetadata>,
    input: &[String],
) -> Result<(Vec<EmbeddingObject>, Usage), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute embeddings for {} chunks", input.len());

    // compute embeddings
    let mut embeddings: Vec<EmbeddingObject> = Vec::new();
    let mut usage = Usage::default();
    for (idx, input) in input.iter().enumerate() {
        // set input
        let tensor_data = input.as_bytes().to_vec();
        graph
            .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
            .map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Backend(BackendError::SetInput(err_msg))
            })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "compute embeddings for chunk {}", idx + 1);

        match graph.compute() {
            Ok(_) => {
                // Retrieve the output.
                let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;

                // convert inference result to string
                let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                    let err_msg = format!(
                        "Failed to decode the buffer of the inference result to a utf-8 string. Reason: {}",
                        e
                    );

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                // deserialize the embedding data
                let embedding = serde_json::from_str::<Embedding>(output).map_err(|e| {
                    let err_msg =
                        format!("Failed to deserialize the embedding data. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                let embedding_object = EmbeddingObject {
                    index: idx as u64,
                    object: String::from("embedding"),
                    embedding: embedding.data,
                };

                embeddings.push(embedding_object);

                // retrieve the number of prompt and completion tokens
                let token_info = get_token_info_by_graph(graph)?;

                usage.prompt_tokens += token_info.prompt_tokens;
                usage.completion_tokens += token_info.completion_tokens;
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
            }
            Err(e) => {
                let err_msg = format!("Failed to compute embeddings. Reason: {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Backend(BackendError::Compute(err_msg)));
            }
        }
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "token usage of embeddings: {} prompt tokens, {} comletion tokens", usage.prompt_tokens, usage.completion_tokens);

    Ok((embeddings, usage))
}

/// Get the dimension of the embedding model.
///
/// # Arguments
///
/// * `name` - The name of the embedding model. If `None`, the dimension of the first model will be returned.
///
/// # Returns
///
/// The dimension of the embedding model.
///
/// # Errors
///
/// * The model does not exist in the embedding graphs.
/// * No embedding model is available.
pub fn dimension(name: Option<&str>) -> Result<u64, LlamaCoreError> {
    let embedding_graphs = match EMBEDDING_GRAPHS.get() {
        Some(embedding_graphs) => embedding_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `EMBEDDING_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let embedding_graphs = embedding_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match name {
        Some(model_name) => match embedding_graphs.get(model_name) {
            Some(graph) => Ok(graph.metadata.ctx_size),
            None => {
                let err_msg = format!(
                    "The model `{}` does not exist in the embedding graphs.",
                    model_name
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg))
            }
        },
        None => {
            if !embedding_graphs.is_empty() {
                let graph = match embedding_graphs.values().next() {
                    Some(graph) => graph,
                    None => {
                        let err_msg = "Fail to get the underlying value of `EMBEDDING_GRAPHS`.";

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.into()));
                    }
                };

                Ok(graph.metadata.ctx_size)
            } else {
                let err_msg = "There is no model available in the embedding graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    #[serde(rename = "n_embedding")]
    len: u64,
    #[serde(rename = "embedding")]
    data: Vec<f64>,
}
