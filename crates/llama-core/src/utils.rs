//! Define utility functions.

use crate::{
    error::{BackendError, LlamaCoreError},
    BaseMetadata, Graph, CHAT_GRAPHS, EMBEDDING_GRAPHS, MAX_BUFFER_SIZE,
};
use chat_prompts::PromptTemplateType;
use serde_json::Value;

pub(crate) fn gen_chat_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

/// Return the names of the chat models.
pub fn chat_model_names() -> Result<Vec<String>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the names of the chat models.");

    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let mut model_names = Vec::new();
    for model_name in chat_graphs.keys() {
        model_names.push(model_name.clone());
    }

    Ok(model_names)
}

/// Return the names of the embedding models.
pub fn embedding_model_names() -> Result<Vec<String>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the names of the embedding models.");

    let embedding_graphs = match EMBEDDING_GRAPHS.get() {
        Some(embedding_graphs) => embedding_graphs,
        None => {
            return Err(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
            )));
        }
    };

    let embedding_graphs = match embedding_graphs.lock() {
        Ok(embedding_graphs) => embedding_graphs,
        Err(e) => {
            let err_msg = format!("Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    let mut model_names = Vec::new();
    for model_name in embedding_graphs.keys() {
        model_names.push(model_name.clone());
    }

    Ok(model_names)
}

/// Get the chat prompt template type from the given model name.
pub fn chat_prompt_template(name: Option<&str>) -> Result<PromptTemplateType, LlamaCoreError> {
    #[cfg(feature = "logging")]
    match name {
        Some(name) => {
            info!(target: "stdout", "Get the chat prompt template type from the chat model named {}.", name)
        }
        None => {
            info!(target: "stdout", "Get the chat prompt template type from the default chat model.")
        }
    }

    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match name {
        Some(model_name) => match chat_graphs.contains_key(model_name) {
            true => {
                let graph = chat_graphs.get(model_name).unwrap();
                let prompt_template = graph.metadata.prompt_template();

                #[cfg(feature = "logging")]
                info!(target: "stdout", "prompt_template: {}", &prompt_template);

                Ok(prompt_template)
            }
            false => match chat_graphs.iter().next() {
                Some((_, graph)) => {
                    let prompt_template = graph.metadata.prompt_template();

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "prompt_template: {}", &prompt_template);

                    Ok(prompt_template)
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match chat_graphs.iter().next() {
            Some((_, graph)) => {
                let prompt_template = graph.metadata.prompt_template();

                #[cfg(feature = "logging")]
                info!(target: "stdout", "prompt_template: {}", &prompt_template);

                Ok(prompt_template)
            }
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

/// Get output buffer generated by model.
pub(crate) fn get_output_buffer<M>(
    graph: &Graph<M>,
    index: usize,
) -> Result<Vec<u8>, LlamaCoreError>
where
    M: BaseMetadata + serde::Serialize + Clone + Default,
{
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the output buffer generated by the model named {}", graph.name());

    let mut output_buffer: Vec<u8> = Vec::with_capacity(MAX_BUFFER_SIZE);

    let output_size: usize = graph.get_output(index, &mut output_buffer).map_err(|e| {
        let err_msg = format!("Fail to get the generated output tensor. {msg}", msg = e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Backend(BackendError::GetOutput(err_msg))
    })?;

    unsafe {
        output_buffer.set_len(output_size);
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Output buffer size: {}", output_size);

    Ok(output_buffer)
}

/// Get output buffer generated by model in the stream mode.
pub(crate) fn get_output_buffer_single<M>(
    graph: &Graph<M>,
    index: usize,
) -> Result<Vec<u8>, LlamaCoreError>
where
    M: BaseMetadata + serde::Serialize + Clone + Default,
{
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get output buffer generated by the model named {} in the stream mode.", graph.name());

    let mut output_buffer: Vec<u8> = Vec::with_capacity(MAX_BUFFER_SIZE);

    let output_size: usize = graph
        .get_output_single(index, &mut output_buffer)
        .map_err(|e| {
            let err_msg = format!("Fail to get plugin metadata. {msg}", msg = e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Backend(BackendError::GetOutput(err_msg))
        })?;

    unsafe {
        output_buffer.set_len(output_size);
    }

    Ok(output_buffer)
}

pub(crate) fn set_tensor_data_u8<M>(
    graph: &mut Graph<M>,
    idx: usize,
    tensor_data: &[u8],
) -> Result<(), LlamaCoreError>
where
    M: BaseMetadata + serde::Serialize + Clone + Default,
{
    if graph
        .set_input(idx, wasmedge_wasi_nn::TensorType::U8, &[1], tensor_data)
        .is_err()
    {
        let err_msg = format!("Fail to set input tensor at index {}", idx);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    };

    Ok(())
}

/// Get the token information from the graph.
pub(crate) fn get_token_info_by_graph<M>(graph: &Graph<M>) -> Result<TokenInfo, LlamaCoreError>
where
    M: BaseMetadata + serde::Serialize + Clone + Default,
{
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get token info from the model named {}.", graph.name());

    let output_buffer = get_output_buffer(graph, 1)?;
    let token_info: Value = match serde_json::from_slice(&output_buffer[..]) {
        Ok(token_info) => token_info,
        Err(e) => {
            let err_msg = format!("Fail to deserialize token info: {msg}", msg = e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    let prompt_tokens = match token_info["input_tokens"].as_u64() {
        Some(prompt_tokens) => prompt_tokens,
        None => {
            let err_msg = "Fail to convert `input_tokens` to u64.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };
    let completion_tokens = match token_info["output_tokens"].as_u64() {
        Some(completion_tokens) => completion_tokens,
        None => {
            let err_msg = "Fail to convert `output_tokens` to u64.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", prompt_tokens, completion_tokens);

    Ok(TokenInfo {
        prompt_tokens,
        completion_tokens,
    })
}

/// Get the token information from the graph by the model name.
pub(crate) fn get_token_info_by_graph_name(
    name: Option<&String>,
) -> Result<TokenInfo, LlamaCoreError> {
    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match name {
        Some(model_name) => match chat_graphs.contains_key(model_name) {
            true => {
                let graph = chat_graphs.get(model_name).unwrap();
                get_token_info_by_graph(graph)
            }
            false => match chat_graphs.iter().next() {
                Some((_, graph)) => get_token_info_by_graph(graph),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match chat_graphs.iter().next() {
            Some((_, graph)) => get_token_info_by_graph(graph),
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

#[derive(Debug)]
pub(crate) struct TokenInfo {
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
}

pub(crate) trait TensorType {
    fn tensor_type() -> wasmedge_wasi_nn::TensorType;
    fn shape(shape: impl AsRef<[usize]>) -> Vec<usize> {
        shape.as_ref().to_vec()
    }
}

impl TensorType for u8 {
    fn tensor_type() -> wasmedge_wasi_nn::TensorType {
        wasmedge_wasi_nn::TensorType::U8
    }
}

impl TensorType for f32 {
    fn tensor_type() -> wasmedge_wasi_nn::TensorType {
        wasmedge_wasi_nn::TensorType::F32
    }
}

pub(crate) fn set_tensor_data<T, M>(
    graph: &mut Graph<M>,
    idx: usize,
    tensor_data: &[T],
    shape: impl AsRef<[usize]>,
) -> Result<(), LlamaCoreError>
where
    T: TensorType,
    M: BaseMetadata + serde::Serialize + Clone + Default,
{
    if graph
        .set_input(idx, T::tensor_type(), &T::shape(shape), tensor_data)
        .is_err()
    {
        let err_msg = format!("Fail to set input tensor at index {}", idx);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    };

    Ok(())
}
