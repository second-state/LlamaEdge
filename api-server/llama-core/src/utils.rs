//! Define utility functions.

use crate::{
    error::{BackendError, LlamaCoreError},
    Graph, CHAT_GRAPHS, EMBEDDING_GRAPHS, MAX_BUFFER_SIZE,
};
use chat_prompts::PromptTemplateType;
use serde_json::Value;

pub(crate) fn print_log_begin_separator(
    title: impl AsRef<str>,
    ch: Option<&str>,
    len: Option<usize>,
) -> usize {
    let title = format!(" [LOG: {}] ", title.as_ref());

    let total_len: usize = len.unwrap_or(100);
    let separator_len: usize = (total_len - title.len()) / 2;

    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str(&title);
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push('\n');
    println!("{}", separator);
    total_len
}

pub(crate) fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push_str("\n\n");
    println!("{}", separator);
}

pub(crate) fn gen_chat_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

/// Return the names of the chat models.
pub fn chat_model_names() -> Result<Vec<String>, LlamaCoreError> {
    let chat_graphs = CHAT_GRAPHS
        .get()
        .ok_or(LlamaCoreError::Operation(String::from(
            "Fail to get the underlying value of `CHAT_GRAPHS`.",
        )))?;

    let chat_graphs = chat_graphs.lock().map_err(|e| {
        LlamaCoreError::Operation(format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e))
    })?;

    let mut model_names = Vec::new();
    for model_name in chat_graphs.keys() {
        model_names.push(model_name.clone());
    }

    Ok(model_names)
}

/// Return the names of the embedding models.
pub fn embedding_model_names() -> Result<Vec<String>, LlamaCoreError> {
    let embedding_graphs =
        EMBEDDING_GRAPHS
            .get()
            .ok_or(LlamaCoreError::Operation(String::from(
                "Fail to get the underlying value of `EMBEDDING_GRAPHS`.",
            )))?;

    let embedding_graphs = embedding_graphs.lock().map_err(|e| {
        LlamaCoreError::Operation(format!(
            "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
            e
        ))
    })?;

    let mut model_names = Vec::new();
    for model_name in embedding_graphs.keys() {
        model_names.push(model_name.clone());
    }

    Ok(model_names)
}

/// Get the chat prompt template type from the given model name.
pub fn chat_prompt_template(name: Option<&str>) -> Result<PromptTemplateType, LlamaCoreError> {
    let chat_graphs = CHAT_GRAPHS
        .get()
        .ok_or(LlamaCoreError::Operation(String::from(
            "Fail to get the underlying value of `CHAT_GRAPHS`.",
        )))?;

    let chat_graphs = chat_graphs.lock().map_err(|e| {
        LlamaCoreError::Operation(format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e))
    })?;

    match name {
        Some(name) => match chat_graphs.get(name) {
            Some(graph) => Ok(graph.prompt_template()),
            None => Err(LlamaCoreError::Operation(format!(
                "Not found `{}` chat model.",
                name
            ))),
        },
        None => match chat_graphs.iter().next() {
            Some((_, graph)) => Ok(graph.prompt_template()),
            None => Err(LlamaCoreError::Operation(String::from(
                "There is no model available in the chat graphs.",
            ))),
        },
    }
}

/// Get output buffer generated by model.
pub(crate) fn get_output_buffer(graph: &Graph, index: usize) -> Result<Vec<u8>, LlamaCoreError> {
    let mut output_buffer: Vec<u8> = Vec::with_capacity(MAX_BUFFER_SIZE);

    let output_size: usize = graph.get_output(index, &mut output_buffer).map_err(|e| {
        LlamaCoreError::Backend(BackendError::GetOutput(format!(
            "Fail to get plugin metadata. {msg}",
            msg = e
        )))
    })?;

    unsafe {
        output_buffer.set_len(output_size);
    }

    Ok(output_buffer)
}

/// Get output buffer generated by model in the stream mode.
pub(crate) fn get_output_buffer_single(
    graph: &Graph,
    index: usize,
) -> Result<Vec<u8>, LlamaCoreError> {
    let mut output_buffer: Vec<u8> = Vec::with_capacity(MAX_BUFFER_SIZE);

    let output_size: usize = graph
        .get_output_single(index, &mut output_buffer)
        .map_err(|e| {
            LlamaCoreError::Backend(BackendError::GetOutput(format!(
                "Fail to get plugin metadata. {msg}",
                msg = e
            )))
        })?;

    unsafe {
        output_buffer.set_len(output_size);
    }

    Ok(output_buffer)
}

pub(crate) fn set_tensor_data_u8(
    graph: &mut Graph,
    idx: usize,
    tensor_data: &[u8],
) -> Result<(), LlamaCoreError> {
    if graph
        .set_input(idx, wasmedge_wasi_nn::TensorType::U8, &[1], tensor_data)
        .is_err()
    {
        let err_msg = format!("Fail to set input tensor at index {}", idx);

        #[cfg(feature = "logging")]
        error!(target: "llama-core", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    };

    Ok(())
}

/// Get the token information from the graph.
pub(crate) fn get_token_info_by_graph(graph: &Graph) -> Result<TokenInfo, LlamaCoreError> {
    let output_buffer = get_output_buffer(graph, 1)?;
    let token_info: Value = match serde_json::from_slice(&output_buffer[..]) {
        Ok(token_info) => token_info,
        Err(e) => {
            return Err(LlamaCoreError::Operation(format!(
                "Fail to deserialize token info: {msg}",
                msg = e
            )));
        }
    };

    let prompt_tokens = match token_info["input_tokens"].as_u64() {
        Some(prompt_tokens) => prompt_tokens,
        None => {
            return Err(LlamaCoreError::Operation(String::from(
                "Fail to convert `input_tokens` to u64.",
            )));
        }
    };
    let completion_tokens = match token_info["output_tokens"].as_u64() {
        Some(completion_tokens) => completion_tokens,
        None => {
            return Err(LlamaCoreError::Operation(String::from(
                "Fail to convert `output_tokens` to u64.",
            )));
        }
    };
    Ok(TokenInfo {
        prompt_tokens,
        completion_tokens,
    })
}

/// Get the token information from the graph by the model name.
pub(crate) fn get_token_info_by_graph_name(
    name: Option<&String>,
) -> Result<TokenInfo, LlamaCoreError> {
    match name {
        Some(model_name) => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;
            match chat_graphs.get(model_name) {
                Some(graph) => get_token_info_by_graph(graph),
                None => Err(LlamaCoreError::Operation(format!(
                    "The model `{}` does not exist in the chat graphs.",
                    &model_name
                ))),
            }
        }
        None => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;

            match chat_graphs.iter().next() {
                Some((_, graph)) => get_token_info_by_graph(graph),
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}

#[derive(Debug)]

pub(crate) struct TokenInfo {
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
}
