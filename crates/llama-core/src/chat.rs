//! Define APIs for chat completion.

use crate::{
    error,
    metadata::ggml::GgmlMetadata,
    running_mode,
    utils::{
        gen_chat_id, get_output_buffer, get_output_buffer_single, get_token_info_by_graph,
        get_token_info_by_graph_name, set_tensor_data_u8,
    },
    Graph, RunningMode, CACHED_UTF8_ENCODINGS, CHAT_GRAPHS, OUTPUT_TENSOR,
};
use chat_prompts::{
    chat::{BuildChatPrompt, ChatPrompt},
    PromptTemplateType,
};
use either::{Either, Left, Right};
use endpoints::{
    chat::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkChoiceDelta,
        ChatCompletionObject, ChatCompletionObjectChoice, ChatCompletionObjectMessage,
        ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRole,
        ChatCompletionUserMessageContent, ContentPart, Function, ToolCall, ToolCallForChunk,
        ToolChoice,
    },
    common::{FinishReason, Usage},
};
use error::{BackendError, LlamaCoreError};
use futures::StreamExt;
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll},
    time::SystemTime,
};

/// Processes a chat-completion request and returns either a stream of ChatCompletionChunk instances or a ChatCompletionObject instance.
pub async fn chat(
    chat_request: &mut ChatCompletionRequest,
) -> Result<
    Either<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, ChatCompletionObject>,
    LlamaCoreError,
> {
    #[cfg(feature = "logging")]
    {
        info!(target: "stdout", "tool choice: {:?}", chat_request.tool_choice.as_ref());
        info!(target: "stdout", "tools: {:?}", chat_request.tools.as_ref());
        info!(target: "stdout", "stream mode: {:?}", chat_request.stream);
    }

    match chat_request.stream {
        Some(true) => match chat_stream(chat_request).await {
            Ok(stream) => Ok(Left(stream)),
            Err(e) => Err(e),
        },
        Some(false) | None => match chat_once(chat_request).await {
            Ok(chat_completion_object) => Ok(Right(chat_completion_object)),
            Err(e) => Err(e),
        },
    }
}

/// Processes a chat-completion request and returns ChatCompletionChunk instances in stream.
#[deprecated(since = "0.10.0", note = "Please use the `chat` function.")]
pub async fn chat_completions_stream(
    chat_request: &mut ChatCompletionRequest,
) -> Result<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, LlamaCoreError> {
    chat_stream(chat_request).await
}

/// Processes a chat-completion request and returns a ChatCompletionObject instance.
#[deprecated(since = "0.10.0", note = "Please use the `chat` function.")]
pub async fn chat_completions(
    chat_request: &mut ChatCompletionRequest,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    chat_once(chat_request).await
}

async fn chat_stream(
    chat_request: &mut ChatCompletionRequest,
) -> Result<impl futures::TryStream<Ok = String, Error = LlamaCoreError>, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Process chat completion request in the stream mode.");

    let running_mode = running_mode()?;
    if running_mode == RunningMode::Embeddings {
        let err_msg = format!(
            "The chat completion is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let model_name = chat_request.model.clone();
    let id = match &chat_request.user {
        Some(id) => id.clone(),
        None => gen_chat_id(),
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "user: {}", &id);

    // parse the `include_usage` option
    let include_usage = match chat_request.stream_options {
        Some(ref stream_options) => stream_options.include_usage.unwrap_or_default(),
        None => false,
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "include_usage: {}", include_usage);

    // update metadata
    let mut metadata = check_model_metadata(chat_request).await?;

    // build prompt
    let (prompt, avaible_completion_tokens, tool_use) =
        build_prompt(model_name.as_ref(), chat_request)?;

    #[cfg(feature = "logging")]
    {
        info!(target: "stdout", "prompt:\n{}", &prompt);
        info!(target: "stdout", "available_completion_tokens: {}", avaible_completion_tokens);
        info!(target: "stdout", "tool_use: {}", tool_use);
    }

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    // set prompt
    set_prompt(chat_request.model.as_ref(), &prompt)?;

    let stream = match tool_use {
        false => ChatStream::new(model_name, id, include_usage, None),
        true => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            match model_name {
                Some(model_name) => match chat_graphs.contains_key(&model_name) {
                    true => {
                        let graph = chat_graphs.get_mut(&model_name).unwrap();
                        chat_stream_by_graph(graph, id, include_usage)?
                    }
                    false => match chat_graphs.iter_mut().next() {
                        Some((_, graph)) => chat_stream_by_graph(graph, id, include_usage)?,
                        None => {
                            let err_msg = "There is no model available in the chat graphs.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg.into()));
                        }
                    },
                },
                None => match chat_graphs.iter_mut().next() {
                    Some((_, graph)) => chat_stream_by_graph(graph, id, include_usage)?,
                    None => {
                        let err_msg = "There is no model available in the chat graphs.";

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.into()));
                    }
                },
            }
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the chat completion stream.");

    Ok(stream)
}

fn chat_stream_by_graph(
    graph: &mut Graph<GgmlMetadata>,
    id: impl Into<String>,
    include_usage: bool,
) -> Result<ChatStream, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Handle chat request with available tools by the model named {}.", graph.name());

    let id = id.into();

    match graph.compute() {
        Ok(_) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw generation:\n{}", output);

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "post-processed generation:\n{}", &message);

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let usage = Some(Usage {
                prompt_tokens: token_info.prompt_tokens,
                completion_tokens: token_info.completion_tokens,
                total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
            });

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            if graph.metadata.prompt_template != PromptTemplateType::MistralTool
                && graph.metadata.prompt_template != PromptTemplateType::ChatMLTool
                && graph.metadata.prompt_template != PromptTemplateType::GroqLlama3Tool
                && graph.metadata.prompt_template != PromptTemplateType::Llama3Tool
                && graph.metadata.prompt_template != PromptTemplateType::InternLM2Tool
            {
                let err_msg = "The tool use is only supported for 'mistral-chat' and 'chatml' prompt templates.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }

            let parsed_result = parse_tool_calls(&message, graph.metadata.prompt_template)?;

            let content = match parsed_result.content {
                Some(content) => Some(content),
                None => Some(parsed_result.raw),
            };

            let tool_calls: Vec<ToolCallForChunk> = parsed_result
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, tool_call)| ToolCallForChunk {
                    index,
                    id: tool_call.id,
                    ty: tool_call.ty,
                    function: tool_call.function,
                })
                .collect();

            // tool_calls chunk
            let tool_call_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkChoiceDelta {
                            role: ChatCompletionRole::Assistant,
                            content,
                            tool_calls,
                        },
                        logprobs: None,
                        finish_reason: None,
                    }],
                    usage: None,
                };
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // uage chunk
            let usage_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![],
                    usage,
                };
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // ending chunk
            let ending_chunk = "data: [DONE]\n\n".to_string();

            let chunks = vec![tool_call_chunk, usage_chunk, ending_chunk];

            Ok(ChatStream::new(
                Some(graph.name().to_owned()),
                id,
                include_usage,
                Some(chunks),
            ))
        }
        Err(wasmedge_wasi_nn::Error::BackendError(wasmedge_wasi_nn::BackendError::ContextFull)) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let usage = Some(Usage {
                prompt_tokens: token_info.prompt_tokens,
                completion_tokens: token_info.completion_tokens,
                total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
            });

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // context full chunk
            let context_full_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkChoiceDelta {
                            role: ChatCompletionRole::Assistant,
                            content: Some(message),
                            tool_calls: vec![],
                        },
                        logprobs: None,
                        finish_reason: Some(FinishReason::length),
                    }],
                    usage: None,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // usage chunk
            let usage_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![],
                    usage,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // ending chunk
            let ending_chunk = "data: [DONE]\n\n".to_string();

            let chunks = vec![context_full_chunk, usage_chunk, ending_chunk];

            Ok(ChatStream::new(
                Some(graph.name().to_owned()),
                id,
                include_usage,
                Some(chunks),
            ))
        }
        Err(wasmedge_wasi_nn::Error::BackendError(
            wasmedge_wasi_nn::BackendError::PromptTooLong,
        )) => {
            #[cfg(feature = "logging")]
            warn!(target: "stdout", "The prompt is too long. Please reduce the length of your input and try again.");

            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion token
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let usage = Some(Usage {
                prompt_tokens: token_info.prompt_tokens,
                completion_tokens: token_info.completion_tokens,
                total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
            });

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // prompt too long chunk
            let prompt_too_long_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkChoiceDelta {
                            role: ChatCompletionRole::Assistant,
                            content: Some(message),
                            tool_calls: vec![],
                        },
                        logprobs: None,
                        finish_reason: Some(FinishReason::length),
                    }],
                    usage: None,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // usage chunk
            let usage_chunk = {
                let chat_completion_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created.as_secs(),
                    model: graph.name().to_owned(),
                    system_fingerprint: "fp_44709d6fcb".to_string(),
                    choices: vec![],
                    usage,
                };

                // serialize chat completion chunk
                let chunk_str = serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                    let err_msg =
                        format!("Failed to serialize chat completion chunk. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                format!("data: {}\n\n", chunk_str)
            };

            // ending chunk
            let ending_chunk = "data: [DONE]\n\n".to_string();

            let chunks = vec![prompt_too_long_chunk, usage_chunk, ending_chunk];

            Ok(ChatStream::new(
                Some(graph.name().to_owned()),
                id,
                include_usage,
                Some(chunks),
            ))
        }
        Err(e) => {
            let err_msg = format!("Failed to compute the chat completion. Reason: {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Backend(BackendError::Compute(err_msg)))
        }
    }
}

async fn chat_once(
    chat_request: &mut ChatCompletionRequest,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Processing chat completion request in non-stream mode.");

    let running_mode = running_mode()?;
    if running_mode == RunningMode::Embeddings {
        let err_msg = format!(
            "The chat completion is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let model_name = chat_request.model.clone();
    let id = match &chat_request.user {
        Some(id) => id.clone(),
        None => gen_chat_id(),
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "user: {}", &id);

    // update metadata
    let mut metadata = check_model_metadata(chat_request).await?;

    // build prompt
    let (prompt, avaible_completion_tokens, tool_use) =
        build_prompt(model_name.as_ref(), chat_request)?;

    #[cfg(feature = "logging")]
    {
        info!(target: "stdout", "prompt:\n{}", &prompt);
        info!(target: "stdout", "available_completion_tokens: {}", avaible_completion_tokens);
        info!(target: "stdout", "tool_use: {}", tool_use);
    }

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    // feed the prompt to the model
    set_prompt(model_name.as_ref(), &prompt)?;

    // compute
    let res = compute(model_name.as_ref(), id, tool_use);

    #[cfg(feature = "logging")]
    info!(target: "stdout", "End of the chat completion.");

    res
}

fn compute(
    model_name: Option<&String>,
    id: impl Into<String>,
    tool_use: bool,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute chat completion.");

    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => match chat_graphs.contains_key(model_name) {
            true => {
                let graph = chat_graphs.get_mut(model_name).unwrap();
                compute_by_graph(graph, id, tool_use)
            }
            false => match chat_graphs.iter_mut().next() {
                Some((_, graph)) => compute_by_graph(graph, id, tool_use),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match chat_graphs.iter_mut().next() {
            Some((_, graph)) => compute_by_graph(graph, id, tool_use),
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

fn compute_by_graph(
    graph: &mut Graph<GgmlMetadata>,
    id: impl Into<String>,
    tool_use: bool,
) -> Result<ChatCompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute chat completion by the model named {}.", graph.name());

    match graph.compute() {
        Ok(_) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw generation: {}", output);

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                LlamaCoreError::Operation(format!("Failed to post-process the output. {}", e))
            })?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "post-processed generation:\n{}", &message);

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            match tool_use {
                true => {
                    if graph.metadata.prompt_template != PromptTemplateType::MistralTool
                        && graph.metadata.prompt_template != PromptTemplateType::ChatMLTool
                        && graph.metadata.prompt_template != PromptTemplateType::GroqLlama3Tool
                        && graph.metadata.prompt_template != PromptTemplateType::Llama3Tool
                        && graph.metadata.prompt_template != PromptTemplateType::InternLM2Tool
                        && graph.metadata.prompt_template != PromptTemplateType::NemotronTool
                        && graph.metadata.prompt_template != PromptTemplateType::FunctionaryV32
                        && graph.metadata.prompt_template != PromptTemplateType::FunctionaryV31
                    {
                        let err_msg = "The tool use is only supported for 'mistral-tool', 'chatml', 'groq-llama3-tool', 'llama-3-tool', 'internlm-2-tool', 'nemotron-tool', 'functionary-31', and 'functionary-32' prompt templates.";

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.into()));
                    }

                    let parsed_result = parse_tool_calls(&message, graph.metadata.prompt_template)?;

                    let finish_reason = if parsed_result.tool_calls.is_empty() {
                        FinishReason::stop
                    } else {
                        FinishReason::tool_calls
                    };

                    let content = match parsed_result.content {
                        Some(content) => Some(content),
                        None => Some(parsed_result.raw),
                    };

                    // create ChatCompletionResponse
                    Ok(ChatCompletionObject {
                        id: id.into(),
                        object: String::from("chat.completion"),
                        created: created.as_secs(),
                        model: graph.name().to_owned(),
                        choices: vec![ChatCompletionObjectChoice {
                            index: 0,
                            message: ChatCompletionObjectMessage {
                                role: ChatCompletionRole::Assistant,
                                content,
                                tool_calls: parsed_result.tool_calls,
                                function_call: None,
                            },
                            finish_reason,
                            logprobs: None,
                        }],
                        usage: Usage {
                            prompt_tokens: token_info.prompt_tokens,
                            completion_tokens: token_info.completion_tokens,
                            total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                        },
                    })
                }
                false => {
                    // create ChatCompletionResponse
                    Ok(ChatCompletionObject {
                        id: id.into(),
                        object: String::from("chat.completion"),
                        created: created.as_secs(),
                        model: graph.name().to_owned(),
                        choices: vec![ChatCompletionObjectChoice {
                            index: 0,
                            message: ChatCompletionObjectMessage {
                                role: ChatCompletionRole::Assistant,
                                content: Some(message),
                                tool_calls: vec![],
                                function_call: None,
                            },
                            finish_reason: FinishReason::stop,
                            logprobs: None,
                        }],
                        usage: Usage {
                            prompt_tokens: token_info.prompt_tokens,
                            completion_tokens: token_info.completion_tokens,
                            total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                        },
                    })
                }
            }
        }
        Err(wasmedge_wasi_nn::Error::BackendError(wasmedge_wasi_nn::BackendError::ContextFull)) => {
            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion tokens
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: id.into(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: graph.name().to_owned(),
                choices: vec![ChatCompletionObjectChoice {
                    index: 0,
                    message: ChatCompletionObjectMessage {
                        role: ChatCompletionRole::Assistant,
                        content: Some(message),
                        tool_calls: vec![],
                        function_call: None,
                    },
                    finish_reason: FinishReason::length,
                    logprobs: None,
                }],
                usage: Usage {
                    prompt_tokens: token_info.prompt_tokens,
                    completion_tokens: token_info.completion_tokens,
                    total_tokens: token_info.prompt_tokens + token_info.completion_tokens,
                },
            })
        }
        Err(wasmedge_wasi_nn::Error::BackendError(
            wasmedge_wasi_nn::BackendError::PromptTooLong,
        )) => {
            #[cfg(feature = "logging")]
            warn!(target: "stdout", "The prompt is too long. Please reduce the length of your input and try again.");

            // Retrieve the output.
            let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;
            let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                let err_msg = format!(
                    "Failed to decode the buffer of the inference result to a utf-8 string. {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // post-process
            let message = post_process(output, &graph.metadata.prompt_template).map_err(|e| {
                let err_msg = format!("Failed to post-process the output. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            // retrieve the number of prompt and completion token
            let token_info = get_token_info_by_graph(graph)?;

            #[cfg(feature = "logging")]
            info!(target: "stdout", "prompt tokens: {}, completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

            let created = SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    let err_msg = format!("Failed to get the current time. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // create ChatCompletionResponse
            Ok(ChatCompletionObject {
                id: id.into(),
                object: String::from("chat.completion"),
                created: created.as_secs(),
                model: graph.name().to_owned(),
                choices: vec![ChatCompletionObjectChoice {
                    index: 0,
                    message: ChatCompletionObjectMessage {
                        role: ChatCompletionRole::Assistant,
                        content: Some(message),
                        tool_calls: vec![],
                        function_call: None,
                    },
                    finish_reason: FinishReason::length,
                    logprobs: None,
                }],
                usage: Usage {
                    prompt_tokens: token_info.prompt_tokens,
                    completion_tokens: token_info.completion_tokens,
                    total_tokens: token_info.completion_tokens + token_info.completion_tokens,
                },
            })
        }
        Err(e) => {
            let err_msg = format!("Failed to compute the chat completion. Reason: {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Backend(BackendError::Compute(err_msg)))
        }
    }
}

fn parse_tool_calls(
    input: &str,
    prompt_template: PromptTemplateType,
) -> Result<ParseResult, LlamaCoreError> {
    match prompt_template {
        PromptTemplateType::MistralTool => match regex::Regex::new(r"\[\{.*?\}\]") {
            Ok(re) => {
                let mut values: Vec<serde_json::Value> = vec![];
                for cap in re.captures_iter(input) {
                    let matched = &cap[0];

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "captured: {}", matched);

                    match serde_json::from_str::<Vec<serde_json::Value>>(matched) {
                        Ok(group) => values.extend(group),
                        Err(e) => {
                            let err_msg = format!(
                                "Failed to deserialize generated tool calls. Reason: {}",
                                e
                            );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    }
                }

                let mut tool_calls: Vec<ToolCall> = vec![];
                for value in values.iter() {
                    let name = match value.get("name") {
                        Some(name) => name.to_string().replace("\"", ""),
                        None => {
                            let err_msg = format!(
                                "Failed to get the name of the function. Tool call: {:?}",
                                value
                            );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    };

                    let arguments = match value.get("arguments") {
                        Some(arguments) => arguments.to_string(),
                        None => {
                            let err_msg = format!(
                                "Failed to get the arguments of the function. Tool call: {:?}",
                                value
                            );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    };

                    let function = Function { name, arguments };

                    let tool_call = ToolCall {
                        id: "call_abc123".to_string(),
                        ty: "function".to_string(),
                        function,
                    };

                    tool_calls.push(tool_call);
                }

                let parsed = ParseResult {
                    raw: input.to_owned(),
                    content: None,
                    tool_calls,
                };

                #[cfg(feature = "logging")]
                info!(target: "stdout", "parsed result: {:?}", parsed);

                Ok(parsed)
            }
            Err(e) => {
                let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg))
            }
        },
        PromptTemplateType::ChatMLTool => {
            match regex::Regex::new(r"<tool_call>(.*?)</tool_call>") {
                Ok(re) => {
                    let mut values: Vec<serde_json::Value> = vec![];
                    for cap in re.captures_iter(input) {
                        let matched = cap[1].replace("\\n", ""); // Remove "\\n" from the captured group

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", &matched);

                        match serde_json::from_str::<serde_json::Value>(&matched) {
                            Ok(value) => values.push(value),
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }

                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for value in values.iter() {
                        let name = match value.get("name") {
                            Some(name) => name.to_string().replace("\"", ""),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the name of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let arguments = match value.get("arguments") {
                            Some(arguments) => arguments.to_string(),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the arguments of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let function = Function { name, arguments };

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function,
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        PromptTemplateType::GroqLlama3Tool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r"(?s)<tool_call>((.|\r|\n)*?)</tool_call>") {
                Ok(re) => {
                    let mut values: Vec<serde_json::Value> = vec![];
                    for cap in re.captures_iter(input) {
                        let matched = cap[1].trim();

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", matched);

                        match serde_json::from_str::<serde_json::Value>(matched) {
                            Ok(value) => values.push(value),
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }

                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for value in values.iter() {
                        let name = match value.get("name") {
                            Some(name) => name.to_string().replace("\"", ""),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the name of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let arguments = match value.get("arguments") {
                            Some(arguments) => arguments.to_string(),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the arguments of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let function = Function { name, arguments };

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function,
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        PromptTemplateType::Llama3Tool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            let re = match regex::Regex::new(r"^\{(.|\r|\n)*\}$") {
                Ok(re) => re,
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            };

            if re.is_match(input) {
                match serde_json::from_str::<serde_json::Value>(input) {
                    Ok(value) => {
                        let values: Vec<serde_json::Value> = vec![value];

                        let mut tool_calls: Vec<ToolCall> = vec![];
                        for value in values.iter() {
                            let name = match value.get("name") {
                                Some(name) => name.to_string().replace("\"", ""),
                                None => {
                                    let err_msg = format!(
                                        "Failed to get the name of the function. Tool call: {:?}",
                                        value
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    return Err(LlamaCoreError::Operation(err_msg));
                                }
                            };

                            let arguments = match value.get("parameters") {
                                Some(arguments) => arguments.to_string(),
                                None => {
                                    let err_msg = format!(
                                        "Failed to get the arguments of the function. Tool call: {:?}",
                                        value
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    return Err(LlamaCoreError::Operation(err_msg));
                                }
                            };

                            let function = Function { name, arguments };

                            let tool_call = ToolCall {
                                id: "call_abc123".to_string(),
                                ty: "function".to_string(),
                                function,
                            };

                            tool_calls.push(tool_call);
                        }

                        let parsed = ParseResult {
                            raw: input.to_owned(),
                            content: None,
                            tool_calls,
                        };

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "parsed result: {:?}", parsed);

                        Ok(parsed)
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed to deserialize generated tool calls. Reason: {}", e);

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        Err(LlamaCoreError::Operation(err_msg))
                    }
                }
            } else {
                let parsed = ParseResult {
                    raw: input.to_owned(),
                    content: None,
                    tool_calls: vec![],
                };

                #[cfg(feature = "logging")]
                info!(target: "stdout", "parsed result: {:?}", parsed);

                Ok(parsed)
            }
        }
        PromptTemplateType::InternLM2Tool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            let blocks: Vec<&str> = input.trim().split("<|action_start|><|plugin|>").collect();

            #[cfg(feature = "logging")]
            info!(target: "stdout", "blocks: {:?}", blocks);

            let mut tool_calls: Vec<ToolCall> = vec![];
            let mut content = String::new();
            for block in blocks {
                let block = block.trim();
                if !block.is_empty() {
                    if block.ends_with("<|action_end|>") {
                        let value = block.trim().trim_end_matches("<|action_end|>");

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "tool call: {}", value);

                        match serde_json::from_str::<serde_json::Value>(value) {
                            Ok(value) => {
                                let name = match value.get("name") {
                                    Some(name) => name.to_string().replace("\"", ""),
                                    None => {
                                        let err_msg = format!(
                                            "Failed to get the name of the function. Tool call: {:?}",
                                            value
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        return Err(LlamaCoreError::Operation(err_msg));
                                    }
                                };

                                let arguments = match value.get("parameters") {
                                    Some(arguments) => arguments.to_string(),
                                    None => {
                                        let err_msg = format!(
                                            "Failed to get the arguments of the function. Tool call: {:?}",
                                            value
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        return Err(LlamaCoreError::Operation(err_msg));
                                    }
                                };

                                let function = Function { name, arguments };

                                let tool_call = ToolCall {
                                    id: "call_abc123".to_string(),
                                    ty: "function".to_string(),
                                    function,
                                };

                                tool_calls.push(tool_call);
                            }
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    } else {
                        content.push_str(block);
                        content.push('\n');
                    }
                }
            }

            let parsed = match content.is_empty() {
                true => ParseResult {
                    raw: input.to_owned(),
                    content: None,
                    tool_calls,
                },
                false => ParseResult {
                    raw: input.to_owned(),
                    content: Some(content.trim().to_owned()),
                    tool_calls,
                },
            };

            #[cfg(feature = "logging")]
            info!(target: "stdout", "parsed result: {:?}", parsed);

            Ok(parsed)
        }
        PromptTemplateType::NemotronTool => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r"(?s)<toolcall>\s*(.*?)\s*</toolcall>") {
                Ok(re) => {
                    let mut values: Vec<serde_json::Value> = vec![];
                    for cap in re.captures_iter(input) {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", &cap[0]);

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "extracted: {}", &cap[1]);

                        let matched = cap[1].trim();

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "captured: {}", matched);

                        match serde_json::from_str::<serde_json::Value>(matched) {
                            Ok(value) => values.push(value),
                            Err(e) => {
                                let err_msg = format!(
                                    "Failed to deserialize generated tool calls. Reason: {}",
                                    e
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }

                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for value in values.iter() {
                        let name = match value.get("name") {
                            Some(name) => name.to_string().replace("\"", ""),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the name of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let arguments = match value.get("arguments") {
                            Some(arguments) => arguments.to_string(),
                            None => {
                                let err_msg = format!(
                                    "Failed to get the arguments of the function. Tool call: {:?}",
                                    value
                                );

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        };

                        let function = Function { name, arguments };

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function,
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let err_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg))
                }
            }
        }
        PromptTemplateType::FunctionaryV32 => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r">>>\s*(\w+)\s*\{(.*)\}<\|eot_id\|>") {
                Ok(re) => {
                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for cap in re.captures_iter(input) {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "func_name: {}", &cap[1]);

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "arguments: {}", &cap[2]);

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function: Function {
                                name: cap[1].to_string(),
                                arguments: cap[2].to_string(),
                            },
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let warn_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    warn!(target: "stdout", "{}", &warn_msg);

                    Ok(ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls: vec![],
                    })
                }
            }
        }
        PromptTemplateType::FunctionaryV31 => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "raw input: {}", input);

            match regex::Regex::new(r"<function=(\w+)>\s*(\{.*?\})</function>") {
                Ok(re) => {
                    let mut tool_calls: Vec<ToolCall> = vec![];
                    for cap in re.captures_iter(input) {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "func_name: {}", &cap[1]);

                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "arguments: {}", &cap[2]);

                        let tool_call = ToolCall {
                            id: "call_abc123".to_string(),
                            ty: "function".to_string(),
                            function: Function {
                                name: cap[1].to_string(),
                                arguments: cap[2].to_string(),
                            },
                        };

                        tool_calls.push(tool_call);
                    }

                    let parsed = ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls,
                    };

                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "parsed result: {:?}", parsed);

                    Ok(parsed)
                }
                Err(e) => {
                    let warn_msg = format!("Failed to create a regex pattern. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    warn!(target: "stdout", "{}", &warn_msg);

                    Ok(ParseResult {
                        raw: input.to_owned(),
                        content: None,
                        tool_calls: vec![],
                    })
                }
            }
        }
        _ => {
            let err_msg = format!(
                "The tool use is only supported for prompt templates: {}, {}, {}, {}, {}, {}, and {}.",
                PromptTemplateType::MistralTool,
                PromptTemplateType::ChatMLTool,
                PromptTemplateType::GroqLlama3Tool,
                PromptTemplateType::Llama3Tool,
                PromptTemplateType::InternLM2Tool,
                PromptTemplateType::NemotronTool,
                PromptTemplateType::FunctionaryV32,
            );

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            Err(LlamaCoreError::Operation(err_msg))
        }
    }
}

async fn check_model_metadata(
    chat_request: &ChatCompletionRequest,
) -> Result<GgmlMetadata, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Check model metadata.");

    let mut should_update = false;
    let mut metadata = get_model_metadata(chat_request.model.as_ref())?;

    // check if necessary to update `image`
    if let Some(ChatCompletionRequestMessage::User(user_message)) = chat_request.messages.last() {
        if let ChatCompletionUserMessageContent::Parts(parts) = user_message.content() {
            for part in parts {
                if let ContentPart::Image(image) = part {
                    let image = image.image();

                    if image.is_url() {
                        // update metadata image
                        let img = download_image(&image.url).await?;

                        metadata.image = Some(img);

                        if !should_update {
                            should_update = true;
                        }

                        // todo: now only support a single image
                        break;
                    }
                }
            }
        }
    }

    // check if necessary to update temperature
    if let Some(temp) = chat_request.temperature {
        if metadata.temperature != temp {
            // update temperature
            metadata.temperature = temp;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update top_p
    if let Some(top_p) = chat_request.top_p {
        if metadata.top_p != top_p {
            // update top_p
            metadata.top_p = top_p;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update frequency_penalty
    if let Some(frequency_penalty) = chat_request.frequency_penalty {
        if metadata.frequency_penalty != frequency_penalty {
            // update frequency_penalty
            metadata.frequency_penalty = frequency_penalty;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if necessary to update presence_penalty
    if let Some(presence_penalty) = chat_request.presence_penalty {
        if metadata.presence_penalty != presence_penalty {
            // update presence_penalty
            metadata.presence_penalty = presence_penalty;

            if !should_update {
                should_update = true;
            }
        }
    }

    // check if the `embedding` option is disabled
    if metadata.embeddings {
        metadata.embeddings = false;

        if !should_update {
            should_update = true;
        }
    }

    if metadata.parallel_tool_calls {
        metadata.parallel_tool_calls = false;

        if !should_update {
            should_update = true;
        }
    }

    if should_update {
        // update the target graph with the new metadata
        update_model_metadata(chat_request.model.as_ref(), &metadata)?;
    }

    Ok(metadata)
}

async fn update_n_predict(
    chat_request: &ChatCompletionRequest,
    metadata: &mut GgmlMetadata,
    available_completion_tokens: u64,
) -> Result<(), LlamaCoreError> {
    let mut should_update = false;

    // check if necessary to update n_predict with max_tokens
    if let Some(max_tokens) = chat_request.max_tokens {
        #[cfg(feature = "logging")]
        info!(target: "stdout", "available_completion_tokens: {}, max_tokens from request: {}, n_predict: {}", available_completion_tokens, max_tokens, metadata.n_predict);

        let max_completion_tokens = match available_completion_tokens < max_tokens {
            true => available_completion_tokens,
            false => max_tokens,
        };

        // update n_predict
        if metadata.n_predict != max_completion_tokens {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "update n_predict from {} to {}", metadata.n_predict, max_completion_tokens);

            metadata.n_predict = max_completion_tokens;

            if !should_update {
                should_update = true;
            }
        }
        if metadata.n_predict < available_completion_tokens {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Update n_predict from {} to {}", metadata.n_predict, available_completion_tokens);

            // update n_predict
            metadata.n_predict = available_completion_tokens;

            if !should_update {
                should_update = true;
            }
        }
    }

    if should_update {
        // update the target graph with the new metadata
        update_model_metadata(chat_request.model.as_ref(), metadata)?;
    }

    Ok(())
}

fn post_process(
    output: impl AsRef<str>,
    template_ty: &PromptTemplateType,
) -> Result<String, String> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Post-process the generated output.");

    let output = if *template_ty == PromptTemplateType::Baichuan2 {
        if output.as_ref().contains("用户:") {
            output.as_ref().trim_end_matches("用户:").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::OpenChat {
        if output.as_ref().contains("<|end_of_turn|>") {
            output
                .as_ref()
                .trim_end_matches("<|end_of_turn|>")
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::GemmaInstruct {
        let s = output.as_ref().trim();
        if s.ends_with("<end_of_turn>") {
            s.trim_end_matches("<end_of_turn>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::ChatML
        || *template_ty == PromptTemplateType::ChatMLTool
        || *template_ty == PromptTemplateType::InternLM2Tool
    {
        if output.as_ref().contains("<|im_start|>") && output.as_ref().contains("<|im_end|>") {
            let idx_start = output.as_ref().find("<|im_start|>").unwrap();
            let idx_end = output.as_ref().find("<|im_end|>").unwrap();

            match idx_start <= idx_end {
                true => output.as_ref().split("<|im_start|>").collect::<Vec<_>>()[0]
                    .trim()
                    .to_owned(),
                false => output.as_ref().split("<|im_end|>").collect::<Vec<_>>()[0]
                    .trim()
                    .to_owned(),
            }
        } else if output.as_ref().contains("<|im_start|>") {
            output.as_ref().split("<|im_start|>").collect::<Vec<_>>()[0]
                .trim()
                .to_owned()
        } else if output.as_ref().contains("<|im_end|>") {
            let output = output.as_ref().trim_end_matches("<|im_end|>").trim();
            if output.starts_with(": ") {
                output.trim_start_matches(": ").to_owned()
            } else {
                output.to_owned()
            }
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::Zephyr
        || *template_ty == PromptTemplateType::MistralLite
        || *template_ty == PromptTemplateType::MistralTool
        || *template_ty == PromptTemplateType::MistralInstruct
        || *template_ty == PromptTemplateType::BreezeInstruct
    {
        if output.as_ref().contains("</s><") {
            output.as_ref().trim_end_matches("</s><").trim().to_owned()
        } else if output.as_ref().contains("</s>") {
            output
                .as_ref()
                .strip_suffix("</s>")
                .unwrap()
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::DeepseekChat {
        if output.as_ref().contains("<|end_of_sentence|>") {
            output
                .as_ref()
                .trim_end_matches("<|end_of_sentence|>")
                .trim()
                .replace("<|end_of_sentence|>", " ")
                .trim()
                .to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::HumanAssistant {
        if output.as_ref().contains("Human:") {
            output.as_ref().trim_end_matches("Human:").trim().to_owned()
        } else {
            output.as_ref().trim().to_owned()
        }
    } else if *template_ty == PromptTemplateType::SolarInstruct {
        let s = output.as_ref().trim();

        if s.starts_with("### Answer") {
            let s = s.trim_start_matches("###").trim();

            if s.starts_with("Answer:\n") {
                s.replace("Answer:\n", "Answer: ")
            } else {
                s.to_owned()
            }
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Llama2Chat {
        let s = output.as_ref().trim();
        if s.ends_with("</s>") {
            s.trim_end_matches("</s>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Llama3Chat
        || *template_ty == PromptTemplateType::GroqLlama3Tool
        || *template_ty == PromptTemplateType::Llama3Tool
        || *template_ty == PromptTemplateType::FunctionaryV32
    {
        let s = output.as_ref().trim();
        if s.ends_with("<|eot_id|>") {
            s.trim_end_matches("<|eot_id|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::Phi3Chat {
        let s = output.as_ref().trim();
        if s.ends_with("<|end|>") {
            s.trim_end_matches("<|end|>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::NemotronTool
        || *template_ty == PromptTemplateType::NemotronChat
    {
        let s = output.as_ref().trim();
        if s.ends_with("</s>") {
            s.trim_end_matches("</s>").trim().to_owned()
        } else {
            s.to_owned()
        }
    } else if *template_ty == PromptTemplateType::FunctionaryV31 {
        let mut s = output.as_ref().trim();
        if s.ends_with("<|eot_id|>") {
            s = s.trim_end_matches("<|eot_id|>").trim();
        }
        if s.ends_with("<|eom_id|>") {
            s = s.trim_end_matches("<|eom_id|>").trim();
        }
        s.to_owned()
    } else {
        output.as_ref().trim().to_owned()
    };

    Ok(output)
}

fn build_prompt(
    model_name: Option<&String>,
    chat_request: &mut ChatCompletionRequest,
) -> Result<(String, u64, bool), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Build the chat prompt from the chat messages.");

    let metadata = get_model_metadata(model_name)?;
    let ctx_size = metadata.ctx_size as u64;
    let chat_prompt = ChatPrompt::from(metadata.prompt_template);

    // compute max prompt tokens, which is 80% of the context size
    let max_prompt_tokens = ctx_size * 4 / 5;

    loop {
        // ! DO NOT REMOVE
        // build prompt
        // let prompt = match chat_prompt.build(&mut chat_request.messages) {
        //     Ok(prompt) => prompt,
        //     Err(e) => {
        //         let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

        //         #[cfg(feature = "logging")]
        //         error!(target: "stdout", "{}", &err_msg);

        //         return Err(LlamaCoreError::Operation(err_msg));
        //     }
        // };

        if chat_request.messages.is_empty() {
            let err_msg = "The messages in the chat request are empty.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.to_owned()));
        }

        let (prompt, tool_use) = match chat_request.tool_choice.as_ref() {
            Some(tool_choice) => match tool_choice {
                ToolChoice::None => {
                    match chat_prompt.build_with_tools(&mut chat_request.messages, Some(&[])) {
                        Ok(prompt) => (prompt, false),
                        Err(e) => {
                            let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    }
                }
                _ => match chat_request.tools.as_ref() {
                    Some(tools) => match chat_prompt
                        .build_with_tools(&mut chat_request.messages, Some(tools.as_slice()))
                    {
                        Ok(prompt) => (prompt, true),
                        Err(e) => {
                            let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        }
                    },
                    None => {
                        #[cfg(feature = "logging")]
                        warn!(target: "stdout", "The tool choice without tools is not supported.");

                        match chat_prompt.build_with_tools(&mut chat_request.messages, None) {
                            Ok(prompt) => (prompt, false),
                            Err(e) => {
                                let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                return Err(LlamaCoreError::Operation(err_msg));
                            }
                        }
                    }
                },
            },
            None => match chat_prompt.build_with_tools(&mut chat_request.messages, None) {
                Ok(prompt) => (prompt, false),
                Err(e) => {
                    let err_msg = format!("Fail to build chat prompts. Reason: {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            },
        };

        // set prompt
        set_prompt(model_name, &prompt)?;

        // Retrieve the number of prompt tokens.
        let token_info = get_token_info_by_graph_name(model_name)?;

        match token_info.prompt_tokens > max_prompt_tokens {
            true => {
                match chat_request.messages[0].role() {
                    ChatCompletionRole::System => {
                        if chat_request.messages.len() >= 3 {
                            // remove user_1 if it exists
                            // For example, `system -> user_1 -> ... -> user_2 -> ... -> user_latest` will be converted to `system -> ... -> user_2 -> ... -> user_latest`
                            if chat_request.messages[1].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(1);
                            }

                            // remove all messages until the message is of `user`
                            // For example, `system -> ... -> user_2 -> ... -> user_latest` will be converted to `system -> user_2 -> ... -> user_latest`
                            while chat_request.messages[1].role() != ChatCompletionRole::User {
                                chat_request.messages.remove(1);
                            }
                        } else if token_info.prompt_tokens > ctx_size {
                            let err_msg = format!(
                                    "The number of prompt tokens is greater than the context size: {} > {}",
                                    token_info.prompt_tokens, ctx_size
                                );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        } else {
                            return Ok((prompt, ctx_size - token_info.prompt_tokens, tool_use));
                        }
                    }
                    ChatCompletionRole::User => {
                        if chat_request.messages.len() >= 3 {
                            // user_1 -> ... -> user_2 -> ... -> user_latest

                            // remove user_1 if it exists
                            // For example, `user_1 -> ... -> user_2 -> ... -> user_latest` will be converted to `... -> user_2 -> ... -> user_latest`
                            if chat_request.messages[0].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(0);
                            }

                            // remove all messages until the message is of `user`
                            // For example, `... -> user_2 -> ... -> user_latest` will be converted to `user_2 -> ... -> user_latest`
                            while chat_request.messages[1].role() != ChatCompletionRole::User {
                                chat_request.messages.remove(1);
                            }
                        } else if chat_request.messages.len() == 2
                            && chat_request.messages[0].role() == ChatCompletionRole::User
                        {
                            // deal with "user_1 -> user_latest"
                            chat_request.messages.remove(0);
                        } else if token_info.prompt_tokens > ctx_size {
                            let err_msg = format!(
                                    "The number of prompt tokens is greater than the context size: {} > {}",
                                    token_info.prompt_tokens, ctx_size
                                );

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            return Err(LlamaCoreError::Operation(err_msg));
                        } else {
                            return Ok((prompt, ctx_size - token_info.prompt_tokens, tool_use));
                        }
                    }
                    _ => {
                        #[cfg(feature = "logging")]
                        info!(target: "stdout", "remove a {} message from the message queue", chat_request.messages[0].role());

                        chat_request.messages.remove(0);
                    }
                }

                continue;
            }
            false => return Ok((prompt, ctx_size - max_prompt_tokens, tool_use)),
        }
    }
}

/// Downloads an image from the given URL and returns the file name.
async fn download_image(image_url: impl AsRef<str>) -> Result<String, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Download image from the URL.");

    let image_url = image_url.as_ref();
    let url = reqwest::Url::parse(image_url).map_err(|e| {
        let err_msg = format!("Fail to parse the image URL: {}. Reason: {}", image_url, e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let response = reqwest::get(url).await.map_err(|e| {
        let err_msg = format!(
            "Fail to download the image from the URL: {}. Reason: {}",
            image_url, e
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let fname = response
        .url()
        .path_segments()
        .and_then(|segments| segments.last())
        .and_then(|name| if name.is_empty() { None } else { Some(name) })
        .ok_or(LlamaCoreError::Operation(format!(
            "Fail to get the file name: {}",
            image_url
        )))?
        .to_string();

    let mut dest = std::fs::File::create(&fname).map_err(|e| {
        let err_msg = format!(
            "Fail to create the file to save the image: {}. Reason: {}",
            &fname, e
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let mut content = response.bytes_stream();
    while let Some(Ok(item)) = content.next().await {
        std::io::copy(&mut item.as_ref(), &mut dest).map_err(|e| {
            let err_msg = format!(
                "Fail to write the image content to the file: {}. Reason: {}",
                &fname, e
            );

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The image is downloaded successfully.");

    Ok(fname)
}

fn set_prompt(model_name: Option<&String>, prompt: impl AsRef<str>) -> Result<(), LlamaCoreError> {
    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Set prompt to the chat model named {}.", model_name);

            match chat_graphs.contains_key(model_name) {
                true => {
                    let graph = chat_graphs.get_mut(model_name).unwrap();
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                false => match chat_graphs.iter_mut().next() {
                    Some((_, graph)) => {
                        let tensor_data = prompt.as_ref().as_bytes().to_vec();
                        set_tensor_data_u8(graph, 0, &tensor_data)
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
        None => {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Set prompt to the default chat model.");

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs while trying to set prompt to the default model.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}

// fn set_tensor_data_u8(
//     graph: &mut Graph,
//     idx: usize,
//     tensor_data: &[u8],
// ) -> Result<(), LlamaCoreError> {
//     if graph
//         .set_input(idx, wasmedge_wasi_nn::TensorType::U8, &[1], tensor_data)
//         .is_err()
//     {
//         return Err(LlamaCoreError::Operation(String::from(
//             "Fail to set input tensor",
//         )));
//     };

//     Ok(())
// }

/// Get a copy of the metadata of the model.
fn get_model_metadata(model_name: Option<&String>) -> Result<GgmlMetadata, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the model metadata.");

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

    match model_name {
        Some(model_name) => match chat_graphs.contains_key(model_name) {
            true => {
                let graph = chat_graphs.get(model_name).unwrap();
                Ok(graph.metadata.clone())
            }
            false => match chat_graphs.iter().next() {
                Some((_, graph)) => Ok(graph.metadata.clone()),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match chat_graphs.iter().next() {
            Some((_, graph)) => Ok(graph.metadata.clone()),
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

fn update_model_metadata(
    model_name: Option<&String>,
    metadata: &GgmlMetadata,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Update the model metadata.");

    let config = match serde_json::to_string(metadata) {
        Ok(config) => config,
        Err(e) => {
            let err_msg = format!("Fail to serialize metadata to a JSON string. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg));
        }
    };

    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. Reason: {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => {
            match chat_graphs.contains_key(model_name) {
                true => {
                    let graph = chat_graphs.get_mut(model_name).unwrap();
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                false => match chat_graphs.iter_mut().next() {
                    Some((_, graph)) => {
                        // update metadata
                        set_tensor_data_u8(graph, 1, config.as_bytes())
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
        None => {
            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    // update metadata
                    set_tensor_data_u8(graph, 1, config.as_bytes())
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextFullState {
    Message,
    Usage,
    Done,
    EndOfSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Usage,
    Done,
    EndOfSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptTooLongState {
    Message,
    Usage,
    Done,
    EndOfSequence,
}

struct ChatStream {
    id: String,
    model: Option<String>,
    include_usage: bool,
    context_full_state: ContextFullState,
    prompt_too_long_state: PromptTooLongState,
    stream_state: StreamState,
    cache: Option<VecDeque<String>>,
}
impl ChatStream {
    fn new(
        model: Option<String>,
        id: String,
        include_usage: bool,
        cache: Option<Vec<String>>,
    ) -> Self {
        let stream_state = if include_usage {
            StreamState::Usage
        } else {
            StreamState::Done
        };

        ChatStream {
            id,
            model,
            include_usage,
            context_full_state: ContextFullState::Message,
            prompt_too_long_state: PromptTooLongState::Message,
            stream_state,
            cache: cache.map(VecDeque::from),
        }
    }
}
impl Drop for ChatStream {
    fn drop(&mut self) {
        if self.cache.is_none() {
            #[cfg(feature = "logging")]
            info!(target: "stdout", "Clean up the context of the stream work environment.");

            match &self.model {
                Some(model_name) => {
                    match CHAT_GRAPHS.get() {
                        Some(chat_graphs) => match chat_graphs.lock() {
                            Ok(mut chat_graphs) => match chat_graphs.contains_key(model_name) {
                                true => {
                                    let graph = chat_graphs.get_mut(model_name).unwrap();

                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        #[cfg(not(feature = "logging"))]
                                        println!(
                                            "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                            &err_msg
                                        );
                                    }
                                }
                                false => match chat_graphs.iter_mut().next() {
                                    Some((_, graph)) => {
                                        if let Err(e) = graph.finish_single() {
                                            let err_msg = format!(
                                                "Failed to clean up the context. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            #[cfg(not(feature = "logging"))]
                                            println!(
                                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                                &err_msg
                                            );
                                        }
                                    }
                                    None => {
                                        let err_msg =
                                            "There is no model available in the chat graphs.";

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        #[cfg(not(feature = "logging"))]
                                        println!(
                                            "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                            &err_msg
                                        );
                                    }
                                },
                            },
                            Err(e) => {
                                let err_msg =
                                    format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                #[cfg(not(feature = "logging"))]
                                println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                            }
                        },
                        None => {
                            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            #[cfg(not(feature = "logging"))]
                            println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                        }
                    };
                }
                None => {
                    match CHAT_GRAPHS.get() {
                        Some(chat_graphs) => match chat_graphs.lock() {
                            Ok(mut chat_graphs) => match chat_graphs.iter_mut().next() {
                                Some((_, graph)) => {
                                    if let Err(e) = graph.finish_single() {
                                        let err_msg = format!(
                                            "Failed to clean up the context. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        #[cfg(not(feature = "logging"))]
                                        println!(
                                        "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                        &err_msg
                                    );
                                    }
                                }
                                None => {
                                    let err_msg = "There is no model available in the chat graphs.";

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", err_msg);

                                    #[cfg(not(feature = "logging"))]
                                    println!(
                                    "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                    err_msg
                                );
                                }
                            },
                            Err(e) => {
                                let err_msg =
                                    format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                #[cfg(not(feature = "logging"))]
                                println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                            }
                        },
                        None => {
                            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            #[cfg(not(feature = "logging"))]
                            println!(
                                "[ERROR][llama_core] Failed to clean up the context. Reason: {}",
                                &err_msg
                            );
                        }
                    };
                }
            }

            #[cfg(feature = "logging")]
            info!(target: "stdout", "Cleanup done!");
        }
    }
}
impl futures::Stream for ChatStream {
    type Item = Result<String, LlamaCoreError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.cache.is_none() {
            let this = self.get_mut();
            let x = compute_stream(
                this.model.clone(),
                this.id.clone(),
                this.include_usage,
                &mut this.prompt_too_long_state,
                &mut this.context_full_state,
                &mut this.stream_state,
            );

            match x {
                Ok(x) => {
                    #[cfg(feature = "logging")]
                    info!(target: "stdout", "next item: {}", &x);

                    if x != "[GGML] End of sequence" && !x.is_empty() {
                        Poll::Ready(Some(Ok(x)))
                    } else {
                        // stopped
                        Poll::Ready(None)
                    }
                }
                Err(e) => Poll::Ready(Some(Err(e))),
            }
        } else {
            let this = self.get_mut();

            let x = this.cache.as_mut().unwrap().pop_front();

            #[cfg(feature = "logging")]
            info!(target: "stdout", "Get the next item from the cache: {:?}", &x);

            match x {
                Some(x) => Poll::Ready(Some(Ok(x))),
                None => Poll::Ready(None),
            }
        }
    }
}

fn compute_stream(
    model_name: Option<String>,
    id: String,
    include_usage: bool,
    prompt_too_long_state: &mut PromptTooLongState,
    context_full_state: &mut ContextFullState,
    stream_state: &mut StreamState,
) -> Result<String, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute the chat stream chunk.");

    if *prompt_too_long_state == PromptTooLongState::EndOfSequence
        || *context_full_state == ContextFullState::EndOfSequence
        || *stream_state == StreamState::EndOfSequence
    {
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Return the chat stream chunk!");

        return Ok("[GGML] End of sequence".to_string());
    }

    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    // get graph
    let res = match &model_name {
        Some(model_name) => {
            match chat_graphs.contains_key(model_name) {
                true => {
                    let graph = chat_graphs.get_mut(model_name).unwrap();
                    // compute
                    match graph.compute_single() {
                        Ok(_) => {
                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "Compute the chat stream chunk successfully.");

                            // Retrieve the output
                            let output_buffer = get_output_buffer_single(graph, OUTPUT_TENSOR)?;

                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "retrieved the output buffer");

                            // decode the output buffer to a utf8 string
                            let output = match String::from_utf8(output_buffer.clone()) {
                                Ok(token) => token,
                                Err(_) => {
                                    let mutex = CACHED_UTF8_ENCODINGS
                                        .get_or_init(|| Mutex::new(Vec::new()));
                                    let mut cached_encodings = mutex.lock().map_err(|e| {
                                            let err_msg = format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);


                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    // cache the bytes for future decoding
                                    cached_encodings.extend_from_slice(&output_buffer[..]);

                                    match String::from_utf8(cached_encodings.to_vec()) {
                                        Ok(token) => {
                                            // clear encodings
                                            cached_encodings.clear();

                                            token
                                        }
                                        Err(_) => {
                                            // TODO This is a temp check. In case, infinite cached encodings happen.
                                            if cached_encodings.len() > 4 {
                                                let err_msg = "The length of the invalid utf8 bytes exceed 4.";

                                                #[cfg(feature = "logging")]
                                                error!(target: "stdout", "{}", &err_msg);

                                                return Err(LlamaCoreError::Operation(
                                                    err_msg.into(),
                                                ));
                                            }

                                            String::new()
                                        }
                                    }
                                }
                            };

                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "decoded the output buffer");

                            let created = SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_err(|e| {
                                let err_msg =
                                    format!("Failed to get the current time. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                LlamaCoreError::Operation(err_msg)
                            })?;

                            let chat_completion_chunk = ChatCompletionChunk {
                                id,
                                object: "chat.completion.chunk".to_string(),
                                created: created.as_secs(),
                                model: graph.name().to_owned(),
                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                choices: vec![ChatCompletionChunkChoice {
                                    index: 0,
                                    delta: ChatCompletionChunkChoiceDelta {
                                        role: ChatCompletionRole::Assistant,
                                        content: Some(output),
                                        tool_calls: vec![],
                                    },
                                    logprobs: None,
                                    finish_reason: None,
                                }],
                                usage: None,
                            };

                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "created chat completion chunk");

                            // serialize chat completion chunk
                            let chunk_str =
                                serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                    let err_msg = format!(
                                        "Failed to serialize chat completion chunk. Reason: {}",
                                        e
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    LlamaCoreError::Operation(err_msg)
                                })?;

                            Ok(format!("data: {}\n\n", chunk_str))
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::EndOfSequence,
                        )) => {
                            match stream_state {
                                StreamState::Usage => {
                                    *stream_state = StreamState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::ContextFull,
                        )) => {
                            match context_full_state {
                                ContextFullState::Message => {
                                    match include_usage {
                                        true => *context_full_state = ContextFullState::Usage,
                                        false => *context_full_state = ContextFullState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: Some(
                                                    "<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string(),
                                                ),
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Usage => {
                                    *context_full_state = ContextFullState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Done => {
                                    *context_full_state = ContextFullState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                ContextFullState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::PromptTooLong,
                        )) => {
                            match prompt_too_long_state {
                                PromptTooLongState::Message => {
                                    match include_usage {
                                        true => *prompt_too_long_state = PromptTooLongState::Usage,
                                        false => *prompt_too_long_state = PromptTooLongState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: None,
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Usage => {
                                    *prompt_too_long_state = PromptTooLongState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Done => {
                                    *prompt_too_long_state = PromptTooLongState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                PromptTooLongState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(e) => {
                            let err_msg =
                                format!("Failed to compute the chat completion. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                err_msg,
                            )))
                        }
                    }
                }
                false => {
                    match chat_graphs.iter_mut().next() {
                        Some((_, graph)) => {
                            // compute
                            match graph.compute_single() {
                                Ok(_) => {
                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "Compute the chat stream chunk successfully.");

                                    // Retrieve the output
                                    let output_buffer =
                                        get_output_buffer_single(graph, OUTPUT_TENSOR)?;

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "retrieved the output buffer");

                                    // decode the output buffer to a utf8 string
                                    let output = match String::from_utf8(output_buffer.clone()) {
                                        Ok(token) => token,
                                        Err(_) => {
                                            let mutex = CACHED_UTF8_ENCODINGS
                                                .get_or_init(|| Mutex::new(Vec::new()));
                                            let mut cached_encodings = mutex.lock().map_err(|e| {
                                            let err_msg = format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);


                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                            // cache the bytes for future decoding
                                            cached_encodings.extend_from_slice(&output_buffer[..]);

                                            match String::from_utf8(cached_encodings.to_vec()) {
                                                Ok(token) => {
                                                    // clear encodings
                                                    cached_encodings.clear();

                                                    token
                                                }
                                                Err(_) => {
                                                    // TODO This is a temp check. In case, infinite cached encodings happen.
                                                    if cached_encodings.len() > 4 {
                                                        let err_msg = "The length of the invalid utf8 bytes exceed 4.";

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        return Err(LlamaCoreError::Operation(
                                                            err_msg.into(),
                                                        ));
                                                    }

                                                    String::new()
                                                }
                                            }
                                        }
                                    };

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "decoded the output buffer");

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: Some(output),
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: None,
                                        }],
                                        usage: None,
                                    };

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "created chat completion chunk");

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                Err(wasmedge_wasi_nn::Error::BackendError(
                                    wasmedge_wasi_nn::BackendError::EndOfSequence,
                                )) => {
                                    match stream_state {
                                        StreamState::Usage => {
                                            *stream_state = StreamState::Done;

                                            // retrieve the number of prompt and completion tokens
                                            let token_info = get_token_info_by_graph(graph)?;

                                            let usage = Some(Usage {
                                                prompt_tokens: token_info.prompt_tokens,
                                                completion_tokens: token_info.completion_tokens,
                                                total_tokens: token_info.prompt_tokens
                                                    + token_info.completion_tokens,
                                            });

                                            #[cfg(feature = "logging")]
                                            info!(target: "stdout", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![],
                                                usage,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        StreamState::Done => {
                                            *stream_state = StreamState::EndOfSequence;

                                            Ok("data: [DONE]\n\n".to_string())
                                        }
                                        StreamState::EndOfSequence => {
                                            Ok("[GGML] End of sequence".to_string())
                                        }
                                    }
                                }
                                Err(wasmedge_wasi_nn::Error::BackendError(
                                    wasmedge_wasi_nn::BackendError::ContextFull,
                                )) => {
                                    match context_full_state {
                                        ContextFullState::Message => {
                                            match include_usage {
                                                true => {
                                                    *context_full_state = ContextFullState::Usage
                                                }
                                                false => {
                                                    *context_full_state = ContextFullState::Done
                                                }
                                            }

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![ChatCompletionChunkChoice {
                                                    index: 0,
                                                    delta: ChatCompletionChunkChoiceDelta {
                                                        role: ChatCompletionRole::Assistant,
                                                        content: Some(
                                                            "<|WASMEDGE-GGML-CONTEXT-FULL|>"
                                                                .to_string(),
                                                        ),
                                                        tool_calls: vec![],
                                                    },
                                                    logprobs: None,
                                                    finish_reason: Some(FinishReason::length),
                                                }],
                                                usage: None,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        ContextFullState::Usage => {
                                            *context_full_state = ContextFullState::Done;

                                            // retrieve the number of prompt and completion tokens
                                            let token_info = get_token_info_by_graph(graph)?;

                                            let usage = Some(Usage {
                                                prompt_tokens: token_info.prompt_tokens,
                                                completion_tokens: token_info.completion_tokens,
                                                total_tokens: token_info.prompt_tokens
                                                    + token_info.completion_tokens,
                                            });

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![],
                                                usage,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        ContextFullState::Done => {
                                            *context_full_state = ContextFullState::EndOfSequence;

                                            Ok("data: [DONE]\n\n".to_string())
                                        }
                                        ContextFullState::EndOfSequence => {
                                            Ok("[GGML] End of sequence".to_string())
                                        }
                                    }
                                }
                                Err(wasmedge_wasi_nn::Error::BackendError(
                                    wasmedge_wasi_nn::BackendError::PromptTooLong,
                                )) => {
                                    match prompt_too_long_state {
                                        PromptTooLongState::Message => {
                                            match include_usage {
                                                true => {
                                                    *prompt_too_long_state =
                                                        PromptTooLongState::Usage
                                                }
                                                false => {
                                                    *prompt_too_long_state =
                                                        PromptTooLongState::Done
                                                }
                                            }

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![ChatCompletionChunkChoice {
                                                    index: 0,
                                                    delta: ChatCompletionChunkChoiceDelta {
                                                        role: ChatCompletionRole::Assistant,
                                                        content: None,
                                                        tool_calls: vec![],
                                                    },
                                                    logprobs: None,
                                                    finish_reason: Some(FinishReason::length),
                                                }],
                                                usage: None,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        PromptTooLongState::Usage => {
                                            *prompt_too_long_state = PromptTooLongState::Done;

                                            // retrieve the number of prompt and completion tokens
                                            let token_info = get_token_info_by_graph(graph)?;

                                            let usage = Some(Usage {
                                                prompt_tokens: token_info.prompt_tokens,
                                                completion_tokens: token_info.completion_tokens,
                                                total_tokens: token_info.prompt_tokens
                                                    + token_info.completion_tokens,
                                            });

                                            let created = SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .map_err(|e| {
                                                    let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                                    #[cfg(feature = "logging")]
                                                    error!(target: "stdout", "{}", &err_msg);

                                                    LlamaCoreError::Operation(err_msg)
                                                })?;

                                            let chat_completion_chunk = ChatCompletionChunk {
                                                id,
                                                object: "chat.completion.chunk".to_string(),
                                                created: created.as_secs(),
                                                model: graph.name().to_owned(),
                                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                                choices: vec![],
                                                usage,
                                            };

                                            // serialize chat completion chunk
                                            let chunk_str =
                                                serde_json::to_string(&chat_completion_chunk)
                                                    .map_err(|e| {
                                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                                        #[cfg(feature = "logging")]
                                                        error!(target: "stdout", "{}", &err_msg);

                                                        LlamaCoreError::Operation(err_msg)
                                                    })?;

                                            Ok(format!("data: {}\n\n", chunk_str))
                                        }
                                        PromptTooLongState::Done => {
                                            *prompt_too_long_state =
                                                PromptTooLongState::EndOfSequence;

                                            Ok("data: [DONE]\n\n".to_string())
                                        }
                                        PromptTooLongState::EndOfSequence => {
                                            Ok("[GGML] End of sequence".to_string())
                                        }
                                    }
                                }
                                Err(e) => {
                                    let err_msg = format!(
                                        "Failed to compute the chat completion. Reason: {}",
                                        e
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                        err_msg,
                                    )))
                                }
                            }
                        }
                        None => {
                            let err_msg = "There is no model available in the chat graphs.";

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            Err(LlamaCoreError::Operation(err_msg.into()))
                        }
                    }
                }
            }
        }
        None => {
            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    // compute
                    match graph.compute_single() {
                        Ok(_) => {
                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "Compute the chat stream chunk successfully.");

                            // Retrieve the output
                            let output_buffer = get_output_buffer_single(graph, OUTPUT_TENSOR)?;

                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "retrieved the output buffer");

                            // decode the output buffer to a utf8 string
                            let output = match String::from_utf8(output_buffer.clone()) {
                                Ok(token) => token,
                                Err(_) => {
                                    let mutex = CACHED_UTF8_ENCODINGS
                                        .get_or_init(|| Mutex::new(Vec::new()));
                                    let mut cached_encodings = mutex.lock().map_err(|e| {
                                            let err_msg = format!(
                                                "Fail to acquire the lock of `UTF8_ENCODINGS`. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    cached_encodings.extend_from_slice(&output_buffer[..]);

                                    match String::from_utf8(cached_encodings.to_vec()) {
                                        Ok(token) => {
                                            // clear encodings
                                            cached_encodings.clear();

                                            token
                                        }
                                        Err(_) => {
                                            // TODO This is a temp check. In case, infinite cached encodings happen.
                                            if cached_encodings.len() > 4 {
                                                let err_msg = "The length of the invalid utf8 bytes exceed 4.";

                                                #[cfg(feature = "logging")]
                                                error!(target: "stdout", "{}", &err_msg);

                                                return Err(LlamaCoreError::Operation(
                                                    err_msg.into(),
                                                ));
                                            }

                                            String::new()
                                        }
                                    }
                                }
                            };

                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "decoded the output buffer");

                            let created = SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_err(|e| {
                                let err_msg =
                                    format!("Failed to get the current time. Reason: {}", e);

                                #[cfg(feature = "logging")]
                                error!(target: "stdout", "{}", &err_msg);

                                LlamaCoreError::Operation(err_msg)
                            })?;

                            let chat_completion_chunk = ChatCompletionChunk {
                                id,
                                object: "chat.completion.chunk".to_string(),
                                created: created.as_secs(),
                                model: graph.name().to_owned(),
                                system_fingerprint: "fp_44709d6fcb".to_string(),
                                choices: vec![ChatCompletionChunkChoice {
                                    index: 0,
                                    delta: ChatCompletionChunkChoiceDelta {
                                        role: ChatCompletionRole::Assistant,
                                        content: Some(output),
                                        tool_calls: vec![],
                                    },
                                    logprobs: None,
                                    finish_reason: None,
                                }],
                                usage: None,
                            };

                            #[cfg(feature = "logging")]
                            info!(target: "stdout", "created chat completion chunk");

                            // serialize chat completion chunk
                            let chunk_str =
                                serde_json::to_string(&chat_completion_chunk).map_err(|e| {
                                    let err_msg = format!(
                                        "Failed to serialize chat completion chunk. Reason: {}",
                                        e
                                    );

                                    #[cfg(feature = "logging")]
                                    error!(target: "stdout", "{}", &err_msg);

                                    LlamaCoreError::Operation(err_msg)
                                })?;

                            Ok(format!("data: {}\n\n", chunk_str))
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::EndOfSequence,
                        )) => {
                            match stream_state {
                                StreamState::Usage => {
                                    *stream_state = StreamState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    #[cfg(feature = "logging")]
                                    info!(target: "stdout", "token_info: {} prompt tokens, {} completion tokens", token_info.prompt_tokens, token_info.completion_tokens);

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                StreamState::Done => {
                                    *stream_state = StreamState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                StreamState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::ContextFull,
                        )) => {
                            match context_full_state {
                                ContextFullState::Message => {
                                    match include_usage {
                                        true => *context_full_state = ContextFullState::Usage,
                                        false => *context_full_state = ContextFullState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: Some(
                                                    "<|WASMEDGE-GGML-CONTEXT-FULL|>".to_string(),
                                                ),
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Usage => {
                                    *context_full_state = ContextFullState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                ContextFullState::Done => {
                                    *context_full_state = ContextFullState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                ContextFullState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(wasmedge_wasi_nn::Error::BackendError(
                            wasmedge_wasi_nn::BackendError::PromptTooLong,
                        )) => {
                            match prompt_too_long_state {
                                PromptTooLongState::Message => {
                                    match include_usage {
                                        true => *prompt_too_long_state = PromptTooLongState::Usage,
                                        false => *prompt_too_long_state = PromptTooLongState::Done,
                                    }

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![ChatCompletionChunkChoice {
                                            index: 0,
                                            delta: ChatCompletionChunkChoiceDelta {
                                                role: ChatCompletionRole::Assistant,
                                                content: None,
                                                tool_calls: vec![],
                                            },
                                            logprobs: None,
                                            finish_reason: Some(FinishReason::length),
                                        }],
                                        usage: None,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Usage => {
                                    *prompt_too_long_state = PromptTooLongState::Done;

                                    // retrieve the number of prompt and completion tokens
                                    let token_info = get_token_info_by_graph(graph)?;

                                    let usage = Some(Usage {
                                        prompt_tokens: token_info.prompt_tokens,
                                        completion_tokens: token_info.completion_tokens,
                                        total_tokens: token_info.prompt_tokens
                                            + token_info.completion_tokens,
                                    });

                                    let created = SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map_err(|e| {
                                            let err_msg = format!(
                                                "Failed to get the current time. Reason: {}",
                                                e
                                            );

                                            #[cfg(feature = "logging")]
                                            error!(target: "stdout", "{}", &err_msg);

                                            LlamaCoreError::Operation(err_msg)
                                        })?;

                                    let chat_completion_chunk = ChatCompletionChunk {
                                        id,
                                        object: "chat.completion.chunk".to_string(),
                                        created: created.as_secs(),
                                        model: graph.name().to_owned(),
                                        system_fingerprint: "fp_44709d6fcb".to_string(),
                                        choices: vec![],
                                        usage,
                                    };

                                    // serialize chat completion chunk
                                    let chunk_str = serde_json::to_string(&chat_completion_chunk)
                                        .map_err(|e| {
                                        let err_msg = format!(
                                            "Failed to serialize chat completion chunk. Reason: {}",
                                            e
                                        );

                                        #[cfg(feature = "logging")]
                                        error!(target: "stdout", "{}", &err_msg);

                                        LlamaCoreError::Operation(err_msg)
                                    })?;

                                    Ok(format!("data: {}\n\n", chunk_str))
                                }
                                PromptTooLongState::Done => {
                                    *prompt_too_long_state = PromptTooLongState::EndOfSequence;

                                    Ok("data: [DONE]\n\n".to_string())
                                }
                                PromptTooLongState::EndOfSequence => {
                                    Ok("[GGML] End of sequence".to_string())
                                }
                            }
                        }
                        Err(e) => {
                            let err_msg =
                                format!("Failed to compute the chat completion. Reason: {}", e);

                            #[cfg(feature = "logging")]
                            error!(target: "stdout", "{}", &err_msg);

                            Err(LlamaCoreError::Backend(BackendError::ComputeSingle(
                                err_msg,
                            )))
                        }
                    }
                }
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            }
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Return the chat stream chunk!");

    res
}

#[derive(Debug)]
struct ParseResult {
    raw: String,
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
}
