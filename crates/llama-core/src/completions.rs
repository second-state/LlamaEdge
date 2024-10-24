//! Define APIs for completions.

use crate::{
    error::{BackendError, LlamaCoreError},
    metadata::ggml::GgmlMetadata,
    running_mode,
    utils::{get_output_buffer, get_token_info_by_graph},
    Graph, RunningMode, CHAT_GRAPHS, OUTPUT_TENSOR,
};
use endpoints::{
    common::{FinishReason, Usage},
    completions::{CompletionChoice, CompletionObject, CompletionPrompt, CompletionRequest},
};
use std::time::SystemTime;

/// Given a prompt, the model will return one or more predicted completions along with the probabilities of alternative tokens at each position.
pub async fn completions(request: &CompletionRequest) -> Result<CompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Generate completions");

    let running_mode = running_mode()?;
    if running_mode == RunningMode::Embeddings || running_mode == RunningMode::Rag {
        let err_msg = format!(
            "The completion is not supported in the {} mode.",
            running_mode
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let prompt = match &request.prompt {
        CompletionPrompt::SingleText(prompt) => prompt.to_owned(),
        CompletionPrompt::MultiText(prompts) => prompts.join(" "),
    };

    compute(prompt.trim(), request.model.as_ref())
}

fn compute(
    prompt: impl AsRef<str>,
    model_name: Option<&String>,
) -> std::result::Result<CompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute completions");

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
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match model_name {
        Some(model_name) => match chat_graphs.contains_key(model_name) {
            true => {
                let graph = chat_graphs.get_mut(model_name).unwrap();
                compute_by_graph(graph, prompt)
            }
            false => match chat_graphs.iter_mut().next() {
                Some((_, graph)) => compute_by_graph(graph, prompt),
                None => {
                    let err_msg = "There is no model available in the chat graphs.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    Err(LlamaCoreError::Operation(err_msg.into()))
                }
            },
        },
        None => match chat_graphs.iter_mut().next() {
            Some((_, graph)) => compute_by_graph(graph, prompt),
            None => {
                let err_msg = "There is no model available in the chat graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg.into()))
            }
        },
    }
}

/// Runs inference on the model with the given name and returns the output.
fn compute_by_graph(
    graph: &mut Graph<GgmlMetadata>,
    prompt: impl AsRef<str>,
) -> std::result::Result<CompletionObject, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Compute completions by graph");

    // check if the `embedding` model is disabled or not
    if graph.metadata.embeddings {
        graph.metadata.embeddings = false;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The `embedding` field of metadata sets to false.");

        graph.update_metadata()?;
    }

    // set input
    let tensor_data = prompt.as_ref().as_bytes().to_vec();
    graph
        .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
        .map_err(|e| {
            let err_msg = format!("Failed to set the input tensor. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Backend(BackendError::SetInput(err_msg))
        })?;

    // execute the inference
    graph.compute().map_err(|e| {
        let err_msg = format!("Failed to execute the inference. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Backend(BackendError::Compute(err_msg))
    })?;

    // Retrieve the output
    let buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;

    // convert inference result to string
    let model_answer = String::from_utf8(buffer).map_err(|e| {
        let err_msg = format!(
            "Failed to decode the buffer of the inference result to a utf-8 string. {}",
            e
        );

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;
    let answer = model_answer.trim();

    // retrieve the number of prompt and completion tokens
    let token_info = get_token_info_by_graph(graph)?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Prompt tokens: {}, Completion tokens: {}", token_info.prompt_tokens, token_info.completion_tokens);

    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| {
            let err_msg = format!("Failed to get the current time. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Completions generated successfully.");

    Ok(CompletionObject {
        id: uuid::Uuid::new_v4().to_string(),
        object: String::from("text_completion"),
        created: created.as_secs(),
        model: graph.name().to_string(),
        choices: vec![CompletionChoice {
            index: 0,
            text: String::from(answer),
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
