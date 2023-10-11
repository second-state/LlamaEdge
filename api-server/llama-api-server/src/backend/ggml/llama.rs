use crate::error;
use hyper::{body::to_bytes, Body, Request, Response};
use prompt::{BuildPrompt, PromptTemplateType};
use xin::{
    chat::{
        ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseMessage,
        ChatCompletionRole, FinishReason,
    },
    common::Usage,
    models::{ListModelsResponse, Model},
};

/// Lists models available
pub(crate) async fn llama_models_handler(created: u64) -> Result<Response<Body>, hyper::Error> {
    let llama_2_7b_chat_q5_k_m = Model {
        id: String::from("llama-2-7b-chat.Q5_K_M.gguf"),
        created: created.clone(),
        object: String::from("model"),
        owned_by: String::from("https://huggingface.co/TheBloke"),
    };

    let codellama_13b_instruct_q4_0 = Model {
        id: String::from("codellama-13b-instruct.Q4_0.gguf"),
        created: created.clone(),
        object: String::from("model"),
        owned_by: String::from("https://huggingface.co/TheBloke"),
    };

    let mistral_7b_instruct_v0_1 = Model {
        id: String::from("Mistral-7B-Instruct-v0.1.gguf"),
        created: created.clone(),
        object: String::from("model"),
        owned_by: String::from("https://huggingface.co/TheBloke"),
    };

    let list_models_response = ListModelsResponse {
        object: String::from("list"),
        data: vec![
            llama_2_7b_chat_q5_k_m,
            codellama_13b_instruct_q4_0,
            mistral_7b_instruct_v0_1,
        ],
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .body(Body::from(
            serde_json::to_string(&list_models_response).unwrap(),
        ));
    match result {
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

pub(crate) async fn _llama_embeddings_handler() -> Result<Response<Body>, hyper::Error> {
    println!("llama_embeddings_handler not implemented");
    error::not_implemented()
}

pub(crate) async fn _llama_completions_handler() -> Result<Response<Body>, hyper::Error> {
    println!("llama_completions_handler not implemented");
    error::not_implemented()
}

/// Processes a chat-completion request and returns a chat-completion response with the answer from the model.
pub(crate) async fn llama_chat_completions_handler(
    mut req: Request<Body>,
    model_name: impl AsRef<str>,
    template_ty: PromptTemplateType,
) -> Result<Response<Body>, hyper::Error> {
    if req.method().eq(&hyper::http::Method::OPTIONS) {
        println!("[CHAT] Empty in, empty out!");

        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .body(Body::empty());

        match result {
            Ok(response) => return Ok(response),
            Err(e) => {
                return error::internal_server_error(e.to_string());
            }
        }
    }

    fn create_prompt_template(template_ty: PromptTemplateType) -> Box<dyn BuildPrompt> {
        match template_ty {
            PromptTemplateType::Llama2Chat => Box::new(prompt::llama::Llama2ChatPrompt::default()),
            PromptTemplateType::MistralInstructV01 => {
                Box::new(prompt::mistral::MistralInstructPrompt::default())
            }
            PromptTemplateType::CodeLlama => {
                Box::new(prompt::llama::CodeLlamaInstructPrompt::default())
            }
        }
    }
    let template = create_prompt_template(template_ty);

    println!("[CHAT] New chat begins ...");

    // parse request
    let body_bytes = to_bytes(req.body_mut()).await?;
    let mut chat_request: xin::chat::ChatCompletionRequest =
        serde_json::from_slice(&body_bytes).unwrap();

    // build prompt
    let prompt = match template.build(chat_request.messages.as_mut()) {
        Ok(prompt) => prompt,
        Err(e) => {
            return error::internal_server_error(e.to_string());
        }
    };

    // run inference
    let buffer = infer(model_name.as_ref(), prompt.trim()).await;

    // convert inference result to string
    let model_answer = String::from_utf8(buffer.clone()).unwrap();
    let assistant_message = model_answer.trim();

    println!("[CHAT] Bot answer: {}", assistant_message);

    println!("[CHAT] New chat ends.");

    // create ChatCompletionResponse
    let chat_completion_obejct = ChatCompletionResponse {
        id: String::new(),
        object: String::from("chat.completion"),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: chat_request.model.clone(),
        choices: vec![ChatCompletionResponseChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: ChatCompletionRole::Assistant,
                content: String::from(assistant_message),
                function_call: None,
            },
            finish_reason: FinishReason::stop,
        }],
        usage: Usage {
            prompt_tokens: 9,
            completion_tokens: 12,
            total_tokens: 21,
        },
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .body(Body::from(
            serde_json::to_string(&chat_completion_obejct).unwrap(),
        ));
    match result {
        Ok(response) => Ok(response),
        Err(e) => error::internal_server_error(e.to_string()),
    }
}

/// Runs inference on the model with the given name and returns the output.
pub(crate) async fn infer(model_name: impl AsRef<str>, prompt: impl AsRef<str>) -> Vec<u8> {
    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::CPU)
            .build_from_cache(model_name.as_ref())
            .unwrap();
    // println!("Loaded model into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    // println!("Created wasi-nn execution context with ID: {:?}", context);

    let tensor_data = prompt.as_ref().trim().as_bytes().to_vec();
    // println!("Read input tensor, size in bytes: {}", tensor_data.len());
    context
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .unwrap();

    // Execute the inference.
    context.compute().unwrap();
    // println!("Executed model inference");

    // Retrieve the output.
    let mut output_buffer = vec![0u8; 2048];
    let size = context.get_output(0, &mut output_buffer).unwrap();
    output_buffer[..size].to_vec()
}
