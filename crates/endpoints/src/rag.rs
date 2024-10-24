//! Define types for the `rag` endpoint.

use crate::{
    chat::{
        ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRequestSampling,
        ChatResponseFormat, StreamOptions, Tool, ToolChoice,
    },
    embeddings::EmbeddingRequest,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEmbeddingRequest {
    #[serde(rename = "embeddings")]
    pub embedding_request: EmbeddingRequest,
    #[serde(rename = "url")]
    pub qdrant_url: String,
    #[serde(rename = "collection_name")]
    pub qdrant_collection_name: String,
}
impl RagEmbeddingRequest {
    pub fn new(
        input: &[String],
        qdrant_url: impl AsRef<str>,
        qdrant_collection_name: impl AsRef<str>,
    ) -> Self {
        RagEmbeddingRequest {
            embedding_request: EmbeddingRequest {
                model: "dummy-embedding-model".to_string(),
                input: input.into(),
                encoding_format: None,
                user: None,
            },
            qdrant_url: qdrant_url.as_ref().to_string(),
            qdrant_collection_name: qdrant_collection_name.as_ref().to_string(),
        }
    }

    pub fn from_embedding_request(
        embedding_request: EmbeddingRequest,
        qdrant_url: impl AsRef<str>,
        qdrant_collection_name: impl AsRef<str>,
    ) -> Self {
        RagEmbeddingRequest {
            embedding_request,
            qdrant_url: qdrant_url.as_ref().to_string(),
            qdrant_collection_name: qdrant_collection_name.as_ref().to_string(),
        }
    }
}

#[test]
fn test_rag_serialize_embedding_request() {
    let embedding_request = EmbeddingRequest {
        model: "model".to_string(),
        input: "Hello, world!".into(),
        encoding_format: None,
        user: None,
    };
    let qdrant_url = "http://localhost:6333".to_string();
    let qdrant_collection_name = "qdrant_collection_name".to_string();
    let rag_embedding_request = RagEmbeddingRequest {
        embedding_request,
        qdrant_url,
        qdrant_collection_name,
    };
    let json = serde_json::to_string(&rag_embedding_request).unwrap();
    assert_eq!(
        json,
        r#"{"embeddings":{"model":"model","input":"Hello, world!"},"url":"http://localhost:6333","collection_name":"qdrant_collection_name"}"#
    );
}

#[test]
fn test_rag_deserialize_embedding_request() {
    let json = r#"{"embeddings":{"model":"model","input":["Hello, world!"]},"url":"http://localhost:6333","collection_name":"qdrant_collection_name"}"#;
    let rag_embedding_request: RagEmbeddingRequest = serde_json::from_str(json).unwrap();
    assert_eq!(rag_embedding_request.qdrant_url, "http://localhost:6333");
    assert_eq!(
        rag_embedding_request.qdrant_collection_name,
        "qdrant_collection_name"
    );
    assert_eq!(rag_embedding_request.embedding_request.model, "model");
    assert_eq!(
        rag_embedding_request.embedding_request.input,
        vec!["Hello, world!"].into()
    );
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct RagChatCompletionsRequest {
    /// The model to use for generating completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_model: Option<String>,
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionRequestMessage>,
    /// ID of the embedding model to use.
    pub embedding_model: String,
    /// The format to return the embeddings in. Can be either float or base64.
    /// Defaults to float.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// The URL of the Qdrant server.
    pub qdrant_url: String,
    /// The name of the collection in Qdrant.
    pub qdrant_collection_name: String,
    /// Max number of retrieved results.
    pub limit: u64,
    /// Adjust the randomness of the generated text. Between 0.0 and 2.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or top_p but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. The value should be between 0.0 and 1.0.
    ///
    /// Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.
    ///
    /// We generally recommend altering this or temperature but not both.
    /// Defaults to 1.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// How many chat completion choices to generate for each input message.
    /// Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_choice: Option<u64>,
    /// [parallel function calling](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling during tool use.
    /// Default to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Whether to stream the results as they are generated. Useful for chatbots.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Options for streaming response. Only set this when you set `stream: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// A list of tokens at which to stop generation. If None, no stop tokens are used. Up to 4 sequences where the API will stop generating further tokens.
    /// Defaults to None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// The maximum number of tokens to generate. The value should be no less than 1.
    /// Defaults to 16.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    /// Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    /// Defaults to None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f64>>,
    /// A unique identifier representing your end-user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Format that the model must output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ChatResponseFormat>,
    /// A list of tools the model may call.
    ///
    /// Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.
    pub tools: Option<Vec<Tool>>,
    /// Controls which (if any) function is called by the model.
    pub tool_choice: Option<ToolChoice>,
}
impl RagChatCompletionsRequest {
    pub fn as_chat_completions_request(&self) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.chat_model.clone(),
            messages: self.messages.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            n_choice: self.n_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            stream: self.stream,
            stream_options: self.stream_options.clone(),
            stop: self.stop.clone(),
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            logit_bias: self.logit_bias.clone(),
            user: self.user.clone(),
            functions: None,
            function_call: None,
            response_format: self.response_format.clone(),
            tool_choice: self.tool_choice.clone(),
            tools: self.tools.clone(),
        }
    }

    pub fn from_chat_completions_request(
        chat_completions_request: ChatCompletionRequest,
        qdrant_url: impl Into<String>,
        qdrant_collection_name: impl Into<String>,
        limit: u64,
    ) -> Self {
        RagChatCompletionsRequest {
            chat_model: chat_completions_request.model,
            messages: chat_completions_request.messages,
            embedding_model: "dummy-embedding-model".to_string(),
            encoding_format: None,
            qdrant_url: qdrant_url.into(),
            qdrant_collection_name: qdrant_collection_name.into(),
            limit,
            temperature: chat_completions_request.temperature,
            top_p: chat_completions_request.top_p,
            n_choice: chat_completions_request.n_choice,
            parallel_tool_calls: chat_completions_request.parallel_tool_calls,
            stream: chat_completions_request.stream,
            stream_options: chat_completions_request.stream_options,
            stop: chat_completions_request.stop,
            max_tokens: chat_completions_request.max_tokens,
            presence_penalty: chat_completions_request.presence_penalty,
            frequency_penalty: chat_completions_request.frequency_penalty,
            logit_bias: chat_completions_request.logit_bias,
            user: chat_completions_request.user,
            response_format: chat_completions_request.response_format,
            tool_choice: chat_completions_request.tool_choice,
            tools: chat_completions_request.tools,
        }
    }
}

/// Request builder for creating a new RAG chat completion request.
pub struct RagChatCompletionRequestBuilder {
    req: RagChatCompletionsRequest,
}
impl RagChatCompletionRequestBuilder {
    /// Creates a new builder with the given model.
    ///
    /// # Arguments
    ///
    /// * `model` - ID of the model to use.
    ///
    /// * `messages` - A list of messages comprising the conversation so far.
    ///
    /// * `sampling` - The sampling method to use.
    pub fn new(
        messages: Vec<ChatCompletionRequestMessage>,
        qdrant_url: impl Into<String>,
        qdrant_collection_name: impl Into<String>,
        limit: u64,
    ) -> Self {
        Self {
            req: RagChatCompletionsRequest {
                chat_model: Some("dummy-chat-model".to_string()),
                messages,
                embedding_model: "dummy-embedding-model".to_string(),
                encoding_format: None,
                qdrant_url: qdrant_url.into(),
                qdrant_collection_name: qdrant_collection_name.into(),
                limit,
                temperature: None,
                top_p: None,
                n_choice: None,
                parallel_tool_calls: None,
                stream: None,
                stream_options: None,
                stop: None,
                max_tokens: None,
                presence_penalty: None,
                frequency_penalty: None,
                logit_bias: None,
                user: None,
                response_format: None,
                tool_choice: None,
                tools: None,
            },
        }
    }

    pub fn with_sampling(mut self, sampling: ChatCompletionRequestSampling) -> Self {
        let (temperature, top_p) = match sampling {
            ChatCompletionRequestSampling::Temperature(t) => (t, 1.0),
            ChatCompletionRequestSampling::TopP(p) => (1.0, p),
        };
        self.req.temperature = Some(temperature);
        self.req.top_p = Some(top_p);
        self
    }

    /// Sets the number of chat completion choices to generate for each input message.
    ///
    /// # Arguments
    ///
    /// * `n` - How many chat completion choices to generate for each input message. If `n` is less than 1, then sets to `1`.
    pub fn with_n_choices(mut self, n: u64) -> Self {
        let n_choice = if n < 1 { 1 } else { n };
        self.req.n_choice = Some(n_choice);
        self
    }

    pub fn enable_parallel_tool_calls(mut self, flag: bool) -> Self {
        self.req.parallel_tool_calls = Some(flag);
        self
    }

    pub fn with_stream(mut self, flag: bool) -> Self {
        self.req.stream = Some(flag);
        self
    }

    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.req.stop = Some(stop);
        self
    }

    /// Sets the maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    ///
    /// # Argument
    ///
    /// * `max_tokens` - The maximum number of tokens to generate in the chat completion. If `max_tokens` is less than 1, then sets to `16`.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        let max_tokens = if max_tokens < 1 { 16 } else { max_tokens };
        self.req.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the presence penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.req.presence_penalty = Some(penalty);
        self
    }

    /// Sets the frequency penalty. Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.req.frequency_penalty = Some(penalty);
        self
    }

    pub fn with_logits_bias(mut self, map: HashMap<String, f64>) -> Self {
        self.req.logit_bias = Some(map);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    pub fn build(self) -> RagChatCompletionsRequest {
        self.req
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunksRequest {
    pub id: String,
    pub filename: String,
    pub chunk_capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunksResponse {
    pub id: String,
    pub filename: String,
    pub chunks: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrieveObject {
    /// The retrieved sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<RagScoredPoint>>,

    /// The number of similar points to retrieve
    pub limit: usize,

    /// The score threshold
    pub score_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagScoredPoint {
    /// Source of the context
    pub source: String,

    /// Points vector distance to the query vector
    pub score: f32,
}

#[test]
fn test_rag_serialize_retrieve_object() {
    {
        let ro = RetrieveObject {
            points: Some(vec![RagScoredPoint {
                source: "source".to_string(),
                score: 0.5,
            }]),
            limit: 1,
            score_threshold: 0.5,
        };
        let json = serde_json::to_string(&ro).unwrap();
        assert_eq!(
            json,
            r#"{"points":[{"source":"source","score":0.5}],"limit":1,"score_threshold":0.5}"#
        );
    }

    {
        let ro = RetrieveObject {
            points: None,
            limit: 1,
            score_threshold: 0.5,
        };
        let json = serde_json::to_string(&ro).unwrap();
        assert_eq!(json, r#"{"limit":1,"score_threshold":0.5}"#);
    }
}

#[test]
fn test_rag_deserialize_retrieve_object() {
    {
        let json =
            r#"{"points":[{"source":"source","score":0.5}],"limit":1,"score_threshold":0.5}"#;
        let ro: RetrieveObject = serde_json::from_str(json).unwrap();
        assert_eq!(ro.limit, 1);
        assert_eq!(ro.score_threshold, 0.5);
        assert!(ro.points.is_some());
        let points = ro.points.unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].source, "source");
        assert_eq!(points[0].score, 0.5);
    }

    {
        let json = r#"{"limit":1,"score_threshold":0.5}"#;
        let ro: RetrieveObject = serde_json::from_str(json).unwrap();
        assert_eq!(ro.limit, 1);
        assert_eq!(ro.score_threshold, 0.5);
        assert!(ro.points.is_none());
    }
}
