//! Define types for building chat completion requests, including messages, tools, and tool choices.
//!
//! **Example 1** Create a normal chat completion request.
//! ```
//! use endpoints::chat::*;
//!
//! let mut messages = Vec::new();
//!
//! // create a system message
//! let system_message = ChatCompletionRequestMessage::System(
//!     ChatCompletionSystemMessage::new("Hello, world!", None),
//! );
//! messages.push(system_message);
//!
//! // create a user message
//! let user_message_content = ChatCompletionUserMessageContent::Parts(vec![
//!     ContentPart::Text(TextContentPart::new("what is in the picture?")),
//!     ContentPart::Image(ImageContentPart::new(Image {
//!         url: "https://example.com/image.png".to_string(),
//!         detail: None,
//!     })),
//! ]);
//! let user_message =
//!     ChatCompletionRequestMessage::new_user_message(user_message_content, None);
//! messages.push(user_message);
//!
//! // create a chat completion request
//! let request = ChatCompletionRequestBuilder::new("model-id", messages)
//!     .with_tool_choice(ToolChoice::None)
//!     .build();
//!
//! // serialize the request to JSON string
//! let json = serde_json::to_string(&request).unwrap();
//! assert_eq!(
//!     json,
//!     r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":[{"type":"text","text":"what is in the picture?"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}],"max_tokens":1024,"tool_choice":"none"}"#
//! );
//! ```
//!
//! **Example 2** Create a chat completion request with available tools.
//! ```
//! use endpoints::chat::*;
//!
//! let mut messages = Vec::new();
//!
//! // create a system message
//! let system_message = ChatCompletionRequestMessage::System(
//!     ChatCompletionSystemMessage::new("Hello, world!", None),
//! );
//! messages.push(system_message);
//!
//! // create a user message
//! let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
//!     ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
//!     None,
//! ));
//! messages.push(user_message);
//! let assistant_message = ChatCompletionRequestMessage::Assistant(
//!     ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
//! );
//! messages.push(assistant_message);
//!
//! // create a tool
//! let params = ToolFunctionParameters {
//!     schema_type: JSONSchemaType::Object,
//!     properties: Some(
//!         vec![
//!             (
//!                 "location".to_string(),
//!                 Box::new(JSONSchemaDefine {
//!                     schema_type: Some(JSONSchemaType::String),
//!                     description: Some(
//!                         "The city and state, e.g. San Francisco, CA".to_string(),
//!                     ),
//!                     enum_values: None,
//!                     properties: None,
//!                     required: None,
//!                     items: None,
//!                     default: None,
//!                     maximum: None,
//!                     minimum: None,
//!                     title: None,
//!                     examples: None,
//!                 }),
//!             ),
//!             (
//!                 "unit".to_string(),
//!                 Box::new(JSONSchemaDefine {
//!                     schema_type: Some(JSONSchemaType::String),
//!                     description: None,
//!                     enum_values: Some(vec![
//!                         "celsius".to_string(),
//!                         "fahrenheit".to_string(),
//!                     ]),
//!                     properties: None,
//!                     required: None,
//!                     items: None,
//!                     default: None,
//!                     maximum: None,
//!                     minimum: None,
//!                     title: None,
//!                     examples: None,
//!                 }),
//!             ),
//!         ]
//!         .into_iter()
//!         .collect(),
//!     ),
//!     required: Some(vec!["location".to_string()]),
//! };
//! let tool = Tool {
//!     ty: "function".to_string(),
//!     function: ToolFunction {
//!         name: "my_function".to_string(),
//!         description: None,
//!         parameters: Some(params),
//!     },
//! };
//!
//! // create a chat completion request
//! let request = ChatCompletionRequestBuilder::new("model-id", messages)
//!     .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
//!     .with_n_choices(3)
//!     .enable_stream(true)
//!     .include_usage()
//!     .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
//!     .with_max_tokens(100)
//!     .with_presence_penalty(0.5)
//!     .with_frequency_penalty(0.5)
//!     .with_reponse_format(ChatResponseFormat::default())
//!     .with_tools(vec![tool])
//!     .with_tool_choice(ToolChoice::Tool(ToolChoiceTool {
//!         ty: "function".to_string(),
//!         function: ToolChoiceToolFunction {
//!             name: "my_function".to_string(),
//!         },
//!     }))
//!     .build();
//!
//! // serialize the request to JSON string
//! let json = serde_json::to_string(&request).unwrap();
//! assert_eq!(
//!     json,
//!     r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}],"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#
//! );
//! ```

use crate::common::{FinishReason, Usage};
use indexmap::IndexMap;
use serde::{
    de::{self, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::Value;
use std::{collections::HashMap, fmt};

/// Request builder for creating a new chat completion request.
pub struct ChatCompletionRequestBuilder {
    req: ChatCompletionRequest,
}
impl ChatCompletionRequestBuilder {
    /// Creates a new builder with the given model.
    ///
    /// # Arguments
    ///
    /// * `model` - ID of the model to use.
    ///
    /// * `messages` - A list of messages comprising the conversation so far.
    ///
    /// * `sampling` - The sampling method to use.
    pub fn new(model: impl Into<String>, messages: Vec<ChatCompletionRequestMessage>) -> Self {
        Self {
            req: ChatCompletionRequest {
                model: Some(model.into()),
                messages,
                temperature: None,
                top_p: None,
                n_choice: None,
                parallel_tool_calls: None,
                stream: None,
                stream_options: None,
                stop: None,
                max_tokens: Some(1024),
                presence_penalty: None,
                frequency_penalty: None,
                logit_bias: None,
                user: None,
                functions: None,
                function_call: None,
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


    /// Enable parallel_tool_calls
    pub fn enable_parallel_tool_calls(mut self, flag: bool) -> Self {
        self.req.parallel_tool_calls = Some(flag);
        self
    }

    /// Enables streaming reponse.
    pub fn enable_stream(mut self, flag: bool) -> Self {
        self.req.stream = Some(flag);
        self
    }

    /// Includes uage in streaming response.
    pub fn include_usage(mut self) -> Self {
        self.req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
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

    pub fn with_functions(mut self, functions: Vec<ChatCompletionRequestFunction>) -> Self {
        self.req.functions = Some(functions);
        self
    }

    pub fn with_function_call(mut self, function_call: impl Into<String>) -> Self {
        self.req.function_call = Some(function_call.into());
        self
    }

    /// Sets response format.
    pub fn with_reponse_format(mut self, response_format: ChatResponseFormat) -> Self {
        self.req.response_format = Some(response_format);
        self
    }

    /// Sets tools
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.req.tools = Some(tools);
        self
    }

    /// Sets tool choice.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.req.tool_choice = Some(tool_choice);
        self
    }

    pub fn build(self) -> ChatCompletionRequest {
        self.req
    }
}

/// Represents a chat completion request.
#[derive(Debug, Serialize, Default)]
pub struct ChatCompletionRequest {
    /// The model to use for generating completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionRequestMessage>,
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
    #[serde(rename = "n")]
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
    /// Defaults to 1024.
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

    //* OpenAI specific parameters
    /// **Deprecated since 0.10.0.** Use `tools` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatCompletionRequestFunction>>,
    /// **Deprecated since 0.10.0.** Use `tool_choice` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<String>,

    /// Format that the model must output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ChatResponseFormat>,
    /// A list of tools the model may call.
    ///
    /// Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Controls which (if any) function is called by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}
impl<'de> Deserialize<'de> for ChatCompletionRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChatCompletionRequestVisitor;

        impl<'de> Visitor<'de> for ChatCompletionRequestVisitor {
            type Value = ChatCompletionRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ChatCompletionRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ChatCompletionRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                // Initialize all fields as None or empty
                let mut model = None;
                let mut messages = None;
                let mut temperature = None;
                let mut top_p = None;
                let mut n_choice = None;
                let mut parallel_tool_calls = None;
                let mut stream = None;
                let mut stream_options = None;
                let mut stop = None;
                let mut max_tokens = None;
                let mut presence_penalty = None;
                let mut frequency_penalty = None;
                let mut logit_bias = None;
                let mut user = None;
                let mut functions = None;
                let mut function_call = None;
                let mut response_format = None;
                let mut tools = None;
                let mut tool_choice = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "model" => model = map.next_value()?,
                        "messages" => messages = map.next_value()?,
                        "temperature" => temperature = map.next_value()?,
                        "top_p" => top_p = map.next_value()?,
                        "n" => n_choice = map.next_value()?,
                        "parallel_tool_calls" => parallel_tool_calls = map.next_value()?,
                        "stream" => stream = map.next_value()?,
                        "stream_options" => stream_options = map.next_value()?,
                        "stop" => stop = map.next_value()?,
                        "max_tokens" => max_tokens = map.next_value()?,
                        "presence_penalty" => presence_penalty = map.next_value()?,
                        "frequency_penalty" => frequency_penalty = map.next_value()?,
                        "logit_bias" => logit_bias = map.next_value()?,
                        "user" => user = map.next_value()?,
                        "functions" => functions = map.next_value()?,
                        "function_call" => function_call = map.next_value()?,
                        "response_format" => response_format = map.next_value()?,
                        "tools" => tools = map.next_value()?,
                        "tool_choice" => tool_choice = map.next_value()?,
                        _ => return Err(de::Error::unknown_field(key.as_str(), FIELDS)),
                    }
                }

                // Ensure all required fields are initialized
                let messages = messages.ok_or_else(|| de::Error::missing_field("messages"))?;

                // Set default value for `max_tokens` if not provided
                if max_tokens.is_none() {
                    max_tokens = Some(1024);
                }

                // Check tools and tool_choice
                // `auto` is the default if tools are present.
                // `none` is the default when no tools are present.
                if tools.is_some() {
                    if tool_choice.is_none() {
                        tool_choice = Some(ToolChoice::Auto);
                    }
                } else if tool_choice.is_none() {
                    tool_choice = Some(ToolChoice::None);
                }

                if n_choice.is_none() {
                    n_choice = Some(1);
                }

                if parallel_tool_calls.is_none() {
                    parallel_tool_calls = Some(false);
                }

                if stream.is_none() {
                    stream = Some(false);
                }

                // Construct ChatCompletionRequest with all fields
                Ok(ChatCompletionRequest {
                    model,
                    messages,
                    temperature,
                    top_p,
                    n_choice,
                    parallel_tool_calls,
                    stream,
                    stream_options,
                    stop,
                    max_tokens,
                    presence_penalty,
                    frequency_penalty,
                    logit_bias,
                    user,
                    functions,
                    function_call,
                    response_format,
                    tools,
                    tool_choice,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "prompt",
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "parallel_tool_calls",
            "stream",
            "stream_options",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "functions",
            "function_call",
            "response_format",
            "tools",
            "tool_choice",
        ];
        deserializer.deserialize_struct(
            "ChatCompletionRequest",
            FIELDS,
            ChatCompletionRequestVisitor,
        )
    }
}

#[test]
fn test_chat_serialize_chat_request() {
    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);
        let request = ChatCompletionRequestBuilder::new("model-id", messages)
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tool_choice(ToolChoice::Auto)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":1024,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tool_choice":"auto"}"#
        );
    }

    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);

        let user_message_content = ChatCompletionUserMessageContent::Parts(vec![
            ContentPart::Text(TextContentPart::new("what is in the picture?")),
            ContentPart::Image(ImageContentPart::new(Image {
                url: "https://example.com/image.png".to_string(),
                detail: None,
            })),
        ]);
        let user_message =
            ChatCompletionRequestMessage::new_user_message(user_message_content, None);
        messages.push(user_message);

        let request = ChatCompletionRequestBuilder::new("model-id", messages)
            .with_tool_choice(ToolChoice::None)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":[{"type":"text","text":"what is in the picture?"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}],"max_tokens":1024,"tool_choice":"none"}"#
        );
    }

    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);

        let params = ToolFunctionParameters {
            schema_type: JSONSchemaType::Object,
            properties: Some(
                vec![
                    (
                        "location".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: Some(
                                "The city and state, e.g. San Francisco, CA".to_string(),
                            ),
                            enum_values: None,
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                    (
                        "unit".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: None,
                            enum_values: Some(vec![
                                "celsius".to_string(),
                                "fahrenheit".to_string(),
                            ]),
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            required: Some(vec!["location".to_string()]),
        };

        let tool = Tool {
            ty: "function".to_string(),
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: Some(params),
            },
        };

        let request = ChatCompletionRequestBuilder::new("model-id", messages)
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_max_tokens(100)
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tools(vec![tool])
            .with_tool_choice(ToolChoice::Tool(ToolChoiceTool {
                ty: "function".to_string(),
                function: ToolChoiceToolFunction {
                    name: "my_function".to_string(),
                },
            }))
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}],"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#
        );
    }

    {
        let mut messages = Vec::new();
        let system_message = ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage::new("Hello, world!", None),
        );
        messages.push(system_message);
        let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
            ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
            None,
        ));
        messages.push(user_message);
        let assistant_message = ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None),
        );
        messages.push(assistant_message);

        let params = ToolFunctionParameters {
            schema_type: JSONSchemaType::Object,
            properties: Some(
                vec![
                    (
                        "location".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: Some(
                                "The city and state, e.g. San Francisco, CA".to_string(),
                            ),
                            enum_values: None,
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                    (
                        "unit".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: None,
                            enum_values: Some(vec![
                                "celsius".to_string(),
                                "fahrenheit".to_string(),
                            ]),
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            required: Some(vec!["location".to_string()]),
        };

        let tool = Tool {
            ty: "function".to_string(),
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: Some(params),
            },
        };

        let request = ChatCompletionRequestBuilder::new("model-id", messages)
            .with_sampling(ChatCompletionRequestSampling::Temperature(0.8))
            .with_n_choices(3)
            .enable_stream(true)
            .include_usage()
            .with_stop(vec!["stop1".to_string(), "stop2".to_string()])
            .with_max_tokens(100)
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.5)
            .with_reponse_format(ChatResponseFormat::default())
            .with_tools(vec![tool])
            .with_tool_choice(ToolChoice::Auto)
            .build();
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(
            json,
            r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}],"tool_choice":"auto"}"#
        );
    }
}

#[test]
fn test_chat_deserialize_chat_request() {
    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stop":["stop1","stop2"],"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"}}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("model-id".to_string()));
        assert_eq!(request.messages.len(), 3);
        assert_eq!(
            request.messages[0],
            ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(
                "Hello, world!",
                None
            ))
        );
        assert_eq!(
            request.messages[1],
            ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
                ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
                None
            ))
        );
        assert_eq!(
            request.messages[2],
            ChatCompletionRequestMessage::Assistant(ChatCompletionAssistantMessage::new(
                Some("Hello, world!".to_string()),
                None,
                None
            ))
        );
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.n_choice, Some(3));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stop,
            Some(vec!["stop1".to_string(), "stop2".to_string()])
        );
        assert_eq!(request.max_tokens, Some(1024));
        assert_eq!(request.presence_penalty, Some(0.5));
        assert_eq!(request.frequency_penalty, Some(0.5));
        assert_eq!(request.tool_choice, Some(ToolChoice::None));
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tool_choice":"auto"}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("model-id".to_string()));
        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.n_choice, Some(3));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stop,
            Some(vec!["stop1".to_string(), "stop2".to_string()])
        );
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.presence_penalty, Some(0.5));
        assert_eq!(request.frequency_penalty, Some(0.5));
        assert_eq!(request.tool_choice, Some(ToolChoice::Auto));
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("model-id".to_string()));
        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.top_p, Some(1.0));
        assert_eq!(request.n_choice, Some(3));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request.stop,
            Some(vec!["stop1".to_string(), "stop2".to_string()])
        );
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.presence_penalty, Some(0.5));
        assert_eq!(request.frequency_penalty, Some(0.5));
        assert_eq!(
            request.tool_choice,
            Some(ToolChoice::Tool(ToolChoiceTool {
                ty: "function".to_string(),
                function: ToolChoiceToolFunction {
                    name: "my_function".to_string(),
                },
            }))
        );
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}],"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tools = request.tools.unwrap();
        let tool = &tools[0];
        assert_eq!(tool.ty, "function");
        assert_eq!(tool.function.name, "my_function");
        assert!(tool.function.description.is_none());
        assert!(tool.function.parameters.is_some());
        let params = tool.function.parameters.as_ref().unwrap();
        assert_eq!(params.schema_type, JSONSchemaType::Object);
        let properties = params.properties.as_ref().unwrap();
        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("unit"));
        assert!(properties.contains_key("location"));
        let unit = properties.get("unit").unwrap();
        assert_eq!(unit.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            unit.enum_values,
            Some(vec!["celsius".to_string(), "fahrenheit".to_string()])
        );
        let location = properties.get("location").unwrap();
        assert_eq!(location.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            location.description,
            Some("The city and state, e.g. San Francisco, CA".to_string())
        );
        let required = params.required.as_ref().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "location");
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"},"tools":[{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tool_choice = request.tool_choice.unwrap();
        assert_eq!(tool_choice, ToolChoice::Auto);
    }

    {
        let json = r#"{"model":"model-id","messages":[{"role":"system","content":"Hello, world!"},{"role":"user","content":"Hello, world!"},{"role":"assistant","content":"Hello, world!"}],"temperature":0.8,"top_p":1.0,"n":3,"stream":true,"stream_options":{"include_usage":true},"stop":["stop1","stop2"],"max_tokens":100,"presence_penalty":0.5,"frequency_penalty":0.5,"response_format":{"type":"text"}}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let tool_choice = request.tool_choice.unwrap();
        assert_eq!(tool_choice, ToolChoice::None);
    }

    {
        let json = r#"{"messages":[{"content":"Send an email to John Doe with the subject 'Hello' and the body 'Hello, John!'. His email is jhon@example.com","role":"user"}],"model":"llama","tool_choice":"auto","tools":[{"function":{"description":"Action to fetch all emails from Gmail.","name":"GMAIL_FETCH_EMAILS","parameters":{"properties":{"include_spam_trash":{"default":false,"description":"Include messages from SPAM and TRASH in the results.","title":"Include Spam Trash","type":"boolean"},"label_ids":{"default":null,"description":"Filter messages by their label IDs. Labels identify the status or category of messages. Some of the in-built labels include 'INBOX', 'SPAM', 'TRASH', 'UNREAD', 'STARRED', 'IMPORTANT', 'CATEGORY_PERSONAL', 'CATEGORY_SOCIAL', 'CATEGORY_PROMOTIONS', 'CATEGORY_UPDATES', and 'CATEGORY_FORUMS'. The 'label_ids' for custom labels can be found in the response of the 'listLabels' action. Note: The label_ids is a list of label IDs to filter the messages by.","items":{"type":"string"},"title":"Label Ids","type":"array"},"max_results":{"default":10,"description":"Maximum number of messages to return.","maximum":500,"minimum":1,"title":"Max Results","type":"integer"},"page_token":{"default":null,"description":"Page token to retrieve a specific page of results in the list. The page token is returned in the response of this action if there are more results to be fetched. If not provided, the first page of results is returned.","title":"Page Token","type":"string"},"query":{"default":null,"description":"Only return messages matching the specified query.","title":"Query","type":"string"},"user_id":{"default":"me","description":"The user's email address or 'me' for the authenticated user.","title":"User Id","type":"string"}},"title":"FetchEmailsRequest","type":"object"}},"type":"function"}]}"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.model.is_some());
        let tools = request.tools.unwrap();
        assert!(tools.len() == 1);
        let tool = &tools[0];
        assert_eq!(tool.ty, "function");
        assert_eq!(tool.function.name, "GMAIL_FETCH_EMAILS");
        assert!(tool.function.parameters.is_some());
        let params = tool.function.parameters.as_ref().unwrap();
        assert!(params.properties.is_some());
        let properties = params.properties.as_ref().unwrap();
        assert!(properties.len() == 6);
        assert!(properties.contains_key("max_results"));
        let max_results = properties.get("max_results").unwrap();
        assert!(max_results.description.is_some());
        assert_eq!(
            max_results.description.as_ref().unwrap(),
            "Maximum number of messages to return."
        );
        assert!(max_results.schema_type.is_some());
        assert_eq!(max_results.schema_type, Some(JSONSchemaType::Integer));
        println!("{:?}", max_results);
    }
}

/// An object specifying the format that the model must output.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatResponseFormat {
    /// Must be one of `text`` or `json_object`. Defaults to `text`.
    #[serde(rename = "type")]
    pub ty: String,
}
impl Default for ChatResponseFormat {
    fn default() -> Self {
        Self {
            ty: "text".to_string(),
        }
    }
}

#[test]
fn test_chat_serialize_response_format() {
    let response_format = ChatResponseFormat {
        ty: "text".to_string(),
    };
    let json = serde_json::to_string(&response_format).unwrap();
    assert_eq!(json, r#"{"type":"text"}"#);

    let response_format = ChatResponseFormat {
        ty: "json_object".to_string(),
    };
    let json = serde_json::to_string(&response_format).unwrap();
    assert_eq!(json, r#"{"type":"json_object"}"#);
}

/// Options for streaming response. Only set this when you set stream: `true``.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Controls which (if any) function is called by the model. Defaults to `None`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model will not call a function and instead generates a message.
    #[serde(rename = "none")]
    None,
    /// The model can pick between generating a message or calling a function.
    #[serde(rename = "auto")]
    Auto,
    /// The model must call one or more tools.
    #[serde(rename = "required")]
    Required,
    /// Specifies a tool the model should use. Use to force the model to call a specific function.
    #[serde(untagged)]
    Tool(ToolChoiceTool),
}
impl Default for ToolChoice {
    fn default() -> Self {
        Self::None
    }
}

#[test]
fn test_chat_serialize_tool_choice() {
    let tool_choice = ToolChoice::None;
    let json = serde_json::to_string(&tool_choice).unwrap();
    assert_eq!(json, r#""none""#);

    let tool_choice = ToolChoice::Auto;
    let json = serde_json::to_string(&tool_choice).unwrap();
    assert_eq!(json, r#""auto""#);

    let tool_choice = ToolChoice::Tool(ToolChoiceTool {
        ty: "function".to_string(),
        function: ToolChoiceToolFunction {
            name: "my_function".to_string(),
        },
    });
    let json = serde_json::to_string(&tool_choice).unwrap();
    assert_eq!(
        json,
        r#"{"type":"function","function":{"name":"my_function"}}"#
    );
}

#[test]
fn test_chat_deserialize_tool_choice() {
    let json = r#""none""#;
    let tool_choice: ToolChoice = serde_json::from_str(json).unwrap();
    assert_eq!(tool_choice, ToolChoice::None);

    let json = r#""auto""#;
    let tool_choice: ToolChoice = serde_json::from_str(json).unwrap();
    assert_eq!(tool_choice, ToolChoice::Auto);

    let json = r#"{"type":"function","function":{"name":"my_function"}}"#;
    let tool_choice: ToolChoice = serde_json::from_str(json).unwrap();
    assert_eq!(
        tool_choice,
        ToolChoice::Tool(ToolChoiceTool {
            ty: "function".to_string(),
            function: ToolChoiceToolFunction {
                name: "my_function".to_string(),
            },
        })
    );
}

/// A tool the model should use.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolChoiceTool {
    /// The type of the tool. Currently, only `function` is supported.
    #[serde(rename = "type")]
    pub ty: String,
    /// The function the model calls.
    pub function: ToolChoiceToolFunction,
}

/// Represents a tool the model should use.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolChoiceToolFunction {
    /// The name of the function to call.
    pub name: String,
}

/// Represents a tool the model may generate JSON inputs for.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    /// The type of the tool. Currently, only `function` is supported.
    #[serde(rename = "type")]
    pub ty: String,
    /// Function the model may generate JSON inputs for.
    pub function: ToolFunction,
}

#[test]
fn test_chat_serialize_tool() {
    {
        let tool = Tool {
            ty: "function".to_string(),
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: None,
            },
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert_eq!(
            json,
            r#"{"type":"function","function":{"name":"my_function"}}"#
        );
    }

    {
        let params = ToolFunctionParameters {
            schema_type: JSONSchemaType::Object,
            properties: Some(
                vec![
                    (
                        "location".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: Some(
                                "The city and state, e.g. San Francisco, CA".to_string(),
                            ),
                            enum_values: None,
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                    (
                        "unit".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: None,
                            enum_values: Some(vec![
                                "celsius".to_string(),
                                "fahrenheit".to_string(),
                            ]),
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            required: Some(vec!["location".to_string()]),
        };

        let tool = Tool {
            ty: "function".to_string(),
            function: ToolFunction {
                name: "my_function".to_string(),
                description: None,
                parameters: Some(params),
            },
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert_eq!(
            json,
            r#"{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}"#
        );
    }

    {
        let tool_1 = Tool {
            ty: "function".to_string(),
            function: ToolFunction {
                name: "my_function_1".to_string(),
                description: None,
                parameters: None,
            },
        };

        let params = ToolFunctionParameters {
            schema_type: JSONSchemaType::Object,
            properties: Some(
                vec![
                    (
                        "location".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: Some(
                                "The city and state, e.g. San Francisco, CA".to_string(),
                            ),
                            enum_values: None,
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                    (
                        "unit".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: None,
                            enum_values: Some(vec![
                                "celsius".to_string(),
                                "fahrenheit".to_string(),
                            ]),
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            required: Some(vec!["location".to_string()]),
        };

        let tool_2 = Tool {
            ty: "function".to_string(),
            function: ToolFunction {
                name: "my_function_2".to_string(),
                description: None,
                parameters: Some(params),
            },
        };

        let tools = vec![tool_1, tool_2];
        let json = serde_json::to_string(&tools).unwrap();
        assert_eq!(
            json,
            r#"[{"type":"function","function":{"name":"my_function_1"}},{"type":"function","function":{"name":"my_function_2","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]"#
        );
    }
}

#[test]
fn test_chat_deserialize_tool() {
    let json = r#"{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}"#;
    let tool: Tool = serde_json::from_str(json).unwrap();
    assert_eq!(tool.ty, "function");
    assert_eq!(tool.function.name, "my_function");
    assert!(tool.function.description.is_none());
    assert!(tool.function.parameters.is_some());
    let params = tool.function.parameters.as_ref().unwrap();
    assert_eq!(params.schema_type, JSONSchemaType::Object);
    let properties = params.properties.as_ref().unwrap();
    assert_eq!(properties.len(), 2);
    assert!(properties.contains_key("unit"));
    assert!(properties.contains_key("location"));
    let unit = properties.get("unit").unwrap();
    assert_eq!(unit.schema_type, Some(JSONSchemaType::String));
    assert_eq!(
        unit.enum_values,
        Some(vec!["celsius".to_string(), "fahrenheit".to_string()])
    );
    let location = properties.get("location").unwrap();
    assert_eq!(location.schema_type, Some(JSONSchemaType::String));
    assert_eq!(
        location.description,
        Some("The city and state, e.g. San Francisco, CA".to_string())
    );
    let required = params.required.as_ref().unwrap();
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "location");
}

/// Function the model may generate JSON inputs for.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolFunction {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// A description of what the function does, used by the model to choose when and how to call the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    // The parameters the functions accepts, described as a JSON Schema object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<ToolFunctionParameters>,
}

#[test]
fn test_chat_serialize_tool_function() {
    let params = ToolFunctionParameters {
        schema_type: JSONSchemaType::Object,
        properties: Some(
            vec![
                (
                    "location".to_string(),
                    Box::new(JSONSchemaDefine {
                        schema_type: Some(JSONSchemaType::String),
                        description: Some("The city and state, e.g. San Francisco, CA".to_string()),
                        enum_values: None,
                        properties: None,
                        required: None,
                        items: None,
                        default: None,
                        maximum: None,
                        minimum: None,
                        title: None,
                        examples: None,
                    }),
                ),
                (
                    "unit".to_string(),
                    Box::new(JSONSchemaDefine {
                        schema_type: Some(JSONSchemaType::String),
                        description: None,
                        enum_values: Some(vec!["celsius".to_string(), "fahrenheit".to_string()]),
                        properties: None,
                        required: None,
                        items: None,
                        default: None,
                        maximum: None,
                        minimum: None,
                        title: None,
                        examples: None,
                    }),
                ),
            ]
            .into_iter()
            .collect(),
        ),
        required: Some(vec!["location".to_string()]),
    };

    let func = ToolFunction {
        name: "my_function".to_string(),
        description: Some("Get the current weather in a given location".to_string()),
        parameters: Some(params),
    };

    let json = serde_json::to_string(&func).unwrap();
    assert_eq!(
        json,
        r#"{"name":"my_function","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}"#
    );
}

/// The parameters the functions accepts, described as a JSON Schema object.
///
/// See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
///
/// To describe a function that accepts no parameters, provide the value
/// `{"type": "object", "properties": {}}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolFunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: JSONSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[test]
fn test_chat_serialize_tool_function_params() {
    {
        let params = ToolFunctionParameters {
            schema_type: JSONSchemaType::Object,
            properties: Some(
                vec![
                    (
                        "location".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: Some(
                                "The city and state, e.g. San Francisco, CA".to_string(),
                            ),
                            enum_values: None,
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                    (
                        "unit".to_string(),
                        Box::new(JSONSchemaDefine {
                            schema_type: Some(JSONSchemaType::String),
                            description: None,
                            enum_values: Some(vec![
                                "celsius".to_string(),
                                "fahrenheit".to_string(),
                            ]),
                            properties: None,
                            required: None,
                            items: None,
                            default: None,
                            maximum: None,
                            minimum: None,
                            title: None,
                            examples: None,
                        }),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            required: Some(vec!["location".to_string()]),
        };

        let json = serde_json::to_string(&params).unwrap();
        assert_eq!(
            json,
            r#"{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}"#
        );
    }
}

#[test]
fn test_chat_deserialize_tool_function_params() {
    {
        let json = r###"
    {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
    }"###;
        let params: ToolFunctionParameters = serde_json::from_str(json).unwrap();
        assert_eq!(params.schema_type, JSONSchemaType::Object);
        let properties = params.properties.as_ref().unwrap();
        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("unit"));
        assert!(properties.contains_key("location"));
        let unit = properties.get("unit").unwrap();
        assert_eq!(unit.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            unit.enum_values,
            Some(vec!["celsius".to_string(), "fahrenheit".to_string()])
        );
        let location = properties.get("location").unwrap();
        assert_eq!(location.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            location.description,
            Some("The city and state, e.g. San Francisco, CA".to_string())
        );
        let required = params.required.as_ref().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "location");
    }

    {
        let json = r###"{
            "properties": {
                "include_spam_trash": {
                    "default": false,
                    "description": "Include messages from SPAM and TRASH in the results.",
                    "title": "Include Spam Trash",
                    "type": "boolean"
                },
                "add_label_ids": {
                    "default": [],
                    "description": "A list of IDs of labels to add to this thread.",
                    "items": {
                        "type": "string"
                    },
                    "title": "Add Label Ids",
                    "type": "array"
                },
                "max_results": {
                    "default": 10,
                    "description": "Maximum number of messages to return.",
                    "examples": [
                        10,
                        50,
                        100
                    ],
                    "maximum": 500,
                    "minimum": 1,
                    "title": "Max Results",
                    "type": "integer"
                },
                "query": {
                    "default": null,
                    "description": "Only return threads matching the specified query.",
                    "examples": [
                        "is:unread",
                        "from:john.doe@example.com"
                    ],
                    "title": "Query",
                    "type": "string"
                }
            },
            "title": "FetchEmailsRequest",
            "type": "object"
        }"###;

        let params: ToolFunctionParameters = serde_json::from_str(json).unwrap();
        assert_eq!(params.schema_type, JSONSchemaType::Object);
        let properties = params.properties.as_ref().unwrap();
        assert_eq!(properties.len(), 4);
        // println!("{:?}", properties);
        assert!(properties.contains_key("include_spam_trash"));
        assert!(properties.contains_key("add_label_ids"));
        assert!(properties.contains_key("max_results"));
        assert!(properties.contains_key("query"));

        let include_spam_trash = properties.get("include_spam_trash").unwrap();
        assert_eq!(
            include_spam_trash.schema_type,
            Some(JSONSchemaType::Boolean)
        );
        assert_eq!(
            include_spam_trash.description,
            Some("Include messages from SPAM and TRASH in the results.".to_string())
        );
        assert_eq!(
            include_spam_trash.title,
            Some("Include Spam Trash".to_string())
        );
        assert_eq!(
            include_spam_trash.default,
            Some(serde_json::Value::Bool(false))
        );

        let add_label_ids = properties.get("add_label_ids").unwrap();
        assert_eq!(add_label_ids.schema_type, Some(JSONSchemaType::Array));
        assert_eq!(
            add_label_ids.description,
            Some("A list of IDs of labels to add to this thread.".to_string())
        );
        assert_eq!(add_label_ids.title, Some("Add Label Ids".to_string()));
        assert_eq!(
            add_label_ids.default,
            Some(serde_json::Value::Array(vec![]))
        );
        let items = add_label_ids.items.as_ref().unwrap();
        assert_eq!(items.schema_type, Some(JSONSchemaType::String));

        let max_results = properties.get("max_results").unwrap();
        assert_eq!(max_results.schema_type, Some(JSONSchemaType::Integer));
        assert_eq!(
            max_results.description,
            Some("Maximum number of messages to return.".to_string())
        );
        assert_eq!(
            max_results.examples,
            Some(vec![
                Value::Number(serde_json::Number::from(10)),
                Value::Number(serde_json::Number::from(50)),
                Value::Number(serde_json::Number::from(100))
            ])
        );
        assert_eq!(
            max_results.maximum,
            Some(Value::Number(serde_json::Number::from(500)))
        );
        assert_eq!(
            max_results.minimum,
            Some(Value::Number(serde_json::Number::from(1)))
        );
        assert_eq!(max_results.title, Some("Max Results".to_string()));
        assert_eq!(
            max_results.default,
            Some(serde_json::Value::Number(10.into()))
        );

        let query = properties.get("query").unwrap();
        assert_eq!(query.schema_type, Some(JSONSchemaType::String));
        assert_eq!(
            query.description,
            Some("Only return threads matching the specified query.".to_string())
        );
        assert_eq!(
            query.examples,
            Some(vec![
                Value::String("is:unread".to_string()),
                Value::String("from:john.doe@example.com".to_string())
            ])
        );
        assert_eq!(query.title, Some("Query".to_string()));
        assert_eq!(query.default, None);
    }
}

/// Message for comprising the conversation.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionRequestMessage {
    System(ChatCompletionSystemMessage),
    User(ChatCompletionUserMessage),
    Assistant(ChatCompletionAssistantMessage),
    Tool(ChatCompletionToolMessage),
}
impl ChatCompletionRequestMessage {
    /// Creates a new system message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the system message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new_system_message(content: impl Into<String>, name: Option<String>) -> Self {
        ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(content, name))
    }

    /// Creates a new user message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the user message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new_user_message(
        content: ChatCompletionUserMessageContent,
        name: Option<String>,
    ) -> Self {
        ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(content, name))
    }

    /// Creates a new assistant message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the assistant message. Required unless `tool_calls` is specified.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    ///
    /// * `tool_calls` - The tool calls generated by the model.
    pub fn new_assistant_message(
        content: Option<String>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        ChatCompletionRequestMessage::Assistant(ChatCompletionAssistantMessage::new(
            content, name, tool_calls,
        ))
    }

    /// Creates a new tool message.
    pub fn new_tool_message(content: impl Into<String>, tool_call_id: Option<String>) -> Self {
        ChatCompletionRequestMessage::Tool(ChatCompletionToolMessage::new(content, tool_call_id))
    }

    /// The role of the messages author.
    pub fn role(&self) -> ChatCompletionRole {
        match self {
            ChatCompletionRequestMessage::System(_) => ChatCompletionRole::System,
            ChatCompletionRequestMessage::User(_) => ChatCompletionRole::User,
            ChatCompletionRequestMessage::Assistant(_) => ChatCompletionRole::Assistant,
            ChatCompletionRequestMessage::Tool(_) => ChatCompletionRole::Tool,
        }
    }

    /// The name of the participant. Provides the model information to differentiate between participants of the same role.
    pub fn name(&self) -> Option<&String> {
        match self {
            ChatCompletionRequestMessage::System(message) => message.name(),
            ChatCompletionRequestMessage::User(message) => message.name(),
            ChatCompletionRequestMessage::Assistant(message) => message.name(),
            ChatCompletionRequestMessage::Tool(_) => None,
        }
    }
}

#[test]
fn test_chat_serialize_request_message() {
    let message = ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(
        "Hello, world!",
        None,
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"role":"system","content":"Hello, world!"}"#);

    let message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
        ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
        None,
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"role":"user","content":"Hello, world!"}"#);

    let message = ChatCompletionRequestMessage::Assistant(ChatCompletionAssistantMessage::new(
        Some("Hello, world!".to_string()),
        None,
        None,
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"role":"assistant","content":"Hello, world!"}"#);

    let message = ChatCompletionRequestMessage::Tool(ChatCompletionToolMessage::new(
        "Hello, world!",
        Some("tool-call-id".into()),
    ));
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(
        json,
        r#"{"role":"tool","content":"Hello, world!","tool_call_id":"tool-call-id"}"#
    );
}

#[test]
fn test_chat_deserialize_request_message() {
    let json = r#"{"content":"Hello, world!","role":"assistant"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::Assistant);

    let json = r#"{"content":"Hello, world!","role":"system"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::System);

    let json = r#"{"content":"Hello, world!","role":"user"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::User);

    let json = r#"{"role":"tool","content":"Hello, world!","tool_call_id":"tool-call-id"}"#;
    let message: ChatCompletionRequestMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::Tool);
}

/// Defines the content of a system message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionSystemMessage {
    /// The contents of the system message.
    content: String,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}
impl ChatCompletionSystemMessage {
    /// Creates a new system message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the system message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new(content: impl Into<String>, name: Option<String>) -> Self {
        Self {
            content: content.into(),
            name,
        }
    }

    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::System
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
}

/// Defines the content of a user message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionUserMessage {
    /// The contents of the user message.
    content: ChatCompletionUserMessageContent,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}
impl ChatCompletionUserMessage {
    /// Creates a new user message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the user message.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    pub fn new(content: ChatCompletionUserMessageContent, name: Option<String>) -> Self {
        Self { content, name }
    }

    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::User
    }

    pub fn content(&self) -> &ChatCompletionUserMessageContent {
        &self.content
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
}

#[test]
fn test_chat_serialize_user_message() {
    let message = ChatCompletionUserMessage::new(
        ChatCompletionUserMessageContent::Text("Hello, world!".to_string()),
        None,
    );
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"content":"Hello, world!"}"#);

    let message = ChatCompletionUserMessage::new(
        ChatCompletionUserMessageContent::Parts(vec![
            ContentPart::Text(TextContentPart::new("Hello, world!")),
            ContentPart::Image(ImageContentPart::new(Image {
                url: "https://example.com/image.png".to_string(),
                detail: Some("auto".to_string()),
            })),
        ]),
        None,
    );
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(
        json,
        r#"{"content":[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}]}"#
    );
}

#[test]
fn test_chat_deserialize_user_message() {
    let json = r#"{"content":"Hello, world!","role":"user"}"#;
    let message: ChatCompletionUserMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.content().ty(), "text");

    let json = r#"{"content":[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}],"role":"user"}"#;
    let message: ChatCompletionUserMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.content().ty(), "parts");
}

/// Defines the content of an assistant message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionAssistantMessage {
    /// The contents of the assistant message. Required unless `tool_calls` is specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// The tool calls generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}
impl ChatCompletionAssistantMessage {
    /// Creates a new assistant message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the assistant message. Required unless `tool_calls` is specified.
    ///
    /// * `name` - An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    ///
    /// * `tool_calls` - The tool calls generated by the model.
    pub fn new(
        content: Option<String>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        match tool_calls.is_some() {
            true => Self {
                content: None,
                name,
                tool_calls,
            },
            false => Self {
                content,
                name,
                tool_calls: None,
            },
        }
    }

    /// The role of the messages author, in this case `assistant`.
    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::Assistant
    }

    /// The contents of the assistant message. If `tool_calls` is specified, then `content` is None.
    pub fn content(&self) -> Option<&String> {
        self.content.as_ref()
    }

    /// An optional name for the participant.
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// The tool calls generated by the model.
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.tool_calls.as_ref()
    }
}

#[test]
fn test_chat_serialize_assistant_message() {
    let message =
        ChatCompletionAssistantMessage::new(Some("Hello, world!".to_string()), None, None);
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(json, r#"{"content":"Hello, world!"}"#);
}

#[test]
fn test_chat_deserialize_assistant_message() {
    let json = r#"{"content":"Hello, world!","role":"assistant"}"#;
    let message: ChatCompletionAssistantMessage = serde_json::from_str(json).unwrap();
    assert_eq!(message.role(), ChatCompletionRole::Assistant);
    assert_eq!(message.content().unwrap().as_str(), "Hello, world!");
}

/// Defines the content of a tool message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatCompletionToolMessage {
    /// The contents of the tool message.
    content: String,
    /// Tool call that this message is responding to.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}
impl ChatCompletionToolMessage {
    /// Creates a new tool message.
    ///
    /// # Arguments
    ///
    /// * `content` - The contents of the tool message.
    ///
    /// * `tool_call_id` - Tool call that this message is responding to.
    pub fn new(content: impl Into<String>, tool_call_id: Option<String>) -> Self {
        Self {
            content: content.into(),
            tool_call_id,
        }
    }

    /// The role of the messages author, in this case `tool`.
    pub fn role(&self) -> ChatCompletionRole {
        ChatCompletionRole::Tool
    }

    /// The contents of the tool message.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Tool call that this message is responding to.
    pub fn tool_call_id(&self) -> Option<String> {
        self.tool_call_id.clone()
    }
}

/// Represents a tool call generated by the model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename = "type")]
    pub ty: String,
    /// The function that the model called.
    pub function: Function,
}

#[test]
fn test_deserialize_tool_call() {
    let json = r#"{"id":"tool-call-id","type":"function","function":{"name":"my_function","arguments":"{\"location\":\"San Francisco, CA\"}"}}"#;
    let tool_call: ToolCall = serde_json::from_str(json).unwrap();
    assert_eq!(tool_call.id, "tool-call-id");
    assert_eq!(tool_call.ty, "function");
    assert_eq!(
        tool_call.function,
        Function {
            name: "my_function".to_string(),
            arguments: r#"{"location":"San Francisco, CA"}"#.to_string()
        }
    );
}

/// Represents a tool call generated by the model.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCallForChunk {
    pub index: usize,
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only function is supported.
    #[serde(rename = "type")]
    pub ty: String,
    /// The function that the model called.
    pub function: Function,
}

#[test]
fn test_deserialize_tool_call_for_chunk() {
    let json = r#"{"index":0, "id":"tool-call-id","type":"function","function":{"name":"my_function","arguments":"{\"location\":\"San Francisco, CA\"}"}}"#;
    let tool_call: ToolCallForChunk = serde_json::from_str(json).unwrap();
    assert_eq!(tool_call.index, 0);
    assert_eq!(tool_call.id, "tool-call-id");
    assert_eq!(tool_call.ty, "function");
    assert_eq!(
        tool_call.function,
        Function {
            name: "my_function".to_string(),
            arguments: r#"{"location":"San Francisco, CA"}"#.to_string()
        }
    );
}

/// The function that the model called.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Function {
    /// The name of the function that the model called.
    pub name: String,
    /// The arguments that the model called the function with.
    pub arguments: String,
}

/// Defines the types of a user message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ChatCompletionUserMessageContent {
    /// The text contents of the message.
    Text(String),
    /// An array of content parts with a defined type, each can be of type `text` or `image_url` when passing in images.
    /// It is required that there must be one content part of type `text` at least. Multiple images are allowed by adding multiple image_url content parts.
    Parts(Vec<ContentPart>),
}
impl ChatCompletionUserMessageContent {
    pub fn ty(&self) -> &str {
        match self {
            ChatCompletionUserMessageContent::Text(_) => "text",
            ChatCompletionUserMessageContent::Parts(_) => "parts",
        }
    }
}

#[test]
fn test_chat_serialize_user_message_content() {
    let content = ChatCompletionUserMessageContent::Text("Hello, world!".to_string());
    let json = serde_json::to_string(&content).unwrap();
    assert_eq!(json, r#""Hello, world!""#);

    let content = ChatCompletionUserMessageContent::Parts(vec![
        ContentPart::Text(TextContentPart::new("Hello, world!")),
        ContentPart::Image(ImageContentPart::new(Image {
            url: "https://example.com/image.png".to_string(),
            detail: Some("auto".to_string()),
        })),
    ]);
    let json = serde_json::to_string(&content).unwrap();
    assert_eq!(
        json,
        r#"[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}]"#
    );
}

#[test]
fn test_chat_deserialize_user_message_content() {
    let json = r#"[{"type":"text","text":"Hello, world!"},{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}]"#;
    let content: ChatCompletionUserMessageContent = serde_json::from_str(json).unwrap();
    assert_eq!(content.ty(), "parts");
    if let ChatCompletionUserMessageContent::Parts(parts) = content {
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].ty(), "text");
        assert_eq!(parts[1].ty(), "image_url");
    }
}

/// Define the content part of a user message.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
// #[serde(untagged)]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text(TextContentPart),
    #[serde(rename = "image_url")]
    Image(ImageContentPart),
}
impl ContentPart {
    pub fn ty(&self) -> &str {
        match self {
            ContentPart::Text(_) => "text",
            ContentPart::Image(_) => "image_url",
        }
    }
}

#[test]
fn test_chat_serialize_content_part() {
    let text_content_part = TextContentPart::new("Hello, world!");
    let content_part = ContentPart::Text(text_content_part);
    let json = serde_json::to_string(&content_part).unwrap();
    assert_eq!(json, r#"{"type":"text","text":"Hello, world!"}"#);

    let image_content_part = ImageContentPart::new(Image {
        url: "https://example.com/image.png".to_string(),
        detail: Some("auto".to_string()),
    });
    let content_part = ContentPart::Image(image_content_part);
    let json = serde_json::to_string(&content_part).unwrap();
    assert_eq!(
        json,
        r#"{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#
    );
}

#[test]
fn test_chat_deserialize_content_part() {
    let json = r#"{"type":"text","text":"Hello, world!"}"#;
    let content_part: ContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(content_part.ty(), "text");

    let json = r#"{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#;
    let content_part: ContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(content_part.ty(), "image_url");
}

/// Represents the text part of a user message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct TextContentPart {
    /// The text content.
    text: String,
}
impl TextContentPart {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// The text content.
    pub fn text(&self) -> &str {
        &self.text
    }
}

#[test]
fn test_chat_serialize_text_content_part() {
    let text_content_part = TextContentPart::new("Hello, world!");
    let json = serde_json::to_string(&text_content_part).unwrap();
    assert_eq!(json, r#"{"text":"Hello, world!"}"#);
}

#[test]
fn test_chat_deserialize_text_content_part() {
    let json = r#"{"type":"text","text":"Hello, world!"}"#;
    let text_content_part: TextContentPart = serde_json::from_str(json).unwrap();
    assert_eq!(text_content_part.text, "Hello, world!");
}

/// Represents the image part of a user message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ImageContentPart {
    #[serde(rename = "image_url")]
    image: Image,
}
impl ImageContentPart {
    pub fn new(image: Image) -> Self {
        Self { image }
    }

    /// The image URL.
    pub fn image(&self) -> &Image {
        &self.image
    }
}

#[test]
fn test_chat_serialize_image_content_part() {
    let image_content_part = ImageContentPart::new(Image {
        url: "https://example.com/image.png".to_string(),
        detail: Some("auto".to_string()),
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(
        json,
        r#"{"image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#
    );

    let image_content_part = ImageContentPart::new(Image {
        url: "https://example.com/image.png".to_string(),
        detail: None,
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(
        json,
        r#"{"image_url":{"url":"https://example.com/image.png"}}"#
    );

    let image_content_part = ImageContentPart::new(Image {
        url: "base64".to_string(),
        detail: Some("auto".to_string()),
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(json, r#"{"image_url":{"url":"base64","detail":"auto"}}"#);

    let image_content_part = ImageContentPart::new(Image {
        url: "base64".to_string(),
        detail: None,
    });
    let json = serde_json::to_string(&image_content_part).unwrap();
    assert_eq!(json, r#"{"image_url":{"url":"base64"}}"#);
}

#[test]
fn test_chat_deserialize_image_content_part() {
    let json = r#"{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"auto"}}"#;
    let image_content_part: ImageContentPart = serde_json::from_str(json).unwrap();
    // assert_eq!(image_content_part.ty, "image_url");
    assert_eq!(
        image_content_part.image.url,
        "https://example.com/image.png"
    );
    assert_eq!(image_content_part.image.detail, Some("auto".to_string()));
}

/// JPEG baseline & progressive (12 bpc/arithmetic not supported, same as stock IJG lib)
/// PNG 1/2/4/8/16-bit-per-channel
///
/// TGA (not sure what subset, if a subset)
/// BMP non-1bpp, non-RLE
/// PSD (composited view only, no extra channels, 8/16 bit-per-channel)
///
/// GIF (*comp always reports as 4-channel)
/// HDR (radiance rgbE format)
/// PIC (Softimage PIC)
/// PNM (PPM and PGM binary only)
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Image {
    /// Either a URL of the image or the base64 encoded image data.
    pub url: String,
    /// Specifies the detail level of the image. Defaults to auto.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}
impl Image {
    pub fn is_url(&self) -> bool {
        url::Url::parse(&self.url).is_ok()
    }
}

#[test]
fn test_chat_serialize_image() {
    let image = Image {
        url: "https://example.com/image.png".to_string(),
        detail: Some("auto".to_string()),
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(
        json,
        r#"{"url":"https://example.com/image.png","detail":"auto"}"#
    );

    let image = Image {
        url: "https://example.com/image.png".to_string(),
        detail: None,
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(json, r#"{"url":"https://example.com/image.png"}"#);

    let image = Image {
        url: "base64".to_string(),
        detail: Some("auto".to_string()),
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(json, r#"{"url":"base64","detail":"auto"}"#);

    let image = Image {
        url: "base64".to_string(),
        detail: None,
    };
    let json = serde_json::to_string(&image).unwrap();
    assert_eq!(json, r#"{"url":"base64"}"#);
}

#[test]
fn test_chat_deserialize_image() {
    let json = r#"{"url":"https://example.com/image.png","detail":"auto"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "https://example.com/image.png");
    assert_eq!(image.detail, Some("auto".to_string()));

    let json = r#"{"url":"https://example.com/image.png"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "https://example.com/image.png");
    assert_eq!(image.detail, None);

    let json = r#"{"url":"base64","detail":"auto"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "base64");
    assert_eq!(image.detail, Some("auto".to_string()));

    let json = r#"{"url":"base64"}"#;
    let image: Image = serde_json::from_str(json).unwrap();
    assert_eq!(image.url, "base64");
    assert_eq!(image.detail, None);
}

/// Sampling methods used for chat completion requests.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
pub enum ChatCompletionRequestSampling {
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    Temperature(f64),
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    TopP(f64),
}

/// The role of the messages author.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRole {
    System,
    User,
    Assistant,
    /// **Deprecated since 0.10.0.** Use [ChatCompletionRole::Tool] instead.
    Function,
    Tool,
}
impl std::fmt::Display for ChatCompletionRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatCompletionRole::System => write!(f, "system"),
            ChatCompletionRole::User => write!(f, "user"),
            ChatCompletionRole::Assistant => write!(f, "assistant"),
            ChatCompletionRole::Function => write!(f, "function"),
            ChatCompletionRole::Tool => write!(f, "tool"),
        }
    }
}

/// **Deprecated since 0.10.0.** Use [Tool] instead.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: ChatCompletionRequestFunctionParameters,
}

/// The parameters the functions accepts, described as a JSON Schema object.
///
/// See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
///
/// To describe a function that accepts no parameters, provide the value
/// `{"type": "object", "properties": {}}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequestFunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: JSONSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JSONSchemaType {
    Object,
    Number,
    Integer,
    String,
    Array,
    Null,
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSONSchemaDefine {
    #[serde(rename = "type")]
    pub schema_type: Option<JSONSchemaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JSONSchemaDefine>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<Value>>,
}

/// Represents a chat completion response returned by model, based on the provided input.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionObject {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// The object type, which is always `chat.completion`.
    pub object: String,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for the chat completion.
    pub model: String,
    /// A list of chat completion choices. Can be more than one if `n_choice` is greater than 1.
    pub choices: Vec<ChatCompletionObjectChoice>,
    /// Usage statistics for the completion request.
    pub usage: Usage,
}

#[test]
fn test_deserialize_chat_completion_object() {
    let json = r#"{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699896916,
  "model": "gpt-3.5-turbo-0125",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\n\"location\": \"Boston, MA\"\n}"
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 82,
    "completion_tokens": 17,
    "total_tokens": 99
  }
}"#;

    let chatcmp_object: ChatCompletionObject = serde_json::from_str(json).unwrap();
    assert_eq!(chatcmp_object.id, "chatcmpl-abc123");
    assert_eq!(chatcmp_object.object, "chat.completion");
    assert_eq!(chatcmp_object.created, 1699896916);
    assert_eq!(chatcmp_object.model, "gpt-3.5-turbo-0125");
    assert_eq!(chatcmp_object.choices.len(), 1);
    assert_eq!(chatcmp_object.choices[0].index, 0);
    assert_eq!(
        chatcmp_object.choices[0].finish_reason,
        FinishReason::tool_calls
    );
    assert_eq!(chatcmp_object.choices[0].message.tool_calls.len(), 1);
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0].id,
        "call_abc123"
    );
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0].ty,
        "function"
    );
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0]
            .function
            .name,
        "get_current_weather"
    );
    assert_eq!(
        chatcmp_object.choices[0].message.tool_calls[0]
            .function
            .arguments,
        "{\n\"location\": \"Boston, MA\"\n}"
    );
    assert_eq!(chatcmp_object.usage.prompt_tokens, 82);
    assert_eq!(chatcmp_object.usage.completion_tokens, 17);
    assert_eq!(chatcmp_object.usage.total_tokens, 99);
}

/// Represents a chat completion choice returned by model.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionObjectChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion message generated by the model.
    pub message: ChatCompletionObjectMessage,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: FinishReason,
    /// Log probability information for the choice.
    pub logprobs: Option<LogProbs>,
}

#[test]
fn test_serialize_chat_completion_object_choice() {
    let tool = ToolCall {
        id: "call_abc123".to_string(),
        ty: "function".to_string(),
        function: Function {
            name: "get_current_weather".to_string(),
            arguments: "{\"location\": \"Boston, MA\"}".to_string(),
        },
    };
    let message = ChatCompletionObjectMessage {
        content: None,
        tool_calls: vec![tool],
        role: ChatCompletionRole::Assistant,
        function_call: None,
    };
    let choice = ChatCompletionObjectChoice {
        index: 0,
        message,
        finish_reason: FinishReason::tool_calls,
        logprobs: None,
    };
    let json = serde_json::to_string(&choice).unwrap();
    assert_eq!(
        json,
        r#"{"index":0,"message":{"content":null,"tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Boston, MA\"}"}}],"role":"assistant"},"finish_reason":"tool_calls","logprobs":null}"#
    );
}

/// Log probability information for the choice.
#[derive(Debug, Deserialize, Serialize)]
pub struct LogProbs;

/// Represents a chat completion message generated by the model.
#[derive(Debug, Serialize)]
pub struct ChatCompletionObjectMessage {
    /// The contents of the message.
    pub content: Option<String>,
    /// The tool calls generated by the model, such as function calls.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The role of the author of this message.
    pub role: ChatCompletionRole,
    /// Deprecated. The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatMessageFunctionCall>,
}
impl<'de> Deserialize<'de> for ChatCompletionObjectMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChatCompletionObjectMessageVisitor;

        impl<'de> Visitor<'de> for ChatCompletionObjectMessageVisitor {
            type Value = ChatCompletionObjectMessage;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ChatCompletionObjectMessage")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ChatCompletionObjectMessage, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut content = None;
                let mut tool_calls = None;
                let mut role = None;
                let mut function_call = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "content" => content = map.next_value()?,
                        "tool_calls" => tool_calls = map.next_value()?,
                        "role" => role = map.next_value()?,
                        "function_call" => function_call = map.next_value()?,
                        _ => return Err(de::Error::unknown_field(key.as_str(), FIELDS)),
                    }
                }

                let content = content;
                let tool_calls = tool_calls.unwrap_or_default();
                let role = role.ok_or_else(|| de::Error::missing_field("role"))?;
                let function_call = function_call;

                Ok(ChatCompletionObjectMessage {
                    content,
                    tool_calls,
                    role,
                    function_call,
                })
            }
        }

        const FIELDS: &[&str] = &["content", "tool_calls", "role", "function_call"];
        deserializer.deserialize_struct(
            "ChatCompletionObjectMessage",
            FIELDS,
            ChatCompletionObjectMessageVisitor,
        )
    }
}

#[test]
fn test_serialize_chat_completion_object_message() {
    let tool = ToolCall {
        id: "call_abc123".to_string(),
        ty: "function".to_string(),
        function: Function {
            name: "get_current_weather".to_string(),
            arguments: "{\"location\": \"Boston, MA\"}".to_string(),
        },
    };
    let message = ChatCompletionObjectMessage {
        content: None,
        tool_calls: vec![tool],
        role: ChatCompletionRole::Assistant,
        function_call: None,
    };
    let json = serde_json::to_string(&message).unwrap();
    assert_eq!(
        json,
        r#"{"content":null,"tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Boston, MA\"}"}}],"role":"assistant"}"#
    );
}

#[test]
fn test_deserialize_chat_completion_object_message() {
    {
        let json = r#"{"content":null,"tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"Boston, MA\"}"}}],"role":"assistant"}"#;
        let message: ChatCompletionObjectMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.content, None);
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.role, ChatCompletionRole::Assistant);
    }

    {
        let json = r#"{"content":null,"role":"assistant"}"#;
        let message: ChatCompletionObjectMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.content, None);
        assert!(message.tool_calls.is_empty());
        assert_eq!(message.role, ChatCompletionRole::Assistant);
    }
}

/// The name and arguments of a function that should be called, as generated by the model.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessageFunctionCall {
    /// The name of the function to call.
    pub name: String,

    /// The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
    pub arguments: String,
}

/// Represents a streamed chunk of a chat completion response returned by model, based on the provided input.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunk {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// A list of chat completion choices. Can be more than one if `n_choice` is greater than 1.
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The model used for the chat completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.
    pub system_fingerprint: String,
    /// The object type, which is always `chat.completion.chunk`.
    pub object: String,
    /// Usage statistics for the completion request.
    ///
    /// An optional field that will only be present when you set stream_options: {"include_usage": true} in your request. When present, it contains a null value except for the last chunk which contains the token usage statistics for the entire request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[test]
fn test_serialize_chat_completion_chunk() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkChoiceDelta {
                content: Some(".".to_owned()),
                tool_calls: vec![],
                role: ChatCompletionRole::Assistant,
            },
            logprobs: None,
            finish_reason: None,
        }],
        created: 1722433423,
        model: "default".to_string(),
        system_fingerprint: "fp_44709d6fcb".to_string(),
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };

    let json = serde_json::to_string(&chunk).unwrap();
    assert_eq!(
        json,
        r#"{"id":"chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a","choices":[{"index":0,"delta":{"content":".","role":"assistant"},"logprobs":null,"finish_reason":null}],"created":1722433423,"model":"default","system_fingerprint":"fp_44709d6fcb","object":"chat.completion.chunk"}"#
    );
}

#[test]
fn test_deserialize_chat_completion_chunk() {
    {
        let json = r#"{"id":"chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a","choices":[{"index":0,"delta":{"content":".","role":"assistant"},"logprobs":null,"finish_reason":null}],"created":1722433423,"model":"default","system_fingerprint":"fp_44709d6fcb","object":"chat.completion.chunk"}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id, "chatcmpl-1d0ff773-e8ab-4254-a222-96e97e3c295a");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(chunk.choices[0].delta.content, Some(".".to_owned()));
        assert!(chunk.choices[0].delta.tool_calls.is_empty());
        assert_eq!(chunk.choices[0].delta.role, ChatCompletionRole::Assistant);
        assert_eq!(chunk.created, 1722433423);
        assert_eq!(chunk.model, "default");
        assert_eq!(chunk.system_fingerprint, "fp_44709d6fcb");
        assert_eq!(chunk.object, "chat.completion.chunk");
    }
}

/// Represents a chat completion choice in a streamed chunk of a chat completion response.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionChunkChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion delta generated by streamed model responses.
    pub delta: ChatCompletionChunkChoiceDelta,
    /// Log probability information for the choice.
    pub logprobs: Option<LogProbs>,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `function_call` if the model called a function.
    pub finish_reason: Option<FinishReason>,
}

/// Represents a chat completion delta generated by streamed model responses.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoiceDelta {
    /// The contents of the chunk message.
    pub content: Option<String>,
    /// The name and arguments of a function that should be called, as generated by the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallForChunk>,
    /// The role of the author of this message.
    pub role: ChatCompletionRole,
}
impl<'de> Deserialize<'de> for ChatCompletionChunkChoiceDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChatCompletionChunkChoiceDeltaVisitor;

        impl<'de> Visitor<'de> for ChatCompletionChunkChoiceDeltaVisitor {
            type Value = ChatCompletionChunkChoiceDelta;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ChatCompletionChunkChoiceDelta")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ChatCompletionChunkChoiceDelta, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut content = None;
                let mut tool_calls = None;
                let mut role = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "content" => content = map.next_value()?,
                        "tool_calls" => tool_calls = map.next_value()?,
                        "role" => role = map.next_value()?,
                        _ => return Err(de::Error::unknown_field(key.as_str(), FIELDS)),
                    }
                }

                let content = content;
                let tool_calls = tool_calls.unwrap_or_default();
                let role = role.ok_or_else(|| de::Error::missing_field("role"))?;
                Ok(ChatCompletionChunkChoiceDelta {
                    content,
                    tool_calls,
                    role,
                })
            }
        }

        const FIELDS: &[&str] = &["content", "tool_calls", "role"];
        deserializer.deserialize_struct(
            "ChatCompletionChunkChoiceDelta",
            FIELDS,
            ChatCompletionChunkChoiceDeltaVisitor,
        )
    }
}
