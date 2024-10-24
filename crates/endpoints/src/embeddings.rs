//! Define types for the `embeddings` endpoint.

use crate::common::Usage;
use serde::{Deserialize, Serialize};

/// Creates an embedding vector representing the input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// ID of the model to use.
    pub model: String,
    /// Input text to embed,encoded as a string or array of tokens.
    ///
    /// To embed multiple inputs in a single request, pass an array of strings or array of token arrays. The input must not exceed the max input tokens for the model (8192 tokens for text-embedding-ada-002), cannot be an empty string, and any array must be 2048 dimensions or less.
    pub input: InputText,
    /// The format to return the embeddings in. Can be either float or base64.
    /// Defaults to float.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[test]
fn test_embedding_serialize_embedding_request() {
    let embedding_request = EmbeddingRequest {
        model: "text-embedding-ada-002".to_string(),
        input: "Hello, world!".into(),
        encoding_format: None,
        user: None,
    };
    let serialized = serde_json::to_string(&embedding_request).unwrap();
    assert_eq!(
        serialized,
        r#"{"model":"text-embedding-ada-002","input":"Hello, world!"}"#
    );

    let embedding_request = EmbeddingRequest {
        model: "text-embedding-ada-002".to_string(),
        input: vec!["Hello, world!", "This is a test string"].into(),
        encoding_format: None,
        user: None,
    };
    let serialized = serde_json::to_string(&embedding_request).unwrap();
    assert_eq!(
        serialized,
        r#"{"model":"text-embedding-ada-002","input":["Hello, world!","This is a test string"]}"#
    );
}

#[test]
fn test_embedding_deserialize_embedding_request() {
    let serialized = r#"{"model":"text-embedding-ada-002","input":"Hello, world!"}"#;
    let embedding_request: EmbeddingRequest = serde_json::from_str(serialized).unwrap();
    assert_eq!(embedding_request.model, "text-embedding-ada-002");
    assert_eq!(embedding_request.input, InputText::from("Hello, world!"));
    assert_eq!(embedding_request.encoding_format, None);
    assert_eq!(embedding_request.user, None);

    let serialized =
        r#"{"model":"text-embedding-ada-002","input":["Hello, world!","This is a test string"]}"#;
    let embedding_request: EmbeddingRequest = serde_json::from_str(serialized).unwrap();
    assert_eq!(embedding_request.model, "text-embedding-ada-002");
    assert_eq!(
        embedding_request.input,
        InputText::from(vec!["Hello, world!", "This is a test string"])
    );
    assert_eq!(embedding_request.encoding_format, None);
    assert_eq!(embedding_request.user, None);
}

/// Defines the input text for the embedding request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum InputText {
    /// The string that will be turned into an embedding.
    String(String),
    /// The array of strings that will be turned into an embedding.
    ArrayOfStrings(Vec<String>),
    /// The array of integers that will be turned into an embedding.
    ArrayOfTokens(Vec<i64>),
    /// The array of arrays containing integers that will be turned into an embedding.
    ArrayOfTokenArrays(Vec<Vec<i64>>),
}
impl From<&str> for InputText {
    fn from(s: &str) -> Self {
        InputText::String(s.to_string())
    }
}
impl From<&String> for InputText {
    fn from(s: &String) -> Self {
        InputText::String(s.to_string())
    }
}
impl From<&[String]> for InputText {
    fn from(s: &[String]) -> Self {
        InputText::ArrayOfStrings(s.to_vec())
    }
}
impl From<Vec<&str>> for InputText {
    fn from(s: Vec<&str>) -> Self {
        InputText::ArrayOfStrings(s.iter().map(|s| s.to_string()).collect())
    }
}
impl From<Vec<String>> for InputText {
    fn from(s: Vec<String>) -> Self {
        InputText::ArrayOfStrings(s)
    }
}
impl From<&[i64]> for InputText {
    fn from(s: &[i64]) -> Self {
        InputText::ArrayOfTokens(s.to_vec())
    }
}
impl From<Vec<i64>> for InputText {
    fn from(s: Vec<i64>) -> Self {
        InputText::ArrayOfTokens(s)
    }
}
impl From<Vec<Vec<i64>>> for InputText {
    fn from(s: Vec<Vec<i64>>) -> Self {
        InputText::ArrayOfTokenArrays(s)
    }
}

/// Defines the embedding response.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

/// Represents an embedding vector returned by embedding endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingObject {
    /// The index of the embedding in the list of embeddings.
    pub index: u64,
    /// The object type, which is always "embedding".
    pub object: String,
    /// The embedding vector, which is a list of floats.
    pub embedding: Vec<f64>,
}
