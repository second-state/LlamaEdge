//! Define common types used by other types.
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[allow(non_camel_case_types)]
pub enum LlamaCppLogitBiasType {
    input_ids,
    tokens,
}

/// Token usage
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u64,
    /// Number of tokens in the generated completion.
    pub completion_tokens: u64,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: u64,
}

/// The reason the model stopped generating tokens.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum FinishReason {
    /// `stop` if the model hit a natural stop point or a provided stop sequence.
    stop,
    /// `length` if the maximum number of tokens specified in the request was reached.
    length,
    /// `tool_calls` if the model called a tool.
    tool_calls,
}
