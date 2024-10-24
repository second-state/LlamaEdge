use endpoints::chat::ChatCompletionRole;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, PromptError>;

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum PromptError {
    #[error("There must be at least one user message to create a prompt from.")]
    NoMessages,
    #[error("No user message to create prompt from.")]
    NoUserMessage,
    #[error("No content in the assistant message when the `tool_calls` is not specified.")]
    NoAssistantMessage,
    #[error("No available tools to create prompt from.")]
    NoAvailableTools,
    #[error("Bad messages. {0}")]
    BadMessages(String),
    #[error("Unknown chat completion role: {0:?}")]
    UnknownRole(ChatCompletionRole),
    #[error("Unknown prompt template type: {0}")]
    UnknownPromptTemplateType(String),
    #[error("Failed to build prompt. Reason: {0}")]
    Operation(String),
}
