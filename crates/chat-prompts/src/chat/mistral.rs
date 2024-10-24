use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionToolMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart, Tool,
};

/// Generate prompts for the `Mistral-instruct` model.
#[derive(Debug, Default, Clone)]
pub struct MistralInstructPrompt;
impl MistralInstructPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => format!(
                "<s>[INST] {user_message} [/INST]",
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}[INST] {user_message} [/INST]",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            // Note that the content is optional if `tool_calls` is specified.
            None => match message.tool_calls().is_some() {
                true => String::new(),
                false => return Err(PromptError::NoAssistantMessage),
            },
        };

        Ok(format!(
            "{chat_history}{assistant_message}</s>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for MistralInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        Ok(prompt)
    }
}

/// Generate prompts for the amazon `MistralLite-7B` model.
#[derive(Debug, Default, Clone)]
pub struct MistralLitePrompt;
impl MistralLitePrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => format!(
                "<|prompter|>{user_message}</s>",
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}<|prompter|>{user_message}</s>",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            // Note that the content is optional if `tool_calls` is specified.
            None => match message.tool_calls().is_some() {
                true => String::new(),
                false => return Err(PromptError::NoAssistantMessage),
            },
        };

        Ok(format!(
            "{chat_history}<|assistant|>{assistant_message}</s>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for MistralLitePrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        // append user/assistant messages
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("<|assistant|>");

        Ok(prompt)
    }
}

/// Generate prompts for the `Mistral-instruct` model.
#[derive(Debug, Default, Clone)]
pub struct MistralToolPrompt;
impl MistralToolPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => format!(
                "<s>[INST] {user_message} [/INST]",
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}[INST] {user_message} [/INST]",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message_tool(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
        tools: Option<&[Tool]>,
        last_user_message: bool,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => match last_user_message {
                true => match tools {
                    Some(tools) => {
                        let json = serde_json::to_string(tools).unwrap();

                        format!(
                            "<s>[AVAILABLE_TOOLS] {available_tools}[/AVAILABLE_TOOLS][INST] {user_message}[/INST]",
                            available_tools = json,
                            user_message = content.trim(),
                        )
                    }
                    None => format!(
                        "<s>[INST] {user_message} [/INST]",
                        user_message = content.trim(),
                    ),
                },
                false => format!(
                    "{chat_history}[INST] {user_message} [/INST]",
                    chat_history = chat_history.as_ref().trim(),
                    user_message = content.trim(),
                ),
            },
            false => match last_user_message {
                true => match tools {
                    Some(tools) => {
                        let json = serde_json::to_string(tools).unwrap();

                        format!(
                            "{chat_history}[AVAILABLE_TOOLS] {available_tools}[/AVAILABLE_TOOLS][INST] {user_message}[/INST]",
                            chat_history = chat_history.as_ref().trim(),
                            available_tools = json,
                            user_message = content.trim(),
                        )
                    }
                    None => format!(
                        "{chat_history}[INST] {user_message} [/INST]",
                        chat_history = chat_history.as_ref().trim(),
                        user_message = content.trim(),
                    ),
                },
                false => format!(
                    "{chat_history}[INST] {user_message} [/INST]",
                    chat_history = chat_history.as_ref().trim(),
                    user_message = content.trim(),
                ),
            },
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            // Note that the content is optional if `tool_calls` is specified.
            None => match message.tool_calls().is_some() {
                true => String::new(),
                false => return Err(PromptError::NoAssistantMessage),
            },
        };

        let content = content.split("\n").next().unwrap_or_default();

        Ok(format!(
            "{chat_history}{assistant_message}</s>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }

    fn append_tool_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionToolMessage,
    ) -> String {
        format!(
            "{chat_history}[TOOL_RESULTS]{tool_result}[/TOOL_RESULTS]",
            chat_history = chat_history.as_ref().trim(),
            tool_result = message.content().trim()
        )
    }
}
impl BuildChatPrompt for MistralToolPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::Tool(message) => {
                    prompt = self.append_tool_message(&prompt, message);
                }
                _ => continue,
            }
        }

        Ok(prompt)
    }

    fn build_with_tools(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        tools: Option<&[endpoints::chat::Tool]>,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // append user/assistant messages
        let mut prompt = String::new();
        for (idx, message) in messages.iter().enumerate() {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    let last = idx == messages.len() - 1;
                    prompt = self.append_user_message_tool(&prompt, message, tools, last);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::Tool(message) => {
                    prompt = self.append_tool_message(&prompt, message);
                }
                _ => continue,
            }
        }

        Ok(prompt)
    }
}
