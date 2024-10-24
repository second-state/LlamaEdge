use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionToolMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart, Tool,
};

/// Generate prompts for the `nemotron-mini-instruct` model.
#[derive(Debug, Default, Clone)]
pub struct NemotronChatPrompt;
impl NemotronChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<extra_id_0>System\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe."),
            false =>format!(
                "<extra_id_0>System\n{system_prompt}", system_prompt=content
            )
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
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
                "{system_prompt}\n<extra_id_1>User\n{user_message}",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}\n<extra_id_1>User\n{user_message}",
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
            "{chat_history}<extra_id_1>Assistant\n{assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for NemotronChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("<extra_id_0>System\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe."),
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<extra_id_1>Assistant\n");

        Ok(prompt)
    }
}

/// Generate prompts for the models using ChatML template.
#[derive(Debug, Default, Clone)]
pub struct NemotronToolPrompt;
impl NemotronToolPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|im_start|>system\nAnswer as concisely as possible.<|im_end|>"),
            false => format!(
                "<|im_start|>system\n{system_prompt}<|im_end|>",
                system_prompt = content
            ),
        }
    }

    fn create_system_prompt_tool(
        &self,
        message: &ChatCompletionSystemMessage,
        tools: Option<&[Tool]>,
    ) -> String {
        let content = message.content();
        match content.is_empty() {
            true => match tools {
                Some(tools) => {
                    let available_tools = serde_json::to_string(tools).unwrap();
                    let tools = format!("<tool> {} </tool>", available_tools);

                    let begin = r#"<extra_id_0>System\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe."#;

                    format!("{}\n\n{}", begin, tools)
                }
                None => {
                    String::from("<extra_id_0>System\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe.")
                }
            },
            false => match tools {
                Some(tools) => {
                    let available_tools = serde_json::to_string(tools).unwrap();
                    let tools = format!("<tool> {} </tool>", available_tools);

                    let begin = format!(
                        "<extra_id_0>System\n{system_prompt}", system_prompt=content
                    );

                    format!("{}\n\n{}", begin, tools)
                }
                None => {
                    format!(
                        "<extra_id_0>System\n{system_prompt}", system_prompt=content
                    )
                }
            },
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
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
                "{system_prompt}\n\n<extra_id_1>User\n{user_message}",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}\n<extra_id_1>User\n{user_message}",
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
            "{chat_history}<extra_id_1>Assistant\n{assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }

    /// create a tool prompt from a chat completion request message.
    fn append_tool_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionToolMessage,
    ) -> String {
        format!(
            "{chat_history}\n<extra_id_1>Tool\n{tool_message}",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for NemotronToolPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => self.create_system_prompt(message),
            _ => String::from("<|im_start|>system\nAnswer as concisely as possible.<|im_end|>"),
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
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

        prompt.push_str("\n<|im_start|>assistant");

        Ok(prompt)
    }

    fn build_with_tools(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt_tool(message, tools)
            }
            _ => match tools {
                Some(tools) => {
                    let mut tools_s = String::new();
                    for tool in tools {
                        let available_tool = serde_json::to_string(&tool.function).unwrap();

                        let tool = format!("<tool> {} </tool>\n", available_tool);

                        tools_s.push_str(&tool);
                    }

                    let begin = r#"<extra_id_0>System\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe."#;

                    format!("{}\n{}", begin, tools_s.trim())
                }
                None => {
                    String::from("<|im_start|>system\nAnswer as concisely as possible.<|im_end|>")
                }
            },
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
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

        prompt.push_str("\n<extra_id_1>Assistant\n");

        Ok(prompt)
    }
}
