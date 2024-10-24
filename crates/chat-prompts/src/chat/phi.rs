use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart,
};

/// Generate instruct prompt for the `microsoft/phi-2` model.
#[derive(Debug, Default, Clone)]
pub struct Phi2InstructPrompt;
impl Phi2InstructPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(&self, message: &ChatCompletionUserMessage) -> String {
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

        format!("Instruct: {user_message}", user_message = content.trim(),)
    }
}
impl BuildChatPrompt for Phi2InstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        let mut prompt = if let Some(user_message) = messages.last() {
            match user_message {
                ChatCompletionRequestMessage::User(message) => self.append_user_message(message),
                _ => {
                    return Err(crate::error::PromptError::NoUserMessage);
                }
            }
        } else {
            return Err(crate::error::PromptError::NoMessages);
        };

        prompt.push_str("\nOutput:");

        Ok(prompt)
    }
}

/// Generate chat prompt for the `microsoft/phi-2` model.
#[derive(Debug, Default, Clone)]
pub struct Phi2ChatPrompt;
impl Phi2ChatPrompt {
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
            true => format!("Alice: {user_message}", user_message = content.trim(),),
            false => format!(
                "{chat_history}\nAlice: {user_message}",
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
            "{chat_history}\nBob: {assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for Phi2ChatPrompt {
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

        prompt.push_str("\nBob:");

        Ok(prompt)
    }
}

/// Generate chat prompt for the `microsoft/phi-3` model.
#[derive(Debug, Default, Clone)]
pub struct Phi3InstructPrompt;
impl Phi3InstructPrompt {
    /// Create a user prompt from a chat completion request message.
    fn append_user_message(&self, message: &ChatCompletionUserMessage) -> String {
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

        format!(
            "<|user|>\n {user_message} <|end|>",
            user_message = content.trim(),
        )
    }
}
impl BuildChatPrompt for Phi3InstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        let mut prompt = if let Some(user_message) = messages.last() {
            match user_message {
                ChatCompletionRequestMessage::User(message) => self.append_user_message(message),
                _ => {
                    return Err(crate::error::PromptError::NoUserMessage);
                }
            }
        } else {
            return Err(crate::error::PromptError::NoMessages);
        };

        prompt.push_str("\n <|assistant|>");

        Ok(prompt)
    }
}

/// Generate chat prompt for the `microsoft/phi-2` model.
#[derive(Debug, Default, Clone)]
pub struct Phi3ChatPrompt;
impl Phi3ChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|system|>\nYou are a friendly chatbot.<|end|>"),
            false => format!("<|system|>\n{content}<|end|>"),
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
                "{system_prompt}\n<|user|>\n{user_message}<|end|>",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}\n<|user|>\n{user_message}<|end|>",
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
            "{chat_history}\n<|assistant|>\n{assistant_message}<|end|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for Phi3ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => self.create_system_prompt(message),
            _ => String::from("<|system|>\nYou are a friendly chatbot.<|end|>"),
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

        prompt.push_str("\n<|assistant|>");

        Ok(prompt)
    }
}
