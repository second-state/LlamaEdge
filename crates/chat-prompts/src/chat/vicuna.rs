use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use base64::{engine::general_purpose, Engine as _};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart,
};
use image::io::Reader as ImageReader;
use std::io::Cursor;

/// Vicuna-1.0 Prompt Template
#[derive(Debug, Default, Clone)]
pub struct VicunaChatPrompt;
impl VicunaChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."),
            false => content.to_string(),
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
                "{system_prompt} USER: {user_message}",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history} USER: {user_message}",
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
            "{chat_history} ASSISTANT: {assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for VicunaChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."),
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

        prompt.push_str(" ASSISTANT:");

        Ok(prompt)
    }
}

/// Vicuna-1.1 Prompt Template
#[derive(Debug, Default, Clone)]
pub struct Vicuna11ChatPrompt;
impl Vicuna11ChatPrompt {
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
            true => format!("USER: {user_message}", user_message = content.trim(),),
            false => format!(
                "{chat_history}\nUSER: {user_message}",
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
            "{chat_history}\nASSISTANT: {assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for Vicuna11ChatPrompt {
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
        prompt.push_str(" ASSISTANT:");

        Ok(prompt)
    }
}

/// Vicuna-1.0 Prompt Template
#[derive(Debug, Default, Clone)]
pub struct VicunaLlavaPrompt;
impl VicunaLlavaPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe."),
            false => content.to_string(),
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> Result<String> {
        let prompt = match message.content() {
            ChatCompletionUserMessageContent::Text(content) => {
                match chat_history.as_ref().is_empty() {
                    true => format!(
                        "{system_prompt}\nUSER: {user_message}",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    ),
                    false => format!(
                        "{chat_history}\nUSER: {user_message}",
                        chat_history = chat_history.as_ref().trim(),
                        user_message = content.trim(),
                    ),
                }
            }
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                let mut image_content = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text(text_content) => {
                            content.push_str(text_content.text());
                            content.push('\n');
                        }
                        ContentPart::Image(part) => {
                            image_content = match part.image().is_url() {
                                true => String::from("<image>"),
                                false => {
                                    let base64_str = part.image().url.as_str();
                                    let format = is_image_format(base64_str)?;
                                    format!(
                                        r#"<img src="data:image/{};base64,{}">"#,
                                        format, base64_str
                                    )
                                }
                            };
                        }
                    }
                }

                match chat_history.as_ref().is_empty() {
                    true => format!(
                        "{system_prompt}\nUSER:{image_embeddings}\n{user_message}",
                        system_prompt = system_prompt.as_ref().trim(),
                        image_embeddings = image_content.trim(),
                        user_message = content.trim(),
                    ),
                    false => format!(
                        "{chat_history}\nUSER:{image_embeddings}\n{user_message}",
                        chat_history = chat_history.as_ref().trim(),
                        image_embeddings = image_content.trim(),
                        user_message = content.trim(),
                    ),
                }
            }
        };

        Ok(prompt)
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
            "{chat_history}\nASSISTANT: {assistant_message}",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for VicunaLlavaPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe."),
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message)?;
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\nASSISTANT:");

        Ok(prompt)
    }
}

fn is_image_format(base64_str: &str) -> Result<String> {
    let image_data = match general_purpose::STANDARD.decode(base64_str) {
        Ok(data) => data,
        Err(_) => {
            return Err(PromptError::Operation(
                "Failed to decode base64 string.".to_string(),
            ))
        }
    };

    let format = ImageReader::new(Cursor::new(image_data))
        .with_guessed_format()
        .unwrap()
        .format();

    let image_format = match format {
        Some(image::ImageFormat::Png) => "png".to_string(),
        Some(image::ImageFormat::Jpeg) => "jpeg".to_string(),
        Some(image::ImageFormat::Tga) => "tga".to_string(),
        Some(image::ImageFormat::Bmp) => "bmp".to_string(),
        Some(image::ImageFormat::Gif) => "gif".to_string(),
        Some(image::ImageFormat::Hdr) => "hdr".to_string(),
        Some(image::ImageFormat::Pnm) => "pnm".to_string(),
        _ => {
            return Err(PromptError::Operation(
                "Unsupported image format.".to_string(),
            ))
        }
    };

    Ok(image_format)
}
