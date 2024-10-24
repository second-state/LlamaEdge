use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionToolMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart, Tool,
};

/// Generate prompts for the models using ChatML template.
#[derive(Debug, Default, Clone)]
pub struct ChatMLPrompt;
impl ChatMLPrompt {
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
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<|im_start|>user\n{user_message}<|im_end|>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n<|im_start|>user\n{user_message}<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user\n{user_message}<|im_end|>",
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
            "{chat_history}\n<|im_start|>assistant\n{assistant_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }

    /// create a tool prompt from a chat completion request message.
    fn append_tool_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionToolMessage,
    ) -> Result<String> {
        let content = message.content();

        Ok(format!(
            "{chat_history}\n<|im_start|>tool\n{tool_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for ChatMLPrompt {
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
                    prompt = self.append_tool_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<|im_start|>assistant");

        Ok(prompt)
    }
}

/// Generate prompts for the models using ChatML template.
#[derive(Debug, Default, Clone)]
pub struct ChatMLToolPrompt;
impl ChatMLToolPrompt {
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
                    let tools = format!("<tools> {} </tools>", available_tools);

                    let begin = r#"<|im_start|>system\nYou are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:"#;

                    let end = r#"Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call><|im_end|>"#;

                    format!("{} {} {}", begin, tools, end)
                }
                None => {
                    String::from("<|im_start|>system\nAnswer as concisely as possible.<|im_end|>")
                }
            },
            false => match tools {
                Some(tools) => {
                    let available_tools = serde_json::to_string(tools).unwrap();
                    let tools = format!("<tools> {} </tools>", available_tools);

                    let begin = format!(
                            "<|im_start|>system\n{system_prompt}\nYou are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:",
                            system_prompt = content
                        );

                    let end = r#"Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call><|im_end|>"#;

                    format!("{} {} {}", begin, tools, end)
                }
                None => {
                    format!(
                        "<|im_start|>system\n{system_prompt}<|im_end|>",
                        system_prompt = content
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
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<|im_start|>user\n{user_message}<|im_end|>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n<|im_start|>user\n{user_message}<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user\n{user_message}<|im_end|>",
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
            "{chat_history}\n<|im_start|>assistant\n{assistant_message}<|im_end|>",
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
            "{chat_history}\n<|im_start|>tool\n<tool_response>\n{tool_message}\n</tool_response>\n<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for ChatMLToolPrompt {
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
                    let available_tools = serde_json::to_string(tools).unwrap();
                    let tools = format!("<tools> {} </tools>", available_tools);

                    let begin = r#"<|im_start|>system\nYou are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:"#;

                    let end = r#"Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call><|im_end|>"#;

                    format!("{} {} {}", begin, tools, end)
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

        prompt.push_str("\n<|im_start|>assistant");

        Ok(prompt)
    }
}

/// Generate prompts for InternLM-2.5 models in tool use scenario.
pub struct InternLM2ToolPrompt;
impl InternLM2ToolPrompt {
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
                    let begin = "<|im_start|>system\nYou are InternLM2-Chat, a harmless AI assistant.<|im_end|>";

                    let available_tools = serde_json::to_string_pretty(tools).unwrap();
                    let tools = format!("<|im_start|>system name=<|plugin|>\n{}\n<|im_end|>", available_tools);

                    format!("{}\n{}", begin, tools)
                }
                None => {
                    String::from("<|im_start|>system\nYou are InternLM2-Chat, a harmless AI assistant.<|im_end|>")
                }
            },
            false => match tools {
                Some(tools) => {
                    let begin = format!("<|im_start|>system\n{}<|im_end|>", content);

                    let available_tools = serde_json::to_string_pretty(tools).unwrap();
                    let tools = format!("<|im_start|>system name=<|plugin|>\n{}\n<|im_end|>", available_tools);

                    format!("{}\n{}", begin, tools)
                }
                None => {
                    format!("<|im_start|>system\n{}<|im_end|>", content)
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
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<|im_start|>user\n{user_message}<|im_end|>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n<|im_start|>user\n{user_message}<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user\n{user_message}<|im_end|>",
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
            "{chat_history}\n<|im_start|>assistant\n{assistant_message}<|im_end|>",
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
            "{chat_history}\n<|im_start|>environment name=<|plugin|>\n{tool_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for InternLM2ToolPrompt {
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
                    let begin = "<|im_start|>system\nYou are InternLM2-Chat, a harmless AI assistant.<|im_end|>";

                    let available_tools = serde_json::to_string_pretty(tools).unwrap();
                    let tools = format!("<|im_start|>system name=<|plugin|>\n{}\n<|im_end|>", available_tools);

                    format!("{}\n{}", begin, tools)
                }
                None => {
                    String::from("<|im_start|>system\nYou are InternLM2-Chat, a harmless AI assistant.<|im_end|>")
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

        prompt.push_str("\n<|im_start|>assistant");

        Ok(prompt)
    }
}
