use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionToolMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart, Tool,
};

/// Generate prompts for the `Llama-2-chat` model.
#[derive(Debug, Default, Clone)]
pub struct Llama2ChatPrompt;
impl Llama2ChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>"),
            false =>format!(
                "<<SYS>>\n{content} <</SYS>>"
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
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<s>[INST] {user_message} [/INST]",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "<s>[INST] {system_prompt}\n\n{user_message} [/INST]",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}<s>[INST] {user_message} [/INST]",
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
            "{prompt} {assistant_message} </s>",
            prompt = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for Llama2ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>"),
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

        Ok(prompt)
    }
}

/// Generate prompts for the `Codellama-instruct` model.
#[derive(Debug, Default, Clone)]
pub struct CodeLlamaInstructPrompt;
impl CodeLlamaInstructPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<<SYS>>\nWrite code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```: <</SYS>>"),
            false => format!(
                "<<SYS>>\n{system_prompt} <</SYS>>", system_prompt=content
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
                "<s>[INST] {system_prompt}\n\n{user_message} [/INST]",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}<s>[INST] {user_message} [/INST]",
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
            "{prompt} {assistant_message} </s>",
            prompt = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for CodeLlamaInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("<<SYS>>\nWrite code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```: <</SYS>>"),
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

        Ok(prompt)
    }
}

/// Generate prompts for the `Codellama-70b-instruct-hf` model.
#[derive(Debug, Default, Clone)]
pub struct CodeLlamaSuperInstructPrompt;
impl CodeLlamaSuperInstructPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<s>Source: system\n\n Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```: <step>"),
            false => format!(
                "<s>Source: system\n\n {content} <step>"
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
                "{system_prompt} Source: user\n\n {user_message} <step>",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history} Source: user\n\n {user_message} <step>",
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
            "{prompt} Source: assistant\n\n {assistant_message} <step>",
            prompt = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for CodeLlamaSuperInstructPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("<s>Source: system\n\n Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```: <step>")
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

        prompt.push_str(" Source: assistant\nDestination: user\n\n ");

        Ok(prompt)
    }
}

/// Generate prompts for the `Llama-3-chat` model.
///
/// Reference: <https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/>
#[derive(Debug, Default, Clone)]
pub struct Llama3ChatPrompt;
impl Llama3ChatPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe.<|eot_id|>"),
            false =>format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>", system_prompt=content
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
                "{system_prompt}<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>",
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
            "{chat_history} <|start_header_id|>assistant<|end_header_id|>\n\n{assistant_message}<|eot_id|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }
}
impl BuildChatPrompt for Llama3ChatPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe.<|eot_id|>"),
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

        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>");

        Ok(prompt)
    }
}

/// Generate prompts for the `Llama-3.1-instruct` model.
///
/// Reference: <https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling>
#[derive(Debug, Default, Clone)]
pub struct Llama3ToolPrompt;
impl Llama3ToolPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe.<|eot_id|>"),
            false =>format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>", system_prompt=content
            )
        }
    }

    /// Create a system prompt for tool use.
    fn create_system_prompt_tool(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal use question.<|eot_id|>"),
            false =>format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>", system_prompt=content
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
                "{system_prompt}<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>",
                system_prompt = system_prompt.as_ref().trim(),
                user_message = content.trim(),
            ),
            false => format!(
                "{chat_history}<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// create a user prompt for tool use.
    fn append_user_message_tool(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
        tools: impl AsRef<[Tool]>,
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
            true => {
                let json = serde_json::to_string(tools.as_ref()).unwrap();

                format!(
                    "{system_prompt}<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {format}. Do not use variables.\n\n{available_tools}\n\nQuestion: {user_message}<|eot_id|>",
                    system_prompt = system_prompt.as_ref().trim(),
                    format = r#"{"name": function name, "parameters": dictionary of argument name and its value}"#,
                    available_tools = json,
                    user_message = content.trim(),
                )
            }
            false => {
                let json = serde_json::to_string(tools.as_ref()).unwrap();

                format!(
                    "{chat_history}<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {format}. Do not use variables.\n\n{available_tools}\n\nQuestion: {user_message}<|eot_id|>",
                    chat_history = chat_history.as_ref().trim(),
                    format = r#"{"name": function name, "parameters": dictionary of argument name and its value}"#,
                    available_tools = json,
                    user_message = content.trim(),
                )
            }
        }
    }

    /// Create an assistant prompt from a chat completion request message.
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
            "{chat_history} <|start_header_id|>assistant<|end_header_id|>\n\n{assistant_message}<|eot_id|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }

    /// Create a tool prompt.
    fn append_tool_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionToolMessage,
    ) -> String {
        format!(
            "{chat_history}<|start_header_id|>ipython<|end_header_id|>\n\n{tool_result}<|eot_id|>",
            chat_history = chat_history.as_ref().trim(),
            tool_result = message.content().trim()
        )
    }
}
impl BuildChatPrompt for Llama3ToolPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                self.create_system_prompt(message)
            }
            _ => String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe.<|eot_id|>"),
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

        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>");

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

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => {
                match tools {
                    Some(available_tools) => match available_tools.is_empty() {
                        true => self.create_system_prompt(message),
                        false => self.create_system_prompt_tool(message),
                    },
                    None => self.create_system_prompt(message)
                }
            }
            _ => String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe.<|eot_id|>"),
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = match tools {
                        Some(available_tools) => match available_tools.is_empty() {
                            true => self.append_user_message(&prompt, &system_prompt, message),
                            false => self.append_user_message_tool(
                                &prompt,
                                &system_prompt,
                                message,
                                available_tools,
                            ),
                        },
                        None => self.append_user_message(&prompt, &system_prompt, message),
                    };
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

        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>");

        Ok(prompt)
    }
}
