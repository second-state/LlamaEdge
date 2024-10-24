use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionToolMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart, Tool,
};

/// Generate prompts for the `second-state/Llama-3-Groq-8B-Tool-Use-GGUF` model.
#[derive(Debug, Default, Clone)]
pub struct GroqLlama3ToolPrompt;
impl GroqLlama3ToolPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt_tool(&self, tools: Option<&[Tool]>) -> Result<String> {
        match tools {
            Some(tools) => {
                let mut available_tools = String::new();
                for tool in tools {
                    if available_tools.is_empty() {
                        available_tools
                            .push_str(&serde_json::to_string_pretty(&tool.function).unwrap());
                    } else {
                        available_tools.push('\n');
                        available_tools
                            .push_str(&serde_json::to_string_pretty(&tool.function).unwrap());
                    }
                }

                let tools = format!(
                    "Here are the available tools:\n<tools> {} </tools>",
                    available_tools
                );

                let format = r#"{"name": <function-name>,"arguments": <args-dict>}"#;
                let begin = format!("<|start_header_id|>system<|end_header_id|>\n\nYou are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{}\n</tool_call>", format);

                let end = r#"<|eot_id|>"#;

                Ok(format!("{}\n\n{}{}", begin, tools, end))
            }
            None => Err(PromptError::NoAvailableTools),
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
            "{chat_history}<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_message}<|eot_id|>",
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
            "{chat_history}<|start_header_id|>tool<|end_header_id|>\n\n<tool_response>\n{tool_message}\n</tool_response><|eot_id|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for GroqLlama3ToolPrompt {
    fn build(&self, _messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        Err(PromptError::Operation("The GroqToolPrompt struct is only designed for `Groq/Llama-3-Groq-8B-Tool-Use` model, which is for tool use ONLY instead of general knowledge or open-ended tasks.".to_string()))
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
        let system_prompt = self.create_system_prompt_tool(tools)?;

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

        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>");

        Ok(prompt)
    }
}
