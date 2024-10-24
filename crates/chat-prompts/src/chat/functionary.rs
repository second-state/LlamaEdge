use super::BuildChatPrompt;
use crate::error::{PromptError, Result};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionToolMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent, ContentPart, Tool,
};

use tera::{Context, Tera};

/// Generate prompts for `functionary-v3.2` models.
#[derive(Debug, Default, Clone)]
pub struct FunctionaryV32ToolPrompt;
impl FunctionaryV32ToolPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, tools: Option<&[Tool]>) -> String {
        match tools {
            Some(tools) if !tools.is_empty() => {
                // Create a Tera context
                let mut context = Context::new();
                context.insert("functions", tools);

                // Initialize Tera template
                let tera = Tera::one_off(
                    r#"
namespace functions {
{% for func in functions %}
    // {{ func.function.description }}
    type {{ func.function.name }} = (_: {
    {% for key, value in func.function.parameters.properties %}
        // {{ value.description }}
        {{ key }}: {{ value.type }},
    {% endfor %}
    }) => any;

{% endfor %}
}"#,
                    &context,
                    true,
                )
                .unwrap();

                let tools = format!("Available functions:\n// Supported function definitions that should be called when necessary.\n{} // namespace functions", tera.trim());

                let begin = r###"<|start_header_id|>system<|end_header_id|>

You are capable of executing available function(s) if required.
Only execute function(s) when absolutely necessary.
Ask for the required input to:recipient==all
Use JSON for function arguments.
Respond in this format:
>>>${recipient}
${content}"###;

                format!("{}\n{}<|eot_id|>", begin, tools)
            }
            _ => String::from("<|start_header_id|>system<|end_header_id|>\n\nAnswer as concisely as possible.<|eot_id|>"),
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
            "{chat_history}<|start_header_id|>assistant<|end_header_id|>\n\n>>>all\n{assistant_message}<|eot_id|>",
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
            "{chat_history}<|start_header_id|>tool<|end_header_id|>\n\n{tool_message}<|eot_id|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for FunctionaryV32ToolPrompt {
    fn build(&self, _messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        unimplemented!()
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
        let system_prompt = self.create_system_prompt(tools);

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

/// Generate prompts for `functionary-v3.1` models.
#[derive(Debug, Default, Clone)]
pub struct FunctionaryV31ToolPrompt;
impl FunctionaryV31ToolPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, tools: Option<&[Tool]>) -> String {
        match tools {
            Some(tools) if !tools.is_empty() => {
                let mut available_tools = String::new();
                for tool in tools {
                    let tool_s = serde_json::to_string(&tool.function).unwrap();
                    println!("tool_s: {}", tool_s);

                    available_tools.push_str(&format!("Use the function '{func_name}' to '{desc}'\n{tool_info}\n\n", func_name = &tool.function.name, desc = &tool.function.description.clone().unwrap_or_default(), tool_info = tool_s));
                }
                let tools = available_tools.trim();

                let begin = r###"<|start_header_id|>system<|end_header_id|>

Environment: ipython

Cutting Knowledge Date: December 2023


You have access to the following functions:"###;

                let end = r###"Think very carefully before calling functions.
If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

<|eot_id|>"###;

                format!("{}\n\n{}\n\n\n{}", begin, tools, end)
            }
            _ => String::from("<|start_header_id|>system<|end_header_id|>\n\nAnswer as concisely as possible.<|eot_id|>"),
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
            "{chat_history}<|start_header_id|>ipython<|end_header_id|>\n\n{tool_message}<|eot_id|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for FunctionaryV31ToolPrompt {
    fn build(&self, _messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        unimplemented!()
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
        let system_prompt = self.create_system_prompt(tools);

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
