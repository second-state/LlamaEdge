//! `chat-prompts` is part of [LlamaEdge API Server](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server) project. It provides a collection of prompt templates that are used to generate prompts for the LLMs (See models in [huggingface.co/second-state](https://huggingface.co/second-state)).
//!
//! For the details of available prompt templates, see [README.md](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server/chat-prompts).

pub mod chat;
pub mod error;

use clap::ValueEnum;
use endpoints::chat::ChatCompletionRequestMessage;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Define the chat prompt template types.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum PromptTemplateType {
    #[value(name = "llama-2-chat")]
    Llama2Chat,
    #[value(name = "llama-3-chat")]
    Llama3Chat,
    #[value(name = "llama-3-tool")]
    Llama3Tool,
    #[value(name = "mistral-instruct")]
    MistralInstruct,
    #[value(name = "mistral-tool")]
    MistralTool,
    #[value(name = "mistrallite")]
    MistralLite,
    #[value(name = "openchat")]
    OpenChat,
    #[value(name = "codellama-instruct")]
    CodeLlama,
    #[value(name = "codellama-super-instruct")]
    CodeLlamaSuper,
    #[value(name = "human-assistant")]
    HumanAssistant,
    #[value(name = "vicuna-1.0-chat")]
    VicunaChat,
    #[value(name = "vicuna-1.1-chat")]
    Vicuna11Chat,
    #[value(name = "vicuna-llava")]
    VicunaLlava,
    #[value(name = "chatml")]
    ChatML,
    #[value(name = "chatml-tool")]
    ChatMLTool,
    #[value(name = "internlm-2-tool")]
    InternLM2Tool,
    #[value(name = "baichuan-2")]
    Baichuan2,
    #[value(name = "wizard-coder")]
    WizardCoder,
    #[value(name = "zephyr")]
    Zephyr,
    #[value(name = "stablelm-zephyr")]
    StableLMZephyr,
    #[value(name = "intel-neural")]
    IntelNeural,
    #[value(name = "deepseek-chat")]
    DeepseekChat,
    #[value(name = "deepseek-coder")]
    DeepseekCoder,
    #[value(name = "deepseek-chat-2")]
    DeepseekChat2,
    #[value(name = "deepseek-chat-25")]
    DeepseekChat25,
    #[value(name = "solar-instruct")]
    SolarInstruct,
    #[value(name = "phi-2-chat")]
    Phi2Chat,
    #[value(name = "phi-2-instruct")]
    Phi2Instruct,
    #[value(name = "phi-3-chat")]
    Phi3Chat,
    #[value(name = "phi-3-instruct")]
    Phi3Instruct,
    #[value(name = "gemma-instruct")]
    GemmaInstruct,
    #[value(name = "octopus")]
    Octopus,
    #[value(name = "glm-4-chat")]
    Glm4Chat,
    #[value(name = "groq-llama3-tool")]
    GroqLlama3Tool,
    #[value(name = "mediatek-breeze")]
    BreezeInstruct,
    #[value(name = "nemotron-chat")]
    NemotronChat,
    #[value(name = "nemotron-tool")]
    NemotronTool,
    #[value(name = "functionary-32")]
    FunctionaryV32,
    #[value(name = "functionary-31")]
    FunctionaryV31,
    #[value(name = "embedding")]
    Embedding,
    #[value(name = "none")]
    Null,
}
impl PromptTemplateType {
    pub fn has_system_prompt(&self) -> bool {
        match self {
            PromptTemplateType::Llama2Chat
            | PromptTemplateType::Llama3Chat
            | PromptTemplateType::Llama3Tool
            | PromptTemplateType::CodeLlama
            | PromptTemplateType::CodeLlamaSuper
            | PromptTemplateType::VicunaChat
            | PromptTemplateType::VicunaLlava
            | PromptTemplateType::ChatML
            | PromptTemplateType::ChatMLTool
            | PromptTemplateType::InternLM2Tool
            | PromptTemplateType::Baichuan2
            | PromptTemplateType::WizardCoder
            | PromptTemplateType::Zephyr
            | PromptTemplateType::IntelNeural
            | PromptTemplateType::DeepseekCoder
            | PromptTemplateType::DeepseekChat2
            | PromptTemplateType::Octopus
            | PromptTemplateType::Phi3Chat
            | PromptTemplateType::Glm4Chat
            | PromptTemplateType::GroqLlama3Tool
            | PromptTemplateType::BreezeInstruct
            | PromptTemplateType::DeepseekChat25
            | PromptTemplateType::NemotronChat
            | PromptTemplateType::NemotronTool => true,
            PromptTemplateType::MistralInstruct
            | PromptTemplateType::MistralTool
            | PromptTemplateType::MistralLite
            | PromptTemplateType::HumanAssistant
            | PromptTemplateType::DeepseekChat
            | PromptTemplateType::GemmaInstruct
            | PromptTemplateType::OpenChat
            | PromptTemplateType::Phi2Chat
            | PromptTemplateType::Phi2Instruct
            | PromptTemplateType::Phi3Instruct
            | PromptTemplateType::SolarInstruct
            | PromptTemplateType::Vicuna11Chat
            | PromptTemplateType::StableLMZephyr
            | PromptTemplateType::FunctionaryV32
            | PromptTemplateType::FunctionaryV31
            | PromptTemplateType::Embedding
            | PromptTemplateType::Null => false,
        }
    }
}
impl FromStr for PromptTemplateType {
    type Err = error::PromptError;

    fn from_str(template: &str) -> std::result::Result<Self, Self::Err> {
        match template {
            "llama-2-chat" => Ok(PromptTemplateType::Llama2Chat),
            "llama-3-chat" => Ok(PromptTemplateType::Llama3Chat),
            "llama-3-tool" => Ok(PromptTemplateType::Llama3Tool),
            "mistral-instruct" => Ok(PromptTemplateType::MistralInstruct),
            "mistral-tool" => Ok(PromptTemplateType::MistralTool),
            "mistrallite" => Ok(PromptTemplateType::MistralLite),
            "codellama-instruct" => Ok(PromptTemplateType::CodeLlama),
            "codellama-super-instruct" => Ok(PromptTemplateType::CodeLlamaSuper),
            "belle-llama-2-chat" => Ok(PromptTemplateType::HumanAssistant),
            "human-assistant" => Ok(PromptTemplateType::HumanAssistant),
            "vicuna-1.0-chat" => Ok(PromptTemplateType::VicunaChat),
            "vicuna-1.1-chat" => Ok(PromptTemplateType::Vicuna11Chat),
            "vicuna-llava" => Ok(PromptTemplateType::VicunaLlava),
            "chatml" => Ok(PromptTemplateType::ChatML),
            "chatml-tool" => Ok(PromptTemplateType::ChatMLTool),
            "internlm-2-tool" => Ok(PromptTemplateType::InternLM2Tool),
            "openchat" => Ok(PromptTemplateType::OpenChat),
            "baichuan-2" => Ok(PromptTemplateType::Baichuan2),
            "wizard-coder" => Ok(PromptTemplateType::WizardCoder),
            "zephyr" => Ok(PromptTemplateType::Zephyr),
            "stablelm-zephyr" => Ok(PromptTemplateType::StableLMZephyr),
            "intel-neural" => Ok(PromptTemplateType::IntelNeural),
            "deepseek-chat" => Ok(PromptTemplateType::DeepseekChat),
            "deepseek-coder" => Ok(PromptTemplateType::DeepseekCoder),
            "deepseek-chat-2" => Ok(PromptTemplateType::DeepseekChat2),
            "deepseek-chat-25" => Ok(PromptTemplateType::DeepseekChat25),
            "solar-instruct" => Ok(PromptTemplateType::SolarInstruct),
            "phi-2-chat" => Ok(PromptTemplateType::Phi2Chat),
            "phi-2-instruct" => Ok(PromptTemplateType::Phi2Instruct),
            "phi-3-chat" => Ok(PromptTemplateType::Phi3Chat),
            "phi-3-instruct" => Ok(PromptTemplateType::Phi3Instruct),
            "gemma-instruct" => Ok(PromptTemplateType::GemmaInstruct),
            "octopus" => Ok(PromptTemplateType::Octopus),
            "glm-4-chat" => Ok(PromptTemplateType::Glm4Chat),
            "groq-llama3-tool" => Ok(PromptTemplateType::GroqLlama3Tool),
            "mediatek-breeze" => Ok(PromptTemplateType::BreezeInstruct),
            "nemotron-chat" => Ok(PromptTemplateType::NemotronChat),
            "nemotron-tool" => Ok(PromptTemplateType::NemotronTool),
            "functionary-32" => Ok(PromptTemplateType::FunctionaryV32),
            "functionary-31" => Ok(PromptTemplateType::FunctionaryV31),
            "embedding" => Ok(PromptTemplateType::Embedding),
            "none" => Ok(PromptTemplateType::Null),
            _ => Err(error::PromptError::UnknownPromptTemplateType(
                template.to_string(),
            )),
        }
    }
}
impl std::fmt::Display for PromptTemplateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptTemplateType::Llama2Chat => write!(f, "llama-2-chat"),
            PromptTemplateType::Llama3Chat => write!(f, "llama-3-chat"),
            PromptTemplateType::Llama3Tool => write!(f, "llama-3-tool"),
            PromptTemplateType::MistralInstruct => write!(f, "mistral-instruct"),
            PromptTemplateType::MistralTool => write!(f, "mistral-tool"),
            PromptTemplateType::MistralLite => write!(f, "mistrallite"),
            PromptTemplateType::OpenChat => write!(f, "openchat"),
            PromptTemplateType::CodeLlama => write!(f, "codellama-instruct"),
            PromptTemplateType::HumanAssistant => write!(f, "human-asistant"),
            PromptTemplateType::VicunaChat => write!(f, "vicuna-1.0-chat"),
            PromptTemplateType::Vicuna11Chat => write!(f, "vicuna-1.1-chat"),
            PromptTemplateType::VicunaLlava => write!(f, "vicuna-llava"),
            PromptTemplateType::ChatML => write!(f, "chatml"),
            PromptTemplateType::ChatMLTool => write!(f, "chatml-tool"),
            PromptTemplateType::InternLM2Tool => write!(f, "internlm-2-tool"),
            PromptTemplateType::Baichuan2 => write!(f, "baichuan-2"),
            PromptTemplateType::WizardCoder => write!(f, "wizard-coder"),
            PromptTemplateType::Zephyr => write!(f, "zephyr"),
            PromptTemplateType::StableLMZephyr => write!(f, "stablelm-zephyr"),
            PromptTemplateType::IntelNeural => write!(f, "intel-neural"),
            PromptTemplateType::DeepseekChat => write!(f, "deepseek-chat"),
            PromptTemplateType::DeepseekCoder => write!(f, "deepseek-coder"),
            PromptTemplateType::DeepseekChat2 => write!(f, "deepseek-chat-2"),
            PromptTemplateType::DeepseekChat25 => write!(f, "deepseek-chat-25"),
            PromptTemplateType::SolarInstruct => write!(f, "solar-instruct"),
            PromptTemplateType::Phi2Chat => write!(f, "phi-2-chat"),
            PromptTemplateType::Phi2Instruct => write!(f, "phi-2-instruct"),
            PromptTemplateType::Phi3Chat => write!(f, "phi-3-chat"),
            PromptTemplateType::Phi3Instruct => write!(f, "phi-3-instruct"),
            PromptTemplateType::CodeLlamaSuper => write!(f, "codellama-super-instruct"),
            PromptTemplateType::GemmaInstruct => write!(f, "gemma-instruct"),
            PromptTemplateType::Octopus => write!(f, "octopus"),
            PromptTemplateType::Glm4Chat => write!(f, "glm-4-chat"),
            PromptTemplateType::GroqLlama3Tool => write!(f, "groq-llama3-tool"),
            PromptTemplateType::BreezeInstruct => write!(f, "mediatek-breeze"),
            PromptTemplateType::NemotronChat => write!(f, "nemotron-chat"),
            PromptTemplateType::NemotronTool => write!(f, "nemotron-tool"),
            PromptTemplateType::FunctionaryV32 => write!(f, "functionary-32"),
            PromptTemplateType::FunctionaryV31 => write!(f, "functionary-31"),
            PromptTemplateType::Embedding => write!(f, "embedding"),
            PromptTemplateType::Null => write!(f, "none"),
        }
    }
}

/// Trait for merging RAG context into chat messages
pub trait MergeRagContext: Send {
    /// Merge RAG context into chat messages.
    ///
    /// Note that the default implementation simply merges the RAG context into the system message. That is, to use the default implementation, `has_system_prompt` should be set to `true` and `policy` set to `MergeRagContextPolicy::SystemMessage`.
    ///
    /// # Arguments
    ///
    /// * `messages` - The chat messages to merge the context into.
    ///
    /// * `context` - The RAG context to merge into the chat messages.
    ///
    /// * `has_system_prompt` - Whether the chat template has a system prompt.
    ///
    /// * `policy` - The policy for merging RAG context into chat messages.
    fn build(
        messages: &mut Vec<endpoints::chat::ChatCompletionRequestMessage>,
        context: &[String],
        has_system_prompt: bool,
        policy: MergeRagContextPolicy,
    ) -> error::Result<()> {
        if (policy == MergeRagContextPolicy::SystemMessage) && has_system_prompt {
            if messages.is_empty() {
                return Err(error::PromptError::NoMessages);
            }

            if context.is_empty() {
                return Err(error::PromptError::Operation(
                    "No context provided.".to_string(),
                ));
            }

            let context = context[0].trim_end();

            // update or insert system message
            match messages[0] {
                ChatCompletionRequestMessage::System(ref message) => {
                    // compose new system message content
                    let content = format!("{original_system_message}\nUse the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}", original_system_message=message.content().trim(), context=context.trim_end());
                    // create system message
                    let system_message = ChatCompletionRequestMessage::new_system_message(
                        content,
                        messages[0].name().cloned(),
                    );
                    // replace the original system message
                    messages[0] = system_message;
                }
                _ => {
                    // prepare system message
                    let content = format!("Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{}", context.trim_end());

                    // create system message
                    let system_message = ChatCompletionRequestMessage::new_system_message(
                        content,
                        messages[0].name().cloned(),
                    );
                    // insert system message
                    messages.insert(0, system_message);
                }
            };
        }

        Ok(())
    }
}

/// Define the strategy for merging RAG context into chat messages.
#[derive(Clone, Debug, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum MergeRagContextPolicy {
    /// Merge RAG context into the system message.
    ///
    /// Note that this policy is only applicable when the chat template has a system message.
    #[default]
    SystemMessage,
    /// Merge RAG context into the last user message.
    LastUserMessage,
}
impl std::fmt::Display for MergeRagContextPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeRagContextPolicy::SystemMessage => write!(f, "system-message"),
            MergeRagContextPolicy::LastUserMessage => write!(f, "last-user-message"),
        }
    }
}
