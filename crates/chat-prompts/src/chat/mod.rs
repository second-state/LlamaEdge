pub mod baichuan;
pub mod belle;
pub mod chatml;
pub mod deepseek;
pub mod functionary;
pub mod gemma;
pub mod glm;
pub mod groq;
pub mod intel;
pub mod llama;
pub mod mediatek;
pub mod mistral;
pub mod nvidia;
pub mod octopus;
pub mod openchat;
pub mod phi;
pub mod solar;
pub mod vicuna;
pub mod wizard;
pub mod zephyr;

use crate::{error::Result, PromptTemplateType};
use baichuan::*;
use belle::*;
use chatml::*;
use deepseek::*;
use endpoints::chat::{ChatCompletionRequestMessage, Tool};
use functionary::{FunctionaryV31ToolPrompt, FunctionaryV32ToolPrompt};
use gemma::*;
use glm::*;
use groq::*;
use intel::*;
use llama::*;
use mediatek::BreezeInstructPrompt;
use mistral::*;
use nvidia::{NemotronChatPrompt, NemotronToolPrompt};
use octopus::*;
use openchat::*;
use phi::*;
use solar::*;
use vicuna::*;
use wizard::*;
use zephyr::*;

/// Trait for building prompts for chat completions.
#[enum_dispatch::enum_dispatch]
pub trait BuildChatPrompt: Send {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String>;

    fn build_with_tools(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        _tools: Option<&[Tool]>,
    ) -> Result<String> {
        self.build(messages)
    }
}

#[enum_dispatch::enum_dispatch(BuildChatPrompt)]
pub enum ChatPrompt {
    Llama2ChatPrompt,
    Llama3ChatPrompt,
    Llama3ToolPrompt,
    MistralInstructPrompt,
    MistralToolPrompt,
    MistralLitePrompt,
    OpenChatPrompt,
    CodeLlamaInstructPrompt,
    CodeLlamaSuperInstructPrompt,
    HumanAssistantChatPrompt,
    /// Vicuna 1.0
    VicunaChatPrompt,
    /// Vicuna 1.1
    Vicuna11ChatPrompt,
    VicunaLlavaPrompt,
    ChatMLPrompt,
    ChatMLToolPrompt,
    InternLM2ToolPrompt,
    Baichuan2ChatPrompt,
    WizardCoderPrompt,
    ZephyrChatPrompt,
    StableLMZephyrChatPrompt,
    NeuralChatPrompt,
    DeepseekChatPrompt,
    DeepseekCoderPrompt,
    DeepseekChat2Prompt,
    DeepseekChat25Prompt,
    SolarInstructPrompt,
    Phi2ChatPrompt,
    Phi2InstructPrompt,
    Phi3ChatPrompt,
    Phi3InstructPrompt,
    GemmaInstructPrompt,
    OctopusPrompt,
    Glm4ChatPrompt,
    GroqLlama3ToolPrompt,
    BreezeInstructPrompt,
    NemotronChatPrompt,
    NemotronToolPrompt,
    FunctionaryV32ToolPrompt,
    FunctionaryV31ToolPrompt,
}
impl From<PromptTemplateType> for ChatPrompt {
    fn from(ty: PromptTemplateType) -> Self {
        match ty {
            PromptTemplateType::Llama2Chat => ChatPrompt::Llama2ChatPrompt(Llama2ChatPrompt),
            PromptTemplateType::Llama3Chat => ChatPrompt::Llama3ChatPrompt(Llama3ChatPrompt),
            PromptTemplateType::Llama3Tool => ChatPrompt::Llama3ToolPrompt(Llama3ToolPrompt),
            PromptTemplateType::MistralInstruct => {
                ChatPrompt::MistralInstructPrompt(MistralInstructPrompt)
            }
            PromptTemplateType::MistralTool => ChatPrompt::MistralToolPrompt(MistralToolPrompt),
            PromptTemplateType::MistralLite => ChatPrompt::MistralLitePrompt(MistralLitePrompt),
            PromptTemplateType::OpenChat => ChatPrompt::OpenChatPrompt(OpenChatPrompt),
            PromptTemplateType::CodeLlama => {
                ChatPrompt::CodeLlamaInstructPrompt(CodeLlamaInstructPrompt)
            }
            PromptTemplateType::CodeLlamaSuper => {
                ChatPrompt::CodeLlamaSuperInstructPrompt(CodeLlamaSuperInstructPrompt)
            }
            PromptTemplateType::HumanAssistant => {
                ChatPrompt::HumanAssistantChatPrompt(HumanAssistantChatPrompt)
            }
            PromptTemplateType::VicunaChat => ChatPrompt::VicunaChatPrompt(VicunaChatPrompt),
            PromptTemplateType::Vicuna11Chat => ChatPrompt::Vicuna11ChatPrompt(Vicuna11ChatPrompt),
            PromptTemplateType::VicunaLlava => ChatPrompt::VicunaLlavaPrompt(VicunaLlavaPrompt),
            PromptTemplateType::ChatML => ChatPrompt::ChatMLPrompt(ChatMLPrompt),
            PromptTemplateType::ChatMLTool => ChatPrompt::ChatMLToolPrompt(ChatMLToolPrompt),
            PromptTemplateType::InternLM2Tool => {
                ChatPrompt::InternLM2ToolPrompt(InternLM2ToolPrompt)
            }
            PromptTemplateType::Baichuan2 => ChatPrompt::Baichuan2ChatPrompt(Baichuan2ChatPrompt),
            PromptTemplateType::WizardCoder => ChatPrompt::WizardCoderPrompt(WizardCoderPrompt),
            PromptTemplateType::Zephyr => ChatPrompt::ZephyrChatPrompt(ZephyrChatPrompt),
            PromptTemplateType::StableLMZephyr => {
                ChatPrompt::StableLMZephyrChatPrompt(StableLMZephyrChatPrompt)
            }
            PromptTemplateType::IntelNeural => ChatPrompt::NeuralChatPrompt(NeuralChatPrompt),
            PromptTemplateType::DeepseekChat => ChatPrompt::DeepseekChatPrompt(DeepseekChatPrompt),
            PromptTemplateType::DeepseekCoder => {
                ChatPrompt::DeepseekCoderPrompt(DeepseekCoderPrompt)
            }
            PromptTemplateType::DeepseekChat2 => {
                ChatPrompt::DeepseekChat2Prompt(DeepseekChat2Prompt)
            }
            PromptTemplateType::DeepseekChat25 => {
                ChatPrompt::DeepseekChat25Prompt(DeepseekChat25Prompt)
            }
            PromptTemplateType::SolarInstruct => {
                ChatPrompt::SolarInstructPrompt(SolarInstructPrompt)
            }
            PromptTemplateType::Phi2Chat => ChatPrompt::Phi2ChatPrompt(Phi2ChatPrompt),
            PromptTemplateType::Phi2Instruct => ChatPrompt::Phi2InstructPrompt(Phi2InstructPrompt),
            PromptTemplateType::Phi3Chat => ChatPrompt::Phi3ChatPrompt(Phi3ChatPrompt),
            PromptTemplateType::Phi3Instruct => ChatPrompt::Phi3InstructPrompt(Phi3InstructPrompt),
            PromptTemplateType::GemmaInstruct => {
                ChatPrompt::GemmaInstructPrompt(GemmaInstructPrompt)
            }
            PromptTemplateType::Octopus => ChatPrompt::OctopusPrompt(OctopusPrompt),
            PromptTemplateType::Glm4Chat => ChatPrompt::Glm4ChatPrompt(Glm4ChatPrompt),
            PromptTemplateType::GroqLlama3Tool => {
                ChatPrompt::GroqLlama3ToolPrompt(GroqLlama3ToolPrompt)
            }
            PromptTemplateType::BreezeInstruct => {
                ChatPrompt::BreezeInstructPrompt(BreezeInstructPrompt)
            }
            PromptTemplateType::NemotronChat => ChatPrompt::NemotronChatPrompt(NemotronChatPrompt),
            PromptTemplateType::NemotronTool => ChatPrompt::NemotronToolPrompt(NemotronToolPrompt),
            PromptTemplateType::FunctionaryV32 => {
                ChatPrompt::FunctionaryV32ToolPrompt(FunctionaryV32ToolPrompt)
            }
            PromptTemplateType::FunctionaryV31 => {
                ChatPrompt::FunctionaryV31ToolPrompt(FunctionaryV31ToolPrompt)
            }
            PromptTemplateType::Embedding => {
                panic!("Embedding prompt template is not used for building chat prompts")
            }
            PromptTemplateType::Null => {
                panic!("Null prompt template is not used for building chat prompts")
            }
        }
    }
}
