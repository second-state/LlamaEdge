pub mod ggml;
pub mod piper;
pub mod whisper;

pub trait BaseMetadata {
    fn model_name(&self) -> &str;
    fn model_alias(&self) -> &str;
}
