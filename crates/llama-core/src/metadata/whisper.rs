use super::BaseMetadata;
use serde::{Deserialize, Serialize};

/// Builder for creating an audio metadata
#[derive(Debug)]
pub struct WhisperMetadataBuilder {
    metadata: WhisperMetadata,
}
impl WhisperMetadataBuilder {
    pub fn new<S: Into<String>>(model_name: S, model_alias: S) -> Self {
        let metadata = WhisperMetadata {
            model_name: model_name.into(),
            model_alias: model_alias.into(),
            ..Default::default()
        };

        Self { metadata }
    }

    pub fn enable_plugin_log(mut self, enable: bool) -> Self {
        self.metadata.log_enable = enable;
        self
    }

    pub fn enable_debug_log(mut self, enable: bool) -> Self {
        self.metadata.debug_log = enable;
        self
    }

    pub fn with_threads(mut self, threads: u64) -> Self {
        self.metadata.threads = threads;
        self
    }

    pub fn enable_translate(mut self, enable: bool) -> Self {
        self.metadata.translate = enable;
        self
    }

    pub fn with_language(mut self, language: String) -> Self {
        self.metadata.language = language;
        self
    }

    pub fn with_processors(mut self, processors: u64) -> Self {
        self.metadata.processors = processors;
        self
    }

    pub fn with_offset_time(mut self, offset_t: u64) -> Self {
        self.metadata.offset_time = offset_t;
        self
    }

    pub fn with_duration(mut self, duration: u64) -> Self {
        self.metadata.duration = duration;
        self
    }

    pub fn with_max_context(mut self, max_context: i32) -> Self {
        self.metadata.max_context = max_context;
        self
    }

    pub fn with_max_len(mut self, max_len: u64) -> Self {
        self.metadata.max_len = max_len;
        self
    }

    pub fn split_on_word(mut self, split_on_word: bool) -> Self {
        self.metadata.split_on_word = split_on_word;
        self
    }

    pub fn output_txt(mut self, output_txt: bool) -> Self {
        self.metadata.output_txt = output_txt;
        self
    }

    pub fn output_vtt(mut self, output_vtt: bool) -> Self {
        self.metadata.output_vtt = output_vtt;
        self
    }

    pub fn output_srt(mut self, output_srt: bool) -> Self {
        self.metadata.output_srt = output_srt;
        self
    }

    pub fn output_lrc(mut self, output_lrc: bool) -> Self {
        self.metadata.output_lrc = output_lrc;
        self
    }

    pub fn output_csv(mut self, output_csv: bool) -> Self {
        self.metadata.output_csv = output_csv;
        self
    }

    pub fn output_json(mut self, output_json: bool) -> Self {
        self.metadata.output_json = output_json;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.metadata.temperature = temperature;
        self
    }

    pub fn detect_language(mut self, detect_language: bool) -> Self {
        self.metadata.detect_language = detect_language;
        self
    }

    pub fn build(self) -> WhisperMetadata {
        self.metadata
    }
}

/// Metadata for whisper model
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WhisperMetadata {
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_name: String,
    // this field not defined for the beckend plugin
    #[serde(skip_serializing)]
    pub model_alias: String,

    #[serde(rename = "enable-log")]
    pub log_enable: bool,
    /// Enable debug mode. Defaults to false.
    #[serde(rename = "enable-debug-log")]
    pub debug_log: bool,

    /// Number of threads to use during computation. Defaults to 4.
    pub threads: u64,
    /// Translate from source language to english. Defaults to false.
    pub translate: bool,
    /// The language of the input audio. `auto` for auto-detection. Defaults to `en`.
    ///
    /// Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.
    pub language: String,
    /// Number of processors to use during computation. Defaults to 1.
    pub processors: u64,
    /// Time offset in milliseconds. Defaults to 0.
    #[serde(rename = "offset-t")]
    pub offset_time: u64,
    /// Duration of audio to process in milliseconds. Defaults to 0.
    pub duration: u64,
    /// Maximum number of text context tokens to store. Defaults to -1.
    pub max_context: i32,
    /// Maximum segment length in characters. Defaults to 0.
    pub max_len: u64,
    /// Split on word rather than on token. Defaults to false.
    pub split_on_word: bool,
    /// Output result in a text file. Defaults to false.
    pub output_txt: bool,
    /// Output result in a vtt file. Defaults to false.
    pub output_vtt: bool,
    /// Output result in a srt file. Defaults to false.
    pub output_srt: bool,
    /// Output result in a lrc file. Defaults to false.
    pub output_lrc: bool,
    /// Output result in a CSV file. Defaults to false.
    pub output_csv: bool,
    /// Output result in a JSON file. Defaults to false.
    pub output_json: bool,
    /// Sampling temperature, between 0 and 1. Defaults to 0.00.
    pub temperature: f64,
    /// Automatically detect the spoken language in the provided audio input.
    pub detect_language: bool,
}
impl Default for WhisperMetadata {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_alias: String::new(),
            log_enable: false,
            debug_log: false,
            threads: 4,
            translate: false,
            language: "en".to_string(),
            processors: 1,
            offset_time: 0,
            duration: 0,
            max_context: -1,
            max_len: 0,
            split_on_word: false,
            output_txt: false,
            output_vtt: false,
            output_srt: false,
            output_lrc: false,
            output_csv: false,
            output_json: false,
            temperature: 0.0,
            detect_language: false,
        }
    }
}
impl BaseMetadata for WhisperMetadata {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_alias(&self) -> &str {
        &self.model_alias
    }
}
