//! Define types for audio transcription.

use crate::files::FileObject;
use serde::{Deserialize, Serialize};

/// Represents a rquest for audio transcription into the input language.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct TranscriptionRequest {
    /// The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    pub file: FileObject,
    /// ID of the model to use.
    pub model: String,
    /// The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// The timestamp granularities to populate for this transcription.
    /// `response_format` must be set `verbose_json` to use timestamp granularities. Either or both of these options are supported: `word`, or `segment`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,
}

/// The timestamp granularities to populate for the transcription.
#[derive(Debug, Deserialize, Serialize)]
pub enum TimestampGranularity {
    /// The model will return timestamps for each word.
    Word,
    /// The model will return timestamps for each segment.
    Segment,
}

/// Represents a transcription response returned by model, based on the provided input.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptionObject {
    /// The transcribed text.
    pub text: String,
}

#[test]
fn test_serialize_transcription_request() {
    let obj = TranscriptionObject {
        text: String::from("Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."),
    };

    let json = serde_json::to_string(&obj).unwrap();
    assert_eq!(
        json,
        r#"{"text":"Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."}"#
    );
}

/// Represents a verbose json transcription response returned by model, based on the provided input.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VerboseTranscriptionObject {
    /// The language of the input audio.
    pub language: String,
    /// The duration of the input audio.
    pub duration: String,
    /// The transcribed text.
    pub text: String,
    /// Extracted words and their corresponding timestamps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,
    /// Segments of the transcribed text and their corresponding details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
}

#[test]
fn test_serialize_verbose_transcription_request() {
    let obj = VerboseTranscriptionObject {
        language: String::from("english"),
        duration: String::from("8.470000267028809"),
        text: String::from("The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing beach volleyball."),
        words: None,
        segments: Some(vec![
            Segment {
                id: 0,
                seek: 0,
                start: 0.0,
                end: 3.319999933242798,
                text: String::from("The beach was a popular spot on a hot summer day."),
                tokens: vec![50364, 440, 7534, 390, 257, 3743, 4008, 322, 257, 2368, 4266, 786, 13, 50530],
                temperature: 0.0,
                avg_logprob: -0.2860786020755768,
                compression_ratio: 1.2363636493682861,
                no_speech_prob: 0.00985979475080967,
            }
        ]),
    };

    let json = serde_json::to_string(&obj).unwrap();
    assert_eq!(
        json,
        r#"{"language":"english","duration":"8.470000267028809","text":"The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing beach volleyball.","segments":[{"id":0,"seek":0,"start":0.0,"end":3.319999933242798,"text":"The beach was a popular spot on a hot summer day.","tokens":[50364,440,7534,390,257,3743,4008,322,257,2368,4266,786,13,50530],"temperature":0.0,"avg_logprob":-0.2860786020755768,"compression_ratio":1.2363636493682861,"no_speech_prob":0.00985979475080967}]}"#
    );
}

/// Represents a word and its corresponding timestamps.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Word {
    /// The text content of the word.
    pub text: String,
    /// Start time of the word in seconds.
    pub start: f64,
    /// End time of the word in seconds.
    pub end: f64,
}

/// Represents a segment of the transcribed text and its corresponding details.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Segment {
    /// Unique identifier of the segment.
    pub id: u64,
    /// Seek offset of the segment.
    pub seek: u64,
    /// Start time of the segment in seconds.
    pub start: f64,
    /// End time of the segment in seconds.
    pub end: f64,
    /// Text content of the segment.
    pub text: String,
    /// Array of token IDs for the text content.
    pub tokens: Vec<u64>,
    /// Temperature parameter used for generating the segment.
    pub temperature: f64,
    /// Average logprob of the segment. If the value is lower than -1, consider the logprobs failed.
    pub avg_logprob: f64,
    /// Compression ratio of the segment. If the value is greater than 2.4, consider the compression failed.
    pub compression_ratio: f64,
    /// Probability of no speech in the segment. If the value is higher than 1.0 and the `avg_logprob` is below -1, consider this segment silent.
    pub no_speech_prob: f64,
}
