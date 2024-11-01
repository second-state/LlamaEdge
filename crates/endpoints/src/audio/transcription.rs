//! Define types for audio transcription.

use crate::files::FileObject;
use serde::{
    de::{self, Deserializer, MapAccess, Visitor},
    Deserialize, Serialize,
};
use std::fmt;

/// Represents a rquest for audio transcription into the input language.
#[derive(Debug, Serialize)]
pub struct TranscriptionRequest {
    /// The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    pub file: FileObject,
    /// ID of the model to use.
    pub model: String,
    /// The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency. Defaults to `en`.
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

    /// Automatically detect the spoken language in the provided audio input. Defaults to false. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect_language: Option<bool>,
    /// Time offset in milliseconds. Defaults to 0. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_time: Option<u64>,
    /// Length of audio (in seconds) to be processed starting from the point defined by the `offset_time` field (or from the beginning by default). Defaults to 0. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u64>,
    /// Maximum amount of text context (in tokens) that the model uses when processing long audio inputs incrementally. Defaults to -1. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context: Option<i32>,
    /// Maximum number of tokens that the model can generate in a single transcription segment (or chunk). Defaults to 0. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none", rename = "max-len")]
    pub max_len: Option<u64>,
    /// Split audio chunks on word rather than on token. Defaults to false. This param is reserved for `whisper.cpp`.
    #[serde(skip_serializing_if = "Option::is_none", rename = "split-on-word")]
    pub split_on_word: Option<bool>,
}
impl Default for TranscriptionRequest {
    fn default() -> Self {
        Self {
            file: FileObject::default(),
            model: String::new(),
            language: Some("en".to_string()),
            prompt: None,
            response_format: Some("json".to_string()),
            temperature: Some(0.0),
            timestamp_granularities: Some(vec![TimestampGranularity::Segment]),
            detect_language: Some(false),
            offset_time: Some(0),
            duration: Some(0),
            max_context: Some(-1),
            max_len: Some(0),
            split_on_word: Some(false),
        }
    }
}
impl<'de> Deserialize<'de> for TranscriptionRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            File,
            Model,
            Language,
            Prompt,
            ResponseFormat,
            Temperature,
            TimestampGranularities,
            DetectLanguage,
            OffsetTime,
            Duration,
            MaxContext,
            MaxLen,
            SplitOnWord,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl Visitor<'_> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`file`, `model`, `language`, `prompt`, `response_format`, `temperature`, `timestamp_granularities`, `detect_language`, `offset_time`, `duration`, `max_context`, `max_len`, or `split_on_word`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "file" => Ok(Field::File),
                            "model" => Ok(Field::Model),
                            "language" => Ok(Field::Language),
                            "prompt" => Ok(Field::Prompt),
                            "response_format" => Ok(Field::ResponseFormat),
                            "temperature" => Ok(Field::Temperature),
                            "timestamp_granularities" => Ok(Field::TimestampGranularities),
                            "detect_language" => Ok(Field::DetectLanguage),
                            "offset_time" => Ok(Field::OffsetTime),
                            "duration" => Ok(Field::Duration),
                            "max_context" => Ok(Field::MaxContext),
                            "max_len" => Ok(Field::MaxLen),
                            "split_on_word" => Ok(Field::SplitOnWord),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TranscriptionRequestVisitor;

        impl<'de> Visitor<'de> for TranscriptionRequestVisitor {
            type Value = TranscriptionRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TranscriptionRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TranscriptionRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut file = None;
                let mut model = None;
                let mut language = None;
                let mut prompt = None;
                let mut response_format = None;
                let mut temperature = None;
                let mut timestamp_granularities = None;
                let mut detect_language = None;
                let mut offset_time = None;
                let mut duration = None;
                let mut max_context = None;
                let mut max_len = None;
                let mut split_on_word = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::File => {
                            if file.is_some() {
                                return Err(de::Error::duplicate_field("file"));
                            }
                            file = Some(map.next_value()?);
                        }
                        Field::Model => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        Field::Language => {
                            if language.is_some() {
                                return Err(de::Error::duplicate_field("language"));
                            }
                            language = Some(map.next_value()?);
                        }
                        Field::Prompt => {
                            if prompt.is_some() {
                                return Err(de::Error::duplicate_field("prompt"));
                            }
                            prompt = Some(map.next_value()?);
                        }
                        Field::ResponseFormat => {
                            if response_format.is_some() {
                                return Err(de::Error::duplicate_field("response_format"));
                            }
                            response_format = Some(map.next_value()?);
                        }
                        Field::Temperature => {
                            if temperature.is_some() {
                                return Err(de::Error::duplicate_field("temperature"));
                            }
                            temperature = Some(map.next_value()?);
                        }
                        Field::TimestampGranularities => {
                            if timestamp_granularities.is_some() {
                                return Err(de::Error::duplicate_field("timestamp_granularities"));
                            }
                            timestamp_granularities = Some(map.next_value()?);
                        }
                        Field::DetectLanguage => {
                            if detect_language.is_some() {
                                return Err(de::Error::duplicate_field("detect_language"));
                            }
                            detect_language = Some(map.next_value()?);
                        }
                        Field::OffsetTime => {
                            if offset_time.is_some() {
                                return Err(de::Error::duplicate_field("offset_time"));
                            }
                            offset_time = Some(map.next_value()?);
                        }
                        Field::Duration => {
                            if duration.is_some() {
                                return Err(de::Error::duplicate_field("duration"));
                            }
                            duration = Some(map.next_value()?);
                        }
                        Field::MaxContext => {
                            if max_context.is_some() {
                                return Err(de::Error::duplicate_field("max_context"));
                            }
                            max_context = Some(map.next_value()?);
                        }
                        Field::MaxLen => {
                            if max_len.is_some() {
                                return Err(de::Error::duplicate_field("max_len"));
                            }
                            max_len = Some(map.next_value()?);
                        }
                        Field::SplitOnWord => {
                            if split_on_word.is_some() {
                                return Err(de::Error::duplicate_field("split_on_word"));
                            }
                            split_on_word = Some(map.next_value()?);
                        }
                    }
                }

                let file = file.ok_or_else(|| de::Error::missing_field("file"))?;
                let model = model.ok_or_else(|| de::Error::missing_field("model"))?;

                if language.is_none() {
                    language = Some("en".to_string());
                }

                if response_format.is_none() {
                    response_format = Some("json".to_string());
                }

                if temperature.is_none() {
                    temperature = Some(0.0);
                }

                if timestamp_granularities.is_none() {
                    timestamp_granularities = Some(vec![TimestampGranularity::Segment]);
                }

                if detect_language.is_none() {
                    detect_language = Some(false);
                }

                if offset_time.is_none() {
                    offset_time = Some(0);
                }

                if duration.is_none() {
                    duration = Some(0);
                }

                if max_context.is_none() {
                    max_context = Some(-1);
                }

                if max_len.is_none() {
                    match &timestamp_granularities {
                        Some(granularities) => {
                            if granularities[0] == TimestampGranularity::Word {
                                max_len = Some(1);
                            } else {
                                max_len = Some(0);
                            }
                        }
                        None => max_len = Some(0),
                    }
                }

                if split_on_word.is_none() {
                    split_on_word = Some(false);
                }

                Ok(TranscriptionRequest {
                    file,
                    model,
                    language,
                    prompt,
                    response_format,
                    temperature,
                    timestamp_granularities,
                    detect_language,
                    offset_time,
                    duration,
                    max_context,
                    max_len,
                    split_on_word,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "file",
            "model",
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
            "detect_language",
            "offset_time",
            "duration",
            "max_context",
            "max_len",
            "split_on_word",
        ];

        deserializer.deserialize_struct("TranscriptionRequest", FIELDS, TranscriptionRequestVisitor)
    }
}

/// The timestamp granularities to populate for the transcription.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
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
