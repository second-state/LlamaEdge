//! Define types for translating audio into English.

use crate::files::FileObject;
use serde::{
    de::{self, MapAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::fmt;

/// Represents a rquest for translating audio into English.
#[derive(Debug, Serialize)]
pub struct TranslationRequest {
    /// The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    pub file: FileObject,
    /// ID of the model to use.
    pub model: Option<String>,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should be in English.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: `json`, `text`, `srt`, `verbose_json`, or `vtt`. Defaults to `json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use [log probability](https://en.wikipedia.org/wiki/Log_probability) to automatically increase the temperature until certain thresholds are hit. Defaults to 0.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Spoken language. `auto` for auto-detect. Defaults to `en`. This param is only supported for `whisper.cpp`.
    pub language: Option<String>,
}
impl<'de> Deserialize<'de> for TranslationRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            File,
            Model,
            Prompt,
            ResponseFormat,
            Temperature,
            Language,
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
                        formatter.write_str("`file`, `model`, `prompt`, `response_format`, `temperature`, or `language`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "file" => Ok(Field::File),
                            "model" => Ok(Field::Model),
                            "prompt" => Ok(Field::Prompt),
                            "response_format" => Ok(Field::ResponseFormat),
                            "temperature" => Ok(Field::Temperature),
                            "language" => Ok(Field::Language),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TranslationRequestVisitor;

        impl<'de> Visitor<'de> for TranslationRequestVisitor {
            type Value = TranslationRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TranslationRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TranslationRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut file = None;
                let mut model = None;
                let mut prompt = None;
                let mut response_format = None;
                let mut temperature = None;
                let mut language = None;

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
                        Field::Language => {
                            if language.is_some() {
                                return Err(de::Error::duplicate_field("language"));
                            }
                            language = Some(map.next_value()?);
                        }
                    }
                }

                let file = file.ok_or_else(|| de::Error::missing_field("file"))?;

                if response_format.is_none() {
                    response_format = Some("json".to_string());
                }

                if temperature.is_none() {
                    temperature = Some(0.0);
                }

                if language.is_none() {
                    language = Some("en".to_string());
                }

                Ok(TranslationRequest {
                    file,
                    model,
                    prompt,
                    response_format,
                    temperature,
                    language,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "file",
            "model",
            "prompt",
            "response_format",
            "temperature",
            "language",
        ];
        deserializer.deserialize_struct("TranslationRequest", FIELDS, TranslationRequestVisitor)
    }
}
impl Default for TranslationRequest {
    fn default() -> Self {
        TranslationRequest {
            file: FileObject::default(),
            model: None,
            prompt: None,
            response_format: Some("json".to_string()),
            temperature: Some(0.0),
            language: Some("en".to_string()),
        }
    }
}

/// Represents a translation object.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranslationObject {
    /// The translated text.
    pub text: String,
}
