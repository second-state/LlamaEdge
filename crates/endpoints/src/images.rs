//! Define types for image generation.

use crate::files::FileObject;
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::{fmt, str::FromStr};

/// Builder for creating a `ImageCreateRequest` instance.
pub struct ImageCreateRequestBuilder {
    req: ImageCreateRequest,
}
impl ImageCreateRequestBuilder {
    /// Create a new builder with the given model and prompt.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            req: ImageCreateRequest {
                model: model.into(),
                prompt: prompt.into(),
                ..Default::default()
            },
        }
    }

    /// Set negative prompt
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.req.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Set the number of images to generate.
    pub fn with_number_of_images(mut self, n: u64) -> Self {
        self.req.n = Some(n);
        self
    }

    /// This param is only supported for OpenAI `dall-e-3`.
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.req.quality = Some(quality.into());
        self
    }

    /// Set the format in which the generated images are returned.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.req.response_format = Some(response_format);
        self
    }

    /// This param is only supported for `dall-e-3`.
    pub fn with_style(mut self, style: impl Into<String>) -> Self {
        self.req.style = Some(style.into());
        self
    }

    /// Set the user id
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    /// Set the unconditional guidance scale. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_cfg_scale(mut self, cfg_scale: f32) -> Self {
        self.req.cfg_scale = Some(cfg_scale);
        self
    }

    /// Set the sampling method. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_sample_method(mut self, sample_method: SamplingMethod) -> Self {
        self.req.sample_method = Some(sample_method);
        self
    }

    /// Set the number of sample steps. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.req.steps = Some(steps);
        self
    }

    /// Set the image size.
    pub fn with_image_size(mut self, height: usize, width: usize) -> Self {
        self.req.height = Some(height);
        self.req.width = Some(width);
        self
    }

    /// Set the strength to apply Control Net. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_control_strength(mut self, control_strength: f32) -> Self {
        self.req.control_strength = Some(control_strength);
        self
    }

    /// Set the RNG seed. Negative value means to use random seed. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_seed(mut self, seed: i32) -> Self {
        self.req.seed = Some(seed);
        self
    }

    /// Build the request.
    pub fn build(self) -> ImageCreateRequest {
        self.req
    }
}

/// Request to create an image by a given prompt.
#[derive(Debug, Serialize)]
pub struct ImageCreateRequest {
    /// A text description of the desired image.
    pub prompt: String,
    /// Negative prompt for the image generation.
    pub negative_prompt: Option<String>,
    /// Name of the model to use for image generation.
    pub model: String,
    /// Number of images to generate. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    /// The quality of the image that will be generated. hd creates images with finer details and greater consistency across the image. Defaults to "standard". This param is only supported for OpenAI `dall-e-3`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// The format in which the generated images are returned. Must be one of `url` or `b64_json`. Defaults to `Url`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// The size of the generated images. Defaults to use the values of `height` and `width` fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The style of the generated images. Must be one of `vivid` or `natural`. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for `dall-e-3`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// A unique identifier representing your end-user, which can help monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Unconditional guidance scale. Defaults to 7.0. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    /// Sampling method. Defaults to "euler_a". This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_method: Option<SamplingMethod>,
    /// Number of sample steps. Defaults to 20. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<usize>,
    /// Image height, in pixel space. Defaults to 512. If `size` is provided, this field will be ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<usize>,
    /// Image width, in pixel space. Defaults to 512. If `size` is provided, this field will be ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    /// strength to apply Control Net. Defaults to 0.9. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_strength: Option<f32>,
    /// RNG seed. Negative value means to use random seed. Defaults to 42. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
}
impl<'de> Deserialize<'de> for ImageCreateRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Prompt,
            NegativePrompt,
            Model,
            N,
            Quality,
            ResponseFormat,
            Size,
            Style,
            User,
            CfgScale,
            SampleMethod,
            Steps,
            Height,
            Width,
            ControlStrength,
            Seed,
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
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "prompt" => Ok(Field::Prompt),
                            "negative_prompt" => Ok(Field::NegativePrompt),
                            "model" => Ok(Field::Model),
                            "n" => Ok(Field::N),
                            "quality" => Ok(Field::Quality),
                            "response_format" => Ok(Field::ResponseFormat),
                            "size" => Ok(Field::Size),
                            "style" => Ok(Field::Style),
                            "user" => Ok(Field::User),
                            "cfg_scale" => Ok(Field::CfgScale),
                            "sample_method" => Ok(Field::SampleMethod),
                            "steps" => Ok(Field::Steps),
                            "height" => Ok(Field::Height),
                            "width" => Ok(Field::Width),
                            "control_strength" => Ok(Field::ControlStrength),
                            "seed" => Ok(Field::Seed),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct CreateImageRequestVisitor;

        impl<'de> Visitor<'de> for CreateImageRequestVisitor {
            type Value = ImageCreateRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CreateImageRequest")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<ImageCreateRequest, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let prompt = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let negative_prompt = seq.next_element()?;
                let model = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let n = seq.next_element()?.unwrap_or(Some(1));
                let quality = seq.next_element()?;
                let response_format = seq.next_element()?.unwrap_or(Some(ResponseFormat::Url));
                let size = seq.next_element()?;
                let style = seq.next_element()?;
                let user = seq.next_element()?;
                let cfg_scale = seq.next_element()?;
                let sample_method = seq.next_element()?;
                let steps = seq.next_element()?;
                let height = seq.next_element()?;
                let width = seq.next_element()?;
                let control_strength = seq.next_element()?;
                let seed = seq.next_element()?;

                Ok(ImageCreateRequest {
                    prompt,
                    negative_prompt,
                    model,
                    n,
                    quality,
                    response_format,
                    size,
                    style,
                    user,
                    cfg_scale,
                    sample_method,
                    steps,
                    height,
                    width,
                    control_strength,
                    seed,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<ImageCreateRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut prompt = None;
                let mut negative_prompt = None;
                let mut model = None;
                let mut n = None;
                let mut quality = None;
                let mut response_format = None;
                let mut size: Option<String> = None;
                let mut style = None;
                let mut user = None;
                let mut cfg_scale = None;
                let mut sample_method = None;
                let mut steps = None;
                let mut height = None;
                let mut width = None;
                let mut control_strength = None;
                let mut seed = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Prompt => {
                            if prompt.is_some() {
                                return Err(de::Error::duplicate_field("prompt"));
                            }
                            prompt = Some(map.next_value()?);
                        }
                        Field::NegativePrompt => {
                            if negative_prompt.is_some() {
                                return Err(de::Error::duplicate_field("negative_prompt"));
                            }
                            negative_prompt = Some(map.next_value()?);
                        }
                        Field::Model => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        Field::N => {
                            if n.is_some() {
                                return Err(de::Error::duplicate_field("n"));
                            }
                            n = Some(map.next_value()?);
                        }
                        Field::Quality => {
                            if quality.is_some() {
                                return Err(de::Error::duplicate_field("quality"));
                            }
                            quality = Some(map.next_value()?);
                        }
                        Field::ResponseFormat => {
                            if response_format.is_some() {
                                return Err(de::Error::duplicate_field("response_format"));
                            }
                            response_format = Some(map.next_value()?);
                        }
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                        Field::Style => {
                            if style.is_some() {
                                return Err(de::Error::duplicate_field("style"));
                            }
                            style = Some(map.next_value()?);
                        }
                        Field::User => {
                            if user.is_some() {
                                return Err(de::Error::duplicate_field("user"));
                            }
                            user = Some(map.next_value()?);
                        }
                        Field::CfgScale => {
                            if cfg_scale.is_some() {
                                return Err(de::Error::duplicate_field("cfg_scale"));
                            }
                            cfg_scale = Some(map.next_value()?);
                        }
                        Field::SampleMethod => {
                            if sample_method.is_some() {
                                return Err(de::Error::duplicate_field("sample_method"));
                            }
                            sample_method = Some(map.next_value()?);
                        }
                        Field::Steps => {
                            if steps.is_some() {
                                return Err(de::Error::duplicate_field("steps"));
                            }
                            steps = Some(map.next_value()?);
                        }
                        Field::Height => {
                            if height.is_some() {
                                return Err(de::Error::duplicate_field("height"));
                            }
                            height = Some(map.next_value()?);
                        }
                        Field::Width => {
                            if width.is_some() {
                                return Err(de::Error::duplicate_field("width"));
                            }
                            width = Some(map.next_value()?);
                        }
                        Field::ControlStrength => {
                            if control_strength.is_some() {
                                return Err(de::Error::duplicate_field("control_strength"));
                            }
                            control_strength = Some(map.next_value()?);
                        }
                        Field::Seed => {
                            if seed.is_some() {
                                return Err(de::Error::duplicate_field("seed"));
                            }
                            seed = Some(map.next_value()?);
                        }
                    }
                }

                if negative_prompt.is_none() {
                    negative_prompt = Some("".to_string());
                }

                if n.is_none() {
                    n = Some(1);
                }

                if response_format.is_none() {
                    response_format = Some(ResponseFormat::Url);
                }

                if cfg_scale.is_none() {
                    cfg_scale = Some(7.0);
                }

                if sample_method.is_none() {
                    sample_method = Some(SamplingMethod::EulerA);
                }

                if steps.is_none() {
                    steps = Some(20);
                }

                if control_strength.is_none() {
                    control_strength = Some(0.9);
                }

                if seed.is_none() {
                    seed = Some(42);
                }

                match &size {
                    Some(size) => {
                        let parts: Vec<&str> = size.split('x').collect();
                        if parts.len() != 2 {
                            return Err(de::Error::custom("invalid size format"));
                        }
                        height = Some(parts[0].parse().unwrap());
                        width = Some(parts[1].parse().unwrap());
                    }
                    None => {
                        if height.is_none() {
                            height = Some(512);
                        }
                        if width.is_none() {
                            width = Some(512);
                        }
                    }
                }

                Ok(ImageCreateRequest {
                    prompt: prompt.ok_or_else(|| de::Error::missing_field("prompt"))?,
                    negative_prompt,
                    model: model.ok_or_else(|| de::Error::missing_field("model"))?,
                    n,
                    quality,
                    response_format,
                    size,
                    style,
                    user,
                    cfg_scale,
                    sample_method,
                    steps,
                    height,
                    width,
                    control_strength,
                    seed,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "prompt",
            "negative_prompt",
            "model",
            "n",
            "quality",
            "response_format",
            "size",
            "style",
            "user",
            "cfg_scale",
            "sample_method",
            "steps",
            "height",
            "width",
        ];
        deserializer.deserialize_struct("CreateImageRequest", FIELDS, CreateImageRequestVisitor)
    }
}
impl Default for ImageCreateRequest {
    fn default() -> Self {
        Self {
            prompt: "".to_string(),
            quality: Some("standard".to_string()),
            negative_prompt: Some("".to_string()),
            model: "".to_string(),
            n: Some(1),
            response_format: Some(ResponseFormat::Url),
            size: None,
            style: Some("natural".to_string()),
            user: None,
            cfg_scale: Some(7.0),
            sample_method: Some(SamplingMethod::EulerA),
            steps: Some(20),
            height: Some(512),
            width: Some(512),
            control_strength: Some(0.9),
            seed: Some(42),
        }
    }
}

/// Sampling method
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum SamplingMethod {
    #[serde(rename = "euler")]
    Euler,
    #[serde(rename = "euler_a")]
    EulerA,
    #[serde(rename = "heun")]
    Heun,
    #[serde(rename = "dpm2")]
    Dpm2,
    #[serde(rename = "dpm++2s_a")]
    DpmPlusPlus2sA,
    #[serde(rename = "dpm++2m")]
    DpmPlusPlus2m,
    #[serde(rename = "dpm++2mv2")]
    DpmPlusPlus2mv2,
    #[serde(rename = "ipndm")]
    Ipndm,
    #[serde(rename = "ipndm_v")]
    IpndmV,
    #[serde(rename = "lcm")]
    Lcm,
}
impl fmt::Display for SamplingMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SamplingMethod::Euler => write!(f, "euler"),
            SamplingMethod::EulerA => write!(f, "euler_a"),
            SamplingMethod::Heun => write!(f, "heun"),
            SamplingMethod::Dpm2 => write!(f, "dpm2"),
            SamplingMethod::DpmPlusPlus2sA => write!(f, "dpm++2s_a"),
            SamplingMethod::DpmPlusPlus2m => write!(f, "dpm++2m"),
            SamplingMethod::DpmPlusPlus2mv2 => write!(f, "dpm++2mv2"),
            SamplingMethod::Ipndm => write!(f, "ipndm"),
            SamplingMethod::IpndmV => write!(f, "ipndm_v"),
            SamplingMethod::Lcm => write!(f, "lcm"),
        }
    }
}
impl From<&str> for SamplingMethod {
    fn from(s: &str) -> Self {
        match s {
            "euler" => SamplingMethod::Euler,
            "euler_a" => SamplingMethod::EulerA,
            "heun" => SamplingMethod::Heun,
            "dpm2" => SamplingMethod::Dpm2,
            "dpm++2s_a" => SamplingMethod::DpmPlusPlus2sA,
            "dpm++2m" => SamplingMethod::DpmPlusPlus2m,
            "dpm++2mv2" => SamplingMethod::DpmPlusPlus2mv2,
            "ipndm" => SamplingMethod::Ipndm,
            "ipndm_v" => SamplingMethod::IpndmV,
            "lcm" => SamplingMethod::Lcm,
            _ => SamplingMethod::EulerA,
        }
    }
}

#[test]
fn test_serialize_image_create_request() {
    {
        let req = ImageCreateRequestBuilder::new("test-model-name", "This is a prompt")
            .with_negative_prompt("This is the negative prompt.")
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"prompt":"This is a prompt","negative_prompt":"This is the negative prompt.","model":"test-model-name","n":1,"quality":"standard","response_format":"url","style":"natural","cfg_scale":7.0,"sample_method":"euler_a","steps":20,"height":512,"width":512,"control_strength":0.9,"seed":42}"#
        );
    }

    {
        let req = ImageCreateRequestBuilder::new("test-model-name", "This is a prompt")
            .with_number_of_images(2)
            .with_response_format(ResponseFormat::B64Json)
            .with_style("vivid")
            .with_user("user")
            .with_cfg_scale(1.0)
            .with_sample_method(SamplingMethod::Euler)
            .with_steps(4)
            .build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"prompt":"This is a prompt","negative_prompt":"","model":"test-model-name","n":2,"quality":"standard","response_format":"b64_json","style":"vivid","user":"user","cfg_scale":1.0,"sample_method":"euler","steps":4,"height":512,"width":512,"control_strength":0.9,"seed":42}"#
        );
    }
}

#[test]
fn test_deserialize_image_create_request() {
    {
        let json = r#"{"prompt":"This is a prompt","negative_prompt":"This is the negative prompt.","model":"test-model-name"}"#;
        let req: ImageCreateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "This is a prompt");
        assert!(req.negative_prompt.is_some());
        assert_eq!(
            req.negative_prompt,
            Some("This is the negative prompt.".to_string())
        );
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(1));
        assert_eq!(req.response_format, Some(ResponseFormat::Url));
        assert_eq!(req.cfg_scale, Some(7.0));
        assert_eq!(req.sample_method, Some(SamplingMethod::EulerA));
        assert_eq!(req.steps, Some(20));
        assert_eq!(req.height, Some(512));
        assert_eq!(req.width, Some(512));
    }

    {
        let json = r#"{"prompt":"This is a prompt","model":"test-model-name","n":2,"response_format":"url","size":"1024x1024","style":"vivid","user":"user","cfg_scale":1.0,"sample_method":"euler","steps":4}"#;
        let req: ImageCreateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "This is a prompt");
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(2));
        assert_eq!(req.response_format, Some(ResponseFormat::Url));
        assert_eq!(req.size, Some("1024x1024".to_string()));
        assert_eq!(req.style, Some("vivid".to_string()));
        assert_eq!(req.user, Some("user".to_string()));
        assert_eq!(req.cfg_scale, Some(1.0));
        assert_eq!(req.sample_method, Some(SamplingMethod::Euler));
        assert_eq!(req.steps, Some(4));
        assert_eq!(req.height, Some(1024));
        assert_eq!(req.width, Some(1024));
    }

    {
        let json = r#"{"prompt":"This is a prompt","model":"test-model-name","n":2,"response_format":"url","size":"1024x1024","style":"vivid","user":"user","cfg_scale":1.0,"sample_method":"euler","steps":4,"height":512,"width":512}"#;
        let req: ImageCreateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "This is a prompt");
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(2));
        assert_eq!(req.response_format, Some(ResponseFormat::Url));
        assert_eq!(req.size, Some("1024x1024".to_string()));
        assert_eq!(req.style, Some("vivid".to_string()));
        assert_eq!(req.user, Some("user".to_string()));
        assert_eq!(req.cfg_scale, Some(1.0));
        assert_eq!(req.sample_method, Some(SamplingMethod::Euler));
        assert_eq!(req.steps, Some(4));
        assert_eq!(req.height, Some(1024));
        assert_eq!(req.width, Some(1024));
    }
}

/// Builder for creating a `ImageEditRequest` instance.
pub struct ImageEditRequestBuilder {
    req: ImageEditRequest,
}
impl ImageEditRequestBuilder {
    /// Create a new builder with the given image, prompt, and mask.
    pub fn new(model: impl Into<String>, image: FileObject, prompt: impl Into<String>) -> Self {
        Self {
            req: ImageEditRequest {
                model: model.into(),
                image,
                prompt: prompt.into(),
                ..Default::default()
            },
        }
    }

    /// Set negative prompt
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.req.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Set an additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. Must have the same dimensions as `image`.
    pub fn with_mask(mut self, mask: FileObject) -> Self {
        self.req.mask = Some(mask);
        self
    }

    /// Set the number of images to generate.
    pub fn with_number_of_images(mut self, n: u64) -> Self {
        self.req.n = Some(n);
        self
    }

    /// Set the size of the generated images.
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.req.size = Some(size.into());
        self
    }

    /// Set the format in which the generated images are returned.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.req.response_format = Some(response_format);
        self
    }

    /// Set the user id
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.req.user = Some(user.into());
        self
    }

    /// Set the unconditional guidance scale. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_cfg_scale(mut self, cfg_scale: f32) -> Self {
        self.req.cfg_scale = Some(cfg_scale);
        self
    }

    /// Set the sampling method. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_sample_method(mut self, sample_method: SamplingMethod) -> Self {
        self.req.sample_method = Some(sample_method);
        self
    }

    /// Set the number of sample steps. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.req.steps = Some(steps);
        self
    }

    /// Set the image size.
    pub fn with_image_size(mut self, height: usize, width: usize) -> Self {
        self.req.height = Some(height);
        self.req.width = Some(width);
        self
    }

    /// Set the strength to apply Control Net. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_control_strength(mut self, control_strength: f32) -> Self {
        self.req.control_strength = Some(control_strength);
        self
    }

    /// Set the RNG seed. Negative value means to use random seed. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_seed(mut self, seed: i32) -> Self {
        self.req.seed = Some(seed);
        self
    }

    /// Set the strength for noising/unnoising. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.req.strength = Some(strength);
        self
    }

    /// Build the request.
    pub fn build(self) -> ImageEditRequest {
        self.req
    }
}

/// Request to create an edited or extended image given an original image and a prompt.
#[derive(Debug, Serialize)]
pub struct ImageEditRequest {
    /// The image to edit. If mask is not provided, image must have transparency, which will be used as the mask.
    pub image: FileObject,
    /// A text description of the desired image(s).
    pub prompt: String,
    /// Negative prompt for the image generation.
    pub negative_prompt: Option<String>,
    /// An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. Must have the same dimensions as `image`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<FileObject>,
    /// The model to use for image generation.
    pub model: String,
    /// The number of images to generate. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    /// The size of the generated images. Defaults to 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The format in which the generated images are returned. Must be one of `url` or `b64_json`. Defaults to `url`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// A unique identifier representing your end-user, which can help monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Unconditional guidance scale. Defaults to 7.0. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    /// Sampling method. Defaults to "euler_a". This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_method: Option<SamplingMethod>,
    /// Number of sample steps. Defaults to 20. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<usize>,
    /// Image height, in pixel space. Defaults to 512. If `size` is provided, this field will be ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<usize>,
    /// Image width, in pixel space. Defaults to 512. If `size` is provided, this field will be ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    /// strength to apply Control Net. Defaults to 0.9. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_strength: Option<f32>,
    /// RNG seed. Negative value means to use random seed. Defaults to 42. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    /// Strength for noising/unnoising. Defaults to 0.75. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
}
impl<'de> Deserialize<'de> for ImageEditRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Image,
            Prompt,
            NegativePrompt,
            Mask,
            Model,
            N,
            Size,
            ResponseFormat,
            User,
            CfgScale,
            SampleMethod,
            Steps,
            Height,
            Width,
            ControlStrength,
            Seed,
            Strength,
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
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "image" => Ok(Field::Image),
                            "prompt" => Ok(Field::Prompt),
                            "negative_prompt" => Ok(Field::NegativePrompt),
                            "mask" => Ok(Field::Mask),
                            "model" => Ok(Field::Model),
                            "n" => Ok(Field::N),
                            "size" => Ok(Field::Size),
                            "response_format" => Ok(Field::ResponseFormat),
                            "user" => Ok(Field::User),
                            "cfg_scale" => Ok(Field::CfgScale),
                            "sample_method" => Ok(Field::SampleMethod),
                            "steps" => Ok(Field::Steps),
                            "height" => Ok(Field::Height),
                            "width" => Ok(Field::Width),
                            "control_strength" => Ok(Field::ControlStrength),
                            "seed" => Ok(Field::Seed),
                            "strength" => Ok(Field::Strength),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct ImageEditRequestVisitor;

        impl<'de> Visitor<'de> for ImageEditRequestVisitor {
            type Value = ImageEditRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ImageEditRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ImageEditRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut image = None;
                let mut prompt = None;
                let mut negative_prompt = None;
                let mut mask = None;
                let mut model = None;
                let mut n = None;
                let mut size: Option<String> = None;
                let mut response_format = None;
                let mut user = None;
                let mut cfg_scale = None;
                let mut sample_method = None;
                let mut steps = None;
                let mut height = None;
                let mut width = None;
                let mut control_strength = None;
                let mut seed = None;
                let mut strength = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Image => {
                            if image.is_some() {
                                return Err(de::Error::duplicate_field("image"));
                            }
                            image = Some(map.next_value()?);
                        }
                        Field::Prompt => {
                            if prompt.is_some() {
                                return Err(de::Error::duplicate_field("prompt"));
                            }
                            prompt = Some(map.next_value()?);
                        }
                        Field::NegativePrompt => {
                            if negative_prompt.is_some() {
                                return Err(de::Error::duplicate_field("negative_prompt"));
                            }
                            negative_prompt = Some(map.next_value()?);
                        }
                        Field::Mask => {
                            if mask.is_some() {
                                return Err(de::Error::duplicate_field("mask"));
                            }
                            mask = Some(map.next_value()?);
                        }
                        Field::Model => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        Field::N => {
                            if n.is_some() {
                                return Err(de::Error::duplicate_field("n"));
                            }
                            n = Some(map.next_value()?);
                        }
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                        Field::ResponseFormat => {
                            if response_format.is_some() {
                                return Err(de::Error::duplicate_field("response_format"));
                            }
                            response_format = Some(map.next_value()?);
                        }
                        Field::User => {
                            if user.is_some() {
                                return Err(de::Error::duplicate_field("user"));
                            }
                            user = Some(map.next_value()?);
                        }
                        Field::CfgScale => {
                            if cfg_scale.is_some() {
                                return Err(de::Error::duplicate_field("cfg_scale"));
                            }
                            cfg_scale = Some(map.next_value()?);
                        }
                        Field::SampleMethod => {
                            if sample_method.is_some() {
                                return Err(de::Error::duplicate_field("sample_method"));
                            }
                            sample_method = Some(map.next_value()?);
                        }
                        Field::Steps => {
                            if steps.is_some() {
                                return Err(de::Error::duplicate_field("steps"));
                            }
                            steps = Some(map.next_value()?);
                        }
                        Field::Height => {
                            if height.is_some() {
                                return Err(de::Error::duplicate_field("height"));
                            }
                            height = Some(map.next_value()?);
                        }
                        Field::Width => {
                            if width.is_some() {
                                return Err(de::Error::duplicate_field("width"));
                            }
                            width = Some(map.next_value()?);
                        }
                        Field::ControlStrength => {
                            if control_strength.is_some() {
                                return Err(de::Error::duplicate_field("control_strength"));
                            }
                            control_strength = Some(map.next_value()?);
                        }
                        Field::Seed => {
                            if seed.is_some() {
                                return Err(de::Error::duplicate_field("seed"));
                            }
                            seed = Some(map.next_value()?);
                        }
                        Field::Strength => {
                            if strength.is_some() {
                                return Err(de::Error::duplicate_field("strength"));
                            }
                            strength = Some(map.next_value()?);
                        }
                    }
                }

                if negative_prompt.is_none() {
                    negative_prompt = Some("".to_string());
                }

                if n.is_none() {
                    n = Some(1);
                }

                if response_format.is_none() {
                    response_format = Some(ResponseFormat::Url);
                }

                if cfg_scale.is_none() {
                    cfg_scale = Some(7.0);
                }

                if sample_method.is_none() {
                    sample_method = Some(SamplingMethod::EulerA);
                }

                if steps.is_none() {
                    steps = Some(20);
                }

                if control_strength.is_none() {
                    control_strength = Some(0.9);
                }

                if seed.is_none() {
                    seed = Some(42);
                }

                if strength.is_none() {
                    strength = Some(0.75);
                }

                match &size {
                    Some(size) => {
                        let parts: Vec<&str> = size.split('x').collect();
                        if parts.len() != 2 {
                            return Err(de::Error::custom("invalid size format"));
                        }
                        height = Some(parts[0].parse().unwrap());
                        width = Some(parts[1].parse().unwrap());
                    }
                    None => {
                        if height.is_none() {
                            height = Some(512);
                        }
                        if width.is_none() {
                            width = Some(512);
                        }
                    }
                }

                Ok(ImageEditRequest {
                    image: image.ok_or_else(|| de::Error::missing_field("image"))?,
                    prompt: prompt.ok_or_else(|| de::Error::missing_field("prompt"))?,
                    negative_prompt,
                    mask,
                    model: model.ok_or_else(|| de::Error::missing_field("model"))?,
                    n,
                    size,
                    response_format,
                    user,
                    cfg_scale,
                    sample_method,
                    steps,
                    height,
                    width,
                    control_strength,
                    seed,
                    strength,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "image",
            "prompt",
            "negative_prompt",
            "mask",
            "model",
            "n",
            "size",
            "response_format",
            "user",
            "cfg_scale",
            "sample_method",
            "steps",
            "height",
            "width",
            "control_strength",
            "seed",
            "strength",
        ];
        deserializer.deserialize_struct("ImageEditRequest", FIELDS, ImageEditRequestVisitor)
    }
}
impl Default for ImageEditRequest {
    fn default() -> Self {
        Self {
            image: FileObject::default(),
            prompt: "".to_string(),
            negative_prompt: Some("".to_string()),
            mask: None,
            model: "".to_string(),
            n: Some(1),
            response_format: Some(ResponseFormat::Url),
            size: None,
            user: None,
            cfg_scale: Some(7.0),
            sample_method: Some(SamplingMethod::EulerA),
            steps: Some(20),
            height: Some(512),
            width: Some(512),
            control_strength: Some(0.9),
            seed: Some(42),
            strength: Some(0.75),
        }
    }
}

#[test]
fn test_serialize_image_edit_request() {
    {
        let req = ImageEditRequestBuilder::new(
            "test-model-name",
            FileObject {
                id: "test-image-id".to_string(),
                bytes: 1024,
                created_at: 1234567890,
                filename: "test-image.png".to_string(),
                object: "file".to_string(),
                purpose: "fine-tune".to_string(),
            },
            "This is a prompt",
        )
        .build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","negative_prompt":"","model":"test-model-name","n":1,"response_format":"url","cfg_scale":7.0,"sample_method":"euler_a","steps":20,"height":512,"width":512,"control_strength":0.9,"seed":42,"strength":0.75}"#
        );
    }

    {
        let req = ImageEditRequestBuilder::new(
            "test-model-name",
            FileObject {
                id: "test-image-id".to_string(),
                bytes: 1024,
                created_at: 1234567890,
                filename: "test-image.png".to_string(),
                object: "file".to_string(),
                purpose: "fine-tune".to_string(),
            },
            "This is a prompt",
        )
        .with_number_of_images(2)
        .with_response_format(ResponseFormat::B64Json)
        .with_size("256x256")
        .with_user("user")
        .build();
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","negative_prompt":"","model":"test-model-name","n":2,"size":"256x256","response_format":"b64_json","user":"user","cfg_scale":7.0,"sample_method":"euler_a","steps":20,"height":512,"width":512,"control_strength":0.9,"seed":42,"strength":0.75}"#
        );
    }
}

#[test]
fn test_deserialize_image_edit_request() {
    {
        let json = r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","model":"test-model-name"}"#;
        let req: ImageEditRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image.id, "test-image-id");
        assert_eq!(req.image.bytes, 1024);
        assert_eq!(req.image.created_at, 1234567890);
        assert_eq!(req.image.filename, "test-image.png");
        assert_eq!(req.image.object, "file");
        assert_eq!(req.image.purpose, "fine-tune");
        assert_eq!(req.prompt, "This is a prompt");
        assert!(req.mask.is_none());
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(1));
        assert_eq!(req.response_format, Some(ResponseFormat::Url));
    }

    {
        let json = r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","model":"test-model-name","n":2,"size":"256x256","response_format":"b64_json","user":"user"}"#;
        let req: ImageEditRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image.id, "test-image-id");
        assert_eq!(req.image.bytes, 1024);
        assert_eq!(req.image.created_at, 1234567890);
        assert_eq!(req.image.filename, "test-image.png");
        assert_eq!(req.image.object, "file");
        assert_eq!(req.image.purpose, "fine-tune");
        assert_eq!(req.prompt, "This is a prompt");
        assert!(req.mask.is_none());
        assert_eq!(req.model, "test-model-name");
        assert_eq!(req.n, Some(2));
        assert_eq!(req.size, Some("256x256".to_string()));
        assert_eq!(req.response_format, Some(ResponseFormat::B64Json));
        assert_eq!(req.user, Some("user".to_string()));
    }
}

/// Request to generate an image variation.
#[derive(Debug, Serialize, Default)]
pub struct ImageVariationRequest {
    /// The image to use as the basis for the variation(s).
    pub image: FileObject,
    /// Name of the model to use for image generation.
    pub model: String,
    /// The number of images to generate. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    /// The format in which the generated images are returned. Must be one of `url` or `b64_json`. Defaults to `b64_json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// The size of the generated images. Defaults to 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// A unique identifier representing your end-user, which can help monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}
impl<'de> Deserialize<'de> for ImageVariationRequest {
    fn deserialize<D>(deserializer: D) -> Result<ImageVariationRequest, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Image,
            Model,
            N,
            ResponseFormat,
            Size,
            User,
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
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "image" => Ok(Field::Image),
                            "model" => Ok(Field::Model),
                            "n" => Ok(Field::N),
                            "response_format" => Ok(Field::ResponseFormat),
                            "size" => Ok(Field::Size),
                            "user" => Ok(Field::User),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct ImageVariationRequestVisitor;

        impl<'de> Visitor<'de> for ImageVariationRequestVisitor {
            type Value = ImageVariationRequest;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ImageVariationRequest")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ImageVariationRequest, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut image = None;
                let mut model = None;
                let mut n = None;
                let mut response_format = None;
                let mut size = None;
                let mut user = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Image => {
                            if image.is_some() {
                                return Err(de::Error::duplicate_field("image"));
                            }
                            image = Some(map.next_value()?);
                        }
                        Field::Model => {
                            if model.is_some() {
                                return Err(de::Error::duplicate_field("model"));
                            }
                            model = Some(map.next_value()?);
                        }
                        Field::N => {
                            if n.is_some() {
                                return Err(de::Error::duplicate_field("n"));
                            }
                            n = Some(map.next_value()?);
                        }
                        Field::ResponseFormat => {
                            if response_format.is_some() {
                                return Err(de::Error::duplicate_field("response_format"));
                            }
                            response_format = Some(map.next_value()?);
                        }
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                        Field::User => {
                            if user.is_some() {
                                return Err(de::Error::duplicate_field("user"));
                            }
                            user = Some(map.next_value()?);
                        }
                    }
                }
                Ok(ImageVariationRequest {
                    image: image.ok_or_else(|| de::Error::missing_field("image"))?,
                    model: model.ok_or_else(|| de::Error::missing_field("model"))?,
                    n: n.unwrap_or(Some(1)),
                    response_format: response_format.unwrap_or(Some(ResponseFormat::B64Json)),
                    size,
                    user,
                })
            }
        }

        const FIELDS: &[&str] = &["image", "model", "n", "response_format", "size", "user"];
        deserializer.deserialize_struct(
            "ImageVariationRequest",
            FIELDS,
            ImageVariationRequestVisitor,
        )
    }
}

/// The format in which the generated images are returned.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    #[serde(rename = "url")]
    Url,
    #[serde(rename = "b64_json")]
    B64Json,
}
impl FromStr for ResponseFormat {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "url" => Ok(ResponseFormat::Url),
            "b64_json" => Ok(ResponseFormat::B64Json),
            _ => Err(ParseError),
        }
    }
}

// Custom error type for conversion errors
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError;
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "provided string did not match any ResponseFormat variants"
        )
    }
}

/// Represents the url or the content of an image generated.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct ImageObject {
    /// The base64-encoded JSON of the generated image, if response_format is `b64_json`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    /// The URL of the generated image, if response_format is `url`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The prompt that was used to generate the image, if there was any revision to the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

/// Represent the response from the `images` endpoint.
#[derive(Debug, Deserialize, Serialize)]
pub struct ListImagesResponse {
    /// The Unix timestamp (in seconds) for when the response was created.
    pub created: u64,
    /// The list of file objects.
    pub data: Vec<ImageObject>,
}
