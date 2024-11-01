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

    /// Set the image to control the generation. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_control_image(mut self, control_image: FileObject) -> Self {
        self.req.control_image = Some(control_image);
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

    /// Set the denoiser sigma scheduler. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_scheduler(mut self, scheduler: Scheduler) -> Self {
        self.req.scheduler = Some(scheduler);
        self
    }

    /// Set whether to apply the canny preprocessor.
    pub fn apply_canny_preprocessor(mut self, apply_canny_preprocessor: bool) -> Self {
        self.req.apply_canny_preprocessor = Some(apply_canny_preprocessor);
        self
    }

    /// Set the strength for keeping input identity.
    pub fn with_style_ratio(mut self, style_ratio: f32) -> Self {
        self.req.style_ratio = Some(style_ratio);
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
    /// Strength to apply Control Net. Defaults to 0.9. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_strength: Option<f32>,
    /// The image to control the generation. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_image: Option<FileObject>,
    /// RNG seed. Negative value means to use random seed. Defaults to 42. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    /// Strength for noising/unnoising. Defaults to 0.75. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    /// Denoiser sigma scheduler. Possible values are `discrete`, `karras`, `exponential`, `ays`, `gits`. Defaults to `discrete`. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<Scheduler>,
    /// Apply canny preprocessor. Defaults to false. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_canny_preprocessor: Option<bool>,
    /// Strength for keeping input identity. Defaults to 0.2. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style_ratio: Option<f32>,
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
            ControlImage,
            Seed,
            Strength,
            Scheduler,
            ApplyCannyPreprocessor,
            StyleRatio,
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
                            "control_image" => Ok(Field::ControlImage),
                            "seed" => Ok(Field::Seed),
                            "strength" => Ok(Field::Strength),
                            "scheduler" => Ok(Field::Scheduler),
                            "apply_canny_preprocessor" => Ok(Field::ApplyCannyPreprocessor),
                            "style_ratio" => Ok(Field::StyleRatio),
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
                let quality = seq.next_element().unwrap_or(Some("standard".to_string()));
                let response_format = seq.next_element().unwrap_or(Some(ResponseFormat::Url));
                let size = seq.next_element().unwrap_or(None);
                let style = seq.next_element().unwrap_or(Some("natural".to_string()));
                let user = seq.next_element().unwrap_or(None);
                let cfg_scale = seq.next_element().unwrap_or(Some(7.0));
                let sample_method = seq.next_element().unwrap_or(Some(SamplingMethod::EulerA));
                let steps = seq.next_element().unwrap_or(Some(20));
                let height = seq.next_element().unwrap_or(Some(512));
                let width = seq.next_element().unwrap_or(Some(512));
                let control_strength = seq.next_element().unwrap_or(Some(0.9));
                let control_image = seq.next_element()?;
                let seed = seq.next_element().unwrap_or(Some(42));
                let strength = seq.next_element().unwrap_or(Some(0.75));
                let scheduler = seq.next_element()?.unwrap_or(Some(Scheduler::Discrete));
                let apply_canny_preprocessor = seq.next_element().unwrap_or(Some(false));
                let style_ratio = seq.next_element().unwrap_or(Some(0.2));
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
                    control_image,
                    seed,
                    strength,
                    scheduler,
                    apply_canny_preprocessor,
                    style_ratio,
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
                let mut control_image = None;
                let mut seed = None;
                let mut strength = None;
                let mut scheduler = None;
                let mut apply_canny_preprocessor = None;
                let mut style_ratio = None;
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
                        Field::ControlImage => {
                            if control_image.is_some() {
                                return Err(de::Error::duplicate_field("control_image"));
                            }
                            control_image = Some(map.next_value()?);
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
                        Field::Scheduler => {
                            if scheduler.is_some() {
                                return Err(de::Error::duplicate_field("scheduler"));
                            }
                            scheduler = Some(map.next_value()?);
                        }
                        Field::ApplyCannyPreprocessor => {
                            if apply_canny_preprocessor.is_some() {
                                return Err(de::Error::duplicate_field("apply_canny_preprocessor"));
                            }
                            apply_canny_preprocessor = Some(map.next_value()?);
                        }
                        Field::StyleRatio => {
                            if style_ratio.is_some() {
                                return Err(de::Error::duplicate_field("style_ratio"));
                            }
                            style_ratio = Some(map.next_value()?);
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

                if strength.is_none() {
                    strength = Some(0.75);
                }

                if scheduler.is_none() {
                    scheduler = Some(Scheduler::Discrete);
                }

                if apply_canny_preprocessor.is_none() {
                    apply_canny_preprocessor = Some(false);
                }

                if style_ratio.is_none() {
                    style_ratio = Some(0.2);
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
                    control_image,
                    seed,
                    strength,
                    scheduler,
                    apply_canny_preprocessor,
                    style_ratio,
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
            "control_strength",
            "control_image",
            "seed",
            "strength",
            "scheduler",
            "apply_canny_preprocessor",
            "style_ratio",
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
            control_image: None,
            seed: Some(42),
            strength: Some(0.75),
            scheduler: Some(Scheduler::Discrete),
            apply_canny_preprocessor: Some(false),
            style_ratio: Some(0.2),
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
            r#"{"prompt":"This is a prompt","negative_prompt":"This is the negative prompt.","model":"test-model-name","n":1,"quality":"standard","response_format":"url","style":"natural","cfg_scale":7.0,"sample_method":"euler_a","steps":20,"height":512,"width":512,"control_strength":0.9,"seed":42,"strength":0.75,"scheduler":"discrete","apply_canny_preprocessor":false,"style_ratio":0.2}"#
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
            r#"{"prompt":"This is a prompt","negative_prompt":"","model":"test-model-name","n":2,"quality":"standard","response_format":"b64_json","style":"vivid","user":"user","cfg_scale":1.0,"sample_method":"euler","steps":4,"height":512,"width":512,"control_strength":0.9,"seed":42,"strength":0.75,"scheduler":"discrete","apply_canny_preprocessor":false,"style_ratio":0.2}"#
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

    /// Set the image to control the generation. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_control_image(mut self, control_image: FileObject) -> Self {
        self.req.control_image = Some(control_image);
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

    /// Set the denoiser sigma scheduler. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_scheduler(mut self, scheduler: Scheduler) -> Self {
        self.req.scheduler = Some(scheduler);
        self
    }

    /// Set whether to apply the canny preprocessor. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_apply_canny_preprocessor(mut self, apply_canny_preprocessor: bool) -> Self {
        self.req.apply_canny_preprocessor = Some(apply_canny_preprocessor);
        self
    }

    /// Set the strength for keeping input identity. This param is only supported for `stable-diffusion.cpp`.
    pub fn with_style_ratio(mut self, style_ratio: f32) -> Self {
        self.req.style_ratio = Some(style_ratio);
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
    /// The image to control the generation. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_image: Option<FileObject>,
    /// RNG seed. Negative value means to use random seed. Defaults to 42. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    /// Strength for noising/unnoising. Defaults to 0.75. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    /// Denoiser sigma scheduler. Possible values are `discrete`, `karras`, `exponential`, `ays`, `gits`. Defaults to `discrete`. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<Scheduler>,
    /// Apply canny preprocessor. Defaults to false. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apply_canny_preprocessor: Option<bool>,
    /// Strength for keeping input identity. Defaults to 0.2. This param is only supported for `stable-diffusion.cpp`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style_ratio: Option<f32>,
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
            ControlImage,
            Seed,
            Strength,
            Scheduler,
            ApplyCannyPreprocessor,
            StyleRatio,
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
                            "control_image" => Ok(Field::ControlImage),
                            "seed" => Ok(Field::Seed),
                            "strength" => Ok(Field::Strength),
                            "scheduler" => Ok(Field::Scheduler),
                            "apply_canny_preprocessor" => Ok(Field::ApplyCannyPreprocessor),
                            "style_ratio" => Ok(Field::StyleRatio),
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
                let mut control_image = None;
                let mut seed = None;
                let mut strength = None;
                let mut scheduler = None;
                let mut apply_canny_preprocessor = None;
                let mut style_ratio = None;

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
                        Field::ControlImage => {
                            if control_image.is_some() {
                                return Err(de::Error::duplicate_field("control_image"));
                            }
                            control_image = Some(map.next_value()?);
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
                        Field::Scheduler => {
                            if scheduler.is_some() {
                                return Err(de::Error::duplicate_field("scheduler"));
                            }
                            scheduler = Some(map.next_value()?);
                        }
                        Field::ApplyCannyPreprocessor => {
                            if apply_canny_preprocessor.is_some() {
                                return Err(de::Error::duplicate_field("apply_canny_preprocessor"));
                            }
                            apply_canny_preprocessor = Some(map.next_value()?);
                        }
                        Field::StyleRatio => {
                            if style_ratio.is_some() {
                                return Err(de::Error::duplicate_field("style_ratio"));
                            }
                            style_ratio = Some(map.next_value()?);
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

                if scheduler.is_none() {
                    scheduler = Some(Scheduler::Discrete);
                }

                if apply_canny_preprocessor.is_none() {
                    apply_canny_preprocessor = Some(false);
                }

                if style_ratio.is_none() {
                    style_ratio = Some(0.2);
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
                    control_image,
                    seed,
                    strength,
                    scheduler,
                    apply_canny_preprocessor,
                    style_ratio,
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
            "control_image",
            "seed",
            "strength",
            "scheduler",
            "apply_canny_preprocessor",
            "style_ratio",
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
            control_image: None,
            seed: Some(42),
            strength: Some(0.75),
            scheduler: Some(Scheduler::Discrete),
            apply_canny_preprocessor: Some(false),
            style_ratio: Some(0.2),
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
            r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","negative_prompt":"","model":"test-model-name","n":1,"response_format":"url","cfg_scale":7.0,"sample_method":"euler_a","steps":20,"height":512,"width":512,"control_strength":0.9,"seed":42,"strength":0.75,"scheduler":"discrete","apply_canny_preprocessor":false,"style_ratio":0.2}"#
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
            r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","negative_prompt":"","model":"test-model-name","n":2,"size":"256x256","response_format":"b64_json","user":"user","cfg_scale":7.0,"sample_method":"euler_a","steps":20,"height":512,"width":512,"control_strength":0.9,"seed":42,"strength":0.75,"scheduler":"discrete","apply_canny_preprocessor":false,"style_ratio":0.2}"#
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
        assert_eq!(req.cfg_scale, Some(7.0));
        assert_eq!(req.sample_method, Some(SamplingMethod::EulerA));
        assert_eq!(req.steps, Some(20));
        assert_eq!(req.height, Some(512));
        assert_eq!(req.width, Some(512));
        assert_eq!(req.control_strength, Some(0.9));
        assert!(req.control_image.is_none());
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.strength, Some(0.75));
        assert_eq!(req.scheduler, Some(Scheduler::Discrete));
    }

    {
        let json = r#"{"image":{"id":"test-image-id","bytes":1024,"created_at":1234567890,"filename":"test-image.png","object":"file","purpose":"fine-tune"},"prompt":"This is a prompt","model":"test-model-name","n":2,"size":"256x256","response_format":"b64_json","user":"user","cfg_scale":7.9,"sample_method":"euler","steps":25, "scheduler":"karras"}"#;
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
        assert_eq!(req.cfg_scale, Some(7.9));
        assert_eq!(req.sample_method, Some(SamplingMethod::Euler));
        assert_eq!(req.steps, Some(25));
        assert_eq!(req.height, Some(256));
        assert_eq!(req.width, Some(256));
        assert_eq!(req.control_strength, Some(0.9));
        assert!(req.control_image.is_none());
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.strength, Some(0.75));
        assert_eq!(req.scheduler, Some(Scheduler::Karras));
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

/// Scheduler type
#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum Scheduler {
    #[serde(rename = "discrete")]
    Discrete,
    #[serde(rename = "karras")]
    Karras,
    #[serde(rename = "exponential")]
    Exponential,
    #[serde(rename = "ays")]
    Ays,
    #[serde(rename = "gits")]
    Gits,
}
impl FromStr for Scheduler {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "discrete" => Ok(Scheduler::Discrete),
            "karras" => Ok(Scheduler::Karras),
            "exponential" => Ok(Scheduler::Exponential),
            "ays" => Ok(Scheduler::Ays),
            "gits" => Ok(Scheduler::Gits),
            _ => Err(ParseError),
        }
    }
}
impl From<&str> for Scheduler {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "discrete" => Scheduler::Discrete,
            "karras" => Scheduler::Karras,
            "exponential" => Scheduler::Exponential,
            "ays" => Scheduler::Ays,
            "gits" => Scheduler::Gits,
            _ => Scheduler::Discrete,
        }
    }
}
impl<'de> Deserialize<'de> for Scheduler {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SchedulerVisitor;

        impl Visitor<'_> for SchedulerVisitor {
            type Value = Scheduler;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string representing a scheduler type")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value.to_lowercase().as_str() {
                    "discrete" => Ok(Scheduler::Discrete),
                    "karras" => Ok(Scheduler::Karras),
                    "exponential" => Ok(Scheduler::Exponential),
                    "ays" => Ok(Scheduler::Ays),
                    "gits" => Ok(Scheduler::Gits),
                    _ => Err(E::custom(format!(
                        "unknown scheduler type: {}, expected one of: discrete, karras, exponential, ays, gits",
                        value
                    ))),
                }
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }
        }

        deserializer.deserialize_str(SchedulerVisitor)
    }
}

pub mod sd_webui {
    use super::Scheduler;
    use serde::{
        de::{self, MapAccess, Visitor},
        Deserialize, Deserializer, Serialize,
    };
    use std::fmt;

    #[derive(Serialize, Debug)]
    pub struct Txt2ImgRequest {
        /// A text description of the desired image.
        pub prompt: String,
        /// Negative prompt for the image generation. Defaults to "".
        #[serde(skip_serializing_if = "Option::is_none")]
        pub negative_prompt: Option<String>,
        /// Name of the model to use for image generation. Defaults to -1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed: Option<i64>,
        /// Subseed for the image generation. Defaults to -1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub subseed: Option<i64>,
        /// Subseed strength for the image generation. Defaults to 0.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub subseed_strength: Option<f64>,
        /// Seed resize from H. Defaults to -1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed_resize_from_h: Option<i64>,
        /// Seed resize from W. Defaults to -1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed_resize_from_w: Option<i64>,
        /// Sampler name. Possible values are `Euler`, `Euler a`, `Heun`, `DPM2`, `DPM2 a`, `DPM fast`, `DPM adaptive`, `DPM++ 2S a`, `DPM++ 2M`, `DPM++ 2M SDE`, `DPM++ 2M SDE Heun`, `DPM++ 3M SDE`, `DPM++ SDE`, `LCM`, `LMS`, `Restart`, `DDIM`, `DDIM CFG++`, `PLMS`, `UniPC`. Defaults to `Euler`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sampler_name: Option<Sampler>,
        /// Denoiser sigma scheduler. Possible values are `discrete`, `karras`, `exponential`, `ays`, `gits`. Defaults to `discrete`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub scheduler: Option<Scheduler>,
        /// The number of images to generate. Defaults to 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub batch_size: Option<u32>,
        /// Number of iterations. Defaults to 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n_iter: Option<u32>,
        /// Number of sample steps. Defaults to 20.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub steps: Option<u32>,
        /// Unconditional guidance scale. Defaults to 7.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cfg_scale: Option<f64>,
        /// Image width, in pixel space. Defaults to 512.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub width: Option<u32>,
        /// Image height, in pixel space. Defaults to 512.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub height: Option<u32>,
        /// Restore faces. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub restore_faces: Option<bool>,
        /// Tiling. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tiling: Option<bool>,
        /// Do not save samples. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub do_not_save_samples: Option<bool>,
        /// Do not save grid. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub do_not_save_grid: Option<bool>,
        /// Eta for the image generation.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub eta: Option<f64>,
        /// Denoising strength.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub denoising_strength: Option<f64>,
        /// S Min Uncond.
        pub s_min_uncond: Option<f64>,
        /// S Churn.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub s_churn: Option<f64>,
        /// S Tmax.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub s_tmax: Option<f64>,
        /// S Tmin.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub s_tmin: Option<f64>,
        /// S Noise.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub s_noise: Option<f64>,
        /// Override settings.
        pub override_settings: OverrideSettings,
        /// Override Settings Restore Afterwards. Defaults to true.
        pub override_settings_restore_afterwards: Option<bool>,
        /// Refiner checkpoint.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub refiner_checkpoint: Option<String>,
        /// Refiner switch at.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub refiner_switch_at: Option<f64>,
        /// Disable extra networks. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub disable_extra_networks: Option<bool>,
        /// Firstpass image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub firstpass_image: Option<String>,
        /// Enable Hr. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub enable_hr: Option<bool>,
        /// Firstphase Width. Defaults to 0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub firstphase_width: Option<u32>,
        /// Firstphase Height. Defaults to 0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub firstphase_height: Option<u32>,
        /// Hr scale. Defaults to 2.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_scale: Option<f64>,
        /// Hr Upscaler.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_upscaler: Option<String>,
        /// Hr Second Pass Steps. Defaults to 0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_second_pass_steps: Option<u32>,
        /// Hr Resize X. Defaults to 0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_resize_x: Option<u32>,
        /// Hr Resize Y. Defaults to 0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_resize_y: Option<u32>,
        /// Hr Checkpoint Name.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_checkpoint_name: Option<String>,
        /// Hr Sampler Name.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_sampler_name: Option<String>,
        /// Hr Scheduler.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_scheduler: Option<String>,
        /// Hr Prompt. Defaults to "".
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_prompt: Option<String>,
        /// Hr Negative Prompt. Defaults to "".
        #[serde(skip_serializing_if = "Option::is_none")]
        pub hr_negative_prompt: Option<String>,
        /// Force Task Id.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub force_task_id: Option<String>,
        /// Sampler index. Possible values are `Euler`, `Euler a`, `Heun`, `DPM2`, `DPM2 a`, `DPM fast`, `DPM adaptive`, `DPM++ 2S a`, `DPM++ 2M`, `DPM++ 2M SDE`, `DPM++ 2M SDE Heun`, `DPM++ 3M SDE`, `DPM++ SDE`, `LCM`, `LMS`, `Restart`, `DDIM`, `DDIM CFG++`, `PLMS`, `UniPC`. Defaults to `Euler`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sampler_index: Option<Sampler>,
        /// Send images. Defaults to true.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub send_images: Option<bool>,
        /// Save images. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub save_images: Option<bool>,
        /// Alwayson scripts.
        pub alwayson_scripts: AlwaysOnScripts,
    }
    impl Default for Txt2ImgRequest {
        fn default() -> Self {
            Self {
                prompt: "".to_string(),
                negative_prompt: Some("".to_string()),
                seed: Some(-1),
                subseed: Some(-1),
                subseed_strength: Some(0.0),
                seed_resize_from_h: Some(-1),
                seed_resize_from_w: Some(-1),
                sampler_name: None,
                scheduler: Some(Scheduler::Discrete),
                batch_size: Some(1),
                n_iter: Some(1),
                steps: Some(20),
                cfg_scale: Some(7.0),
                width: Some(512),
                height: Some(512),
                restore_faces: Some(false),
                tiling: Some(false),
                do_not_save_samples: Some(false),
                do_not_save_grid: Some(false),
                eta: None,
                denoising_strength: None,
                s_min_uncond: None,
                s_churn: None,
                s_tmax: None,
                s_tmin: None,
                s_noise: None,
                override_settings: OverrideSettings::default(),
                override_settings_restore_afterwards: Some(true),
                refiner_checkpoint: None,
                refiner_switch_at: None,
                disable_extra_networks: None,
                firstpass_image: None,
                enable_hr: Some(false),
                firstphase_width: Some(0),
                firstphase_height: Some(0),
                hr_scale: Some(2.0),
                hr_upscaler: None,
                hr_second_pass_steps: Some(0),
                hr_resize_x: Some(0),
                hr_resize_y: Some(0),
                hr_checkpoint_name: None,
                hr_sampler_name: None,
                hr_scheduler: None,
                hr_prompt: Some("".to_string()),
                hr_negative_prompt: Some("".to_string()),
                force_task_id: None,
                sampler_index: Some(Sampler::Euler),
                send_images: Some(true),
                save_images: Some(false),
                alwayson_scripts: AlwaysOnScripts::default(),
            }
        }
    }
    impl<'de> Deserialize<'de> for Txt2ImgRequest {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            enum Field {
                Prompt,
                NegativePrompt,
                Seed,
                Subseed,
                SubseedStrength,
                SeedResizeFromH,
                SeedResizeFromW,
                SamplerName,
                Scheduler,
                BatchSize,
                NIter,
                Steps,
                CfgScale,
                Width,
                Height,
                RestoreFaces,
                Tiling,
                DoNotSaveSamples,
                DoNotSaveGrid,
                Eta,
                DenoisingStrength,
                SMinUncond,
                SChurn,
                STmax,
                STmin,
                SNoise,
                OverrideSettings,
                OverrideSettingsRestoreAfterwards,
                RefinerCheckpoint,
                RefinerSwitchAt,
                DisableExtraNetworks,
                FirstpassImage,
                EnableHr,
                FirstphaseWidth,
                FirstphaseHeight,
                HrScale,
                HrUpscaler,
                HrSecondPassSteps,
                HrResizeX,
                HrResizeY,
                HrCheckpointName,
                HrSamplerName,
                HrScheduler,
                HrPrompt,
                HrNegativePrompt,
                ForceTaskId,
                SamplerIndex,
                SendImages,
                SaveImages,
                AlwaysonScripts,
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
                                "seed" => Ok(Field::Seed),
                                "subseed" => Ok(Field::Subseed),
                                "subseed_strength" => Ok(Field::SubseedStrength),
                                "seed_resize_from_h" => Ok(Field::SeedResizeFromH),
                                "seed_resize_from_w" => Ok(Field::SeedResizeFromW),
                                "sampler_name" => Ok(Field::SamplerName),
                                "scheduler" => Ok(Field::Scheduler),
                                "batch_size" => Ok(Field::BatchSize),
                                "n_iter" => Ok(Field::NIter),
                                "steps" => Ok(Field::Steps),
                                "cfg_scale" => Ok(Field::CfgScale),
                                "width" => Ok(Field::Width),
                                "height" => Ok(Field::Height),
                                "restore_faces" => Ok(Field::RestoreFaces),
                                "tiling" => Ok(Field::Tiling),
                                "do_not_save_samples" => Ok(Field::DoNotSaveSamples),
                                "do_not_save_grid" => Ok(Field::DoNotSaveGrid),
                                "eta" => Ok(Field::Eta),
                                "denoising_strength" => Ok(Field::DenoisingStrength),
                                "s_min_uncond" => Ok(Field::SMinUncond),
                                "s_churn" => Ok(Field::SChurn),
                                "s_tmax" => Ok(Field::STmax),
                                "s_tmin" => Ok(Field::STmin),
                                "s_noise" => Ok(Field::SNoise),
                                "override_settings" => Ok(Field::OverrideSettings),
                                "override_settings_restore_afterwards" => {
                                    Ok(Field::OverrideSettingsRestoreAfterwards)
                                }
                                "refiner_checkpoint" => Ok(Field::RefinerCheckpoint),
                                "refiner_switch_at" => Ok(Field::RefinerSwitchAt),
                                "disable_extra_networks" => Ok(Field::DisableExtraNetworks),
                                "firstpass_image" => Ok(Field::FirstpassImage),
                                "enable_hr" => Ok(Field::EnableHr),
                                "firstphase_width" => Ok(Field::FirstphaseWidth),
                                "firstphase_height" => Ok(Field::FirstphaseHeight),
                                "hr_scale" => Ok(Field::HrScale),
                                "hr_upscaler" => Ok(Field::HrUpscaler),
                                "hr_second_pass_steps" => Ok(Field::HrSecondPassSteps),
                                "hr_resize_x" => Ok(Field::HrResizeX),
                                "hr_resize_y" => Ok(Field::HrResizeY),
                                "hr_checkpoint_name" => Ok(Field::HrCheckpointName),
                                "hr_sampler_name" => Ok(Field::HrSamplerName),
                                "hr_scheduler" => Ok(Field::HrScheduler),
                                "hr_prompt" => Ok(Field::HrPrompt),
                                "hr_negative_prompt" => Ok(Field::HrNegativePrompt),
                                "force_task_id" => Ok(Field::ForceTaskId),
                                "sampler_index" => Ok(Field::SamplerIndex),
                                "send_images" => Ok(Field::SendImages),
                                "save_images" => Ok(Field::SaveImages),
                                "alwayson_scripts" => Ok(Field::AlwaysonScripts),
                                _ => Err(de::Error::unknown_field(value, FIELDS)),
                            }
                        }
                    }

                    deserializer.deserialize_identifier(FieldVisitor)
                }
            }

            struct Txt2ImgRequestVisitor;

            impl<'de> Visitor<'de> for Txt2ImgRequestVisitor {
                type Value = Txt2ImgRequest;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("struct Txt2ImgRequest")
                }

                fn visit_map<V>(self, mut map: V) -> Result<Txt2ImgRequest, V::Error>
                where
                    V: MapAccess<'de>,
                {
                    let mut prompt = None;
                    let mut negative_prompt = None;
                    let mut seed = None;
                    let mut subseed = None;
                    let mut subseed_strength = None;
                    let mut seed_resize_from_h = None;
                    let mut seed_resize_from_w = None;
                    let mut sampler_name = None;
                    let mut scheduler = None;
                    let mut batch_size = None;
                    let mut n_iter = None;
                    let mut steps = None;
                    let mut cfg_scale = None;
                    let mut width = None;
                    let mut height = None;
                    let mut restore_faces = None;
                    let mut tiling = None;
                    let mut do_not_save_samples = None;
                    let mut do_not_save_grid = None;
                    let mut eta = None;
                    let mut denoising_strength = None;
                    let mut s_min_uncond = None;
                    let mut s_churn = None;
                    let mut s_tmax = None;
                    let mut s_tmin = None;
                    let mut s_noise = None;
                    let mut override_settings = None;
                    let mut override_settings_restore_afterwards = None;
                    let mut refiner_checkpoint = None;
                    let mut refiner_switch_at = None;
                    let mut disable_extra_networks = None;
                    let mut firstpass_image = None;
                    let mut enable_hr = None;
                    let mut firstphase_width = None;
                    let mut firstphase_height = None;
                    let mut hr_scale = None;
                    let mut hr_upscaler = None;
                    let mut hr_second_pass_steps = None;
                    let mut hr_resize_x = None;
                    let mut hr_resize_y = None;
                    let mut hr_checkpoint_name = None;
                    let mut hr_sampler_name = None;
                    let mut hr_scheduler = None;
                    let mut hr_prompt = None;
                    let mut hr_negative_prompt = None;
                    let mut force_task_id = None;
                    let mut sampler_index = None;
                    let mut send_images = None;
                    let mut save_images = None;
                    let mut alwayson_scripts = None;

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
                            Field::Seed => {
                                if seed.is_some() {
                                    return Err(de::Error::duplicate_field("seed"));
                                }
                                seed = Some(map.next_value()?);
                            }
                            Field::Subseed => {
                                if subseed.is_some() {
                                    return Err(de::Error::duplicate_field("subseed"));
                                }
                                subseed = Some(map.next_value()?);
                            }
                            Field::SubseedStrength => {
                                if subseed_strength.is_some() {
                                    return Err(de::Error::duplicate_field("subseed_strength"));
                                }
                                subseed_strength = Some(map.next_value()?);
                            }
                            Field::SeedResizeFromH => {
                                if seed_resize_from_h.is_some() {
                                    return Err(de::Error::duplicate_field("seed_resize_from_h"));
                                }
                                seed_resize_from_h = Some(map.next_value()?);
                            }
                            Field::SeedResizeFromW => {
                                if seed_resize_from_w.is_some() {
                                    return Err(de::Error::duplicate_field("seed_resize_from_w"));
                                }
                                seed_resize_from_w = Some(map.next_value()?);
                            }
                            Field::SamplerName => {
                                if sampler_name.is_some() {
                                    return Err(de::Error::duplicate_field("sampler_name"));
                                }
                                sampler_name = Some(map.next_value()?);
                            }
                            Field::Scheduler => {
                                if scheduler.is_some() {
                                    return Err(de::Error::duplicate_field("scheduler"));
                                }
                                scheduler = Some(map.next_value()?);
                            }
                            Field::BatchSize => {
                                if batch_size.is_some() {
                                    return Err(de::Error::duplicate_field("batch_size"));
                                }
                                batch_size = Some(map.next_value()?);
                            }
                            Field::NIter => {
                                if n_iter.is_some() {
                                    return Err(de::Error::duplicate_field("n_iter"));
                                }
                                n_iter = Some(map.next_value()?);
                            }
                            Field::Steps => {
                                if steps.is_some() {
                                    return Err(de::Error::duplicate_field("steps"));
                                }
                                steps = Some(map.next_value()?);
                            }
                            Field::CfgScale => {
                                if cfg_scale.is_some() {
                                    return Err(de::Error::duplicate_field("cfg_scale"));
                                }
                                cfg_scale = Some(map.next_value()?);
                            }
                            Field::Width => {
                                if width.is_some() {
                                    return Err(de::Error::duplicate_field("width"));
                                }
                                width = Some(map.next_value()?);
                            }
                            Field::Height => {
                                if height.is_some() {
                                    return Err(de::Error::duplicate_field("height"));
                                }
                                height = Some(map.next_value()?);
                            }
                            Field::RestoreFaces => {
                                if restore_faces.is_some() {
                                    return Err(de::Error::duplicate_field("restore_faces"));
                                }
                                restore_faces = Some(map.next_value()?);
                            }
                            Field::Tiling => {
                                if tiling.is_some() {
                                    return Err(de::Error::duplicate_field("tiling"));
                                }
                                tiling = Some(map.next_value()?);
                            }
                            Field::DoNotSaveSamples => {
                                if do_not_save_samples.is_some() {
                                    return Err(de::Error::duplicate_field("do_not_save_samples"));
                                }
                                do_not_save_samples = Some(map.next_value()?);
                            }
                            Field::DoNotSaveGrid => {
                                if do_not_save_grid.is_some() {
                                    return Err(de::Error::duplicate_field("do_not_save_grid"));
                                }
                                do_not_save_grid = Some(map.next_value()?);
                            }
                            Field::Eta => {
                                if eta.is_some() {
                                    return Err(de::Error::duplicate_field("eta"));
                                }
                                eta = Some(map.next_value()?);
                            }
                            Field::DenoisingStrength => {
                                if denoising_strength.is_some() {
                                    return Err(de::Error::duplicate_field("denoising_strength"));
                                }
                                denoising_strength = Some(map.next_value()?);
                            }
                            Field::SMinUncond => {
                                if s_min_uncond.is_some() {
                                    return Err(de::Error::duplicate_field("s_min_uncond"));
                                }
                                s_min_uncond = Some(map.next_value()?);
                            }
                            Field::SChurn => {
                                if s_churn.is_some() {
                                    return Err(de::Error::duplicate_field("s_churn"));
                                }
                                s_churn = Some(map.next_value()?);
                            }
                            Field::STmax => {
                                if s_tmax.is_some() {
                                    return Err(de::Error::duplicate_field("s_tmax"));
                                }
                                s_tmax = Some(map.next_value()?);
                            }
                            Field::STmin => {
                                if s_tmin.is_some() {
                                    return Err(de::Error::duplicate_field("s_tmin"));
                                }
                                s_tmin = Some(map.next_value()?);
                            }
                            Field::SNoise => {
                                if s_noise.is_some() {
                                    return Err(de::Error::duplicate_field("s_noise"));
                                }
                                s_noise = Some(map.next_value()?);
                            }
                            Field::OverrideSettings => {
                                if override_settings.is_some() {
                                    return Err(de::Error::duplicate_field("override_settings"));
                                }
                                override_settings = Some(map.next_value()?);
                            }
                            Field::OverrideSettingsRestoreAfterwards => {
                                if override_settings_restore_afterwards.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "override_settings_restore_afterwards",
                                    ));
                                }
                                override_settings_restore_afterwards = Some(map.next_value()?);
                            }
                            Field::RefinerCheckpoint => {
                                if refiner_checkpoint.is_some() {
                                    return Err(de::Error::duplicate_field("refiner_checkpoint"));
                                }
                                refiner_checkpoint = Some(map.next_value()?);
                            }
                            Field::RefinerSwitchAt => {
                                if refiner_switch_at.is_some() {
                                    return Err(de::Error::duplicate_field("refiner_switch_at"));
                                }
                                refiner_switch_at = Some(map.next_value()?);
                            }
                            Field::DisableExtraNetworks => {
                                if disable_extra_networks.is_some() {
                                    return Err(de::Error::duplicate_field(
                                        "disable_extra_networks",
                                    ));
                                }
                                disable_extra_networks = Some(map.next_value()?);
                            }
                            Field::FirstpassImage => {
                                if firstpass_image.is_some() {
                                    return Err(de::Error::duplicate_field("firstpass_image"));
                                }
                                firstpass_image = Some(map.next_value()?);
                            }
                            Field::EnableHr => {
                                if enable_hr.is_some() {
                                    return Err(de::Error::duplicate_field("enable_hr"));
                                }
                                enable_hr = Some(map.next_value()?);
                            }
                            Field::FirstphaseWidth => {
                                if firstphase_width.is_some() {
                                    return Err(de::Error::duplicate_field("firstphase_width"));
                                }
                                firstphase_width = Some(map.next_value()?);
                            }
                            Field::FirstphaseHeight => {
                                if firstphase_height.is_some() {
                                    return Err(de::Error::duplicate_field("firstphase_height"));
                                }
                                firstphase_height = Some(map.next_value()?);
                            }
                            Field::HrScale => {
                                if hr_scale.is_some() {
                                    return Err(de::Error::duplicate_field("hr_scale"));
                                }
                                hr_scale = Some(map.next_value()?);
                            }
                            Field::HrUpscaler => {
                                if hr_upscaler.is_some() {
                                    return Err(de::Error::duplicate_field("hr_upscaler"));
                                }
                                hr_upscaler = Some(map.next_value()?);
                            }
                            Field::HrSecondPassSteps => {
                                if hr_second_pass_steps.is_some() {
                                    return Err(de::Error::duplicate_field("hr_second_pass_steps"));
                                }
                                hr_second_pass_steps = Some(map.next_value()?);
                            }
                            Field::HrResizeX => {
                                if hr_resize_x.is_some() {
                                    return Err(de::Error::duplicate_field("hr_resize_x"));
                                }
                                hr_resize_x = Some(map.next_value()?);
                            }
                            Field::HrResizeY => {
                                if hr_resize_y.is_some() {
                                    return Err(de::Error::duplicate_field("hr_resize_y"));
                                }
                                hr_resize_y = Some(map.next_value()?);
                            }
                            Field::HrCheckpointName => {
                                if hr_checkpoint_name.is_some() {
                                    return Err(de::Error::duplicate_field("hr_checkpoint_name"));
                                }
                                hr_checkpoint_name = Some(map.next_value()?);
                            }
                            Field::HrSamplerName => {
                                if hr_sampler_name.is_some() {
                                    return Err(de::Error::duplicate_field("hr_sampler_name"));
                                }
                                hr_sampler_name = Some(map.next_value()?);
                            }
                            Field::HrScheduler => {
                                if hr_scheduler.is_some() {
                                    return Err(de::Error::duplicate_field("hr_scheduler"));
                                }
                                hr_scheduler = Some(map.next_value()?);
                            }
                            Field::HrPrompt => {
                                if hr_prompt.is_some() {
                                    return Err(de::Error::duplicate_field("hr_prompt"));
                                }
                                hr_prompt = Some(map.next_value()?);
                            }
                            Field::HrNegativePrompt => {
                                if hr_negative_prompt.is_some() {
                                    return Err(de::Error::duplicate_field("hr_negative_prompt"));
                                }
                                hr_negative_prompt = Some(map.next_value()?);
                            }
                            Field::ForceTaskId => {
                                if force_task_id.is_some() {
                                    return Err(de::Error::duplicate_field("force_task_id"));
                                }
                                force_task_id = Some(map.next_value()?);
                            }
                            Field::SamplerIndex => {
                                if sampler_index.is_some() {
                                    return Err(de::Error::duplicate_field("sampler_index"));
                                }
                                sampler_index = Some(map.next_value()?);
                            }
                            Field::SendImages => {
                                if send_images.is_some() {
                                    return Err(de::Error::duplicate_field("send_images"));
                                }
                                send_images = Some(map.next_value()?);
                            }
                            Field::SaveImages => {
                                if save_images.is_some() {
                                    return Err(de::Error::duplicate_field("save_images"));
                                }
                                save_images = Some(map.next_value()?);
                            }
                            Field::AlwaysonScripts => {
                                if alwayson_scripts.is_some() {
                                    return Err(de::Error::duplicate_field("alwayson_scripts"));
                                }
                                alwayson_scripts = Some(map.next_value()?);
                            }
                        }
                    }

                    let prompt = prompt.ok_or_else(|| de::Error::missing_field("prompt"))?;

                    if negative_prompt.is_none() {
                        negative_prompt = Some("".to_string());
                    }

                    if seed.is_none() {
                        seed = Some(-1);
                    }

                    if subseed.is_none() {
                        subseed = Some(-1);
                    }

                    if subseed_strength.is_none() {
                        subseed_strength = Some(0.0);
                    }

                    if seed_resize_from_h.is_none() {
                        seed_resize_from_h = Some(-1);
                    }

                    if seed_resize_from_w.is_none() {
                        seed_resize_from_w = Some(-1);
                    }

                    if scheduler.is_none() {
                        scheduler = Some(Scheduler::Discrete);
                    }

                    if batch_size.is_none() {
                        batch_size = Some(1);
                    }

                    if n_iter.is_none() {
                        n_iter = Some(1);
                    }

                    if steps.is_none() {
                        steps = Some(20);
                    }

                    if cfg_scale.is_none() {
                        cfg_scale = Some(7.0);
                    }

                    if width.is_none() {
                        width = Some(512);
                    }

                    if height.is_none() {
                        height = Some(512);
                    }

                    if restore_faces.is_none() {
                        restore_faces = Some(false);
                    }

                    if tiling.is_none() {
                        tiling = Some(false);
                    }

                    if do_not_save_samples.is_none() {
                        do_not_save_samples = Some(false);
                    }

                    if do_not_save_grid.is_none() {
                        do_not_save_grid = Some(false);
                    }

                    let override_settings = override_settings.unwrap_or_default();

                    if override_settings_restore_afterwards.is_none() {
                        override_settings_restore_afterwards = Some(true);
                    }

                    if enable_hr.is_none() {
                        enable_hr = Some(false);
                    }

                    if firstphase_width.is_none() {
                        firstphase_width = Some(0);
                    }

                    if firstphase_height.is_none() {
                        firstphase_height = Some(0);
                    }

                    if hr_scale.is_none() {
                        hr_scale = Some(2.0);
                    }

                    if hr_second_pass_steps.is_none() {
                        hr_second_pass_steps = Some(0);
                    }

                    if hr_resize_x.is_none() {
                        hr_resize_x = Some(0);
                    }

                    if hr_resize_y.is_none() {
                        hr_resize_y = Some(0);
                    }

                    if hr_prompt.is_none() {
                        hr_prompt = Some("".to_string());
                    }

                    if hr_negative_prompt.is_none() {
                        hr_negative_prompt = Some("".to_string());
                    }

                    if sampler_index.is_none() {
                        sampler_index = Some(Sampler::Euler);
                    }

                    if send_images.is_none() {
                        send_images = Some(true);
                    }

                    if save_images.is_none() {
                        save_images = Some(false);
                    }

                    let alwayson_scripts = alwayson_scripts.unwrap_or_default();

                    Ok(Txt2ImgRequest {
                        prompt,
                        negative_prompt,
                        seed,
                        subseed,
                        subseed_strength,
                        seed_resize_from_h,
                        seed_resize_from_w,
                        sampler_name,
                        scheduler,
                        batch_size,
                        n_iter,
                        steps,
                        cfg_scale,
                        width,
                        height,
                        restore_faces,
                        tiling,
                        do_not_save_samples,
                        do_not_save_grid,
                        eta,
                        denoising_strength,
                        s_min_uncond,
                        s_churn,
                        s_tmax,
                        s_tmin,
                        s_noise,
                        override_settings,
                        override_settings_restore_afterwards,
                        refiner_checkpoint,
                        refiner_switch_at,
                        disable_extra_networks,
                        firstpass_image,
                        enable_hr,
                        firstphase_width,
                        firstphase_height,
                        hr_scale,
                        hr_upscaler,
                        hr_second_pass_steps,
                        hr_resize_x,
                        hr_resize_y,
                        hr_checkpoint_name,
                        hr_sampler_name,
                        hr_scheduler,
                        hr_prompt,
                        hr_negative_prompt,
                        force_task_id,
                        sampler_index,
                        send_images,
                        save_images,
                        alwayson_scripts,
                    })
                }
            }

            const FIELDS: &[&str] = &[
                "prompt",
                "negative_prompt",
                "seed",
                "subseed",
                "subseed_strength",
                "seed_resize_from_h",
                "seed_resize_from_w",
                "sampler_name",
                "scheduler",
                "batch_size",
                "n_iter",
                "steps",
                "cfg_scale",
                "width",
                "height",
                "restore_faces",
                "tiling",
                "do_not_save_samples",
                "do_not_save_grid",
                "eta",
                "denoising_strength",
                "s_min_uncond",
                "s_churn",
                "s_tmax",
                "s_tmin",
                "s_noise",
                "override_settings",
                "override_settings_restore_afterwards",
                "refiner_checkpoint",
                "refiner_switch_at",
                "disable_extra_networks",
                "firstpass_image",
                "enable_hr",
                "firstphase_width",
                "firstphase_height",
                "hr_scale",
                "hr_upscaler",
                "hr_second_pass_steps",
                "hr_resize_x",
                "hr_resize_y",
                "hr_checkpoint_name",
                "hr_sampler_name",
                "hr_scheduler",
                "hr_prompt",
                "hr_negative_prompt",
                "force_task_id",
                "sampler_index",
                "send_images",
                "save_images",
                "alwayson_scripts",
            ];

            deserializer.deserialize_struct("Txt2ImgRequest", FIELDS, Txt2ImgRequestVisitor)
        }
    }

    #[cfg(test)]
    #[test]
    fn test_deserialize_txt2img_request() {
        let json = r###"{
    "prompt": "1girl,intricate,highly detailed,Mature,seductive gaze,teasing expression,sexy posture,solo,Moderate breasts,Charm,alluring,Hot,tsurime,lipstick,stylish_pose,long hair,long_eyelashes,black hair,bar,dress,",
    "negative_prompt": "",
    "seed": -1,
    "batch_size": 2,
    "steps": 25,
    "scheduler": "Karras",
    "cfg_scale": 7.0,
    "width": 540,
    "height": 960,
    "restore_faces": false,
    "tiling": false,
    "override_settings": {
        "sd_model_checkpoint": "waiANINSFWPONYXL_v90.safetensors"
    },
    "sampler_index": "DPM++ 2M",
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "enabled": true,
                    "pixel_perfect": true,
                    "image": "iVBORw0KGgoAAAANSUhEUgAABDgAAAeACAI",
                    "module": "reference_only",
                    "guidance_start": 0.0,
                    "guidance_end": 0.2
                }
            ]
        }
    }
}"###;
        let req: Txt2ImgRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "1girl,intricate,highly detailed,Mature,seductive gaze,teasing expression,sexy posture,solo,Moderate breasts,Charm,alluring,Hot,tsurime,lipstick,stylish_pose,long hair,long_eyelashes,black hair,bar,dress,");
        assert_eq!(req.negative_prompt, Some("".to_string()));
        assert_eq!(req.seed, Some(-1));
        assert_eq!(req.batch_size, Some(2));
        assert_eq!(req.steps, Some(25));
        assert_eq!(req.scheduler, Some(Scheduler::Karras));
        assert_eq!(req.cfg_scale, Some(7.0));
        assert_eq!(req.width, Some(540));
        assert_eq!(req.height, Some(960));
        assert_eq!(req.restore_faces, Some(false));
        assert_eq!(req.tiling, Some(false));
        assert_eq!(req.sampler_index, Some(Sampler::DpmPlusPlus2M));
        assert_eq!(req.alwayson_scripts.controlnet.args.len(), 1);
        assert_eq!(req.alwayson_scripts.controlnet.args[0].enabled, Some(true));
        assert_eq!(
            req.alwayson_scripts.controlnet.args[0].pixel_perfect,
            Some(true)
        );
        assert_eq!(
            req.alwayson_scripts.controlnet.args[0].module,
            "reference_only"
        );
        assert_eq!(
            req.alwayson_scripts.controlnet.args[0].guidance_start,
            Some(0.0)
        );
        assert_eq!(
            req.alwayson_scripts.controlnet.args[0].guidance_end,
            Some(0.2)
        );
        assert_eq!(
            req.alwayson_scripts.controlnet.args[0].image,
            "iVBORw0KGgoAAAANSUhEUgAABDgAAAeACAI"
        );
        assert_eq!(
            req.override_settings.sd_model_checkpoint,
            "waiANINSFWPONYXL_v90.safetensors"
        );
    }

    #[test]
    fn test_serialize_txt2img_request() {
        let req = Txt2ImgRequest {
            prompt: "1girl,intricate,highly detailed,Mature,seductive gaze,teasing expression,sexy posture,solo,Moderate breasts,Charm,alluring,Hot,tsurime,lipstick,stylish_pose,long hair,long_eyelashes,black hair,bar,dress,".to_string(),
            negative_prompt: Some("".to_string()),
            seed: Some(-1),
            subseed: Some(-1),
            subseed_strength: Some(0.0),
            seed_resize_from_h: Some(-1),
            seed_resize_from_w: Some(-1),
            scheduler: Some(Scheduler::Discrete),
            batch_size: Some(2),
            n_iter: Some(1),
            steps: Some(25),
            cfg_scale: Some(7.0),
            width: Some(540),
            height: Some(960),
            restore_faces: Some(false),
            tiling: Some(false),
            override_settings: OverrideSettings {
                sd_model_checkpoint: "waiANINSFWPONYXL_v90.safetensors".to_string(),
            },
            sampler_index: Some(Sampler::DpmPlusPlus2M),
            alwayson_scripts: AlwaysOnScripts {
                controlnet: ControlNet { args: vec![
                    ControlNetArgs {
                        enabled: Some(true),
                        pixel_perfect: Some(true),
                        image: "iVBORw0KGgoAAAANSUhEUgAABDgAAAeACAI".to_string(),
                        module: "reference_only".to_string(),
                        guidance_start: Some(0.0),
                        guidance_end: Some(0.2),
                    }
                ] },
            },
            ..Default::default()
        };
        let serialized = serde_json::to_string_pretty(&req).unwrap();

        let json = r###"{
  "prompt": "1girl,intricate,highly detailed,Mature,seductive gaze,teasing expression,sexy posture,solo,Moderate breasts,Charm,alluring,Hot,tsurime,lipstick,stylish_pose,long hair,long_eyelashes,black hair,bar,dress,",
  "negative_prompt": "",
  "seed": -1,
  "subseed": -1,
  "subseed_strength": 0.0,
  "seed_resize_from_h": -1,
  "seed_resize_from_w": -1,
  "scheduler": "discrete",
  "batch_size": 2,
  "n_iter": 1,
  "steps": 25,
  "cfg_scale": 7.0,
  "width": 540,
  "height": 960,
  "restore_faces": false,
  "tiling": false,
  "do_not_save_samples": false,
  "do_not_save_grid": false,
  "s_min_uncond": null,
  "override_settings": {
    "sd_model_checkpoint": "waiANINSFWPONYXL_v90.safetensors"
  },
  "override_settings_restore_afterwards": true,
  "enable_hr": false,
  "firstphase_width": 0,
  "firstphase_height": 0,
  "hr_scale": 2.0,
  "hr_second_pass_steps": 0,
  "hr_resize_x": 0,
  "hr_resize_y": 0,
  "hr_prompt": "",
  "hr_negative_prompt": "",
  "sampler_index": "DPM++ 2M",
  "send_images": true,
  "save_images": false,
  "alwayson_scripts": {
    "controlnet": {
      "args": [
        {
          "enabled": true,
          "pixel_perfect": true,
          "image": "iVBORw0KGgoAAAANSUhEUgAABDgAAAeACAI",
          "module": "reference_only",
          "guidance_start": 0.0,
          "guidance_end": 0.2
        }
      ]
    }
  }
}"###;

        assert_eq!(serialized, json);
    }

    /// Sampling method
    #[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
    pub enum Sampler {
        #[serde(rename = "Euler")]
        Euler,
        #[serde(rename = "Euler a")]
        EulerA,
        #[serde(rename = "Heun")]
        Heun,
        #[serde(rename = "DPM2")]
        Dpm2,
        #[serde(rename = "DPM2 a")]
        Dpm2a,
        #[serde(rename = "DPM fast")]
        DpmFast,
        #[serde(rename = "DPM adaptive")]
        DpmAdaptive,
        #[serde(rename = "DPM++ 2S a")]
        DpmPlusPlus2sA,
        #[serde(rename = "DPM++ 2M")]
        DpmPlusPlus2M,
        #[serde(rename = "DPM++ 2M SDE")]
        DpmPlusPlus2MSde,
        #[serde(rename = "DPM++ 2M SDE Heun")]
        DpmPlusPlus2MSdeHeun,
        #[serde(rename = "DPM++ 3M SDE")]
        DpmPlusPlus3MSde,
        #[serde(rename = "DPM++ SDE")]
        DpmPlusPlusSde,
        #[serde(rename = "LCM")]
        Lcm,
        #[serde(rename = "LMS")]
        LMS,
        #[serde(rename = "Restart")]
        Restart,
        #[serde(rename = "DDIM")]
        Ddim,
        #[serde(rename = "DDIM CFG++")]
        DdimCfgPlusPlus,
        #[serde(rename = "PLMS")]
        Plms,
        #[serde(rename = "UniPC")]
        UniPC,
    }
    impl fmt::Display for Sampler {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Sampler::Euler => write!(f, "Euler"),
                Sampler::EulerA => write!(f, "Euler a"),
                Sampler::Heun => write!(f, "Heun"),
                Sampler::Dpm2 => write!(f, "DPM2"),
                Sampler::Dpm2a => write!(f, "DPM2 a"),
                Sampler::DpmFast => write!(f, "DPM fast"),
                Sampler::DpmAdaptive => write!(f, "DPM adaptive"),
                Sampler::DpmPlusPlus2sA => write!(f, "DPM++ 2S a"),
                Sampler::DpmPlusPlus2M => write!(f, "DPM++ 2M"),
                Sampler::DpmPlusPlus2MSde => write!(f, "DPM++ 2M SDE"),
                Sampler::DpmPlusPlus2MSdeHeun => write!(f, "DPM++ 2M SDE Heun"),
                Sampler::DpmPlusPlus3MSde => write!(f, "DPM++ 3M SDE"),
                Sampler::DpmPlusPlusSde => write!(f, "DPM++ SDE"),
                Sampler::Lcm => write!(f, "LCM"),
                Sampler::LMS => write!(f, "LMS"),
                Sampler::Restart => write!(f, "Restart"),
                Sampler::Ddim => write!(f, "DDIM"),
                Sampler::DdimCfgPlusPlus => write!(f, "DDIM CFG++"),
                Sampler::Plms => write!(f, "PLMS"),
                Sampler::UniPC => write!(f, "UniPC"),
            }
        }
    }
    impl From<&str> for Sampler {
        fn from(s: &str) -> Self {
            match s {
                "Euler" => Sampler::Euler,
                "Euler a" => Sampler::EulerA,
                "Heun" => Sampler::Heun,
                "DPM2" => Sampler::Dpm2,
                "DPM2 a" => Sampler::Dpm2a,
                "DPM fast" => Sampler::DpmFast,
                "DPM adaptive" => Sampler::DpmAdaptive,
                "DPM++ 2S a" => Sampler::DpmPlusPlus2sA,
                "DPM++ 2M" => Sampler::DpmPlusPlus2M,
                "DPM++ 2M SDE" => Sampler::DpmPlusPlus2MSde,
                "DPM++ 2M SDE Heun" => Sampler::DpmPlusPlus2MSdeHeun,
                "DPM++ 3M SDE" => Sampler::DpmPlusPlus3MSde,
                "DPM++ SDE" => Sampler::DpmPlusPlusSde,
                "LCM" => Sampler::Lcm,
                "LMS" => Sampler::LMS,
                "Restart" => Sampler::Restart,
                "DDIM" => Sampler::Ddim,
                "DDIM CFG++" => Sampler::DdimCfgPlusPlus,
                "PLMS" => Sampler::Plms,
                "UniPC" => Sampler::UniPC,
                _ => Sampler::EulerA,
            }
        }
    }
    impl<'de> Deserialize<'de> for Sampler {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            struct SamplerVisitor;

            impl Visitor<'_> for SamplerVisitor {
                type Value = Sampler;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a string representing a sampler type")
                }

                fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                where
                    E: de::Error,
                {
                    match value {
                        "Euler" => Ok(Sampler::Euler),
                        "Euler a" => Ok(Sampler::EulerA),
                        "Heun" => Ok(Sampler::Heun),
                        "DPM2" => Ok(Sampler::Dpm2),
                        "DPM2 a" => Ok(Sampler::Dpm2a),
                        "DPM fast" => Ok(Sampler::DpmFast),
                        "DPM adaptive" => Ok(Sampler::DpmAdaptive),
                        "DPM++ 2S a" => Ok(Sampler::DpmPlusPlus2sA),
                        "DPM++ 2M" => Ok(Sampler::DpmPlusPlus2M),
                        "DPM++ 2M SDE" => Ok(Sampler::DpmPlusPlus2MSde),
                        "DPM++ 2M SDE Heun" => Ok(Sampler::DpmPlusPlus2MSdeHeun),
                        "DPM++ 3M SDE" => Ok(Sampler::DpmPlusPlus3MSde),
                        "DPM++ SDE" => Ok(Sampler::DpmPlusPlusSde),
                        "LCM" => Ok(Sampler::Lcm),
                        "LMS" => Ok(Sampler::LMS),
                        "Restart" => Ok(Sampler::Restart),
                        "DDIM" => Ok(Sampler::Ddim),
                        "DDIM CFG++" => Ok(Sampler::DdimCfgPlusPlus),
                        "PLMS" => Ok(Sampler::Plms),
                        "UniPC" => Ok(Sampler::UniPC),
                        _ => Err(E::custom(format!(
                            "unknown sampler type: {}, expected one of: Euler, Euler a, Heun, DPM2, DPM2 a, DPM fast, DPM adaptive, DPM++ 2S a, DPM++ 2M, DPM++ 2M SDE, DPM++ 2M SDE Heun, DPM++ 3M SDE, DPM++ SDE, LCM, LMS, Restart, DDIM, DDIM CFG++, PLMS, UniPC",
                            value
                        ))),
                    }
                }

                fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
                where
                    E: de::Error,
                {
                    self.visit_str(&value)
                }
            }

            deserializer.deserialize_str(SamplerVisitor)
        }
    }

    #[derive(Serialize, Deserialize, Debug, Default)]
    pub struct OverrideSettings {
        pub sd_model_checkpoint: String,
    }

    #[derive(Serialize, Deserialize, Debug, Default)]
    pub struct AlwaysOnScripts {
        pub controlnet: ControlNet,
    }

    #[derive(Serialize, Deserialize, Debug, Default)]
    pub struct ControlNet {
        pub args: Vec<ControlNetArgs>,
    }

    #[derive(Serialize, Debug)]
    pub struct ControlNetArgs {
        /// Enable the control net. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub enabled: Option<bool>,
        /// Pixel perfect. Defaults to false.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub pixel_perfect: Option<bool>,
        /// Image. Store image as a Base64 string.
        pub image: String,
        /// ControlNet module.
        pub module: String,
        /// Guidance start. Defaults to 0.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub guidance_start: Option<f64>,
        /// Guidance end. Defaults to 1.0.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub guidance_end: Option<f64>,
    }
    impl Default for ControlNetArgs {
        fn default() -> Self {
            Self {
                enabled: Some(false),
                pixel_perfect: Some(false),
                image: "".to_string(),
                module: "reference_only".to_string(),
                guidance_start: Some(0.0),
                guidance_end: Some(1.0),
            }
        }
    }
    impl<'de> Deserialize<'de> for ControlNetArgs {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            enum Field {
                Enabled,
                PixelPerfect,
                Image,
                Module,
                GuidanceStart,
                GuidanceEnd,
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
                                "enabled" => Ok(Field::Enabled),
                                "pixel_perfect" => Ok(Field::PixelPerfect),
                                "image" => Ok(Field::Image),
                                "module" => Ok(Field::Module),
                                "guidance_start" => Ok(Field::GuidanceStart),
                                "guidance_end" => Ok(Field::GuidanceEnd),
                                _ => Err(de::Error::unknown_field(value, FIELDS)),
                            }
                        }
                    }

                    deserializer.deserialize_identifier(FieldVisitor)
                }
            }

            struct ControlNetArgsVisitor;

            impl<'de> Visitor<'de> for ControlNetArgsVisitor {
                type Value = ControlNetArgs;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("struct ControlNetArgs")
                }

                fn visit_map<V>(self, mut map: V) -> Result<ControlNetArgs, V::Error>
                where
                    V: MapAccess<'de>,
                {
                    let mut enabled = None;
                    let mut pixel_perfect = None;
                    let mut image = None;
                    let mut module = None;
                    let mut guidance_start = None;
                    let mut guidance_end = None;

                    while let Some(key) = map.next_key()? {
                        match key {
                            Field::Enabled => {
                                if enabled.is_some() {
                                    return Err(de::Error::duplicate_field("enabled"));
                                }
                                enabled = Some(map.next_value()?);
                            }
                            Field::PixelPerfect => {
                                if pixel_perfect.is_some() {
                                    return Err(de::Error::duplicate_field("pixel_perfect"));
                                }
                                pixel_perfect = Some(map.next_value()?);
                            }
                            Field::Image => {
                                if image.is_some() {
                                    return Err(de::Error::duplicate_field("image"));
                                }
                                image = Some(map.next_value()?);
                            }
                            Field::Module => {
                                if module.is_some() {
                                    return Err(de::Error::duplicate_field("module"));
                                }
                                module = Some(map.next_value()?);
                            }
                            Field::GuidanceStart => {
                                if guidance_start.is_some() {
                                    return Err(de::Error::duplicate_field("guidance_start"));
                                }
                                guidance_start = Some(map.next_value()?);
                            }
                            Field::GuidanceEnd => {
                                if guidance_end.is_some() {
                                    return Err(de::Error::duplicate_field("guidance_end"));
                                }
                                guidance_end = Some(map.next_value()?);
                            }
                        }
                    }

                    let image = image.ok_or_else(|| de::Error::missing_field("image"))?;
                    let module = module.ok_or_else(|| de::Error::missing_field("module"))?;

                    if enabled.is_none() {
                        enabled = Some(false);
                    }

                    if pixel_perfect.is_none() {
                        pixel_perfect = Some(false);
                    }

                    if guidance_start.is_none() {
                        guidance_start = Some(0.0);
                    }

                    if guidance_end.is_none() {
                        guidance_end = Some(1.0);
                    }

                    Ok(ControlNetArgs {
                        enabled,
                        pixel_perfect,
                        image,
                        module,
                        guidance_start,
                        guidance_end,
                    })
                }
            }

            const FIELDS: &[&str] = &[
                "enabled",
                "pixel_perfect",
                "image",
                "module",
                "guidance_start",
                "guidance_end",
            ];

            deserializer.deserialize_struct("ControlNetArgs", FIELDS, ControlNetArgsVisitor)
        }
    }
}
