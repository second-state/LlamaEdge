//! Llama Core, abbreviated as `llama-core`, defines a set of APIs. Developers can utilize these APIs to build applications based on large models, such as chatbots, RAG, and more.

#[cfg(feature = "logging")]
#[macro_use]
extern crate log;

pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod graph;
pub mod images;
pub mod metadata;
pub mod models;
pub mod rag;
#[cfg(feature = "search")]
pub mod search;
pub mod utils;

pub use error::LlamaCoreError;
pub use graph::{EngineType, Graph, GraphBuilder};
pub use metadata::{
    ggml::GgmlMetadata, piper::PiperMetadata, whisper::WhisperMetadata, BaseMetadata,
};

use once_cell::sync::OnceCell;
use std::{
    collections::HashMap,
    path::Path,
    sync::{Mutex, RwLock},
};
use utils::get_output_buffer;
use wasmedge_stable_diffusion::*;

// key: model_name, value: Graph
pub(crate) static CHAT_GRAPHS: OnceCell<Mutex<HashMap<String, Graph<GgmlMetadata>>>> =
    OnceCell::new();
// key: model_name, value: Graph
pub(crate) static EMBEDDING_GRAPHS: OnceCell<Mutex<HashMap<String, Graph<GgmlMetadata>>>> =
    OnceCell::new();
// cache bytes for decoding utf8
pub(crate) static CACHED_UTF8_ENCODINGS: OnceCell<Mutex<Vec<u8>>> = OnceCell::new();
// running mode
pub(crate) static RUNNING_MODE: OnceCell<RwLock<RunningMode>> = OnceCell::new();
// stable diffusion context for the text-to-image task
pub(crate) static SD_TEXT_TO_IMAGE: OnceCell<Mutex<TextToImage>> = OnceCell::new();
// stable diffusion context for the image-to-image task
pub(crate) static SD_IMAGE_TO_IMAGE: OnceCell<Mutex<ImageToImage>> = OnceCell::new();
// context for the audio task
pub(crate) static AUDIO_GRAPH: OnceCell<Mutex<Graph<WhisperMetadata>>> = OnceCell::new();
// context for the piper task
pub(crate) static PIPER_GRAPH: OnceCell<Mutex<Graph<PiperMetadata>>> = OnceCell::new();

pub(crate) const MAX_BUFFER_SIZE: usize = 2usize.pow(14) * 15 + 128;
pub(crate) const OUTPUT_TENSOR: usize = 0;
const PLUGIN_VERSION: usize = 1;

/// Initialize the core context
pub fn init_core_context(
    metadata_for_chats: Option<&[GgmlMetadata]>,
    metadata_for_embeddings: Option<&[GgmlMetadata]>,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the core context");

    if metadata_for_chats.is_none() && metadata_for_embeddings.is_none() {
        let err_msg = "Failed to initialize the core context. Please set metadata for chat completions and/or embeddings.";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::InitContext(err_msg.into()));
    }

    let mut mode = RunningMode::Embeddings;

    if let Some(metadata_chats) = metadata_for_chats {
        let mut chat_graphs = HashMap::new();
        for metadata in metadata_chats {
            let graph = Graph::new(metadata.clone())?;

            chat_graphs.insert(graph.name().to_string(), graph);
        }
        CHAT_GRAPHS.set(Mutex::new(chat_graphs)).map_err(|_| {
            let err_msg = "Failed to initialize the core context. Reason: The `CHAT_GRAPHS` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

        mode = RunningMode::Chat
    }

    if let Some(metadata_embeddings) = metadata_for_embeddings {
        let mut embedding_graphs = HashMap::new();
        for metadata in metadata_embeddings {
            let graph = Graph::new(metadata.clone())?;

            embedding_graphs.insert(graph.name().to_string(), graph);
        }
        EMBEDDING_GRAPHS
            .set(Mutex::new(embedding_graphs))
            .map_err(|_| {
                let err_msg = "Failed to initialize the core context. Reason: The `EMBEDDING_GRAPHS` has already been initialized";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg.into())
            })?;

        if mode == RunningMode::Chat {
            mode = RunningMode::ChatEmbedding;
        }
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "running mode: {}", mode);

    RUNNING_MODE.set(RwLock::new(mode)).map_err(|_| {
        let err_msg = "Failed to initialize the core context. Reason: The `RUNNING_MODE` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The core context has been initialized");

    Ok(())
}

/// Initialize the core context for RAG scenarios.
pub fn init_rag_core_context(
    metadata_for_chats: &[GgmlMetadata],
    metadata_for_embeddings: &[GgmlMetadata],
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the core context for RAG scenarios");

    // chat models
    if metadata_for_chats.is_empty() {
        let err_msg = "The metadata for chat models is empty";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::InitContext(err_msg.into()));
    }
    let mut chat_graphs = HashMap::new();
    for metadata in metadata_for_chats {
        let graph = Graph::new(metadata.clone())?;

        chat_graphs.insert(graph.name().to_string(), graph);
    }
    CHAT_GRAPHS.set(Mutex::new(chat_graphs)).map_err(|_| {
        let err_msg = "Failed to initialize the core context. Reason: The `CHAT_GRAPHS` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

    // embedding models
    if metadata_for_embeddings.is_empty() {
        let err_msg = "The metadata for embeddings is empty";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::InitContext(err_msg.into()));
    }
    let mut embedding_graphs = HashMap::new();
    for metadata in metadata_for_embeddings {
        let graph = Graph::new(metadata.clone())?;

        embedding_graphs.insert(graph.name().to_string(), graph);
    }
    EMBEDDING_GRAPHS
        .set(Mutex::new(embedding_graphs))
        .map_err(|_| {
            let err_msg = "Failed to initialize the core context. Reason: The `EMBEDDING_GRAPHS` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    let running_mode = RunningMode::Rag;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "running mode: {}", running_mode);

    // set running mode
    RUNNING_MODE.set(RwLock::new(running_mode)).map_err(|_| {
            let err_msg = "Failed to initialize the core context. Reason: The `RUNNING_MODE` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The core context for RAG scenarios has been initialized");

    Ok(())
}

/// Get the plugin info
///
/// Note that it is required to call `init_core_context` before calling this function.
pub fn get_plugin_info() -> Result<PluginInfo, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Getting the plugin info");

    match running_mode()? {
        RunningMode::Embeddings => {
            let embedding_graphs = match EMBEDDING_GRAPHS.get() {
                Some(embedding_graphs) => embedding_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `EMBEDDING_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let embedding_graphs = embedding_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            let graph = match embedding_graphs.values().next() {
                Some(graph) => graph,
                None => {
                    let err_msg = "Fail to get the underlying value of `EMBEDDING_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            get_plugin_info_by_graph(graph)
        }
        _ => {
            let chat_graphs = match CHAT_GRAPHS.get() {
                Some(chat_graphs) => chat_graphs,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            let chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            let graph = match chat_graphs.values().next() {
                Some(graph) => graph,
                None => {
                    let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", err_msg);

                    return Err(LlamaCoreError::Operation(err_msg.into()));
                }
            };

            get_plugin_info_by_graph(graph)
        }
    }
}

fn get_plugin_info_by_graph<M: BaseMetadata + serde::Serialize + Clone + Default>(
    graph: &Graph<M>,
) -> Result<PluginInfo, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Getting the plugin info by the graph named {}", graph.name());

    // get the plugin metadata
    let output_buffer = get_output_buffer(graph, PLUGIN_VERSION)?;
    let metadata: serde_json::Value = serde_json::from_slice(&output_buffer[..]).map_err(|e| {
        let err_msg = format!("Fail to deserialize the plugin metadata. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    // get build number of the plugin
    let plugin_build_number = match metadata.get("llama_build_number") {
        Some(value) => match value.as_u64() {
            Some(number) => number,
            None => {
                let err_msg = "Failed to convert the build number of the plugin to u64";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        },
        None => {
            let err_msg = "Metadata does not have the field `llama_build_number`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    // get commit id of the plugin
    let plugin_commit = match metadata.get("llama_commit") {
        Some(value) => match value.as_str() {
            Some(commit) => commit,
            None => {
                let err_msg = "Failed to convert the commit id of the plugin to string";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        },
        None => {
            let err_msg = "Metadata does not have the field `llama_commit`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Plugin info: b{}(commit {})", plugin_build_number, plugin_commit);

    Ok(PluginInfo {
        build_number: plugin_build_number,
        commit_id: plugin_commit.to_string(),
    })
}

/// Version info of the `wasi-nn_ggml` plugin, including the build number and the commit id.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub build_number: u64,
    pub commit_id: String,
}
impl std::fmt::Display for PluginInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "wasinn-ggml plugin: b{}(commit {})",
            self.build_number, self.commit_id
        )
    }
}

/// Running mode
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RunningMode {
    Chat,
    Embeddings,
    ChatEmbedding,
    Rag,
}
impl std::fmt::Display for RunningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunningMode::Chat => write!(f, "chat"),
            RunningMode::Embeddings => write!(f, "embeddings"),
            RunningMode::ChatEmbedding => write!(f, "chat-embeddings"),
            RunningMode::Rag => write!(f, "rag"),
        }
    }
}

/// Return the current running mode.
pub fn running_mode() -> Result<RunningMode, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Get the running mode.");

    let mode = match RUNNING_MODE.get() {
        Some(mode) => match mode.read() {
            Ok(mode) => mode.to_owned(),
            Err(e) => {
                let err_msg = format!("Fail to get the underlying value of `RUNNING_MODE`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::Operation(err_msg));
            }
        },
        None => {
            let err_msg = "Fail to get the underlying value of `RUNNING_MODE`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "running mode: {}", &mode);

    Ok(mode.to_owned())
}

/// Initialize the stable diffusion context with the given full diffusion model
///
/// # Arguments
///
/// * `model_file` - Path to the stable diffusion model file.
///
/// * `ctx` - The context type to create.
pub fn init_sd_context_with_full_model(
    model_file: impl AsRef<str>,
    ctx: SDContextType,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the stable diffusion context with the full model");

    // create the stable diffusion context for the text-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::TextToImage {
        let sd = StableDiffusion::new(Task::TextToImage, model_file.as_ref());

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::TextToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the text-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_TEXT_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
        let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_TEXT_TO_IMAGE` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion text-to-image context has been initialized");
    }

    // create the stable diffusion context for the image-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::ImageToImage {
        let sd = StableDiffusion::new(Task::ImageToImage, model_file.as_ref());

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::ImageToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the image-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_IMAGE_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
            let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_IMAGE_TO_IMAGE` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion image-to-image context has been initialized");
    }

    Ok(())
}

/// Initialize the stable diffusion context with the given standalone diffusion model
///
/// # Arguments
///
/// * `model_file` - Path to the standalone diffusion model file.
///
/// * `vae` - Path to the VAE model file.
///
/// * `clip_l` - Path to the CLIP model file.
///
/// * `t5xxl` - Path to the T5-XXL model file.
///
/// * `lora_model_dir` - Path to the Lora model directory.
///
/// * `n_threads` - Number of threads to use.
///
/// * `ctx` - The context type to create.
pub fn init_sd_context_with_standalone_model(
    model_file: impl AsRef<str>,
    vae: impl AsRef<str>,
    clip_l: impl AsRef<str>,
    t5xxl: impl AsRef<str>,
    lora_model_dir: impl AsRef<str>,
    n_threads: i32,
    ctx: SDContextType,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the stable diffusion context with the standalone diffusion model");

    // create the stable diffusion context for the text-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::TextToImage {
        let sd = SDBuidler::new_with_standalone_model(Task::TextToImage, model_file.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_vae_path(vae.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_clip_l_path(clip_l.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_t5xxl_path(t5xxl.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_lora_model_dir(lora_model_dir.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_n_threads(n_threads)
            .build();

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::TextToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the text-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_TEXT_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
            let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_TEXT_TO_IMAGE` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion text-to-image context has been initialized");
    }

    // create the stable diffusion context for the image-to-image task
    if ctx == SDContextType::Full || ctx == SDContextType::ImageToImage {
        let sd = SDBuidler::new_with_standalone_model(Task::ImageToImage, model_file.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_vae_path(vae.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_clip_l_path(clip_l.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_t5xxl_path(t5xxl.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_lora_model_dir(lora_model_dir.as_ref())
            .map_err(|e| {
                let err_msg = format!(
                    "Failed to initialize the stable diffusion context. Reason: {}",
                    e
                );

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                LlamaCoreError::InitContext(err_msg)
            })?
            .with_n_threads(n_threads)
            .build();

        let ctx = sd.create_context().map_err(|e| {
            let err_msg = format!("Fail to create the context. {}", e);

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::InitContext(err_msg)
        })?;

        let ctx = match ctx {
            Context::ImageToImage(ctx) => ctx,
            _ => {
                let err_msg = "Fail to get the context for the image-to-image task";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", err_msg);

                return Err(LlamaCoreError::InitContext(err_msg.into()));
            }
        };

        SD_IMAGE_TO_IMAGE.set(Mutex::new(ctx)).map_err(|_| {
        let err_msg = "Failed to initialize the stable diffusion context. Reason: The `SD_IMAGE_TO_IMAGE` has already been initialized";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        LlamaCoreError::InitContext(err_msg.into())
    })?;

        #[cfg(feature = "logging")]
        info!(target: "stdout", "The stable diffusion image-to-image context has been initialized");
    }

    Ok(())
}

/// The context to create for the stable diffusion model
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum SDContextType {
    /// `text_to_image` context
    TextToImage,
    /// `image_to_image` context
    ImageToImage,
    /// Both `text_to_image` and `image_to_image` contexts
    Full,
}

/// Initialize the whisper context
pub fn init_whisper_context(
    whisper_metadata: &WhisperMetadata,
    model_file: impl AsRef<Path>,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the audio context");

    // create and initialize the audio context
    let graph = GraphBuilder::new(EngineType::Whisper)?
        .with_config(whisper_metadata.clone())?
        .use_cpu()
        .build_from_files([model_file.as_ref()])?;

    AUDIO_GRAPH.set(Mutex::new(graph)).map_err(|_| {
            let err_msg = "Failed to initialize the audio context. Reason: The `AUDIO_GRAPH` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The audio context has been initialized");

    Ok(())
}

/// Initialize the piper context
///
/// # Arguments
///
/// * `voice_model` - Path to the voice model file.
///
/// * `voice_config` - Path to the voice config file.
///
/// * `espeak_ng_data` - Path to the espeak-ng data directory.
///
pub fn init_piper_context(
    piper_metadata: &PiperMetadata,
    voice_model: impl AsRef<Path>,
    voice_config: impl AsRef<Path>,
    espeak_ng_data: impl AsRef<Path>,
) -> Result<(), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Initializing the piper context");

    let config = serde_json::json!({
        "model": voice_model.as_ref().to_owned(),
        "config": voice_config.as_ref().to_owned(),
        "espeak_data": espeak_ng_data.as_ref().to_owned(),
    });

    // create and initialize the audio context
    let graph = GraphBuilder::new(EngineType::Piper)?
        .with_config(piper_metadata.clone())?
        .use_cpu()
        .build_from_buffer([config.to_string()])?;

    PIPER_GRAPH.set(Mutex::new(graph)).map_err(|_| {
            let err_msg = "Failed to initialize the piper context. Reason: The `PIPER_GRAPH` has already been initialized";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            LlamaCoreError::InitContext(err_msg.into())
        })?;

    #[cfg(feature = "logging")]
    info!(target: "stdout", "The piper context has been initialized");

    Ok(())
}
