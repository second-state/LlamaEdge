//! Define Graph and GraphBuilder APIs for creating a new computation graph.

use crate::{error::LlamaCoreError, utils::set_tensor_data_u8, BaseMetadata};
use wasmedge_wasi_nn::{
    Error as WasiNnError, Graph as WasiNnGraph, GraphExecutionContext, TensorType,
};

/// Builder for creating a new computation graph.
#[derive(Debug)]
pub struct GraphBuilder<M: BaseMetadata + serde::Serialize + Clone + Default> {
    metadata: Option<M>,
    wasi_nn_graph_builder: wasmedge_wasi_nn::GraphBuilder,
}
impl<M: BaseMetadata + serde::Serialize + Clone + Default> GraphBuilder<M> {
    /// Create a new computation graph builder.
    pub fn new(ty: EngineType) -> Result<Self, LlamaCoreError> {
        let encoding = match ty {
            EngineType::Ggml => wasmedge_wasi_nn::GraphEncoding::Ggml,
            EngineType::Whisper => wasmedge_wasi_nn::GraphEncoding::Whisper,
            EngineType::Piper => wasmedge_wasi_nn::GraphEncoding::Piper,
        };

        let wasi_nn_graph_builder =
            wasmedge_wasi_nn::GraphBuilder::new(encoding, wasmedge_wasi_nn::ExecutionTarget::AUTO);

        Ok(Self {
            metadata: None,
            wasi_nn_graph_builder,
        })
    }

    pub fn with_config(mut self, metadata: M) -> Result<Self, LlamaCoreError> {
        let config = serde_json::to_string(&metadata).map_err(|e| {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;
        self.wasi_nn_graph_builder = self.wasi_nn_graph_builder.config(config);
        self.metadata = Some(metadata.clone());

        Ok(self)
    }

    pub fn use_cpu(mut self) -> Self {
        self.wasi_nn_graph_builder = self.wasi_nn_graph_builder.cpu();
        self
    }

    pub fn use_gpu(mut self) -> Self {
        self.wasi_nn_graph_builder = self.wasi_nn_graph_builder.gpu();
        self
    }

    pub fn use_tpu(mut self) -> Self {
        self.wasi_nn_graph_builder = self.wasi_nn_graph_builder.tpu();
        self
    }

    pub fn build_from_buffer<B>(
        self,
        bytes_array: impl AsRef<[B]>,
    ) -> Result<Graph<M>, LlamaCoreError>
    where
        B: AsRef<[u8]>,
    {
        // load the model
        let graph = self
            .wasi_nn_graph_builder
            .build_from_bytes(bytes_array)
            .map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

        // initialize the execution context
        let context = graph.init_execution_context().map_err(|e| {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

        Ok(Graph {
            created,
            metadata: self.metadata.clone().unwrap_or_default(),
            _graph: graph,
            context,
        })
    }

    pub fn build_from_files<P>(self, files: impl AsRef<[P]>) -> Result<Graph<M>, LlamaCoreError>
    where
        P: AsRef<std::path::Path>,
    {
        // load the model
        let graph = self
            .wasi_nn_graph_builder
            .build_from_files(files)
            .map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

        // initialize the execution context
        let context = graph.init_execution_context().map_err(|e| {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

        Ok(Graph {
            created,
            metadata: self.metadata.clone().unwrap_or_default(),
            _graph: graph,
            context,
        })
    }

    pub fn build_from_cache(self) -> Result<Graph<M>, LlamaCoreError> {
        match &self.metadata {
            Some(metadata) => {
                // load the model
                let graph = self
                    .wasi_nn_graph_builder
                    .build_from_cache(metadata.model_alias())
                    .map_err(|e| {
                        let err_msg = e.to_string();

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        LlamaCoreError::Operation(err_msg)
                    })?;

                // initialize the execution context
                let context = graph.init_execution_context().map_err(|e| {
                    let err_msg = e.to_string();

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|e| {
                        let err_msg = e.to_string();

                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "{}", &err_msg);

                        LlamaCoreError::Operation(err_msg)
                    })?;

                Ok(Graph {
                    created,
                    metadata: metadata.clone(),
                    _graph: graph,
                    context,
                })
            }
            None => {
                let err_msg =
                    "Failed to create a Graph from cache. Reason: Metadata is not provided."
                        .to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                Err(LlamaCoreError::Operation(err_msg))
            }
        }
    }
}

/// Wrapper of the `wasmedge_wasi_nn::Graph` struct
#[derive(Debug)]
pub struct Graph<M: BaseMetadata + serde::Serialize + Clone + Default> {
    pub created: std::time::Duration,
    pub metadata: M,
    _graph: WasiNnGraph,
    context: GraphExecutionContext,
}
impl<M: BaseMetadata + serde::Serialize + Clone + Default> Graph<M> {
    /// Create a new computation graph from the given metadata.
    pub fn new(metadata: M) -> Result<Self, LlamaCoreError> {
        let config = serde_json::to_string(&metadata).map_err(|e| {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;

        // load the model
        let graph = wasmedge_wasi_nn::GraphBuilder::new(
            wasmedge_wasi_nn::GraphEncoding::Ggml,
            wasmedge_wasi_nn::ExecutionTarget::AUTO,
        )
        .config(config)
        .build_from_cache(metadata.model_alias())
        .map_err(|e| {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;

        // initialize the execution context
        let context = graph.init_execution_context().map_err(|e| {
            let err_msg = e.to_string();

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            LlamaCoreError::Operation(err_msg)
        })?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

        Ok(Self {
            created,
            metadata: metadata.clone(),
            _graph: graph,
            context,
        })
    }

    /// Get the name of the model
    pub fn name(&self) -> &str {
        self.metadata.model_name()
    }

    /// Get the alias of the model
    pub fn alias(&self) -> &str {
        self.metadata.model_alias()
    }

    /// Update metadata
    pub fn update_metadata(&mut self) -> Result<(), LlamaCoreError> {
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Update metadata for the model named {}", self.name());

        // update metadata
        let config = match serde_json::to_string(&self.metadata) {
            Ok(config) => config,
            Err(e) => {
                let err_msg = format!("Failed to update metadta. Reason: Fail to serialize metadata to a JSON string. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg));
            }
        };

        let res = set_tensor_data_u8(self, 1, config.as_bytes());

        #[cfg(feature = "logging")]
        info!(target: "stdout", "Metadata updated successfully.");

        res
    }

    /// Set input uses the data, not only [u8](https://doc.rust-lang.org/nightly/std/primitive.u8.html), but also [f32](https://doc.rust-lang.org/nightly/std/primitive.f32.html), [i32](https://doc.rust-lang.org/nightly/std/primitive.i32.html), etc.
    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>,
    ) -> Result<(), WasiNnError> {
        self.context.set_input(index, tensor_type, dimensions, data)
    }

    /// Compute the inference on the given inputs.
    pub fn compute(&mut self) -> Result<(), WasiNnError> {
        self.context.compute()
    }

    /// Compute the inference on the given inputs.
    ///
    /// Note that this method is used for the stream mode. It generates one token at a time.
    pub fn compute_single(&mut self) -> Result<(), WasiNnError> {
        self.context.compute_single()
    }

    /// Copy output tensor to out_buffer, return the output’s **size in bytes**.
    pub fn get_output<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output(index, out_buffer)
    }

    /// Copy output tensor to out_buffer, return the output’s **size in bytes**.
    ///
    /// Note that this method is used for the stream mode. It returns one token at a time.
    pub fn get_output_single<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut [T],
    ) -> Result<usize, WasiNnError> {
        self.context.get_output_single(index, out_buffer)
    }

    /// Clear the computation context.
    ///
    /// Note that this method is used for the stream mode. It clears the context after the stream mode is finished.
    pub fn finish_single(&mut self) -> Result<(), WasiNnError> {
        self.context.fini_single()
    }
}

/// Engine type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EngineType {
    Ggml,
    Whisper,
    Piper,
}
