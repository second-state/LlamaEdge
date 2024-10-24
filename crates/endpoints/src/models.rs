//! Define types for the `models` endpoint.

use serde::{Deserialize, Serialize};

/// Lists the currently available models, and provides basic information about each one such as the owner and availability.
#[derive(Debug, Deserialize, Serialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

/// Describes a model offering that can be used with the API.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Model {
    /// The model identifier, which can be referenced in the API endpoints.
    pub id: String,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u64,
    /// The object type, which is always "model".
    pub object: String,
    /// The organization that owns the model.
    pub owned_by: String,
}
