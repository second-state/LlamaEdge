//! Define APIs for querying models.

use crate::{error::LlamaCoreError, CHAT_GRAPHS, EMBEDDING_GRAPHS};
use endpoints::models::{ListModelsResponse, Model};

/// Lists models available
pub async fn models() -> Result<ListModelsResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "List models");

    let mut models = vec![];

    {
        if let Some(chat_graphs) = CHAT_GRAPHS.get() {
            let chat_graphs = chat_graphs.lock().map_err(|e| {
                let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Operation(err_msg)
            })?;

            for (name, graph) in chat_graphs.iter() {
                models.push(Model {
                    id: name.clone(),
                    created: graph.created.as_secs(),
                    object: String::from("model"),
                    owned_by: String::from("Not specified"),
                });
            }
        }
    }

    {
        if let Some(embedding_graphs) = EMBEDDING_GRAPHS.get() {
            let embedding_graphs = embedding_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `EMBEDDING_GRAPHS`. {}",
                    e
                ))
            })?;

            if !embedding_graphs.is_empty() {
                for (name, graph) in embedding_graphs.iter() {
                    models.push(Model {
                        id: name.clone(),
                        created: graph.created.as_secs(),
                        object: String::from("model"),
                        owned_by: String::from("Not specified"),
                    });
                }
            }
        }
    }

    Ok(ListModelsResponse {
        object: String::from("list"),
        data: models,
    })
}
