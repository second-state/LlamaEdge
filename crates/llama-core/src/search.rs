use crate::{error::LlamaCoreError, CHAT_GRAPHS};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Possible input/output Content Types. Currently only supports JSON.
#[derive(Debug, Eq, PartialEq)]
pub enum ContentType {
    JSON,
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                ContentType::JSON => "application/json",
            }
        )
    }
}

/// The base Search Configuration holding all relevant information to access a search api and retrieve results.
#[derive(Debug)]
pub struct SearchConfig {
    /// The search engine we're currently focusing on. Currently only one supported, to ensure stability.
    #[allow(dead_code)]
    pub search_engine: String,
    /// The total number of results.
    pub max_search_results: u8,
    /// The size limit of every search result.
    pub size_limit_per_result: u16,
    /// The endpoint for the search API.
    pub endpoint: String,
    /// The content type of the input.
    pub content_type: ContentType,
    /// The (expected) content type of the output.
    pub output_content_type: ContentType,
    /// Method expected by the api endpoint.
    pub method: String,
    /// Additional headers for any other purpose.
    pub additional_headers: Option<std::collections::HashMap<String, String>>,
    /// Callback function to parse the output of the api-service. Implementation left to the user.
    pub parser: fn(&serde_json::Value) -> Result<SearchOutput, Box<dyn std::error::Error>>,
    /// Prompts for use with summarization functionality. If set to `None`, use hard-coded prompts.
    pub summarization_prompts: Option<(String, String)>,
    /// Context size for summary generation. If `None`, will use the 4 char ~ 1 token metric to generate summary.
    pub summarize_ctx_size: Option<usize>,
}

/// output format for individual results in the final output.
#[derive(Serialize, Deserialize)]
pub struct SearchResult {
    pub url: String,
    pub site_name: String,
    pub text_content: String,
}

/// Final output format for consumption by the LLM.
#[derive(Serialize, Deserialize)]
pub struct SearchOutput {
    pub results: Vec<SearchResult>,
}

impl SearchConfig {
    /// Wrapper for the parser() function.
    pub fn parse_into_results(
        &self,
        raw_results: &serde_json::Value,
    ) -> Result<SearchOutput, Box<dyn std::error::Error>> {
        (self.parser)(raw_results)
    }
    pub fn new(
        search_engine: String,
        max_search_results: u8,
        size_limit_per_result: u16,
        endpoint: String,
        content_type: ContentType,
        output_content_type: ContentType,
        method: String,
        additional_headers: Option<std::collections::HashMap<String, String>>,
        parser: fn(&serde_json::Value) -> Result<SearchOutput, Box<dyn std::error::Error>>,
        summarization_prompts: Option<(String, String)>,
        summarize_ctx_size: Option<usize>,
    ) -> SearchConfig {
        SearchConfig {
            search_engine,
            max_search_results,
            size_limit_per_result,
            endpoint,
            content_type,
            output_content_type,
            method,
            additional_headers,
            parser,
            summarization_prompts,
            summarize_ctx_size,
        }
    }
    /// Perform a web search with a `Serialize`-able input. The `search_input` is used as is to query the search endpoint.
    pub async fn perform_search<T: Serialize>(
        &self,
        search_input: &T,
    ) -> Result<SearchOutput, LlamaCoreError> {
        let client = Client::new();
        let url = match Url::parse(&self.endpoint) {
            Ok(url) => url,
            Err(_) => {
                let msg = "Malformed endpoint url";
                #[cfg(feature = "logging")]
                error!(target: "stdout", "perform_search: {}", msg);
                return Err(LlamaCoreError::Search(format!(
                    "When parsing endpoint url: {}",
                    msg
                )));
            }
        };

        let method_as_string = match reqwest::Method::from_bytes(self.method.as_bytes()) {
            Ok(method) => method,
            _ => {
                let msg = "Non Standard or unknown method";
                #[cfg(feature = "logging")]
                error!(target: "stdout", "perform_search: {}", msg);
                return Err(LlamaCoreError::Search(format!(
                    "When converting method from bytes: {}",
                    msg
                )));
            }
        };

        let mut req = client.request(method_as_string.clone(), url);

        // check headers.
        req = req.headers(
            match (&self
                .additional_headers
                .clone()
                .unwrap_or_else(|| std::collections::HashMap::new()))
                .try_into()
            {
                Ok(headers) => headers,
                Err(_) => {
                    let msg = "Failed to convert headers from HashMaps to HeaderMaps";
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "On converting headers: {}",
                        msg
                    )));
                }
            },
        );

        // For POST requests, search_input goes into the request body. For GET requests, in the
        // params.
        req = match method_as_string {
            reqwest::Method::POST => match self.content_type {
                ContentType::JSON => req.json(search_input),
            },
            reqwest::Method::GET => req.query(search_input),
            _ => {
                let msg = format!(
                    "Unsupported request method: {}",
                    method_as_string.to_owned()
                );
                #[cfg(feature = "logging")]
                error!(target: "stdout", "perform_search: {}", msg);
                return Err(LlamaCoreError::Search(msg));
            }
        };

        let res = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                #[cfg(feature = "logging")]
                error!(target: "stdout", "perform_search: {}", msg);
                return Err(LlamaCoreError::Search(format!(
                    "When recieving response: {}",
                    msg
                )));
            }
        };

        match res.content_length() {
            Some(length) => {
                if length == 0 {
                    let msg = "Empty response from server";
                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "perform_search: {}", msg);
                    return Err(LlamaCoreError::Search(format!(
                        "Unexpected content length: {}",
                        msg
                    )));
                }
            }
            None => {
                let msg = "Content length returned None";
                #[cfg(feature = "logging")]
                error!(target: "stdout", "perform_search: {}", msg);
                return Err(LlamaCoreError::Search(format!(
                    "Content length field not found: {}",
                    msg
                )));
            }
        }

        // start parsing the output.
        //
        // only checking for JSON as the output content type since it's the most common and widely
        // supported.
        let raw_results: Value;
        match self.output_content_type {
            ContentType::JSON => {
                let body_text = match res.text().await {
                    Ok(body) => body,
                    Err(e) => {
                        let msg = e.to_string();
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "perform_search: {}", msg);
                        return Err(LlamaCoreError::Search(format!(
                            "When accessing response body: {}",
                            msg
                        )));
                    }
                };
                println!("{}", body_text);
                raw_results = match serde_json::from_str(body_text.as_str()) {
                    Ok(value) => value,
                    Err(e) => {
                        let msg = e.to_string();
                        #[cfg(feature = "logging")]
                        error!(target: "stdout", "perform_search: {}", msg);
                        return Err(LlamaCoreError::Search(format!(
                            "When converting to a JSON object: {}",
                            msg
                        )));
                    }
                };
            }
        };

        // start cleaning the output.

        // produce SearchOutput instance with the raw results obtained from the endpoint.
        let mut search_output: SearchOutput = match self.parse_into_results(&raw_results) {
            Ok(search_output) => search_output,
            Err(e) => {
                let msg = e.to_string();
                #[cfg(feature = "logging")]
                error!(target: "stdout", "perform_search: {}", msg);
                return Err(LlamaCoreError::Search(format!(
                    "When calling parse_into_results: {}",
                    msg
                )));
            }
        };

        // apply maximum search result limit.
        search_output
            .results
            .truncate(self.max_search_results as usize);

        // apply per result character limit.
        //
        // since the clipping only happens when split_at_checked() returns Some, the results will
        // remain unchanged should split_at_checked() return None.
        for result in search_output.results.iter_mut() {
            if let Some(clipped_content) = result
                .text_content
                .split_at_checked(self.size_limit_per_result as usize)
            {
                result.text_content = clipped_content.0.to_string();
            }
        }

        // Search Output cleaned and finalized.
        Ok(search_output)
    }
    /// Perform a search and summarize the corresponding search results
    pub async fn summarize_search<T: Serialize>(
        &self,
        search_input: &T,
    ) -> Result<String, LlamaCoreError> {
        let search_output = self.perform_search(&search_input).await?;

        let summarization_prompts = self.summarization_prompts.clone().unwrap_or((
            "The following are search results I found on the internet:\n\n".to_string(),
            "\n\nTo sum up them up: ".to_string(),
        ));

        // the fallback context size limit for the search summary to be generated.
        let summarize_ctx_size = self
            .summarize_ctx_size
            .unwrap_or((self.size_limit_per_result * self.max_search_results as u16) as usize);

        summarize(
            search_output,
            summarize_ctx_size,
            summarization_prompts.0,
            summarization_prompts.1,
        )
    }
}

/// Summarize the search output provided
fn summarize(
    search_output: SearchOutput,
    summarize_ctx_size: usize,
    initial_prompt: String,
    final_prompt: String,
) -> Result<String, LlamaCoreError> {
    let mut search_output_string: String = String::new();

    // Add the text content of every result together.
    search_output
        .results
        .iter()
        .for_each(|result| search_output_string.push_str(result.text_content.as_str()));

    // Error on embedding running mode.
    if crate::running_mode()? == crate::RunningMode::Embeddings {
        let err_msg = "Summarization is not supported in the EMBEDDINGS running mode.";

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", err_msg);

        return Err(LlamaCoreError::Search(err_msg.into()));
    }

    // Get graphs and pick the first graph.
    let chat_graphs = match CHAT_GRAPHS.get() {
        Some(chat_graphs) => chat_graphs,
        None => {
            let err_msg = "Fail to get the underlying value of `CHAT_GRAPHS`.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Search(err_msg.into()));
        }
    };

    let mut chat_graphs = chat_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Search(err_msg)
    })?;

    // Prepare input prompt.
    let input = initial_prompt + search_output_string.as_str() + final_prompt.as_str();
    let tensor_data = input.as_bytes().to_vec();

    // Use first available chat graph
    let graph: &mut crate::Graph = match chat_graphs.values_mut().next() {
        Some(graph) => graph,
        None => {
            let err_msg = "No available chat graph.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", err_msg);

            return Err(LlamaCoreError::Search(err_msg.into()));
        }
    };

    graph
        .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set prompt as the input tensor");

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Generating a summary for search results...");
    // Execute the inference.
    graph.compute().expect("Failed to complete inference");

    // Retrieve the output.
    let mut output_buffer = vec![0u8; summarize_ctx_size];
    let mut output_size = graph
        .get_output(0, &mut output_buffer)
        .expect("Failed to get output tensor");
    output_size = std::cmp::min(summarize_ctx_size, output_size);

    // Compute lossy UTF-8 output (text only).
    let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Summary generated.");

    Ok(output)
}
