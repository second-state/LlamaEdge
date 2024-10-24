# LlamaEdge API Server

LlamaEdge API server offers OpenAI-compatible REST APIs. It can accelerate developers to build LLM-driven applications, AI infrastructure, and etc. In addition, LlamaEdge is also friendly to AI frameworks, such as LangChain and LlamaIndex.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [LlamaEdge API Server](#llamaedge-api-server)
  - [Dependencies](#dependencies)
  - [Get LlamaEdge API server](#get-llamaedge-api-server)
  - [Get model](#get-model)
  - [Run LlamaEdge API server](#run-llamaedge-api-server)
  - [Endpoints](#endpoints)
    - [`/v1/models` endpoint](#v1models-endpoint)
    - [`/v1/chat/completions` endpoint](#v1chatcompletions-endpoint)
    - [`/v1/files` endpoint](#v1files-endpoint)
    - [`/v1/chunks` endpoint](#v1chunks-endpoint)
    - [`/v1/embeddings` endpoint](#v1embeddings-endpoint)
    - [`/v1/completions` endpoint](#v1completions-endpoint)
  - [Add a web UI](#add-a-web-ui)
  - [CLI options for the API server](#cli-options-for-the-api-server)
  - [Set Log Level](#set-log-level)

<!-- /code_chunk_output -->

## Dependencies

Install the latest WasmEdge with plugins:

<details> <summary> For macOS (apple silicon) </summary>

```console
# install WasmEdge with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s

# Assuming you use zsh (the default shell on macOS), run the following command to activate the environment
source $HOME/.zshenv
```

</details>

<details> <summary> For Ubuntu (>= 20.04) </summary>

```console
# install libopenblas-dev
apt update && apt install -y libopenblas-dev

# install WasmEdge with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

<details> <summary> For General Linux </summary>

```console
# install WasmEdge with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

## Get LlamaEdge API server

- Download LlamaEdge API server with the support for `HTTP` scheme only:

  ```console
  curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
  ```

- Download LlamaEdge API server with the support for `HTTP` and `WebSocket` schemes:

  ```console
  curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server-full.wasm
  ```

## Get model

[huggingface.co/second-state](https://huggingface.co/second-state?search_models=GGUF) maintains a group of GGUF models for different usages. In addition, you can also pick a GGUF model from the [https://huggingface.co/models?sort=trending&search=gguf](https://huggingface.co/models?sort=trending&search=gguf).

## Run LlamaEdge API server

Run the API server with the following command:

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Meta-Llama-3-8B-Instruct-Q5_K_M.gguf \
  llama-api-server.wasm \
  --prompt-template llama-3-chat \
  --ctx-size 4096 \
  --model-name llama-3-8b

```

The command above starts the API server on the default socket address. Besides, there are also some other options specified in the command:

- The `--dir .:.` option specifies the current directory as the root directory of the WASI file system.

- The `--nn-preload default:GGML:AUTO:Meta-Llama-3-8B-Instruct-Q5_K_M.gguf` option specifies the Llama model to be used by the API server. The pattern of the argument is `<name>:<encoding>:<target>:<model path>`. Here, the model used is `Meta-Llama-3-8B-Instruct-Q5_K_M.gguf`; and we give it an alias `default` as its name in the runtime environment. You can change the model name here if you're not using llama-3-8b.
- The `--prompt-template llama-3-chat` is the prompt template for the model.
- The `--model-name llama-3-8b` specifies the model name. It is used in the chat request.

## Endpoints

### `/v1/models` endpoint

`/v1/models` endpoint is used to list models running on LlamaEdge API server.

<details> <summary> Example </summary>

You can use `curl` to test it on a new terminal:

```bash
curl -X GET http://localhost:8080/v1/models -H 'accept:application/json'
```

If the command is successful, you should see the similar output as below in your terminal:

```json
{
    "object":"list",
    "data":[
        {
            "id":"llama-3-8b",
            "created":1697084821,
            "object":"model",
            "owned_by":"Not specified"
        }
    ]
}
```

</details>

### `/v1/chat/completions` endpoint

`/v1/chat/completions` endpoint is used for multi-turn conversations between human users and LLM models.

<details> <summary> Example </summary>

The following command sends a chat request with a user's question to the LLM model named `llama-3-8b`:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-3-8b"}'
```

Here is the response from LlamaEdge API server:

```json
{
    "id":"",
    "object":"chat.completion",
    "created":1697092593,
    "model":"llama-3-8b",
    "choices":[
        {
            "index":0,
            "message":{
                "role":"assistant",
                "content":"Robert Oppenheimer was an American theoretical physicist and director of the Manhattan Project, which developed the atomic bomb during World War II. He is widely regarded as one of the most important physicists of the 20th century and is known for his contributions to the development of quantum mechanics and the theory of the atomic nucleus. Oppenheimer was also a prominent figure in the post-war nuclear weapons debate, advocating for international control and regulation of nuclear weapons."
            },
            "finish_reason":"stop"
        }
    ],
    "usage":{
        "prompt_tokens":9,
        "completion_tokens":12,
        "total_tokens":21
    }
}
```

</details>

### `/v1/files` endpoint

`/v1/files` endpoint is used for uploading text and markdown files to LlamaEdge API server.

<details> <summary> Example: Upload files </summary>

The following command upload a text file [paris.txt](https://huggingface.co/datasets/gaianet/paris/raw/main/paris.txt) to the API server via the `/v1/files` endpoint:

```bash
curl -X POST http://127.0.0.1:8080/v1/files -F "file=@paris.txt"
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "id": "file_4bc24593-2a57-4646-af16-028855e7802e",
    "bytes": 2161,
    "created_at": 1711611801,
    "filename": "paris.txt",
    "object": "file",
    "purpose": "assistants"
}
```

The `id` and `filename` fields are important for the next step, for example, to segment the uploaded file to chunks for computing embeddings.

If you'd like to build a RAG chatbot, it's strongly recommended to visit [LlamaEdge-RAG API Server](https://github.com/LlamaEdge/rag-api-server).

</details>

<details> <summary> Example: List files </summary>

The following command lists all files on the server via the `/v1/files` endpoint:

```bash
curl -X GET http://127.0.0.1:8080/v1/files
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "object": "list",
    "data": [
        {
            "id": "file_33d9188d-5060-4141-8c52-ae148fd15f6a",
            "bytes": 17039,
            "created_at": 1718296362,
            "filename": "test-123.m4a",
            "object": "file",
            "purpose": "assistants"
        },
        {
            "id": "file_8c6439da-df59-4b9a-bb5e-dba4b2f23c04",
            "bytes": 17039,
            "created_at": 1718294169,
            "filename": "test-123.m4a",
            "object": "file",
            "purpose": "assistants"
        },
        {
            "id": "file_6c601277-7deb-44c9-bfb3-57ce9da856c9",
            "bytes": 17039,
            "created_at": 1718296350,
            "filename": "test-123.m4a",
            "object": "file",
            "purpose": "assistants"
        },
        {
            "id": "file_137b1ea2-c01d-44da-83ad-6b4aa2ff71de",
            "bytes": 244596,
            "created_at": 1718337557,
            "filename": "audio16k.wav",
            "object": "file",
            "purpose": "assistants"
        },
        {
            "id": "file_21fde6a7-18dc-4d42-a5bb-1a27d4b7a32e",
            "bytes": 17039,
            "created_at": 1718294739,
            "filename": "test-123.m4a",
            "object": "file",
            "purpose": "assistants"
        },
        {
            "id": "file_b892bc81-35e9-44a6-8c01-ae915c1d3832",
            "bytes": 2161,
            "created_at": 1715832065,
            "filename": "paris.txt",
            "object": "file",
            "purpose": "assistants"
        },
        {
            "id": "file_6a6d8046-fd98-410a-b70e-0a0142ec9a39",
            "bytes": 17039,
            "created_at": 1718332593,
            "filename": "test-123.m4a",
            "object": "file",
            "purpose": "assistants"
        }
    ]
}
```

</details>

<details> <summary> Example: Retrieve information about a specific file </summary>

The following command retrieves information about a specific file on the server via the `/v1/files/{file_id}` endpoint:

```bash
curl -X GET http://localhost:10086/v1/files/file_b892bc81-35e9-44a6-8c01-ae915c1d3832
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "id": "file_b892bc81-35e9-44a6-8c01-ae915c1d3832",
    "bytes": 2161,
    "created_at": 1715832065,
    "filename": "paris.txt",
    "object": "file",
    "purpose": "assistants"
}
```

</details>

<details> <summary> Example: Delete a specific file </summary>

The following command deletes a specific file on the server via the `/v1/files/{file_id}` endpoint:

```bash
curl -X DELETE http://localhost:10086/v1/files/file_6a6d8046-fd98-410a-b70e-0a0142ec9a39
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "id": "file_6a6d8046-fd98-410a-b70e-0a0142ec9a39",
    "object": "file",
    "deleted": true
}
```

</details>

### `/v1/chunks` endpoint

To segment the uploaded file to chunks for computing embeddings, use the `/v1/chunks` API.

<details> <summary> Example </summary>

The following command sends the uploaded file ID and filename to the API server and gets the chunks:

```bash
curl -X POST http://localhost:8080/v1/chunks \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id":"file_4bc24593-2a57-4646-af16-028855e7802e", "filename":"paris.txt", "chunk_capacity":100}'
```

The following is an example return with the generated chunks:

```json
{
    "id": "file_4bc24593-2a57-4646-af16-028855e7802e",
    "filename": "paris.txt",
    "chunks": [
        "Paris, city and capital of France, ... and far beyond both banks of the Seine.",
        "Paris occupies a central position in the rich agricultural region ... metropolitan area, 890 square miles (2,300 square km).",
        "Pop. (2020 est.) city, 2,145,906; (2020 est.) urban agglomeration, 10,858,874.",
        "For centuries Paris has been one of the world’s ..., for Paris has retained its importance as a centre for education and intellectual pursuits.",
        "Paris’s site at a crossroads of both water and land routes ... The Frankish king Clovis I had taken Paris from the Gauls by 494 CE and later made his capital there.",
        "Under Hugh Capet (ruled 987–996) and the Capetian dynasty ..., drawing to itself much of the talent and vitality of the provinces."
    ]
}
```

If you'd like to build a RAG chatbot, it's strongly recommended to visit [LlamaEdge-RAG API Server](https://github.com/LlamaEdge/rag-api-server).

</details>

### `/v1/embeddings` endpoint

To compute embeddings for user query or file chunks, use the `/v1/embeddings` API.

<details> <summary> Example </summary>

The following command sends a query to the API server and gets the embeddings as return:

```bash
curl -X POST http://localhost:8080/v1/embeddings \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"model": "e5-mistral-7b-instruct-Q5_K_M", "input":["Paris, city and capital of France, ... and far beyond both banks of the Seine.","Paris occupies a central position in the rich agricultural region ... metropolitan area, 890 square miles (2,300 square km).","Pop. (2020 est.) city, 2,145,906; (2020 est.) urban agglomeration, 10,858,874.","For centuries Paris has been one of the world’s ..., for Paris has retained its importance as a centre for education and intellectual pursuits.","Paris’s site at a crossroads of both water and land routes ... The Frankish king Clovis I had taken Paris from the Gauls by 494 CE and later made his capital there.","Under Hugh Capet (ruled 987–996) and the Capetian dynasty ..., drawing to itself much of the talent and vitality of the provinces."]}'
```

The embeddings returned are like below:

```json
{
    "object": "list",
    "data": [
        {
            "index": 0,
            "object": "embedding",
            "embedding": [
                0.1477311701,
                -0.00002238310481,
                ...,
                0.01931835897,
                -0.02496444248
            ]
        },
        {
            "index": 1,
            "object": "embedding",
            "embedding": [
                0.1766036302,
                -0.009940749966,
                ...,
                0.0156990625,
                -0.02616829611
            ]
        },
        {
            "index": 2,
            "object": "embedding",
            "embedding": [
                0.04604972154,
                -0.07207781076,
                ...,
                0.00005568400593,
                0.04646552354
            ]
        },
        {
            "index": 3,
            "object": "embedding",
            "embedding": [
                0.1065238863,
                -0.04788689688,
                ...,
                0.0301867798,
                0.0275206212
            ]
        },
        {
            "index": 4,
            "object": "embedding",
            "embedding": [
                0.05383823439,
                0.03193736449,
                ...,
                0.01904040016,
                -0.02546775527
            ]
        },
        {
            "index": 5,
            "object": "embedding",
            "embedding": [
                0.05341234431,
                0.005945806392,
                ...,
                0.06845153868,
                0.02127391472
            ]
        }
    ],
    "model": "all-MiniLM-L6-v2-ggml-model-f16",
    "usage": {
        "prompt_tokens": 495,
        "completion_tokens": 0,
        "total_tokens": 495
    }
}
```

If you'd like to build a RAG chatbot, it's strongly recommended to visit [LlamaEdge-RAG API Server](https://github.com/LlamaEdge/rag-api-server).

</details>

### `/v1/completions` endpoint

To obtain the completion for a single prompt, use the `/v1/completions` API.

<details> <summary> Example </summary>

The following command sends a prompt to the API server and gets the completion:

```bash
curl -X POST http://localhost:8080/v1/completions \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"prompt":["Long long ago, "], "model":"tinyllama"}'
```

The response looks like below:

```json
{
    "id": "b68bfc92-8b23-4435-bbe1-492e11c973a3",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "text": "in a galaxy far, far away, a group of Jedi Knights lived and trained to defend the peace and security of the galaxy. They were the last hope for peace and justice in a world where the dark side of the force was rife with corruption and injustice. The Knights were a select few, and their training and abilities were the envy of the galaxy. They were the chosen ones. They were the ones who would bring peace and justice to the galaxy. ..."
        }
    ],
    "created": 1702046592,
    "model": "tinyllama",
    "object": "text_completion",
    "usage": {
        "prompt_tokens": 3,
        "completion_tokens": 804,
        "total_tokens": 807
    }
}
```

</details>

## Add a web UI

We provide a front-end Web UI for you to easily interact with the API. You can download and extract it by running:

```bash
curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
```

After that, you can use the same command line to create the API server

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:Meta-Llama-3-8B-Instruct-Q5_K_M.gguf \
  llama-api-server.wasm \
  --prompt-template llama-3-chat \
  --ctx-size 4096 \
  --model-name llama-3-8b
```

Then, you will be asked to open `http://127.0.0.1:8080` from your browser.

<a id="cli-options"></a>

## CLI options for the API server

The `-h` or `--help` option can list the available options of the `llama-api-server` wasm app:

```console
$ wasmedge llama-api-server.wasm -h

LlamaEdge API Server

Usage: llama-api-server.wasm [OPTIONS] --prompt-template <PROMPT_TEMPLATE>

Options:
  -m, --model-name <MODEL_NAME>
          Sets names for chat and/or embedding models. To run both chat and embedding models, the names should be separated by comma without space, for example, '--model-name Llama-2-7b,all-minilm'. The first value is for the chat model, and the second is for the embedding model [default: default]
  -a, --model-alias <MODEL_ALIAS>
          Model aliases for chat and embedding models [default: default,embedding]
  -c, --ctx-size <CTX_SIZE>
          Sets context sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--ctx-size 4096,384'. The first value is for the chat model, and the second is for the embedding model [default: 4096,384]
  -b, --batch-size <BATCH_SIZE>
          Sets batch sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--batch-size 128,64'. The first value is for the chat model, and the second is for the embedding model [default: 512,512]
  -p, --prompt-template <PROMPT_TEMPLATE>
          Sets prompt templates for chat and/or embedding models, respectively. To run both chat and embedding models, the prompt templates should be separated by comma without space, for example, '--prompt-template llama-2-chat,embedding'. The first value is for the chat model, and the second is for the embedding model [possible values: llama-2-chat, llama-3-chat, llama-3-tool, mistral-instruct, mistral-tool, mistrallite, openchat, codellama-instruct, codellama-super-instruct, human-assistant, vicuna-1.0-chat, vicuna-1.1-chat, vicuna-llava, chatml, chatml-tool, internlm-2-tool, baichuan-2, wizard-coder, zephyr, stablelm-zephyr, intel-neural, deepseek-chat, deepseek-coder, deepseek-chat-2, deepseek-chat-25, solar-instruct, phi-2-chat, phi-2-instruct, phi-3-chat, phi-3-instruct, gemma-instruct, octopus, glm-4-chat, groq-llama3-tool, mediatek-breeze, nemotron-chat, nemotron-tool, functionary-32, functionary-31, embedding, none]
  -r, --reverse-prompt <REVERSE_PROMPT>
          Halt generation at PROMPT, return control
  -n, --n-predict <N_PREDICT>
          Number of tokens to predict [default: 1024]
  -g, --n-gpu-layers <N_GPU_LAYERS>
          Number of layers to run on the GPU [default: 100]
      --main-gpu <MAIN_GPU>
          The main GPU to use
      --tensor-split <TENSOR_SPLIT>
          How split tensors should be distributed accross GPUs. If None the model is not split; otherwise, a comma-separated list of non-negative values, e.g., "3,2" presents 60% of the data to GPU 0 and 40% to GPU 1
      --threads <THREADS>
          Number of threads to use during computation [default: 2]
      --no-mmap <NO_MMAP>
          Disable memory mapping for file access of chat models [possible values: true, false]
      --temp <TEMP>
          Temperature for sampling [default: 1.0]
      --top-p <TOP_P>
          An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 1.0 = disabled [default: 1.0]
      --repeat-penalty <REPEAT_PENALTY>
          Penalize repeat sequence of tokens [default: 1.1]
      --presence-penalty <PRESENCE_PENALTY>
          Repeat alpha presence penalty. 0.0 = disabled [default: 0.0]
      --frequency-penalty <FREQUENCY_PENALTY>
          Repeat alpha frequency penalty. 0.0 = disabled [default: 0.0]
      --grammar <GRAMMAR>
          BNF-like grammar to constrain generations (see samples in grammars/ dir) [default: ]
      --json-schema <JSON_SCHEMA>
          JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
      --llava-mmproj <LLAVA_MMPROJ>
          Path to the multimodal projector file
      --socket-addr <SOCKET_ADDR>
          Socket address of LlamaEdge API Server instance. For example, `0.0.0.0:8080`
      --port <PORT>
          Port number [default: 8080]
      --web-ui <WEB_UI>
          Root path for the Web UI files [default: chatbot-ui]
      --log-prompts
          Deprecated. Print prompt strings to stdout
      --log-stat
          Deprecated. Print statistics to stdout
      --log-all
          Deprecated. Print all log information to stdout
  -h, --help
          Print help
  -V, --version
          Print version
```

Please guarantee that the port is not occupied by other processes. If the port specified is available on your machine and the command is successful, you should see the following output in the terminal:

```console
[INFO] LlamaEdge HTTP listening on 8080
```

If the Web UI is ready, you can navigate to `http://127.0.0.1:8080` to open the chatbot, it will interact with the API of your server.

## Set Log Level

You can set the log level of the API server by setting the `LLAMA_LOG` environment variable. For example, to set the log level to `debug`, you can run the following command:

```bash
wasmedge --dir .:. --env RUST_LOG=debug \
    --nn-preload default:GGML:AUTO:Meta-Llama-3-8B-Instruct-Q5_K_M.gguf \
    llama-api-server.wasm \
    --model-name llama-3-8b \
    --prompt-template llama-3-chat \
    --ctx-size 4096
```

The log level can be one of the following values: `trace`, `debug`, `info`, `warn`, `error`. The default log level is `info`.
