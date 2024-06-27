# NVIDIA NIM RAG-Based Chatbot

## Project Overview

This project is a state-of-the-art chatbot that utilizes NVIDIA NIM RAG (Retrieval-Augmented Generation) and cutting-edge AI models. The chatbot is designed to provide precise and contextually relevant answers by retrieving information from a vast corpus of documents and generating human-like responses.

## Project Highlights

- **NVIDIA AI Integration**: The chatbot integrates the latest community-built AI models, optimized and accelerated by NVIDIA, ensuring high-performance and accurate responses.
- **RAG Implementation**: Employs Retrieval-Augmented Generation to deliver precise answers by retrieving and generating responses.
- **NVIDIA NIM Inference Microservices**: Deployed using NVIDIA NIM inference microservices, enabling seamless and scalable deployment across various platforms.
- **User-Friendly Interface**: Features an interactive and aesthetically pleasing interface built with Streamlit, making it easy for users to interact and get their queries answered efficiently.

## Explore More

This project showcases the potential of NVIDIA's robust AI infrastructure in creating sophisticated and responsive chatbots. By utilizing NVIDIA’s optimized models and microservices, this solution is not only powerful but also flexible for deployment anywhere.

## How It Works

1. **Embeddings**: Uses NVIDIAEmbeddings to process and transform document data.
2. **Document Loader**: Loads documents with PyPDFDirectoryLoader.
3. **Text Splitting**: Splits text into manageable chunks for efficient processing.
4. **Vector Store**: Utilizes FAISS for creating a vector store of the documents.
5. **Prompt Template**: Defines the interaction template for the chatbot.
6. **Retrieval Chain**: Combines document retrieval with LLM for precise responses.

## Installation and Setup

To set up and run the project locally, follow these steps:

1. Clone the repository:

    ```sh
    git clone https://github.com/anirxudh/NVIDIA_NIM-demo.git
    ```

2. Change into the project directory:

    ```sh
    cd NVIDIA_NIM-demo
    ```

3. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate  # MacOS/Linux
    ```

4. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

5. Set up the NVIDIA API Key:

    ```sh
    export NVIDIA_API_KEY=your_nvidia_api_key
    ```

6. Run the Streamlit application:

    ```sh
    streamlit run app.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License.

---

![NVIDIA Logo](https://www.nvidia.com/etc/clientlibs/nvidia/clientlibs/clientlib-site/resources/images/nvidia-logo.png)

## Images

![Sample Interface 1](images/Screenshot%202024-06-23%20214847.png)

![Sample Interface 2](images/Screenshot%202024-06-23%20215513.png)

![Sample Interface 3](images/Screenshot%202024-06-23%20215620.png)

