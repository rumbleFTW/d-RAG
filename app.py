import gradio as gr

from langchain_community.vectorstores import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import text
from langchain_community.embeddings import HuggingFaceEmbeddings


from core import LLM

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = LLM()


def infer(system_prompt, user_prompt, file_input):
    loader = text.TextLoader(file_path=file_input)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    splitted_docs = text_splitter.split_documents(documents=documents)
    db = faiss.FAISS.from_documents(embedding=embeddings, documents=splitted_docs)
    retriever = db.as_retriever()
    context = retriever.invoke(user_prompt)[0].page_content

    system = {"role": "system", "content": system_prompt}
    user = {
        "role": "user",
        "content": f"prompt: ```{user_prompt}``\ncontext:```{context}```",
    }
    # print(system, user)
    response = llm([system, user]).split("<|assistant|>")[-1]
    return response


with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            system_prompt = gr.Textbox(
                label="System prompt",
                lines=4,
                value="You are an assistant. Answer to a user's query based on a given context.",
            )
            user_prompt = gr.Textbox(label="User prompt", lines=4)
            file_input = gr.File(
                label="Enter you .txt document.", file_types=["text"], type="filepath"
            )
            submit = gr.Button("Submit")
        with gr.Column():
            text_output = gr.TextArea(lines=26, label="Response")

        submit.click(
            infer,
            inputs=[
                system_prompt,
                user_prompt,
                file_input,
            ],
            outputs=[text_output],
        )


if __name__ == "__main__":
    iface.launch(share=True)
