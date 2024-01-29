import gradio as gr

from llms.tiny_llama import TinyLlama
from knowledgebase import KnowledgeBase
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


LLM = None
KNOWLEDGEBASE = None
SYTEM_PROMPT = (
    "You are an assistant. Answer to a user's query based on a given context."
)

# MEMORY = ConversationBufferMemory(
#     memory_key="chat_history", output_key="answer", return_messages=True
# )
QA_CHAIN = None


# def init_qa_chain():
#     QA_CHAIN = ConversationalRetrievalChain.from_llm(
#         LLM,
#         retriever=KNOWLEDGEBASE,
#         chain_type="stuff",
#         memory=MEMORY,
#         return_source_documents=True,
#         verbose=True,
#     )


def init_llm(llm="TinyLlama"):
    global LLM
    LLM = TinyLlama()


def chat(message, history):
    global LLM, KNOWLEDGEBASE, SYTEM_PROMPT
    context = KNOWLEDGEBASE.invoke(message)[0].page_content
    system = {"role": "system", "content": SYTEM_PROMPT}
    user = {
        "role": "user",
        "content": f"prompt: ```{message}``\ncontext:```{context}```",
    }
    response = LLM([system, user]).split("<|assistant|>")[-1]
    return response


def init_rag(system_prompt, url_input, file_input):
    global SYTEM_PROMPT
    if SYTEM_PROMPT != system_prompt:
        SYTEM_PROMPT = system_prompt
        gr.Info("Saved new system prompt")
    if url_input and file_input:
        gr.Error(message="Provide either an URL or a File")
    path = url_input if url_input else file_input
    load_knowledgebase(path)


def load_knowledgebase(path):
    global KNOWLEDGEBASE
    if not KNOWLEDGEBASE:
        KNOWLEDGEBASE = KnowledgeBase()
    print("Loading knowledgebase:", path)
    if not path:
        return
    if "https://" in path:
        KNOWLEDGEBASE.load_url(path)
        gr.Info(message="Succesfully loaded URL")
    else:
        if path.split(".")[-1] == "pdf":
            KNOWLEDGEBASE.load_pdf(path)
            gr.Info(message="Succesfully loaded pdf")
        else:
            KNOWLEDGEBASE.load_txt(path)
            gr.Info(message="Succesfully loaded txt")


def show_file(file):
    print(file)
    return file


with gr.Blocks(title="d-RAG") as iface:
    gr.Markdown(
        """# d-RAG &nbsp;[![Watch on GitHub](https://img.shields.io/github/watchers/rumbleFTW/d-RAG.svg?style=social)](https://github.com/rumbleFTW/d-RAG/watchers) &nbsp; [![Star on GitHub](https://img.shields.io/github/stars/rumbleFTW/d-RAG.svg?style=social)](https://github.com/rumbleFTW/d-RAG/stargazers)
"""
    )
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Row():
                model = gr.Dropdown(
                    label="Model",
                    choices=[
                        "TinyLlama-1.1B-Chat-v1.0",
                        "Mixtral-8x7B-Instruct-v0.1",
                        "Mistral-7B-Instruct-v0.2",
                    ],
                    value="TinyLlama-1.1B-Chat-v1.0",
                    scale=1,
                    interactive=True,
                )
                system_prompt = gr.Textbox(
                    label="System prompt",
                    value="You are an assistant. Answer to a user's query based on a given context.",
                    scale=2,
                )
            with gr.Accordion(label="Knowledge base", open=True):
                url_input = gr.Textbox(placeholder="URL", value=None)
                gr.Markdown("OR")
                file_input = gr.File(
                    file_count="multiple",
                    file_types=[".txt", ".pdf"],
                    show_label=True,
                    visible=True,
                )
            submit = gr.Button("Submit")
            submit.click(
                fn=init_rag,
                inputs=[system_prompt, url_input, file_input],
            )
        with gr.Column():
            demo = gr.ChatInterface(fn=chat, examples=["Namaste!", "Hello!", "Hola!"])

if __name__ == "__main__":
    init_llm()
    iface.launch(debug=True)
