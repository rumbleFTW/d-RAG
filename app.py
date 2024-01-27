import gradio as gr

from llms.tiny_llama import TinyLlama
from knowledgebase import KnowledgeBase
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


LLM = None
KNOWLEDGEBASE = KnowledgeBase()

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
    LLM = TinyLlama()


def echo(message, history):
    return message


def load_knowledgebase(path):
    print("docu:", path)
    if not path:
        return
    if "https://" in path:
        KNOWLEDGEBASE.load_url(path)
        print("Succesfully loaded URL")
    else:
        if path.split(".")[-1] == "pdf":
            KNOWLEDGEBASE.load_pdf(path)
            print("Succesfully loaded PDF")
        else:
            KNOWLEDGEBASE.load_txt(path)
            print("Succesfully loaded TXT")


with gr.Blocks() as iface:
    with gr.Row(equal_height=True):
        with gr.Column():
            system_prompt = gr.Textbox(
                label="System prompt",
                placeholder="You are an assistant. Answer to a user's query based on a given context.",
            )
            with gr.Accordion(label="Knowledge base", open=True):
                url_input = gr.Textbox(
                    placeholder="URL",
                )
                gr.Markdown("OR")
                file_input = gr.File(
                    file_count="multiple",
                    file_types=[".txt", ".pdf"],
                    interactive=True,
                )
                file_input.upload(load_knowledgebase(path=file_input.value))
            submit = gr.Button("Submit")
        with gr.Column():
            demo = gr.ChatInterface(fn=echo, examples=["Namaste!", "Hello!", "Hola!"])

if __name__ == "__main__":
    # init_llm()
    iface.launch(debug=True)
