import chainlit as cl

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains import ConversationalRetrievalChain

from src import get_vector_db
from src import get_huggingface_llm


LLM = get_huggingface_llm()

welcome_message = """Welcome to the PDF QA! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]

    msg = cl.Message(
        content=f"Processing '{file.name}'...", disable_feedback=True
    )
    await msg.send()

    vector_db = await cl.make_async(get_vector_db)(file)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"k": 3}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"'{file.name}' processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    answer = answer[answer.rfind("Helpful Answer: ") + 16 :]
    # answer = res["answer"]
    # source_documents = res["source_documents"]
    # text_elements = []

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content,
    #                     name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNo sources found"

    await cl.Message(content=answer).send()
