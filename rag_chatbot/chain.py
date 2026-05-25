from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma

LLM_MODEL = "llama-3.3-70b-versatile"


def build_chain(vectorstore: Chroma):
    llm = ChatGroq(model=LLM_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the conversation history and a follow-up question, "
            "rewrite it as a standalone question. Return only the question."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer the user's question using only the context below. "
            "If the answer is not in the context, say so clearly.\n\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    condense_chain = condense_prompt | llm | StrOutputParser()
    qa_chain = qa_prompt | llm | StrOutputParser()

    def retrieve(inputs: dict) -> dict:
        question = inputs["input"]
        if inputs.get("chat_history"):
            question = condense_chain.invoke(inputs)
        docs = retriever.invoke(question)
        return {**inputs, "context": "\n\n".join(d.page_content for d in docs), "context_docs": docs}

    def generate(inputs: dict) -> dict:
        answer = qa_chain.invoke(inputs)
        return {"answer": answer, "context": inputs.get("context_docs", [])}

    return RunnableLambda(retrieve) | RunnableLambda(generate)


def messages_to_langchain(messages: list[dict]) -> list:
    result = []
    for msg in messages:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result
