from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="""{utterance}""",
    input_variables=["utterance"],
)

llm = OllamaLLM(
    model="llama3.1:8b",
    temperature=0,
)

turn_chain = prompt | llm | StrOutputParser()

while True:
    utterance = input(">")
    if utterance == "quit":
        break

    response = turn_chain.invoke(
        {"utterance": utterance}
    )
    print(f"{response}")
