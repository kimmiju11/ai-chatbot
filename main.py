from fastapi import FastAPI
import func

app = FastAPI()

@app.get("/chat")
def chat(question: str):
    qa = func.chainSet()
    # 체인 실행
    result = qa.invoke({
        "question": question
    })
    return result['answer']