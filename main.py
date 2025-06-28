from fastapi import FastAPI
from fastapi.responses import FileResponse
import func
from langsmith import traceable
from ragas_evaluation import run_ragas_evaluation
import os
import tempfile
from datetime import datetime

app = FastAPI()

@app.get("/chat")
@traceable(name="chat_endpoint")
def chat(question: str):
    """
    챗봇 질문 응답 API 엔드포인트
    
    Args:
        question: 사용자가 입력한 질문
    
    Returns:
        dict: 챗봇의 응답 메시지
    """
    qa = func.chainSet()
    # 체인 실행
    result = qa.invoke({
        "question": question
    })
    return {"message": result['answer']}

@app.post("/evaluate")
def evaluate_ragas():
    """
    RAGAS 평가를 실행하는 API 엔드포인트
    
    Returns:
        FileResponse: 평가 결과파일 다운로드
    """
    print("=== RAGAS 평가 API 호출 시작 ===")
    save_results = True  # 결과 저장 여부, 필요에 따라 변경 가능    
    try:
        # 임시 파일 경로 생성
        temp_dir = tempfile.gettempdir()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_evaluation_results_{timestamp}"
        
        # 확장자가 없으면 .xlsx 추가
        if not filename.endswith(('.xlsx', '.xls')):
            filename = filename + '.xlsx'
            
        temp_file_path = os.path.join(temp_dir, filename)
        print(f"임시 파일 경로: {temp_file_path}")
        
        print("RAGAS 평가 함수 호출 시작...")
        result = run_ragas_evaluation(output_path=temp_file_path, save_results=save_results )
        
        if result is None:
            print("평가 결과가 None입니다.")
            return {"status": "error", "message": "평가 실행 중 오류가 발생했습니다."}
        
        
        
        if save_results and os.path.exists(temp_file_path):
            print(f"파일 다운로드 응답 반환: {temp_file_path}")
            # 파일이 성공적으로 생성되었으면 파일 다운로드 응답 반환
            return FileResponse(
                path=temp_file_path,
                filename=filename,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            print("JSON 응답 반환")
            # 파일 저장하지 않거나 실패한 경우 JSON 응답 반환
            return {
                "status": "success",
                "message": "RAGAS 평가가 완료되었습니다.",
            }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"평가 중 상세 오류:\n{error_detail}")
        
        # LangSmith에서 오류를 제대로 추적할 수 있도록 예외를 다시 발생
        raise Exception(f"RAGAS 평가 실패: {str(e)}")
