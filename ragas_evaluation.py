import pandas as pd
from func import llmSet, embeddingSet, vectordbSet, chainSet
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
    faithfulness
)
from datasets import Dataset

def load_evaluation_data(excel_path):
    """
    Excel 파일에서 평가 데이터를 로드합니다.
    
    Args:
        excel_path (str): Excel 파일 경로
    
    Returns:
        pd.DataFrame: 평가 데이터
    """
    try:
        df = pd.read_excel(excel_path)
        print(f"평가 데이터 로드 완료: {len(df)}개 항목")
        print(f"컬럼: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Excel 파일 로드 중 오류 발생: {e}")
        return None
    
def generate_answers_with_context(questions, qa_chain, database):
    """
    질문에 대한 답변과 관련 컨텍스트를 생성합니다.
    
    Args:
        questions (list): 질문 리스트
        qa_chain: QA 체인
        database: Vector 데이터베이스
    
    Returns:
        tuple: (answers, contexts) 
    """
    answers = []
    contexts = []
    
    if not questions:
        print("질문 리스트가 비어있습니다.")
        return [], []
    
    print(f"총 {len(questions)}개 질문 처리 시작...")
    
    for i, question in enumerate(questions):
        try:
            print(f"질문 {i+1}/{len(questions)} 처리 중: {question[:50]}...")
            
            # QA 체인을 통해 답변 생성
            result = qa_chain.invoke({"question": question})
            answer = result.get('answer', '답변 생성 실패') if isinstance(result, dict) else str(result)
            
            # 관련 컨텍스트 검색
            try:
                retriever = database.as_retriever()
                relevant_docs = retriever.get_relevant_documents(question)
                context = [doc.page_content for doc in relevant_docs] if relevant_docs else ["컨텍스트 없음"]
            except Exception as ctx_error:
                print(f"컨텍스트 검색 오류: {ctx_error}")
                context = ["컨텍스트 검색 실패"]
            
            answers.append(str(answer))
            contexts.append(context)
            
            print(f"질문 {i+1} 처리 완료")
            
        except Exception as e:
            print(f"질문 {i+1} 처리 중 오류 발생: {e}")
            answers.append("답변 생성 실패")
            contexts.append(["컨텍스트 검색 실패"])
    
    print(f"답변 생성 완료: {len(answers)}개")
    
    # 길이 검증
    if len(answers) != len(questions) or len(contexts) != len(questions):
        print(f"경고: 데이터 길이 불일치 - questions: {len(questions)}, answers: {len(answers)}, contexts: {len(contexts)}")
    
    return answers, contexts

    
def prepare_ragas_dataset(df, qa_chain, database):
    """
    RAGAS 평가를 위한 데이터셋을 준비합니다.
    
    Args:
        df (pd.DataFrame): 평가 데이터
        qa_chain: QA 체인  
        database: Vector 데이터베이스
    
    Returns:
        Dataset: RAGAS 평가용 데이터셋
    """
    # Excel 파일의 컬럼 확인
    print(f"사용 가능한 컬럼: {df.columns.tolist()}")
    
    # 컬럼명 매핑
    question_col = 'questions'  # Excel 파일에서 확인된 컬럼명
    ground_truth_col = 'ground_truths'  # Excel 파일에서 확인된 컬럼명
    
    if question_col not in df.columns:
        print(f"'{question_col}' 컬럼을 찾을 수 없습니다. 첫 번째 컬럼을 사용합니다.")
        question_col = df.columns[0]
    
    if ground_truth_col not in df.columns:
        print(f"'{ground_truth_col}' 컬럼을 찾을 수 없습니다. 두 번째 컬럼을 사용합니다.")
        ground_truth_col = df.columns[1] if len(df.columns) > 1 else None
    
    questions = df[question_col].tolist()
    ground_truths = df[ground_truth_col].tolist() if ground_truth_col else [""] * len(questions)
    
    # 답변과 컨텍스트 생성
    print("답변 및 컨텍스트 생성 중...")
    answers, contexts = generate_answers_with_context(questions, qa_chain, database)
    
    # RAGAS 데이터셋 형식으로 변환
    ragas_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(ragas_data)
    return dataset

def run_ragas_evaluation_with_df(df, output_path, save_results=True):
    """
    DataFrame으로 RAGAS 평가를 실행합니다.
    
    Args:
        df (pd.DataFrame): 평가 데이터
        save_results (bool): 결과 저장 여부
        output_path (str): 결과 저장 경로 (save_results=True일 때 필수)
    
    Returns:
        dict: 평가 결과
    """
    print("RAGAS 평가 시작...")
    
    # 1. LLM, embedding, vector DB 설정
    print("모델 및 데이터베이스 설정 중...")
    try:
        llm = llmSet()
        database = vectordbSet()
        qa_chain = chainSet()
        embeddings = embeddingSet()
        print("모델 설정 완료")
    except Exception as e:
        print(f"모델 설정 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"모델 설정 실패: {e}")
    
    # 2. RAGAS 데이터셋 준비
    print("RAGAS 데이터셋 준비 중...")
    try:
        dataset = prepare_ragas_dataset(df, qa_chain, database)
        print("데이터셋 준비 완료")
    except Exception as e:
        print(f"데이터셋 준비 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"데이터셋 준비 실패: {e}")
    
    # 3. 평가 메트릭 설정
    metrics = [
        answer_relevancy,
        answer_correctness,
        context_recall,
        context_precision,
        faithfulness 
    ]
    
    # 4. 평가 실행
    print("RAGAS 평가 실행 중...")
    try:
        print(f"데이터셋 크기: {len(dataset)}")
        
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
        print("평가 완료!")
        print("=" * 50)
        print("평가 결과:")
        print(result)
        
        # 5. 결과 저장
        if save_results:
            try:
                # 결과를 DataFrame으로 변환하여 저장
                if hasattr(result, 'to_pandas'):
                    results_df = result.to_pandas()
                    # 확장자가 없으면 .xlsx 추가
                    if not output_path.endswith(('.xlsx', '.xls')):
                        output_path = output_path + '.xlsx'
                    
                    results_df.to_excel(output_path, index=True)
                    print(f"상세 결과가 '{output_path}'에 저장되었습니다.")
                else:
                    print(f"ERROR: 결과를 DataFrame으로 변환할 수 없습니다.")
                    raise Exception("결과 DataFrame 변환 실패")
            except Exception as e:
                print(f"결과 저장 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
        return result
        
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        print("오류 상세:", str(e))
        import traceback
        traceback.print_exc()
        raise Exception(f"RAGAS 평가 실행 실패: {e}")
    
def run_ragas_evaluation(output_path, save_results=True):
    """
    RAGAS 평가를 실행합니다.
    
    Args:
        output_path (str): 결과 저장 경로 (save_results=True일 때 필수)
        save_results (bool): 결과 저장 여부
    
    Returns:
        dict: 평가 결과
    """
    excel_path="doc/check_dataset.xlsx"
    # 1. 평가 데이터 로드
    df = load_evaluation_data(excel_path)
    if df is None:
        return None
    
    # 2. DataFrame으로 평가 실행
    return run_ragas_evaluation_with_df(df, output_path, save_results)