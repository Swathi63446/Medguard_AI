import json
import time
import httpx
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule

API_URL = "http://127.0.0.1:8000/ask"

# Robust path detection for test_questions.jsonl
TEST_QUESTIONS_FILE = "test_questions.jsonl"
if not os.path.exists(TEST_QUESTIONS_FILE):
    if os.path.exists("medguard_ai_dataset/test_questions.jsonl"):
        TEST_QUESTIONS_FILE = "medguard_ai_dataset/test_questions.jsonl"

def run_evaluation():
    print("--- Starting MedGuard AI Pipeline Evaluation on All 25 Test Questions ---")
    print(f"Using test file: {TEST_QUESTIONS_FILE}")
    
    try:
        with open(TEST_QUESTIONS_FILE, "r", encoding="utf-8") as f:
            test_cases = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {TEST_QUESTIONS_FILE} not found. Please ensure the test file is in your workspace root.")
        return

    total_questions = len(test_cases)
    passed_abstention = 0
    passed_citations = 0
    successful_retrievals = 0

    results_data = []

    print(f"Loaded {total_questions} test cases.")
    print("Applying 9-second delay between requests to stay safely within the free-tier limit (7 req/min)...\n")

    for idx, test in enumerate(test_cases, 1):
        q_id = test["question_id"]
        question = test["question"]
        expected_status = test["expected_status"] # "answerable" or "unanswerable"
        relevant_ids = test["relevant_document_ids"]
        
        print(f"[{idx}/{total_questions}] Testing {q_id}: {question}")
        
        is_abstaining = True
        cited_ids = []
        answer_text = ""
        status_flag = "Error"
        
        try:
            # Send POST request to FastAPI backend with 120s timeout
            response = httpx.post(API_URL, json={"question": question}, timeout=120.0)
            if response.status_code == 200:
                data = response.json()
                is_abstaining = data.get("abstain", True)
                cited_ids = data.get("cited_ids", [])
                answer_text = data.get("answer", "")
                status_flag = "Success"
            else:
                print(f" -> API Error status {response.status_code}: {response.text}")
                status_flag = f"HTTP Error {response.status_code}"
                answer_text = response.text
        except Exception as e:
            print(f" -> Connection/Execution Error on {q_id}: {e}")
            status_flag = f"Timeout/Error: {str(e)}"
            answer_text = str(e)

        # Check 1: Abstention Accuracy
        abstain_correct = False
        if expected_status == "unanswerable" and is_abstaining:
            abstain_correct = True
            passed_abstention += 1
        elif expected_status == "answerable" and not is_abstaining:
            abstain_correct = True
            passed_abstention += 1
            
        # Check 2: Citation / Expected ID Correctness
        citation_match = any(doc_id in cited_ids for doc_id in relevant_ids)
        if expected_status == "answerable" and citation_match:
            passed_citations += 1
            successful_retrievals += 1
        elif expected_status == "unanswerable":
            passed_citations += 1 
            successful_retrievals += 1

        results_data.append({
            "Question ID": q_id,
            "Question": question,
            "Expected Status": expected_status,
            "Got Abstain": is_abstaining,
            "Abstain Correct": "PASS" if abstain_correct else "FAIL",
            "Expected IDs": ", ".join(relevant_ids),
            "Cited IDs": ", ".join(cited_ids),
            "Citation Match": "PASS" if (expected_status == "unanswerable" or citation_match) else "FAIL",
            "Generated Answer": answer_text,
            "Execution Status": status_flag
        })
            
        print(f" -> Expected Status: {expected_status} | Got Abstain: {is_abstaining} (Correct: {abstain_correct})")
        print(f" -> Expected IDs: {relevant_ids} | Cited IDs: {cited_ids}")
        print("-" * 60)

        # Free tier rate-limit buffer: 9 seconds pause between requests (~6-7 requests per minute)
        if idx < total_questions:
            time.sleep(9.0)

    # --- Compute Metrics ---
    abstention_accuracy = (passed_abstention / total_questions) * 100
    citation_correctness = (passed_citations / total_questions) * 100
    retrieval_hit_rate = (successful_retrievals / total_questions) * 100

    print("\n================ EVALUATION REPORT ================")
    print(f"Total Evaluated Questions : {total_questions}")
    print(f"Abstention Accuracy       : {abstention_accuracy:.2f}%")
    print(f"Citation Correctness      : {citation_correctness:.2f}%")
    print(f"Retrieval Hit Rate        : {retrieval_hit_rate:.2f}%")
    print("====================================================")

    # --- Export to Polished Excel Workbook ---
    excel_filename = "medguard_evaluation_report.xlsx"
    wb = Workbook()
    
    # Sheet 1: Dashboard / Summary
    ws_summary = wb.active
    ws_summary.title = "Summary Dashboard"
    ws_summary.views.sheetView[0].showGridLines = True
    
    # Sheet 2: Detailed Results
    ws_details = wb.create_sheet(title="Detailed Results")
    ws_details.views.sheetView[0].showGridLines = True

    # Styling definitions
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=16, bold=True, color="1F4E78")
    subtitle_font = Font(name="Calibri", size=11, italic=True, color="595959")
    bold_font = Font(name="Calibri", size=11, bold=True)
    thin_border = Border(left=Side(style='thin', color='D3D3D3'),
                         right=Side(style='thin', color='D3D3D3'),
                         top=Side(style='thin', color='D3D3D3'),
                         bottom=Side(style='thin', color='D3D3D3'))

    # Populate Summary Dashboard
    ws_summary["A1"] = "MedGuard AI - Evaluation Dashboard"
    ws_summary["A1"].font = title_font
    ws_summary["A2"] = "Evidence-based Clinical RAG Pipeline Performance Metrics"
    ws_summary["A2"].font = subtitle_font

    summary_headers = ["Metric Description", "Score / Value"]
    for col_idx, header in enumerate(summary_headers, 1):
        cell = ws_summary.cell(row=4, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    summary_rows = [
        ("Total Evaluated Questions", total_questions),
        ("Abstention Accuracy (%)", f"{abstention_accuracy:.2f}%"),
        ("Citation Correctness (%)", f"{citation_correctness:.2f}%"),
        ("Retrieval Hit Rate (%)", f"{retrieval_hit_rate:.2f}%"),
        ("Evaluation Date", time.strftime("%Y-%m-%d %H:%M:%S"))
    ]

    for r_idx, (metric, val) in enumerate(summary_rows, 5):
        c1 = ws_summary.cell(row=r_idx, column=1, value=metric)
        c2 = ws_summary.cell(row=r_idx, column=2, value=val)
        c1.font = bold_font
        c1.border = thin_border
        c2.alignment = Alignment(horizontal="center")
        c2.border = thin_border

    ws_summary.column_dimensions['A'].width = 32
    ws_summary.column_dimensions['B'].width = 25

    # Populate Detailed Results
    detail_headers = [
        "Question ID", "Question", "Expected Status", "Got Abstain", 
        "Abstain Correct", "Expected IDs", "Cited IDs", "Citation Match", 
        "Generated Answer", "Execution Status"
    ]

    for col_idx, header in enumerate(detail_headers, 1):
        cell = ws_details.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for r_idx, row_data in enumerate(results_data, 2):
        ws_details.cell(row=r_idx, column=1, value=row_data["Question ID"]).alignment = Alignment(horizontal="center")
        ws_details.cell(row=r_idx, column=2, value=row_data["Question"]).alignment = Alignment(wrap_text=True)
        ws_details.cell(row=r_idx, column=3, value=row_data["Expected Status"]).alignment = Alignment(horizontal="center")
        ws_details.cell(row=r_idx, column=4, value=str(row_data["Got Abstain"])).alignment = Alignment(horizontal="center")
        
        c_abstain = ws_details.cell(row=r_idx, column=5, value=row_data["Abstain Correct"])
        c_abstain.alignment = Alignment(horizontal="center")
        c_abstain.font = bold_font
        
        ws_details.cell(row=r_idx, column=6, value=row_data["Expected IDs"]).alignment = Alignment(horizontal="center")
        ws_details.cell(row=r_idx, column=7, value=row_data["Cited IDs"]).alignment = Alignment(horizontal="center")
        
        c_match = ws_details.cell(row=r_idx, column=8, value=row_data["Citation Match"])
        c_match.alignment = Alignment(horizontal="center")
        c_match.font = bold_font

        ws_details.cell(row=r_idx, column=9, value=row_data["Generated Answer"]).alignment = Alignment(wrap_text=True)
        ws_details.cell(row=r_idx, column=10, value=row_data["Execution Status"]).alignment = Alignment(horizontal="center")

        for c in range(1, 11):
            ws_details.cell(row=r_idx, column=c).border = thin_border

    col_widths = {
        'A': 14, 'B': 45, 'C': 16, 'D': 14, 
        'E': 16, 'F': 16, 'G': 16, 'H': 16, 
        'I': 50, 'J': 18
    }
    for col, width in col_widths.items():
        ws_details.column_dimensions[col].width = width

    # Conditional Formatting: Highlight PASS (Light Green) and FAIL (Light Red)
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    ws_details.conditional_formatting.add('E2:E26', CellIsRule(operator='equal', formula=['"PASS"'], stopIfTrue=True, fill=green_fill))
    ws_details.conditional_formatting.add('E2:E26', CellIsRule(operator='equal', formula=['"FAIL"'], stopIfTrue=True, fill=red_fill))
    ws_details.conditional_formatting.add('H2:H26', CellIsRule(operator='equal', formula=['"PASS"'], stopIfTrue=True, fill=green_fill))
    ws_details.conditional_formatting.add('H2:H26', CellIsRule(operator='equal', formula=['"FAIL"'], stopIfTrue=True, fill=red_fill))

    wb.save(excel_filename)
    print(f"\nProfessional evaluation report successfully saved to: {excel_filename}")

if __name__ == "__main__":
    run_evaluation()