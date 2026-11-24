from playwright.sync_api import sync_playwright
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall
from ragas.run_config import RunConfig
import json
import os

APP_URL = "https://llm-rag-testing-workshop.streamlit.app/Application"

# Load test cases
with open("data/spice_garden_test.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

evaluation_data = []

# Launch browser and submit prompts
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(APP_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(10000)
    app = page.frame_locator('iframe[title="streamlitApp"]')

    for case in test_cases:
        chat = app.get_by_test_id("stChatInputTextArea")
        chat.fill(case["user_input"])
        chat.press("Enter")

        page.wait_for_timeout(10000)

        messages = app.get_by_test_id("stChatMessage").all_inner_texts()
        response = messages[-1] if messages else ""

        if response.strip():
            evaluation_data.append({
                "user_input": case["user_input"],
                "response": response,
                "retrieved_contexts": case["retrieved_contexts"],
                "reference": case["reference"]
            })

    browser.close()

# Create RAGAs dataset
if not evaluation_data:
    raise ValueError("No responses captured. Evaluation cannot proceed.")

dataset = EvaluationDataset.from_list(evaluation_data)

# Configure Ollama judge
judge = ChatOpenAI(
    model="qwen2.5:3b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.3
)

# Evaluate RAG application
results = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), LLMContextRecall()],
    llm=LangchainLLMWrapper(judge),
    run_config=RunConfig(timeout=180, max_workers=1)
)

# Save results
os.makedirs("results", exist_ok=True)
results.to_pandas().to_csv("results/spice_garden_eval.csv", index=False)

print(results)
print("Evaluation completed!")