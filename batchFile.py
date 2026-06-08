import json
import os

from mistralai.client import Mistral
from sklearn.metrics import precision_score, recall_score, f1_score


def create_batch_file(df, prompt, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            request = {
                "custom_id": f"req_{idx}",
                "body": {
                    "model": "mistral-small-2603",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Antworte nur mit ja oder nein"
                        },
                        {
                            "role": "user",
                            "content": f"{prompt} {row['text']}"}
                    ],
                    "max_tokens": 5
                }
            }
            f.write(json.dumps(request) + "\n")
    return filename


def start_batch_file(filename):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    batch_file = client.files.upload(
        file={"file_name": filename, "content": open(filename, "rb")}, purpose='batch'
    )
    job = client.batch.jobs.create(
        input_files=[batch_file.id],
        model="mistral-small-2603",
        endpoint="/v1/chat/completions"
    )
    print(f"Batch-Job gestartet! ID: {job.id}")


def evaluate_batch_results(job_id, original_df):
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    job = client.batch.jobs.get(job_id=job_id)

    output_file_id = job.output_file

    response = client.files.download(file_id=output_file_id)

    raw_content = response.read().decode('utf-8')
    result_content = raw_content.strip().split('\n')

    predictions = [None] * len(original_df)

    for line in result_content:
        if not line.strip():
            continue

        data = json.loads(line)
        idx = int(data["custom_id"].split("_")[1])

        try:
            content = data["response"]["body"]["choices"][0]["message"]["content"].lower()
            predictions[idx] = 1 if "ja" in content else 0
        except (KeyError, TypeError):
            predictions[idx] = 0

    y_true = original_df['label_hs'].tolist()
    y_pred = predictions

    y_pred = [p if p is not None else 0 for p in y_pred]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- Batch Auswertung ({len(y_pred)} Texte) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    return precision, recall, f1
