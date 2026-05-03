import base64
import json
import os
import re
import requests


class ImageBenchmarkAgent:
    def __init__(self, api_model) -> None:
        self.api_url = api_model._APIModel__api_url
        self.api_key = api_model._APIModel__api_key
        self.model = "gpt-4o"
        self.records = []

    def _parse_json(self, text):
        if not text:
            return {}
        cleaned = text.strip()
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

    def evaluate_image(self, image_path, topic, section_name, caption, context_text):
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            prompt = f"""
You are benchmarking an academic survey figure.
Topic: {topic}
Section: {section_name}
Caption: {caption}
Context:
{context_text[:5000]}

Judge the image on:
1. relevance to the section
2. factual/content accuracy
3. text legibility
4. structural clarity
5. overall suitability for a survey figure

Return strict JSON only:
{{
  "pass": true,
  "scores": {{
    "relevance": 1,
    "accuracy": 1,
    "legibility": 1,
    "clarity": 1,
    "overall": 1
  }},
  "issues": ["..."],
  "suggestions": ["..."]
}}
"""
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 500,
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = self._parse_json(content)
            if not parsed:
                parsed = {
                    "pass": False,
                    "scores": {"relevance": 1, "accuracy": 1, "legibility": 1, "clarity": 1, "overall": 1},
                    "issues": ["Failed to parse benchmark output"],
                    "suggestions": [],
                    "raw": content,
                }

            record = {
                "image": image_path,
                "topic": topic,
                "section": section_name,
                "caption": caption,
                "result": parsed,
            }
            self.records.append(record)
            return record
        except Exception as e:
            record = {
                "image": image_path,
                "topic": topic,
                "section": section_name,
                "caption": caption,
                "result": {
                    "pass": False,
                    "scores": {"relevance": 0, "accuracy": 0, "legibility": 0, "clarity": 0, "overall": 0},
                    "issues": [str(e)],
                    "suggestions": [],
                },
            }
            self.records.append(record)
            return record

    def save(self, saving_path):
        if not self.records:
            return None
        os.makedirs(saving_path, exist_ok=True)
        json_path = os.path.join(saving_path, "image_benchmark.json")
        txt_path = os.path.join(saving_path, "image_benchmark.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)

        avg = {}
        keys = ["relevance", "accuracy", "legibility", "clarity", "overall"]
        for key in keys:
            vals = []
            for record in self.records:
                scores = record.get("result", {}).get("scores", {})
                if key in scores:
                    vals.append(scores[key])
            avg[key] = sum(vals) / len(vals) if vals else 0
        pass_rate = 0
        if self.records:
            pass_rate = sum(1 for record in self.records if record.get("result", {}).get("pass")) / len(self.records)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"avg_scores": avg, "pass_rate": pass_rate, "count": len(self.records)}, ensure_ascii=False, indent=4))
        return json_path
