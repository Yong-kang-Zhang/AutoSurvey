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

    def evaluate_image(self, image_path, topic, section_name, caption, context_text, persist=True):
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

Scoring rules:
- Use a 0-100 scale for every score.
- 100 means excellent, publication-ready performance on that dimension.
- 80 means strong with minor weaknesses.
- 60 means acceptable but clearly flawed.
- 40 means weak.
- 20 means poor.
- 0 means unusable.
- Do not use 1-5 or 1-10 scales.

Return strict JSON only:
{{
  "scores": {{
    "relevance": 0,
    "accuracy": 0,
    "legibility": 0,
    "clarity": 0,
    "overall": 0
  }},
  "summary_score": 0,
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
                    "scores": {"relevance": 0, "accuracy": 0, "legibility": 0, "clarity": 0, "overall": 0},
                    "summary_score": 0,
                    "issues": ["Failed to parse benchmark output"],
                    "suggestions": [],
                    "raw": content,
                }
            parsed = self._normalize_result(parsed)

            record = {
                "image": image_path,
                "topic": topic,
                "section": section_name,
                "caption": caption,
                "result": parsed,
            }
            if persist:
                self.records.append(record)
            return record
        except Exception as e:
            result_record = {
                "image": image_path,
                "topic": topic,
                "section": section_name,
                "caption": caption,
                "result": {
                    "scores": {"relevance": 0, "accuracy": 0, "legibility": 0, "clarity": 0, "overall": 0},
                    "summary_score": 0,
                    "issues": [str(e)],
                    "suggestions": [],
                },
            }
            if persist:
                self.records.append(result_record)
            return result_record

    def _normalize_score(self, value):
        try:
            score = float(value)
        except Exception:
            return 0
        if score <= 1:
            score *= 100
        elif score <= 5:
            score *= 20
        elif score <= 10:
            score *= 10
        return max(0, min(100, round(score, 2)))

    def _normalize_result(self, parsed):
        scores = parsed.get("scores", {}) or {}
        normalized_scores = {}
        for key in ["relevance", "accuracy", "legibility", "clarity", "overall"]:
            normalized_scores[key] = self._normalize_score(scores.get(key, 0))
        parsed["scores"] = normalized_scores
        raw_summary = parsed.get("summary_score", None)
        normalized_summary = self._normalize_score(raw_summary) if raw_summary is not None else 0
        average_summary = round(sum(normalized_scores.values()) / len(normalized_scores), 2) if normalized_scores else 0

        if normalized_summary <= 0:
            parsed["summary_score"] = average_summary
        elif abs(normalized_summary - average_summary) > 25:
            parsed["summary_score"] = average_summary
        else:
            parsed["summary_score"] = normalized_summary
        return parsed

    def _deduplicate_records(self):
        latest_by_image = {}
        for record in self.records:
            image = record.get("image", "")
            if not image:
                continue
            latest_by_image[image] = record
        return list(latest_by_image.values())

    def save(self, saving_path):
        if not self.records:
            return None
        os.makedirs(saving_path, exist_ok=True)
        json_path = os.path.join(saving_path, "image_benchmark.json")
        txt_path = os.path.join(saving_path, "image_benchmark.txt")
        final_records = self._deduplicate_records()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_records, f, indent=4, ensure_ascii=False)

        avg = {}
        keys = ["relevance", "accuracy", "legibility", "clarity", "overall"]
        for key in keys:
            vals = []
            for record in final_records:
                scores = record.get("result", {}).get("scores", {})
                if key in scores:
                    vals.append(scores[key])
            avg[key] = sum(vals) / len(vals) if vals else 0
        summary_scores = [record.get("result", {}).get("summary_score", 0) for record in final_records]
        overall_summary = sum(summary_scores) / len(summary_scores) if summary_scores else 0

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"avg_scores": avg, "summary_score": overall_summary, "count": len(final_records)}, ensure_ascii=False, indent=4))
        return json_path
