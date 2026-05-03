import os
import re
import requests
import base64
import shutil
import time


class IllustratorAgent:
    """
    专业的科研绘图智能体 (The Academic Illustrator Agent)
    目标：为综述正文补充高质量机制图。
    """
    def __init__(self, api_model, image_api_key: str, image_model: str = "gpt-image-2-all", image_api_url: str = "", benchmark_agent=None):
        self.api_model = api_model
        self.image_api_key = image_api_key
        self.image_model = image_model
        self.api_url = api_model._APIModel__api_url 
        self.image_api_url = image_api_url.strip() if image_api_url else self._infer_image_api_url()
        self.image_api_url_candidates = self._build_image_api_candidates(self.image_api_url)
        self.benchmark_agent = benchmark_agent
        self.image_timeout = 120
        self.max_image_retries = 3
        self.max_benchmark_repairs = 2
        self.min_benchmark_score = 86
        self.min_legibility_score = 82
        self.min_clarity_score = 82

    def select_best_subsection_index(self, section_name, subsection_titles, subsection_contents):
        if not subsection_titles:
            return 0
        candidate_blocks = []
        for idx, (title, content) in enumerate(zip(subsection_titles, subsection_contents), 1):
            candidate_blocks.append(
                f"[{idx}] {title}\n{content[:1800]}"
            )
        prompt = f"""
You are placing one academic survey figure into a section.
Section: {section_name}

Choose the single best subsection index for placing a figure.
Prefer subsections that:
1. explain mechanisms, pipelines, taxonomies, architectures, or evaluation setups
2. would genuinely benefit from a visual summary
3. are not purely introductory or purely future-work discussion

Return only the integer index.

Candidates:
---
{chr(10).join(candidate_blocks)}
---
"""
        response = self.api_model.chat(prompt, temperature=0)
        match = re.search(r'\d+', response)
        if match:
            chosen = max(1, min(len(subsection_titles), int(match.group(0))))
            return chosen - 1
        return 0

    def score_section_for_figure(self, section_name, subsection_titles, subsection_contents):
        text = " ".join([section_name] + list(subsection_titles) + [c[:1200] for c in subsection_contents]).lower()
        weights = {
            'taxonomy': 5,
            'framework': 4,
            'architecture': 4,
            'mechanism': 4,
            'pipeline': 4,
            'workflow': 4,
            'benchmark': 3,
            'evaluation': 3,
            'application': 2,
            'retrieval': 2,
            'reasoning': 2,
            'adaptation': 2,
            'attention': 2,
            'training': 2,
        }
        penalties = {'introduction': 4, 'future': 5, 'outlook': 5, 'conclusion': 5}
        score = 0
        for key, weight in weights.items():
            if key in text:
                score += weight
        lowered_name = section_name.lower()
        for key, weight in penalties.items():
            if key in lowered_name:
                score -= weight
        score += min(3, len(re.findall(r'\[[^\]]+\]', " ".join(subsection_contents))) // 12)
        if any(len(content.split()) > 140 for content in subsection_contents):
            score += 1
        return score

    def insert_block_after_nth_paragraph(self, text, block, paragraph_number=1):
        block = (block or '').strip()
        body = (text or '').strip()
        if not block:
            return body
        if not body:
            return block

        parts = [part.strip() for part in re.split(r'\n\s*\n', body) if part.strip()]
        if len(parts) <= 1:
            return body + "\n\n" + block
        insert_after = max(1, min(paragraph_number, len(parts)))
        rebuilt = []
        for idx, part in enumerate(parts, 1):
            rebuilt.append(part)
            if idx == insert_after:
                rebuilt.append(block)
        return "\n\n".join(rebuilt)

    def _normalize_size_hint(self, value):
        text = str(value or "").strip().lower()
        if text in {"wide", "landscape", "horizontal"}:
            return "wide"
        if text in {"portrait", "tall", "vertical"}:
            return "portrait"
        return "square"

    def infer_size_hint(self, subsection_title, subsection_text):
        text = f"{subsection_title}\n{subsection_text}".lower()
        if any(key in text for key in ["pipeline", "workflow", "architecture", "benchmark", "comparison", "mechanism"]):
            return "wide"
        if any(key in text for key in ["taxonomy", "challenge", "future", "applications", "dimensions"]):
            return "portrait"
        return "square"

    def _resolve_size_candidates(self, size_hint):
        hint = self._normalize_size_hint(size_hint)
        if hint == "wide":
            return ["1536x1024", "1024x1024"]
        if hint == "portrait":
            return ["1024x1536", "1024x1024"]
        return ["1024x1024"]

    def _build_benchmark_repair_prompt(self, base_prompt, benchmark_result):
        issues = benchmark_result.get("issues", []) or []
        suggestions = benchmark_result.get("suggestions", []) or []
        issue_text = "; ".join(issues[:4]) if issues else "low survey suitability"
        suggestion_text = "; ".join(suggestions[:4]) if suggestions else "increase legibility, tighten structure, and stay closer to the subsection semantics"
        return (
            base_prompt
            + "\n\nRepair instruction: the previous image underperformed on survey-figure benchmarking."
            + f"\nWeaknesses to fix: {issue_text}."
            + f"\nConcrete repair actions: {suggestion_text}."
            + "\nMandatory fixes: preserve only subsection-supported content, switch to a cleaner publication layout, enlarge every label, use 3-5 large panels instead of many micro-panels, reduce label count if needed, increase whitespace, strengthen visual grouping, and keep a publication-quality academic figure style."
        )

    def _benchmark_passes_threshold(self, record):
        result = record.get("result", {}) or {}
        scores = result.get("scores", {}) or {}
        return (
            result.get("summary_score", 0) >= self.min_benchmark_score
            and scores.get("legibility", 0) >= self.min_legibility_score
            and scores.get("clarity", 0) >= self.min_clarity_score
        )

    def _generate_benchmarked_image(self, prompt, save_path, size_hint, topic, section_name, caption, context_text):
        if not self.benchmark_agent:
            return self.call_nano_api(prompt, save_path, size_hint=size_hint), None

        base, ext = os.path.splitext(save_path)
        best_record = None
        best_path = None
        temp_paths = []
        current_prompt = prompt

        for repair_round in range(self.max_benchmark_repairs + 1):
            attempt_path = f"{base}_attempt{repair_round}{ext}"
            temp_paths.append(attempt_path)
            success = self.call_nano_api(current_prompt, attempt_path, size_hint=size_hint)
            if not success:
                continue

            record = self.benchmark_agent.evaluate_image(
                image_path=attempt_path,
                topic=topic,
                section_name=section_name,
                caption=caption,
                context_text=context_text,
                persist=False,
            )
            score = record.get("result", {}).get("summary_score", 0)
            if best_record is None or score > best_record.get("result", {}).get("summary_score", 0):
                best_record = record
                best_path = attempt_path

            if self._benchmark_passes_threshold(record):
                break

            if repair_round < self.max_benchmark_repairs:
                current_prompt = self._build_benchmark_repair_prompt(prompt, record.get("result", {}))

        if best_record and best_path and os.path.exists(best_path):
            shutil.copyfile(best_path, save_path)
            best_record["image"] = save_path
            self.benchmark_agent.records.append(best_record)
            for temp_path in temp_paths:
                if temp_path != best_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            if best_path != save_path and os.path.exists(best_path):
                try:
                    os.remove(best_path)
                except OSError:
                    pass
            return True, best_record

        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        return False, None

    def _infer_image_api_url(self):
        if "image" in self.image_model and self.api_url.endswith("/v1/chat/completions"):
            return self.api_url.replace("/v1/chat/completions", "/v1/images/generations")
        if self.image_model.startswith("gpt-image-"):
            return "https://api.openai.com/v1/images/generations"
        return self.api_url

    def _build_image_api_candidates(self, primary_url):
        candidates = []

        def add(url):
            if url and url not in candidates:
                candidates.append(url)

        add(primary_url)
        if self.api_url.endswith("/v1/chat/completions"):
            add(self.api_url.replace("/v1/chat/completions", "/v1/images/generations"))
        if self.image_model.startswith("gpt-image-"):
            add("https://yunwu.ai/v1/images/generations")
            add("https://api.openai.com/v1/images/generations")
        return candidates

    def _save_image_bytes(self, img_data, save_path):
        with open(save_path, "wb") as f:
            f.write(img_data)
        return True

    def _build_retry_prompt_variants(self, prompt):
        variants = [
            ("primary", prompt),
            (
                "fallback_simple",
                prompt +
                "\n\nFallback instruction: simplify the composition. Use 4-6 large modules only, fewer labels, cleaner spacing, and prioritize rendering reliability over extreme density. Keep the figure academic, accurate, and visually sharp.",
            ),
            (
                "fallback_minimal",
                prompt +
                "\n\nEmergency fallback instruction: generate a clean 2D academic mechanism figure with 3-5 core modules, very short labels, minimal decorative details, no tiny text, and a strong focus on successful rendering and correctness.",
            ),
        ]

        deduped = []
        seen = set()
        for name, text in variants:
            if text not in seen:
                deduped.append((name, text))
                seen.add(text)
        return deduped

    def _call_nano_api_once(self, prompt, save_path, size):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.image_api_key}"
            }

            if self.image_model.startswith("gpt-image-"):
                payload = {
                    "model": self.image_model,
                    "prompt": prompt,
                    "size": size,
                }
                last_error = None
                for endpoint in self.image_api_url_candidates:
                    try:
                        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.image_timeout)
                        response.raise_for_status()
                        data = response.json()
                        if data.get("data"):
                            item = data["data"][0]
                            if item.get("b64_json"):
                                img_data = base64.b64decode(item["b64_json"])
                                self.image_api_url = endpoint
                                return self._save_image_bytes(img_data, save_path), None
                            if item.get("url"):
                                img_res = requests.get(item["url"], timeout=30)
                                img_res.raise_for_status()
                                self.image_api_url = endpoint
                                return self._save_image_bytes(img_res.content, save_path), None
                        last_error = f"No image payload found in response from {endpoint}."
                    except Exception as endpoint_error:
                        last_error = f"{endpoint} :: {endpoint_error}"
                        continue
                return False, last_error or "No image payload found in image response."

            payload = {
                "model": self.image_model,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(self.image_api_url, headers=headers, json=payload, timeout=self.image_timeout)
            response.raise_for_status() 
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            base64_match = re.search(r'base64,([A-Za-z0-9+/=]+)', content)
            if base64_match:
                img_data = base64.b64decode(base64_match.group(1))
                return self._save_image_bytes(img_data, save_path), None
            elif len(content) > 1000 and "http" not in content:
                clean_b64 = re.sub(r'[^A-Za-z0-9+/=]', '', content) 
                img_data = base64.b64decode(clean_b64)
                return self._save_image_bytes(img_data, save_path), None
            else:
                url_match = re.search(r'(https?://[^\s\)]+)', content)
                if url_match:
                    img_res = requests.get(url_match.group(1), timeout=30)
                    img_res.raise_for_status()
                    return self._save_image_bytes(img_res.content, save_path), None
                else:
                    return False, "No image data or URL found in response."
        except Exception as e:
            return False, str(e)

    def call_nano_api(self, prompt, save_path, size_hint="square"):
        prompt_variants = self._build_retry_prompt_variants(prompt)
        size_candidates = self._resolve_size_candidates(size_hint)
        last_error = "unknown error"

        for size in size_candidates:
            for variant_idx, (variant_name, variant_prompt) in enumerate(prompt_variants, 1):
                for attempt in range(1, self.max_image_retries + 1):
                    print(
                        f"    > [High-Density Diagram Gen] {variant_name} {size} attempt {attempt}/{self.max_image_retries}: "
                        f"{variant_prompt[:60]}..."
                    )
                    success, error = self._call_nano_api_once(variant_prompt, save_path, size=size)
                    if success:
                        if variant_idx > 1 or attempt > 1 or size != size_candidates[0]:
                            print(f"    > [Recovery] Image generation succeeded via {variant_name} {size} attempt {attempt}.")
                        return True

                    last_error = error or "unknown error"
                    print(f"    > [Retry] {variant_name} {size} attempt {attempt} failed: {last_error}")
                    if attempt < self.max_image_retries:
                        time.sleep(min(10, 2 * attempt))

                if variant_idx < len(prompt_variants):
                    print(f"    > [Fallback] Switching to a simpler prompt after {variant_name} failed.")

        print(f"    > [Error] Image generation failed after retries and fallback prompts: {last_error}")
        return False

    def enrich_sections_with_diagrams(self, topic, parsed_outline, section_contents, saving_path):
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
            
        print("\n[*] [Academic Art Director] Executing Survey Figure Generation...")
        prompts = []
        prompt_meta = []
        max_figures = max(3, min(5, round(len(parsed_outline['sections']) * 0.5)))
        section_rankings = []

        for i, section_name in enumerate(parsed_outline['sections']):
            score = self.score_section_for_figure(
                section_name,
                parsed_outline['subsections'][i],
                section_contents[i],
            )
            section_rankings.append((score, i))

        candidate_indices = [
            idx for score, idx in sorted(section_rankings, reverse=True)
            if score > 0
        ][: min(len(parsed_outline['sections']), max_figures + 2)]
        
        for i in candidate_indices:
            section_name = parsed_outline['sections'][i]
            target_sub_idx = self.select_best_subsection_index(
                section_name,
                parsed_outline['subsections'][i],
                section_contents[i],
            )
            target_title = parsed_outline['subsections'][i][target_sub_idx] if parsed_outline['subsections'][i] else section_name
            target_text = section_contents[i][target_sub_idx][:8000] if section_contents[i] else ""
            section_context = "\n\n".join(section_contents[i])[:5000]
            
            p = f"""
            Topic: {topic}
            Section: {section_name}
            Best insertion subsection: {target_title}

            [Task: ULTRA-HIGH-DENSITY ACADEMIC SURVEY FIGURE]
            You are designing one figure for a top-tier academic survey. Read the target subsection carefully and generate a figure prompt that keeps an ultra-high-density BioRender-style visual language, while prioritizing factual correctness, readability, and direct usefulness for the chosen subsection.

            <TargetSubsectionTitle>{target_title}</TargetSubsectionTitle>
            <TargetSubsectionText>{target_text}</TargetSubsectionText>
            <BroaderSectionContext>{section_context}</BroaderSectionContext>

            [DESIGN RULES - CRITICAL]
            - 1. First decide the figure type that best fits the target subsection: mechanism pipeline, taxonomy, comparison map, architecture overview, or evaluation workflow.
            - 2. Also decide the best canvas orientation: wide landscape for pipelines/comparisons, portrait for taxonomies/challenge stacks, square only when structure is compact.
            - 3. Preserve a high-information-density BioRender-style look, but organize it with a strong visual hierarchy, obvious reading order, and balanced spacing.
            - 4. Prefer 3-5 large panels or 4-6 major modules with rich scientific detail, instead of a cluttered wall of tiny elements.
            - 5. Every text label must remain readable after insertion into a conference-style PDF page. Prefer fewer, larger labels over many tiny labels.
            - 6. Use short labels with 2-5 words, larger title banners, and thick arrows. Avoid micro-text, crowded legends, or dense unlabeled icon fields.
            - 7. Prioritize factual fidelity and text legibility over decorative density. Every visual element must be supported by the target subsection text.
            - 8. Use clean academic color grouping, crisp arrows, compact but readable labels, and subsection-faithful terminology.
            - 9. Include dense but meaningful substructures such as grouped panels, token flows, benchmark comparison callouts, or concept clusters only when they are truly relevant.
            - 10. Avoid speculative content, fake placeholder text, illegible tiny labels, visual noise, and complex equations.
            - 11. Use short labels with flawless spelling. No alien text. No fake bullet lines. No empty blocks.
            - 12. The figure should land naturally after the first or second paragraph of the target subsection, so the content should summarize what that subsection is about rather than the whole paper.
            - 13. End the prompt with this style command exactly:
              "Masterpiece, award-winning Nature/CVPR journal survey figure, ultra-high-information-density BioRender style, but still readable and faithful. Publication-quality, balanced layout, crisp arrows, rich scientific detail, accurate content, compact yet legible labels, professional conference-paper survey style."
            
            [OUTPUT FORMAT - STRICTLY FOLLOW]
            [IMAGE_SIZE]
            wide or portrait or square
            [/IMAGE_SIZE]

            [INSERT_AFTER_PARAGRAPH]
            1 or 2
            [/INSERT_AFTER_PARAGRAPH]

            [NANO_PROMPT]
            // STRICTLY 100% ENGLISH. Your ultra-dense survey-figure prompt here. ABSOLUTELY NO CHINESE CHARACTERS in this block. Must use quotes like "exact text 'XXX'" for short words, and include the visual style command at the end.
            [/NANO_PROMPT]

            [CAPTION]
            Figure: Ultra-dense BioRender-style survey figure summarizing the key concepts, mechanisms, or comparisons in this subsection.
            [/CAPTION]

            If the text is purely introductory and structurally cannot support a useful survey figure, output [NO_DIAGRAM].
            """
            prompts.append(p)
            prompt_meta.append((i, target_sub_idx, target_title, target_text, section_context))
        
        codes = self.api_model.batch_chat(prompts, temperature=0.3) 
        img_counter = 1 
        placed = 0
        
        for (i, target_sub_idx, target_title, target_text, section_context), c in zip(prompt_meta, codes):
            if placed >= max_figures:
                break
            if "[NO_DIAGRAM]" in c:
                continue
                
            size_match = re.search(r'\[IMAGE_SIZE\](.*?)\[/IMAGE_SIZE\]', c, re.DOTALL | re.IGNORECASE)
            paragraph_match = re.search(r'\[INSERT_AFTER_PARAGRAPH\](.*?)\[/INSERT_AFTER_PARAGRAPH\]', c, re.DOTALL | re.IGNORECASE)
            nano_match = re.search(r'\[NANO_PROMPT\](.*?)\[/NANO_PROMPT\]', c, re.DOTALL | re.IGNORECASE)
            caption_match = re.search(r'\[CAPTION\](.*?)\[/CAPTION\]', c, re.DOTALL | re.IGNORECASE)
            
            if nano_match and caption_match:
                nano_prompt = nano_match.group(1).strip()
                caption = caption_match.group(1).strip()
                size_hint = self.infer_size_hint(target_title, target_text)
                if size_match:
                    size_hint = self._normalize_size_hint(size_match.group(1))
                insert_after_paragraph = 1
                if paragraph_match:
                    match = re.search(r'\d+', paragraph_match.group(1))
                    if match:
                        insert_after_paragraph = max(1, min(2, int(match.group(0))))
                
                safe_topic = re.sub(r'[^a-zA-Z0-9]', '_', topic)
                img_fn = f"{safe_topic}_{img_counter}_mechanism.jpg"
                img_path = os.path.join(saving_path, img_fn)
                
                success, benchmark_record = self._generate_benchmarked_image(
                    nano_prompt,
                    img_path,
                    size_hint,
                    topic,
                    parsed_outline['sections'][i],
                    caption,
                    section_context,
                )

                if success:
                    md_injection = f"\n**{caption}**\n\n"
                    md_injection += f"![Mechanism Diagram](./{img_fn})\n\n"
                    target_idx = max(0, min(target_sub_idx, len(section_contents[i]) - 1))
                    section_contents[i][target_idx] = self.insert_block_after_nth_paragraph(
                        section_contents[i][target_idx],
                        md_injection,
                        paragraph_number=insert_after_paragraph,
                    )
                    placed += 1
                elif self.benchmark_agent:
                    self.benchmark_agent.records.append(
                        {
                            "image": img_path,
                            "topic": topic,
                            "section": parsed_outline['sections'][i],
                            "caption": caption,
                            "result": {
                                "scores": {"relevance": 0, "accuracy": 0, "legibility": 0, "clarity": 0, "overall": 0},
                                "summary_score": 0,
                                "issues": ["Image generation failed after retries and fallback prompts"],
                                "suggestions": ["Retry with a shorter prompt or a more stable image provider"],
                            },
                        }
                    )
                img_counter += 1

        if self.benchmark_agent:
            self.benchmark_agent.save(saving_path)

        return section_contents
        
