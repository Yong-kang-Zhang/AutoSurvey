import os
import re
import requests
import base64
class IllustratorAgent:
    """
    专业的科研绘图智能体 (The Academic Illustrator Agent)
    目标：为综述正文补充高质量机制图。
    """
    def __init__(self, api_model, image_api_key: str, image_model: str = "gpt-image-2-all", image_api_url: str = ""):
        self.api_model = api_model
        self.image_api_key = image_api_key
        self.image_model = image_model
        self.api_url = api_model._APIModel__api_url 
        self.image_api_url = image_api_url.strip() if image_api_url else self._infer_image_api_url()

    def _infer_image_api_url(self):
        if "image" in self.image_model and self.api_url.endswith("/v1/chat/completions"):
            return self.api_url.replace("/v1/chat/completions", "/v1/images/generations")
        if self.image_model.startswith("gpt-image-"):
            return "https://api.openai.com/v1/images/generations"
        return self.api_url

    def _save_image_bytes(self, img_data, save_path):
        with open(save_path, "wb") as f:
            f.write(img_data)
        return True

    def call_nano_api(self, prompt, save_path):
        print(f"    > [High-Density Diagram Gen] Requesting image model for prompt: {prompt[:60]}...")
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.image_api_key}"
            }

            if self.image_model.startswith("gpt-image-"):
                payload = {
                    "model": self.image_model,
                    "prompt": prompt,
                    "size": "1024x1024",
                }
                response = requests.post(self.image_api_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                if data.get("data"):
                    item = data["data"][0]
                    if item.get("b64_json"):
                        img_data = base64.b64decode(item["b64_json"])
                        return self._save_image_bytes(img_data, save_path)
                    if item.get("url"):
                        img_res = requests.get(item["url"], timeout=30)
                        img_res.raise_for_status()
                        return self._save_image_bytes(img_res.content, save_path)
                print("    > [Error] No image payload found in OpenAI image response.")
                return False

            payload = {
                "model": self.image_model,
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(self.image_api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status() 
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            base64_match = re.search(r'base64,([A-Za-z0-9+/=]+)', content)
            if base64_match:
                img_data = base64.b64decode(base64_match.group(1))
                return self._save_image_bytes(img_data, save_path)
            elif len(content) > 1000 and "http" not in content:
                clean_b64 = re.sub(r'[^A-Za-z0-9+/=]', '', content) 
                img_data = base64.b64decode(clean_b64)
                return self._save_image_bytes(img_data, save_path)
            else:
                url_match = re.search(r'(https?://[^\s\)]+)', content)
                if url_match:
                    img_res = requests.get(url_match.group(1), timeout=30)
                    img_res.raise_for_status()
                    return self._save_image_bytes(img_res.content, save_path)
                else:
                    print("    > [Error] No Image Data/URL found in response.")
                    return False
        except Exception as e:
            print(f"    > [Error] Image API call failed: {e}")
            return False

    def enrich_sections_with_diagrams(self, topic, parsed_outline, section_contents, saving_path):
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
            
        print("\n[*] [Academic Art Director] Executing Ultra-High-Density Structural Generation...")
        prompts = []
        
        for i, section_name in enumerate(parsed_outline['sections']):
            txt = "\n".join(section_contents[i])[:15000]
            
            p = f"""
            Topic: {topic}
            Section: {section_name}

            [Task: ULTRA-HIGH-DENSITY ACADEMIC MECHANISM DIAGRAM]
            你现在是一名顶会（Nature/CVPR）级别的顶级学术绘图设计师。你需要阅读以下文献内容，深入理解核心机制、关键方法与数据流向，生成一张【宏观结构严密、微观干货满满、信息密度极高且画面极其干净】的 BioRender 风格机制图提示词。

            <Text>{txt}</Text>

            [DESIGN RULES - CRITICAL]

            ROLE: The Infographic Designer (Focus: Ultra-Dense BioRender-Style via Image Generator)
            - 1. 智能版式决策与视觉层级（决定构图与主次）：
              * 若偏向【核心算法/机制】：设计【连续的科学流水线 (Scientific Pipeline)】。像讲故事一样描述数据流转，并且**必须突出视觉中心**（例如：将核心算法模块放大并置于中心，预处理和输出模块置于两侧）。
              * 若偏向【引言/分类/挑战】：设计【高级分类导图 (Taxonomy) 或 层次系统图】。通过严格的对称性、主次节点大小的递减来展现严密的逻辑树。
            - 2. 宏观秩序与线面逻辑（排版框架）：使用带浅色背景的大容器框（Container Panels）进行分组。排版必须保持严谨的对齐感（Grid-aligned）。**新增限制：连接线必须具有语义区分（例如：实线代表数据流，虚线代表反向传播或逻辑关联），箭头必须犀利精准。**
            - 3. 微观密度与数据抽象（科学填充）：大框内部绝对不能是空洞的色块！更不允许有大块空白！必须用**具象的数据抽象形态**填满框架。例如：多维张量矩阵(Tensor matrices)、彩色特征图网格(Feature map grids)、3D点云/散点图、连续信号波形等。

            - 4. [CRITICAL SPELLING & FORMULA RULE - 绝对防乱码安全锁]: 
              * 严格提取 4-6 个（绝不能超过6个）最核心的极短缩写（如 MAML, CNN, GNN, Loss），强迫模型将其作为**加粗的主标题**渲染在核心节点上。
              * **【严禁复杂公式】**：绝对不能要求生图模型画类似 $\mathcal{{L}}$ 这样的 LaTeX 公式！如果要表现数学概念，只能要求画单纯的字母或英文单词（如 "the word 'Loss'", "the letter 'L'", "the word 'Theta'"）。
              * **严禁使用“假小字占位符(fake text/bullet lines)”！** 如果有空白，请用微型图表（Mini-graphs）或几何线框去填补，绝不准出现任何视觉乱码噪点。

            - 5. [ADVANCED VISUAL STYLE COMMAND - CRITICAL]: 提示词末尾必须一字不差地包含以下英文咒语，这是融合顶会最高审美与极高信息密度的终极指令：
              "Masterpiece, award-winning Nature/CVPR journal survey figure, ultimate high-information-density BioRender style. Professional visual hierarchy and strictly aligned macro-layout: central core mechanisms must be prominent. Group modules into large, elegant pastel container panels with crisp, ultra-thin deep navy borders. These panels MUST BE HIGHLY ENRICHED with intricate scientific abstractions: stacked tensor matrices, detailed feature map grids, 3D latent space projections, and distinct mathematical symbols. Connect modules with precise, varying orthogonal arrows (solid for data flow, dashed for logic). Sophisticated academic color palette (e.g., deep navy accents, muted teal, pale orange pastel fills). ZERO visual noise, absolutely NO messy fake tiny text, NO empty wasted blocks, NO cheap 3D bevels. 4K resolution, professional print quality, razor-sharp typography, ultra-clear and legible text. TYPOGRAPHY COMMAND: Every single letter and word must be flawless. NO typographical errors, NO alien text, NO complex math equations. Ensure perfect spelling of the short acronyms."
            
            [OUTPUT FORMAT - STRICTLY FOLLOW]
            [NANO_PROMPT]
            // STRICTLY 100% ENGLISH. Your Ultra-Dense BioRender-style prompt here. ABSOLUTELY NO CHINESE CHARACTERS in this block. Must use quotes like "exact text 'XXX'" for short words, and include the visual style command at the end.
            [/NANO_PROMPT]

            [CAPTION]
            Figure: Ultra-dense BioRender-style mechanism diagram illustrating core methodologies, packed with micro-structures and perfectly spelled labels.
            [/CAPTION]

            If the text is purely introductory and structurally cannot support a dense mechanism diagram, output [NO_DIAGRAM].
            """
            prompts.append(p)
        
        codes = self.api_model.batch_chat(prompts, temperature=0.3) 
        img_counter = 1 
        
        for i, c in enumerate(codes):
            if "[NO_DIAGRAM]" in c: continue
                
            nano_match = re.search(r'\[NANO_PROMPT\](.*?)\[/NANO_PROMPT\]', c, re.DOTALL | re.IGNORECASE)
            caption_match = re.search(r'\[CAPTION\](.*?)\[/CAPTION\]', c, re.DOTALL | re.IGNORECASE)
            
            if nano_match and caption_match:
                nano_prompt = nano_match.group(1).strip()
                caption = caption_match.group(1).strip()
                
                safe_topic = re.sub(r'[^a-zA-Z0-9]', '_', topic)
                img_fn = f"{safe_topic}_{img_counter}_mechanism.jpg"
                img_path = os.path.join(saving_path, img_fn)
                
                # 第一步：调用 Nano 生成图片
                success = self.call_nano_api(nano_prompt, img_path)
                
                md_injection = f"\n**{caption}**\n\n"
                if success:
                    md_injection += f"![Mechanism Diagram](./{img_fn})\n\n"

                section_contents[i][0] = md_injection + section_contents[i][0]
                img_counter += 1
                
        return section_contents
        
