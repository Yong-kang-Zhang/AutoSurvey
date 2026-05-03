import time
import requests
import json
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class APIModel:

    def __init__(self, model, api_key, api_url) -> None:
        self.__api_key = api_key
        self.__api_url = api_url
        self.model = model
        
    def __req(self, text, temperature, max_try=5):
        # 构造 Header
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.__api_key}',
            'Content-Type': 'application/json'
        }
        
        # 构造 Body
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": text}
            ],
            "temperature": temperature
        }

        # 循环重试
        for attempt in range(max_try):
            try:
                # === 超时时间延长到 120秒 (写长文必备) ===
                response = requests.post(
                    self.__api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=120 
                )
                
                # 检查状态码
                if response.status_code == 200:
                    try:
                        res_json = response.json()
                        content = res_json['choices'][0]['message']['content']
                        if content:
                            return content
                        else:
                            print(f"    [API Warning] 返回内容为空")
                            return ""
                    except Exception as e:
                        print(f"    [API Error] JSON解析失败: {e}")
                        return None
                else:
                    # 👇 [修改点 1]：增强 429 错误处理逻辑，动态退避并跳过下方报错
                    if response.status_code == 429:
                        wait_time = 5 + attempt * 5
                        print(f"    [API Warning] 429 Too Many Requests. Waiting {wait_time}s to retry ({attempt+1}/{max_try})...")
                        time.sleep(wait_time)
                        continue  # 遇到 429 直接等待后重试，不执行后面的逻辑
                        
                    print(f"    [API Error] HTTP {response.status_code}: {response.text[:100]}")
                    # 401/404/403 错误直接退出，不重试
                    if response.status_code in [401, 403, 404]:
                        return None
            
            except requests.exceptions.Timeout:
                print(f"    [API Error] 请求超时 (120秒) - 第 {attempt+1}/{max_try} 次重试...")
            except requests.exceptions.ConnectionError:
                print(f"    [API Error] 连接中断 - 第 {attempt+1}/{max_try} 次重试...")
            except Exception as e:
                print(f"    [API Error] 未知错误: {e}")
            
            # === 失败后递增等待时间 (3s, 5s, 7s...) ===
            time.sleep(3 + attempt * 2)

        print("    [API Fail] 重试次数耗尽。")
        return None
    
    def chat(self, text, temperature=1):
        res = self.__req(text, temperature, max_try=5)
        if res is None:
            return "" # 返回空串防止崩溃
        return res

    def batch_chat(self, text_batch, temperature=0):
        results = [""] * len(text_batch)
        
        # 👇 [修改点 2]：将并发数降为 2，并加入错峰缓冲时间 (降低网络压力与风控概率)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_idx = {}
            for i, text in enumerate(text_batch):
                future_to_idx[executor.submit(self.chat, text, temperature)] = i
                time.sleep(1) # 强制加入1秒延迟，错峰提交请求，避免瞬间触发 429
                
            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(future_to_idx), total=len(text_batch), desc="Batch API Call"):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results[idx] = res
                except Exception as e:
                    print(f"    [Batch Error] 索引 {idx} 处理失败: {e}")
                    results[idx] = ""
                    
        return results