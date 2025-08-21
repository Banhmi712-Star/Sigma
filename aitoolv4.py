import os
import time
import requests
import random
import json
import pyfiglet
from datetime import datetime
from colorama import Fore, Style, init
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import hashlib
import binascii
from urllib.parse import urlparse, parse_qs
import threading
from collections import Counter, defaultdict
import math
import google.generativeai as genai

# Thiết lập logging
logging.basicConfig(filename='lich_su_ai.log', level=logging.INFO, 
                    format='%(asctime)s | %(message)s', encoding='utf-8')

# Native entropy
def entropy_native(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p)) if len(p) > 0 else 0

# Native simple kmeans and vq (fix k dynamic)
def kmeans_native(obs, k, iter=20, thresh=1e-05):
    if len(obs) == 0:
        return np.array([]), np.array([])
    k = min(k, len(obs))
    centroids = obs[np.random.choice(len(obs), k, replace=False)]
    for _ in range(iter):
        dist = np.sum((obs[:, np.newaxis] - centroids[np.newaxis, :])**2, axis=2)
        codes = np.argmin(dist, axis=1)
        new_centroids = np.array([obs[codes == i].mean(axis=0) if np.sum(codes == i) > 0 else centroids[i] for _ in range(k)])
        if np.sum((new_centroids - centroids)**2) < thresh:
            break
        centroids = new_centroids
    return centroids, codes

def vq_native(obs, centroids):
    if len(centroids) == 0:
        return np.zeros(len(obs), dtype=int)
    dist = np.sum((obs[:, np.newaxis] - centroids[np.newaxis, :])**2, axis=2)
    codes = np.argmin(dist, axis=1)
    return codes

def derive_key(master_password):
    return hashlib.sha256(master_password.encode()).digest()

def encrypt_data(data, key):
    data_bytes = data.encode()
    key = (key * (len(data_bytes) // len(key) + 1))[:len(data_bytes)]
    encrypted = bytes(a ^ b for a, b in zip(data_bytes, key))
    return binascii.hexlify(encrypted).decode()

def decrypt_data(encrypted, key):
    encrypted_bytes = binascii.unhexlify(encrypted)
    key = (key * (len(encrypted_bytes) // len(key) + 1))[:len(encrypted_bytes)]
    decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key))
    return decrypted.decode()

# Khởi tạo colorama
init(autoreset=True)

def choose_key_smart_all_in_one(
    base_counts: dict,
    recent_records: list,
    *,
    prior: float = 1.0,           
    beta: float = 0.7,            
    decay_lambda: float = 0.25,   
    z: float = 1.96,              

    streak_window: int = 10,
    streak_penalty_coef: float = 0.2,   
    gap_bonus_coef: float = 0.12,       
    gap_bonus_log_base: float = 1.5,
    recent_power: float = 1.3,          
    ema_alpha: float = 0.25,            
    ema_penalty_coef: float = 0.15,
    entropy_boost_coef: float = 0.08,   
    jitter_std: float = 1e-4,           

    safe_boost_top: float = 0.03,
    safe_boost_decay: float = 0.015,

    safe_random_n: int = 4,       
    safe2_n: int = 4,             
    avoid_min: float = 0.80,
    avoid_max: float = 0.95,
    safe3_n: int = 4,             
    include_newest_in_random: bool = True,
    random_seed: int = 42,

    trust_state: dict | None = None,    
    trust_decay_when_picked: float = 0.03,  
    trust_recover_when_not: float = 0.03    
) -> str:
    if random_seed is not None:
        random.seed(random_seed)

    counts = {str(k): float(v) for k, v in base_counts.items()}
    keys = list(counts.keys())
    if not keys:
        raise ValueError("base_counts rỗng.")
    K = len(keys)

    recent_ids = [str(r["killed_room_id"]) for r in recent_records]
    M = len(recent_ids)

    w_count = Counter()
    if M > 0:
        weights = [math.exp(-decay_lambda * i) for i in range(M)]  
        w_sum = sum(weights)
        for i, rid in enumerate(recent_ids):
            w_count[rid] += weights[i]
        if w_sum > 0:
            for k in list(w_count.keys()):
                w_count[k] /= w_sum  

    comp_counts = {k: counts.get(k, 0.0) + beta * w_count.get(k, 0.0) for k in keys}

    alpha_post = {k: comp_counts[k] + prior for k in keys}
    S = sum(alpha_post.values())

    p_mean  = {k: alpha_post[k] / S for k in keys}
    p_var   = {k: (alpha_post[k] * (S - alpha_post[k])) / (S**2 * (S + 1.0)) for k in keys}
    p_sd    = {k: math.sqrt(max(v, 0.0)) for k, v in p_var.items()}
    p_upper = {k: p_mean[k] + z * p_sd[k] for k in keys}  

    streak_penalty = {k: 0.0 for k in keys}
    if M > 0 and streak_window > 0:
        window_ids = recent_ids[:streak_window]
        if window_ids:
            head = window_ids[0]
            streak_len = 1
            for x in window_ids[1:]:
                if x == head: streak_len += 1
                else: break
            if head in streak_penalty:
                streak_penalty[head] += streak_penalty_coef * streak_len

    last_seen_idx = {k: None for k in keys}
    for idx, rid in enumerate(recent_ids):
        if last_seen_idx.get(rid) is None:
            last_seen_idx[rid] = idx
    gap_bonus = {k: 0.0 for k in keys}
    for k in keys:
        if last_seen_idx[k] is None:
            gap_bonus[k] = gap_bonus_coef * math.log(1 + (streak_window + 1), gap_bonus_log_base)
        else:
            gap = last_seen_idx[k]
            gap_bonus[k] = gap_bonus_coef * math.log(1 + max(0, gap), gap_bonus_log_base)

    recent_ratio = {k: w_count.get(k, 0.0) for k in keys}
    recency_penalty = {k: (recent_ratio[k] ** recent_power) for k in keys}

    ema_hot = defaultdict(float)
    if M > 0:
        for rid in reversed(recent_ids):  
            for k in keys:
                ema_hot[k] = (1 - ema_alpha) * ema_hot[k] + (1.0 if k == rid else 0.0)
    ema_penalty = {k: ema_penalty_coef * ema_hot[k] for k in keys}

    avg_p = 1.0 / K
    p_boost = {k: p_mean[k] + entropy_boost_coef * (avg_p - p_mean[k]) for k in keys}
    s_boost = sum(p_boost.values())
    if s_boost > 0:
        for k in keys:
            p_boost[k] /= s_boost
    else:
        p_boost = p_mean.copy()

    score = {}
    for k in keys:
        val = (
            p_upper[k]
            + streak_penalty[k]
            + recency_penalty[k]
            + ema_penalty[k]
            - gap_bonus[k]
            - 0.05 * p_boost[k]
        )
        if jitter_std > 0:
            val += random.gauss(0.0, jitter_std)
        score[k] = val

    def tie_key_tmp(k):
        return (score[k], p_upper[k], p_mean[k], int(k) if k.isdigit() else k)
    ranking_tmp = sorted(keys, key=tie_key_tmp)

    safe_size = (K + 1) // 2  
    safe_list = ranking_tmp[:safe_size]

    if M > 0:
        newest = recent_ids[0]
        if newest in safe_list:
            safe_list.remove(newest)
        safe_list.insert(0, newest)
        safe_list = safe_list[:safe_size]

    for idx, k in enumerate(safe_list):
        bonus = max(0.0, safe_boost_top - idx * safe_boost_decay)
        score[k] -= bonus

    pool = safe_list.copy()
    if not include_newest_in_random and M > 0:
        newest = recent_ids[0]
        pool = [k for k in pool if k != newest] or pool

    n4 = min(safe_random_n, len(pool))
    _safe4 = random.sample(pool, n4) if n4 > 0 else []

    n2 = min(safe2_n, len(safe_list))
    _safe2 = random.sample(safe_list, n2) if n2 > 0 else []

    avoid = {k: 1.0 - p_upper[k] for k in keys}
    cand = [k for k in keys if avoid_min <= avoid[k] <= avoid_max]
    if len(cand) < safe3_n:
        filler = sorted([k for k in keys if k not in cand], key=lambda x: avoid[x], reverse=True)
        cand.extend(filler[: safe3_n - len(cand)])
    cand = list(dict.fromkeys(cand))  
    n3 = min(safe3_n, len(cand))
    _safe3 = random.sample(cand, n3) if n3 > 0 else []

    if trust_state is None:
        trust_state = {}
    for k in keys:
        trust_state.setdefault(k, 1.0)  

    for k in keys:
        t = max(0.01, float(trust_state.get(k, 1.0)))  
        score[k] *= (1.0 / t)

    def tie_key_final(k):
        return (score[k], p_upper[k], p_mean[k], int(k) if k.isdigit() else k)
    best_key = min(keys, key=tie_key_final)

    trust_state[best_key] = max(0.0, trust_state.get(best_key, 1.0) * (1.0 - trust_decay_when_picked))
    for k in keys:
        if k == best_key:
            continue
        trust_state[k] = min(1.0, trust_state.get(k, 1.0) + trust_recover_when_not)

    return best_key

CONFIG_FILE = "config.json"

class GameBot:
    def __init__(self):
        self.tool_running = True
        self.cuoc_ban_dau = 0
        self.amount = 0
        self.he_so_gap = 0
        self.so_du_ban_dau = 0
        self.tong_loi_lo = 0.0
        self.cuoc_dang_cho = None
        self.lich_su_ket_qua = []
        self.model_ensemble = None
        self.model_lstm = None
        self.optimizer_ensemble = None
        self.optimizer_lstm = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(Fore.YELLOW + f"Device: {self.device}")
        self.MODEL_ENSEMBLE_PATH = "ai_ensemble.pth"
        self.MODEL_LSTM_PATH = "ai_lstm.pth"
        self.data_history = []  
        self.cluster_centroids = None
        self.rf_model_path = "rf_model.pth"
        self.calibrated_rf = None
        self.api_cache = {}  
        self.cache_expire = 60  
        self.current_time = int(time.time() * 1000)
        self.api_login = f"https://user.3games.io/user/regist?is_cwallet=1&is_mission_setting=true&version=&time={self.current_time}"
        self.room_mapping = {
            1: "Nhà Kho",
            2: "Phòng Họp",
            3: "Phòng Giám Đốc",
            4: "Phòng Trò Chuyện",
            5: "Phòng Giám Sát",
            6: "Văn Phòng",
            7: "Phòng Tài Vụ",
            8: "Phòng Nhân Sự"
        }
        self.room_mapping_name2id = {v: k for k, v in self.room_mapping.items()}
        self.headers = {}
        self.fixed_headers = {
            "user-id": "2372036",
            "user-secret-key": "c9662eb28bc74cac95ad0ea91a79f1ef5d91774cc1f56a149bebe055771559f4",
            "user-login": "login_v2"
        }
        self.history_lock = threading.Lock()
        self.bet_type = "BUILD"
        self.bet_type_key = {"BUILD": "ctoken_contribute", "USDT": "ctoken_kusdt", "WORLD": "ctoken_kther"}
        self.api_history = f"https://xworld.info/vi-VN/battle/record?asset={self.bet_type}"
        self.api_cuoc = "https://api.escapemaster.net/escape_game/bet"
        self.api_10_van = f"https://api.escapemaster.net/escape_game/recent_10_issues?asset={self.bet_type}"
        self.api_100_van = f"https://api.escapemaster.net/escape_game/recent_100_issues?asset={self.bet_type}"
        self.api_my_joined = f"https://api.escapemaster.net/escape_game/my_joined?asset={self.bet_type}&page=1&page_size=10"
        self.thong_ke_thuat_toan = {}  
        self.lich_su_thang_thua = []  
        self.chuoi_thang_hien_tai = 0  
        self.chuoi_thang_max = 0  
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBC9VdGtwSlJ7BQr6Q1b3OCt4j4AFUJDnM")
        try:
            genai.configure(api_key=self.GEMINI_API_KEY)
        except:
            self.GEMINI_API_KEY = None

    def load_or_create_config(self):
        if os.path.exists(CONFIG_FILE):
            choice = input("\n 🔎 Đã lưu config, bạn có muốn dùng lại không? (y/n): ").strip().lower()
            if choice in ["y","yes",""]:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.bet_type = config["bet_type"]
                self.cuoc_ban_dau = config["cuoc_ban_dau"]
                self.he_so_gap = config["he_so_gap"]
                return config
            if choice in ["n","no"]:
                print("♻️ Nhập lại config mới:")
        else:
            print("⚠️ Chưa có config, hãy nhập mới:")

        self.bet_type = input("Nhập Loại Tiền cược BUILD/USDT/WORLD: ").strip().upper()
        self.cuoc_ban_dau = float(input(Fore.YELLOW + "Nhập số BUILD cược ban đầu ( >0 ): "))
        self.he_so_gap = int(input(Fore.YELLOW + "Nhập hệ số gấp khi thua (VD: 10, >1): "))
        config = {
            "bet_type": self.bet_type,
            "cuoc_ban_dau": self.cuoc_ban_dau,
            "he_so_gap": self.he_so_gap,
        }

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(Fore.GREEN + f"✅ Đã lưu config vào {CONFIG_FILE}")
        return config

    def history_collector(self):
        while self.tool_running:
            try:
                resp = requests.get(self.api_history, headers=self.fixed_headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == 0:
                        issues = data.get("data", [])
                        with self.history_lock:
                            self.data_history = [{"issue_id": i["issue_id"], "killed_room_id": i["killed_room_id"]} for i in issues[:100]]
                        print(Fore.GREEN + "✅ Updated history with 100 latest matches.")
            except Exception as e:
                logging.error(f"History collector error: {e}")
            time.sleep(30)

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def banner(self):
        self.clear_screen()
        print(Fore.LIGHTRED_EX + pyfiglet.figlet_format("TOOLVTH"))
        print(Fore.CYAN + "═" * 60)
        print(Fore.YELLOW + "🔥 AI dự đoán phòng VUA THOÁT HIỂM - Bản Giao Diện Đẹp Nhất 🔥")
        print(Fore.CYAN + "═" * 60 + "\n")

    def ghi_log_ai(self, room_name, ket_qua, so_tien, do_tin_cay, logic, reasoning=""):
        global thong_ke_thuat_toan
        if logic not in self.thong_ke_thuat_toan:
            self.thong_ke_thuat_toan[logic] = {"thang": 0, "thua": 0, "tong": 0}
        self.thong_ke_thuat_toan[logic]["tong"] += 1
        if ket_qua == "Thắng":
            self.thong_ke_thuat_toan[logic]["thang"] += 1
        else:
            self.thong_ke_thuat_toan[logic]["thua"] += 1
        logging.info(f"{logic} | {room_name} | {so_tien} | {ket_qua} | {do_tin_cay:.2f}% | Reasoning: {reasoning}")

    def luu_thong_ke_thuat_toan(self):
        try:
            with open("thong_ke_thuat_toan.json", "w", encoding="utf-8") as f:
                json.dump(self.thong_ke_thuat_toan, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(Fore.RED + f"Lỗi lưu thống kê: {e}")

    def tai_thong_ke_thuat_toan(self):
        try:
            with open("thong_ke_thuat_toan.json", "r", encoding="utf-8") as f:
                self.thong_ke_thuat_toan = json.load(f)
        except FileNotFoundError:
            self.thong_ke_thuat_toan = {}
        except Exception as e:
            print(Fore.RED + f"Lỗi tải thống kê: {e}")
            self.thong_ke_thuat_toan = {}

    def hien_thi_thong_ke_thuat_toan(self):
        if not self.thong_ke_thuat_toan:
            return
        
        print(Fore.CYAN + "\n" + "═" * 80)
        print(Fore.YELLOW + "📊 THỐNG KÊ HIỆU SUẤT CÁC THUẬT TOÁN AI")
        print(Fore.CYAN + "═" * 80)
        print(Fore.CYAN + "╔════════════════════════════════╤═════════╤═════════╤══════════╤══════════╗")
        print(Fore.CYAN + "║ Thuật toán                    │ Thắng   │ Thua    │ Tổng     │ Tỉ lệ %  ║")
        print(Fore.CYAN + "╠════════════════════════════════╪═════════╪═════════╪══════════╪══════════╣")
        
        sorted_algorithms = sorted(self.thong_ke_thuat_toan.items(), 
                                 key=lambda x: x[1]["thang"]/x[1]["tong"] if x[1]["tong"] > 0 else 0, 
                                 reverse=True)
        
        for logic, stats in sorted_algorithms:
            tong = stats["tong"]
            thang = stats["thang"]
            thua = stats["thua"]
            ti_le = (thang / tong * 100) if tong > 0 else 0
            
            if ti_le >= 70:
                color = Fore.GREEN
            elif ti_le >= 50:
                color = Fore.YELLOW
            else:
                color = Fore.RED
                
            print(f"║ {logic:<30}│ {thang:<7}│ {thua:<7}│ {tong:<8}│ {color}{ti_le:<8.1f}%{Fore.CYAN}║")
        
        print(Fore.CYAN + "╚════════════════════════════════╧═════════╧═════════╧══════════╧══════════╝")
        
        if sorted_algorithms:
            best_logic, best_stats = sorted_algorithms[0]
            best_ti_le = (best_stats["thang"] / best_stats["tong"] * 100) if best_stats["tong"] > 0 else 0
            print(Fore.GREEN + f"\n🏆 Thuật toán tốt nhất: {best_logic} ({best_ti_le:.1f}%)")
            
            self.luu_thong_ke_thuat_toan()

    def hien_thi_thong_ke_chinh_xac(self):
        if not self.lich_su_thang_thua:
            return
        
        so_du_hien_tai = self.lay_so_du_hien_tai()
        
        tong_tran = len(self.lich_su_thang_thua)
        so_tran_thang = self.lich_su_thang_thua.count("Thắng")
        so_tran_thua = self.lich_su_thang_thua.count("Thua")
        loi_hien_tai = so_du_hien_tai - self.so_du_ban_dau
        
        print(Fore.CYAN + "\n" + "═" * 80)
        print(Fore.YELLOW + "📊 THỐNG KÊ CHI TIẾT")
        print(Fore.CYAN + "═" * 80)
        
        print(Fore.LIGHTGREEN_EX + f"💰 Số dư ban đầu: {self.so_du_ban_dau:.2f} {self.bet_type}")
        print(Fore.LIGHTGREEN_EX + f"💰 Số dư hiện tại: {so_du_hien_tai:.2f} {self.bet_type}")
        print(Fore.LIGHTGREEN_EX + f"💰 Lời/Lỗ hiện tại: {loi_hien_tai:.2f} {self.bet_type}")
        
        print(Fore.LIGHTBLUE_EX + f"\n🎯 Thống kê trận đấu:")
        print(Fore.LIGHTBLUE_EX + f"   Tổng trận: {tong_tran}")
        print(Fore.GREEN + f"   Thắng: {so_tran_thang}")
        print(Fore.RED + f"   Thua: {so_tran_thua}")
        print(Fore.YELLOW + f"   Tỉ lệ thắng: {(so_tran_thang/tong_tran*100):.1f}%")
        
        print(Fore.LIGHTMAGENTA_EX + f"\n🔥 Chuỗi thắng:")
        print(Fore.LIGHTMAGENTA_EX + f"   Hiện tại: {self.chuoi_thang_hien_tai}")
        print(Fore.LIGHTMAGENTA_EX + f"   Tối đa: {self.chuoi_thang_max}")
        
        print(Fore.CYAN + "═" * 80)

    def hien_thi_thong_ke_nhanh(self):
        if not self.lich_su_thang_thua:
            return
        
        so_du_hien_tai = self.lay_so_du_hien_tai()
        
        tong_tran = len(self.lich_su_thang_thua)
        so_tran_thang = self.lich_su_thang_thua.count("Thắng")
        loi_hien_tai = so_du_hien_tai - self.so_du_ban_dau
        
        print(Fore.LIGHTGREEN_EX + f"💰 {self.bet_type}: {so_du_hien_tai:.2f}")
        print(Fore.LIGHTBLUE_EX + f"📊 Thắng: {so_tran_thang}/{tong_tran} | Chuỗi: {self.chuoi_thang_hien_tai}(max:{self.chuoi_thang_max}) | Lời: {loi_hien_tai:.2f} {self.bet_type}")

    def get_api_data(self, url):
        now = time.time()
        if url in self.api_cache and now - self.api_cache[url]['time'] < self.cache_expire:
            return self.api_cache[url]['data']
        for attempt in range(3):  
            try:
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == 0:
                        self.api_cache[url] = {'data': data, 'time': now}
                        return data
            except Exception as e:
                logging.error(f"Lỗi API {url} attempt {attempt+1}: {e}")
                time.sleep(1)  
        print(Fore.RED + f"Lỗi API sau retry: {url}")
        return None

    def get_my_joined(self):
        data = self.get_api_data(self.api_my_joined)
        if data:
            return data.get("data", {}).get("items", [])
        return []

    def thong_ke_xu_huong(self, so_van):
        with self.history_lock:
            issues = self.data_history[:so_van]
        if issues:
            thong_ke = {}
            for issue in issues:
                room_id = issue.get("killed_room_id")
                room_name = self.room_mapping.get(room_id, "Không xác định")
                thong_ke[room_name] = thong_ke.get(room_name, 0) + 1
            return thong_ke
        return {}

    def thong_ke_xu_huong_va_do_tin_cay(self, so_van):
        thong_ke = self.thong_ke_xu_huong(so_van)
        if thong_ke:
            phong_max = max(thong_ke, key=thong_ke.get)
            tile_max = thong_ke[phong_max] / so_van * 100

            print(Fore.MAGENTA + f"\n📊 Thống kê {so_van} ván gần nhất:")
            print(Fore.CYAN + "╔════════════════╤═════════╤══════════╗")
            print(Fore.CYAN + "║ Phòng          │ Số lần  │ % Tỉ lệ  ║")
            print(Fore.CYAN + "╠════════════════╪═════════╪══════════╣")
            for room, count in sorted(thong_ke.items(), key=lambda x: -x[1]):
                tile = (count / so_van) * 100
                mark = Fore.GREEN + " ← Ưu tiên" if room == phong_max else ""
                print(f"║ {room:<14}│ {count:<7}│ {tile:<8.2f}║{mark}")
            print(Fore.CYAN + "╚════════════════╧═════════╧══════════╝")
            return phong_max, tile_max
        return None, 0

    def phong_xuat_hien_nhieu_nhat(self, so_van):
        thong_ke = self.thong_ke_xu_huong(so_van)
        if thong_ke:
            return max(thong_ke, key=thong_ke.get)
        return None

    def phong_it_nhat(self, so_van):
        thong_ke = self.thong_ke_xu_huong(so_van)
        if thong_ke:
            return min(thong_ke, key=thong_ke.get)
        return None

    def phong_trung_binh(self, so_van):
        thong_ke = self.thong_ke_xu_huong(so_van)
        if thong_ke:
            values = list(thong_ke.values())
            avg = sum(values) / len(values)
            closest = min(values, key=lambda x: abs(x - avg))
            for room, count in thong_ke.items():
                if count == closest:
                    return room
        return None

    def phong_xu_huong_tang(self, so_van):
        with self.history_lock:
            issues = self.data_history[:so_van]
        if issues:
            nua_cu = issues[:so_van//2]
            nua_moi = issues[so_van//2:]
            thong_ke_cu = {}
            thong_ke_moi = {}
            for issue in nua_cu:
                room_name = self.room_mapping.get(issue["killed_room_id"], "Không xác định")
                thong_ke_cu[room_name] = thong_ke_cu.get(room_name, 0) + 1
            for issue in nua_moi:
                room_name = self.room_mapping.get(issue["killed_room_id"], "Không xác định")
                thong_ke_moi[room_name] = thong_ke_moi.get(room_name, 0) + 1

            tang_truong = {}
            for room in self.room_mapping.values():
                cu = thong_ke_cu.get(room, 0)
                moi = thong_ke_moi.get(room, 0)
                if cu > 0:
                    tang_truong[room] = ((moi - cu) / cu) * 100
                else:
                    tang_truong[room] = 100 if moi > 0 else 0

            return max(tang_truong, key=tang_truong.get)
        return None

    def phong_khac_voi_van_truoc(self):
        if self.lich_su_ket_qua:
            return random.choice([r for r in self.room_mapping.values() if r != self.lich_su_ket_qua[-1]])
        return random.choice(list(self.room_mapping.values()))

    def phong_theo_quy_luat_fibonacci(self):
        fib = [1, 1, 2, 3, 5, 8, 13, 21]
        index = fib[len(self.lich_su_ket_qua) % len(fib)] % 8
        return list(self.room_mapping.values())[index]

    def phong_theo_thoi_gian(self):
        gio = datetime.now().hour
        return list(self.room_mapping.values())[gio % 8]

    def phong_theo_so_du(self):
        try:
            response = requests.get(self.api_login, headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 200:
                    so_du = round(data["data"]["cwallet"][self.bet_type_key[self.bet_type]])
                    return list(self.room_mapping.values())[so_du % 8]
        except Exception as e:
            logging.error(f"Lỗi lấy số dư: {e}")
        return random.choice(list(self.room_mapping.values()))

    def phong_ngau_nhien_co_trong_so(self):
        trong_so = {room: random.randint(1, 100) for room in self.room_mapping.values()}
        return max(trong_so, key=trong_so.get)

    def phong_theo_mau(self):
        mau_sac = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
        index = len(self.lich_su_ket_qua) % len(mau_sac)
        return list(self.room_mapping.values())[index]

    def phong_markov_chain(self):
        if len(self.lich_su_ket_qua) < 3:
            return random.choice(list(self.room_mapping.values()))
        
        transition_matrix = {}
        for i in range(len(self.lich_su_ket_qua) - 2):
            state = (self.lich_su_ket_qua[i], self.lich_su_ket_qua[i+1])
            next_state = self.lich_su_ket_qua[i+2]
            
            if state not in transition_matrix:
                transition_matrix[state] = {}
            transition_matrix[state][next_state] = transition_matrix[state].get(next_state, 0) + 1
        
        current_state = (self.lich_su_ket_qua[-2], self.lich_su_ket_qua[-1])
        if current_state in transition_matrix:
            next_rooms = transition_matrix[current_state]
            if next_rooms:
                return max(next_rooms, key=next_rooms.get)
        
        return random.choice(list(self.room_mapping.values()))

    def phong_ml_linear_regression(self):
        if len(self.lich_su_ket_qua) < 10:
            return random.choice(list(self.room_mapping.values()))
        
        room_to_num = {room: i for i, room in enumerate(self.room_mapping.values())}
        num_to_room = {i: room for room, i in room_to_num.items()}
        
        X = []
        y = []
        
        for i, room in enumerate(self.lich_su_ket_qua):
            X.append([i, i**2, i**3])  
            y.append(room_to_num[room])
        
        n = len(X)
        if n >= 3:
            x_mean = sum(x[0] for x in X) / n
            y_mean = sum(y) / n
            
            numerator = sum((X[i][0] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((X[i][0] - x_mean) ** 2 for i in range(n))
            
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                next_x = n
                predicted_num = int(round(slope * next_x + intercept)) % 8
                return num_to_room[predicted_num]
        
        return random.choice(list(self.room_mapping.values()))

    def phong_pattern_recognition(self):
        if len(self.lich_su_ket_qua) < 6:
            return random.choice(list(self.room_mapping.values()))
        
        for pattern_length in range(2, min(6, len(self.lich_su_ket_qua)//2 + 1)):
            recent_pattern = self.lich_su_ket_qua[-pattern_length:]
            
            for i in range(len(self.lich_su_ket_qua) - pattern_length * 2):
                if self.lich_su_ket_qua[i:i+pattern_length] == recent_pattern:
                    if i + pattern_length < len(self.lich_su_ket_qua):
                        return self.lich_su_ket_qua[i + pattern_length]
        
        return random.choice(list(self.room_mapping.values()))

    def phong_multi_factor_analysis(self):
        factors = {}
        
        xu_huong = self.phong_xuat_hien_nhieu_nhat(20)
        if xu_huong:
            factors[xu_huong] = factors.get(xu_huong, 0) + 30
        
        it_xuat_hien = self.phong_it_nhat(20)
        if it_xuat_hien:
            factors[it_xuat_hien] = factors.get(it_xuat_hien, 0) + 25
        
        khac_van_truoc = self.phong_khac_voi_van_truoc()
        if khac_van_truoc:
            factors[khac_van_truoc] = factors.get(khac_van_truoc, 0) + 20
        
        theo_gio = self.phong_theo_thoi_gian()
        if theo_gio:
            factors[theo_gio] = factors.get(theo_gio, 0) + 15
        
        theo_so_du = self.phong_theo_so_du()
        if theo_so_du:
            factors[theo_so_du] = factors.get(theo_so_du, 0) + 10
        
        if factors:
            return max(factors, key=factors.get)
        
        return random.choice(list(self.room_mapping.values()))

    def phong_bayesian_probability(self):
        if len(self.lich_su_ket_qua) < 5:
            return random.choice(list(self.room_mapping.values()))
        
        prior_prob = {}
        for room in self.room_mapping.values():
            prior_prob[room] = self.lich_su_ket_qua.count(room) / len(self.lich_su_ket_qua)
        
        recent_10 = self.lich_su_ket_qua[-10:] if len(self.lich_su_ket_qua) >= 10 else self.lich_su_ket_qua
        likelihood_prob = {}
        for room in self.room_mapping.values():
            likelihood_prob[room] = recent_10.count(room) / len(recent_10)
        
        posterior_prob = {}
        for room in self.room_mapping.values():
            posterior_prob[room] = prior_prob[room] * likelihood_prob[room]
        
        total = sum(posterior_prob.values())
        if total > 0:
            for room in posterior_prob:
                posterior_prob[room] /= total
        
        return min(posterior_prob, key=posterior_prob.get)

    def phong_neural_network(self):
        if len(self.lich_su_ket_qua) < 8:
            return random.choice(list(self.room_mapping.values()))
        
        room_to_vector = {}
        for i, room in enumerate(self.room_mapping.values()):
            vector = [0] * 8
            vector[i] = 1
            room_to_vector[room] = vector
        
        X = []  
        y = []  
        
        for i in range(len(self.lich_su_ket_qua) - 3):
            input_rooms = self.lich_su_ket_qua[i:i+3]
            output_room = self.lich_su_ket_qua[i+3]
            
            input_vector = []
            for room in input_rooms:
                input_vector.extend(room_to_vector[room])
            
            X.append(input_vector)
            y.append(room_to_vector[output_room])
        
        weights = [random.uniform(-1, 1) for _ in range(24)]  
        
        learning_rate = 0.1
        for epoch in range(100):
            for i in range(len(X)):
                output = [0] * 8
                for j in range(8):
                    for k in range(24):
                        output[j] += X[i][k] * weights[k]
                
                max_val = max(output)
                exp_output = [math.exp(o - max_val) for o in output]
                sum_exp = sum(exp_output)
                output = [o / sum_exp for o in exp_output]
                
                for j in range(8):
                    error = y[i][j] - output[j]
                    for k in range(24):
                        weights[k] += learning_rate * error * X[i][k]
        
        if len(self.lich_su_ket_qua) >= 3:
            recent_3 = self.lich_su_ket_qua[-3:]
            input_vector = []
            for room in recent_3:
                input_vector.extend(room_to_vector[room])
            
            output = [0] * 8
            for j in range(8):
                for k in range(24):
                    output[j] += input_vector[k] * weights[k]
            
            predicted_index = output.index(max(output))
            return list(self.room_mapping.values())[predicted_index]
        
        return random.choice(list(self.room_mapping.values()))

    def phong_genetic_algorithm(self):
        if len(self.lich_su_ket_qua) < 10:
            return random.choice(list(self.room_mapping.values()))
        
        population_size = 20
        population = []
        for _ in range(population_size):
            individual = {
                'weights': [random.uniform(0, 1) for _ in range(8)],
                'fitness': 0
            }
            population.append(individual)
        
        for individual in population:
            correct_predictions = 0
            for i in range(len(self.lich_su_ket_qua) - 1):
                scores = [0] * 8
                for j, room in enumerate(self.room_mapping.values()):
                    count = self.lich_su_ket_qua[:i+1].count(room)
                    scores[j] = count * individual['weights'][j]
                
                predicted_room = list(self.room_mapping.values())[scores.index(max(scores))]
                if predicted_room == self.lich_su_ket_qua[i+1]:
                    correct_predictions += 1
            
            individual['fitness'] = correct_predictions / (len(self.lich_su_ket_qua) - 1)
        
        population.sort(key=lambda x: x['fitness'], reverse=True)
        best_individuals = population[:population_size//2]
        
        new_population = best_individuals.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(best_individuals)
            parent2 = random.choice(best_individuals)
            
            child = {'weights': [], 'fitness': 0}
            for i in range(8):
                if random.random() < 0.5:
                    child['weights'].append(parent1['weights'][i])
                else:
                    child['weights'].append(parent2['weights'][i])
            
            for i in range(8):
                if random.random() < 0.1:
                    child['weights'][i] = random.uniform(0, 1)
            
            new_population.append(child)
        
        best_individual = max(new_population, key=lambda x: x['fitness'])
        scores = [0] * 8
        for j, room in enumerate(self.room_mapping.values()):
            count = self.lich_su_ket_qua.count(room)
            scores[j] = count * best_individual['weights'][j]
        
        return list(self.room_mapping.values())[scores.index(max(scores))]

    def phong_ensemble_learning(self):
        algorithms = [
            self.phong_markov_chain,
            self.phong_ml_linear_regression,
            self.phong_pattern_recognition,
            self.phong_multi_factor_analysis,
            self.phong_bayesian_probability,
            self.phong_neural_network,
            self.phong_genetic_algorithm
        ]
        
        predictions = {}
        for algo in algorithms:
            try:
                prediction = algo()
                predictions[prediction] = predictions.get(prediction, 0) + 1
            except:
                continue
        
        if predictions:
            return max(predictions, key=predictions.get)
        
        return random.choice(list(self.room_mapping.values()))

    def phong_gemini_ai(self):
        try:
            prompt = f"""
            Phân tích dữ liệu game Vua Thoát Hiểm:
            
            Lịch sử phòng bị sát: {self.lich_su_ket_qua[-20:] if len(self.lich_su_ket_qua) >= 20 else self.lich_su_ket_qua}
            Lịch sử thắng/thua: {self.lich_su_thang_thua[-10:] if len(self.lich_su_thang_thua) >= 10 else self.lich_su_thang_thua}
            
            Các phòng: Nhà Kho, Phòng Họp, Phòng Giám Đốc, Phòng Trò Chuyện, Phòng Giám Sát, Văn Phòng, Phòng Tài Vụ, Phòng Nhân Sự
            
            Dự đoán phòng nào sẽ AN TOÀN trong ván tiếp theo.
            Chỉ trả lời tên phòng.
            """
            
            genai.configure(api_key=self.GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            prediction = (resp.text or "").strip()
            
            room_mapping_reverse = {v: v for v in self.room_mapping.values()}
            if prediction in room_mapping_reverse:
                return prediction
            else:
                return random.choice(list(self.room_mapping.values()))
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(Fore.YELLOW + "⚠️ Gemini API hết quota, chuyển sang thuật toán nội bộ")
                return self.phong_genetic_algorithm()
            else:
                print(Fore.RED + f"❌ Lỗi Gemini API: {e}")
                return random.choice(list(self.room_mapping.values()))

    def dat_cuoc(self, room_id, logic, do_tin_cay=0, reasoning=""):
        if self.amount > self.so_du_ban_dau:
            print(Fore.RED + "⚠️ Số cược vượt số dư, reset về cược ban đầu.")
            self.amount = self.cuoc_ban_dau
        issue_id = self.lay_vong_hien_tai()
        if issue_id is None:
            print(Fore.RED + "Không lấy được vòng hiện tại, skip đặt cược.")
            return
        body = {
            "asset_type": self.bet_type,
            "bet_amount": self.amount,
            "room_id": room_id,
            "user_id": self.headers["user-id"]
        }
        try:
            resp = requests.post(self.api_cuoc, headers=self.headers, json=body, timeout=5)
            if resp.status_code == 200:
                print(Fore.LIGHTGREEN_EX + f"🚀 Đặt {self.amount} {self.bet_type} vào {self.room_mapping[room_id]} (Logic: {logic})")
                self.cuoc_dang_cho = {
                    "issue_id": issue_id,
                    "room_id": room_id,
                    "amount": self.amount,
                    "logic": logic,
                    "do_tin_cay": do_tin_cay
                }
            else:
                print(Fore.RED + f"Lỗi đặt cược: {resp.status_code}")
        except Exception as e:
            logging.error(f"Lỗi đặt cược: {e}")
            print(Fore.RED + f"Lỗi đặt cược: {e}")

    def lay_vong_hien_tai(self):
        with self.history_lock:
            if self.data_history:
                return self.data_history[0]["issue_id"] + 1
        return None

    def lay_lich_su(self, so_van):
        with self.history_lock:
            issues = self.data_history[:so_van]
        if issues:
            return [self.room_mapping.get(issue["killed_room_id"], "Không xác định") for issue in issues]
        return []

    def calc_overall_acc(self):
        corrects = [d['correct'] for d in self.data_history if d.get('correct') is not None]
        if len(corrects) > 0:
            return np.mean(corrects) * 100
        return 0

    def cap_nhat_chuoi_thang(self, ket_qua):
        if ket_qua == "Thắng":
            self.chuoi_thang_hien_tai += 1
            if self.chuoi_thang_hien_tai > self.chuoi_thang_max:
                self.chuoi_thang_max = self.chuoi_thang_hien_tai
        else:
            self.chuoi_thang_hien_tai = 0

    def lay_so_du_hien_tai(self):
        try:
            response = requests.get(self.api_login, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 200:
                    return data["data"]["cwallet"][self.bet_type_key[self.bet_type]]  
        except:
            pass
        return self.so_du_ban_dau

    def kiem_tra_ket_qua(self):
        if not self.cuoc_dang_cho:
            return

        try:
            items = self.get_my_joined()
            for item in items:
                if item["issue_id"] == self.cuoc_dang_cho["issue_id"]:
                    award = round(item.get("award_amount", 0), 3)
                    net = award - self.cuoc_dang_cho["amount"]
                    with self.history_lock:
                        for issue in self.data_history:
                            if issue["issue_id"] == self.cuoc_dang_cho["issue_id"]:
                                phong_bi_sat = self.room_mapping.get(issue["killed_room_id"], "Không xác định")
                                break
                        else:
                            print(Fore.RED + "Không tìm thấy kết quả vòng trong lịch sử.")
                            return

                    phong_dat_cuoc = self.room_mapping.get(self.cuoc_dang_cho["room_id"], "Không xác định")
                    logic = self.cuoc_dang_cho["logic"]
                    do_tin_cay = self.cuoc_dang_cho["do_tin_cay"]

                    self.lich_su_ket_qua.append(phong_bi_sat)
                    if len(self.lich_su_ket_qua) > 50:
                        self.lich_su_ket_qua.pop(0)

                    issue_id = item["issue_id"]
                    if not any(d['issue_id'] == issue_id for d in self.data_history):
                        self.data_history.append({
                            "issue_id": issue_id,
                            "killed_room_id": issue["killed_room_id"],
                            "prediction": self.room_mapping_name2id.get(phong_dat_cuoc, 0),
                            "correct": 1 if phong_bi_sat != phong_dat_cuoc else 0
                        })

                    if phong_bi_sat != phong_dat_cuoc:
                        print(Fore.GREEN + f"✅ Thắng vòng #{self.cuoc_dang_cho['issue_id']}! Reward: {award}")
                        self.tong_loi_lo += net
                        self.ghi_log_ai(phong_dat_cuoc, "Thắng", self.cuoc_dang_cho["amount"], do_tin_cay, logic)
                        self.amount = self.cuoc_ban_dau  
                        self.lich_su_thang_thua.append("Thắng")
                        self.cap_nhat_chuoi_thang("Thắng")
                    else:
                        print(Fore.RED + f"❌ Thua vòng #{self.cuoc_dang_cho['issue_id']}!")        
                        self.tong_loi_lo += net
                        self.ghi_log_ai(phong_dat_cuoc, "Thua", self.cuoc_dang_cho["amount"], do_tin_cay, logic)
                        self.amount *= self.he_so_gap  
                        self.lich_su_thang_thua.append("Thua")
                        self.cap_nhat_chuoi_thang("Thua")

                    print(Fore.CYAN + f"📈 Tổng lời/lỗ: {self.tong_loi_lo:.2f} {self.bet_type}\n")
                    acc = self.calc_overall_acc()
                    print(Fore.CYAN + f"📊 Overall Accuracy: {acc:.2f}%")

                    self.fine_tune_models()
                    self.update_rf_model()

                    self.cuoc_dang_cho = None
                    break
        except Exception as e:
            logging.error(f"Lỗi kiểm tra kết quả: {e}")
            print(Fore.RED + f"Lỗi kiểm tra kết quả: {e}")

    def wait_for_result(self):
        if not self.cuoc_dang_cho:
            return

        stop_event = threading.Event()

        def time_re():
            for i in range(100):
                if stop_event.is_set():
                    break
                print(f"Chờ [ {i} ] giây", end="\r", flush=True)
                time.sleep(1)

        def rq_history():
            try:
                initial_items = self.get_my_joined()
                initial_issue = initial_items[0]["issue_id"] if initial_items else None
            except:
                initial_issue = None
            while not stop_event.is_set():
                time.sleep(2)
                items = self.get_my_joined()
                if items and items[0]["issue_id"] != initial_issue:
                    print(f"Phiên [ {items[0]['issue_id']} ] đã kết thúc")
                    stop_event.set()

        t1 = threading.Thread(target=time_re)
        t2 = threading.Thread(target=rq_history)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def luu_tai_khoan(self, uid, user_login, secret_key, master):
        try:
            key = derive_key(master)
            encrypted_secret = encrypt_data(secret_key, key)
            try:
                with open("taikhoan.json", "r", encoding="utf-8") as f:
                    danh_sach = json.load(f)
            except FileNotFoundError:
                danh_sach = []
            if not any(tk["uid"] == uid and tk["user_login"] == user_login for tk in danh_sach):
                danh_sach.append({
                    "uid": uid,
                    "user_login": user_login,
                    "secret_key": encrypted_secret
                })
                with open("taikhoan.json", "w", encoding="utf-8") as f:
                    json.dump(danh_sach, f, ensure_ascii=False, indent=2)
                os.chmod("taikhoan.json", 0o600)
                print(Fore.GREEN + "✅ Đã lưu tài khoản thành công (encrypted)!")
            else:
                print(Fore.YELLOW + "ℹ️ Tài khoản đã tồn tại")
        except Exception as e:
            logging.error(f"Lỗi lưu tài khoản: {e}")
            print(Fore.RED + f"❌ Lỗi khi lưu tài khoản: {e}")

    def chon_tai_khoan(self, master):
        try:
            with open("taikhoan.json", "r", encoding="utf-8") as f:
                danh_sach = json.load(f)
            if not danh_sach:
                return None
            print(Fore.CYAN + "\n" + "═" * 50)
            print(Fore.YELLOW + "DANH SÁCH TÀI KHOẢN ĐÃ LƯU:")
            for i, tk in enumerate(danh_sach):
                print(Fore.LIGHTBLUE_EX + f"[{i+1}] UID: {tk['uid']} | Login: {tk['user_login']}")
            print(Fore.CYAN + "═" * 50)
            lua_chon = input(Fore.YELLOW + "Chọn tài khoản (0 để nhập mới): ")
            if lua_chon.isdigit():
                index = int(lua_chon) - 1
                if 0 <= index < len(danh_sach):
                    tk = danh_sach[index]
                    key = derive_key(master)
                    try:                        
                        decrypted_secret = decrypt_data(tk['secret_key'], key)
                        tk['secret_key'] = decrypted_secret
                    except Exception as e:
                        logging.error(f"Lỗi decrypt: {e}")
                        print(Fore.RED + f"❌ Lỗi decrypt secret_key: {e}")
                        return None
                    return tk
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.error(f"Lỗi chọn tài khoản: {e}")
        return None

    class EnsembleNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_ensembles=3):
            super().__init__()
            self.num_ensembles = num_ensembles
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_ensembles)])

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            outputs = [head(x) for head in self.heads]
            return torch.mean(torch.stack(outputs), dim=0)

    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.bn(out[:, -1, :])
            out = self.fc(out)
            return out

    class SimpleRFNet(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_classes=8):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.bn2 = nn.BatchNorm1d(hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, num_classes)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            return self.fc3(x)

    def load_or_init_models(self):
        input_size = 10
        hidden_size = 64
        num_classes = len(self.room_mapping)
        num_layers = 2

        self.model_ensemble = self.EnsembleNet(input_size, hidden_size, num_classes).to(self.device)
        self.optimizer_ensemble = optim.Adam(self.model_ensemble.parameters(), lr=0.001, weight_decay=1e-5)

        self.model_lstm = self.LSTMNet(1, hidden_size, num_layers, num_classes).to(self.device)
        self.optimizer_lstm = optim.Adam(self.model_lstm.parameters(), lr=0.001, weight_decay=1e-5)

        self.criterion = nn.CrossEntropyLoss()

        if os.path.exists(self.MODEL_ENSEMBLE_PATH):
            self.model_ensemble.load_state_dict(torch.load(self.MODEL_ENSEMBLE_PATH))
        if os.path.exists(self.MODEL_LSTM_PATH):
            self.model_lstm.load_state_dict(torch.load(self.MODEL_LSTM_PATH))

        self.calibrated_rf = self.SimpleRFNet().to(self.device)
        if os.path.exists(self.rf_model_path):
            self.calibrated_rf.load_state_dict(torch.load(self.rf_model_path))

    def prepare_data(self, issues, for_lstm=False):
        if len(issues) == 0:
            return torch.tensor([]), torch.tensor([])
        rooms = [issue["killed_room_id"] - 1 for issue in issues]
        sequences = []
        labels = []
        window = 10
        if len(rooms) < window + 1:
            pad = [random.randint(0, 7) for _ in range(window + 1 - len(rooms))]
            rooms = pad + rooms
        for i in range(len(rooms) - window):
            seq = rooms[i:i+window]
            sequences.append(seq if not for_lstm else np.array(seq).reshape(window, 1))
            labels.append(rooms[i+window])
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if sequences.shape[0] > 0:
            k = min(5, max(1, len(sequences)//20))
            if k > 1:
                flat_seq = sequences.reshape(sequences.shape[0], -1)
                cluster_centroids, codes = kmeans_native(flat_seq, k)
                unique, counts = np.unique(codes, return_counts=True)
                common_clusters = unique[counts > 1]
                mask = np.isin(codes, common_clusters)
                sequences = sequences[mask]
                labels = labels[mask]

        mean = np.mean(sequences) if sequences.size > 0 else 0
        std = np.std(sequences) if sequences.size > 0 else 1e-8
        sequences = (sequences - mean) / std

        sequences += np.random.normal(0, 0.01, sequences.shape) if sequences.size > 0 else 0

        return torch.tensor(sequences), torch.tensor(labels)

    def train_models(self, epochs=10):
        try:
            with self.history_lock:
                issues = self.data_history[:]
            if len(issues) < 20:
                print(Fore.YELLOW + "⚠️ Data quá ít, skip train ban đầu.")
                return

            existing_ids = {d['issue_id'] for d in self.data_history}
            new_entries = [{"issue_id": i["issue_id"], "killed_room_id": i["killed_room_id"], "prediction": None, "correct": None} for i in issues if i["issue_id"] not in existing_ids]
            self.data_history.extend(new_entries)
            print(Fore.GREEN + f"✅ Populated {len(new_entries)} initial history entries from web.")

            sequences, labels = self.prepare_data(issues)
            if len(labels) < 10:
                return
            split = int(0.8 * len(labels))
            train_seq, val_seq = sequences[:split], sequences[split:]
            train_lbl, val_lbl = labels[:split], labels[split:]
            dataset_train = RoomDataset(train_seq, train_lbl)
            loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
            self.model_ensemble.train()
            best_loss = float('inf')
            for epoch in range(epochs):
                epoch_loss = 0
                for seq, lbl in loader:
                    seq, lbl = seq.to(self.device), lbl.to(self.device)
                    self.optimizer_ensemble.zero_grad()
                    out = self.model_ensemble(seq)
                    loss = self.criterion(out, lbl)
                    loss.backward()
                    self.optimizer_ensemble.step()
                    epoch_loss += loss.item()
                if epoch_loss >= best_loss:  
                    break
                best_loss = epoch_loss
            self.model_ensemble.eval()
            with torch.no_grad():
                out = self.model_ensemble(val_seq.to(self.device))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                acc = np.mean(preds == val_lbl.numpy())
                print(Fore.GREEN + f"✅ Ensemble Val Acc: {acc:.2f}")
            torch.save(self.model_ensemble.state_dict(), self.MODEL_ENSEMBLE_PATH)

            sequences_lstm, labels_lstm = self.prepare_data(issues, for_lstm=True)
            train_seq_lstm, val_seq_lstm = sequences_lstm[:split], sequences_lstm[split:]
            train_lbl_lstm, val_lbl_lstm = labels_lstm[:split], labels_lstm[split:]
            dataset_lstm = RoomDataset(train_seq_lstm, train_lbl_lstm)
            loader_lstm = DataLoader(dataset_lstm, batch_size=32, shuffle=True)
            self.model_lstm.train()
            best_loss = float('inf')
            for epoch in range(epochs):
                epoch_loss = 0
                for seq, lbl in loader_lstm:
                    seq, lbl = seq.to(self.device), lbl.to(self.device)
                    self.optimizer_lstm.zero_grad()
                    out = self.model_lstm(seq)
                    loss = self.criterion(out, lbl)
                    loss.backward()
                    self.optimizer_lstm.step()
                    epoch_loss += loss.item()
                if epoch_loss >= best_loss:
                    break
                best_loss = epoch_loss
            self.model_lstm.eval()
            with torch.no_grad():
                out = self.model_lstm(val_seq_lstm.to(self.device))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                acc = np.mean(preds == val_lbl_lstm.numpy())
                print(Fore.GREEN + f"✅ LSTM Val Acc: {acc:.2f}")
            torch.save(self.model_lstm.state_dict(), self.MODEL_LSTM_PATH)

            print(Fore.GREEN + f"✅ Models trained/updated and saved.")
        except Exception as e:
            logging.error(f"Error training models: {e}")
            print(Fore.RED + f"❌ Error training models: {e}")

    def fine_tune_models(self):
        if len(self.data_history) > 20:
            recent_issues = self.data_history[-20:]
            sequences, labels = self.prepare_data(recent_issues)
            if len(labels) < 5:
                return
            split = int(0.8 * len(labels))
            train_seq, val_seq = sequences[:split], sequences[split:]
            train_lbl, val_lbl = labels[:split], labels[split:]
            dataset = RoomDataset(train_seq, train_lbl)
            loader = DataLoader(dataset, batch_size=8, shuffle=True)  
            self.model_ensemble.train()
            for epoch in range(5):
                for seq, lbl in loader:
                    seq, lbl = seq.to(self.device), lbl.to(self.device)
                    self.optimizer_ensemble.zero_grad()
                    out = self.model_ensemble(seq)
                    loss = self.criterion(out, lbl)
                    loss.backward()
                    self.optimizer_ensemble.step()
            self.model_ensemble.eval()
            with torch.no_grad():
                out = self.model_ensemble(val_seq.to(self.device))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                acc = np.mean(preds == val_lbl.numpy())
                print(Fore.GREEN + f"✅ Ensemble Fine-tune Val Acc: {acc:.2f}")
            torch.save(self.model_ensemble.state_dict(), self.MODEL_ENSEMBLE_PATH)

            sequences_lstm, labels_lstm = self.prepare_data(recent_issues, for_lstm=True)
            train_seq_lstm, val_seq_lstm = sequences_lstm[:split], sequences_lstm[split:]
            train_lbl_lstm, val_lbl_lstm = labels_lstm[:split], labels_lstm[split:]
            dataset_lstm = RoomDataset(train_seq_lstm, train_lbl_lstm)
            loader_lstm = DataLoader(dataset_lstm, batch_size=8, shuffle=True)
            self.model_lstm.train()
            for epoch in range(5):
                for seq, lbl in loader_lstm:
                    seq, lbl = seq.to(self.device), lbl.to(self.device)
                    self.optimizer_lstm.zero_grad()
                    out = self.model_lstm(seq)
                    loss = self.criterion(out, lbl)
                    loss.backward()
                    self.optimizer_lstm.step()
            self.model_lstm.eval()
            with torch.no_grad():
                out = self.model_lstm(val_seq_lstm.to(self.device))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                acc = np.mean(preds == val_lbl_lstm.numpy())
                print(Fore.GREEN + f"✅ LSTM Fine-tune Val Acc: {acc:.2f}")
            torch.save(self.model_lstm.state_dict(), self.MODEL_LSTM_PATH)

            print(Fore.GREEN + "✅ Models fine-tuned with new data.")

    def predict_with_ensemble(self, history):
        self.model_ensemble.eval()
        with torch.no_grad():
            seq = np.array(history[-10:], dtype=np.float32).reshape(1, -1)
            mean = np.mean(seq)
            std = np.std(seq) or 1e-8
            seq = (seq - mean) / std
            seq = torch.tensor(seq).to(self.device)
            out = self.model_ensemble(seq)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            return probs  

    def predict_with_lstm(self, history):
        self.model_lstm.eval()
        with torch.no_grad():
            seq = np.array(history[-10:], dtype=np.float32).reshape(1, 10, 1)
            mean = np.mean(seq)
            std = np.std(seq) or 1e-8
            seq = (seq - mean) / std
            seq = torch.tensor(seq).to(self.device)
            out = self.model_lstm(seq)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            return probs

    def predict_with_rf(self):
        if self.calibrated_rf:
            if self.lich_su_ket_qua:
                last_room = self.room_mapping_name2id.get(self.lich_su_ket_qua[-1], 1)
            else:
                last_history = self.lay_lich_su(1)
                last_room = self.room_mapping_name2id.get(last_history[0] if last_history else "Nhà Kho", 1)
            last_feat = torch.tensor([[last_room - 1]], dtype=torch.float32).to(self.device)
            self.calibrated_rf.eval()
            with torch.no_grad():
                out = self.calibrated_rf(last_feat)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            return probs
        return np.ones(8) / 8

    def hybrid_predict_with_confidence(self, history):
        probs_ensemble = self.predict_with_ensemble(history)
        probs_lstm = self.predict_with_lstm(history)
        probs_rf = self.predict_with_rf()

        w_ensemble = 1/3
        w_lstm = 1/3
        w_rf = 1/3
        probs_hybrid = w_ensemble * probs_ensemble + w_lstm * probs_lstm + w_rf * probs_rf

        ent = entropy_native(probs_hybrid)
        uncertainty = ent / np.log(len(self.room_mapping)) if np.log(len(self.room_mapping)) > 0 else 0
        confidence = (1 - uncertainty) * 100

        if confidence < 50:
            print(Fore.YELLOW + "⚠️ Confidence thấp, fallback rule-based nếu cần.")

        room_id = np.argmax(probs_hybrid) + 1

        reasoning = f"Hybrid probs: {probs_hybrid}. Entropy: {ent:.2f}. Sequence trend: {history[-5:]}"

        return room_id, confidence, reasoning

    def train_or_load_rf_model(self):
        self.calibrated_rf = self.SimpleRFNet().to(self.device)
        if os.path.exists(self.rf_model_path):
            self.calibrated_rf.load_state_dict(torch.load(self.rf_model_path))
            print(Fore.GREEN + "✅ Loaded existing RF-like model.")
        else:
            print(Fore.YELLOW + "ℹ️ Initializing new RF-like model.")

    def update_rf_model(self):
        if self.calibrated_rf and len(self.lich_su_ket_qua) > 20:
            new_history = self.lay_lich_su(100)
            if len(new_history) < 2:
                return
            y_new = np.array([self.room_mapping_name2id.get(name, 1) - 1 for name in new_history[1:]])
            X_new = np.array([self.room_mapping_name2id.get(name, 1) - 1 for name in new_history[:-1]]).reshape(-1, 1)
            X_new = torch.tensor(X_new, dtype=torch.float32).to(self.device)
            y_new = torch.tensor(y_new, dtype=torch.long).to(self.device)

            split = int(0.8 * len(y_new))
            X_train, X_val = X_new[:split], X_new[split:]
            y_train, y_val = y_new[:split], y_new[split:]

            optimizer_rf = optim.Adam(self.calibrated_rf.parameters(), lr=0.001, weight_decay=1e-5)
            criterion_rf = nn.CrossEntropyLoss()
            self.calibrated_rf.train()
            for epoch in range(5):
                out = self.calibrated_rf(X_train)
                loss = criterion_rf(out, y_train)
                optimizer_rf.zero_grad()
                loss.backward()
                optimizer_rf.step()
            self.calibrated_rf.eval()
            with torch.no_grad():
                out = self.calibrated_rf(X_val)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                acc = np.mean(preds == y_val.cpu().numpy())
                print(Fore.GREEN + f"✅ RF Val Acc: {acc:.2f}")
            torch.save(self.calibrated_rf.state_dict(), self.rf_model_path)
            print(Fore.GREEN + "✅ RF-like model updated with new data.")

    def calculate_confidence(self, thong_ke, so_van):
        if thong_ke:
            probs = np.array(list(thong_ke.values())) / so_van
            ent = entropy_native(probs)
            return (1 - ent / np.log(len(self.room_mapping))) * 100 if np.log(len(self.room_mapping)) > 0 else 0
        return 0

    def xu_ly_thuat_toan(self, logic_id):
        if logic_id == "max10":
            return self.phong_xuat_hien_nhieu_nhat(10)
        elif logic_id == "min10":
            return self.phong_it_nhat(10)
        elif logic_id == "max50":
            return self.phong_xuat_hien_nhieu_nhat(50)
        elif logic_id == "min50":
            return self.phong_it_nhat(50)
        elif logic_id == "max100":
            return self.phong_xuat_hien_nhieu_nhat(100)
        elif logic_id == "min100":
            return self.phong_it_nhat(100)
        elif logic_id == "trung_binh30":
            return self.phong_trung_binh(30)
        elif logic_id == "tang_truong40":
            return self.phong_xu_huong_tang(40)
        elif logic_id == "khac_van_truoc":
            return self.phong_khac_voi_van_truoc()
        elif logic_id == "fibonacci":
            return self.phong_theo_quy_luat_fibonacci()
        elif logic_id == "theo_gio":
            return self.phong_theo_thoi_gian()
        elif logic_id == "theo_so_du":
            return self.phong_theo_so_du()
        elif logic_id == "trong_so_ngau_nhien":
            return self.phong_ngau_nhien_co_trong_so()
        elif logic_id == "theo_mau":
            return self.phong_theo_mau()
        elif logic_id == "markov_chain":
            return self.phong_markov_chain()
        elif logic_id == "ml_linear":
            return self.phong_ml_linear_regression()
        elif logic_id == "pattern_recognition":
            return self.phong_pattern_recognition()
        elif logic_id == "multi_factor":
            return self.phong_multi_factor_analysis()
        elif logic_id == "bayesian":
            return self.phong_bayesian_probability()
        elif logic_id == "neural_network":
            return self.phong_neural_network()
        elif logic_id == "genetic_algo":
            return self.phong_genetic_algorithm()
        elif logic_id == "ensemble":
            return self.phong_ensemble_learning()
        elif logic_id == "gemini_ai":
            return self.phong_gemini_ai()
        elif logic_id == "hybrid_ai":
            history = [self.room_mapping_name2id.get(r, random.randint(1,8)) - 1 for r in self.lich_su_ket_qua[-10:]]
            if len(history) < 10:
                history = [random.randint(0,7) for _ in range(10 - len(history))] + history
            room_id, do_tin_cay, reasoning = self.hybrid_predict_with_confidence(history)
            return self.room_mapping.get(room_id, "Không xác định")
        elif logic_id == "smart":
            thong_ke = self.thong_ke_xu_huong(100)
            base_counts = {str(self.room_mapping_name2id[name]): count for name, count in thong_ke.items()}
            with self.history_lock:
                recent_records = self.data_history[:10]
            trust = {}
            best_key = choose_key_smart_all_in_one(base_counts, recent_records, trust_state=trust)
            return self.room_mapping.get(int(best_key), "Không xác định")
        return random.choice(list(self.room_mapping.values()))

    def chon_phong_va_cuoc(self):
        danh_sach_logic = [
            ("max10", "Xu hướng 10 ván gần nhất"),
            ("min10", "Phòng ít xuất hiện nhất 10 ván"),
            ("max50", "Xu hướng 50 ván gần nhất"),
            ("min50", "Phòng ít xuất hiện nhất 50 ván"),
            ("max100", "Xu hướng 100 ván gần nhất"),
            ("min100", "Phòng ít xuất hiện nhất 100 ván"),
            ("trung_binh30", "Phòng trung bình 30 ván"),
            ("tang_truong40", "Phòng tăng trưởng mạnh 40 ván"),
            ("khac_van_truoc", "Khác với ván trước"),
            ("fibonacci", "Theo dãy Fibonacci"),
            ("theo_gio", "Theo giờ hiện tại"),
            ("theo_so_du", "Theo số dư tài khoản"),
            ("trong_so_ngau_nhien", "Trọng số ngẫu nhiên"),
            ("theo_mau", "Theo chu kỳ màu sắc"),
            ("hybrid_ai", "Hybrid AI (Ensemble + LSTM + RF)"),
            ("hybrid_ai", "Hybrid AI (Ensemble + LSTM + RF)"),  # Add duplicate to increase probability
            ("smart", "Smart All In One"),
            ("markov_chain", "Markov Chain - Chuỗi xác suất"),
            ("ml_linear", "Machine Learning - Linear Regression"),
            ("pattern_recognition", "Pattern Recognition - Nhận diện mẫu"),
            ("multi_factor", "Multi-Factor Analysis - Phân tích đa yếu tố"),
            ("bayesian", "Bayesian Probability - Xác suất Bayes"),
            ("neural_network", "Neural Network - Mạng nơ-ron"),
            ("genetic_algo", "Genetic Algorithm - Thuật toán di truyền"),
            ("ensemble", "Ensemble Learning - Học tập tổng hợp")
        ]
        if self.su_dung_ai == "2":
            danh_sach_logic.extend([
                ("gemini_ai", "Gemini AI - Google AI")
            ])
        logic_id, logic_name = random.choice(danh_sach_logic)
        print(Fore.LIGHTYELLOW_EX + f"🔮 Đang sử dụng logic: {logic_name.upper()}")
        room_name = None
        do_tin_cay = 0
        reasoning = ""
        room_id = None

        try:
            room_name = self.xu_ly_thuat_toan(logic_id)
            do_tin_cay = self.calculate_confidence(self.thong_ke_xu_huong(100), 100)
            reasoning = "Processed with " + logic_name

            if room_name:
                room_id = self.room_mapping_name2id.get(room_name, random.randint(1,8))
            else:
                room_id = random.randint(1,8)
                room_name = self.room_mapping[room_id]

            print(Fore.YELLOW + f"🤖 Độ tin cậy: {do_tin_cay:.2f}% | Reasoning: {reasoning}")
            self.dat_cuoc(room_id, logic_name, do_tin_cay, reasoning)
        except Exception as e:
            logging.error(f"Lỗi chọn phòng: {e}")
            print(Fore.RED + f"Lỗi chọn phòng, fallback random: {e}")
            room_id = random.randint(1,8)
            self.dat_cuoc(room_id, "Fallback Random", 12.5, "Error fallback")

    def run(self):
        self.banner()

        master = input(Fore.YELLOW + "Nhập master password cho encryption (bắt buộc): ")
        if not master:
            print(Fore.RED + "Master password không được rỗng!")
            return

        tai_khoan_chon = self.chon_tai_khoan(master)
        if tai_khoan_chon:
            user_id = tai_khoan_chon["uid"]
            user_login = tai_khoan_chon["user_login"]
            user_secret_key = tai_khoan_chon["secret_key"]
            print(Fore.GREEN + f"\nĐã chọn tài khoản: UID={user_id}")
        else:
            link = input(Fore.YELLOW + "Nhập link của bạn: ")
            parsed_url = urlparse(link)
            query_params = parse_qs(parsed_url.query)
            user_id = query_params.get('userId', [''])[0]
            user_secret_key = query_params.get('secretKey', [''])[0]
            user_login = "login_v2"  # Mặc định
            if not user_id or not user_secret_key:
                print(Fore.RED + "Link không hợp lệ!")
                return
            if input(Fore.YELLOW + "Bạn có muốn lưu tài khoản này? (y/n): ").lower() == "y":
                self.luu_tai_khoan(user_id, user_login, user_secret_key, master)

        self.load_or_create_config()
        self.amount = self.cuoc_ban_dau
        self.headers = {
            "user-id": user_id,
            "user-login": user_login,
            "user-secret-key": user_secret_key
        }
        if not self.login():
            print(Fore.RED + "Login fail, exit.")
            return

        self.load_or_init_models()
        self.train_models()
        self.train_or_load_rf_model()
        self.tai_thong_ke_thuat_toan()

        print(Fore.CYAN + "\n" + "═" * 60)
        print(Fore.YELLOW + "🤖 THIẾT LẬP AI API:")
        print(Fore.CYAN + "═" * 60)
        print(Fore.LIGHTBLUE_EX + "[1] Không sử dụng AI API")
        print(Fore.LIGHTGREEN_EX + "[2] Sử dụng AI API (Gemini)")
        print(Fore.CYAN + "═" * 60)
        
        while True:
            self.su_dung_ai = input(Fore.YELLOW + "Chọn có sử dụng AI API không? (1/2): ")
            if self.su_dung_ai in ["1", "2"]:
                break
            else:
                print(Fore.RED + "❌ Vui lòng chọn 1 hoặc 2")
        
        if self.su_dung_ai == "2":
            if not self.setup_ai_apis():
                print(Fore.RED + "❌ Không thể thiết lập AI API, chuyển về chế độ thường")
                self.su_dung_ai = "1"
            else:
                print(Fore.GREEN + "✅ Đã thiết lập AI API thành công!")

        print(Fore.CYAN + "\n" + "═" * 60)
        print(Fore.YELLOW + "🤖 CHỌN CHẾ ĐỘ THUẬT TOÁN:")
        print(Fore.CYAN + "═" * 60)
        if self.su_dung_ai == "1":
            print(Fore.LIGHTBLUE_EX + "[1] Tất cả thuật toán (20 thuật toán)")
            print(Fore.LIGHTGREEN_EX + "[2] Chỉ AI nâng cao (8 thuật toán)")
        else:
            print(Fore.LIGHTBLUE_EX + "[1] Tất cả thuật toán (21 thuật toán)")
            print(Fore.LIGHTGREEN_EX + "[2] Chỉ AI API (Gemini)")
        print(Fore.CYAN + "═" * 60)
        
        while True:
            self.che_do = input(Fore.YELLOW + "Chọn chế độ (1/2): ")
            if self.che_do in ["1", "2"]:
                break
            else:
                print(Fore.RED + "❌ Vui lòng chọn 1 hoặc 2")

        threading.Thread(target=self.history_collector, daemon=True).start()

        self.thong_ke_xu_huong_va_do_tin_cay(100)

        try:
            van_dem = 0
            while self.tool_running:
                self.kiem_tra_ket_qua()
                if not self.cuoc_dang_cho:
                    self.chon_phong_va_cuoc()
                    van_dem += 1
                    
                    if van_dem % 10 == 0:
                        self.hien_thi_thong_ke_thuat_toan()
                        self.hien_thi_thong_ke_chinh_xac()
                    
                    self.hien_thi_thong_ke_nhanh()
                self.wait_for_result()
                self.kiem_tra_ket_qua()
                time.sleep(1)
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n🛑 Dừng bởi người dùng ")
            self.hien_thi_thong_ke_thuat_toan()
            self.hien_thi_thong_ke_chinh_xac()

class RoomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

if __name__ == "__main__":
    bot = GameBot()
    bot.run()