#!/usr/bin/env python3
"""Замер Share of Voice бренда в ответах ИИ-ассистентов.

Гоняет пул запросов из конфига по нескольким моделям с повторами,
сохраняет сырые ответы в JSONL и считает метрики SoV по брендам:
mention rate, top-1 rate, средний ранг.

Использование:
    export OPENAI_API_KEY=... YANDEX_API_KEY=... GIGACHAT_AUTH_KEY=...
    python3 sov_measure.py queries.yaml -o results/

Зависимости: pip install requests pyyaml
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

TIMEOUT = 120
RETRIES = 3

# GigaChat работает с сертификатом НУЦ Минцифры. Укажите путь к нему в
# GIGACHAT_CA_BUNDLE; без него проверка TLS у GigaChat отключается (с warning).
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


def post_with_retries(url, **kwargs):
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(url, timeout=TIMEOUT, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp
        except Exception as e:  # noqa: BLE001 - сетевые ошибки ретраим все
            last_err = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"запрос не удался после {RETRIES} попыток: {last_err}")


class OpenAICompatClient:
    def __init__(self, cfg):
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.api_key = os.environ.get(cfg["key_env"], "")
        if not self.api_key:
            raise SystemExit(f"не задана переменная окружения {cfg['key_env']}")

    def ask(self, question):
        resp = post_with_retries(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": question}],
            },
        )
        return resp.json()["choices"][0]["message"]["content"]


class GigaChatClient:
    def __init__(self, cfg):
        self.model = cfg.get("model", "GigaChat")
        self.auth_key = os.environ.get(cfg["key_env"], "")
        if not self.auth_key:
            raise SystemExit(f"не задана переменная окружения {cfg['key_env']}")
        self.verify = os.environ.get("GIGACHAT_CA_BUNDLE") or False
        if self.verify is False:
            print("warning: GIGACHAT_CA_BUNDLE не задан, TLS-проверка GigaChat отключена", file=sys.stderr)
        self._token = None
        self._token_exp = 0

    def _get_token(self):
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        resp = post_with_retries(
            GIGACHAT_OAUTH_URL,
            headers={
                "Authorization": f"Basic {self.auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")},
            verify=self.verify,
        )
        data = resp.json()
        self._token = data["access_token"]
        self._token_exp = data["expires_at"] / 1000
        return self._token

    def ask(self, question):
        resp = post_with_retries(
            GIGACHAT_CHAT_URL,
            headers={"Authorization": f"Bearer {self._get_token()}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": question}],
            },
            verify=self.verify,
        )
        return resp.json()["choices"][0]["message"]["content"]


def make_client(cfg):
    if cfg["api"] == "openai":
        return OpenAICompatClient(cfg)
    if cfg["api"] == "gigachat":
        return GigaChatClient(cfg)
    raise SystemExit(f"неизвестный api: {cfg['api']}")


def find_mentions(text, brands):
    """Возвращает [(brand_name, позиция первого упоминания)], отсортированный по позиции."""
    found = []
    for brand in brands:
        best = None
        for alias in brand["aliases"]:
            # Границы слова: не матчим алиас внутри другого слова.
            pattern = r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])"
            m = re.search(pattern, text, re.IGNORECASE)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            found.append((brand["name"], best))
    found.sort(key=lambda x: x[1])
    return found


def run_measurement(config, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    raw_path = out_dir / f"raw-{stamp}.jsonl"

    brands = config["brands"]
    runs = int(config.get("runs_per_query", 5))
    clients = {m["id"]: make_client(m) for m in config["models"]}

    total = len(config["models"]) * len(config["queries"]) * runs
    done = 0
    records = []

    with raw_path.open("a", encoding="utf-8") as raw_f:
        for model_cfg in config["models"]:
            client = clients[model_cfg["id"]]
            for q in config["queries"]:
                for run_idx in range(runs):
                    done += 1
                    try:
                        answer = client.ask(q["text"])
                        error = None
                    except Exception as e:  # noqa: BLE001
                        answer, error = "", str(e)
                    mentions = find_mentions(answer, brands) if answer else []
                    rec = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "model": model_cfg["id"],
                        "query": q["text"],
                        "query_type": q.get("type", "category"),
                        "run": run_idx + 1,
                        "answer": answer,
                        "error": error,
                        "mentions": [{"brand": b, "pos": p, "rank": i + 1}
                                     for i, (b, p) in enumerate(mentions)],
                    }
                    raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    records.append(rec)
                    status = "ERR " + error[:60] if error else f"{len(mentions)} брендов"
                    print(f"[{done}/{total}] {model_cfg['id']} | {q['text'][:50]!r} #{run_idx + 1}: {status}")
    return records, raw_path, stamp


def build_report(records, brands, out_dir, stamp):
    ok = [r for r in records if not r["error"]]
    groups = defaultdict(list)  # (model, query_type) -> records; плюс (model, "all")
    for r in ok:
        groups[(r["model"], r["query_type"])].append(r)
        groups[(r["model"], "all")].append(r)

    rows = []
    for (model, qtype), recs in sorted(groups.items()):
        n = len(recs)
        for brand in brands:
            name = brand["name"]
            hits = [m for r in recs for m in r["mentions"] if m["brand"] == name]
            mentioned_in = sum(1 for r in recs if any(m["brand"] == name for m in r["mentions"]))
            top1 = sum(1 for r in recs if r["mentions"] and r["mentions"][0]["brand"] == name)
            avg_rank = round(sum(m["rank"] for m in hits) / len(hits), 2) if hits else ""
            rows.append({
                "model": model,
                "query_type": qtype,
                "brand": name,
                "is_client": bool(brand.get("is_client")),
                "answers": n,
                "mention_rate_%": round(100 * mentioned_in / n, 1) if n else 0,
                "top1_rate_%": round(100 * top1 / n, 1) if n else 0,
                "avg_rank": avg_rank,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"sov-report-{stamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = len(records) - len(ok)
    print(f"\n=== SoV, сводка по всем типам запросов (ответов: {len(ok)}, ошибок: {errors}) ===")
    print(f"{'модель':<12} {'бренд':<20} {'mention%':>9} {'top1%':>7} {'ср.ранг':>8}")
    for row in rows:
        if row["query_type"] != "all":
            continue
        mark = " <- клиент" if row["is_client"] else ""
        print(f"{row['model']:<12} {row['brand']:<20} {row['mention_rate_%']:>9} "
              f"{row['top1_rate_%']:>7} {str(row['avg_rank']):>8}{mark}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="YAML-конфиг (см. queries.example.yaml)")
    parser.add_argument("-o", "--out", default="results", help="каталог результатов")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    records, raw_path, stamp = run_measurement(config, out_dir)
    csv_path = build_report(records, config["brands"], out_dir, stamp)
    print(f"\nСырые ответы: {raw_path}\nОтчёт: {csv_path}")


if __name__ == "__main__":
    main()
