"""Cache local en mémoire : LRU (taille max) + TTL (expiration).

Choix volontaire : pas de Redis, un cache process-local suffit pour un
chatbot personnel à trafic faible/modéré. Verrouillé explicitement (`_lock`)
car ce cache est un singleton partagé (app/cache/caches.py) entre toutes les
requêtes/utilisateurs et `OrderedDict` n'est pas thread-safe nativement —
reste sûr même si l'exécution devient un jour multi-thread."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Generic, TypeVar

V = TypeVar("V")


class TTLLRUCache(Generic[V]):
    def __init__(self, max_size: int, ttl_seconds: int) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[V, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None

            value, expires_at = entry
            if expires_at < time.monotonic():
                del self._store[key]
                self.misses += 1
                return None

            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: V) -> None:
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._max_size:
                self._store.popitem(last=False)  # évince l'entrée la plus ancienne

            self._store[key] = (value, time.monotonic() + self._ttl)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(self.hits / total, 3) if total else 0.0,
            }
