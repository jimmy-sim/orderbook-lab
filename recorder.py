"""Record Coinbase Exchange order book snapshots and trades to Parquet.

Research only: this connects to Coinbase's free public market-data feed. It needs no
account or API key and cannot place orders.

What it does
------------
* Subscribes to `level2_batch` (book updates batched every 50ms), `matches` (every trade)
  and `heartbeat` (once a second, carries the latest trade id).
* Keeps a full local copy of each order book: starts from the `snapshot` message, then
  applies every `l2update`.
* Every SAMPLE_MS milliseconds it saves the top DEPTH levels of each book (one row).
* Saves every trade as it arrives.
* Buffers rows in memory and writes a new Parquet file every FLUSH_S seconds into
  hourly folders: data/{book,trades}/product=BTC-USD/date=YYYY-MM-DD/hour=HH/*.parquet
* If the connection drops, goes quiet, or the book looks wrong (best bid >= best ask),
  it reconnects. Reconnecting re-subscribes, which sends a fresh full snapshot, so the
  book is rebuilt from scratch. Every such gap is appended to data/gaps.jsonl so the
  analysis can skip those windows.
* Missed trades are detected from jumps in Coinbase's consecutive trade ids.
* After a pause (laptop sleep, process frozen) the local book is stale. The recorder
  notices the late clock tick, discards the book and resyncs; it also stops saving rows
  whenever no message has arrived for QUIET_ROW_S seconds.

Run:  uv run python recorder.py               (BTC-USD and ETH-USD)
      uv run python recorder.py --products BTC-USD --minutes 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import websockets
from sortedcontainers import SortedDict

URL = "wss://ws-feed.exchange.coinbase.com"
DEPTH = 10          # levels per side saved in each snapshot
SAMPLE_MS = 100     # snapshot interval
FLUSH_S = 300       # how often buffered rows are written to disk
STALE_S = 15        # reconnect if no message for this long
QUIET_ROW_S = 3     # don't save book rows if no message has arrived for this long
JUMP_S = 2          # a sampler tick this late means the process was asleep/frozen
DATA = Path(__file__).parent / "data"

log = logging.getLogger("recorder")


def iso_to_ns(s: str) -> int:
    """Coinbase timestamps look like 2026-11-23T21:52:15.469740Z."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.timestamp()) * 1_000_000_000 + dt.microsecond * 1000


class Book:
    """One product's order book: price -> size on each side."""

    def __init__(self) -> None:
        self.bids: SortedDict = SortedDict()  # ascending; best bid is the last key
        self.asks: SortedDict = SortedDict()  # ascending; best ask is the first key
        self.synced = False
        self.exch_ns = 0      # exchange time of the last update applied
        self.updates = 0      # l2update messages since the last snapshot row

    def load_snapshot(self, msg: dict) -> None:
        self.bids = SortedDict({float(p): float(s) for p, s in msg["bids"]})
        self.asks = SortedDict({float(p): float(s) for p, s in msg["asks"]})
        self.exch_ns = iso_to_ns(msg["time"]) if msg.get("time") else time.time_ns()
        self.synced = True

    def apply(self, msg: dict) -> None:
        for side, p, s in msg["changes"]:
            book = self.bids if side == "buy" else self.asks
            price, size = float(p), float(s)
            if size == 0.0:
                book.pop(price, None)
            else:
                book[price] = size
        self.exch_ns = iso_to_ns(msg["time"])
        self.updates += 1

    def crossed(self) -> bool:
        return bool(self.bids and self.asks) and self.bids.peekitem(-1)[0] >= self.asks.peekitem(0)[0]

    def top(self, n: int) -> tuple[list, list, list, list]:
        nb, na = min(n, len(self.bids)), min(n, len(self.asks))
        bp = [self.bids.peekitem(-1 - i)[0] for i in range(nb)]
        bs = [self.bids[p] for p in bp]
        ap = [self.asks.peekitem(i)[0] for i in range(na)]
        az = [self.asks[p] for p in ap]
        pad = lambda xs: xs + [float("nan")] * (n - len(xs))  # noqa: E731
        return pad(bp), pad(bs), pad(ap), pad(az)


class Sink:
    """Buffers rows per (kind, product) and writes them to hour-partitioned Parquet."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: dict[tuple[str, str], list[dict]] = {}

    def add(self, kind: str, product: str, row: dict) -> None:
        self.rows.setdefault((kind, product), []).append(row)

    def flush(self) -> None:
        for (kind, product), rows in self.rows.items():
            if not rows:
                continue
            # split rows by the hour of their local receive time
            by_hour: dict[str, list[dict]] = {}
            for r in rows:
                h = datetime.fromtimestamp(r["ts_ns"] / 1e9, tz=timezone.utc).strftime("%Y-%m-%d/%H")
                by_hour.setdefault(h, []).append(r)
            for h, hr in by_hour.items():
                date, hour = h.split("/")
                d = self.root / kind / f"product={product}" / f"date={date}" / f"hour={hour}"
                d.mkdir(parents=True, exist_ok=True)
                tmp = d / f".part-{hr[0]['ts_ns']}.tmp"
                pq.write_table(pa.Table.from_pylist(hr), tmp, compression="zstd")
                tmp.rename(d / f"part-{hr[0]['ts_ns']}.parquet")  # atomic: readers never see half a file
            log.info("flushed %d %s rows for %s", len(rows), kind, product)
        self.rows = {}


class Recorder:
    def __init__(self, products: list[str], root: Path) -> None:
        self.products = products
        self.root = root
        self.sink = Sink(root)
        self.books = {p: Book() for p in products}
        self.last_trade_id: dict[str, int] = {}
        self.last_msg_ns = time.time_ns()   # wall clock: keeps counting through laptop sleep
        self.ws = None
        self.disconnected_at_ns: int | None = None
        self.stop = asyncio.Event()

    # ---- gap log -------------------------------------------------------------
    def gap(self, kind: str, **info) -> None:
        rec = {"kind": kind, "ts_ns": time.time_ns(), **info}
        log.warning("gap: %s", rec)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "gaps.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- trades --------------------------------------------------------------
    def check_trade_id(self, product: str, tid: int) -> None:
        last = self.last_trade_id.get(product)
        if last is not None and tid > last + 1:
            self.gap("missed_trades", product=product, after_trade_id=last, before_trade_id=tid,
                     missed=tid - last - 1)
        if last is None or tid > last:
            self.last_trade_id[product] = tid

    def on_match(self, msg: dict, recv_ns: int) -> None:
        p, tid = msg["product_id"], int(msg["trade_id"])
        if msg["type"] == "match":
            last = self.last_trade_id.get(p)
            if last is not None and tid <= last:
                return  # duplicate (e.g. replayed after reconnect)
            self.sink.add("trades", p, {
                "ts_ns": recv_ns,
                "exch_ns": iso_to_ns(msg["time"]),
                "trade_id": tid,
                "price": float(msg["price"]),
                "size": float(msg["size"]),
                # Coinbase's `side` is the resting (maker) order's side, so the
                # aggressor (taker) is the opposite: maker sell => buyer lifted the ask.
                "taker_side": "buy" if msg["side"] == "sell" else "sell",
            })
        self.check_trade_id(p, tid)

    # ---- book sampling -------------------------------------------------------
    async def sampler(self) -> None:
        step = SAMPLE_MS * 1_000_000
        nxt = (time.time_ns() // step + 1) * step
        last_tick = time.time_ns()
        while not self.stop.is_set():
            await asyncio.sleep(max(0.0, (nxt - time.time_ns()) / 1e9))
            now = time.time_ns()
            if now - last_tick > JUMP_S * 1e9:
                # We were asleep (lid closed) or frozen. The local book missed every
                # update in between, so it is stale: drop it and force a fresh snapshot.
                self.force_resync(f"woke after {(now - last_tick) / 1e9:.1f}s pause", since_ns=last_tick)
            last_tick = now
            if now - self.last_msg_ns > QUIET_ROW_S * 1e9:
                # feed has gone quiet: the book may be stale, so save nothing until it's back
                nxt = max(nxt + step, (now // step + 1) * step)
                continue
            for p, b in self.books.items():
                if not b.synced or not b.bids or not b.asks:
                    continue
                bp, bs, ap, az = b.top(DEPTH)
                row = {"ts_ns": nxt, "exch_ns": b.exch_ns, "n_updates": b.updates}
                for i in range(DEPTH):
                    row[f"bid_px_{i + 1}"] = bp[i]
                    row[f"bid_sz_{i + 1}"] = bs[i]
                    row[f"ask_px_{i + 1}"] = ap[i]
                    row[f"ask_sz_{i + 1}"] = az[i]
                self.sink.add("book", p, row)
                b.updates = 0
            # skip missed slots if we fell behind (e.g. the laptop slept)
            nxt = max(nxt + step, (now // step + 1) * step)

    def force_resync(self, reason: str, since_ns: int) -> None:
        log.warning("forcing resync: %s", reason)
        for b in self.books.values():
            b.synced = False
        if self.disconnected_at_ns is None:
            self.disconnected_at_ns = since_ns
        if self.ws is not None:
            asyncio.create_task(self.ws.close())  # session() sees the close and reconnects

    async def flusher(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=FLUSH_S)
            except asyncio.TimeoutError:
                pass
            await asyncio.to_thread(self.sink.flush)

    # ---- connection ----------------------------------------------------------
    async def session(self) -> None:
        sub = {"type": "subscribe", "product_ids": self.products,
               "channels": ["level2_batch", "matches", "heartbeat"]}
        async with websockets.connect(URL, max_size=2**25, ping_interval=20, ping_timeout=20,
                                      close_timeout=1) as ws:
            self.ws = ws
            await ws.send(json.dumps(sub))
            log.info("connected, subscribed to %s", self.products)
            for b in self.books.values():
                b.synced = False  # wait for a fresh snapshot
            while not self.stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=STALE_S)
                except asyncio.TimeoutError:
                    raise ConnectionError(f"no message for {STALE_S}s")
                recv_ns = time.time_ns()
                self.last_msg_ns = recv_ns
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "l2update":
                    b = self.books[msg["product_id"]]
                    if b.synced:
                        b.apply(msg)
                        if b.crossed():
                            b.synced = False
                            raise ConnectionError(f"crossed book on {msg['product_id']}, resyncing")
                elif t == "snapshot":
                    self.books[msg["product_id"]].load_snapshot(msg)
                    if self.disconnected_at_ns is not None:
                        self.gap("book_resync", product=msg["product_id"],
                                 start_ns=self.disconnected_at_ns, end_ns=recv_ns,
                                 seconds=round((recv_ns - self.disconnected_at_ns) / 1e9, 3))
                    log.info("book snapshot for %s: %d bids, %d asks", msg["product_id"],
                             len(msg["bids"]), len(msg["asks"]))
                    if all(bk.synced for bk in self.books.values()):
                        self.disconnected_at_ns = None
                elif t in ("match", "last_match"):
                    self.on_match(msg, recv_ns)
                elif t == "error":
                    raise ConnectionError(f"feed error: {msg}")

    async def connect_loop(self) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                await self.session()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - any failure means reconnect
                if self.disconnected_at_ns is None:
                    # the outage really began at the last message we received
                    self.disconnected_at_ns = self.last_msg_ns
                for b in self.books.values():
                    b.synced = False
                if time.monotonic() - started > 60:
                    backoff = 1.0  # the last session was healthy; reconnect quickly
                wait = backoff + random.random()
                log.warning("connection lost (%s: %s); reconnecting in %.1fs", type(e).__name__, e, wait)
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60.0)

    async def run(self, minutes: float | None) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop.set)
        tasks = [asyncio.create_task(c) for c in (self.connect_loop(), self.sampler(), self.flusher())]
        try:
            if minutes:
                await asyncio.wait_for(self.stop.wait(), timeout=minutes * 60)
            else:
                await self.stop.wait()
        except asyncio.TimeoutError:
            pass
        self.stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sink.flush()  # never lose the buffer on a clean stop
        log.info("stopped")


def main() -> None:
    global FLUSH_S
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--products", nargs="+", default=["BTC-USD", "ETH-USD"])
    ap.add_argument("--minutes", type=float, default=None, help="stop after this long (default: run forever)")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--flush", type=float, default=FLUSH_S, help="seconds between disk writes")
    args = ap.parse_args()
    FLUSH_S = args.flush
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(Recorder(args.products, args.data).run(args.minutes))


if __name__ == "__main__":
    main()
