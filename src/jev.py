"""Client for the evidence judge: TypeSafe Jev, a model trained for calibrated decisions.

The judge receives text only (a candidate's evidence record), never images, and returns calibrated probabilities.
Credentials: TYPESAFE_API_KEY, or a file named by TYPESAFE_API_KEY_FILE. Neither is ever logged.
Optional: SCOUT_JEV_LOG=<file.jsonl> appends one line per request (time, model, input tokens, latency) for cost
accounting; SCOUT_JEV_ADDRS=<ip,ip> pins the API host's address (for networks whose resolver fails on it, or where
IPv6 connections to it stall).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENDPOINT = "https://api.typesafe.ai/v1/systemone"

if os.environ.get("SCOUT_JEV_ADDRS"):
    import socket
    _getaddrinfo, _pinned = socket.getaddrinfo, os.environ["SCOUT_JEV_ADDRS"].split(",")

    def _pinned_getaddrinfo(host, port, *args, **kw):
        """Pins only the API host; TLS still verifies the certificate against its name."""
        if host == "api.typesafe.ai":
            return [r for ip in _pinned for r in _getaddrinfo(ip, port, *args, **kw)]
        return _getaddrinfo(host, port, *args, **kw)
    socket.getaddrinfo = _pinned_getaddrinfo


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None                   # never forward the credential to another host


class JevClient:
    """judge(state, questions) -> {name: answer}. One retry on a timeout or a 5xx; the caller measures the latency
    (retry included) and charges it to the mission clock."""

    RETRIES = 1

    def __init__(self, key_file=None, timeout=15.0, model="jev-1.13.0"):
        self.key_file = key_file or os.environ.get("TYPESAFE_API_KEY_FILE")
        self.timeout = timeout
        self.model = model
        self.log = os.environ.get("SCOUT_JEV_LOG")

    def judge(self, state, questions: dict):
        """Evaluate one state against a map of typed questions, answered together in one request. Each answer gains
        a `value`: the probability for a Noul, the expected level for a Score, the winner's confidence for a Choice."""
        reply = self._post({"model": self.model, "state": state, "questions": questions})
        answers = reply.get("answers", {})
        if set(answers) != set(questions):
            raise RuntimeError("TypeSafe returned answers for a different question set")
        scalar = {"noul": "noul", "score": "score", "choice": "confidence"}
        for answer in answers.values():
            if answer.get("type") not in scalar:
                raise RuntimeError("TypeSafe returned an unknown answer type")
            answer["value"] = answer[scalar[answer["type"]]]
        return answers

    def _key(self):
        key = os.environ.get("TYPESAFE_API_KEY") or (Path(self.key_file).read_text().strip() if self.key_file else "")
        if not key:
            raise ValueError("No TypeSafe credential: set TYPESAFE_API_KEY or TYPESAFE_API_KEY_FILE")
        return key

    def _post(self, payload: dict):
        for attempt in range(self.RETRIES + 1):
            try:
                return self._post_once(payload)
            except Exception as exc:
                text = str(exc)
                retryable = isinstance(exc, (TimeoutError, OSError)) or \
                    text.startswith("TypeSafe HTTP status 5") or "timed out" in text
                if not retryable or attempt == self.RETRIES:
                    raise
                time.sleep(1.0)

    def _post_once(self, payload: dict):
        body = json.dumps(payload, allow_nan=False).encode()
        request = Request(ENDPOINT, data=body, method="POST", headers={
            "Authorization": "Bearer " + self._key(), "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with build_opener(_NoRedirect).open(request, timeout=self.timeout) as response:
                reply = json.load(response)
        except HTTPError as exc:
            raise RuntimeError("TypeSafe HTTP status %s" % exc.code) from None
        except (URLError, TimeoutError):
            raise RuntimeError("TypeSafe connection failed or timed out") from None
        self.last_usage = reply.get("usage")
        if self.log:
            with open(self.log, "a") as fh:
                fh.write(json.dumps({"t": round(t0, 3), "model": self.model, "latency_s": round(time.time() - t0, 3),
                                     "input_tokens": (self.last_usage or {}).get("input_tokens")}) + "\n")
        return reply
