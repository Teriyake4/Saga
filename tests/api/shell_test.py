#!/usr/bin/env python3
"""Finite real-process/HTTP tests; no model weights or mock generation."""
import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from unittest import mock


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.getheaders(), response.read()
    finally:
        connection.close()


def raw_exchange(port, headers, suffix=b"", method="POST", path="/v1/chat/completions",
                 half_close=False, timeout=5):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.settimeout(timeout)
        sock.sendall(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n".encode("ascii") +
                     headers + b"\r\n" + suffix)
        if half_close:
            sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            try:
                chunk = sock.recv(4096)
            except ConnectionResetError:
                # Closing a socket with unread pipelined bytes may send RST.
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


def check_raw_response(response, expected_status, error_code=None, context="raw request"):
    check(b"\r\n\r\n" in response, "complete raw response: " + repr(response))
    status, rest = response.split(b"\r\n", 1)
    check(status.startswith(f"HTTP/1.1 {expected_status} ".encode("ascii")),
          f"{context}: expected {expected_status}: " + repr(response))
    response_headers, body = rest.split(b"\r\n\r\n", 1)
    check(b"connection: close" in response_headers.lower(), "raw connection must close")
    if error_code is not None:
        check(json.loads(body)["error"]["code"] == error_code, "structured raw rejection: " + repr(body))
    check(response.count(b"HTTP/1.1 ") == 1, "no interim 100 or pipelined response")


def reject_raw_framing(port, headers, suffix=b"", method="POST", path="/v1/chat/completions"):
    # Deliberately withhold the body: a body-first check would hit the 1s read
    # timeout instead of replying immediately. Also require EOF, not just a
    # Connection: close response header, to catch a live keep-alive read loop.
    response = raw_exchange(port, headers, suffix, method, path, timeout=0.75)
    check_raw_response(response, 400, "invalid_request_framing", context=f"framing headers {headers!r}")


def framing_tests(port):
    for headers in (
        b"Content-Length: 1\r\nContent-Length: 2\r\n",
        b"content-length: 2\r\nCONTENT-LENGTH: 1\r\n",
        b"Content-Length: 0\r\nContent-Length: 1000\r\n",
        b"Content-Length: 1000\r\nContent-Length: 0\r\n",
        b"Content-Length: 1, 2\r\n",
        b"Content-Length: 1\r\nContent-Length: 1\r\n",
        b"Transfer-Encoding: chunked\r\nContent-Length: 1000\r\n",
        b"Content-Length: 0\r\ntransfer-encoding: chunked\r\n",
        b"Transfer-Encoding: identity\r\nContent-Length: 0\r\n",
        b"Transfer-Encoding: chunked\r\nTransfer-Encoding: identity\r\n",
        b"Transfer-Encoding: gzip, chunked\r\n",
        b"Content-Length: -1\r\n", b"Content-Length: +1\r\n",
        b"Content-Length: 1x\r\n", b"Content-Length: \r\n",
        b"Content-Length: 18446744073709551616\r\n",
    ):
        reject_raw_framing(port, headers)
    # These are invalid literal header bytes, not alternate encodings of 0 or
    # chunked. With Expect they must fail before even an interim 100 response.
    for headers in (b"Content-Length: %30\r\n", b"Content-Length: %u0030\r\n",
                    b"Transfer-Encoding: %63hunked\r\n"):
        reject_raw_framing(port, headers)
        reject_raw_framing(port, headers + b"Expect: 100-continue\r\n")
    ambiguous = b"Content-Length: 1000\r\nTransfer-Encoding: chunked\r\n"
    reject_raw_framing(port, ambiguous + b"Expect: 100-continue\r\n")
    pipeline = b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reject_raw_framing(port, b"Content-Length: 0\r\nContent-Length: 1\r\n", pipeline,
                       method="GET", path="/healthz")
    reject_raw_framing(port, ambiguous, b"0\r\n\r\n" + pipeline)
    check(request(port, "GET", "/healthz")[0] == 200, "malformed connection isolation")
    chunk_tests(port)


def chunk_tests(port):
    for chunk_size in (b"+0", b"0x0", b"0junk", b"1garbage", b"-1", b"-0",
                       b"10000000000000000", b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"):
        response = raw_exchange(port, b"Transfer-Encoding: chunked\r\n",
                                chunk_size + b"\r\nx\r\n0\r\n\r\n", timeout=0.75)
        check_raw_response(response, 400, "http_error", context=f"chunk size {chunk_size!r}")
    for body in (b"0\r\n", b"2\r\n{}\r\n0\r\n", b"2\r\n{}\r\n0\r\n\r"):
        # EOF is not the empty trailer line terminating a chunked message.
        # The vendor may suppress writes after SHUT_WR. Accept silent close here;
        # deterministic vendor parser tests must also verify rejection, not success
        # followed by a suppressed write. Never accept a successful HTTP response.
        response = raw_exchange(port, b"Transfer-Encoding: chunked\r\n", body, half_close=True)
        if response:
            check_raw_response(response, 400, "http_error", context=f"missing terminator {body!r}")
    for encoding in (b"chunked", b"Chunked", b"CHUNKED", b"cHuNkEd"):
        for body in (b"2\r\n{}\r\n0\r\n\r\n",
                     b"2;name=value;flag;quoted=\"a;b\\\"c\"\r\n{}\r\n0;end=yes\r\n\r\n"):
            response = raw_exchange(port, b"Transfer-Encoding: " + encoding + b"\r\n", body)
            check_raw_response(response, 503, "model_not_ready")
    check(request(port, "GET", "/healthz")[0] == 200, "chunk rejection isolation")


def authentication_tests(port, key):
    for scheme in ("Bearer", "bearer", "BEARER", "bEaReR"):
        for spaces in (" ", "  ", "     "):
            check(request(port, "GET", "/v1/models", headers={"Authorization": scheme + spaces + key})[0] == 200,
                  "bearer scheme is ASCII case-insensitive and accepts 1+ SP")
    for authorization in ("Bearer\t" + key, "Bearer \t" + key, "Bearer\t " + key,
                          "Bearer" + key, "Bearer   ", "Bearer " + key.swapcase(),
                          "Bearer " + key + "=", "Bearer " + key.rstrip("=")):
        check(request(port, "GET", "/v1/models", headers={"Authorization": authorization})[0] == 401,
              "separator must be SP; token bytes and padding are exact")
    raw_key = key.encode("ascii")
    for headers in (b"Authorization: Bearer " + raw_key + b"\r\nAuthorization: Bearer " + raw_key + b"\r\n",
                    b"authorization: Bearer " + raw_key + b"\r\nAUTHORIZATION: Bearer wrong\r\n"):
        response = raw_exchange(port, headers, method="GET", path="/v1/models")
        check_raw_response(response, 401, "invalid_api_key")
    # Encoded spellings must not authenticate as the raw configured token.
    for encoded_key in (f"%{raw_key[0]:02X}".encode("ascii") + raw_key[1:],
                        f"%u{raw_key[0]:04X}".encode("ascii") + raw_key[1:],
                        raw_key.replace(b"+", b"%2B"), raw_key.replace(b"=", b"%3D")):
        check(encoded_key != raw_key, "raw token probe must change the wire bytes")
        response = raw_exchange(port, b"Authorization: Bearer " + encoded_key + b"\r\n",
                                method="GET", path="/v1/models")
        check_raw_response(response, 401, "invalid_api_key")
    response = raw_exchange(port, b"Authorization: bEaReR   " + raw_key + b"\r\n",
                            method="GET", path="/v1/models")
    check_raw_response(response, 200)


class RunningServer:
    def __init__(self, binary, config, directory, key=None, port=0):
        self.log_path = directory / ("auth.log" if key else "server.log")
        self.log_file = self.log_path.open("wb")
        self.process = None
        self.port = None
        self.shutdown_signal = signal.SIGTERM
        try:
            environment = os.environ.copy()
            environment.pop("SAGA_API_KEY", None)
            if key:
                environment["SAGA_API_KEY"] = key
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self.process = subprocess.Popen([binary, "--config", str(config), "--port", str(port)],
                                            stdout=subprocess.DEVNULL, stderr=self.log_file,
                                            stdin=subprocess.DEVNULL, env=environment, creationflags=flags)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                check(self.process.poll() is None, "server exited during startup: " + self.logs())
                for line in self.logs().splitlines():
                    event = json.loads(line)
                    if event["event"] == "listener_bound":
                        self.port = int(event["detail"].rsplit(":", 1)[1])
                if self.port:
                    try:
                        if request(self.port, "GET", "/healthz")[0] == 200:
                            return
                    except (OSError, http.client.HTTPException):
                        pass
                time.sleep(0.02)
            raise AssertionError("startup deadline exceeded")
        except BaseException:
            try:
                if self.process is not None and self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=5)
            finally:
                self.log_file.close()
            raise

    def logs(self):
        return self.log_path.read_text(encoding="utf-8")

    def stop(self):
        try:
            if self.process.poll() is None:
                shutdown_signal = signal.CTRL_BREAK_EVENT if os.name == "nt" else self.shutdown_signal
                self.process.send_signal(shutdown_signal)
            return_code = self.process.wait(timeout=10)
            check(return_code == 0, f"shutdown failed with exit code {return_code}: " + self.logs())
            events = [json.loads(line).get("event") for line in self.logs().splitlines()]
            check("listener_stopped" in events, "successful listener_stopped event: " + self.logs())
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            self.log_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    binary = str(Path(args.server).resolve())
    environment = os.environ.copy()
    environment.pop("SAGA_API_KEY", None)
    with tempfile.TemporaryDirectory(prefix="saga-api-") as temp:
        directory = Path(temp)
        for flag in ("--help", "--version"):
            result = subprocess.run([binary, flag], capture_output=True, timeout=10, env=environment)
            check(result.returncode == 0 and b"inference" in result.stdout, flag)
        result = subprocess.run([binary, "--config", args.config, "--check-config"], capture_output=True, timeout=10, env=environment)
        check(result.returncode == 0 and b"NOT validated" in result.stdout, "example config static validation")
        for flags in (["--port", "-1"], ["--host", "0.0.0.0", "--port", "0"], ["--unknown"]):
            result = subprocess.run([binary] + flags, capture_output=True, timeout=10, env=environment)
            check(result.returncode != 0 and json.loads(result.stderr)["event"] == "startup_failed", "invalid startup must fail")
        model_ini = directory / "single model \u6a21\u578b.ini"
        model_ini.write_bytes('[model]\r\nmodel="weights/model.engine"\r\nalias=qwen3.8-27b-gsq3\r\nctx-size=4096\r\n'.encode("utf-8-sig"))
        checked_model = subprocess.run([binary, "--model-config", str(model_ini), "--check-config"],
                                       capture_output=True, timeout=10, env=environment)
        check(checked_model.returncode == 0 and b"NOT validated" in checked_model.stdout,
              "one Unicode-path INI validates without loading or listing models")
        unavailable_model = subprocess.run([binary, "--model-config", str(model_ini)],
                                           capture_output=True, timeout=10, env=environment)
        check(unavailable_model.returncode != 0 and b"not implemented" in unavailable_model.stderr,
              "model INI must not bypass unsupported artifact startup")
        model_ini.write_text('[model]\nctx-size=4096\n[another-model]\nmodel=other.engine\n', encoding="utf-8")
        rejected_catalog = subprocess.run([binary, "--model-config", str(model_ini), "--check-config"],
                                          capture_output=True, timeout=10, env=environment)
        check(rejected_catalog.returncode != 0, "reject multiple model profiles in one file")
        artifact = directory / "artifact.toml"
        artifact.write_text('[model]\nartifact="nonexistent.engine"\n', encoding="utf-8")
        result = subprocess.run([binary, "--config", str(artifact)], capture_output=True, timeout=10, env=environment)
        check(result.returncode != 0 and b"not implemented" in result.stderr, "do not silently ignore model loading")

        config = directory / "\u8bbe\u7f6e.toml"
        config.write_text('[server]\nport=0\nmax_body_bytes=1024\nhttp_workers=2\nmax_pending_connections=4\nio_timeout_seconds=1\n', encoding="utf-8")
        # Config file order does not override CLI: even --config last leaves the CLI value authoritative.
        precedence = subprocess.run([binary, "--port", "5", "--config", str(config), "--check-config"],
                                    capture_output=True, timeout=10, env=environment)
        check(precedence.returncode == 0, "CLI/config order")
        for invalid_key in ("bad%41", "bad%", "bad key", "bad\tkey", "bad:key", "a" * 4097,
                            "=bad", "====", "bad=middle", "bad==middle", "bad=+"):
            key_environment = environment.copy()
            key_environment["SAGA_API_KEY"] = invalid_key
            result = subprocess.run([binary, "--config", str(config)], capture_output=True,
                                    timeout=10, env=key_environment)
            check(result.returncode != 0 and b"[A-Za-z0-9._~+/-]" in result.stderr and
                  b"optional '=' padding" in result.stderr and b"4096" in result.stderr,
                  "unsupported key startup error explains bounded b64token grammar")
            check(b"decod" not in result.stderr, "key startup error must not blame header decoding")
            check(invalid_key.encode() not in result.stderr, "unsupported key is not logged")
        original_path_open = Path.open
        opened_logs = []

        def track_open(path, *open_args, **open_kwargs):
            opened = original_path_open(path, *open_args, **open_kwargs)
            opened_logs.append(opened)
            return opened

        with mock.patch.object(Path, "open", track_open):
            try:
                RunningServer(str(directory / "missing-saga-server"), config, directory)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("missing executable must fail to spawn")
        check(len(opened_logs) == 1 and opened_logs[0].closed,
              "spawn failure closes the opened server log")

        abnormal = RunningServer(binary, config, directory)
        abnormal.process.kill()
        abnormal.process.wait(timeout=5)
        try:
            abnormal.stop()
        except AssertionError as error:
            check("exit code" in str(error), "already-exited abnormal process reports its exit status")
        else:
            raise AssertionError("stop must reject an already-exited abnormal process")

        # A real occupied listener must fail as structured startup, then the
        # exact same port must be reusable after releasing the blocker and server.
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == "nt":
                blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            blocker.bind(("127.0.0.1", 0))
            blocker.listen()
            occupied_port = blocker.getsockname()[1]
            failed = subprocess.run([binary, "--config", str(config), "--host", "127.0.0.1",
                                     "--port", str(occupied_port)], capture_output=True, timeout=10, env=environment)
            check(failed.returncode != 0 and json.loads(failed.stderr)["event"] == "startup_failed",
                  "occupied port must fail with structured startup_failed")
        finally:
            blocker.close()
        first = RunningServer(binary, config, directory, port=occupied_port)
        first.stop()
        second = RunningServer(binary, config, directory, port=occupied_port)
        second.stop()

        for repetition in range(2):
            server = RunningServer(binary, config, directory)
            idle = None
            try:
                port = server.port
                status, _, body = request(port, "GET", "/healthz")
                check(status == 200 and json.loads(body)["status"] == "ok", "liveness")
                status, _, body = request(port, "GET", "/readyz")
                check(status == 503 and json.loads(body)["error"]["code"] == "model_not_ready", "never fake readiness")
                status, _, body = request(port, "GET", "/v1/models")
                check(status == 200 and json.loads(body) == {"object": "list", "data": []}, "no fake registered model")
                for streaming in (False, True):
                    status, headers, body = request(port, "POST", "/v1/chat/completions",
                        json.dumps({"model": "qwen3.8-27b-gsq3", "stream": streaming, "messages": []}),
                        {"Content-Type": "application/json"})
                    check(status == 503 and json.loads(body)["error"]["code"] == "model_not_ready", "generation unavailable before headers/SSE")
                    check("text/event-stream" not in str(headers), "no fake stream")
                status, _, body = request(port, "POST", "/v1/chat/completions", b"x" * 1025)
                check(status == 413 and json.loads(body)["error"]["code"] == "request_too_large", "body bound")
                status, _, body = request(port, "GET", "/unknown")
                check(status == 404 and json.loads(body)["error"]["code"] == "not_found", "structured 404")
                status, _, body = request(port, "GET", "/metrics")
                check(status == 200 and b"saga_engine_ready 0" in body, "truthful metrics")
                framing_tests(port)
                # Excessive header count is rejected before routing, never accepted as healthy.
                with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                    sock.settimeout(5)
                    sock.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n" + b"X-Test: a\r\n" * 80 + b"\r\n")
                    check(b" 200 " not in sock.recv(4096).split(b"\r\n", 1)[0], "header count bound")
                # Disconnect and idle partial headers do not poison the next connection or hang shutdown.
                with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                    sock.sendall(b"GET /healthz HTTP/1.1\r\n")
                check(request(port, "GET", "/healthz")[0] == 200, "disconnect isolation")
                if repetition == 1:
                    idle = socket.create_connection(("127.0.0.1", port), timeout=5)
                    idle.sendall(b"GET /healthz HTTP/1.1\r\n")
            finally:
                try:
                    if os.name != "nt" and repetition == 0:
                        server.shutdown_signal = signal.SIGINT
                    server.stop()
                finally:
                    if idle:
                        idle.close()
        key = "test-only-secret-DO-NOT-LOG_0123456789.ABCxyz~+/-="
        server = RunningServer(binary, config, directory, key)
        try:
            for path in ("/readyz", "/v1/models", "/metrics"):
                check(request(server.port, "GET", path)[0] == 401, "auth protects non-liveness routes")
                check(request(server.port, "GET", path, headers={"Authorization": "Bearer wrong"})[0] == 401, "invalid key")
            check(request(server.port, "GET", "/v1/models", headers={"Authorization": "Bearer " + key})[0] == 200, "valid key")
            authentication_tests(server.port, key)
            reject_raw_framing(server.port, b"Content-Length: 1000\r\nContent-Length: 1\r\n")
            reject_raw_framing(server.port, b"Transfer-Encoding: chunked\r\nContent-Length: 1\r\nExpect: 100-continue\r\n")
            check(key not in server.logs(), "secret not logged")
        finally:
            server.stop()
        check(key not in server.logs(), "secret absent from shutdown logs")
    print("PASS: finite HTTP/process shell, raw framing/chunk grammar, exact bearer tokens, limits, signals and restart")


if __name__ == "__main__":
    main()
