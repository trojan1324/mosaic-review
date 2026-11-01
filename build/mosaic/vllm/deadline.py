"""Opt-in deadline for an entire HTTP exchange, including DNS and trickle headers.

Caller deadlines never wait for worker cleanup. A late connection cannot start a
request after expiration. Already-started sends are conservatively unknown; no
retry authority is added. A daemon worker may outlive a DNS timeout but must close
its eventual connection without sending. This is not server-side cancellation.
"""
import http.client
import math
import queue
import socket
import threading
import time
import urllib.parse

from mosaic.vllm.client import TransportError


class DeadlineTransport:
    def __init__(self, *, maximum_response_bytes=16 * 1024 * 1024):
        if type(maximum_response_bytes) is not int or maximum_response_bytes <= 0:
            raise ValueError('response_size_bound_required')
        self.maximum_response_bytes = maximum_response_bytes

    def request(self, method, url, headers, body, timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise TransportError('positive_finite_request_deadline_required', sent=False)
        try:
            parts = urllib.parse.urlsplit(url)
            port = parts.port
        except ValueError:
            raise TransportError('bounded_transport_url_invalid', sent=False) from None
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise TransportError('bounded_transport_url_invalid', sent=False)
        deadline = time.monotonic() + timeout
        lock = threading.Lock(); results = queue.Queue(maxsize=1)
        state = {'expired': False, 'sent': False, 'connection': None, 'socket': None}

        def remaining():
            with lock:
                seconds = deadline - time.monotonic()
                if state['expired'] or seconds <= 0:
                    raise TimeoutError('absolute_request_deadline')
                conn = state['connection']
            if conn is not None:
                conn.timeout = seconds
                if conn.sock is not None:
                    conn.sock.settimeout(seconds)
            return seconds

        def exchange():
            conn = None
            try:
                cls = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
                conn = cls(parts.hostname, port, timeout=remaining())
                with lock:
                    state['connection'] = conn
                remaining(); conn.connect()
                with lock:
                    state['socket'] = conn.sock
                remaining()
                # This is the irreversible send boundary. Expiration and send
                # authorization share a lock, so a late DNS/connect completion
                # cannot start a request after the caller has timed out.
                with lock:
                    if state['expired'] or time.monotonic() >= deadline:
                        raise TimeoutError('absolute_request_deadline')
                    state['sent'] = True
                conn.request(method, (parts.path or '/') + ('?' + parts.query if parts.query else ''),
                             body=body, headers=headers)
                remaining(); response = conn.getresponse(); chunks = []; size = 0
                while True:
                    remaining(); chunk = response.read1(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.maximum_response_bytes:
                        raise ValueError('response_size_bound_exceeded')
                    chunks.append(chunk)
                remaining()
                results.put((True, (response.status, {k.lower(): v for k, v in response.getheaders()}, b''.join(chunks))))
            except Exception as exc:
                # No headers, credentials, response text or exception message.
                with lock:
                    sent = state['sent']
                results.put((False, TransportError('bounded_http_' + type(exc).__name__, sent=sent)))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass

        worker = threading.Thread(target=exchange, name='mosaic-bounded-http', daemon=True)
        worker.start()
        try:
            ok, result = results.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            with lock:
                state['expired'] = True
                sent = state['sent']; sock = state['socket']
            # Interrupt reads/writes without closing a BufferedReader from a
            # different thread (which can block on that reader's internal lock).
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            raise TransportError('absolute_request_deadline', sent=sent) from None
        if not ok:
            raise result
        if time.monotonic() >= deadline:
            raise TransportError('absolute_request_deadline', sent=True)
        return result
