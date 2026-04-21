#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

try:
    import websockets
except ImportError:  # pragma: no cover - runtime dependency check
    websockets = None

try:
    from bleak import BleakClient, BleakScanner
except ImportError:  # pragma: no cover - runtime dependency check
    BleakClient = None
    BleakScanner = None

BleakClientType: TypeAlias = Any


OPENGLASS_NAME = "OpenGlass"
PHOTO_DATA_UUID = "19b10005-e8f2-537e-4f6c-d104768a1214"
PHOTO_CONTROL_UUID = "19b10006-e8f2-537e-4f6c-d104768a1214"
SINGLE_PHOTO_COMMAND = bytes([0xFF])

DEFAULT_WS_URL = "ws://127.0.0.1:8000/xiaozhi/v1/"
DEFAULT_DEVICE_ID = "openglass-bridge"
DEFAULT_CLIENT_ID = "openglass-bridge"


class BridgeError(RuntimeError):
    pass


@dataclass
class VisionTarget:
    url: str = ""
    token: str = ""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sanitize_stem(value: str | None) -> str:
    if not value:
        return time.strftime("openglass_%Y%m%d_%H%M%S")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return stem.strip("._-") or time.strftime("openglass_%Y%m%d_%H%M%S")


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    if websockets is None:
        missing.append("websockets")
    if BleakClient is None or BleakScanner is None:
        missing.append("bleak")
    if missing:
        joined = " ".join(missing)
        raise BridgeError(f"Missing Python package(s): {joined}. Install with: pip install {joined}")


class OpenGlassCamera:
    def __init__(self, ble_name: str, address: str | None, capture_timeout: float) -> None:
        self.ble_name = ble_name
        self.address = address
        self.capture_timeout = capture_timeout
        self.client: BleakClientType | None = None
        self._capture_lock = asyncio.Lock()
        self._current_future: asyncio.Future[bytes] | None = None
        self._buffer = bytearray()
        self._previous_packet_id = -1

    async def connect(self) -> None:
        require_runtime_dependencies()
        target = self.address

        if not target:
            log(f"Scanning BLE devices for {self.ble_name!r}...")
            device = await BleakScanner.find_device_by_filter(
                lambda dev, adv: dev.name == self.ble_name or adv.local_name == self.ble_name,
                timeout=10.0,
            )
            if device is None:
                raise BridgeError(f"Could not find BLE device named {self.ble_name!r}")
            target = device.address

        log(f"Connecting to OpenGlass BLE target {target}...")
        self.client = BleakClient(target)
        await self.client.connect()
        await self.client.start_notify(PHOTO_DATA_UUID, self._on_photo_data)
        log("OpenGlass BLE connected; photo notifications enabled.")

    async def close(self) -> None:
        if self.client is None:
            return
        try:
            if self.client.is_connected:
                await self.client.stop_notify(PHOTO_DATA_UUID)
                await self.client.disconnect()
        finally:
            self.client = None

    async def take_photo(self) -> bytes:
        if self.client is None or not self.client.is_connected:
            raise BridgeError("OpenGlass BLE is not connected")

        async with self._capture_lock:
            loop = asyncio.get_running_loop()
            self._current_future = loop.create_future()
            self._buffer.clear()
            self._previous_packet_id = -1

            await self.client.write_gatt_char(PHOTO_CONTROL_UUID, SINGLE_PHOTO_COMMAND, response=True)
            try:
                photo = await asyncio.wait_for(self._current_future, timeout=self.capture_timeout)
            finally:
                self._current_future = None
            if not photo.startswith(b"\xff\xd8"):
                raise BridgeError("OpenGlass returned data that does not look like a JPEG")
            return photo

    def _on_photo_data(self, _: Any, data: bytearray) -> None:
        if self._current_future is None or self._current_future.done():
            return

        packet = bytes(data)
        if len(packet) >= 2 and packet[0] == 0xFF and packet[1] == 0xFF:
            self._current_future.set_result(bytes(self._buffer))
            return

        if len(packet) < 2:
            self._current_future.set_exception(BridgeError("Received malformed OpenGlass photo packet"))
            return

        packet_id = packet[0] | (packet[1] << 8)
        expected = self._previous_packet_id + 1
        if packet_id != expected:
            self._current_future.set_exception(
                BridgeError(f"OpenGlass photo packet gap: expected {expected}, got {packet_id}")
            )
            return

        self._previous_packet_id = packet_id
        self._buffer.extend(packet[2:])


def build_multipart(question: str, image_bytes: bytes, file_name: str) -> tuple[bytes, str]:
    boundary = f"----openglass-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def add(value: bytes) -> None:
        parts.append(value)

    add(f"--{boundary}\r\n".encode())
    add(b'Content-Disposition: form-data; name="question"\r\n')
    add(b"Content-Type: text/plain; charset=utf-8\r\n\r\n")
    add(question.encode("utf-8"))
    add(b"\r\n")

    add(f"--{boundary}\r\n".encode())
    add(f'Content-Disposition: form-data; name="image"; filename="{file_name}"\r\n'.encode())
    add(b"Content-Type: image/jpeg\r\n\r\n")
    add(image_bytes)
    add(b"\r\n")

    add(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def post_vision(
    target: VisionTarget,
    device_id: str,
    client_id: str,
    question: str,
    image_bytes: bytes,
    file_name: str,
    timeout: float,
) -> dict[str, Any]:
    if not target.url:
        raise BridgeError("Vision URL is empty; wait for xiaozhi MCP initialize or pass --vision-url")

    body, boundary = build_multipart(question, image_bytes, file_name)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Device-Id": device_id,
        "Client-Id": client_id,
    }
    if target.token:
        headers["Authorization"] = f"Bearer {target.token}"

    request = urllib.request.Request(target.url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BridgeError(f"Vision HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BridgeError(f"Vision request failed: {exc.reason}") from exc

    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"Vision returned non-JSON response: {payload[:200]}") from exc
    return result


async def connect_websocket(url: str, headers: dict[str, str]) -> Any:
    assert websockets is not None
    try:
        return await websockets.connect(url, additional_headers=headers)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers)


class XiaozhiOpenGlassBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vision = VisionTarget(args.vision_url, args.vision_token)
        self.camera = OpenGlassCamera(args.ble_name, args.ble_address, args.capture_timeout)
        self.session_id: str | None = None

    async def run(self) -> None:
        require_runtime_dependencies()
        await self.camera.connect()
        try:
            await self._run_websocket()
        finally:
            await self.camera.close()

    async def _run_websocket(self) -> None:
        headers = {
            "Device-Id": self.args.device_id,
            "Client-Id": self.args.client_id,
        }
        if self.args.authorization:
            headers["Authorization"] = self.args.authorization

        log(f"Connecting to xiaozhi websocket {self.args.xiaozhi_ws} as {self.args.device_id}...")
        ws = await connect_websocket(self.args.xiaozhi_ws, headers)
        try:
            await self._send_hello(ws)
            log("Bridge is online. Trigger self.camera.take_photo from xiaozhi.")
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                await self._handle_message(ws, raw)
        finally:
            await ws.close()

    async def _send_hello(self, ws: Any) -> None:
        payload = {
            "type": "hello",
            "version": 1,
            "transport": "websocket",
            "features": {"mcp": True},
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))

    async def _handle_message(self, ws: Any, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            log(f"Ignoring non-JSON websocket text: {raw[:120]}")
            return

        msg_type = message.get("type")
        if msg_type == "hello":
            self.session_id = message.get("session_id") or self.session_id
            log(f"Received xiaozhi hello; session_id={self.session_id or 'unknown'}")
            return

        if msg_type != "mcp":
            return

        payload = message.get("payload")
        if not isinstance(payload, dict):
            return
        response = await self._handle_mcp_payload(payload)
        if response is not None:
            outbound: dict[str, Any] = {"type": "mcp", "payload": response}
            if self.session_id:
                outbound["session_id"] = self.session_id
            await ws.send(json.dumps(outbound, ensure_ascii=False))

    async def _handle_mcp_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        method = payload.get("method")
        request_id = payload.get("id")
        if method == "initialize":
            self._capture_vision_target(payload.get("params", {}))
            return self._mcp_result(
                request_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "openglass-xiaozhi-bridge", "version": "0.1.0"},
                },
            )

        if method == "tools/list":
            return self._mcp_result(request_id, {"tools": [self._camera_tool_definition()]})

        if method == "tools/call":
            params = payload.get("params", {})
            return await self._handle_tool_call(request_id, params)

        if request_id is None:
            return None
        return self._mcp_error(request_id, -32601, f"Unsupported MCP method: {method}")

    def _capture_vision_target(self, params: dict[str, Any]) -> None:
        vision = params.get("vision") if isinstance(params, dict) else None
        capabilities = params.get("capabilities") if isinstance(params, dict) else None
        if not isinstance(vision, dict) and isinstance(capabilities, dict):
            vision = capabilities.get("vision")
        if isinstance(vision, dict):
            self.vision.url = vision.get("url") or self.vision.url
            self.vision.token = vision.get("token") or self.vision.token
        if self.vision.url:
            log(f"Vision endpoint ready: {self.vision.url}")
        else:
            log("MCP initialized, but no vision endpoint was provided.")

    async def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if name != "self.camera.take_photo":
            return self._mcp_error(request_id, -32602, f"Unsupported tool: {name}")

        arguments = params.get("arguments") or {}
        question = str(arguments.get("question") or "Please describe the current scene and give experiment guidance.")
        photo_name = sanitize_stem(arguments.get("photo_name"))
        file_name = f"{photo_name}.jpg"

        try:
            log(f"Taking OpenGlass photo for question: {question[:80]}")
            image_bytes = await self.camera.take_photo()
            image_path = self._save_capture(file_name, image_bytes)
            log(f"Saved capture: {image_path}")
            vision_result = await asyncio.to_thread(
                post_vision,
                self.vision,
                self.args.device_id,
                self.args.client_id,
                question,
                image_bytes,
                file_name,
                self.args.vision_timeout,
            )
            return self._tool_text_result(request_id, json.dumps(vision_result, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - MCP should return errors as data
            log(f"Tool call failed: {exc}")
            return self._tool_text_result(request_id, str(exc), is_error=True)

    def _save_capture(self, file_name: str, image_bytes: bytes) -> Path:
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        target = save_dir / file_name
        counter = 2
        while target.exists():
            target = save_dir / f"{Path(file_name).stem}_{counter}.jpg"
            counter += 1
        target.write_bytes(image_bytes)
        return target

    @staticmethod
    def _camera_tool_definition() -> dict[str, Any]:
        return {
            "name": "self.camera.take_photo",
            "description": "Capture one JPEG photo from OpenGlass and submit it to xiaozhi vision analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question or instruction for the vision model.",
                    },
                    "photo_name": {
                        "type": "string",
                        "description": "Optional local filename stem for the captured JPEG.",
                    },
                },
                "required": ["question"],
            },
        }

    @staticmethod
    def _mcp_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _mcp_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_text_result(request_id: Any, text: str, is_error: bool = False) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose OpenGlass BLE camera as xiaozhi MCP tool self.camera.take_photo."
    )
    parser.add_argument("--xiaozhi-ws", default=DEFAULT_WS_URL, help="Xiaozhi websocket URL.")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID, help="Device-Id registered in xiaozhi.")
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID, help="Client-Id sent to xiaozhi.")
    parser.add_argument("--authorization", default="", help="Optional Authorization header for websocket auth.")
    parser.add_argument("--vision-url", default="", help="Fallback vision URL if MCP initialize omits it.")
    parser.add_argument("--vision-token", default="", help="Fallback vision bearer token.")
    parser.add_argument("--ble-name", default=OPENGLASS_NAME, help="BLE advertised name.")
    parser.add_argument("--ble-address", default="", help="BLE address; skips scanning when provided.")
    parser.add_argument("--capture-timeout", type=float, default=20.0, help="Seconds to wait for one photo.")
    parser.add_argument("--vision-timeout", type=float, default=120.0, help="Seconds to wait for vision response.")
    parser.add_argument("--save-dir", default="captures", help="Directory for local JPEG captures.")
    return parser.parse_args(argv)


async def amain(argv: list[str]) -> int:
    args = parse_args(argv)
    bridge = XiaozhiOpenGlassBridge(args)
    await bridge.run()
    return 0


def main() -> int:
    try:
        return asyncio.run(amain(sys.argv[1:]))
    except KeyboardInterrupt:
        log("Stopped.")
        return 130
    except BridgeError as exc:
        log(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
