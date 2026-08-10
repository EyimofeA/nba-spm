"""Small local HTTP server for the ratings query contract."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from nba_impact.api.ratings import RatingsStore


def _single(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    values = query.get(key)
    if not values:
        if default is None:
            raise ValueError(f"missing query parameter: {key}")
        return default
    return values[-1]


def dispatch(store: RatingsStore, target: str) -> tuple[int, dict]:
    parsed = urlparse(target)
    query = parse_qs(parsed.query)
    if parsed.path == "/v1/health":
        return HTTPStatus.OK, {"status": "ok", "contract_version": store.config.contract_version}
    if parsed.path == "/v1/meta":
        return HTTPStatus.OK, store.metadata()
    if parsed.path == "/v1/leaderboards/annual":
        return HTTPStatus.OK, store.annual_leaderboard(
            int(_single(query, "season")),
            _single(query, "metric", "aio_net"),
            limit=int(_single(query, "limit", str(store.config.default_limit))),
            offset=int(_single(query, "offset", "0")),
            minimum_possessions=int(_single(query, "minimum_possessions", "0")),
        )
    if parsed.path == "/v1/leaderboards/peaks":
        return HTTPStatus.OK, store.peak_leaderboard(
            int(_single(query, "window")),
            _single(query, "component", "net"),
            limit=int(_single(query, "limit", str(store.config.default_limit))),
            offset=int(_single(query, "offset", "0")),
        )
    if parsed.path == "/v1/players/search":
        return HTTPStatus.OK, store.search_players(
            _single(query, "q"),
            limit=int(_single(query, "limit", str(store.config.default_limit))),
        )
    if parsed.path.startswith("/v1/players/"):
        player = store.player(int(parsed.path.rsplit("/", 1)[-1]))
        if player is None:
            return HTTPStatus.NOT_FOUND, {"error": "player_not_found"}
        return HTTPStatus.OK, player
    return HTTPStatus.NOT_FOUND, {"error": "route_not_found"}


def make_handler(store: RatingsStore) -> type[BaseHTTPRequestHandler]:
    class RatingsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            try:
                status, payload = dispatch(store, self.path)
            except (TypeError, ValueError) as exc:
                status, payload = HTTPStatus.BAD_REQUEST, {"error": str(exc)}
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return RatingsHandler


def serve(store: RatingsStore, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(store))
    print(f"Ratings API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
