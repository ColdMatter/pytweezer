import argparse

from pytweezer.drivers.imagemX2 import run_server
from pytweezer.servers.configreader import ConfigReader


def _resolve_server_name(token: str, conf: dict) -> str:
    if isinstance(token, str) and token.startswith("Servers/"):
        return token.split("/", 1)[1]
    if isinstance(token, str) and token in conf.get("Servers", {}):
        return token
    return "ImagEM X2 Camera"


def main():
    parser = argparse.ArgumentParser(description="ImagEM X2 sipyco RPC server launcher")
    parser.add_argument(
        "name",
        nargs="?",
        default="Servers/ImagEM X2 Camera",
        help="process-manager label or explicit server name",
    )
    parser.add_argument(
        "--stream-name",
        default=None,
        help="image stream name used by experiment-side clients",
    )
    parser.add_argument("--image-dir", default=None, help="optional TIFF autosave directory")
    parser.add_argument("--timeout", type=float, default=None, help="frame wait timeout in seconds")
    parser.add_argument("--host", default=None, help="override RPC bind host")
    parser.add_argument("--port", type=int, default=None, help="override RPC bind port")
    parser.add_argument("--simulate", action="store_true", help="force synthetic camera backend")
    args = parser.parse_args()

    conf = ConfigReader.getConfiguration()
    server_name = _resolve_server_name(args.name, conf)
    server_conf = conf.get("Servers", {}).get(server_name, {})

    host = args.host or server_conf.get("host", "127.0.0.1")
    port = args.port if args.port is not None else int(server_conf.get("port", 3251))
    stream_name = args.stream_name or server_conf.get("stream_name", "imagemx2")
    timeout = args.timeout if args.timeout is not None else float(server_conf.get("timeout", 5.0))
    simulate = bool(args.simulate or server_conf.get("simulate", False))

    run_server(
        host=host,
        port=port,
        stream_name=stream_name,
        image_dir=args.image_dir,
        timeout=timeout,
        simulate=simulate,
    )


if __name__ == "__main__":
    main()
