"""Connect a reference companion to a *running* Nova, over the real socket.

This exists to answer one question that logs alone cannot: when an offload does
not happen, is it Iridium's routing or the phone? Point this at the deployed
endpoint and the answer is immediate — if the reference peer is offered work and
Iridium uses its answer, the server side is sound and the investigation belongs
on the device.

It is also how the offload path is verified when no phone is available, which is
most of the time: phones lock, suspend their sockets, and go out of the house.

**This peer answers with placeholder content.** That is harmless for
``classify_icon`` — a wrong glyph in the dashboard — and decidedly not harmless
for ``render_response``, whose answer is *spoken aloud in the house*. Iridium
only offers what a peer advertises, so the advertised list is the safety gate:
it defaults to ``classify_icon`` and refuses to advertise a hot-path workload
without ``--i-know-this-speaks-aloud``.

Run it on Iridium, where the CA and an identity can be issued:

    ops/issue-satellite-identity.sh companion-probe /tmp/probe-identity
    /opt/nova-voice/venv/bin/python ops/companion_live_peer.py \
        --endpoint wss://iridium.local:8766/v1/companion \
        --cert /tmp/probe-identity/client.crt \
        --key /tmp/probe-identity/client.key \
        --ca /etc/nova-voice/tls/ca.crt \
        --announced-id companion-probe --jobs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import time
from pathlib import Path

import websockets
from cryptography.hazmat.primitives import serialization

from nova_voice.companion.protocol import JobOffer
from nova_voice.companion.reference import ReferenceCompanion

# Workloads whose answer reaches the household directly. Advertising one of
# these means this process decides what the house says out loud.
SPOKEN_WORKLOADS = frozenset({"interpret", "render_response"})


def answer(offer: JobOffer) -> dict | None:
    """A schema-valid placeholder answer for each routable workload.

    Deliberately not clever. The point is to prove the *path* — offer, accept,
    result, and Iridium using it instead of its own model — so the content only
    has to satisfy the result schema and be obviously synthetic in a log.
    """

    workload = offer.envelope.workload
    payload = offer.payload or {}

    if workload == "classify_icon":
        icons = [icon for icon in (payload.get("icons") or []) if isinstance(icon, str)]
        # The first allowed icon: recognisable in the dashboard as "the peer
        # answered", and always inside the vocabulary the caller sent.
        return {"icon": icons[0] if icons else None}

    if workload == "extract_self_profile_update":
        return {"update": None}

    if workload == "confirm_objective":
        pending = [item for item in (payload.get("pending") or []) if isinstance(item, dict)]
        # Unconfirmed, always. This peer observes no devices, and claiming a
        # target had settled would end a verification loop on a device that
        # never did.
        return {
            "items": [
                {
                    "target": str(item.get("target", "unknown"))[:120],
                    "confirmed": False,
                    "reason": "reference peer does not observe device state",
                }
                for item in pending
            ],
            "all_confirmed": False,
        }

    if workload == "render_response":
        return {"text": "This reply came from the reference companion peer."}

    return None


async def run(args: argparse.Namespace) -> int:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=args.ca)
    context.load_cert_chain(certfile=args.cert, keyfile=args.key)

    private_key = serialization.load_pem_private_key(
        Path(args.key).read_bytes(), password=None
    )
    certificate_pem = Path(args.cert).read_text()

    offered: list[tuple[str, float]] = []

    async def handler(offer: JobOffer) -> dict | None:
        started = time.perf_counter()
        result = answer(offer)
        elapsed = (time.perf_counter() - started) * 1000
        offered.append((offer.envelope.workload, elapsed))
        print(
            f"offered workload={offer.envelope.workload} "
            f"job={offer.envelope.job_id} "
            f"payloadKeys={sorted((offer.payload or {}).keys())} "
            f"answered={'yes' if result is not None else 'no'} "
            f"answer_ms={elapsed:.1f}",
            flush=True,
        )
        return result

    async with websockets.connect(args.endpoint, ssl=context) as socket:
        peer = ReferenceCompanion(
            announced_id=args.announced_id,
            private_key=private_key,
            certificate_pem=certificate_pem,
            send=socket.send,
            receive=socket.recv,
            workloads=tuple(args.workloads),
            display_name="Reference Peer (ops)",
            job_handler=handler,
        )
        ack = await peer.authenticate()
        print(f"session established: {json.dumps(ack, default=str)}", flush=True)
        print(f"advertising {list(args.workloads)}; waiting for {args.jobs} job(s)", flush=True)
        try:
            await asyncio.wait_for(peer.run(until=args.jobs), timeout=args.timeout)
        except TimeoutError:
            # Not a failure of this tool: it means Iridium offered nothing in
            # the window, which is itself the finding. Say so plainly rather
            # than reporting a crash.
            print(
                f"no work offered within {args.timeout}s "
                f"({len(offered)} job(s) serviced)",
                flush=True,
            )
            return 2

    print(f"serviced {len(offered)} job(s): {offered}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="wss://iridium.local:8766/v1/companion")
    parser.add_argument("--cert", required=True, help="client certificate PEM")
    parser.add_argument("--key", required=True, help="client private key PEM (unencrypted)")
    parser.add_argument("--ca", default="/etc/nova-voice/tls/ca.crt")
    parser.add_argument("--announced-id", required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["classify_icon"],
        help="what to advertise; Iridium offers nothing else",
    )
    parser.add_argument("--jobs", type=int, default=1, help="stop after this many jobs")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--i-know-this-speaks-aloud",
        action="store_true",
        help="allow advertising interpret/render_response, whose answers are spoken",
    )
    args = parser.parse_args()

    spoken = SPOKEN_WORKLOADS.intersection(args.workloads)
    if spoken and not args.i_know_this_speaks_aloud:
        parser.error(
            f"{sorted(spoken)} would make this process decide what the house says out "
            "loud. Pass --i-know-this-speaks-aloud if that is genuinely what you want."
        )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
