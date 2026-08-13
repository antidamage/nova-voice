"""Optional companion host: a second inference host and personal-context source.

Iridium's llama.cpp unit runs ``--parallel 1``. Every auxiliary pass queues
behind the hot interpretation path on that single slot. A companion device is a
second, independent inference host that does not contend for it, and — because
it is the owner's own phone — a source of Calendar, Reminders, location and
Health context Iridium has no other way to see.

Everything in this package is off by default. The single-Iridium deployment
remains the supported baseline: with ``companion_enabled`` false, no route in
here is consulted and the service behaves exactly as it did before.
"""
