# Security Policy

## Reporting a Vulnerability

Report any suspected vulnerability privately and confidentially to
`opensource@limxdynamics.com`.

Do not open a public issue for a suspected vulnerability. Use a minimal,
redacted description that explains the affected component and reproduction
steps. Do not send real credentials, customer data, field data, private logs,
model weights, private endpoints, or other confidential material.

## Supported Network Deployment Boundary

All current runtime network interfaces in this repository—including the policy
server and client, TRON2 robot control, and Bridge observations—are supported
only on a controlled robot LAN that is accessible to authorized systems. Do not
expose them to the Internet or use them on an untrusted or shared network.

Not every current transport provides application authentication or TLS. The
policy-serving and robot-control paths must not be treated as authenticated or
encrypted merely because they run inside a LAN. A configured `wss://` Bridge
does not secure the other links or expand the supported trust boundary.

Any Internet-facing, cross-site, or cloud deployment requires a separate
security review before use.

## Robot Safety Scope

Source disclosure of the current TRON2 tasks, prompts, RTC and training design,
robot mapping, calibration, initialization poses, or replay behavior is not a
functional safety approval or real-robot certification. It does not assert that
this repository implements authentication, TLS, emergency stop, motion limits,
collision protection, or watchdog behavior. Real-robot operation requires the
applicable operator, hardware, and site safety procedures.
